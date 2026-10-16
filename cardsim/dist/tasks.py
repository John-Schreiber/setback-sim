"""Work units for distributed studies.

A study is one big job over a range of items: deals for a survey, hand classes for an
every-hand table. A worker runs any slice [start, end) of it and returns numpy arrays;
merging the slices in order gives exactly what one machine running the whole range
would produce, because every deal and every search decision is seeded by its global
index.

Kinds:
  survey    params: preset, players, search, seed           items: deals
            -> rows: fast.survey columns (trump, hand, own, opp, dealt, sweep)
  allhands  params: preset, players, deals, search, seed    items: hand classes
            (optional limit: only the first `limit` classes)
            -> pts: every team's points (the bidder's team first), shape (classes, deals,
               teams), int8; sweep: 1 where the bidder's team took every trick, shape
               (classes, deals), int8; trump: the suit each class names, shape (classes,)
  allhands_reduced  params: preset, players, deals, search, seed, level, items   items: groups
            hands merged into groups of near-identical value (cardsim/pitch/reduced.py); each
            deal plays one member hand drawn by weight. -> pts, sweep as allhands (groups,
            deals[, teams]); member: the hand class played in each deal, (groups, deals)
"""
import hashlib
import os

import numpy as np

KINDS = ("survey", "allhands", "allhands_reduced")
COMPUTE_FILES = ["cardsim/cards.py", "cardsim/hands.py", "cardsim/pitch/rules.py", "cardsim/pitch/tables.py",
                 "cardsim/pitch/core.py", "cardsim/pitch/fast.py", "cardsim/pitch/gpu.py",
                 "cardsim/pitch/rollout.cu", "cardsim/pitch/gpu_cpp.py", "cardsim/pitch/reduced.py",
                 "cardsim/dist/tasks.py", "setback.py", "trump_weights.py"]
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def code_version():
    """Fingerprint of the code that decides how games are played: a study's slices must
    all come from the same. Tools that only submit studies or read results (client,
    charts, coordinator) are left out, so changing them needs no worker redeploy."""
    h = hashlib.sha256()
    files = [os.path.join(ROOT, f) for f in COMPUTE_FILES]
    for fn in files:
        with open(fn, "rb") as f:
            source = f.read().replace(b"\r\n", b"\n")  # same fingerprint on Windows and Linux
        h.update(os.path.relpath(fn, ROOT).replace(os.sep, "/").encode() + b"\n" + source)
    return h.hexdigest()[:16]


def study_params(kind, preset, players, search, seed, **extra):
    """Parameters for a new study, pinning the trump-choice weights and code version
    so every slice, on any machine, plays exactly the same way."""
    from cardsim.pitch import fast
    tw, tl, tx = fast.trump_weights(preset, players)
    return dict(preset=preset, players=players, search=search, seed=seed, code=code_version(),
                weights=[tw.tolist(), tl.tolist(), tx.tolist()], **extra)


def _game(params):
    from cardsim.pitch import PitchRules, fast
    w = params.get("weights")
    weights = tuple(np.array(x, np.int64) for x in w) if w else None
    return fast.Game(PitchRules.preset(params["preset"]), int(params["players"]), weights=weights)


def run(kind, params, start, end, device):
    """Run items [start, end) of a study on 'gpu' or 'cpu'. Returns {name: array}."""
    from cardsim.pitch import fast
    if params.get("code") and params["code"] != code_version():
        raise ValueError(f"this worker runs code {code_version()}, the study needs {params['code']}: "
                         "update the worker (git pull / rebuild the image)")
    count = end - start
    g = _game(params)
    r = g.rules
    search = np.full(g.nteams, int(params.get("search", 0)), np.int64)
    seed = int(params.get("seed", 1))
    if kind == "survey":
        if device == "gpu":
            from cardsim.pitch import gpu
            rows = gpu.survey(count, g, search, seed=seed, start=start)
        else:
            rows = fast.survey(count, g.nplayers, r.deal_size, r.hand_size, r.refill, g.team, g.nteams, True,
                               g.rules_bt, search, g.tab, g.gf, g.tw, g.tl, g.tx, seed, start)
        return {"rows": rows}
    if kind in ("allhands", "allhands_reduced"):
        if r.deal_size != 6 or r.refill or r.jokers:
            raise ValueError(f"{kind} needs a 52-card, 6-card, no-refill preset (not {r.name})")
        deals = int(params["deals"])
        rules = g.rules_bt[g.team[0]]
        if kind == "allhands":
            from cardsim.hands import hand_classes
            keys, _ = hand_classes()
            hands = keys[start:end]
            trumps = np.array([fast.pick_trump(np.int64(h), g.tab, g.tw, g.tl, g.tx) for h in hands], np.int64)
            sc, won = _deal_and_play(g, rules, search, seed, hands, trumps, np.arange(start, end, dtype=np.int64),
                                     deals, device)
            extra = {"trump": trumps.astype(np.int8)}
        else:
            from cardsim.pitch import reduced
            level = params.get("level", "side")
            member = reduced.members(g, level, seed, start, end, deals)  # (groups, deals) class indices
            _, _, _, _, _, keys, ctrumps, _ = reduced.groups(g, level)
            flat = member.ravel()
            sc, won = _deal_and_play(g, rules, search, seed, keys[flat], ctrumps[flat],
                                     np.arange(start * deals, end * deals, dtype=np.int64), 1, device)
            extra = {"member": member.astype(np.int32)}
        bt = g.team[0]
        order = [bt] + [k for k in range(g.nteams) if k != bt]
        sweep = (np.delete(won, bt, axis=1) == 0).all(axis=1)
        return {"pts": sc[:, order].reshape(count, deals, g.nteams).astype(np.int8),
                "sweep": sweep.reshape(count, deals).astype(np.int8), **extra}
    raise ValueError(f"unknown kind {kind!r}")


def _deal_and_play(g, rules, search, seed, hands, trumps, ids, per_item, device):
    """Deal per_item layouts around each bidder hand (layout j of item ids[i] uses seed
    ids[i]*per_item + j, ids contiguous) and play them; on the GPU, the next batch is dealt
    while one plays. Returns (points per team, cards each team captured) per game."""
    from cardsim.pitch import fast
    if device == "gpu":
        from cardsim.pitch import gpu
        per = max(1, 500_000 // per_item)

        def deal(a):
            return fast.deal_with_hand_batch(hands[a:a + per], trumps[a:a + per], ids[a:a + per], per_item,
                                             g.nplayers, 6, 6, False, g.team, g.tab, g.gf, seed)

        def play(a, dealt):
            return gpu.play_batch(dealt[0], np.repeat(trumps[a:a + per], per_item), dealt[1], g.team, rules,
                                  search, g.tab, g.gf, seed=seed, game_offset=int(ids[a]) * per_item,
                                  return_won=True)

        parts = [res for _, res in gpu.overlapped(range(0, len(hands), per), deal, play)]
        return np.concatenate([q[0] for q in parts]), np.concatenate([q[1] for q in parts])
    dealt, _ = fast.deal_with_hand_batch(hands, trumps, ids, per_item, g.nplayers, 6, 6, False, g.team, g.tab,
                                         g.gf, seed)
    return fast.play_many(dealt, np.repeat(trumps, per_item), g.team, g.nteams, rules, search, g.tab, g.gf, seed,
                          int(ids[0]) * per_item)


def total_items(kind, params):
    if kind == "survey":
        return int(params["hands"])
    if kind == "allhands":
        from cardsim.hands import N_CLASSES
        return min(N_CLASSES, int(params.get("limit", N_CLASSES)))
    if kind == "allhands_reduced":
        return int(params["items"])  # number of groups, worked out when the study is submitted
    raise ValueError(kind)


def merge(parts):
    """parts: list of (start, {name: array}) -> {name: array} concatenated in order."""
    parts = sorted(parts, key=lambda p: p[0])
    return {k: np.concatenate([p[1][k] for p in parts]) for k in parts[0][1]}
