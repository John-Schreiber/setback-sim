"""Fit the bidder's trump-choice weights for each house-rule preset.

For random dealt hands, every suit the bidder could name is played with search
players on the same layouts (GPU). The weights (a value per natural rank, the
off-jack and each joker, plus a bonus per trump length) are then fitted by
coordinate search to lose the fewest points to naming the wrong suit, on half the
hands, and checked on the other half. Results go to cardsim/pitch/trump_weights.json,
which fast.Game loads.

    .venv/Scripts/python -m cardsim.pitch.fit_trump --preset ten_point -p 4
"""
import argparse
import json
import os
import time

import numpy as np

from cardsim.cards import COLOR_MATE, card, deck_mask
from cardsim.pitch import PRESETS, PitchRules
from cardsim.pitch import fast, gpu

HERE = os.path.dirname(os.path.abspath(__file__))
JSON = os.path.join(HERE, "trump_weights.json")
N_LEN = 10  # length bonus for 0..9 trumps


def collect(game, nhands, deals, search, seed):
    """Mean bidder-team points for each (hand, suit held) -> (hands, points[nhands, 4])."""
    r = game.rules
    rng = np.random.default_rng(seed)
    deck = [c for c in range(54) if (deck_mask(r.jokers) >> c) & 1]
    hands = np.array([sum(1 << int(c) for c in rng.choice(deck, r.deal_size, replace=False))
                      for _ in range(nhands)], np.int64)
    pts = np.full((nhands, 4), np.nan)
    srch = np.full(game.nteams, search, np.int64)
    for s in range(4):
        idx = np.nonzero(((hands >> (13 * s)) & 0x1FFF) != 0)[0]
        step = max(1, 200_000 // deals)
        for a in range(0, len(idx), step):
            ids = idx[a:a + step].astype(np.int64)
            tr = np.full(len(ids), s, np.int64)
            dealt, lo = fast.deal_with_hand_batch(hands[ids], tr, ids, deals, game.nplayers, r.deal_size,
                                                  r.hand_size, r.refill, game.team, game.tab, game.gf, seed)
            sc = gpu.play_batch(dealt, np.repeat(tr, deals), lo, game.team, game.rules_bt[game.team[0]], srch,
                                game.tab, game.gf, seed=seed + 7 * s + a)
            pts[ids, s] = sc[:, game.team[0]].reshape(len(ids), deals).mean(axis=1)
    return hands, pts


def features(hands, rules):
    """x[hand, suit, :] = 13 natural-rank flags, off-jack / big joker / little joker
    held, then one-hot trump length 0..9."""
    x = np.zeros((len(hands), 4, 13 + 3 + N_LEN))
    for s in range(4):
        nat = (hands >> (13 * s)) & 0x1FFF
        for r in range(13):
            x[:, s, r] = (nat >> r) & 1
        if rules.off_jack:
            x[:, s, 13] = (hands >> card(11, COLOR_MATE[s])) & 1
        if rules.jokers >= 1:
            x[:, s, 14] = (hands >> 53) & 1
        if rules.jokers >= 2:
            x[:, s, 15] = (hands >> 52) & 1
        n = x[:, s, :16].sum(axis=1).astype(int)
        x[np.arange(len(hands)), s, 16 + np.minimum(n, N_LEN - 1)] = 1
    return x


def regret(theta, x, val, held):
    score = x @ theta
    score[~held] = -np.inf
    pick = np.argmax(score, axis=1)  # ties -> lowest suit, as in the engine
    return np.mean(np.nanmax(val, axis=1) - val[np.arange(len(val)), pick])


def fit(x, val, held, theta):
    best = regret(theta, x, val, held)
    for step in (100, 50, 25, 10, 5):
        improved = True
        while improved:
            improved = False
            for i in range(len(theta)):
                for d in (-step, step):
                    t = theta.copy()
                    t[i] += d
                    r = regret(t, x, val, held)
                    if r < best - 1e-9:
                        theta, best, improved = t, r, True
    return theta


def run(preset, nplayers, nhands, deals, search, seed):
    rules = PitchRules.preset(preset)
    game = fast.Game(rules, nplayers)
    t0 = time.time()
    hands, pts = collect(game, nhands, deals, search, seed)
    held = ~np.isnan(pts)
    keep = held.sum(axis=1) >= 2
    x, val, held = features(hands[keep], rules), pts[keep], held[keep]
    half = len(x) // 2
    tr, te = slice(0, half), slice(half, None)
    tw, tl, tx = fast.trump_weights(preset, nplayers)
    start = np.concatenate([tw, tx, tl[:N_LEN]]).astype(float)
    theta = fit(x[tr], val[tr], held[tr], start.copy())
    r_old, r_new = regret(start, x[te], val[te], held[te]), regret(theta, x[te], val[te], held[te])
    print(f"{preset} {nplayers}-handed: {keep.sum():,} hands with a choice, {time.time() - t0:.0f}s; "
          f"points lost to the wrong trump (held-out half): {r_old:.3f} -> {r_new:.3f}", flush=True)
    data = json.load(open(JSON)) if os.path.exists(JSON) else {}
    th = [int(round(v)) for v in theta]
    data.setdefault(preset, {})[str(nplayers)] = {
        "rank": th[:13], "extra": th[13:16], "length": th[16:16 + N_LEN],
        "regret_before": round(r_old, 4), "regret_after": round(r_new, 4),
        "hands": int(keep.sum()), "deals": deals, "search": search}
    json.dump(data, open(JSON, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="all", help="a preset name or 'all' (all but setback4)")
    ap.add_argument("-p", "--players", type=int, nargs="+", default=[4, 6])
    ap.add_argument("--hands", type=int, default=20000)
    ap.add_argument("--deals", type=int, default=32)
    ap.add_argument("--search", type=int, default=16)
    ap.add_argument("--seed", type=int, default=31337)
    a = ap.parse_args()
    presets = [p for p in PRESETS if p != "setback4"] if a.preset == "all" else [a.preset]
    for preset in presets:
        for n in a.players:
            r = PitchRules.preset(preset)
            if n * r.deal_size > 52 + r.jokers:
                continue
            run(preset, n, a.hands, a.deals, a.search, a.seed)


if __name__ == "__main__":
    main()
