"""Bidding charts for any house-rule preset, from a large GPU survey with search players.

Each hand is reduced to the honors it holds in the suit it names as trump (A K Q J,
plus the variant's own trumps and point cards: off-jack, jokers, 10, 9, 5, 3 and the 2
for Low) and how many other small trumps. For bidding variants that deal more cards
than are played (Nine-Five), this is the hand as dealt, which is what you bid on.

For each group, the expected points of every bid in the variant's range are worked out
from the bidding team's points (making a bid scores the points taken, being set loses
the bid), plus a moon bid where the rules have one. The chart gives the highest bid with
expected points >= 0, by honors (rows) and number of small trumps (columns).

The .csv also has each bid's net expected points (net2, net3, ..., net_moon): the bidding
team's score change minus what everyone else scores on the hand (opp, the average of
which is also given), i.e. how far the bid moves you against the field. Points nobody
scores (a tie for Game) count for neither side. The play does not depend on the bid, so
net and ev differ by the same opp for every bid and rank the bids alike; they differ
only on whether bidding beats passing, where net >= 0 is the stricter (too strict:
passing also lets the opponents bid) test.

    .venv/Scripts/python -m cardsim.pitch.bidchart --preset ten_point -p 4
    .venv/Scripts/python -m cardsim.pitch.bidchart --cluster http://localhost:8700   # all presets, on the cluster
    .venv/Scripts/python -m cardsim.pitch.bidchart --cluster http://localhost:8700 --hands 4000000 --rebuild
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

from cardsim.cards import BIG_JOKER, COLOR_MATE, LITTLE_JOKER, card
from cardsim.pitch import PRESETS, PitchRules
from cardsim.pitch.notes import COMMON, NOTES
from cardsim.pitch.rules import describe
from cardsim.pitch import fast, gpu

OUT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "charts"))
MIN_HANDS = 200  # groups with fewer hands are left blank


def honor_list(rules):
    """(label, which card for trump t) for this variant's honors, strongest first."""
    h = [("A", lambda t: card(14, t)), ("K", lambda t: card(13, t)), ("Q", lambda t: card(12, t)),
         ("J", lambda t: card(11, t))]
    if rules.off_jack:
        h.append(("offJ", lambda t: card(11, COLOR_MATE[t])))
    if rules.jokers >= 1:
        h.append(("BJ", lambda t: BIG_JOKER))
    if rules.jokers >= 2:
        h.append(("LJ", lambda t: LITTLE_JOKER))
    h.append(("10", lambda t: card(10, t)))
    for r, pts in ((9, rules.nine_pts), (5, rules.five_pts), (3, rules.three_pts)):
        if pts:
            h.append((str(r), lambda t, r=r: card(r, t)))
    h.append(("2", lambda t: card(2, t)))
    return h


def features(dealt, trumps, rules, tab):
    """Honors label and number of other small trumps for each (dealt hand, trump)."""
    hon = honor_list(rules)
    labels, small = [], []
    for m, t in zip(dealt.tolist(), trumps.tolist()):
        cards = [lab for lab, f in hon if (m >> f(t)) & 1]
        labels.append(" ".join(cards))
        trump_mask = int(tab[t, 6, t])  # F_ESM
        n_tr = bin(m & trump_mask).count("1")
        small.append(n_tr - len(cards))
    return labels, small


def build(preset, nplayers, nhands, search, seed=2027):
    """Run the survey on this machine's GPU and aggregate it."""
    rules = PitchRules.preset(preset)
    game = fast.Game(rules, nplayers)
    t0 = time.time()
    s = np.full(game.nteams, search, np.int64)
    out = gpu.survey(nhands, game, s, seed=seed)
    return rules, game, aggregate(rules, game, out), time.time() - t0, nhands


def aggregate(rules, game, out):
    """Survey rows (fast.S_* columns) -> one row per (honors, small trumps) group with the
    expected points of every bid and the highest profitable bid."""
    labels, small = features(out[:, fast.S_DEALT], out[:, fast.S_TRUMP], rules, game.tab)
    own = out[:, fast.S_OWN]
    opp = out[:, fast.S_OPP]
    moon_ok = (out[:, fast.S_SWEEP] == 1) & (own == rules.points_total)
    df = pd.DataFrame({"honors": labels, "small": small, "own": own, "opp": opp, "moon": moon_ok})
    bids = list(range(rules.min_bid, rules.max_bid + 1))
    for k in bids:
        df[f"ev{k}"] = np.where(own >= k, own, -k)
        df[f"net{k}"] = df[f"ev{k}"] - opp
    if rules.moon_bid:
        df["net_moon"] = np.where(moon_ok, rules.moon_value, -rules.moon_value) - opp
    nets = [f"net{k}" for k in bids] + (["net_moon"] if rules.moon_bid else [])
    g = df.groupby(["honors", "small"]).agg(
        hands=("own", "size"), avg=("own", "mean"), opp=("opp", "mean"), p_moon=("moon", "mean"),
        **{f"ev{k}": (f"ev{k}", "mean") for k in bids}, **{c: (c, "mean") for c in nets}).reset_index()
    ev = g[[f"ev{k}" for k in bids]].to_numpy()
    ok = ev >= 0
    g["max_bid"] = np.where(ok.any(axis=1), np.array(bids)[np.where(ok, np.arange(len(bids)), -1).max(axis=1)], 0)
    if rules.moon_bid:
        g["ev_moon"] = rules.moon_value * (2 * g.p_moon - 1)
    # the line that wins whole games (passline.LINES): highest bid with net >= the variant's line
    from cardsim.pitch.lines import LINES
    line = LINES.get((rules.name, game.nplayers))
    if line is not None:
        okn = g[[f"net{k}" for k in bids]].to_numpy() >= line
        g["bid_line"] = np.where(okn.any(axis=1), np.array(bids)[np.where(okn, np.arange(len(bids)), -1).max(axis=1)], 0)
    first = ["honors", "small", "hands", "avg", "opp", "p_moon"]
    g = g[first + [f"ev{k}" for k in bids] + ["max_bid"] + (["bid_line"] if line is not None else []) +
          (["ev_moon"] if rules.moon_bid else []) + nets]
    return g[g.hands >= 1]


def write_chart(rules, nplayers, g, nhands, search):
    base = os.path.join(OUT, f"{rules.name}_{nplayers}p")
    g.to_csv(base + ".csv", index=False, float_format="%.4f")
    with open(base + ".md", "w", encoding="utf-8") as f:
        f.write(markdown(rules, nplayers, g, nhands, search))
    return base + ".md"


def run_cluster(url, jobs, search, seed, poll=60, rebuild=False):
    """Submit one survey study per (preset, players, hands) to the coordinator (or reuse
    an existing one with the same settings and code), wait, fetch, and write the charts.
    rebuild: submit nothing; rewrite the charts from finished studies with the same
    settings, whatever code they ran on (for changes to the charts, not to the play)."""
    import io
    from cardsim.dist.client import token_from_env
    from cardsim.dist.tasks import code_version, study_params
    from cardsim.dist.worker import Client
    c = Client(url, token_from_env())
    studies = c.get("/studies")
    existing = {(s["name"], s["params"].get("code")): s["id"] for s in studies}
    finished = {s["name"]: s["id"] for s in studies if s["complete"]}
    ids = {}
    for preset, n, nh in jobs:
        name = f"bidchart {preset} {n}p {nh} deals search {search} seed {seed}"
        if rebuild:
            if name not in finished:
                raise SystemExit(f"no finished study named {name!r} to rebuild from")
            ids[(preset, n, nh)] = finished[name]
            continue
        sid = existing.get((name, code_version()))
        if sid is None:
            params = study_params("survey", preset, n, search, seed, hands=nh)
            sid = c.post("/studies", {"kind": "survey", "params": params, "engine": "any", "name": name,
                                      "calib": 4000})["id"]
        ids[(preset, n, nh)] = sid
    print(f"{len(ids)} studies on {url}", flush=True)
    t0 = time.time()
    while True:
        st = {s["id"]: s for s in c.get("/studies")}
        done = sum(st[i]["done"] for i in ids.values())
        total = sum(st[i]["total"] for i in ids.values())
        if all(st[i]["complete"] for i in ids.values()):
            break
        print(f"  {done:,}/{total:,} deals ({100 * done / total:.1f}%) after {(time.time() - t0) / 60:.1f} min",
              flush=True)
        time.sleep(poll)
    for (preset, n, nh), sid in ids.items():
        rows = np.load(io.BytesIO(c.get(f"/studies/{sid}/result")))["rows"]
        rules = PitchRules.preset(preset)
        g = aggregate(rules, fast.Game(rules, n), rows)
        print(f"{preset} {n}-handed: {nh:,} deals -> {write_chart(rules, n, g, nh, search)}", flush=True)
    write_index()


def strength_key(label, rules):
    order = [lab for lab, _ in honor_list(rules)]
    return tuple(order.index(x) for x in label.split()) + (99,) if label else (99,)


def markdown(rules, nplayers, g, nhands, search):
    from cardsim.pitch.lines import LINES
    pass_label = "pass"
    cols = sorted(g.small.unique())
    line = LINES.get((rules.name, nplayers))
    what = (f"Highest bid worth making by the line that wins the most whole games: net expected points "
            f"(score change minus the opponents' points on the hand) >= {line:+g} (cardsim/pitch/passline.py; "
            f"the .csv's max_bid is the older line, expected points >= 0)" if "bid_line" in g else
            "Highest bid with expected points >= 0")
    lines = [f"## {rules.name}, {nplayers}-handed",
             "",
             f"{what} (bids {rules.min_bid}-{rules.max_bid}; '{pass_label}' = "
             f"none), by trump honors held (rows) and number of other small trumps (columns). From "
             f"{nhands:,} deals with search players; blank = fewer than {MIN_HANDS} hands."
             + (f" A moon bid ({rules.moon_bid}, worth {rules.moon_value}) is marked * where it is also "
                f"worth bidding." if rules.moon_bid else ""),
             "",
             "| trump honors | " + " | ".join(f"+{c}" for c in cols) + " |",
             "|---|" + "---|" * len(cols)]
    piv = {(r.honors, r.small): r for r in g.itertuples()}
    for hon in sorted(g.honors.unique(), key=lambda h: strength_key(h, rules)):
        cells = []
        for c in cols:
            r = piv.get((hon, c))
            if r is None or r.hands < MIN_HANDS:
                cells.append("")
            else:
                b = r.bid_line if "bid_line" in g else r.max_bid
                cell = str(b) if b else pass_label
                if rules.moon_bid and r.ev_moon > 0:
                    cell += "*"
                cells.append(cell)
        if any(cells):
            lines.append(f"| {hon or '(none)'} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="all")
    ap.add_argument("-p", "--players", type=int, nargs="+", default=[4, 6])
    ap.add_argument("--hands", type=int, default=1_000_000)
    ap.add_argument("--search", type=int, default=16)
    ap.add_argument("--seed", type=int, default=2027)
    ap.add_argument("--cluster", help="coordinator URL: run the surveys on the cluster instead of this GPU")
    ap.add_argument("--rebuild", action="store_true",
                    help="with --cluster: only rewrite the charts from the finished surveys (no new deals)")
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    os.makedirs(OUT, exist_ok=True)
    presets = list(PRESETS) if a.preset == "all" else [a.preset]
    jobs = []
    for preset in presets:
        for n in a.players:
            r = PitchRules.preset(preset)
            if n * r.deal_size > 52 + r.jokers:
                continue
            jobs.append((preset, n, a.hands if n == 4 else a.hands // 2))
    if a.cluster:
        run_cluster(a.cluster, jobs, a.search, a.seed, rebuild=a.rebuild)
        return
    for preset, n, nh in jobs:
        rules, game, g, secs, nh = build(preset, n, nh, a.search, a.seed)
        print(f"{preset} {n}-handed: {nh:,} deals in {secs:.0f}s -> {write_chart(rules, n, g, nh, a.search)}",
              flush=True)
    write_index()


def write_index():
    lines = ["# Bidding charts by house rules", "",
             "Each chart gives the highest bid worth making (by the line that wins the most whole games: net "
             "expected points of at least a per-variant threshold, cardsim/pitch/passline.py) by the honors you hold in "
             "your trump suit and how many other small trumps you have. Built by "
             "`python -m cardsim.pitch.bidchart` from large GPU surveys with search players; the `.csv` next "
             "to each chart has the expected points of every bid for every group, and the net expected points "
             "(score change minus the opponents' points on the hand).", "",
             "| Rules | 4-handed | 6-handed | What changes |", "|---|---|---|---|"]
    for name in PRESETS:
        r = PitchRules.preset(name)
        links = []
        for n in (4, 6):
            fn = os.path.join(OUT, f"{name}_{n}p.md")
            links.append(f"[chart]({name}_{n}p.md)" if os.path.exists(fn) else "-")
        lines.append(f"| **{name}** | {links[0]} | {links[1]} | {describe(r)} |")
    lines += ["", "## The variants", "", COMMON]
    for name in PRESETS:
        lines += ["", f"**{name}**: {NOTES[name]}"]
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
