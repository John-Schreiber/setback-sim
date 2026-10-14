"""Bidding charts for any house-rule preset, from a large GPU survey with search players.

Each hand is reduced to the honors it holds in the suit it names as trump (A K Q J,
plus the variant's own trumps and point cards: off-jack, jokers, 10, 9, 5, 3 and the 2
for Low) and how many other small trumps. For bidding variants that deal more cards
than are played (Nine-Five), this is the hand as dealt, which is what you bid on.

For each group, the expected points of every bid in the variant's range are worked out
from the bidding team's points (making a bid scores the points taken, being set loses
the bid), plus a moon bid where the rules have one. The chart gives the highest bid with
expected points >= 0, by honors (rows) and number of small trumps (columns).

    .venv/Scripts/python -m cardsim.pitch.bidchart --preset ten_point -p 4
"""
import argparse
import os
import time
import warnings

import numpy as np
import pandas as pd

from cardsim.cards import BIG_JOKER, COLOR_MATE, LITTLE_JOKER, card
from cardsim.pitch import PRESETS, PitchRules
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
    rules = PitchRules.preset(preset)
    game = fast.Game(rules, nplayers)
    t0 = time.time()
    s = np.full(game.nteams, search, np.int64)
    out = gpu.survey(nhands, game, s, seed=seed)
    labels, small = features(out[:, fast.S_DEALT], out[:, fast.S_TRUMP], rules, game.tab)
    own = out[:, fast.S_OWN]
    moon_ok = (out[:, fast.S_SWEEP] == 1) & (own == rules.points_total)
    df = pd.DataFrame({"honors": labels, "small": small, "own": own, "moon": moon_ok})
    bids = list(range(rules.min_bid, rules.max_bid + 1))
    for k in bids:
        df[f"ev{k}"] = np.where(own >= k, own, -k)
    g = df.groupby(["honors", "small"]).agg(
        hands=("own", "size"), avg=("own", "mean"), p_moon=("moon", "mean"),
        **{f"ev{k}": (f"ev{k}", "mean") for k in bids}).reset_index()
    ev = g[[f"ev{k}" for k in bids]].to_numpy()
    ok = ev >= 0
    g["max_bid"] = np.where(ok.any(axis=1), np.array(bids)[np.where(ok, np.arange(len(bids)), -1).max(axis=1)], 0)
    if rules.moon_bid:
        g["ev_moon"] = rules.moon_value * (2 * g.p_moon - 1)
    g = g[g.hands >= 1]
    return rules, game, g, time.time() - t0, nhands


def strength_key(label, rules):
    order = [lab for lab, _ in honor_list(rules)]
    return tuple(order.index(x) for x in label.split()) + (99,) if label else (99,)


def markdown(rules, nplayers, g, nhands, search):
    pass_label = "pass"
    cols = sorted(g.small.unique())
    lines = [f"## {rules.name}, {nplayers}-handed",
             "",
             f"Highest bid with expected points >= 0 (bids {rules.min_bid}-{rules.max_bid}; '{pass_label}' = "
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
                cell = str(r.max_bid) if r.max_bid else pass_label
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
    a = ap.parse_args()
    warnings.filterwarnings("ignore")
    os.makedirs(OUT, exist_ok=True)
    presets = list(PRESETS) if a.preset == "all" else [a.preset]
    for preset in presets:
        for n in a.players:
            r = PitchRules.preset(preset)
            if n * r.deal_size > 52 + r.jokers:
                continue
            nh = a.hands if n == 4 else a.hands // 2
            rules, game, g, secs, nh = build(preset, n, nh, a.search)
            base = os.path.join(OUT, f"{preset}_{n}p")
            g.to_csv(base + ".csv", index=False, float_format="%.4f")
            with open(base + ".md", "w", encoding="utf-8") as f:
                f.write(markdown(rules, n, g, nh, a.search))
            print(f"{preset} {n}-handed: {nh:,} deals in {secs:.0f}s -> {base}.md", flush=True)
    write_index()


def describe(rules):
    """One-line summary of a preset's rules."""
    pts = [f"High {rules.high}", f"Low {rules.low}" + ("" if rules.low_to_captor else " (to whoever was dealt it)"),
           f"Jack {rules.jack}"]
    if rules.off_jack:
        pts.append(f"off-jack {rules.off_jack_pts}")
    if rules.jokers:
        pts.append(f"{rules.jokers} joker{'s' if rules.jokers > 1 else ''} {rules.joker_pts} each")
    for lab, v in (("10 of trump", rules.ten_pts), ("3 of trump", rules.three_pts),
                   ("5 of trump", rules.five_pts), ("9 of trump", rules.nine_pts)):
        if v:
            pts.append(f"{lab} {v}")
    pts.append(f"Game {rules.game}")
    if rules.last_trick:
        pts.append(f"last trick {rules.last_trick}")
    deal = f"{rules.deal_size} cards" + (", discard non-trumps and draw back to 6" if rules.refill or
                                         rules.deal_size > rules.hand_size else "")
    play = "may trump in" if rules.may_trump else "must follow suit"
    bids = f"bids {rules.min_bid}-{rules.max_bid}" + (f", moon {rules.moon_bid} = {rules.moon_value}"
                                                      if rules.moon_bid else "")
    return f"{deal}; {play}; {', '.join(pts)} ({rules.points_total} total); {bids}; game to {rules.target}"


def write_index():
    lines = ["# Bidding charts by house rules", "",
             "Each chart gives the highest bid worth making (expected points >= 0) by the honors you hold in "
             "your trump suit and how many other small trumps you have. Built by "
             "`python -m cardsim.pitch.bidchart` from large GPU surveys with search players; the `.csv` next "
             "to each chart has the expected points of every bid for every group.", "",
             "| Rules | 4-handed | 6-handed | What changes |", "|---|---|---|---|"]
    for name in PRESETS:
        r = PitchRules.preset(name)
        links = []
        for n in (4, 6):
            fn = os.path.join(OUT, f"{name}_{n}p.md")
            links.append(f"[chart]({name}_{n}p.md)" if os.path.exists(fn) else "-")
        lines.append(f"| **{name}** | {links[0]} | {links[1]} | {describe(r)} |")
    with open(os.path.join(OUT, "README.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
