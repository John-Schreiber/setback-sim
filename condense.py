"""Condensed bid charts from the every-hand search tables (allhands.py).

A hand is reduced to which trump honors it holds (A K Q J 10 2) and how many
small trumps (3-9). For each group the expected points of each bid are averaged
over every real hand in it (weighted by how often it's dealt), and the
recommended bid is the highest with expected points >= 0. Adding off-suit Aces
or 10s as a third feature barely changes any bid (see agreement()).

Writes condensed_<P>p.csv (every group), bidchart_condensed.txt (fewest small
trumps per bid, every honors combination) and bidchart_condensed.md (the same,
as a 16 x 4 grid per game).

    .venv/Scripts/python condense.py
"""
import numpy as np
import pandas as pd

HONORS = "AKQJT2"
GAMES = ((4, 224), (6, 112))  # players, deals per hand in the every-hand table


def features(df):
    hon, nx, offa, offt = [], [], [], []
    for h, t in zip(df.hand, df.trump):
        cards = h.split()
        tr = [c[0] for c in cards if c[1] == t]
        hon.append("".join(r for r in HONORS if r in tr))
        nx.append(sum(1 for r in tr if r in "3456789"))
        offa.append(min(2, sum(1 for c in cards if c[1] != t and c[0] == "A")))
        offt.append(min(2, sum(1 for c in cards if c[1] != t and c[0] == "T")))
    return df.assign(hon=hon, nx=nx, offa=offa, offt=offt)


def bid_of(ev2, ev3, ev4):
    return np.select([ev4 >= 0, ev3 >= 0, ev2 >= 0], [4, 3, 2], 0)


def grouped(df, keys):
    w = df.weight
    g = df.assign(w2=w * df.ev2, w3=w * df.ev3, w4=w * df.ev4).groupby(keys).agg(
        w=("weight", "sum"), w2=("w2", "sum"), w3=("w3", "sum"), w4=("w4", "sum"))
    for k in (2, 3, 4):
        g[f"ev{k}"] = g[f"w{k}"] / g.w
    g["bid"] = bid_of(g.ev2, g.ev3, g.ev4)
    return g[["w", "ev2", "ev3", "ev4", "bid"]]


def load(p, d):
    df = pd.read_csv(f"allhands_{p}p_d{d}_s16.csv.gz", usecols=["hand", "trump", "weight", "ev2", "ev3", "ev4"])
    df = features(df)
    df["exact"] = bid_of(df.ev2, df.ev3, df.ev4)
    return df


def agreement(p, df):
    """How often each condensing level gives the same bid as the exact table."""
    W = df.weight.sum()
    print(f"{p}-handed: how much each condensing level keeps")
    for name, keys in (("trump honors + # small trumps", ["hon", "nx"]),
                       ("  + off-suit Aces", ["hon", "nx", "offa"]),
                       ("  + off-suit Aces and 10s", ["hon", "nx", "offa", "offt"])):
        g = grouped(df, keys)
        b = df.join(g.bid.rename("cbid"), on=keys).cbid
        print(f"  {name:32} {len(g):5} groups   same bid as the exact table for "
              f"{df.weight[b == df.exact].sum() / W:.1%} of real hands")


def thresholds(g):
    """{honors: {bid: fewest small trumps needed, or None}}, plus the honors where
    adding a small trump ever lowers the bid."""
    need, nonmono = {}, []
    for hon, sub in g.groupby(level="hon"):
        bids = sub.droplevel("hon").sort_index().bid
        need[hon] = {b: next((n for n, v in bids.items() if v >= b), None) for b in (2, 3, 4)}
        if (np.diff(bids.to_numpy()) < 0).any():
            nonmono.append(hon)
    return need, nonmono


def label(hon):
    return " ".join("10" if h == "T" else h for h in hon) or "(none)"


def text_chart(p, g, need, nonmono):
    W = g.w.groupby(level="hon").sum() / g.w.sum()
    fmt = lambda v: "-" if v is None else str(v)
    lines = [f"{p}-handed: fewest small trumps (3-9) needed for each bid ('-' = never; "
             f"'0' = enough with none)",
             f"  {len(need)} honor combinations; adding a small trump never lowers the bid in "
             f"{len(need) - len(nonmono)} of them", "",
             f"  {'honors':14} {'bid 2':>6} {'bid 3':>6} {'bid 4':>6}   share of hands"]
    order = sorted(need, key=lambda h: [HONORS.index(c) for c in h] + [9])
    for hon in order:
        n = need[hon]
        lines.append(f"  {label(hon):14} {fmt(n[2]):>6} {fmt(n[3]):>6} {fmt(n[4]):>6}   {W[hon]:6.2%}")
    return "\n".join(lines)


def grid_markdown(p, need):
    """16 x 4 grid: rows = which of A K Q J you hold; columns = + nothing / 10 / 2 /
    10 and 2. Cell = fewest small trumps needed to bid 2 / 3 / 4 ('-' = never)."""
    fmt = lambda v: "-" if v is None else str(v)
    lines = [f"| {p}-handed | + nothing | + 10 | + 2 | + 10 and 2 |", "|---|---|---|---|---|"]
    for top in ["AKQJ", "AKQ", "AKJ", "AQJ", "KQJ", "AK", "AQ", "AJ", "KQ", "KJ", "QJ",
                "A", "K", "Q", "J", ""]:
        cells = []
        for extra in ["", "T", "2", "T2"]:
            n = need.get("".join(h for h in HONORS if h in top + extra))
            cells.append(" / ".join(fmt(n[b]) for b in (2, 3, 4)) if n else "")
        lines.append(f"| **{' '.join(top) or 'none'}** | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    texts, grids = [], []
    for p, d in GAMES:
        df = load(p, d)
        agreement(p, df)
        g = grouped(df, ["hon", "nx"])
        g.reset_index().to_csv(f"condensed_{p}p.csv", index=False, float_format="%.3f")
        need, nonmono = thresholds(g)
        texts.append(text_chart(p, g, need, nonmono))
        grids.append(grid_markdown(p, need))
    with open("bidchart_condensed.txt", "w", encoding="utf-8") as f:
        f.write("\n\n".join(texts) + "\n")
    with open("bidchart_condensed.md", "w", encoding="utf-8") as f:
        f.write("Fewest small trumps (3-9) needed to bid 2 / 3 / 4 ('-' = never).\n"
                "Rows: which of A K Q J of trump you hold; columns: whether you also hold "
                "the 10 and/or the 2 of trump.\n\n" + "\n\n".join(grids) + "\n")
    print("wrote condensed_4p.csv, condensed_6p.csv, bidchart_condensed.txt, bidchart_condensed.md")


if __name__ == "__main__":
    main()
