"""Targeted GPU experiments with search players.

  worst2     the weakest hands that are still a profitable 2-bid: reruns the
             every-hand table's weakest-looking candidates with many more deals
  bareace    bare Ace of trump vs two non-Ace trumps: the same hands played on the
             same deals, naming the Ace's suit vs the two-card suit
  trumpdata  random hands played with every held suit as trump (same deals), for
             fitting the bidder's choice of trump (fit_trump.py)

    .venv/Scripts/python experiments.py worst2 -p 4
    .venv/Scripts/python experiments.py bareace -p 6
"""
import argparse
import random

import numpy as np
import pandas as pd

import allhands as ah
import setback as sb
import setback_fast as fast
import setback_gpu as gpu

RANKS = "23456789TJQKA"


def evaluate(hands, trumps, nplayers, deals, search=16, seed=777):
    """Bidder-team points, shape (hands, deals); same deals for the same hands and seed."""
    team = np.array(sb.make_teams(nplayers, 2), np.int64)
    rules = fast.rules_array(sb.team_rules(nplayers, list(team)))
    s = np.full(int(team.max()) + 1, search, np.int64)
    hands = np.asarray(hands, np.int64)
    trumps = np.asarray(trumps, np.int64)
    out = np.zeros((len(hands), deals), np.int64)
    step = max(1, 400_000 // deals)
    for a in range(0, len(hands), step):
        b = min(len(hands), a + step)
        dealt, lo = ah.deal_many(hands[a:b], trumps[a:b], deals, nplayers, seed, a)
        sc = gpu.play_batch(dealt, np.repeat(trumps[a:b], deals), team[lo], team, rules, s, seed=seed + a)
        out[a:b] = sc[:, team[0]].reshape(b - a, deals)
    return out


def ev(own, k):
    return np.where(own >= k, own, -k).mean(axis=-1)


def worst2(nplayers, deals_in_table, candidates=2000, deals=2048):
    df = pd.read_csv(f"allhands_{nplayers}p_d{deals_in_table}_s16.csv.gz")
    cand = df[df.ev2 >= -0.2].nsmallest(candidates, "mean")
    hands = [sum(1 << (s * 13 + RANKS.index(c[0])) for c in h.split() for s in ["SHDC".index(c[1])])
             for h in cand.hand]
    trumps = ["SHDC".index(t) for t in cand.trump]
    own = evaluate(hands, trumps, nplayers, deals)
    cand = cand.assign(mean2=own.mean(axis=1), p2_2=(own >= 2).mean(axis=1), ev2_2=ev(own, 2),
                       se=np.where(own >= 2, own, -2).std(axis=1) / np.sqrt(deals))
    pos = cand[cand.ev2_2 >= 0].nsmallest(6, "mean2")
    print(f"\n{nplayers}-handed: weakest hands that are still a profitable 2-bid "
          f"({candidates} candidates rerun with {deals} deals each)")
    for _, r in pos.iterrows():
        print(f"  {r.hand:20} trump {r.trump}   avg {r.mean2:.2f}  P(make 2) {r.p2_2:.2f}  "
              f"EV2 {r.ev2_2:+.2f} +/- {r.se:.2f}")
    print(f"  (of the {candidates} weakest-looking candidates, {(cand.ev2_2 >= 0).mean():.0%} are really "
          f"profitable at 2; break-even needs P(make 2) of about "
          f"{cand.loc[(cand.ev2_2.abs() < 0.05), 'p2_2'].mean():.2f})")


PATTERNS = ["KQ", "KJ", "KT", "K5", "K2", "QJ", "QT", "Q5", "Q2", "JT", "J5", "J2", "T5", "T2", "95", "52"]


def bareace(nplayers, nhands=1500, deals=128, seed=4242):
    rng = random.Random(seed)
    rows = []
    for pat in PATTERNS:
        hands, ace_suit, two_suit = [], [], []
        for _ in range(nhands):
            a, t, o1, o2 = rng.sample(range(4), 4)
            m = 1 << (a * 13 + 12)
            for r in pat:
                m |= 1 << (t * 13 + RANKS.index(r))
            # three more cards from the other two suits, no Aces (keeps the comparison clean)
            pool = [s * 13 + r for s in (o1, o2) for r in range(12)]
            for c in rng.sample(pool, 3):
                m |= 1 << c
            hands.append(m)
            ace_suit.append(a)
            two_suit.append(t)
        oa = evaluate(hands, ace_suit, nplayers, deals, seed=seed)
        ot = evaluate(hands, two_suit, nplayers, deals, seed=seed)  # same hands, same deals
        d = ev(ot, 2) - ev(oa, 2)
        rows.append((pat, ev(oa, 2).mean(), ev(ot, 2).mean(), d.mean(), d.std() / np.sqrt(nhands),
                     ev(oa, 3).mean(), ev(ot, 3).mean()))
    print(f"\n{nplayers}-handed: bare Ace vs two non-Ace trumps (same hands, same deals; "
          f"{nhands} hands x {deals} deals per holding; other 3 cards random, no Aces)")
    print("  two-card trump   EV2 naming bare A   EV2 naming the two   difference        EV3 A / two")
    for pat, ea, et, dm, se, ea3, et3 in rows:
        better = "two-card suit" if dm > 2 * se else "bare Ace" if dm < -2 * se else "even"
        print(f"  {pat[0]}-{pat[1]:12}   {ea:+.2f}               {et:+.2f}            {dm:+.2f} +/- {se:.2f}  "
              f"{ea3:+.2f} / {et3:+.2f}   -> {better}")


# ---------------------------------------------------------------- trump choice data

from numba import njit, prange  # noqa: E402
from engine_core import SM, bit, lowest  # noqa: E402


@njit(parallel=True, cache=True)
def deal_ids(hands, trumps, ids, deals, nplayers, seed):
    """Like allhands.deal_many, but the layouts depend only on ids[i], so the same
    hand gets the same layouts whichever suit is trump."""
    nh = hands.shape[0]
    out = np.zeros((nh * deals, nplayers), np.int64)
    lo_owner = np.zeros(nh * deals, np.int64)
    for g in prange(nh * deals):
        i = g // deals
        j = g % deals
        hand = hands[i]
        rest = np.empty(46, np.int64)
        k = 0
        for d in range(52):
            if not (hand >> d) & 1:
                rest[k] = d
                k += 1
        np.random.seed(seed + ids[i] * deals + j)
        perm = np.random.permutation(46)
        out[g, 0] = hand
        for q in range(1, nplayers):
            for t in range(6):
                out[g, q] |= bit(rest[perm[(q - 1) * 6 + t]])
        lo_c = 99
        for q in range(nplayers):
            tm = out[g, q] & SM[trumps[i]]
            if tm and lowest(tm) < lo_c:
                lo_c, lo_owner[g] = lowest(tm), q
    return out, lo_owner


def trumpdata(nplayers, nhands=40000, deals=64, search=16, seed=9090):
    """For random hands, play every suit the bidder holds as trump (same layouts per
    hand). Saves trumpdata_<P>p.npz: hands, and per suit mean/EV2/EV3 (NaN if not held)."""
    team = np.array(sb.make_teams(nplayers, 2), np.int64)
    rules = fast.rules_array(sb.team_rules(nplayers, list(team)))
    s_arr = np.full(int(team.max()) + 1, search, np.int64)
    rng = np.random.default_rng(seed)
    hands = np.array([sum(1 << int(c) for c in rng.choice(52, 6, replace=False)) for _ in range(nhands)],
                     np.int64)
    res = {k: np.full((nhands, 4), np.nan) for k in ("mean", "ev2", "ev3")}
    for suit in range(4):
        idx = np.nonzero((hands & SM[suit]) != 0)[0]
        step = max(1, 400_000 // deals)
        for a in range(0, len(idx), step):
            ids = idx[a:a + step]
            tr = np.full(len(ids), suit, np.int64)
            dealt, lo = deal_ids(hands[ids], tr, ids.astype(np.int64), deals, nplayers, seed)
            sc = gpu.play_batch(dealt, np.repeat(tr, deals), team[lo], team, rules, s_arr, seed=seed + suit * 7 + a)
            own = sc[:, team[0]].reshape(len(ids), deals)
            res["mean"][ids, suit] = own.mean(axis=1)
            res["ev2"][ids, suit] = ev(own, 2)
            res["ev3"][ids, suit] = ev(own, 3)
        print(f"  suit {suit} done ({len(idx)} hands)", flush=True)
    np.savez(f"trumpdata_{nplayers}p.npz", hands=hands, **res)
    print(f"wrote trumpdata_{nplayers}p.npz", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiment", choices=["worst2", "bareace", "trumpdata"])
    ap.add_argument("-p", "--players", type=int, default=4, choices=[4, 6])
    a = ap.parse_args()
    if a.experiment == "worst2":
        worst2(a.players, 224 if a.players == 4 else 112)
    elif a.experiment == "trumpdata":
        trumpdata(a.players)
    else:
        bareace(a.players)
