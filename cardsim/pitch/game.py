"""Whole games with an auction, for any house-rule preset.

Bidding uses the variant's chart (bidchart.py): each player reduces their dealt hand
to trump honors + small trumps and looks up the highest profitable bid (and whether a
moon bid is worth it). Strategies, per team (as in auction.py):
  table   bid your highest profitable bid if it beats the high bid
  cheap   bid just enough to beat the high bid, if that bid is profitable
  add "-ns" to never overcall your partner.
--pairs A:B plays team A's strategy against B on every other team (6-handed: two teams).
Rules handled: bid range, moon bid (value or game win), stuck dealer or redeal,
dealer's privilege at listed levels, discard and refill, target, win by, bidder goes
out first, no junk points, exact-bid bonus, backdoor loss.

    .venv/Scripts/python -m cardsim.pitch.game --preset ten_point --pairs cheap:table
"""
import argparse
import os

import numpy as np
import pandas as pd
from numba import njit, prange

from cardsim.pitch import PitchRules
from cardsim.pitch import fast
from cardsim.pitch.bidchart import OUT, honor_list
from cardsim.pitch.core import bit, lowest, popcount
from cardsim.pitch.tables import F_ESM, F_POWER, G_DECK

STRATS = {"table": 1, "cheap": 2}
MAX_SMALL = 10
# out columns: winner, hands, redeals, then one block of MAXT columns per counter (team k at +k)
MAXT = 3
C_WIN, C_HANDS, C_REDEALS = range(3)
C_S0, C_AUC0, C_SET0, C_MOON0, C_MOONMADE0 = (3 + MAXT * i for i in range(5))
N_COLS = 3 + 5 * MAXT


class Brain:
    """Bid lookup for one variant: key = honors bitmask * MAX_SMALL + small trumps."""

    def __init__(self, rules, nplayers):
        df = pd.read_csv(os.path.join(OUT, f"{rules.name}_{nplayers}p.csv"), keep_default_na=False)
        hon = honor_list(rules)
        self.honor_cards = np.array([[f(t) for _, f in hon] for t in range(4)], np.int64)
        H = len(hon)
        names = [lab for lab, _ in hon]
        maxbid = np.zeros((1 << H, MAX_SMALL), np.int64)
        moon = np.zeros((1 << H, MAX_SMALL), np.int64)
        seen = np.zeros((1 << H, MAX_SMALL), bool)
        for r in df.itertuples():
            m = sum(1 << names.index(x) for x in str(r.honors).split()) if r.honors else 0
            s = min(int(r.small), MAX_SMALL - 1)
            maxbid[m, s] = r.max_bid
            moon[m, s] = int(rules.moon_bid > 0 and r.ev_moon > 0) if "ev_moon" in df else 0
            seen[m, s] = True
        # hands the survey never saw: take the best bid of any weaker hand that was seen
        # (fewer honors and/or fewer small trumps); adding cards never lowers a bid here
        for m in sorted(range(1 << H), key=lambda x: bin(x).count("1")):
            for s in range(MAX_SMALL):
                if seen[m, s]:
                    continue
                cands = [(maxbid[m, s - 1], moon[m, s - 1])] if s else []
                cands += [(maxbid[m & ~(1 << b), s], moon[m & ~(1 << b), s]) for b in range(H) if m >> b & 1]
                if cands:
                    maxbid[m, s] = max(c[0] for c in cands)
                    moon[m, s] = max(c[1] for c in cands)
        self.maxbid = maxbid.ravel()
        self.moon = moon.ravel()


@njit(cache=True)
def hand_key(dealt, t, honor_cards, tab):
    m, nh = 0, 0
    for i in range(honor_cards.shape[1]):
        if (dealt >> honor_cards[t, i]) & 1:
            m |= 1 << i
            nh += 1
    small = popcount(dealt & tab[t, F_ESM, t]) - nh
    return m * MAX_SMALL + min(small, MAX_SMALL - 1)


@njit(parallel=True, cache=True)
def play_games(ngames, nplayers, team, strat, nosteal, rp, take, maxbid, moon, honor_cards,
               rules_bt, tab, gf, tw, tl, tx, seed):
    """rp = [deal_size, hand_size, refill, min_bid, max_bid, moon_bid, moon_value,
    moon_wins, dealer_stuck, target, win_by, bidder_out, no_junk, exact_bonus, backdoor,
    points_total]; take[level] = 1 if the dealer may take that bid level."""
    deal_size, hand_size, do_refill, min_bid, max_bid = rp[0], rp[1], rp[2], rp[3], rp[4]
    moon_bid, moon_value, moon_wins, stuck, target = rp[5], rp[6], rp[7], rp[8], rp[9]
    win_by, bidder_out, no_junk, exact_bonus, backdoor, ptotal = rp[10], rp[11], rp[12], rp[13], rp[14], rp[15]
    nteams = team.max() + 1
    zero = np.zeros(nteams, np.int64)
    out = np.zeros((ngames, N_COLS), np.int64)
    for gi in prange(ngames):
        np.random.seed(seed + gi)
        score = np.zeros(nteams, np.int64)
        dealer = gi % nplayers
        nh, winner = 0, -1
        deck = gf[G_DECK]
        cards = np.empty(54, np.int64)
        nd = 0
        for c in range(54):
            if (deck >> c) & 1:
                cards[nd] = c
                nd += 1
        while winner < 0 and nh < 1000:
            perm = np.random.permutation(nd)
            hands = np.zeros(nplayers, np.int64)
            for q in range(nplayers):
                for j in range(deal_size):
                    hands[q] |= bit(cards[perm[q * deal_size + j]])
            # auction: one round from the dealer's left
            high, bidder = 0, -1
            for k in range(1, nplayers + 1):
                p = (dealer + k) % nplayers
                st = strat[team[p]]
                if bidder >= 0 and team[bidder] == team[p] and nosteal[team[p]]:
                    continue
                t = fast.pick_trump(hands[p], tab, tw, tl, tx)
                key = hand_key(hands[p], t, honor_cards, tab)
                mx = maxbid[key]
                if moon_bid > 0 and moon[key] and high < moon_bid:
                    high, bidder = moon_bid, p
                    continue
                if high >= max_bid:
                    continue
                if k == nplayers and high >= min_bid and take[high] and mx >= high:
                    bidder = p  # dealer's privilege: take it at the current high
                    continue
                need = high + 1 if high >= min_bid else min_bid
                want = mx if st == 1 else (need if need <= mx else 0)
                if want >= need and want > high:
                    high, bidder = want, p
            if bidder < 0:
                if not stuck:
                    out[gi, C_REDEALS] += 1
                    dealer = (dealer + 1) % nplayers
                    continue
                high, bidder = min_bid, dealer
            t = fast.pick_trump(hands[bidder], tab, tw, tl, tx)
            if do_refill or deal_size > hand_size:
                used = nplayers * deal_size
                stock = np.empty(nd - used, np.int64)
                for i in range(nd - used):
                    stock[i] = cards[perm[used + i]]
                fast.refill(hands, stock, nd - used, t, hand_size, (dealer + 1) % nplayers, tab)
            bt = team[bidder]
            sc, won, _ = fast.play_game_won(hands, bidder, t, team, nteams, True, rules_bt[bt], zero, tab, gf)
            change = np.zeros(nteams, np.int64)
            if high == moon_bid and moon_bid > 0:
                made = sc[bt] == ptotal
                for k2 in range(nteams):
                    if k2 != bt and won[k2] != 0:
                        made = False
                change[bt] = moon_value if made else -moon_value
                out[gi, C_MOON0 + bt] += 1
                if made:
                    out[gi, C_MOONMADE0 + bt] += 1
                    if moon_wins:
                        winner = bt
            else:
                made = sc[bt] >= high
                change[bt] = sc[bt] + (exact_bonus if sc[bt] == high else 0) if made else -high
            if not made:
                out[gi, C_SET0 + bt] += 1
            for k2 in range(nteams):
                if k2 != bt and not no_junk:
                    change[k2] = sc[k2]
            for k2 in range(nteams):
                score[k2] += change[k2]
            out[gi, C_AUC0 + bt] += 1
            nh += 1
            dealer = (dealer + 1) % nplayers
            if winner >= 0:
                break
            # winning: reach the target, ahead of everyone by win_by; the bidder first
            best, bs = -1, -(1 << 60)
            for k2 in range(nteams):
                ahead = True
                for k3 in range(nteams):
                    if k3 != k2 and score[k2] - score[k3] < win_by:
                        ahead = False
                if score[k2] >= target and ahead:
                    if bidder_out and k2 == bt:
                        best, bs = k2, 1 << 60
                    elif score[k2] > bs:
                        best, bs = k2, score[k2]
            if best >= 0:
                winner = best
            elif backdoor < 0:
                for k2 in range(nteams):
                    if score[k2] <= backdoor:
                        top, ts = -1, -(1 << 60)
                        for k3 in range(nteams):
                            if k3 != k2 and score[k3] > ts:
                                top, ts = k3, score[k3]
                        winner = top
        out[gi, C_WIN] = winner
        out[gi, C_HANDS] = nh
        for k2 in range(nteams):
            out[gi, C_S0 + k2] = score[k2]
    return out


def rule_params(r):
    return np.array([r.deal_size, r.hand_size, int(r.refill), r.min_bid, r.max_bid, r.moon_bid, r.moon_value,
                     int(r.moon_wins_game), int(r.dealer_stuck), r.target, r.win_by, int(r.bidder_out_first),
                     int(r.no_junk), r.exact_bonus, r.backdoor, r.points_total], np.int64)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="setback4")
    ap.add_argument("-p", "--players", type=int, default=4)
    ap.add_argument("--games", type=int, default=100_000)
    ap.add_argument("--pairs", default="cheap:table,cheap:cheap-ns")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()
    rules = PitchRules.preset(a.preset)
    g = fast.Game(rules, a.players)
    brain = Brain(rules, a.players)
    take = np.zeros(64, np.int64)
    for lv in rules.dealer_takes:
        take[lv] = 1
    rp = rule_params(rules)
    print(f"{a.preset}, {a.players}-handed, {a.games:,} games per pairing (rule-based play); "
          f"bids {rules.min_bid}-{rules.max_bid}{f', moon {rules.moon_bid}' if rules.moon_bid else ''}, "
          f"target {rules.target}")
    print(f"{'team A':>10} vs {'team B':<10} A wins          hands/game  A sets  B sets  moon bids A/B per 1000 (made)")
    for pair in a.pairs.split(","):
        sa, sb = pair.split(":")
        sides = (sa,) + (sb,) * (g.nteams - 1)  # team A against B on every other team
        strat = np.array([STRATS[x.replace("-ns", "")] for x in sides], np.int64)
        ns = np.array([int(x.endswith("-ns")) for x in sides], np.int64)
        o = play_games(a.games, a.players, g.team, strat, ns, rp, take, brain.maxbid, brain.moon,
                       brain.honor_cards, g.rules_bt, g.tab, g.gf, g.tw, g.tl, g.tx, a.seed)
        tot = o.sum(axis=0)
        win = (o[:, C_WIN] == 0).mean()
        se = np.sqrt(win * (1 - win) / a.games)
        hands = tot[C_HANDS]
        mm = lambda k: tot[C_MOONMADE0 + k] / max(1, tot[C_MOON0 + k])
        print(f"{sa:>10} vs {sb:<10} {win:6.1%} +/-{se:.1%}   {o[:, C_HANDS].mean():6.1f}     "
              f"{tot[C_SET0] / max(1, tot[C_AUC0]):5.1%}  {tot[C_SET0 + 1] / max(1, tot[C_AUC0 + 1]):5.1%}   "
              f"{1000 * tot[C_MOON0] / hands:.2f} ({mm(0):.0%}) / {1000 * tot[C_MOON0 + 1] / hands:.2f} ({mm(1):.0%})")


if __name__ == "__main__":
    main()
