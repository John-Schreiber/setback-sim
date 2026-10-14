"""Whole games with an auction: which bidding strategy wins more games?

Each team bids by one strategy. Add "-ns" to a name for "never overcall your
partner", or "-big" for "overcall your partner only with a hand worth a full bid
more than needed" (by default a strategy takes the bid from its partner whenever
its own hand supports the higher bid), and "@T" to also bid 11 when the hand's chance of a
sweep (all 4 points and all 6 tricks, sweep.py) is at least T, e.g. cheap-ns@0.6:
  always2  bid 2 if nobody has bid yet, otherwise pass
  table    bid the highest bid with expected points >= 0 in the every-hand table
           (allhands.py), if it beats the current high bid
  cheap    bid just enough to beat the high bid (2 if none), if the table says
           that bid has expected points >= 0
  nopass   never pass: always bid just enough to beat the high bid (2 if none),
           whatever the hand (passes only over a 4)
  open2    bid 2 whenever nobody has bid, whatever the hand; over a bid, play as cheap
  take2    never pass a chance to bid 2: bid 2 if nobody has bid, and as dealer take a
           2 bid for 2 (dealer's privilege), whatever the hand; otherwise bid as table
  take2c   take2, but otherwise bid as cheap (raise only as far as needed)

Auction: one round, starting left of the dealer; each bid must beat the high
bid; if everyone passes the dealer is stuck with 2. With --dealer-takes the
dealer may instead take the bid at the current high when it is one of the listed
levels ("4" = only a 4, "2-4" = any); cheap and table do so when the table says
that bid is profitable. Bids are 2, 3, 4 and 11. The bidder names trump and
pitches. Scoring: the bidding team scores its points if it makes the bid, or
loses the bid if set; an 11-bid needs all 4 points and every trick and scores
11 (both sides then play for tricks). The other team scores what it takes. Dealer rotates. First team to the target wins; if both
cross on the same hand, the bidder goes out first. Every pairing plays the same
deals (seeded per game).

    .venv/Scripts/python auction.py --pairs cheap:cheap-ns,table:table-ns --dealer-takes 4
    .venv/Scripts/python auction.py --steal-analysis
"""
import argparse
import os

import numpy as np
import pandas as pd
from numba import njit, prange

import allhands as ah
import setback as sb
import setback_fast as fast
from engine_core import bit, pick_trump
from sweep import rules_for_11

STRATS = {"always2": 0, "table": 1, "cheap": 2, "nopass": 3, "open2": 4, "take2": 5, "take2c": 6}

# play_games output columns
(C_WIN, C_HANDS, C_S0, C_S1, C_AUC0, C_AUC1, C_SET0, C_SET1, C_STEAL0, C_STEAL1,
 C_STEALSET0, C_STEALSET1, C_DEALERNET, C_TOOK4, C_TOOK4MADE, C_B11_0, C_B11_1,
 C_B11MADE0, C_B11MADE1) = range(19)


@njit(parallel=True, cache=True)
def play_games(ngames, nplayers, target, strat, nosteal, t11, keys, maxbid, psweep,
               rules_by_bidder_team, rules11_by_bidder_team, team, seed, take_lo=0, take_hi=-1):
    """strat / nosteal / t11 (sweep chance needed to bid 11; > 1 = never): per team.
    rules_by_bidder_team[t]: rules to use when team t holds the bid (rules11_... for
    an 11-bid). The dealer may take the bid at the current high when it is between
    take_lo and take_hi. One row per game (see the C_* columns)."""
    nteams = team.max() + 1
    out = np.zeros((ngames, 19), np.int64)
    for gi in prange(ngames):
        np.random.seed(seed + gi)
        score = np.zeros(nteams, np.int64)
        dealer = gi % nplayers
        nh = 0
        winner = -1
        while winner < 0 and nh < 1000:
            perm = np.random.permutation(52)
            hands = np.zeros(nplayers, np.int64)
            for q in range(nplayers):
                for j in range(6):
                    hands[q] |= bit(perm[q * 6 + j])
            high, bidder, stole, took = 0, -1, False, False
            for k in range(1, nplayers + 1):
                p = (dealer + k) % nplayers
                st = strat[team[p]]
                if bidder >= 0 and team[bidder] == team[p] and nosteal[team[p]]:
                    if nosteal[team[p]] == 1:
                        continue  # partner holds the bid: leave it
                    # "-big": take it from partner only with a hand worth a full bid more
                    if maxbid[np.searchsorted(keys, ah.canonical(hands[p]))] < high + 2:
                        continue
                if high < 11 and t11[team[p]] <= 1.0:
                    if psweep[np.searchsorted(keys, ah.canonical(hands[p]))] >= t11[team[p]]:
                        stole = bidder >= 0 and team[bidder] == team[p]
                        high, bidder, took = 11, p, False
                        continue
                if high >= 11:
                    continue
                need = high + 1 if high >= 2 else 2
                if k == nplayers and take_lo <= high <= take_hi and high >= 2:
                    # dealer's privilege: take the bid at the current high
                    take = False
                    if (st == 5 or st == 6) and high == 2:
                        take = True
                    elif st in (1, 2, 6):
                        mx = maxbid[np.searchsorted(keys, ah.canonical(hands[p]))]
                        take = mx == high or ((st == 2 or st == 6) and mx >= high)
                    if take:
                        stole = team[bidder] == team[p]
                        bidder, took = p, True
                        continue
                if st == 0:
                    want = 2
                elif st == 3:
                    want = need if need <= 4 else 0
                elif (st == 4 or st == 5 or st == 6) and high == 0:
                    want = 2
                else:
                    mx = maxbid[np.searchsorted(keys, ah.canonical(hands[p]))]
                    if st == 1 or st == 5:
                        want = mx
                    else:
                        want = need if need <= mx else 0
                if want > high:
                    stole = bidder >= 0 and team[bidder] == team[p]
                    high, bidder, took = want, p, False
            if bidder < 0:  # everyone passed: dealer is stuck with 2
                high, bidder = 2, dealer
            trump = pick_trump(hands[bidder], nplayers)
            bt = team[bidder]
            change = np.zeros(nteams, np.int64)
            if high == 11:
                sc, won = fast.play_game_won(hands, bidder, trump, team, nteams, True, True,
                                             rules11_by_bidder_team[bt], np.zeros(nteams, np.int64))
                made = sc[bt] == 4
                for t in range(nteams):
                    if t != bt and won[t] != 0:
                        made = False
                change[bt] = 11 if made else -11
                out[gi, C_B11_0 + bt] += 1
                if made:
                    out[gi, C_B11MADE0 + bt] += 1
            else:
                sc = fast.play_game(hands, bidder, trump, team, nteams, True, True,
                                    rules_by_bidder_team[bt], np.zeros(nteams, np.int64))
                made = sc[bt] >= high
                change[bt] = sc[bt] if made else -high
            for t in range(nteams):
                if t != bt:
                    change[t] = sc[t]
            for t in range(nteams):
                score[t] += change[t]
            out[gi, C_AUC0 + bt] += 1
            if not made:
                out[gi, C_SET0 + bt] += 1
            if stole:
                out[gi, C_STEAL0 + bt] += 1
                if not made:
                    out[gi, C_STEALSET0 + bt] += 1
            if took and high == 4:
                out[gi, C_TOOK4] += 1
                if made:
                    out[gi, C_TOOK4MADE] += 1
            dt = team[dealer]
            for t in range(nteams):
                out[gi, C_DEALERNET] += change[t] if t == dt else -change[t]
            nh += 1
            dealer = (dealer + 1) % nplayers
            if score[bt] >= target:
                winner = bt
            else:
                best = -1
                for t in range(nteams):
                    if score[t] >= target and (best < 0 or score[t] > score[best]):
                        best = t
                winner = best
        out[gi, C_WIN] = winner
        out[gi, C_HANDS] = nh
        out[gi, C_S0] = score[0]
        out[gi, C_S1] = score[1]
    return out


@njit(parallel=True, cache=True)
def steal_pairs(ndeals, keys, maxbid, rules_bidteam0, team, seed):
    """Partner (seat 0) holds the bid at 2; I (seat 2) could take it at 3.
    Same deal played both ways. Returns per deal: partner's table max bid, my table
    max bid, and my team's net swing (our score change minus theirs) with me at 3
    minus with partner at 2."""
    out = np.zeros((ndeals, 3), np.int64)
    zero = np.zeros(2, np.int64)
    for i in prange(ndeals):
        np.random.seed(seed + i)
        perm = np.random.permutation(52)
        hands = np.zeros(4, np.int64)
        for q in range(4):
            for j in range(6):
                hands[q] |= bit(perm[q * 6 + j])
        net = np.zeros(2, np.int64)
        for v, (bidder, bid) in enumerate(((0, 2), (2, 3))):
            sc = fast.play_game(hands, bidder, pick_trump(hands[bidder], 4), team, 2, True, True,
                                rules_bidteam0, zero)
            ours = sc[0] if sc[0] >= bid else -bid
            net[v] = ours - sc[1]
        out[i, 0] = maxbid[np.searchsorted(keys, ah.canonical(hands[0]))]
        out[i, 1] = maxbid[np.searchsorted(keys, ah.canonical(hands[2]))]
        out[i, 2] = net[1] - net[0]
    return out


def load_maxbid(players, table):
    deals = 224 if players == 4 else 112
    fn = os.path.join(ah.HERE, f"allhands_{players}p_d{deals}_{table}.csv.gz")
    df = pd.read_csv(fn, usecols=["hand", "ev2", "ev3", "ev4"])
    keys, _ = ah.hand_classes()
    assert df.hand.iloc[0] == ah.hand_str(int(keys[0])) and df.hand.iloc[-1] == ah.hand_str(int(keys[-1]))
    return keys, np.select([df.ev4 >= 0, df.ev3 >= 0, df.ev2 >= 0], [4, 3, 2], 0).astype(np.int64), fn


def parse_strat(name):
    """'cheap-ns@0.6' -> (strategy code, never-steal flag, sweep chance needed to bid 11)."""
    name, _, t = name.partition("@")
    ns = 1 if name.endswith("-ns") else 2 if name.endswith("-big") else 0
    base = name.rsplit("-", 1)[0] if ns else name
    return STRATS[base], ns, float(t) if t else 2.0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--players", type=int, default=4, choices=[4])
    ap.add_argument("--games", type=int, default=200_000)
    ap.add_argument("--table", default="s0", help="every-hand table for the table strategies: s0 or s16")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--dealer-takes", default="none",
                    help='bid levels the dealer may take at the current high: "4", "3-4", "2-4" or "none"')
    ap.add_argument("--pairs", default="cheap:cheap-ns,table:table-ns,cheap:table",
                    help="comma-separated A:B strategy pairings")
    ap.add_argument("--targets", default="11,21")
    ap.add_argument("--steal-analysis", action="store_true",
                    help="same deal: partner plays at 2 vs you take it at 3, by both hands' strength")
    a = ap.parse_args()

    keys, maxbid, fn = load_maxbid(a.players, a.table)
    team = np.array(sb.make_teams(a.players, 2), np.int64)
    nteams = int(team.max()) + 1
    rules_bt = np.stack([fast.rules_array(sb.team_rules(a.players, list(team), bidder=b))
                         for b in range(nteams)])  # seat b is on team b for b < nteams
    rules11_bt = rules_for_11(a.players, team)
    z = np.load(os.path.join(ah.HERE, "sweep_4p_d224.npz"))
    assert np.array_equal(z["keys"], keys)
    psweep = z["p"].astype(np.float64)
    if a.dealer_takes == "none":
        take_lo, take_hi = 0, -1
    else:
        lo, _, hi = a.dealer_takes.partition("-")
        take_lo, take_hi = int(lo), int(hi or lo)

    if a.steal_analysis:
        n = a.games * 10
        o = steal_pairs(n, keys, maxbid, rules_bt[0], team, a.seed)
        df = pd.DataFrame(o, columns=["partner", "me", "gain"])
        lab = {0: "pass", 2: "2", 3: "3", 4: "4"}
        print(f"Partner holds the bid at 2; you could take it at 3. Your team's net gain from taking it, "
              f"points per hand ({n:,} deals, rule-based players; hands rated by {os.path.basename(fn)}):\n")
        t = df.groupby(["partner", "me"]).gain.agg(["mean", "sem", "size"])
        print(f"{'partner hand':>12} {'your hand':>10}   gain from stealing   share of deals")
        for (pm, mm), r in t.iterrows():
            print(f"{'worth ' + lab[pm]:>12} {'worth ' + lab[mm]:>10}   {r['mean']:+.3f} +/- {r['sem']:.3f}"
                  f"        {r['size'] / n:5.1%}")
        return

    print(f"{a.players}-handed, {a.games:,} games per pairing, rule-based players, table strategies read "
          f"{os.path.basename(fn)}; dealer's privilege: "
          f"{a.dealer_takes if a.dealer_takes != 'none' else 'none'}")
    for target in (int(t) for t in a.targets.split(",")):
        print(f"\n--- games to {target} ---")
        print(f"{'team A':>15} vs {'team B':<15} A wins        A pts/hand  B pts/hand  steals A/B   "
              f"set when stealing A/B   dealer net/hand  dealer takes 4 (made)  11-bids A/B per 1000 (made)")
        for pair in a.pairs.split(","):
            sa, sb_ = pair.split(":")
            (ca, na, ta), (cb, nb, tb) = parse_strat(sa), parse_strat(sb_)
            o = play_games(a.games, a.players, target, np.array([ca, cb], np.int64),
                           np.array([na, nb], np.int64), np.array([ta, tb]), keys, maxbid, psweep,
                           rules_bt, rules11_bt, team, a.seed, take_lo, take_hi)
            tot = o.sum(axis=0)
            hands = tot[C_HANDS]
            win = (o[:, C_WIN] == 0).mean()
            se = np.sqrt(win * (1 - win) / a.games)
            ss = lambda t: tot[C_STEALSET0 + t] / max(1, tot[C_STEAL0 + t])
            m11 = lambda t: tot[C_B11MADE0 + t] / max(1, tot[C_B11_0 + t])
            print(f"{sa:>15} vs {sb_:<15} {win:6.1%} +/-{se:.1%}  {tot[C_S0] / hands:+.3f}      {tot[C_S1] / hands:+.3f}"
                  f"      {tot[C_STEAL0] / hands:4.1%}/{tot[C_STEAL1] / hands:4.1%}   "
                  f"{ss(0):5.1%}/{ss(1):5.1%}            {tot[C_DEALERNET] / hands:+.3f}          "
                  f"{tot[C_TOOK4] / hands:4.2%} ({tot[C_TOOK4MADE] / max(1, tot[C_TOOK4]):.0%})          "
                  f"{1000 * tot[C_B11_0] / hands:.2f} ({m11(0):.0%}) / {1000 * tot[C_B11_1] / hands:.2f} ({m11(1):.0%})")


if __name__ == "__main__":
    main()
