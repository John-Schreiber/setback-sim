"""Whole games with look-ahead players, cheaply: a per-deal outcome cache.

A hand's play doesn't depend on the bid, only on the deal, who won the auction and the
trump they name. And game.play_games draws its deals from one seeded stream per game, so
deal j of game g is the same whatever the bidding strategies. So each deal is played once
per possible bidder (with search players, on the GPU) and stored; then any number of
bidding strategies run as auctions and scoring over the stored outcomes.

Deals are cached lazily: a game that reaches a deal not yet played stops and asks for it;
the missing deals are played and the run repeats until no game asks. With rule-based
players (search 0) the cached games equal game.play_games exactly (tests/test_gamecache.py).
"""
import numpy as np
from numba import njit, prange

from cardsim.pitch import fast
from cardsim.pitch.core import bit, lowest
from cardsim.pitch.game import (C_AUC0, C_HANDS, C_MOON0, C_MOONMADE0, C_REDEALS, C_S0, C_SET0, C_WIN, N_COLS,
                                hand_key)
from cardsim.pitch.tables import F_ESM, F_POWER, G_DECK

C_NEED = N_COLS  # extra column: the deal a game needed and didn't find (-1 = finished)


@njit(cache=True)
def _deck(gf):
    cards = np.empty(54, np.int64)
    nd = 0
    for c in range(54):
        if (gf[G_DECK] >> c) & 1:
            cards[nd] = c
            nd += 1
    return cards, nd


@njit(parallel=True, cache=True)
def deals_for(games, j0, j1, nplayers, rp, team, tab, gf, tw, tl, tx, seed):
    """For each game index games[i], deals j0[i] .. j1[i]-1 and every possible bidder: the
    hands as played (after any refill for that bidder's trump), the trump, and the team dealt
    the lowest trump in play. Flattened in order (game, deal, bidder)."""
    deal_size, hand_size, do_refill = rp[0], rp[1], rp[2]
    cards, nd = _deck(gf)
    count = np.zeros(len(games) + 1, np.int64)
    for i in range(len(games)):
        count[i + 1] = count[i] + (j1[i] - j0[i]) * nplayers
    tot = count[-1]
    hands_out = np.zeros((tot, nplayers), np.int64)
    trumps = np.zeros(tot, np.int64)
    lo_team = np.zeros(tot, np.int64)
    for i in prange(len(games)):
        gi = games[i]
        np.random.seed(seed + gi)
        at = count[i]
        for j in range(j1[i]):
            perm = np.random.permutation(nd)
            if j < j0[i]:
                continue
            dealer = (gi + j) % nplayers
            for b in range(nplayers):
                hands = np.zeros(nplayers, np.int64)
                for q in range(nplayers):
                    for k in range(deal_size):
                        hands[q] |= bit(cards[perm[q * deal_size + k]])
                t = fast.pick_trump(hands[b], tab, tw, tl, tx)
                if do_refill or deal_size > hand_size:
                    used = nplayers * deal_size
                    stock = np.empty(nd - used, np.int64)
                    for k in range(nd - used):
                        stock[k] = cards[perm[used + k]]
                    fast.refill(hands, stock, nd - used, t, hand_size, (dealer + 1) % nplayers, tab)
                lo_p, lt = 1 << 30, 0
                for q in range(nplayers):  # as play_game_won: who holds the lowest trump in play
                    x = hands[q] & tab[t, F_ESM, t]
                    while x:
                        c = lowest(x)
                        x &= x - 1
                        if tab[t, F_POWER, c] < lo_p:
                            lo_p, lt = tab[t, F_POWER, c], team[q]
                hands_out[at] = hands
                trumps[at] = t
                lo_team[at] = lt
                at += 1
    return hands_out, trumps, lo_team


@njit(parallel=True, cache=True)
def play_cached(ngames, nplayers, team, strat, nosteal, rp, take, maxbid, moon, honor_cards, tab, gf, tw, tl, tx,
                seed, pts, swept, have):
    """game.play_games with each hand's outcome read from the cache: pts[g, j, bidder, team],
    swept[g, j, bidder] (the bidder's team took every trick), deals j < have[g] cached. A game
    reaching an uncached deal stops with column C_NEED = that deal."""
    deal_size, min_bid, max_bid = rp[0], rp[3], rp[4]
    moon_bid, moon_value, moon_wins, stuck, target = rp[5], rp[6], rp[7], rp[8], rp[9]
    win_by, bidder_out, no_junk, exact_bonus, backdoor, ptotal = rp[10], rp[11], rp[12], rp[13], rp[14], rp[15]
    nteams = team.max() + 1
    cards, nd = _deck(gf)
    out = np.zeros((ngames, N_COLS + 1), np.int64)
    for gi in prange(ngames):
        out[gi, C_NEED] = -1
        np.random.seed(seed + gi)
        score = np.zeros(nteams, np.int64)
        dealer = gi % nplayers
        nh, winner, j = 0, -1, 0
        while winner < 0 and nh < 1000:
            if j >= have[gi]:
                out[gi, C_NEED] = j
                break
            perm = np.random.permutation(nd)
            hands = np.zeros(nplayers, np.int64)
            for q in range(nplayers):
                for k in range(deal_size):
                    hands[q] |= bit(cards[perm[q * deal_size + k]])
            high, bidder = 0, -1
            for k in range(1, nplayers + 1):
                p = (dealer + k) % nplayers
                st = strat[team[p]]
                if bidder >= 0 and team[bidder] == team[p] and nosteal[team[p]]:
                    continue
                t = fast.pick_trump(hands[p], tab, tw, tl, tx)
                key = hand_key(hands[p], t, honor_cards, tab)
                mx = maxbid[team[p], key]
                if moon_bid > 0 and moon[team[p], key] and high < moon_bid:
                    high, bidder = moon_bid, p
                    continue
                if high >= max_bid:
                    continue
                if k == nplayers and high >= min_bid and take[high] and mx >= high:
                    bidder = p
                    continue
                need = high + 1 if high >= min_bid else min_bid
                want = mx if st == 1 else (need if need <= mx else 0)
                if want >= need and want > high:
                    high, bidder = want, p
            if bidder < 0:
                if not stuck:
                    out[gi, C_REDEALS] += 1
                    dealer = (dealer + 1) % nplayers
                    j += 1
                    continue
                high, bidder = min_bid, dealer
            bt = team[bidder]
            sc = pts[gi, j, bidder]
            change = np.zeros(nteams, np.int64)
            if high == moon_bid and moon_bid > 0:
                made = sc[bt] == ptotal and swept[gi, j, bidder] == 1
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
            j += 1
            dealer = (dealer + 1) % nplayers
            if winner >= 0:
                break
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


class Cache:
    """The outcome cache for `ngames` games of one variant, filled on demand."""

    def __init__(self, game, ngames, seed, search, max_deals=64, log=print):
        from cardsim.pitch.game import rule_params
        self.g, self.ngames, self.seed, self.log = game, ngames, seed, log
        self.search = np.asarray(search, np.int64)
        self.rp = rule_params(game.rules)
        n, nt = game.nplayers, game.nteams
        self.pts = np.zeros((ngames, max_deals, n, nt), np.int8)
        self.swept = np.zeros((ngames, max_deals, n), np.int8)
        self.have = np.zeros(ngames, np.int64)
        self.played = 0
        self.batch = 0

    def fill(self, games, upto):
        """Play deals have[g] .. upto[g]-1 of these games for every bidder."""
        g = self.g
        n = g.nplayers
        upto = np.minimum(upto, self.pts.shape[1])
        j0 = self.have[games]
        hands, trumps, lo = deals_for(games, j0, upto, n, self.rp, g.team, g.tab, g.gf, g.tw, g.tl, g.tx,
                                      self.seed)
        bidder = np.tile(np.arange(n), len(hands) // n)
        sc = np.zeros((len(hands), g.nteams), np.int64)
        won = np.zeros((len(hands), g.nteams), np.int64)
        for b in range(n):  # one batch per bidding seat (play_batch takes one bidder)
            sel = np.flatnonzero(bidder == b)
            if not len(sel):
                continue
            if self.search.max() > 0:
                from cardsim.pitch import gpu
                s_, w_ = gpu.play_batch(hands[sel], trumps[sel], lo[sel], g.team, g.rules_bt[g.team[b]], self.search,
                                        g.tab, g.gf, bidder=b, seed=self.seed, return_won=True,
                                        game_offset=self.batch)
                self.batch += len(sel)
            else:
                s_, w_ = _play_rules(hands[sel], trumps[sel], b, g.team, g.nteams, g.rules_bt[g.team[b]], g.tab, g.gf)
            sc[sel], won[sel] = s_, w_
        at = 0
        for i, gi in enumerate(games):
            for j in range(j0[i], upto[i]):
                for b in range(n):
                    bt = g.team[b]
                    self.pts[gi, j, b] = sc[at]
                    self.swept[gi, j, b] = all(won[at, k] == 0 for k in range(g.nteams) if k != bt)
                    at += 1
            self.have[gi] = upto[i]
        self.played += len(hands)

    def run(self, strat, nosteal, take, maxbid, moon, honor_cards):
        """play_cached for one set of bidding tables, filling the cache until every game finishes."""
        g = self.g
        while True:
            out = play_cached(self.ngames, g.nplayers, g.team, strat, nosteal, self.rp, take, maxbid, moon,
                              honor_cards, g.tab, g.gf, g.tw, g.tl, g.tx, self.seed, self.pts, self.swept, self.have)
            need = out[:, C_NEED]
            games = np.flatnonzero(need >= 0)
            if not len(games):
                return out[:, :C_NEED]
            if (need[games] >= self.pts.shape[1]).any():
                raise RuntimeError("a game needs more deals than the cache holds (raise max_deals)")
            self.fill(games, need[games] + 4)  # a few deals ahead, so fewer refills


@njit(parallel=True, cache=True)
def _play_rules(hands, trumps, bidder, team, nteams, rules, tab, gf):
    zero = np.zeros(nteams, np.int64)
    sc = np.zeros((hands.shape[0], nteams), np.int64)
    won = np.zeros((hands.shape[0], nteams), np.int64)
    for i in prange(hands.shape[0]):
        s_, w_, _ = fast.play_game_won(hands[i], bidder, trumps[i], team, nteams, True, rules, zero, tab, gf)
        sc[i], won[i] = s_, w_
    return sc, won
