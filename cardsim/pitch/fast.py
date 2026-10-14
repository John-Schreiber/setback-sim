"""CPU (Numba) engine for Pitch under any house rules: dealing, whole games, search.

Built on core.py; the GPU engine runs the same core. `Game(rules, nplayers)`
bundles the tables, flags and default player rules for one variant.
"""
import numpy as np
from numba import njit, prange

from cardsim.pitch import core
from cardsim.pitch.core import (bit, heur_pick, lowest, note_void, options, popcount, rollout,
                                score_into, search_value, trick_winner)
from cardsim.pitch.tables import (F_ESM, F_ESUIT, F_NRANK, F_POWER, F_PTS, G_DECK, build_flags,
                                  build_tables)


# ---------------------------------------------------------------- variant bundle

class Game:
    """Tables, flags, teams and rule-based player settings for one variant."""

    def __init__(self, rules, nplayers=4, team_size=2):
        import setback as sb
        from trump_weights import TL, TW
        self.rules, self.nplayers = rules, nplayers
        self.tab = build_tables(rules)
        self.gf = build_flags(rules, nplayers)
        self.team = np.array([i % (nplayers // team_size) for i in range(nplayers)], np.int64)
        self.nteams = int(self.team.max()) + 1
        self.tw = TW[1 if nplayers == 6 else 0].copy()
        self.tl = TL[1 if nplayers == 6 else 0].copy()

        def rules_for(bidder_seat):
            return np.array([rules_row(r) for r in sb.team_rules(nplayers, list(self.team), bidder_seat)],
                            np.int64)
        # player rules when the bid is held by team b (seat b is on team b for b < nteams)
        self.rules_bt = np.stack([rules_for(b) for b in range(self.nteams)])


def rules_row(r):
    """setback.py RULES dict -> one row of the engine's rules array."""
    return (core.PITCH_CODES[r["pitch"]], r["ruff_min"], int(r["pull"]), bool(r["lead_aces"]),
            bool(r.get("schoolboy", False)), r.get("bank_low", 0), r.get("bank_jack", 0),
            r.get("lead10", 0), r.get("presmear", 0), bool(r.get("pull_t2", False)))


# ---------------------------------------------------------------- trump choice and the deal

@njit(cache=True)
def pick_trump(hand, tab, tw, tl):
    """The suit the bidder names: per-rank weights and a length bonus (trump_weights.py).
    The off-jack and jokers count for the suit they would join; the 3/5/9/10 of trump
    add 100 per point they're worth. Ties go to the lower suit."""
    best, bv = -1, np.int64(-(1 << 60))
    for s in range(4):
        tm = hand & tab[s, F_ESM, s]
        if tm == 0:
            continue
        v = tl[min(popcount(tm), 6)]
        x = tm
        while x:
            c = lowest(x)
            x &= x - 1
            if c < 52 and c // 13 == s:
                v += tw[c % 13]
                if tab[s, F_NRANK, c] != 11:
                    v += 100 * tab[s, F_PTS, c]
            else:
                v += tw[9]  # off-jack or joker: about a Jack
        if v > bv:
            best, bv = s, v
    return best


@njit(cache=True)
def refill(hands, stock, nstock, t, hand_size, first, tab):
    """Everyone discards non-trumps and draws back to hand_size from the stock,
    starting with seat `first`. Keeps point cards first, then the highest trumps,
    if a player holds too many; keeps their best non-trumps if the stock runs out."""
    n = hands.shape[0]
    pos = 0
    tm_all = tab[t, F_ESM, t]
    for k in range(n):
        p = (first + k) % n
        tm = hands[p] & tm_all
        keep = np.int64(0)
        for _ in range(min(hand_size, popcount(tm))):
            best, bk = -1, -1
            x = tm & ~keep
            while x:
                c = lowest(x)
                x &= x - 1
                key = tab[t, F_PTS, c] * 1000 + tab[t, F_POWER, c]
                if key > bk:
                    best, bk = c, key
            keep |= bit(best)
        need = hand_size - popcount(keep)
        while need > 0 and pos < nstock:
            keep |= bit(stock[pos])
            pos += 1
            need -= 1
        rest = hands[p] & ~tm_all
        while need > 0 and rest:
            best, bk = -1, -1
            x = rest
            while x:
                c = lowest(x)
                x &= x - 1
                if tab[t, F_POWER, c] > bk:
                    best, bk = c, tab[t, F_POWER, c]
            keep |= bit(best)
            rest &= ~bit(best)
            need -= 1
        hands[p] = keep
    return pos


@njit(cache=True)
def deal(seed, nplayers, deal_size, hand_size, do_refill, bidder, tab, gf, tw, tl):
    """Shuffle (seeded) and deal; the bidder names trump from the dealt cards; then the
    discard-and-refill if the rules have one. Returns (hands, trump)."""
    np.random.seed(seed)
    deck = gf[G_DECK]
    cards = np.empty(54, np.int64)
    nd = 0
    for c in range(54):
        if (deck >> c) & 1:
            cards[nd] = c
            nd += 1
    perm = np.random.permutation(nd)
    hands = np.zeros(nplayers, np.int64)
    for q in range(nplayers):
        for j in range(deal_size):
            hands[q] |= bit(cards[perm[q * deal_size + j]])
    t = pick_trump(hands[bidder], tab, tw, tl)
    if do_refill or deal_size > hand_size:
        used = nplayers * deal_size
        stock = np.empty(nd - used, np.int64)
        for i in range(nd - used):
            stock[i] = cards[perm[used + i]]
        refill(hands, stock, nd - used, t, hand_size, bidder, tab)
    return hands, t


# ---------------------------------------------------------------- whole games

@njit(cache=True)
def play_game_won(hands0, bidder, t, team, nteams, track, rules, search, tab, gf):
    """Play one hand. Returns (points per team, cards each team captured, team that
    took the last trick). search: sampled deals per decision, per team (0 = rules)."""
    n = hands0.shape[0]
    lo_c, lo_p, lo_team = -1, 1 << 30, 0
    for q in range(n):
        x = hands0[q] & tab[t, F_ESM, t]
        while x:
            c = lowest(x)
            x &= x - 1
            if tab[t, F_POWER, c] < lo_p:
                lo_c, lo_p, lo_team = c, tab[t, F_POWER, c], team[q]
    hands = hands0.copy()
    gone = np.int64(0)
    won = np.zeros(nteams, np.int64)
    voids = np.zeros(n, np.int64)
    tp = np.zeros(n, np.int64)
    tc = np.zeros(n, np.int64)
    tlen = 0
    leader = bidder
    last_team = -1
    while True:
        if tlen == n:
            wp, _ = trick_winner(tp, tc, tlen, t, tab)
            for i in range(n):
                gone |= bit(tc[i])
                won[team[wp]] |= bit(tc[i])
            tlen = 0
            leader = wp
            last_team = team[wp]
            if hands[leader] == 0:
                break
        p = (leader + tlen) % n
        if search[team[p]] > 0:
            c = search_pick(p, hands, tp, tc, tlen, gone, won, voids, t, team, n, track, rules,
                            search[team[p]], tab, gf)
        else:
            c = heur_pick(p, hands, tp, tc, tlen, gone, voids, t, team, n, rules, tab, gf)
        hands[p] &= ~bit(c)
        if track:
            note_void(voids, p, tc, tlen, c, t, tab, gf)
        tp[tlen] = p
        tc[tlen] = c
        tlen += 1
    sc = np.zeros(nteams, np.int64)
    score_into(won, t, lo_team, last_team, sc, tab, gf)
    return sc, won, last_team


@njit(cache=True)
def play_game(hands0, bidder, t, team, nteams, track, rules, search, tab, gf):
    return play_game_won(hands0, bidder, t, team, nteams, track, rules, search, tab, gf)[0]


# ---------------------------------------------------------------- search player

@njit(cache=True)
def sample_world(me, hands, unseen, nun, voids, n, tab, t):
    """Deal the unseen cards to the other players, honoring known (effective-suit)
    voids. Leftover unseen cards are the undealt stock."""
    out = np.zeros(n, np.int64)
    out[me] = hands[me]
    order = np.empty(n - 1, np.int64)
    k = 0
    for nv in range(4, -1, -1):  # most constrained first
        for q in range(n):
            if q != me and popcount(voids[q]) == nv:
                order[k] = q
                k += 1
    for attempt in range(31):
        perm = np.random.permutation(nun)
        used = np.zeros(nun, np.bool_)
        ok = True
        for j in range(n - 1):
            q = order[j]
            need = popcount(hands[q])
            got = 0
            out[q] = 0
            for i in range(nun):
                if got == need:
                    break
                d = unseen[perm[i]]
                if used[perm[i]]:
                    continue
                if attempt < 30 and ((voids[q] >> tab[t, F_ESUIT, d]) & 1):
                    continue  # last attempt ignores voids (shouldn't be needed)
                out[q] |= bit(d)
                used[perm[i]] = True
                got += 1
            if got < need:
                ok = False
                break
        if ok:
            return out
    return out


@njit(cache=True)
def search_values(p, hands, tp, tc, tlen, gone, won, voids, t, team, n, track, rules, nsamp, tab, gf):
    """Legal cards (mask) and each card's summed search value (indexed by card)."""
    hand = hands[p]
    opts = options(hand, gone, tc, tlen, t, tab, gf)
    totals = np.zeros(54, np.float64)
    if popcount(opts) == 1:
        return opts, totals
    seen = gone | hand
    for i in range(tlen):
        seen |= bit(tc[i])
    unseen = np.empty(54, np.int64)
    nun = 0
    rest = gf[G_DECK] & ~seen
    while rest:
        c = lowest(rest)
        rest &= rest - 1
        unseen[nun] = c
        nun += 1
    lead_p = tp[0] if tlen else p
    for _ in range(nsamp):
        world = sample_world(p, hands, unseen, nun, voids, n, tab, t)
        x = opts
        while x:
            c = lowest(x)
            x &= x - 1
            hs = world.copy()
            hs[p] = hand & ~bit(c)
            vs = voids.copy()
            if track:
                note_void(vs, p, tc, tlen, c, t, tab, gf)
            tp2 = tp.copy()
            tc2 = tc.copy()
            tp2[tlen] = p
            tc2[tlen] = c
            w = won.copy()
            last = rollout(hs, tp2, tc2, tlen + 1, lead_p, gone, w, vs, t, team, n, track, rules, tab, gf)
            totals[c] += search_value(w, t, team[p], last, tab, gf)
    return opts, totals


@njit(cache=True)
def search_pick(p, hands, tp, tc, tlen, gone, won, voids, t, team, n, track, rules, nsamp, tab, gf):
    opts, totals = search_values(p, hands, tp, tc, tlen, gone, won, voids, t, team, n, track, rules,
                                 nsamp, tab, gf)
    if popcount(opts) == 1:
        return lowest(opts)
    best, bt, bp = -1, -1e300, 1 << 30
    x = opts
    while x:
        c = lowest(x)
        x &= x - 1
        pw = tab[t, F_POWER, c]
        if totals[c] > bt or (totals[c] == bt and pw < bp):
            best, bt, bp = c, totals[c], pw
    return best


# ---------------------------------------------------------------- batch runs

@njit(parallel=True, cache=True)
def survey(nhands, nplayers, deal_size, hand_size, do_refill, team, nteams, track, rules_bt,
           search, tab, gf, tw, tl, seed):
    """Random deals; seat 0 bids, names trump and pitches. Returns per hand: trump,
    the bidder's dealt-hand mask... (cols: trump, bidder hand after refill, own pts,
    others' pts)."""
    out = np.zeros((nhands, 4), np.int64)
    for i in prange(nhands):
        hands, t = deal(seed + i, nplayers, deal_size, hand_size, do_refill, 0, tab, gf, tw, tl)
        sc = play_game(hands, 0, t, team, nteams, True, rules_bt[team[0]], search, tab, gf)
        out[i, 0] = t
        out[i, 1] = hands[0]
        out[i, 2] = sc[team[0]]
        out[i, 3] = sc.sum() - sc[team[0]]
    return out


@njit(cache=True)
def play_game_log(hands0, bidder, t, team, nteams, track, rules, search, tab, gf):
    """play_game_won, also returning every card played in order (players, cards)."""
    n = hands0.shape[0]
    total = 0
    for q in range(n):
        total += popcount(hands0[q])
    log_p = np.zeros(total, np.int64)
    log_c = np.zeros(total, np.int64)
    lo_p, lo_team = 1 << 30, 0
    for q in range(n):
        x = hands0[q] & tab[t, F_ESM, t]
        while x:
            c = lowest(x)
            x &= x - 1
            if tab[t, F_POWER, c] < lo_p:
                lo_p, lo_team = tab[t, F_POWER, c], team[q]
    hands = hands0.copy()
    gone = np.int64(0)
    won = np.zeros(nteams, np.int64)
    voids = np.zeros(n, np.int64)
    tp = np.zeros(n, np.int64)
    tc = np.zeros(n, np.int64)
    tlen, leader, last_team, k = 0, bidder, -1, 0
    while True:
        if tlen == n:
            wp, _ = trick_winner(tp, tc, tlen, t, tab)
            for i in range(n):
                gone |= bit(tc[i])
                won[team[wp]] |= bit(tc[i])
            tlen = 0
            leader = wp
            last_team = team[wp]
            if hands[leader] == 0:
                break
        p = (leader + tlen) % n
        if search[team[p]] > 0:
            c = search_pick(p, hands, tp, tc, tlen, gone, won, voids, t, team, n, track, rules,
                            search[team[p]], tab, gf)
        else:
            c = heur_pick(p, hands, tp, tc, tlen, gone, voids, t, team, n, rules, tab, gf)
        hands[p] &= ~bit(c)
        if track:
            note_void(voids, p, tc, tlen, c, t, tab, gf)
        tp[tlen] = p
        tc[tlen] = c
        tlen += 1
        log_p[k] = p
        log_c[k] = c
        k += 1
    sc = np.zeros(nteams, np.int64)
    score_into(won, t, lo_team, last_team, sc, tab, gf)
    return sc, log_p, log_c


@njit(parallel=True, cache=True)
def deal_batch(start, count, nplayers, deal_size, hand_size, do_refill, team, tab, gf, tw, tl, seed):
    """deal() for hands start..start+count-1 (seed + index, as survey() uses).
    Returns hands (count, nplayers), trump, and the team dealt the lowest trump."""
    hands = np.zeros((count, nplayers), np.int64)
    trumps = np.zeros(count, np.int64)
    lo_team = np.zeros(count, np.int64)
    for k in prange(count):
        h, t = deal(seed + start + k, nplayers, deal_size, hand_size, do_refill, 0, tab, gf, tw, tl)
        hands[k] = h
        trumps[k] = t
        lo_p = 1 << 30
        for q in range(nplayers):
            x = h[q] & tab[t, F_ESM, t]
            while x:
                c = lowest(x)
                x &= x - 1
                if tab[t, F_POWER, c] < lo_p:
                    lo_p, lo_team[k] = tab[t, F_POWER, c], team[q]
    return hands, trumps, lo_team
