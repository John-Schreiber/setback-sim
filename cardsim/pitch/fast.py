"""CPU (Numba) engine for Pitch under any house rules: dealing, whole games, search.

Built on core.py; the GPU engine runs the same core. `Game(rules, nplayers)`
bundles the tables, flags and default player rules for one variant.
"""
import numpy as np
from numba import njit, prange

from cardsim.pitch import core
from cardsim.pitch.core import (bit, heur_pick, lowest, note_void, options, popcount, rng_seed,
                                rollout, sample_world, score_into, search_value, trick_winner)
from cardsim.pitch.tables import (F_ESM, F_ESUIT, F_NRANK, F_POWER, F_PTS, G_DECK, build_flags,
                                  build_tables)


# ---------------------------------------------------------------- variant bundle

class Game:
    """Tables, flags, teams, trump-choice weights and rule-based player settings for
    one variant."""

    def __init__(self, rules, nplayers=4, team_size=2, weights=None):
        import setback as sb
        self.rules, self.nplayers = rules, nplayers
        self.tab = build_tables(rules)
        self.gf = build_flags(rules, nplayers)
        self.team = np.array([i % (nplayers // team_size) for i in range(nplayers)], np.int64)
        self.nteams = int(self.team.max()) + 1
        self.tw, self.tl, self.tx = weights if weights is not None else trump_weights(rules.name, nplayers)

        def rules_for(bidder_seat):
            return np.array([rules_row(r) for r in sb.team_rules(nplayers, list(self.team), bidder_seat)],
                            np.int64)
        # player rules when the bid is held by team b (seat b is on team b for b < nteams)
        self.rules_bt = np.stack([rules_for(b) for b in range(self.nteams)])


def trump_weights(preset, nplayers):
    """(rank weights[13], length bonus[10], [off-jack, big joker, little joker]) for the
    bidder's trump choice, in hundredths. Fitted per variant by fit_trump.py
    (cardsim/pitch/trump_weights.json); setback4 keeps trump_weights.py; a variant not
    fitted yet starts from setback4's weights with the off-jack and jokers worth a Jack."""
    import json
    import os
    from trump_weights import LEN_W, RANK_W
    g = 6 if nplayers == 6 else 4
    fn = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trump_weights.json")
    fitted = json.load(open(fn)) if os.path.exists(fn) else {}
    w = fitted.get(preset, {}).get(str(g))
    if w is None or preset == "setback4":
        tw = list(RANK_W[g])
        return (np.array(tw, np.int64), np.array(LEN_W[g] + [LEN_W[g][-1]] * 3, np.int64),
                np.array([tw[9]] * 3, np.int64))
    return (np.array(w["rank"], np.int64), np.array(w["length"], np.int64), np.array(w["extra"], np.int64))


def rules_row(r):
    """setback.py RULES dict -> one row of the engine's rules array."""
    return (core.PITCH_CODES[r["pitch"]], r["ruff_min"], int(r["pull"]), bool(r["lead_aces"]),
            bool(r.get("schoolboy", False)), r.get("bank_low", 0), r.get("bank_jack", 0),
            r.get("lead10", 0), r.get("presmear", 0), bool(r.get("pull_t2", False)))


# ---------------------------------------------------------------- trump choice and the deal

@njit(cache=True)
def pick_trump(hand, tab, tw, tl, tx):
    """The suit the bidder names: a weight per natural rank (tw), a bonus for the number
    of trumps (tl), and weights for the off-jack and jokers (tx: off-jack, big joker,
    little joker), which count for the suit they would join. Only suits with a natural
    card are considered. Ties go to the lower suit."""
    best, bv = -1, np.int64(-(1 << 60))
    for s in range(4):
        tm = hand & tab[s, F_ESM, s]
        if (tm & (np.int64(0x1FFF) << np.int64(13 * s))) == 0:
            continue
        v = tl[min(popcount(tm), tl.shape[0] - 1)]
        x = tm
        while x:
            c = lowest(x)
            x &= x - 1
            if c < 52 and c // 13 == s:
                v += tw[c % 13]
            elif c == 53:
                v += tx[1]
            elif c == 52:
                v += tx[2]
            else:
                v += tx[0]
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
def deal(seed, nplayers, deal_size, hand_size, do_refill, bidder, tab, gf, tw, tl, tx):
    """Shuffle (seeded) and deal; the bidder names trump from the dealt cards; then the
    discard-and-refill if the rules have one. Returns (hands, trump, the bidder's dealt
    cards before any discard)."""
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
    t = pick_trump(hands[bidder], tab, tw, tl, tx)
    dealt = hands[bidder]
    if do_refill or deal_size > hand_size:
        used = nplayers * deal_size
        stock = np.empty(nd - used, np.int64)
        for i in range(nd - used):
            stock[i] = cards[perm[used + i]]
        refill(hands, stock, nd - used, t, hand_size, bidder, tab)
    return hands, t, dealt


# ---------------------------------------------------------------- whole games

@njit(cache=True)
def search_states(search, seed, game_index):
    """Random streams for one game's search: stream s is seeded by (seed, game_index *
    slots + s), exactly as the GPU engine seeds them, so search plays identically."""
    nsamp = max(1, search.max())
    states = np.zeros((nsamp, 2), np.uint64)
    for s in range(nsamp):
        rng_seed(states, s, seed, game_index * nsamp + s)
    return states


@njit(cache=True)
def play_game_won(hands0, bidder, t, team, nteams, track, rules, search, tab, gf, seed=0, game_index=0):
    """Play one hand. Returns (points per team, cards each team captured, team that
    took the last trick). search: sampled deals per decision, per team (0 = rules);
    its random numbers come from (seed, game_index), as on the GPU."""
    n = hands0.shape[0]
    states = search_states(search, seed, game_index)
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
                            search[team[p]], tab, gf, states)
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
def play_game(hands0, bidder, t, team, nteams, track, rules, search, tab, gf, seed=0, game_index=0):
    return play_game_won(hands0, bidder, t, team, nteams, track, rules, search, tab, gf, seed, game_index)[0]


# ---------------------------------------------------------------- search player

@njit(cache=True)
def search_values(p, hands, tp, tc, tlen, gone, won, voids, t, team, n, track, rules, nsamp, tab, gf,
                  states):
    """Legal cards (mask) and each card's summed search value (indexed by card).
    states: the game's random streams (search_states); sample s uses stream s."""
    hand = hands[p]
    opts = options(hand, gone, tc, tlen, t, tab, gf)
    totals = np.zeros(54, np.float64)
    if popcount(opts) == 1:
        return opts, totals
    seen = gone | hand
    for i in range(tlen):
        seen |= bit(tc[i])
    unseen = gf[G_DECK] & ~seen
    lead_p = tp[0] if tlen else p
    world = np.zeros(n, np.int64)
    order = np.empty(n, np.int64)
    hs, vs, tp2, tc2 = np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n, np.int64), np.empty(n, np.int64)
    w = np.empty(won.shape[0], np.int64)
    for s in range(nsamp):
        sample_world(p, hands, unseen, voids, n, t, tab, world, order, states, s)
        x = opts
        while x:
            c = lowest(x)
            x &= x - 1
            hs[:] = world  # scratch reused across play-outs (no allocation per play-out)
            hs[p] = hand & ~bit(c)
            vs[:] = voids
            if track:
                note_void(vs, p, tc, tlen, c, t, tab, gf)
            tp2[:] = tp
            tc2[:] = tc
            tp2[tlen] = p
            tc2[tlen] = c
            w[:] = won
            last = rollout(hs, tp2, tc2, tlen + 1, lead_p, gone, w, vs, t, team, n, track, rules, tab, gf)
            totals[c] += search_value(w, t, team[p], last, tab, gf)
    return opts, totals


@njit(cache=True)
def search_pick(p, hands, tp, tc, tlen, gone, won, voids, t, team, n, track, rules, nsamp, tab, gf,
                states):
    opts, totals = search_values(p, hands, tp, tc, tlen, gone, won, voids, t, team, n, track, rules,
                                 nsamp, tab, gf, states)
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

S_TRUMP, S_HAND, S_OWN, S_OPP, S_DEALT, S_SWEEP = range(6)  # survey() columns

@njit(parallel=True, cache=True)
def survey(nhands, nplayers, deal_size, hand_size, do_refill, team, nteams, track, rules_bt,
           search, tab, gf, tw, tl, tx, seed, start=0):
    """Random deals; seat 0 bids, names trump and pitches. One row per hand (S_* columns):
    trump, the bidder's hand in play, bidder-team points, everyone else's points, the
    bidder's dealt cards (before any discard), and whether the bidder's team took every
    trick. start: global index of the first deal (deal i uses seed + start + i), so pieces
    of a study reproduce the whole, on CPU or GPU."""
    out = np.zeros((nhands, 6), np.int64)
    for i in prange(nhands):
        hands, t, dealt = deal(seed + start + i, nplayers, deal_size, hand_size, do_refill, 0, tab, gf, tw,
                               tl, tx)
        sc, won, _ = play_game_won(hands, 0, t, team, nteams, True, rules_bt[team[0]], search, tab, gf,
                                   seed, start + i)
        out[i, S_TRUMP] = t
        out[i, S_HAND] = hands[0]
        out[i, S_OWN] = sc[team[0]]
        out[i, S_OPP] = sc.sum() - sc[team[0]]
        out[i, S_DEALT] = dealt
        sweep = 1
        for k in range(nteams):
            if k != team[0] and won[k] != 0:
                sweep = 0
        out[i, S_SWEEP] = sweep
    return out


@njit(cache=True)
def play_game_log(hands0, bidder, t, team, nteams, track, rules, search, tab, gf, seed=0, game_index=0):
    """play_game_won, also returning every card played in order (players, cards)."""
    n = hands0.shape[0]
    states = search_states(search, seed, game_index)
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
                            search[team[p]], tab, gf, states)
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
def deal_batch(start, count, nplayers, deal_size, hand_size, do_refill, team, tab, gf, tw, tl, tx, seed):
    """deal() for hands start..start+count-1 (seed + index, as survey() uses).
    Returns hands (count, nplayers), trump, the team dealt the lowest trump, and the
    bidder's dealt cards."""
    hands = np.zeros((count, nplayers), np.int64)
    trumps = np.zeros(count, np.int64)
    lo_team = np.zeros(count, np.int64)
    dealt0 = np.zeros(count, np.int64)
    for k in prange(count):
        h, t, d0 = deal(seed + start + k, nplayers, deal_size, hand_size, do_refill, 0, tab, gf, tw, tl, tx)
        hands[k] = h
        trumps[k] = t
        dealt0[k] = d0
        lo_p = 1 << 30
        for q in range(nplayers):
            x = h[q] & tab[t, F_ESM, t]
            while x:
                c = lowest(x)
                x &= x - 1
                if tab[t, F_POWER, c] < lo_p:
                    lo_p, lo_team[k] = tab[t, F_POWER, c], team[q]
    return hands, trumps, lo_team, dealt0


@njit(cache=True)
def deal_with_hand(hand, t, seed, nplayers, deal_size, hand_size, do_refill, tab, gf):
    """Seat 0 holds `hand` (its dealt cards) and names trump t; the rest of the deck is
    shuffled (seeded) and dealt, then the discard and refill if the rules have one.
    The same seed gives the same layout whichever trump is named."""
    np.random.seed(seed)
    rest = gf[G_DECK] & ~hand
    cards = np.empty(54, np.int64)
    nr = 0
    while rest:
        c = lowest(rest)
        rest &= rest - 1
        cards[nr] = c
        nr += 1
    perm = np.random.permutation(nr)
    hands = np.zeros(nplayers, np.int64)
    hands[0] = hand
    for q in range(1, nplayers):
        for j in range(deal_size):
            hands[q] |= bit(cards[perm[(q - 1) * deal_size + j]])
    if do_refill or deal_size > hand_size:
        used = (nplayers - 1) * deal_size
        stock = np.empty(nr - used, np.int64)
        for i in range(nr - used):
            stock[i] = cards[perm[used + i]]
        refill(hands, stock, nr - used, t, hand_size, 0, tab)
    return hands


@njit(parallel=True, cache=True)
def deal_with_hand_batch(hands_in, trumps, ids, deals, nplayers, deal_size, hand_size, do_refill, team,
                         tab, gf, seed):
    """`deals` layouts for each (hand, trump); layout j of hand ids[i] uses seed
    seed + ids[i] * deals + j. Returns hands and the team dealt the lowest trump."""
    nh = hands_in.shape[0]
    out = np.zeros((nh * deals, nplayers), np.int64)
    lo_team = np.zeros(nh * deals, np.int64)
    for g in prange(nh * deals):
        i = g // deals
        t = trumps[i]
        h = deal_with_hand(hands_in[i], t, seed + ids[i] * deals + g % deals, nplayers, deal_size,
                           hand_size, do_refill, tab, gf)
        out[g] = h
        lo_p = 1 << 30
        for q in range(nplayers):
            x = h[q] & tab[t, F_ESM, t]
            while x:
                c = lowest(x)
                x &= x - 1
                if tab[t, F_POWER, c] < lo_p:
                    lo_p, lo_team[g] = tab[t, F_POWER, c], team[q]
    return out, lo_team


@njit(parallel=True, cache=True)
def play_many(hands, trumps, team, nteams, rules, search, tab, gf, seed, offset):
    """Play every dealt game (CPU). Game i's search draws from (seed, offset + i), as
    gpu.play_batch(seed=seed, game_offset=offset) does. Returns points per team."""
    out = np.zeros((hands.shape[0], nteams), np.int64)
    for i in prange(hands.shape[0]):
        out[i] = play_game(hands[i], 0, trumps[i], team, nteams, True, rules, search, tab, gf, seed, offset + i)
    return out
