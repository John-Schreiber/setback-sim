"""Numba engine for setback.py.

Same players and rules as the pure-Python engine in setback.py, compiled with
Numba. The per-card game logic lives in engine_core.py (shared with the GPU
engine); this file holds what needs arrays: search, full games and batch runs.

The pure-Python engine stays the reference: feed it hands sorted by
(suit, rank) and the rule-based players here make identical choices (see
test_fast.py). The search player here uses its own random numbers, so it
matches the Python search only statistically. The experimental odds-based
smearing (smear_p) is Python-only.
"""
import numpy as np
from numba import njit, prange

from engine_core import (  # noqa: F401  (re-exported: callers use fast.<name>)
    GV,
    PITCH_CODES,
    R_PITCH,
    R_RUFF_MIN,
    R_PULL,
    R_LEAD_ACES,
    R_SCHOOLBOY,
    R_BANK_LOW,
    R_BANK_JACK,
    R_LEAD10,
    R_PRESMEAR,
    R_PULL_T2,
    N_RULES,
    SM,
    TL,
    TW,
    above,
    beats,
    bit,
    choose,
    could_beat,
    danger,
    game_points,
    heur_pick,
    highest,
    holds,
    legal,
    lowest,
    min_cheap,
    min_shed,
    note_void,
    pick_trump,
    pitch_card,
    popcount,
    rank,
    rollout,
    score_into,
    search_value,
    suit,
    trick_winner,
)


def rules_array(rules_by_team):
    """List of RULES dicts (one per team) -> int64 array for the engine."""
    out = np.zeros((len(rules_by_team), N_RULES), np.int64)
    for i, r in enumerate(rules_by_team):
        out[i] = (PITCH_CODES[r["pitch"]], r["ruff_min"], int(r["pull"]), bool(r["lead_aces"]),
                  bool(r.get("schoolboy", False)), r.get("bank_low", 0), r.get("bank_jack", 0),
                  r.get("lead10", 0), r.get("presmear", 0), bool(r.get("pull_t2", False)))
    return out


# ---------------------------------------------------------------- scoring and play

@njit(cache=True)
def score(won, trump, lowcap, lo_team):
    sc = np.zeros(won.shape[0], np.int64)
    score_into(won, trump, lowcap, lo_team, sc)
    return sc


@njit(cache=True)
def sample_world(me, hands, unseen, nun, voids, n):
    """Deal the unseen cards to the other players, honoring known voids.
    Leftover unseen cards are the undealt stock."""
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
                if attempt < 30 and ((voids[q] >> suit(d)) & 1):
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
def search_pick(p, hands, tp, tc, tlen, leader, gone, won, voids, trump, team, n, lowcap,
                track, rules, nsamp):
    """Monte Carlo search: try each legal card against nsamp guessed deals of the
    hidden cards, play out the rest with the rule-based players, keep the card with
    the best average (own team points - everyone else's)."""
    opts, totals = search_values(p, hands, tp, tc, tlen, gone, won, voids, trump, team, n,
                                 track, rules, nsamp)
    if popcount(opts) == 1:
        return lowest(opts)
    best, bt, br = -1, -1e300, 99
    x = opts
    while x:
        c = lowest(x)
        x &= x - 1
        if totals[c] > bt or (totals[c] == bt and rank(c) < br):
            best, bt, br = c, totals[c], rank(c)
    return best


@njit(cache=True)
def search_values(p, hands, tp, tc, tlen, gone, won, voids, trump, team, n, track, rules, nsamp):
    """Legal cards (mask) and each card's summed search value (indexed by card;
    divide by nsamp for the average own-minus-others points). Nothing is sampled
    when only one card is legal."""
    hand = hands[p]
    if gone == 0 and tlen == 0:
        opts = hand & SM[trump]  # pitch: any trump
    else:
        opts = legal(hand, suit(tc[0]) if tlen else -1, trump)
    totals = np.zeros(52, np.float64)
    if popcount(opts) == 1:
        return opts, totals
    seen = gone | hand
    for i in range(tlen):
        seen |= bit(tc[i])
    unseen = np.empty(52, np.int64)
    nun = 0
    for d in range(52):
        if not (seen >> d) & 1:
            unseen[nun] = d
            nun += 1
    lead_p = tp[0] if tlen else p
    for _ in range(nsamp):
        world = sample_world(p, hands, unseen, nun, voids, n)
        x = opts
        while x:
            c = lowest(x)
            x &= x - 1
            hs = world.copy()
            hs[p] = hand & ~bit(c)
            vs = voids.copy()
            if track:
                note_void(vs, p, tc, tlen, c, trump)
            tp2 = tp.copy()
            tc2 = tc.copy()
            tp2[tlen] = p
            tc2[tlen] = c
            w = won.copy()
            rollout(hs, tp2, tc2, tlen + 1, lead_p, gone, w, vs, trump, team, n, True, track, rules)
            totals[c] += search_value(w, trump, team[p])
    return opts, totals


@njit(cache=True)
def play_game(hands0, bidder, trump, team, nteams, lowcap, track, rules, search):
    """Play one hand. hands0: int64 mask per player. search: samples per team (0 = rules).
    Returns points per team."""
    return play_game_won(hands0, bidder, trump, team, nteams, lowcap, track, rules, search)[0]


@njit(cache=True)
def play_game_won(hands0, bidder, trump, team, nteams, lowcap, track, rules, search):
    """play_game, also returning the cards each team captured (a team that took
    every trick holds all the cards played)."""
    n = hands0.shape[0]
    lo_c, lo_team = 99, 0
    for q in range(n):
        t = hands0[q] & SM[trump]
        if t and lowest(t) < lo_c:
            lo_c, lo_team = lowest(t), team[q]
    hands = hands0.copy()
    gone = np.int64(0)
    won = np.zeros(nteams, np.int64)
    voids = np.zeros(n, np.int64)
    tp = np.zeros(n, np.int64)
    tc = np.zeros(n, np.int64)
    tlen = 0
    leader = bidder
    while True:
        if tlen == n:
            wp, _ = trick_winner(tp, tc, tlen, trump)
            for i in range(n):
                gone |= bit(tc[i])
                won[team[wp]] |= bit(tc[i])
            tlen = 0
            leader = wp
            if hands[leader] == 0:
                break
        p = (leader + tlen) % n
        if search[team[p]] > 0:
            c = search_pick(p, hands, tp, tc, tlen, leader, gone, won, voids, trump, team, n,
                            lowcap, track, rules, search[team[p]])
        else:
            c = heur_pick(p, hands, tp, tc, tlen, gone, voids, trump, team, n, lowcap, rules)
        hands[p] &= ~bit(c)
        if track:
            note_void(voids, p, tc, tlen, c, trump)
        tp[tlen] = p
        tc[tlen] = c
        tlen += 1
    return score(won, trump, lowcap, lo_team), won


# ---------------------------------------------------------------- batch runs

# survey output columns
F_TOP, F_NTR, F_ACES, F_JACK, F_LOW, F_OWN, F_OPP = range(7)


@njit(parallel=True, cache=True)
def survey(nhands, nplayers, team, nteams, lowcap, track, rules, search, seed):
    """Random deals; player 0 bids, names their best suit and pitches.
    Returns one row per hand: top trump rank, #trump, off-suit Aces (max 2),
    holds Jack, holds 2/3 of trump, bidder-team points, everyone else's points."""
    out = np.zeros((nhands, 7), np.int64)
    for i in prange(nhands):
        np.random.seed(seed + i)
        perm = np.random.permutation(52)
        hands = np.zeros(nplayers, np.int64)
        for q in range(nplayers):
            for j in range(6):
                hands[q] |= bit(perm[q * 6 + j])
        trump = pick_trump(hands[0], nplayers)
        sc = play_game(hands, 0, trump, team, nteams, lowcap, track, rules, search)
        h = hands[0]
        t = h & SM[trump]
        aces = 0
        for s in range(4):
            if s != trump and (h >> (s * 13 + 12)) & 1:
                aces += 1
        out[i, F_TOP] = rank(highest(t))
        out[i, F_NTR] = popcount(t)
        out[i, F_ACES] = min(aces, 2)
        out[i, F_JACK] = (t >> (trump * 13 + 9)) & 1
        out[i, F_LOW] = (t & (bit(trump * 13) | bit(trump * 13 + 1))) != 0
        out[i, F_OWN] = sc[team[0]]
        out[i, F_OPP] = sc.sum() - sc[team[0]]
    return out


@njit(parallel=True, cache=True)
def sim_hand(hand, trump, nhands, nplayers, team, nteams, lowcap, track, rules, search, seed):
    """Player 0 holds `hand` (mask) and bids `trump`; the rest is dealt at random.
    Returns (own, opp) points per deal."""
    rest = np.empty(52 - popcount(hand), np.int64)
    k = 0
    for d in range(52):
        if not (hand >> d) & 1:
            rest[k] = d
            k += 1
    out = np.zeros((nhands, 2), np.int64)
    for i in prange(nhands):
        np.random.seed(seed + i)
        perm = np.random.permutation(rest.shape[0])
        hands = np.zeros(nplayers, np.int64)
        hands[0] = hand
        for q in range(1, nplayers):
            for j in range(6):
                hands[q] |= bit(rest[perm[(q - 1) * 6 + j]])
        sc = play_game(hands, 0, trump, team, nteams, lowcap, track, rules, search)
        out[i, 0] = sc[team[0]]
        out[i, 1] = sc.sum() - sc[team[0]]
    return out


@njit(parallel=True, cache=True)
def head_to_head(nhands, nplayers, team, nteams, lowcap, track, rules_a, search_a,
                 rules_b, search_b, seed, need_ak=False):
    """Same deals played two ways. Returns bidder-team points under setup A and B.
    need_ak: only deals where the bidder holds the Ace and King of trump."""
    out = np.zeros((nhands, 2), np.int64)
    for i in prange(nhands):
        np.random.seed(seed + i)
        hands = np.zeros(nplayers, np.int64)
        trump = 0
        while True:
            perm = np.random.permutation(52)
            hands[:] = 0
            for q in range(nplayers):
                for j in range(6):
                    hands[q] |= bit(perm[q * 6 + j])
            trump = pick_trump(hands[0], nplayers)
            if not need_ak or (hands[0] >> (trump * 13 + 11)) & 3 == 3:
                break
        out[i, 0] = play_game(hands, 0, trump, team, nteams, lowcap, track, rules_a, search_a)[team[0]]
        out[i, 1] = play_game(hands, 0, trump, team, nteams, lowcap, track, rules_b, search_b)[team[0]]
    return out


@njit(parallel=True, cache=True)
def deal_batch(start, count, nplayers, seed):
    """The same deals survey() plays for hands start..start+count-1 (same seeds).
    Returns hands (count, nplayers), trump bid by player 0, and the team-agnostic
    owner (player) of the lowest dealt trump, for the low-dealt rule."""
    hands = np.zeros((count, nplayers), np.int64)
    trumps = np.zeros(count, np.int64)
    lo_owner = np.zeros(count, np.int64)
    for k in prange(count):
        np.random.seed(seed + start + k)
        perm = np.random.permutation(52)
        for q in range(nplayers):
            for j in range(6):
                hands[k, q] |= bit(perm[q * 6 + j])
        trump = pick_trump(hands[k, 0], nplayers)
        trumps[k] = trump
        lo_c = 99
        for q in range(nplayers):
            t = hands[k, q] & SM[trump]
            if t and lowest(t) < lo_c:
                lo_c, lo_owner[k] = lowest(t), q
    return hands, trumps, lo_owner


@njit(parallel=True, cache=True)
def features_batch(hands, trumps):
    """Bidder-hand features as in survey(): top trump rank, #trump, off-suit Aces
    (max 2), holds Jack, holds 2/3 of trump."""
    out = np.zeros((hands.shape[0], 5), np.int64)
    for i in prange(hands.shape[0]):
        h = hands[i, 0]
        trump = trumps[i]
        t = h & SM[trump]
        aces = 0
        for s in range(4):
            if s != trump and (h >> (s * 13 + 12)) & 1:
                aces += 1
        out[i, F_TOP] = rank(highest(t))
        out[i, F_NTR] = popcount(t)
        out[i, F_ACES] = min(aces, 2)
        out[i, F_JACK] = (t >> (trump * 13 + 9)) & 1
        out[i, F_LOW] = (t & (bit(trump * 13) | bit(trump * 13 + 1))) != 0
    return out


@njit(parallel=True, cache=True)
def deal_hand_batch(hand, trump, count, nplayers, seed):
    """The same deals sim_hand() plays: player 0 holds `hand`, the rest is dealt at
    random. Returns hands (count, nplayers) and the owner of the lowest dealt trump."""
    rest = np.empty(52 - popcount(hand), np.int64)
    k = 0
    for d in range(52):
        if not (hand >> d) & 1:
            rest[k] = d
            k += 1
    hands = np.zeros((count, nplayers), np.int64)
    lo_owner = np.zeros(count, np.int64)
    for i in prange(count):
        np.random.seed(seed + i)
        perm = np.random.permutation(rest.shape[0])
        hands[i, 0] = hand
        for q in range(1, nplayers):
            for j in range(6):
                hands[i, q] |= bit(rest[perm[(q - 1) * 6 + j]])
        lo_c = 99
        for q in range(nplayers):
            t = hands[i, q] & SM[trump]
            if t and lowest(t) < lo_c:
                lo_c, lo_owner[i] = lowest(t), q
    return hands, lo_owner
