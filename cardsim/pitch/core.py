"""Pitch game logic for any house rules, shared by the CPU and GPU engines.

A port of engine_core.py in which every suit and rank lookup goes through the
per-trump tables (tables.py), so the off-jack and jokers can be trumps, point
schemes vary, and play rules (must follow, first lead, trump broken) are flags.
With the setback4 preset the tables reduce to plain suits and ranks and this code
makes exactly the same choices as engine_core.py (see tests/test_pitch_core.py).

Allocation-free, so it compiles as CUDA device functions too: imported normally it
compiles for the CPU; the GPU engine loads this file again with TARGET = "cuda".
"""
import numpy as np

from cardsim.pitch.tables import (F_ABOVE, F_CASH, F_CHEAP, F_DANGER, F_ESM, F_ESUIT, F_GV, F_MASKS,
                                  F_NRANK, F_POWER, F_PTS, F_SAFE, F_SHED, F_SMEAR, M_ACES, M_BJ,
                                  M_JACKS, M_KINGS, M_LJ, M_MID, M_MID2, M_OFFJ, M_POINTS, M_QUEENS,
                                  M_SAFE_TR, M_TENS, M_TR_HIGH, M_TR_LOW,
                                  G_DECK, G_FIRST_LEAD, G_GAME, G_HIGH, G_LAST, G_LOW, G_LOWCAP,
                                  G_MAY_TRUMP, G_TRUMP_BROKEN)

TARGET = globals().get("TARGET", "cpu")


def _clear_stale_numba_cache():
    """Numba's on-disk cache only notices edits to a function's own file, but the
    compiled CPU engine (fast.py) inlines this file's functions. When core.py changes,
    drop every cached function in this package so nothing runs stale code."""
    import glob
    import hashlib
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, "core.py"), "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    cache = os.path.join(here, "__pycache__")
    stamp = os.path.join(cache, "core.sources.sha256")
    try:
        with open(stamp) as fh:
            old = fh.read()
    except OSError:
        old = ""
    if old != digest:
        for f in glob.glob(os.path.join(cache, "*.nbi")) + glob.glob(os.path.join(cache, "*.nbc")):
            try:
                os.remove(f)
            except OSError:
                pass
        os.makedirs(cache, exist_ok=True)
        with open(stamp, "w") as fh:
            fh.write(digest)


if TARGET == "cpu":
    _clear_stale_numba_cache()
    from numba import njit

    def kernel_fn(f):
        return njit(cache=True)(f)
else:
    from numba import cuda

    def kernel_fn(f):
        return cuda.jit(device=True)(f)

# rules array columns (one row per team), as in engine_core.py
R_PITCH, R_RUFF_MIN, R_PULL, R_LEAD_ACES, R_SCHOOLBOY, R_BANK_LOW = 0, 1, 2, 3, 4, 5
R_BANK_JACK, R_LEAD10, R_PRESMEAR, R_PULL_T2 = 6, 7, 8, 9
N_RULES = 10
PITCH_CODES = {"top": 0, "smart": 1, "keepjack": 2, "smart2": 3}

N_BITS = 54


# ---------------------------------------------------------------- bits and table lookups

@kernel_fn
def bit(c):
    return np.int64(1) << np.int64(c)


# Lowest set bit and bit count are the hottest operations in the engine. Each compiles to a
# single instruction: tzcnt / popcnt on the CPU (LLVM intrinsics), ffs / popc on the GPU.
# Both give identical results (lowest(0) = -1), so CPU and GPU still play identically.
if TARGET == "cpu":
    from llvmlite import ir
    from numba.core import types
    from numba.extending import intrinsic

    @intrinsic
    def _cttz(typingctx, x):
        def codegen(context, builder, signature, args):
            return builder.cttz(args[0], ir.Constant(ir.IntType(1), 0))  # cttz(0) = 64
        return types.int64(types.int64), codegen

    @intrinsic
    def _ctlz(typingctx, x):
        def codegen(context, builder, signature, args):
            return builder.ctlz(args[0], ir.Constant(ir.IntType(1), 0))  # ctlz(0) = 64
        return types.int64(types.int64), codegen

    @intrinsic
    def _ctpop(typingctx, x):
        def codegen(context, builder, signature, args):
            return builder.ctpop(args[0])
        return types.int64(types.int64), codegen

    @kernel_fn
    def lowest(m):
        z = _cttz(np.int64(m))
        return z if z < 64 else -1

    @kernel_fn
    def popcount(m):
        return _ctpop(np.int64(m))

    @kernel_fn
    def highest(m):
        return 63 - _ctlz(np.int64(m))  # -1 for no bits
else:
    @kernel_fn
    def lowest(m):
        return cuda.ffs(np.int64(m)) - 1  # ffs is 1-based, 0 for no bits

    @kernel_fn
    def popcount(m):
        return cuda.popc(np.int64(m))

    @kernel_fn
    def highest(m):
        return 63 - cuda.clz(np.int64(m))  # -1 for no bits


@kernel_fn
def top_trump(m, t, tab):
    """Highest-power card of m, a mask of trumps only (-1 if empty). Trump power order is
    natural 2..10 < little joker < big joker < off-jack < natural J..A, and natural cards
    sit in rank order in the bits, so this is a highest-bit plus three special checks."""
    hi = m & tab[t, F_MASKS, M_TR_HIGH]
    if hi:
        return highest(hi)
    c = tab[t, F_MASKS, M_OFFJ]
    if c >= 0 and (m >> c) & 1:
        return c
    c = tab[t, F_MASKS, M_BJ]
    if c >= 0 and (m >> c) & 1:
        return c
    c = tab[t, F_MASKS, M_LJ]
    if c >= 0 and (m >> c) & 1:
        return c
    return highest(m & tab[t, F_MASKS, M_TR_LOW])


@kernel_fn
def low_trump(m, t, tab):
    """Lowest-power card of m, a mask of trumps only (-1 if empty); see top_trump."""
    lo = m & tab[t, F_MASKS, M_TR_LOW]
    if lo:
        return lowest(lo)
    c = tab[t, F_MASKS, M_LJ]
    if c >= 0 and (m >> c) & 1:
        return c
    c = tab[t, F_MASKS, M_BJ]
    if c >= 0 and (m >> c) & 1:
        return c
    c = tab[t, F_MASKS, M_OFFJ]
    if c >= 0 and (m >> c) & 1:
        return c
    return lowest(m & tab[t, F_MASKS, M_TR_HIGH])


@kernel_fn
def natural_card(rank, suit):
    return suit * 13 + rank - 2


# ---------------------------------------------------------------- rules of play

@kernel_fn
def danger(c, t, tab, gf):
    """How costly it is to hand this card to the other side (precomputed per trump)."""
    return tab[t, F_DANGER, c]


@kernel_fn
def legal(hand, lead, t, tab, gf):
    if lead < 0:
        return hand
    follow = hand & tab[t, F_ESM, lead]
    if follow == 0:
        return hand
    if lead == t or not gf[G_MAY_TRUMP]:
        return follow
    return follow | (hand & tab[t, F_ESM, t])  # may always trump in


@kernel_fn
def lead_mask(hand, t, gone, tab, gf):
    """Cards this player may lead (first lead / trump-broken rules)."""
    tm = tab[t, F_ESM, t]
    nt = hand & ~tm
    if nt:
        if gone == 0 and gf[G_FIRST_LEAD] == 2:
            return nt
        if gf[G_TRUMP_BROKEN] and (gone & tm) == 0:
            return nt
    return hand


@kernel_fn
def beats(a, b, lead, t, tab):
    sa, sb = tab[t, F_ESUIT, a], tab[t, F_ESUIT, b]
    if sa == t:
        return sb != t or tab[t, F_POWER, a] > tab[t, F_POWER, b]
    if sb == t:
        return False
    return sa == lead and tab[t, F_POWER, a] > tab[t, F_POWER, b]


@kernel_fn
def trick_winner(tp, tc, tlen, t, tab):
    lead = tab[t, F_ESUIT, tc[0]]
    wp, wc = tp[0], tc[0]
    for i in range(1, tlen):
        if beats(tc[i], wc, lead, t, tab):
            wp, wc = tp[i], tc[i]
    return wp, wc


@kernel_fn
def could_beat(p, c, lead, t, seen, voids, tab):
    """Could player p possibly beat card c? seen = cards the deciding player has seen."""
    can_trump = ((voids[p] >> t) & 1) == 0 and (tab[t, F_ESM, t] & ~seen) != 0
    above = tab[t, F_ABOVE, c] & ~seen
    if tab[t, F_ESUIT, c] == t:
        return can_trump and above != 0
    if can_trump:
        return True
    return ((voids[p] >> lead) & 1) == 0 and tab[t, F_ESUIT, c] == lead and above != 0


@kernel_fn
def note_void(voids, p, tc, tlen, c, t, tab, gf):
    if tlen == 0:
        return
    lead = tab[t, F_ESUIT, tc[0]]
    s = tab[t, F_ESUIT, c]
    if lead == t and s != t:
        voids[p] |= 1 << t
    elif lead != t and s != lead and s != t:
        voids[p] |= 1 << lead  # may still hold trump (chose not to ruff)
    elif lead != t and s == t and not gf[G_MAY_TRUMP]:
        voids[p] |= 1 << lead  # must follow suit: trumping in shows a void


@kernel_fn
def min_shed(m, t, tab, gf):
    best, bk = -1, np.int64(1) << 60
    x = m
    while x:
        c = lowest(x)
        x &= x - 1
        k = tab[t, F_SHED, c]
        if k < bk:
            best, bk = c, k
    return best


@kernel_fn
def min_cheap(m, t, tab, gf):
    best, bk = -1, np.int64(1) << 60
    x = m
    while x:
        c = lowest(x)
        x &= x - 1
        k = tab[t, F_CHEAP, c]
        if k < bk:
            best, bk = c, k
    return best


@kernel_fn
def holds(c, me, tlen, n, team, lead, t, seen, voids, tab):
    """Card c wins the trick against every opponent still to play (certain)."""
    for i in range(1, n - tlen):
        p = (me + i) % n
        if team[p] != team[me] and could_beat(p, c, lead, t, seen, voids, tab):
            return False
    return True


@kernel_fn
def choose(me, hand, leads, tp, tc, tlen, t, gone, team, n, voids, rules, tab, gf):
    """Rule-based player; mirrors engine_core.choose. leads: cards it may lead."""
    TM = tab[t, F_ESM, t]
    ruff_min, pull, lead_aces = rules[R_RUFF_MIN], rules[R_PULL], rules[R_LEAD_ACES]

    if tlen == 0:  # leading
        seen = gone | hand
        unseen_tr = TM & ~seen
        opp_trump = False
        for p in range(n):
            if team[p] != team[me] and ((voids[p] >> t) & 1) == 0 and unseen_tr != 0:
                opp_trump = True
        top = top_trump(hand & TM, t, tab)
        boss = top >= 0 and (tab[t, F_ABOVE, top] & ~seen) == 0 and ((leads >> top) & 1) == 1
        if boss and opp_trump:  # pull trump while opponents may still hold some
            return top
        # opponents can't beat it (out of trump / suit is boss): cash Aces and 10s
        best, bk = -1, -1
        x = leads & ~TM
        while x:
            c = lowest(x)
            x &= x - 1
            safe = True
            for p in range(n):
                if team[p] == team[me]:
                    continue
                if could_beat(p, c, tab[t, F_ESUIT, c], t, seen, voids, tab):
                    safe = False
                    break
            if safe:
                k = tab[t, F_CASH, c]
                if k > bk:
                    best, bk = c, k
        if best >= 0:
            return best
        if boss:
            return top
        if pull > 0 and opp_trump and (not rules[R_PULL_T2] or popcount(gone) == n):
            plain = leads & tab[t, F_MASKS, M_SAFE_TR]
            if popcount(plain) >= pull:
                return low_trump(plain, t, tab)
        off = leads & ~TM
        if lead_aces and off & tab[t, F_MASKS, M_ACES]:
            return lowest(off & tab[t, F_MASKS, M_ACES])
        if rules[R_LEAD10] >= 2:
            x = off & tab[t, F_MASKS, M_TENS]
            while x:
                c = lowest(x)
                x &= x - 1
                if popcount(tab[t, F_ABOVE, c] & ~seen) <= rules[R_LEAD10] - 1:
                    return c
        best, bk = -1, np.int64(1) << 60
        x = off & ~tab[t, F_MASKS, M_TENS]
        while x:
            c = lowest(x)
            x &= x - 1
            k = tab[t, F_SAFE, c]
            if k < bk:
                best, bk = c, k
        if best >= 0:
            return best
        if rules[R_LEAD10] and off & tab[t, F_MASKS, M_TENS]:
            return lowest(off & tab[t, F_MASKS, M_TENS])
        return min_shed(leads, t, tab, gf)

    lead = tab[t, F_ESUIT, tc[0]]
    opts = legal(hand, lead, t, tab, gf)
    wp, wc = trick_winner(tp, tc, tlen, t, tab)
    last = tlen == n - 1
    pts = 0
    tmask = np.int64(0)
    for i in range(tlen):
        pts += danger(tc[i], t, tab, gf)
        tmask |= bit(tc[i])

    beaters = tab[t, F_ABOVE, wc]
    if tab[t, F_ESUIT, wc] != t:
        beaters |= TM
    beaters &= opts

    bj = rules[R_BANK_JACK]
    jack = natural_card(11, t)
    if bj and (opts >> jack) & 1:
        unseen_hi = tab[t, F_ABOVE, jack] & ~(gone | hand | tmask)
        if unseen_hi and (bj == 1 or popcount(tab[t, F_ABOVE, jack] & hand) < popcount(unseen_hi)):
            # a Jack that isn't boss and isn't backed up: play it the moment it's safe
            if team[wp] == team[me]:
                if last or holds(wc, me, tlen, n, team, lead, t, gone | hand | tmask, voids, tab):
                    return jack
            elif beats(jack, wc, lead, t, tab) and (
                    last or holds(jack, me, tlen, n, team, lead, t,
                                  gone | hand | tmask | bit(jack), voids, tab)):
                return jack

    if team[wp] == team[me]:
        secure = last or holds(wc, me, tlen, n, team, lead, t, gone | hand | tmask, voids, tab)
        losers = opts & ~beaters
        if losers == 0:
            losers = opts
        if secure:  # smear points onto partner's trick
            best, bk = -1, -(np.int64(1) << 60)
            for part in range(2):  # on ties, smear a card of the led suit before a trump
                x = losers & tab[t, F_ESM, lead] if part == 0 else losers & ~tab[t, F_ESM, lead]
                while x:
                    c = lowest(x)
                    x &= x - 1
                    k = tab[t, F_SMEAR, c]
                    if k > bk:
                        best, bk = c, k
            return best
        return min_shed(losers, t, tab, gf)

    if rules[R_BANK_LOW] and lead != t:
        # trump with the 2/3 of trump: takes the trick and banks the Low point
        low = beaters & tab[t, F_ESM, t] & (bit(natural_card(2, t)) | bit(natural_card(3, t)))
        if low:
            c = low_trump(low, t, tab)
            m = rules[R_BANK_LOW]
            if m == 3 or last or (m == 2 and pts > 0) or (m == 4 and pts >= ruff_min) or \
                    holds(c, me, tlen, n, team, lead, t, gone | hand | tmask | bit(c), voids, tab):
                return c
    winners = beaters
    if lead != t and pts < ruff_min:  # not worth a trump
        winners &= ~TM
    if winners:
        if last:
            return min_cheap(winners, t, tab, gf)
        sure = np.int64(0)
        x = winners
        while x:
            c = lowest(x)
            x &= x - 1
            if holds(c, me, tlen, n, team, lead, t, gone | hand | tmask | bit(c), voids, tab):
                sure |= bit(c)
        if sure and (pts > 0 or (sure & ~TM) != 0):
            return min_cheap(sure, t, tab, gf)
        if winners & ~TM:
            return min_cheap(winners & ~TM, t, tab, gf)
        if pts > 0:
            return min_cheap(winners, t, tab, gf)
    ps = rules[R_PRESMEAR]
    if ps:
        # partner still to play could beat the winner: throw them an off-suit 10
        pard_can = False
        for i in range(1, n - tlen):
            q = (me + i) % n
            if team[q] == team[me] and (ps == 2 or i == n - tlen - 1) and \
                    could_beat(q, wc, lead, t, gone | hand | tmask, voids, tab):
                pard_can = True
        if pard_can:
            x = opts & ~TM & ~beaters
            while x:
                c = lowest(x)
                x &= x - 1
                if tab[t, F_NRANK, c] == 10:
                    return c
    return min_shed(opts, t, tab, gf)


@kernel_fn
def pitch_card(hand, t, mode, schoolboy, tab):
    """The bidder's first lead. mode: 0 top, 1 smart, 2 keepjack, 3 smart2."""
    tm = hand & tab[t, F_ESM, t]
    top = top_trump(tm, t, tab)
    if top < 0:  # no trump to pitch (only possible with unusual deals): lead anything
        return lowest(hand)
    ace, king, jack = natural_card(14, t), natural_card(13, t), natural_card(11, t)
    if schoolboy and top == ace and (tm >> king) & 1:
        return king  # schoolboy pitch: the King, hiding the Ace
    if mode == 0:
        return top
    not_jack = tm & ~bit(jack)
    natural = tm & tab[t, F_ESM, t] & (np.int64(0x1FFF) << np.int64(13 * t))
    top_rank = tab[t, F_NRANK, top] if (natural >> top) & 1 else 11  # off-jack, jokers: Jack level
    if mode == 2:  # keepjack
        if top_rank >= 12:
            return top
        if natural & tab[t, F_MASKS, M_MID]:
            return lowest(natural & tab[t, F_MASKS, M_MID])
        return low_trump(not_jack, t, tab) if not_jack else top
    if top == ace:
        return top
    # no Ace: lead a middle trump, keeping the Jack, 10 and low trump to score later
    if natural & tab[t, F_MASKS, M_MID2]:
        return lowest(natural & tab[t, F_MASKS, M_MID2])
    if mode == 3 and not_jack:
        return low_trump(not_jack, t, tab)
    return top


@kernel_fn
def heur_pick(p, hands, tp, tc, tlen, gone, voids, t, team, n, rules, tab, gf):
    r = rules[team[p]]
    if gone == 0 and tlen == 0 and gf[G_FIRST_LEAD] == 0:  # the pitch
        return pitch_card(hands[p], t, r[R_PITCH], r[R_SCHOOLBOY], tab)
    leads = lead_mask(hands[p], t, gone, tab, gf) if tlen == 0 else hands[p]
    return choose(p, hands[p], leads, tp, tc, tlen, t, gone, team, n, voids, r, tab, gf)


@kernel_fn
def options(hand, gone, tc, tlen, t, tab, gf):
    """Every card a player may play now (what search tries)."""
    if tlen:
        return legal(hand, tab[t, F_ESUIT, tc[0]], t, tab, gf)
    if gone == 0 and gf[G_FIRST_LEAD] == 0:
        return hand & tab[t, F_ESM, t]  # the pitch: any trump
    return lead_mask(hand, t, gone, tab, gf)


# ---------------------------------------------------------------- scoring

@kernel_fn
def game_points(m, t, tab):
    """Game count: 10 per Ten, 4 per Ace, 3 per King, 2 per Queen, 1 per Jack."""
    return (10 * popcount(m & tab[t, F_MASKS, M_TENS]) + 4 * popcount(m & tab[t, F_MASKS, M_ACES]) +
            3 * popcount(m & tab[t, F_MASKS, M_KINGS]) + 2 * popcount(m & tab[t, F_MASKS, M_QUEENS]) +
            popcount(m & tab[t, F_MASKS, M_JACKS]))


@kernel_fn
def card_points(m, t, tab):
    """Points for captured point cards (Jack, off-jack, jokers, 10/3/5/9 of trump)."""
    g = 0
    x = m & tab[t, F_MASKS, M_POINTS]
    while x:
        c = lowest(x)
        x &= x - 1
        g += tab[t, F_PTS, c]
    return g


@kernel_fn
def score_into(won, t, lo_team, last_team, sc, tab, gf):
    """Points per team from the cards each team captured, written into sc."""
    nteams = won.shape[0]
    alltr = np.int64(0)
    for k in range(nteams):
        alltr |= won[k] & tab[t, F_ESM, t]
        sc[k] = 0
    hi = top_trump(alltr, t, tab)
    lo = low_trump(alltr, t, tab)
    best, nbest, bt = -1, 0, -1
    for k in range(nteams):
        if hi >= 0 and (won[k] >> hi) & 1:
            sc[k] += gf[G_HIGH]
        sc[k] += card_points(won[k], t, tab)
        if gf[G_LOWCAP] and lo >= 0 and (won[k] >> lo) & 1:
            sc[k] += gf[G_LOW]
        g = game_points(won[k], t, tab)
        if g > best:
            best, nbest, bt = g, 1, k
        elif g == best:
            nbest += 1
    if not gf[G_LOWCAP] and lo >= 0:
        sc[lo_team] += gf[G_LOW]
    if nbest == 1:  # tie for game: nobody scores it
        sc[bt] += gf[G_GAME]
    if last_team >= 0:
        sc[last_team] += gf[G_LAST]


@kernel_fn
def search_value(won, t, my_team, last_team, tab, gf):
    """What search maximizes: my team's points minus everyone else's (Low to the captor)."""
    nteams = won.shape[0]
    alltr = np.int64(0)
    for k in range(nteams):
        alltr |= won[k] & tab[t, F_ESM, t]
    hi = top_trump(alltr, t, tab)
    lo = low_trump(alltr, t, tab)
    own, total = 0, 0
    best, nbest, bt = -1, 0, -1
    for k in range(nteams):
        pts = card_points(won[k], t, tab)
        if hi >= 0 and (won[k] >> hi) & 1:
            pts += gf[G_HIGH]
        if lo >= 0 and (won[k] >> lo) & 1:
            pts += gf[G_LOW]
        if k == last_team:
            pts += gf[G_LAST]
        total += pts
        if k == my_team:
            own += pts
        g = game_points(won[k], t, tab)
        if g > best:
            best, nbest, bt = g, 1, k
        elif g == best:
            nbest += 1
    if nbest == 1:
        total += gf[G_GAME]
        if bt == my_team:
            own += gf[G_GAME]
    return 2 * own - total


@kernel_fn
def rollout(hands, tp, tc, tlen, leader, gone, won, voids, t, team, n, track, rules, tab, gf):
    """Play out from any point with rule-based players. Mutates its arguments (won
    holds the result); returns the team that took the last trick."""
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
                return last_team
        p = (leader + tlen) % n
        c = heur_pick(p, hands, tp, tc, tlen, gone, voids, t, team, n, rules, tab, gf)
        hands[p] &= ~bit(c)
        if track:
            note_void(voids, p, tc, tlen, c, t, tab, gf)
        tp[tlen] = p
        tc[tlen] = c
        tlen += 1


# ---------------------------------------------------------------- search sampling (CPU and GPU)
#
# Search draws its random numbers from a small xoroshiro128+ generator with one stream per
# (game, sample slot), seeded by a hash of (study seed, global game index * slots + slot),
# and deals the hidden cards with the same integer-only Fisher-Yates shuffle on CPU and
# GPU. So search plays identically on either, and a study can mix CPU and GPU workers.
# states: uint64 array (streams, 2).

@kernel_fn
def splitmix64(z):
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


@kernel_fn
def rng_seed(states, i, seed, index):
    """Seed stream i from (seed, index)."""
    z = splitmix64(np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15) + np.uint64(index))
    s0 = splitmix64(z)
    s1 = splitmix64(s0)
    states[i, 0] = s0
    states[i, 1] = s1 if (s0 | s1) != np.uint64(0) else np.uint64(1)


@kernel_fn
def rng_next(states, i):
    """xoroshiro128+ step on stream i."""
    s0 = states[i, 0]
    s1 = states[i, 1]
    result = s0 + s1
    s1 = s1 ^ s0
    states[i, 0] = ((s0 << np.uint64(24)) | (s0 >> np.uint64(40))) ^ s1 ^ (s1 << np.uint64(16))
    states[i, 1] = (s1 << np.uint64(37)) | (s1 >> np.uint64(27))
    return result


@kernel_fn
def rng_below(states, i, m):
    """Integer in [0, m) from stream i, m < 65536 (integer math only, so CPU and GPU agree).
    Same value as (r >> 11) % m, computed from the two 32-bit halves: GPUs emulate 64-bit
    division slowly, 32-bit remainders are cheap."""
    x = rng_next(states, i) >> np.uint64(11)
    mm = np.uint32(m)
    hi = np.uint32(x >> np.uint64(32)) % mm
    lo = np.uint32(x & np.uint64(0xFFFFFFFF)) % mm
    p = np.uint32(np.uint32(0xFFFFFFFF) % mm + np.uint32(1))  # 2^32 mod m (up to one m)
    return np.int64(np.uint32(np.uint32(hi * p + lo) % mm))


@kernel_fn
def select_bit(m, k):
    """Position of the k-th (0-based) set bit of m (m has more than k bits, bit 63 clear):
    a six-step binary search on bit counts."""
    pos = 0
    x = m
    w = 32
    while w:
        low = x & ((np.int64(1) << np.int64(w)) - 1)
        c = popcount(low)
        if k >= c:
            k -= c
            x >>= np.int64(w)
            pos += w
        else:
            x = low
        w >>= 1
    return pos


@kernel_fn
def sample_world(me, hands, unseen, voids, n, t, tab, out, order, states, si):
    """Guess where the cards player `me` can't see are: deal the `unseen` mask to the other
    players (honoring known effective-suit voids) into `out`, using random stream si.
    Each player, most constrained first, draws uniformly from the unseen cards left that
    their voids allow; what nobody draws is the undealt stock. Work is proportional to the
    cards dealt, not to the unseen pool. order (len >= n) is scratch space."""
    out[me] = hands[me]
    k = 0
    for nv in range(4, -1, -1):  # most constrained first
        for q in range(n):
            if q != me and popcount(voids[q]) == nv:
                order[k] = q
                k += 1
    for attempt in range(31):
        rest = unseen
        ok = True
        for jj in range(n - 1):
            q = order[jj]
            need = popcount(hands[q])
            allowed = rest
            if attempt < 30:  # last attempt ignores voids (shouldn't be needed)
                for v in range(4):
                    if (voids[q] >> v) & 1:
                        allowed &= ~tab[t, F_ESM, v]
            avail = popcount(allowed)
            if avail < need:
                ok = False
                break
            m = np.int64(0)
            for _ in range(need):
                b = bit(select_bit(allowed, rng_below(states, si, avail)))
                m |= b
                allowed &= ~b
                avail -= 1
            out[q] = m
            rest &= ~m
        if ok:
            return
