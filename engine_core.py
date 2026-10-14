"""Game logic shared by the CPU (setback_fast.py) and GPU (setback_gpu.py) engines.

Everything here is allocation-free so it can compile as CUDA device functions.
Imported normally it compiles for the CPU with Numba. setback_gpu.py loads this
same file a second time with TARGET = "cuda" set first, so both engines run
exactly the same rules.

Cards are ints 0..51 (suit * 13 + rank - 2) and a hand is an int64 bitmask, so
iterating a hand's bits from low to high visits cards in (suit, rank) order.
"""
import numpy as np

TARGET = globals().get("TARGET", "cpu")


def _clear_stale_numba_cache():
    """Numba's on-disk cache only notices edits to a function's own file, but the
    compiled CPU engine also bakes in this file and trump_weights.py (callers in
    other files inline these functions and constants). When either changes, drop
    every cached function in this folder so nothing runs stale code."""
    import glob
    import hashlib
    import os
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for f in ("engine_core.py", "trump_weights.py"):
        with open(os.path.join(here, f), "rb") as fh:
            h.update(fh.read())
    cache = os.path.join(here, "__pycache__")
    stamp = os.path.join(cache, "engine_core.sources.sha256")
    try:
        with open(stamp) as fh:
            old = fh.read()
    except OSError:
        old = ""
    if old != h.hexdigest():
        for f in glob.glob(os.path.join(cache, "*.nbi")) + glob.glob(os.path.join(cache, "*.nbc")):
            try:
                os.remove(f)
            except OSError:
                pass
        os.makedirs(cache, exist_ok=True)
        with open(stamp, "w") as fh:
            fh.write(h.hexdigest())


if TARGET == "cpu":
    _clear_stale_numba_cache()
    from numba import njit

    def kernel_fn(f):
        return njit(cache=True)(f)
else:
    from numba import cuda

    def kernel_fn(f):
        return cuda.jit(device=True)(f)

# game-count value by rank index (rank - 2): 10=10, J=1, Q=2, K=3, A=4
GV = np.array([0, 0, 0, 0, 0, 0, 0, 0, 10, 1, 2, 3, 4], np.int64)
SM = np.array([np.int64(0x1FFF) << (13 * s) for s in range(4)], np.int64)  # suit masks
# pick_trump weights by rank index
from trump_weights import TL, TW  # noqa: E402  (bidder's trump-choice weights, per game)

# rules array columns (one row per team)
R_PITCH, R_RUFF_MIN, R_PULL, R_LEAD_ACES, R_SCHOOLBOY, R_BANK_LOW = 0, 1, 2, 3, 4, 5
R_BANK_JACK, R_LEAD10, R_PRESMEAR, R_PULL_T2 = 6, 7, 8, 9
N_RULES = 10
PITCH_CODES = {"top": 0, "smart": 1, "keepjack": 2, "smart2": 3}



# ---------------------------------------------------------------- bit helpers

@kernel_fn
def bit(c):
    return np.int64(1) << np.int64(c)


@kernel_fn
def rank(c):
    return c % 13 + 2


@kernel_fn
def suit(c):
    return c // 13


@kernel_fn
def highest(m):
    for i in range(51, -1, -1):
        if (m >> i) & 1:
            return i
    return -1


@kernel_fn
def lowest(m):
    for i in range(52):
        if (m >> i) & 1:
            return i
    return -1


@kernel_fn
def popcount(m):
    n = 0
    while m:
        m &= m - 1
        n += 1
    return n


@kernel_fn
def above(c):
    """Mask of the cards in c's suit ranked above c."""
    return SM[suit(c)] & ~((bit(c) << 1) - 1)


# ---------------------------------------------------------------- rules of play

@kernel_fn
def danger(c, trump, lowcap):
    d = GV[c % 13]
    if suit(c) == trump:
        r = rank(c)
        if r == 11:
            d += 15
        if lowcap and r <= 3:
            d += 8
    return d


@kernel_fn
def legal(hand, lead, trump):
    if lead < 0:
        return hand
    follow = hand & SM[lead]
    if follow == 0:
        return hand
    if lead == trump:
        return follow
    return follow | (hand & SM[trump])  # may always trump in


@kernel_fn
def beats(a, b, lead, trump):
    if suit(a) == trump:
        return suit(b) != trump or a > b
    if suit(b) == trump:
        return False
    return suit(a) == lead and a > b


@kernel_fn
def trick_winner(tp, tc, tlen, trump):
    lead = suit(tc[0])
    wp, wc = tp[0], tc[0]
    for i in range(1, tlen):
        if beats(tc[i], wc, lead, trump):
            wp, wc = tp[i], tc[i]
    return wp, wc


@kernel_fn
def could_beat(p, c, lead, trump, seen, voids):
    """Could player p possibly beat card c? seen = cards the deciding player has seen."""
    can_trump = ((voids[p] >> trump) & 1) == 0 and (SM[trump] & ~seen) != 0
    if suit(c) == trump:
        return can_trump and (above(c) & ~seen) != 0
    if can_trump:
        return True
    return ((voids[p] >> lead) & 1) == 0 and suit(c) == lead and (above(c) & ~seen) != 0


@kernel_fn
def note_void(voids, p, tc, tlen, c, trump):
    if tlen == 0:
        return
    lead = suit(tc[0])
    s = suit(c)
    if lead == trump and s != trump:
        voids[p] |= 1 << trump
    elif lead != trump and s != lead and s != trump:
        voids[p] |= 1 << lead  # may still hold trump (chose not to ruff)


@kernel_fn
def min_shed(m, trump, lowcap):
    best, bk = -1, 1 << 60
    x = m
    while x:
        c = lowest(x)
        x &= x - 1
        k = danger(c, trump, lowcap) * 1000 + (suit(c) == trump) * 100 + rank(c)
        if k < bk:
            best, bk = c, k
    return best


@kernel_fn
def min_cheap(m, trump, lowcap):
    best, bk = -1, 1 << 60
    x = m
    while x:
        c = lowest(x)
        x &= x - 1
        k = (suit(c) == trump) * 100000 + danger(c, trump, lowcap) * 100 + rank(c)
        if k < bk:
            best, bk = c, k
    return best


@kernel_fn
def choose(me, hand, tp, tc, tlen, trump, gone, team, n, lowcap, voids, rules):
    """Rule-based player; mirrors setback.choose."""
    TM = SM[trump]
    ruff_min, pull, lead_aces = rules[R_RUFF_MIN], rules[R_PULL], rules[R_LEAD_ACES]

    if tlen == 0:  # leading
        seen = gone | hand
        unseen_tr = TM & ~seen
        opp_trump = False
        for p in range(n):
            if team[p] != team[me] and ((voids[p] >> trump) & 1) == 0 and unseen_tr != 0:
                opp_trump = True
        top = highest(hand & TM)
        boss = top >= 0 and (above(top) & ~seen) == 0
        if boss and opp_trump:  # pull trump while opponents may still hold some
            return top
        # opponents can't beat it (out of trump / suit is boss): cash Aces and 10s
        best, bk = -1, -1
        x = hand & ~TM
        while x:
            c = lowest(x)
            x &= x - 1
            safe = True
            for p in range(n):
                if team[p] == team[me]:
                    continue
                if could_beat(p, c, suit(c), trump, seen, voids):
                    safe = False
                    break
            if safe:
                k = GV[c % 13] * 100 + rank(c)
                if k > bk:
                    best, bk = c, k
        if best >= 0:
            return best
        if boss:
            return top
        if pull > 0 and opp_trump and (not rules[R_PULL_T2] or popcount(gone) == n):
            plain = np.int64(0)
            x = hand & TM
            while x:
                c = lowest(x)
                x &= x - 1
                if danger(c, trump, lowcap) == 0:
                    plain |= bit(c)
            if popcount(plain) >= pull:
                return lowest(plain)
        off = hand & ~TM
        if lead_aces:
            x = off
            while x:
                c = lowest(x)
                x &= x - 1
                if rank(c) == 14:
                    return c
        if rules[R_LEAD10] >= 2:
            x = off
            while x:
                c = lowest(x)
                x &= x - 1
                if rank(c) == 10 and popcount(above(c) & ~seen) <= rules[R_LEAD10] - 1:
                    return c
        best, bk = -1, 1 << 60
        x = off
        while x:
            c = lowest(x)
            x &= x - 1
            if rank(c) != 10:
                k = rank(c) * 4 + suit(c)
                if k < bk:
                    best, bk = c, k
        if best >= 0:
            return best
        if rules[R_LEAD10]:
            x = off
            while x:
                c = lowest(x)
                x &= x - 1
                if rank(c) == 10:
                    return c
        return min_shed(hand, trump, lowcap)

    lead = suit(tc[0])
    opts = legal(hand, lead, trump)
    wp, wc = trick_winner(tp, tc, tlen, trump)
    last = tlen == n - 1
    pts = 0
    tmask = np.int64(0)
    for i in range(tlen):
        pts += danger(tc[i], trump, lowcap)
        tmask |= bit(tc[i])

    beaters = np.int64(0)
    x = opts
    while x:
        c = lowest(x)
        x &= x - 1
        if beats(c, wc, lead, trump):
            beaters |= bit(c)

    bj = rules[R_BANK_JACK]
    jack = trump * 13 + 9
    if bj and (opts >> jack) & 1:
        unseen_hi = above(jack) & ~(gone | hand | tmask)
        if unseen_hi and (bj == 1 or popcount(above(jack) & hand) < popcount(unseen_hi)):
            # a Jack that isn't boss and isn't backed up: play it the moment it's safe
            if team[wp] == team[me]:
                if last or holds(wc, me, tlen, n, team, lead, trump, gone | hand | tmask, voids):
                    return jack
            elif beats(jack, wc, lead, trump) and (
                    last or holds(jack, me, tlen, n, team, lead, trump,
                                  gone | hand | tmask | bit(jack), voids)):
                return jack

    if team[wp] == team[me]:
        secure = last or holds(wc, me, tlen, n, team, lead, trump, gone | hand | tmask, voids)
        losers = opts & ~beaters
        if losers == 0:
            losers = opts
        if secure:  # smear points onto partner's trick
            best, bk = -1, -(1 << 60)
            for part in range(2):  # on ties, smear a card of the led suit before a trump
                x = losers & SM[lead] if part == 0 else losers & ~SM[lead]
                while x:
                    c = lowest(x)
                    x &= x - 1
                    k = danger(c, trump, lowcap) * 100 - rank(c)
                    if k > bk:
                        best, bk = c, k
            return best
        return min_shed(losers, trump, lowcap)

    if rules[R_BANK_LOW] and lead != trump:
        # trump with the 2/3 of trump: takes the trick and banks the Low point
        low = beaters & SM[trump] & (bit(trump * 13) | bit(trump * 13 + 1))
        if low:
            c = lowest(low)
            m = rules[R_BANK_LOW]
            if m == 3 or last or (m == 2 and pts > 0) or (m == 4 and pts >= ruff_min) or \
                    holds(c, me, tlen, n, team, lead, trump, gone | hand | tmask | bit(c), voids):
                return c
    winners = beaters
    if lead != trump and pts < ruff_min:  # not worth a trump
        winners &= ~TM
    if winners:
        if last:
            return min_cheap(winners, trump, lowcap)
        sure = np.int64(0)
        x = winners
        while x:
            c = lowest(x)
            x &= x - 1
            if holds(c, me, tlen, n, team, lead, trump, gone | hand | tmask | bit(c), voids):
                sure |= bit(c)
        if sure and (pts > 0 or (sure & ~TM) != 0):
            return min_cheap(sure, trump, lowcap)
        if winners & ~TM:
            return min_cheap(winners & ~TM, trump, lowcap)
        if pts > 0:
            return min_cheap(winners, trump, lowcap)
    ps = rules[R_PRESMEAR]
    if ps:
        # partner still to play could beat the winner: throw them an off-suit 10
        pard_can = False
        for i in range(1, n - tlen):
            q = (me + i) % n
            if team[q] == team[me] and (ps == 2 or i == n - tlen - 1) and \
                    could_beat(q, wc, lead, trump, gone | hand | tmask, voids):
                pard_can = True
        if pard_can:
            x = opts & ~TM & ~beaters
            while x:
                c = lowest(x)
                x &= x - 1
                if rank(c) == 10:
                    return c
    return min_shed(opts, trump, lowcap)


@kernel_fn
def holds(c, me, tlen, n, team, lead, trump, seen, voids):
    """Card c wins the trick against every opponent still to play (certain)."""
    for i in range(1, n - tlen):
        p = (me + i) % n
        if team[p] != team[me] and could_beat(p, c, lead, trump, seen, voids):
            return False
    return True


@kernel_fn
def pitch_card(hand, trump, mode, schoolboy):
    """mode: 0 top, 1 smart, 2 keepjack, 3 smart2 (see setback.OLD_RULES["pitch"])."""
    t = hand & SM[trump]
    top = highest(t)
    if schoolboy and rank(top) == 14 and (t >> (trump * 13 + 11)) & 1:
        return trump * 13 + 11  # schoolboy pitch: the King, hiding the Ace
    if mode == 0:
        return top
    not_jack = t & ~bit(trump * 13 + 9)
    if mode == 2:  # keepjack
        if rank(top) >= 12:
            return top
        x = t
        while x:
            c = lowest(x)
            x &= x - 1
            if 4 <= rank(c) <= 9:
                return c
        return lowest(not_jack) if not_jack else top
    if rank(top) == 14:
        return top
    # no Ace: lead a middle trump, keeping the Jack, 10 and low trump to score later
    x = t
    while x:
        c = lowest(x)
        x &= x - 1
        r = rank(c)
        if 4 <= r <= 9 or r == 12 or r == 13:
            return c
    if mode == 3 and not_jack:
        return lowest(not_jack)
    return top


@kernel_fn
def heur_pick(p, hands, tp, tc, tlen, gone, voids, trump, team, n, lowcap, rules):
    r = rules[team[p]]
    if gone == 0 and tlen == 0:  # the pitch
        return pitch_card(hands[p], trump, r[R_PITCH], r[R_SCHOOLBOY])
    return choose(p, hands[p], tp, tc, tlen, trump, gone, team, n, lowcap, voids, r)


# ---------------------------------------------------------------- scoring

@kernel_fn
def game_points(m):
    g = 0
    x = m
    while x:
        c = lowest(x)
        x &= x - 1
        g += GV[c % 13]
    return g


@kernel_fn
def score_into(won, trump, lowcap, lo_team, sc):
    """Points per team from the cards each team captured, written into sc."""
    nteams = won.shape[0]
    alltr = np.int64(0)
    for t in range(nteams):
        alltr |= won[t] & SM[trump]
        sc[t] = 0
    hi, lo = highest(alltr), lowest(alltr)
    jack = bit(trump * 13 + 9)
    best, nbest, bt = -1, 0, -1
    for t in range(nteams):
        if won[t] & bit(hi):
            sc[t] += 1
        if won[t] & jack:
            sc[t] += 1
        if lowcap and (won[t] & bit(lo)):
            sc[t] += 1
        g = game_points(won[t])
        if g > best:
            best, nbest, bt = g, 1, t
        elif g == best:
            nbest += 1
    if not lowcap:
        sc[lo_team] += 1
    if nbest == 1:  # tie for game: nobody scores it
        sc[bt] += 1


@kernel_fn
def search_value(won, trump, my_team):
    """What search maximizes: my team's points minus everyone else's (Low to the captor)."""
    nteams = won.shape[0]
    alltr = np.int64(0)
    for t in range(nteams):
        alltr |= won[t] & SM[trump]
    hi, lo = highest(alltr), lowest(alltr)
    jack = bit(trump * 13 + 9)
    own, total = 0, 0
    best, nbest, bt = -1, 0, -1
    for t in range(nteams):
        pts = 0
        if won[t] & bit(hi):
            pts += 1
        if won[t] & jack:
            pts += 1
        if won[t] & bit(lo):
            pts += 1
        total += pts
        if t == my_team:
            own += pts
        g = game_points(won[t])
        if g > best:
            best, nbest, bt = g, 1, t
        elif g == best:
            nbest += 1
    if nbest == 1:
        total += 1
        if bt == my_team:
            own += 1
    return 2 * own - total


@kernel_fn
def rollout(hands, tp, tc, tlen, leader, gone, won, voids, trump, team, n, lowcap, track, rules):
    """Play out from any point with rule-based players. Mutates its arguments (won holds the result)."""
    while True:
        if tlen == n:
            wp, _ = trick_winner(tp, tc, tlen, trump)
            for i in range(n):
                gone |= bit(tc[i])
                won[team[wp]] |= bit(tc[i])
            tlen = 0
            leader = wp
            if hands[leader] == 0:
                return
        p = (leader + tlen) % n
        c = heur_pick(p, hands, tp, tc, tlen, gone, voids, trump, team, n, lowcap, rules)
        hands[p] &= ~bit(c)
        if track:
            note_void(voids, p, tc, tlen, c, trump)
        tp[tlen] = p
        tc[tlen] = c
        tlen += 1


@kernel_fn
def pick_trump(hand, nplayers):
    """The suit the bidder names (trump_weights.py); ties go to the lower suit."""
    g = 1 if nplayers == 6 else 0
    best, bv = 0, np.int64(-(1 << 60))
    for s in range(4):
        x = hand & SM[s]
        if x == 0:
            continue
        v = TL[g, popcount(x)]
        while x:
            c = lowest(x)
            x &= x - 1
            v += TW[g, c % 13]
        if v > bv:
            best, bv = s, v
    return best
