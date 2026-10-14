"""Per-trump card tables and game flags for the compiled engines.

For each trump suit t, every card c gets its effective suit, its power (higher
beats lower within an effective suit), the mask of cards that outrank it, its
point value when captured, its Game count and its natural rank. With the
off-jack or jokers in play they become trumps; otherwise the tables reduce to
plain suits and ranks, so the original game is unchanged.

Everything is packed into one int64 array TAB[t, field, i] (and a small flags
array GF) so the same code runs on the CPU and as CUDA device functions.
"""
import numpy as np

from ..cards import BIG_JOKER, COLOR_MATE, LITTLE_JOKER, N_CARDS, card, deck_mask, rank_of, suit_of

# TAB fields (third index is the card, except ESM: third index is the suit 0..3)
F_ESUIT, F_POWER, F_ABOVE, F_PTS, F_GV, F_NRANK, F_ESM = range(7)
# precomputed per-trump sort keys of the rule-based player (one lookup instead of
# recomputing through several reads and branches each time a card is compared)
F_DANGER, F_SHED, F_CHEAP, F_SMEAR, F_CASH, F_SAFE = range(7, 13)
# F_MASKS: per-trump masks and card ids (third index one of the M_* slots below) that turn
# card-by-card scans into a bit operation or two
F_MASKS = 13
N_FIELDS = 14
(M_TR_LOW, M_TR_HIGH, M_LJ, M_BJ, M_OFFJ, M_SAFE_TR, M_POINTS, M_ACES, M_TENS, M_KINGS, M_QUEENS,
 M_JACKS, M_MID, M_MID2) = range(14)

# GF flags
(G_DECK, G_MAY_TRUMP, G_FIRST_LEAD, G_TRUMP_BROKEN, G_LOWCAP, G_HIGH, G_LOW, G_GAME, G_LAST,
 G_HAND, G_NPLAYERS) = range(11)
N_FLAGS = 11
FIRST_LEAD = {"trump": 0, "any": 1, "not_trump": 2}

GAME_COUNT = {10: 10, 14: 4, 13: 3, 12: 2, 11: 1}
POWER_JACK, POWER_OFF_JACK, POWER_BIG_JOKER, POWER_LITTLE_JOKER = 110, 105, 104, 103


def build_tables(rules):
    """TAB[4 trumps, N_FIELDS, 54] for these rules."""
    deck = deck_mask(rules.jokers)
    tab = np.zeros((4, N_FIELDS, N_CARDS), np.int64)
    for t in range(4):
        off_jack = card(11, COLOR_MATE[t]) if rules.off_jack else -1
        for c in range(N_CARDS):
            if not (deck >> c) & 1:
                tab[t, F_ESUIT, c] = -1
                continue
            r, s = rank_of(c), suit_of(c)
            if c in (BIG_JOKER, LITTLE_JOKER):
                es, power = t, POWER_BIG_JOKER if c == BIG_JOKER else POWER_LITTLE_JOKER
            elif c == off_jack:
                es, power = t, POWER_OFF_JACK
            else:
                es, power = s, r * 10
            tab[t, F_ESUIT, c] = es
            tab[t, F_POWER, c] = power
            tab[t, F_NRANK, c] = r
            tab[t, F_GV, c] = GAME_COUNT.get(r, 0)
            pts = 0
            if es == t:
                if c == card(11, t):
                    pts += rules.jack
                elif c == off_jack:
                    pts += rules.off_jack_pts
                elif c in (BIG_JOKER, LITTLE_JOKER):
                    pts += rules.joker_pts
                elif s == t:
                    pts += {10: rules.ten_pts, 3: rules.three_pts, 5: rules.five_pts,
                            9: rules.nine_pts}.get(r, 0)
            tab[t, F_PTS, c] = pts
        for c in range(N_CARDS):
            es = tab[t, F_ESUIT, c]
            if es < 0:
                continue
            tab[t, F_ESM, es] |= np.int64(1) << np.int64(c)
        for c in range(N_CARDS):
            es = tab[t, F_ESUIT, c]
            if es < 0:
                continue
            above = 0
            for d in range(N_CARDS):
                if tab[t, F_ESUIT, d] == es and tab[t, F_POWER, d] > tab[t, F_POWER, c]:
                    above |= 1 << d
            tab[t, F_ABOVE, c] = above
    return add_keys(tab, rules)


def add_keys(tab, rules):
    """Fill the precomputed sort keys (danger and the orderings built from it)."""
    for t in range(4):
        for c in range(N_CARDS):
            es = tab[t, F_ESUIT, c]
            if es < 0:
                continue
            power, trump = tab[t, F_POWER, c], int(es == t)
            d = tab[t, F_GV, c] + 15 * tab[t, F_PTS, c]
            if rules.low_to_captor and c < 52 and c // 13 == t and tab[t, F_NRANK, c] <= 3:
                d += 8
            tab[t, F_DANGER, c] = d
            tab[t, F_SHED, c] = d * 10000 + trump * 1000 + power
            tab[t, F_CHEAP, c] = trump * 100000000 + d * 1000 + power
            tab[t, F_SMEAR, c] = d * 1000 - power
            tab[t, F_CASH, c] = tab[t, F_GV, c] * 1000 + power
            tab[t, F_SAFE, c] = power * 4 + es
    deck = deck_mask(rules.jokers)

    def ranks(*rs):
        return sum(1 << c for c in range(52) if (deck >> c) & 1 and rank_of(c) in rs)

    for t in range(4):
        M = tab[t, F_MASKS]
        # trump power order: natural 2..10 < little joker < big joker < off-jack < natural J..A
        M[M_TR_LOW] = 0x1FF << (13 * t)
        M[M_TR_HIGH] = 0xF << (13 * t + 9)
        M[M_LJ] = LITTLE_JOKER if tab[t, F_ESUIT, LITTLE_JOKER] == t else -1
        M[M_BJ] = BIG_JOKER if tab[t, F_ESUIT, BIG_JOKER] == t else -1
        oj = card(11, COLOR_MATE[t])
        M[M_OFFJ] = oj if tab[t, F_ESUIT, oj] == t else -1
        for c in range(N_CARDS):
            if tab[t, F_ESUIT, c] == t and tab[t, F_DANGER, c] == 0:
                M[M_SAFE_TR] |= 1 << c
            if tab[t, F_ESUIT, c] >= 0 and tab[t, F_PTS, c] != 0:
                M[M_POINTS] |= 1 << c
        M[M_ACES], M[M_TENS], M[M_KINGS], M[M_QUEENS], M[M_JACKS] = (ranks(14), ranks(10), ranks(13),
                                                                    ranks(12), ranks(11))
        M[M_MID] = ranks(4, 5, 6, 7, 8, 9)
        M[M_MID2] = ranks(4, 5, 6, 7, 8, 9, 12, 13)
        assert GAME_COUNT == {10: 10, 14: 4, 13: 3, 12: 2, 11: 1}  # game_points relies on it
        for c in range(N_CARDS):  # the order top_trump / low_trump rely on
            if tab[t, F_ESUIT, c] == t and c < 52 and c // 13 == t:
                assert (tab[t, F_POWER, c] < POWER_LITTLE_JOKER) == (rank_of(c) <= 10)
    return tab


def build_flags(rules, nplayers):
    gf = np.zeros(N_FLAGS, np.int64)
    gf[G_DECK] = deck_mask(rules.jokers)
    gf[G_MAY_TRUMP] = rules.may_trump
    gf[G_FIRST_LEAD] = FIRST_LEAD[rules.first_lead]
    gf[G_TRUMP_BROKEN] = rules.trump_broken
    gf[G_LOWCAP] = rules.low_to_captor
    gf[G_HIGH], gf[G_LOW], gf[G_GAME], gf[G_LAST] = rules.high, rules.low, rules.game, rules.last_trick
    gf[G_HAND] = rules.hand_size
    gf[G_NPLAYERS] = nplayers
    return gf
