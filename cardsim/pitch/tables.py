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
N_FIELDS = 7

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
