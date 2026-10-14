"""Cards as small ints and hands as int64 bitmasks.

Standard cards are 0..51 = suit * 13 + (rank - 2), suits S H D C, ranks 2..14
(A = 14), so a suit's cards are 13 consecutive bits in rank order. Jokers, when
a game uses them, are 52 (little joker) and 53 (big joker).
"""
SUITS = "SHDC"
RANKS = "23456789TJQKA"
LITTLE_JOKER, BIG_JOKER = 52, 53
N_CARDS = 54

COLOR_MATE = (3, 2, 1, 0)  # same-colour suit: S<->C (black), H<->D (red)


def card(rank, suit):
    """rank 2..14, suit 0..3 -> card id."""
    return suit * 13 + rank - 2


def rank_of(c):
    return c % 13 + 2 if c < 52 else 0


def suit_of(c):
    return c // 13 if c < 52 else -1


def name(c):
    if c == LITTLE_JOKER:
        return "LJ"
    if c == BIG_JOKER:
        return "BJ"
    return RANKS[c % 13] + SUITS[c // 13]


def parse(tok):
    """'AS', '10H', 'TH', 'LJ', 'BJ' -> card id."""
    t = tok.strip().upper().replace("10", "T")
    if t in ("LJ", "LITTLE", "JOKER-", "JL"):
        return LITTLE_JOKER
    if t in ("BJ", "BIG", "JOKER+", "JB"):
        return BIG_JOKER
    return card(RANKS.index(t[0]) + 2, SUITS.index(t[1]))


def mask(cards):
    m = 0
    for c in cards:
        m |= 1 << c
    return m


def cards_in(m):
    return [c for c in range(N_CARDS) if (m >> c) & 1]


def deck_mask(jokers=0):
    """52 cards plus 0, 1 (big joker) or 2 jokers."""
    m = (1 << 52) - 1
    if jokers >= 1:
        m |= 1 << BIG_JOKER
    if jokers >= 2:
        m |= 1 << LITTLE_JOKER
    return m
