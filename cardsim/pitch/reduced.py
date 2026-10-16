"""Reduced hand classes for every-hand tables: hands that differ only in cards that barely
matter are played as one group.

The 962,988 hand classes (hands.py, up to suit relabeling) are merged further: for the trump
the bidder names, a card is *counted* rather than tracked when it carries no points, no Game
value and (for trumps) can't be Low, i.e. side-suit 2-9 (level "side") and also trump 4-9
(level "trump"). Two hands fall in one group when they hold the same tracked cards and the
same number of counted cards in each suit (the three side suits in any order).

Measured on the setback4 tables (2026-10-15): level "side" leaves 68,677 groups 4-handed
(14x fewer) with no detectable difference between a hand and its group; level "trump"
leaves 18,759 (51x fewer) at about 0.02 points RMS, below the tables' sampling noise.

A group is played by drawing, for each deal, one member hand in proportion to how many real
hands it stands for, so the group's average is unbiased; every member gets the group's numbers.
"""
import numpy as np
from numba import njit

from cardsim.pitch.fast import pick_trump
from cardsim.pitch.tables import F_ESUIT, F_GV, F_NRANK, F_PTS

LEVELS = ("side", "trump")
_CACHE = {}


def counted_ranks(tab, level):
    """(side, trump): 13-bit masks of the ranks counted rather than tracked, read from this
    variant's tables (trump 0, side suit 1; eligible presets have no off-jack or jokers)."""
    side = trump = 0
    for r in range(13):
        sc, tc = 13 + r, r  # the rank in side suit 1 and in trump suit 0, trump 0
        if tab[0, F_ESUIT, sc] == 1 and tab[0, F_PTS, sc] == 0 and tab[0, F_GV, sc] == 0:
            side |= 1 << r
        if level == "trump" and tab[0, F_PTS, tc] == 0 and tab[0, F_GV, tc] == 0 and tab[0, F_NRANK, tc] >= 4:
            trump |= 1 << r
    return side, trump


@njit(cache=True)
def _describe(keys, trumps, side_m, trump_m):
    """One row per class: trump suit (tracked cards, count), then the three side suits sorted."""
    out = np.zeros((keys.shape[0], 8), np.int64)
    for i in range(keys.shape[0]):
        h, t = keys[i], trumps[i]
        tm = (h >> (13 * t)) & 0x1FFF
        out[i, 0] = tm & ~trump_m
        c = tm & trump_m
        n = 0
        while c:
            c &= c - 1
            n += 1
        out[i, 1] = n
        side = np.empty(3, np.int64)
        k = 0
        for s in range(4):
            if s == t:
                continue
            sm = (h >> (13 * s)) & 0x1FFF
            c = sm & side_m
            n = 0
            while c:
                c &= c - 1
                n += 1
            side[k] = ((sm & ~side_m) << 4) | n  # tracked cards and count, one sortable number
            k += 1
        side.sort()
        out[i, 2], out[i, 3], out[i, 4] = side[0], side[1], side[2]
    return out


def groups(game, level="side"):
    """For a fast.Game (its tables and trump weights): (gid per class, number of groups, members
    of each group as CSR: order (class indices sorted by group), starts, cumulative weights
    within the group, class keys, class trumps, class weights)."""
    from cardsim.hands import hand_classes
    ck = (game.rules.name, game.nplayers, level, game.tw.tobytes(), game.tl.tobytes(), game.tx.tobytes())
    if ck in _CACHE:
        return _CACHE[ck]
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    keys, counts = hand_classes()
    trumps = _trumps(keys, game.tab, game.tw, game.tl, game.tx)
    side_m, trump_m = counted_ranks(game.tab, level)
    desc = _describe(keys, trumps, side_m, trump_m)
    _, gid = np.unique(desc, axis=0, return_inverse=True)
    gid = gid.ravel()
    ng = int(gid.max()) + 1
    order = np.argsort(gid, kind="stable")
    starts = np.searchsorted(gid[order], np.arange(ng + 1))
    w = counts[order].astype(np.float64)
    cum = np.empty_like(w)
    for g in range(ng):  # cumulative weight within each group
        a, b = starts[g], starts[g + 1]
        cum[a:b] = np.cumsum(w[a:b])
    res = (gid, ng, order, starts, cum, keys, trumps, counts)
    _CACHE[ck] = res
    return res


@njit(cache=True)
def _trumps(keys, tab, tw, tl, tx):
    out = np.empty(keys.shape[0], np.int64)
    for i in range(keys.shape[0]):
        out[i] = pick_trump(keys[i], tab, tw, tl, tx)
    return out


def members(game, level, seed, start, end, deals):
    """For groups [start, end): the class index played in each deal, (end-start, deals). Group
    r's draws depend only on (seed, r), so slices of a study reproduce the whole."""
    gid, ng, order, starts, cum, keys, trumps, counts = groups(game, level)
    out = np.empty((end - start, deals), np.int64)
    for i, r in enumerate(range(start, end)):
        a, b = starts[r], starts[r + 1]
        u = np.random.default_rng([seed, r]).random(deals) * cum[b - 1]
        out[i] = order[a + np.minimum(np.searchsorted(cum[a:b], u, side="right"), b - a - 1)]
    return out
