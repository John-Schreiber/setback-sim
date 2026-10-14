"""Distinct 6-card hands from a 52-card deck, up to relabeling suits.

All C(52,6) = 20,358,520 hands reduce to 962,988 classes; hands in a class play the
same once the suits are renamed. The class key sorts the four 13-bit suit masks
(largest first) and repacks them.
"""
import numpy as np
from numba import njit

N_HANDS = 20_358_520
N_CLASSES = 962_988


@njit(cache=True)
def canonical(hand):
    m = np.empty(4, np.int64)
    for s in range(4):
        m[s] = (hand >> (13 * s)) & 0x1FFF
    m.sort()
    return m[3] | (m[2] << 13) | (m[1] << 26) | (m[0] << 39)


@njit(cache=True)
def _all_canonical():
    out = np.empty(N_HANDS, np.int64)
    k = 0
    for a in range(52):
        for b in range(a + 1, 52):
            for c in range(b + 1, 52):
                for d in range(c + 1, 52):
                    for e in range(d + 1, 52):
                        for f in range(e + 1, 52):
                            h = (np.int64(1) << a) | (np.int64(1) << b) | (np.int64(1) << c) | \
                                (np.int64(1) << d) | (np.int64(1) << e) | (np.int64(1) << f)
                            out[k] = canonical(h)
                            k += 1
    return out


_CACHE = {}


def hand_classes():
    """(class keys sorted, how many of the 20,358,520 hands each stands for)."""
    if "classes" not in _CACHE:
        keys, counts = np.unique(_all_canonical(), return_counts=True)
        assert len(keys) == N_CLASSES and counts.sum() == N_HANDS
        _CACHE["classes"] = (keys, counts)
    return _CACHE["classes"]
