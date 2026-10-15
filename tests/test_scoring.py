"""Hand scoring and the bid charts' expected points: a tie for Game scores nobody, and
the charts' net columns charge every bid the opponents' points on the hand.

    .venv/Scripts/python tests/test_scoring.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cardsim.cards import card  # noqa: E402
from cardsim.pitch import PitchRules  # noqa: E402
from cardsim.pitch import fast  # noqa: E402
from cardsim.pitch.bidchart import aggregate  # noqa: E402
from cardsim.pitch.core import bit, score_into  # noqa: E402


def mask(*cards):
    m = 0
    for c in cards:
        m |= bit(c)
    return np.int64(m)


def test_game_tie():
    """setback4, suit 0 trump: team 0 takes High (A, 4 for game) and Low plus a plain Ten
    (14 game); team 1 takes the Jack (1), a plain Ten and a plain King (14 game). Game is
    tied and nobody scores it. One more plain Jack for team 0 breaks the tie."""
    g = fast.Game(PitchRules.preset("setback4"), 4)
    t = 0
    a = [card(14, t), card(2, t), card(10, 1)]
    b = [card(11, t), card(10, 2), card(13, 3)]
    sc = np.zeros(2, np.int64)
    score_into(np.array([mask(*a), mask(*b)]), t, 0, -1, sc, g.tab, g.gf)
    assert list(sc) == [2, 1], sc
    score_into(np.array([mask(*a, card(11, 3)), mask(*b)]), t, 0, -1, sc, g.tab, g.gf)
    assert list(sc) == [3, 1], sc
    print("  tie for Game scores nobody; broken tie scores the leader")


def test_net_columns():
    """Bid 2 in setback4: made with 2 while the opponents take 2 nets 0; set (1 point)
    while they take 3 nets -2 - 3 = -5."""
    rules = PitchRules.preset("setback4")
    g = fast.Game(rules, 4)
    out = np.zeros((2, 6), np.int64)
    out[:, fast.S_OWN] = [2, 1]
    out[:, fast.S_OPP] = [2, 3]
    r = aggregate(rules, g, out).iloc[0]
    assert r.hands == 2 and r.opp == 2.5, r
    assert r.ev2 == (2 - 2) / 2 and r.net2 == (0 - 5) / 2, r
    assert r.ev3 == (-3 - 3) / 2 and r.net3 == (-3 - 2 - 3 - 3) / 2, r
    print("  net = score change minus the opponents' points")


if __name__ == "__main__":
    test_game_tie()
    test_net_columns()
