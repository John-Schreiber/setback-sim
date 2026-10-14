"""Whole games with an auction (cardsim.pitch.game), for every house-rule preset.

Checks the chart lookup (Brain) and the hand key it uses, then plays short matches
under each preset: every game must end with a winner who met the rules' winning
condition, sets can't outnumber won auctions, the same seed must give the same games,
and two teams with the same strategy must win about equally often.

    .venv/Scripts/python tests/test_pitch_game.py
"""
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cardsim.cards import mask, parse  # noqa: E402
from cardsim.pitch import PRESETS, PitchRules, fast  # noqa: E402
from cardsim.pitch.game import (C_AUC0, C_HANDS, C_MOON0, C_MOONMADE0, C_S0, C_SET0, C_WIN,  # noqa: E402
                                MAX_SMALL, STRATS, Brain, hand_key, play_games, rule_params)


def setup(preset, n):
    rules = PitchRules.preset(preset)
    g = fast.Game(rules, n)
    brain = Brain(rules, n)
    take = np.zeros(64, np.int64)
    for lv in rules.dealer_takes:
        take[lv] = 1
    return rules, g, brain, take


def run(rules, g, brain, take, n, strats, games, seed):
    strat = np.array([STRATS[s.replace("-ns", "")] for s in strats], np.int64)
    ns = np.array([int(s.endswith("-ns")) for s in strats], np.int64)
    return play_games(games, n, g.team, strat, ns, rule_params(rules), take, brain.maxbid, brain.moon,
                      brain.honor_cards, g.rules_bt, g.tab, g.gf, g.tw, g.tl, g.tx, seed)


def test_hand_key():
    rules, g, brain, _ = setup("setback4", 4)
    # spades trump: A and J are honors (bits 0 and 3 of A K Q J 10 2), 5 and 3 are small
    hand = np.int64(mask(parse(c) for c in "AS JS 5S 3S KH 2D".split()))
    assert hand_key(hand, 0, brain.honor_cards, g.tab) == (1 | 8) * MAX_SMALL + 2
    # hearts trump: K is the only honor, nothing small
    assert hand_key(hand, 1, brain.honor_cards, g.tab) == 2 * MAX_SMALL + 0
    print("  hand key: honors and small trumps counted right")


def test_brain(preset, n):
    rules, g, brain, _ = setup(preset, n)
    mb = brain.maxbid
    assert ((mb == 0) | ((mb >= rules.min_bid) & (mb <= rules.max_bid))).all(), "bid out of range"
    assert (mb > 0).any(), "a chart with no biddable hand"
    if not rules.moon_bid:
        assert not brain.moon.any(), "moon bids under rules without one"


def test_games(preset, n, games=400):
    rules, g, brain, take = setup(preset, n)
    o = run(rules, g, brain, take, n, ("table", "cheap"), games, 5)
    assert (o == run(rules, g, brain, take, n, ("table", "cheap"), games, 5)).all(), "same seed, different games"
    assert (o[:, C_WIN] >= 0).all(), "a game without a winner"
    assert (o[:, C_HANDS] > 0).all()
    k = g.nteams
    assert (o[:, C_SET0:C_SET0 + k] <= o[:, C_AUC0:C_AUC0 + k]).all(), "more sets than auctions won"
    assert (o[:, C_MOONMADE0:C_MOONMADE0 + k] <= o[:, C_MOON0:C_MOON0 + k]).all()
    assert (o[:, C_AUC0:C_AUC0 + k].sum(axis=1) == o[:, C_HANDS]).all(), "one auction per hand played"
    for row in o:
        w = row[C_WIN]
        s = row[C_S0:C_S0 + k]
        moon_win = rules.moon_wins_game and row[C_MOONMADE0 + w] > 0
        back = rules.backdoor < 0 and s.min() <= rules.backdoor
        ahead = all(s[w] - s[j] >= rules.win_by for j in range(k) if j != w)
        assert moon_win or back or (s[w] >= rules.target and ahead), f"team {w} won with scores {s}"
    print(f"  {preset:20} {n}-handed: {games} games, all finished legally, "
          f"{o[:, C_HANDS].mean():.1f} hands per game")


def test_fair(preset="setback4", n=4, games=20_000):
    """Same strategy for every team: each wins 1/teams of the games (dealer rotates)."""
    rules, g, brain, take = setup(preset, n)
    o = run(rules, g, brain, take, n, ("table",) * g.nteams, games, 9)
    p = 1 / g.nteams
    se = np.sqrt(p * (1 - p) / games)
    wins = [(o[:, C_WIN] == k).mean() for k in range(g.nteams)]
    for k, w in enumerate(wins):
        assert abs(w - p) < 4 * se, f"team {k} wins {w:.3f} (expected {p:.3f}) with identical strategies"
    print(f"  {preset} {n}-handed, table for every team: wins " + " / ".join(f"{w:.1%}" for w in wins) +
          f" (+/- {se:.1%} each)")


if __name__ == "__main__":
    print("whole games with an auction:")
    test_hand_key()
    for preset in PRESETS:
        for n in (4, 6):
            r = PitchRules.preset(preset)
            if n * r.deal_size > 52 + r.jokers:
                continue
            test_brain(preset, n)
            test_games(preset, n)
    test_fair()
    test_fair(n=6, games=30_000)
