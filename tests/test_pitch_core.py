"""The generalized Pitch engine (cardsim.pitch) must reproduce the original engine
(setback_fast.py) exactly under the setback4 preset: same deals, same trump, same
cards captured, same points, for rule-based and search players.

    .venv/Scripts/python tests/test_pitch_core.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import setback as sb  # noqa: E402
import setback_fast as old  # noqa: E402
from cardsim.pitch import PitchRules  # noqa: E402
from cardsim.pitch import fast as new  # noqa: E402

RULE_VARIANTS = [
    lambda n: [sb.OLD_RULES],
    lambda n: [sb.default_rules(n)],
    lambda n: [dict(sb.OLD_RULES, pull=1, lead_aces=False, ruff_min=10, pitch="smart")],
    lambda n: [dict(sb.OLD_RULES, ruff_min=5, pitch="keepjack", bank_low=1, schoolboy=True)],
    lambda n: [dict(sb.OLD_RULES, ruff_min=8, pitch="smart2", bank_low=2)],
    lambda n: [dict(sb.default_rules(n), bank_jack=1, lead10=1, presmear=1, bank_low=4)],
    lambda n: [dict(sb.default_rules(n), bank_jack=2, presmear=2, lead10=3)],
    lambda n: sb.team_rules(n, sb.make_teams(n, 2)),
]


def check_rules(nplayers, preset, deals=3000):
    g = new.Game(PitchRules.preset(preset), nplayers)
    lowcap = g.rules.low_to_captor
    zero = np.zeros(g.nteams, np.int64)
    for vi, make in enumerate(RULE_VARIANTS):
        rl = make(nplayers)
        rules_old = old.rules_array(rl * g.nteams if len(rl) == 1 else rl)
        rules_new = np.array([new.rules_row(r) for r in (rl * g.nteams if len(rl) == 1 else rl)], np.int64)
        for i in range(deals):
            seed = 1000 * vi + i
            hands, t = new.deal(seed, nplayers, 6, 6, False, 0, g.tab, g.gf, g.tw, g.tl)
            oh, ot, _ = old.deal_batch(seed, 1, nplayers, 0)
            assert np.array_equal(hands, oh[0]) and t == ot[0], ("deal", seed)
            sc_o, won_o = old.play_game_won(hands, 0, t, g.team, g.nteams, lowcap, True, rules_old, zero)
            sc_n, won_n, _ = new.play_game_won(hands, 0, t, g.team, g.nteams, True, rules_new, zero,
                                               g.tab, g.gf)
            assert np.array_equal(sc_o, sc_n) and np.array_equal(won_o, won_n), (vi, seed, sc_o, sc_n)
    print(f"  {nplayers}-handed {preset}: {deals * len(RULE_VARIANTS)} deals identical (rule-based)")


def check_search(nplayers, deals=300, samples=8):
    g = new.Game(PitchRules.preset("setback4"), nplayers)
    rules = g.rules_bt[g.team[0]]
    s = np.full(g.nteams, samples, np.int64)
    a = old.survey(deals, nplayers, g.team, g.nteams, True, True, rules, s, 4242)
    b = new.survey(deals, nplayers, 6, 6, False, g.team, g.nteams, True, g.rules_bt, s, g.tab, g.gf,
                   g.tw, g.tl, 4242)
    assert np.array_equal(a[:, old.F_OWN], b[:, 2]) and np.array_equal(a[:, old.F_OPP], b[:, 3])
    print(f"  {nplayers}-handed setback4: {deals} deals identical (search, {samples} samples)")


if __name__ == "__main__":
    print("generalized engine vs original engine:")
    for n in (4, 6):
        check_rules(n, "setback4")
        check_rules(n, "setback4_lowdealt", deals=1000)
        check_search(n)
