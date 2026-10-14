"""Check the Numba engine against the pure-Python reference engine.

Rule-based play must match card for card (hands are fed to Python sorted by
suit then rank, which is the Numba engine's iteration order). Search uses
different random numbers, so it is compared statistically.

    .venv/Scripts/python test_fast.py
"""
import random
import time

import numpy as np

import setback as sb
import setback_fast as fast


def to_mask(cards):
    m = 0
    for r, s in cards:
        m |= 1 << (s * 13 + r - 2)
    return m


def check_rules_match(nplayers, low_captured, deals=5000):
    rng = random.Random(nplayers * 10 + low_captured)
    team = sb.make_teams(nplayers, 2)
    nteams = max(team) + 1
    team_np = np.array(team, np.int64)
    variants = [sb.OLD_RULES, sb.default_rules(nplayers),
                dict(sb.OLD_RULES, pull=1, lead_aces=False, ruff_min=10, pitch="smart"),
                dict(sb.OLD_RULES, ruff_min=5, pitch="keepjack", bank_low=1, schoolboy=True),
                dict(sb.OLD_RULES, ruff_min=8, pitch="smart2", bank_low=2),
                dict(sb.OLD_RULES, pitch="smart2", bank_low=3),
                dict(sb.default_rules(nplayers), bank_jack=1, lead10=1, presmear=1, bank_low=4),
                dict(sb.default_rules(nplayers), bank_jack=2, presmear=2),
                sb.team_rules(nplayers, team),
                dict(sb.default_rules(nplayers), pull=2, pull_t2=True, lead10=3)]
    for rules in variants:
        rules_np = fast.rules_array(rules if isinstance(rules, list) else [rules] * nteams)
        for _ in range(deals):
            deck = sb.DECK[:]
            rng.shuffle(deck)
            hands = [sorted(deck[i * 6:(i + 1) * 6], key=lambda c: (c[1], c[0])) for i in range(nplayers)]
            trump = sb.pick_trump(hands[0], nplayers)
            assert fast.pick_trump(np.int64(to_mask(hands[0])), nplayers) == trump
            py = sb.play(hands, 0, trump, team, low_captured, rules=rules)
            nb = fast.play_game(np.array([to_mask(h) for h in hands], np.int64), 0, trump, team_np,
                                nteams, low_captured, True, rules_np, np.zeros(nteams, np.int64))
            assert list(nb) == py, (hands, trump, rules, py, list(nb))
    print(f"  {nplayers}-handed, low {'captured' if low_captured else 'dealt'}: "
          f"{deals * len(variants)} deals identical")


def check_search_stats(nplayers, deals, samples=16):
    """Search for the bidder vs rules for everyone: gain in bidder-team points, both engines."""
    team = sb.make_teams(nplayers, 2)
    nteams = max(team) + 1
    rules = sb.team_rules(nplayers, team)
    rules_np = fast.rules_array(rules)
    search = np.zeros(nteams, np.int64)
    search[team[0]] = samples
    t0 = time.time()
    r = fast.head_to_head(deals, nplayers, np.array(team, np.int64), nteams, True, True,
                          rules_np, np.zeros(nteams, np.int64), rules_np, search, 12345)
    t_fast = time.time() - t0
    d = r[:, 1] - r[:, 0]
    print(f"  {nplayers}-handed numba: search gain {d.mean():+.3f} +/- {d.std() / np.sqrt(len(d)):.3f}"
          f"  ({deals} deals, {t_fast:.1f}s)")

    rng = random.Random(7)
    py_d = []
    t0 = time.time()
    pdeals = max(100, deals // 40)
    pol = [0] * nteams
    pol[team[0]] = samples
    for _ in range(pdeals):
        deck = sb.DECK[:]
        rng.shuffle(deck)
        hands = [deck[i * 6:(i + 1) * 6] for i in range(nplayers)]
        trump = sb.pick_trump(hands[0], nplayers)
        a = sb.play(hands, 0, trump, team, rules=rules)[team[0]]
        b = sb.play(hands, 0, trump, team, rules=rules, search=pol, rng=rng)[team[0]]
        py_d.append(b - a)
    py_d = np.array(py_d)
    print(f"  {nplayers}-handed python: search gain {py_d.mean():+.3f} +/- "
          f"{py_d.std() / np.sqrt(len(py_d)):.3f}  ({pdeals} deals, {time.time() - t0:.1f}s)")


if __name__ == "__main__":
    print("rule-based players, card-for-card:")
    for n in (4, 6):
        for lc in (True, False):
            check_rules_match(n, lc)
    print("search player, statistically:")
    for n in (4, 6):
        check_search_stats(n, 8000)
