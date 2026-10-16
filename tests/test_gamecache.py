"""The outcome cache (cardsim/pitch/gamecache.py) against game.play_games: with rule-based
players the cached games must be identical, game for game, for every preset. Then a
look-ahead run fills the cache on the GPU and must finish every game.

    .venv/Scripts/python tests/test_gamecache.py
"""
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cardsim.pitch import PRESETS, PitchRules, fast  # noqa: E402
from cardsim.pitch.game import STRATS, Brain, play_games, rule_params  # noqa: E402
from cardsim.pitch.gamecache import Cache  # noqa: E402


def setup(preset, n):
    rules = PitchRules.preset(preset)
    if n * rules.deal_size > 52 + rules.jokers:
        return None
    g = fast.Game(rules, n)
    brain = Brain(rules, n)
    take = np.zeros(64, np.int64)
    for lv in rules.dealer_takes:
        take[lv] = 1
    return rules, g, brain, take


def check_rules(preset, n, games=3000, seed=17):
    s = setup(preset, n)
    if s is None:
        return
    rules, g, brain, take = s
    strat = np.array([STRATS["cheap"], STRATS["table"]] + [STRATS["table"]] * (g.nteams - 2), np.int64)
    ns = np.zeros(g.nteams, np.int64)
    mb, mo = np.stack([brain.maxbid] * g.nteams), np.stack([brain.moon] * g.nteams)
    ref = play_games(games, n, g.team, strat, ns, rule_params(rules), take, mb, mo, brain.honor_cards, g.rules_bt,
                     g.tab, g.gf, g.tw, g.tl, g.tx, seed)
    cache = Cache(g, games, seed, np.zeros(g.nteams, np.int64), max_deals=200)
    got = cache.run(strat, ns, take, mb, mo, brain.honor_cards)
    assert np.array_equal(ref, got), (preset, n, np.argwhere((ref != got).any(axis=1))[:5])
    print(f"  {preset:21} {n}-handed: {games} cached games identical to play_games "
          f"({cache.played:,} hands cached, {ref[:, 1].sum():,} played)")


def check_search(preset="setback4", n=4, games=2000):
    rules, g, brain, take = setup(preset, n)
    strat = np.full(g.nteams, STRATS["cheap"], np.int64)
    ns = np.zeros(g.nteams, np.int64)
    mb, mo = np.stack([brain.maxbid] * g.nteams), np.stack([brain.moon] * g.nteams)
    t0 = time.time()
    cache = Cache(g, games, 5, np.full(g.nteams, 16, np.int64), max_deals=200)
    out = cache.run(strat, ns, take, mb, mo, brain.honor_cards)
    assert (out[:, 0] >= 0).all()
    print(f"  {preset} {n}-handed, look-ahead players: {games} games finished, {cache.played:,} hands played "
          f"on the GPU in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    print("cached games vs play_games, rule-based players:")
    for preset in PRESETS:
        for n in (4, 6):
            check_rules(preset, n)
    print("look-ahead players:")
    check_search()
