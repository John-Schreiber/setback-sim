"""The generalized GPU engine against the generalized CPU engine, for every preset.

Both rule-based and search games must match exactly: same deals, same points. Search
draws from the same seeded streams with the same shuffle on both (core.sample_world).

    .venv/Scripts/python tests/test_pitch_gpu.py
"""
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cardsim.pitch import PRESETS, PitchRules  # noqa: E402
from cardsim.pitch import fast, gpu  # noqa: E402


def game_for(preset, n):
    r = PitchRules.preset(preset)
    if n * r.deal_size > 52 + r.jokers:
        return None
    return fast.Game(r, n)


def check_rules_exact(preset, n, deals=50_000):
    g = game_for(preset, n)
    if g is None:
        return
    r = g.rules
    zero = np.zeros(g.nteams, np.int64)
    cpu = fast.survey(deals, n, r.deal_size, r.hand_size, r.refill, g.team, g.nteams, True, g.rules_bt,
                      zero, g.tab, g.gf, g.tw, g.tl, g.tx, 777)
    gp = gpu.survey(deals, g, zero, seed=777)
    assert np.array_equal(cpu, gp), (preset, n, np.argwhere((cpu != gp).any(axis=1))[:5])
    print(f"  {preset:20} {n}-handed: {deals} rule-based games identical")


def check_search_exact(preset, n, deals=1500, samples=8):
    g = game_for(preset, n)
    if g is None:
        return
    r = g.rules
    s = np.full(g.nteams, samples, np.int64)
    t0 = time.time()
    cpu = fast.survey(deals, n, r.deal_size, r.hand_size, r.refill, g.team, g.nteams, True, g.rules_bt, s,
                      g.tab, g.gf, g.tw, g.tl, g.tx, 999, 5000)
    tc = time.time() - t0
    t0 = time.time()
    gp = gpu.survey(deals, g, s, seed=999, start=5000, batch=600)
    tg = time.time() - t0
    assert np.array_equal(cpu, gp), (preset, n, np.argwhere((cpu != gp).any(axis=1))[:5])
    print(f"  {preset:20} {n}-handed: {deals} search games identical (cpu {tc:.0f}s, gpu {tg:.0f}s)")


if __name__ == "__main__":
    print("rule-based players, GPU vs CPU, exact:")
    for preset in PRESETS:
        for n in (4, 6):
            check_rules_exact(preset, n)
    print("search players, GPU vs CPU, exact:")
    for preset in PRESETS:
        for n in (4, 6):
            check_search_exact(preset, n)
