"""Check the GPU engine against the CPU (Numba) engine.

Both engines play the same deals for the same seed. Rule-based games must match
exactly; search uses different random numbers, so on the same deals the
per-deal differences must average out to zero.

    .venv/Scripts/python test_gpu.py
"""
import time

import numpy as np

import setback as sb
import setback_fast as fast
import setback_gpu as gpu


def setup(nplayers, search):
    team = np.array(sb.make_teams(nplayers, 2), np.int64)
    nteams = int(team.max()) + 1
    rules = fast.rules_array(sb.team_rules(nplayers, list(team)))
    return team, nteams, rules, np.full(nteams, search, np.int64)


def check_rules_exact(nplayers, lowcap, n=200_000):
    team, nteams, rules, s = setup(nplayers, 0)
    cpu = fast.survey(n, nplayers, team, nteams, lowcap, True, rules, s, 555)
    g = gpu.survey(n, nplayers, team, rules, s, lowcap, True, seed=555)
    same = np.array_equal(cpu, g)
    print(f"  {nplayers}-handed, low {'captured' if lowcap else 'dealt'}: {n} games "
          f"{'identical' if same else 'DIFFERENT'}")
    assert same, np.argwhere((cpu != g).any(axis=1))[:5]


def check_search_stats(nplayers, n, samples=16):
    team, nteams, rules, s = setup(nplayers, samples)
    t0 = time.time()
    cpu = fast.survey(n, nplayers, team, nteams, True, True, rules, s, 777)
    t_cpu = time.time() - t0
    t0 = time.time()
    g = gpu.survey(n, nplayers, team, rules, s, True, True, seed=777)
    t_gpu = time.time() - t0
    d = g[:, fast.F_OWN] - cpu[:, fast.F_OWN]
    print(f"  {nplayers}-handed, {n} deals, search {samples}: bidder avg cpu "
          f"{cpu[:, fast.F_OWN].mean():.3f} gpu {g[:, fast.F_OWN].mean():.3f}, "
          f"paired diff {d.mean():+.4f} +/- {d.std() / np.sqrt(n):.4f}   "
          f"(cpu {t_cpu:.1f}s, gpu {t_gpu:.1f}s)")
    for k in (2, 3, 4):
        print(f"      made {k}: cpu {(cpu[:, fast.F_OWN] >= k).mean():.4f}  "
              f"gpu {(g[:, fast.F_OWN] >= k).mean():.4f}")


if __name__ == "__main__":
    print("rule-based players, exact:")
    for npl in (4, 6):
        for lc in (True, False):
            check_rules_exact(npl, lc)
    print("search players, statistically on the same deals:")
    for npl in (4, 6):
        check_search_stats(npl, 30_000)
