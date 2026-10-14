"""Chance each hand makes an 11-bid: all 4 points AND all 6 tricks (4-handed).

Every distinct hand (allhands.py) is played as bidder against the same random
layouts the every-hand tables use, by rule-based players who play for tricks:
both sides trump in whenever they can (an 11-bid falls to a single lost trick).
The Jack must be in play to make all 4 points. Saves sweep_4p_d<deals>.npz with
the chance per hand class, in allhands key order.

    .venv/Scripts/python sweep.py --deals 224
"""
import argparse
import os
import time

import numpy as np
from numba import njit, prange

import allhands as ah
import setback as sb
import setback_fast as fast
from engine_core import R_RUFF_MIN


def rules_for_11(nplayers, team, bidder=0):
    """Default rules, but everyone plays for tricks."""
    r = np.stack([fast.rules_array(sb.team_rules(nplayers, list(team), bidder=b))
                  for b in range(int(team.max()) + 1)])
    r[:, :, R_RUFF_MIN] = 0
    return r


@njit(parallel=True, cache=True)
def sweeps(dealt, trumps, team, rules):
    """dealt[g]: hands with the bidder in seat 0. True where seat 0's team took all
    4 points and every trick."""
    nteams = team.max() + 1
    zero = np.zeros(nteams, np.int64)
    out = np.zeros(dealt.shape[0], np.bool_)
    for g in prange(dealt.shape[0]):
        sc, won = fast.play_game_won(dealt[g], 0, trumps[g], team, nteams, True, True, rules, zero)
        ok = sc[team[0]] == 4
        for t in range(nteams):
            if t != team[0] and won[t] != 0:
                ok = False
        out[g] = ok
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deals", type=int, default=224)
    ap.add_argument("--seed", type=int, default=20261013)  # same layouts as allhands.py
    ap.add_argument("--chunk", type=int, default=8192)
    a = ap.parse_args()
    team = np.array(sb.make_teams(4, 2), np.int64)
    rules = rules_for_11(4, team)[team[0]]  # bidder is seat 0
    keys, counts = ah.hand_classes()
    trumps = ah.best_suits(keys, 4)
    p = np.zeros(len(keys), np.float32)
    t0 = time.time()
    for s in range(0, len(keys), a.chunk):
        e = min(len(keys), s + a.chunk)
        dealt, _ = ah.deal_many(keys[s:e], trumps[s:e], a.deals, 4, a.seed, s)
        ok = sweeps(dealt, np.repeat(trumps[s:e], a.deals), team, rules)
        p[s:e] = ok.reshape(e - s, a.deals).mean(axis=1)
        if (s // a.chunk) % 20 == 0:
            print(f"  {e:,}/{len(keys):,} hands  {time.time() - t0:5.0f}s", flush=True)
    fn = os.path.join(ah.HERE, f"sweep_4p_d{a.deals}.npz")
    np.savez(fn, p=p, keys=keys, counts=counts)
    w = counts / counts.sum()
    print(f"wrote {fn}: chance of a sweep averaged over all hands {np.sum(w * p):.3%}; "
          f"hands with >= 50% chance: {w[p >= 0.5].sum():.3%}", flush=True)


if __name__ == "__main__":
    main()
