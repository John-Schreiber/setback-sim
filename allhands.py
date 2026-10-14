"""Every distinct 6-card hand, played out with search players on the GPU.

Hands that differ only by relabeling suits play the same, so all C(52,6) =
20,358,520 hands reduce to 962,988 classes. For each class the bidder (seat 0)
names their best suit and the hand is played against `--deals` random layouts
of the other cards. Per hand we store the bidder team's average points, the
chance of making 2/3/4 and the expected points of each bid.

Results are saved in chunks under allhands_<P>p_d<deals>_s<search>/ so an
interrupted run resumes where it stopped (rerun the same command); when every
chunk is done they are merged into allhands_<P>p_d<deals>_s<search>.csv.gz.

    .venv/Scripts/python allhands.py -p 4 --deals 256
"""
import argparse
import glob
import os
import time

import numpy as np
import pandas as pd
from numba import njit, prange

import setback as sb
import setback_fast as fast
import setback_gpu as gpu
from engine_core import SM, bit, lowest, popcount, pick_trump

HERE = os.path.dirname(os.path.abspath(__file__))


@njit(cache=True)
def canonical(hand):
    """Sort the four 13-card suit masks, largest first, and repack: one key per class."""
    m = np.empty(4, np.int64)
    for s in range(4):
        m[s] = (hand >> (13 * s)) & 0x1FFF
    m.sort()
    return m[3] | (m[2] << 13) | (m[1] << 26) | (m[0] << 39)


@njit(cache=True)
def all_canonical():
    """Canonical key of every one of the C(52,6) hands."""
    out = np.empty(20358520, np.int64)
    k = 0
    for a in range(52):
        for b in range(a + 1, 52):
            for c in range(b + 1, 52):
                for d in range(c + 1, 52):
                    for e in range(d + 1, 52):
                        for f in range(e + 1, 52):
                            out[k] = canonical(bit(a) | bit(b) | bit(c) | bit(d) | bit(e) | bit(f))
                            k += 1
    return out


def hand_classes():
    """(canonical hand masks, how many of the 20,358,520 hands fall in each class)."""
    keys, counts = np.unique(all_canonical(), return_counts=True)
    return keys, counts


@njit(parallel=True, cache=True)
def deal_many(hands, trumps, deals, nplayers, seed, first):
    """`deals` random layouts for each hand; hand i (global index first + i) uses
    seeds seed + (first + i) * deals + j, so every layout is reproducible."""
    nh = hands.shape[0]
    out = np.zeros((nh * deals, nplayers), np.int64)
    lo_owner = np.zeros(nh * deals, np.int64)
    for g in prange(nh * deals):
        i = g // deals
        j = g % deals
        hand = hands[i]
        rest = np.empty(46, np.int64)
        k = 0
        for d in range(52):
            if not (hand >> d) & 1:
                rest[k] = d
                k += 1
        np.random.seed(seed + (first + i) * deals + j)
        perm = np.random.permutation(46)
        out[g, 0] = hand
        for q in range(1, nplayers):
            for t in range(6):
                out[g, q] |= bit(rest[perm[(q - 1) * 6 + t]])
        trump = trumps[i]
        lo_c = 99
        for q in range(nplayers):
            tm = out[g, q] & SM[trump]
            if tm and lowest(tm) < lo_c:
                lo_c, lo_owner[g] = lowest(tm), q
    return out, lo_owner


@njit(parallel=True, cache=True)
def best_suits(hands, nplayers):
    out = np.zeros(hands.shape[0], np.int64)
    for i in prange(hands.shape[0]):
        out[i] = pick_trump(hands[i], nplayers)
    return out


def hand_str(mask):
    return " ".join("23456789TJQKA"[c % 13] + "SHDC"[c // 13] for c in range(52) if (mask >> c) & 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-p", "--players", type=int, default=4, choices=[4, 6])
    ap.add_argument("--deals", type=int, default=256, help="random layouts per hand")
    ap.add_argument("--search", type=int, default=16, help="sampled deals per search decision")
    ap.add_argument("--chunk", type=int, default=2048, help="hands per saved chunk")
    ap.add_argument("--limit", type=int, default=0, help="only the first N hand classes (testing)")
    ap.add_argument("--seed", type=int, default=20261013)
    a = ap.parse_args()

    team = np.array(sb.make_teams(a.players, 2), np.int64)
    nteams = int(team.max()) + 1
    rules = fast.rules_array(sb.team_rules(a.players, list(team)))
    search = np.full(nteams, a.search, np.int64)
    tag = f"allhands_{a.players}p_d{a.deals}_s{a.search}"  # settings in the name: chunks never mix
    outdir = os.path.join(HERE, tag)
    os.makedirs(outdir, exist_ok=True)

    keys, counts = hand_classes()
    assert len(keys) == 962988 and counts.sum() == 20358520, (len(keys), counts.sum())
    if a.limit:
        keys, counts = keys[:a.limit], counts[:a.limit]
    trumps = best_suits(keys, a.players)
    starts = list(range(0, len(keys), a.chunk))
    todo = [s for s in starts if not os.path.exists(os.path.join(outdir, f"chunk_{s:07d}.npz"))]
    print(f"{a.players}-handed: {len(keys):,} hand classes x {a.deals} deals, search {a.search}; "
          f"{len(starts) - len(todo)}/{len(starts)} chunks already done", flush=True)

    t0 = time.time()
    for n_done, s in enumerate(todo, 1):
        e = min(len(keys), s + a.chunk)
        hands, lo_owner = deal_many(keys[s:e], trumps[s:e], a.deals, a.players, a.seed, s)
        tr = np.repeat(trumps[s:e], a.deals)
        sc = gpu.play_batch(hands, tr, team[lo_owner], team, rules, search, seed=a.seed + s)
        own = sc[:, team[0]].reshape(e - s, a.deals)
        stats = {"mean": own.mean(axis=1)}
        for k in (2, 3, 4):
            stats[f"p{k}"] = (own >= k).mean(axis=1)
            stats[f"ev{k}"] = np.where(own >= k, own, -k).mean(axis=1)
        tmp = os.path.join(outdir, f"chunk_{s:07d}.tmp.npz")
        np.savez(tmp, **{k: v.astype(np.float32) for k, v in stats.items()})
        os.replace(tmp, os.path.join(outdir, f"chunk_{s:07d}.npz"))
        el = time.time() - t0
        print(f"  chunk {s:>7,}-{e:>7,}  done {n_done}/{len(todo)}  "
              f"elapsed {el / 60:6.1f} min  eta {el / n_done * (len(todo) - n_done) / 60:6.1f} min",
              flush=True)

    # merge
    parts = {}
    for s in starts:
        z = np.load(os.path.join(outdir, f"chunk_{s:07d}.npz"))
        for k in z.files:
            parts.setdefault(k, []).append(z[k])
    df = pd.DataFrame({k: np.concatenate(v) for k, v in parts.items()})
    df.insert(0, "hand", [hand_str(int(h)) for h in keys])
    df.insert(1, "trump", ["SHDC"[t] for t in trumps])
    df.insert(2, "weight", counts)  # how many real hands this class stands for
    df.insert(3, "deals", a.deals)
    fn = os.path.join(HERE, f"{tag}{f'_first{a.limit}' if a.limit else ''}.csv.gz")
    df.to_csv(fn, index=False, float_format="%.4f")
    print(f"wrote {fn}  ({len(df):,} hands)", flush=True)


if __name__ == "__main__":
    main()
