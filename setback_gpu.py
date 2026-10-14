"""CUDA engine for setback.py: many games in lockstep on the GPU.

The CPU deals (setback_fast.deal_batch, same seeds as the CPU engine, so both
engines play identical deals). All games then advance one card at a time
together; every hand has exactly 6 * players cards played. For each card:

  1. k_sample   one thread per (game, sample): a searching player guesses the
                hidden cards (honoring known voids), using a GPU random stream
  2. k_rollout  one thread per (game, sample, legal card): play the rest of the
                hand out with the rule-based players, record the result
  3. k_advance  one thread per game: pick the card (best summed search value, or
                the rules), play it, and finish the trick if it's complete

The game logic is engine_core.py compiled as CUDA device functions, so it is
the same code the CPU engine runs. Rule-based games match the CPU engine
exactly; search uses different random numbers, so it matches statistically
(see test_gpu.py).

Kernels stay short on purpose: Windows kills GPU kernels that run for more than
about 2 seconds while a display is attached.
"""
import importlib.util
import math
import os

import numpy as np
from numba import cuda
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float64

# engine_core.py again, compiled for the GPU this time
_spec = importlib.util.spec_from_file_location(
    "engine_core_gpu", os.path.join(os.path.dirname(os.path.abspath(__file__)), "engine_core.py"))
core = importlib.util.module_from_spec(_spec)
core.TARGET = "cuda"
_spec.loader.exec_module(core)

SM = core.SM
bit, rank, suit = core.bit, core.rank, core.suit
lowest, popcount, legal = core.lowest, core.popcount, core.legal
trick_winner, note_void = core.trick_winner, core.note_void
heur_pick, rollout = core.heur_pick, core.rollout
search_value, score_into = core.search_value, core.score_into

MAXP = 6    # players
MAXT = 3    # teams
MAXOPT = 6  # legal cards in one decision
TPB = 128   # threads per block


@cuda.jit(device=True)
def options(hand, gone, tc, tlen, trump):
    if gone == 0 and tlen == 0:
        return hand & SM[trump]  # pitch: any trump
    return legal(hand, suit(tc[0]) if tlen else -1, trump)


@cuda.jit(device=True)
def kth_card(m, k):
    x = m
    for _ in range(k):
        x &= x - 1
    return lowest(x)


@cuda.jit
def k_sample(hands, gone, voids, tp, tc, tlen, leader, trumps, team, n, search, nsamp,
             worlds, rng):
    """A searching player guesses where the cards it can't see are."""
    idx = cuda.grid(1)
    g = idx // nsamp
    s = idx % nsamp
    if g >= hands.shape[0]:
        return
    p = (leader[g] + tlen) % n
    if s >= search[team[p]]:
        return
    trump = trumps[g]
    hand = hands[g, p]
    if popcount(options(hand, gone[g], tc[g], tlen, trump)) <= 1:
        return
    seen = gone[g] | hand
    for i in range(tlen):
        seen |= bit(tc[g, i])
    unseen = cuda.local.array(52, np.int64)
    nun = 0
    for d in range(52):
        if not (seen >> d) & 1:
            unseen[nun] = d
            nun += 1
    order = cuda.local.array(MAXP, np.int64)
    k = 0
    for nv in range(4, -1, -1):  # most constrained first
        for q in range(n):
            if q != p and popcount(voids[g, q]) == nv:
                order[k] = q
                k += 1
    perm = cuda.local.array(52, np.int64)
    used = cuda.local.array(52, np.int8)
    worlds[g, s, p] = hand
    for attempt in range(31):
        for i in range(nun):  # Fisher-Yates shuffle
            perm[i] = i
            used[i] = 0
        for i in range(nun - 1, 0, -1):
            j = int(xoroshiro128p_uniform_float64(rng, idx) * (i + 1))
            if j > i:
                j = i
            perm[i], perm[j] = perm[j], perm[i]
        ok = True
        for jj in range(n - 1):
            q = order[jj]
            need = popcount(hands[g, q])
            got = 0
            m = np.int64(0)
            for i in range(nun):
                if got == need:
                    break
                if used[perm[i]]:
                    continue
                d = unseen[perm[i]]
                if attempt < 30 and ((voids[g, q] >> suit(d)) & 1):
                    continue  # last attempt ignores voids (shouldn't be needed)
                m |= bit(d)
                used[perm[i]] = 1
                got += 1
            worlds[g, s, q] = m
            if got < need:
                ok = False
                break
        if ok:
            return


@cuda.jit
def k_rollout(hands, gone, won, voids, tp, tc, tlen, leader, trumps, team, n, nteams, search,
              nsamp, rules, track, worlds, vals):
    """Play one legal card on one guessed deal, then finish the hand with the rules."""
    idx = cuda.grid(1)
    g = idx // (nsamp * MAXOPT)
    r = idx % (nsamp * MAXOPT)
    s = r // MAXOPT
    k = r % MAXOPT
    if g >= hands.shape[0]:
        return
    p = (leader[g] + tlen) % n
    if s >= search[team[p]]:
        return
    trump = trumps[g]
    hand = hands[g, p]
    opts = options(hand, gone[g], tc[g], tlen, trump)
    if popcount(opts) <= 1 or k >= popcount(opts):
        return
    c = kth_card(opts, k)
    hs = cuda.local.array(MAXP, np.int64)
    vs = cuda.local.array(MAXP, np.int64)
    tp2 = cuda.local.array(MAXP, np.int64)
    tc2 = cuda.local.array(MAXP, np.int64)
    w = cuda.local.array(MAXT, np.int64)
    for q in range(MAXP):
        hs[q] = worlds[g, s, q] if q < n else 0
        vs[q] = voids[g, q] if q < n else 0
        tp2[q] = tp[g, q] if q < n else 0
        tc2[q] = tc[g, q] if q < n else 0
    for t in range(MAXT):
        w[t] = won[g, t] if t < nteams else 0
    hs[p] = hand & ~bit(c)
    if track:
        note_void(vs, p, tc2, tlen, c, trump)
    tp2[tlen] = p
    tc2[tlen] = c
    lead_p = tp[g, 0] if tlen else p
    rollout(hs, tp2, tc2, tlen + 1, lead_p, gone[g], w, vs, trump, team, n, True, track, rules)
    vals[g, s, k] = search_value(w, trump, team[p])


@cuda.jit
def k_advance(hands, gone, won, voids, tp, tc, tlen, leader, trumps, team, n, search, rules,
              lowcap, track, vals):
    """Each game plays its next card."""
    g = cuda.grid(1)
    if g >= hands.shape[0]:
        return
    p = (leader[g] + tlen) % n
    trump = trumps[g]
    hand = hands[g, p]
    opts = options(hand, gone[g], tc[g], tlen, trump)
    ns = search[team[p]]
    if ns > 0 and popcount(opts) > 1:
        best, bt, br = -1, -1e300, 99
        x = opts
        k = 0
        while x:  # ascending card order, as on the CPU
            c = lowest(x)
            x &= x - 1
            tot = 0.0
            for s in range(ns):
                tot += vals[g, s, k]
            if tot > bt or (tot == bt and rank(c) < br):
                best, bt, br = c, tot, rank(c)
            k += 1
        c = best
    else:
        c = heur_pick(p, hands[g], tp[g], tc[g], tlen, gone[g], voids[g], trump, team, n, lowcap,
                      rules)
    hands[g, p] = hand & ~bit(c)
    if track:
        note_void(voids[g], p, tc[g], tlen, c, trump)
    tp[g, tlen] = p
    tc[g, tlen] = c
    if tlen + 1 == n:  # trick complete
        wp, _ = trick_winner(tp[g], tc[g], n, trump)
        for i in range(n):
            gone[g] |= bit(tc[g, i])
            won[g, team[wp]] |= bit(tc[g, i])
        leader[g] = wp


@cuda.jit
def k_score(won, trumps, lowcap, lo_team, out):
    g = cuda.grid(1)
    if g < won.shape[0]:
        score_into(won[g], trumps[g], lowcap, lo_team[g], out[g])


def _blocks(nthreads):
    return max(1, math.ceil(nthreads / TPB))


def play_batch(hands0, trumps, lo_team, team, rules, search, lowcap=True, track=True, bidder=0,
               seed=0, max_rollout_threads=4_000_000):
    """Play every deal in hands0 (games x players masks) on the GPU.

    team: team of each player; rules: setback_fast.rules_array(...); search: samples
    per team (0 = rule-based); lo_team: team dealt the lowest trump (low-dealt rule).
    Returns points per team for each game."""
    hands0 = np.ascontiguousarray(hands0, np.int64)
    B, n = hands0.shape
    team = np.asarray(team, np.int64)
    nteams = int(team.max()) + 1
    search = np.asarray(search, np.int64)
    nsamp = max(1, int(search.max()))
    # keep each rollout launch small enough for the Windows display watchdog
    chunk = max(1, min(B, max_rollout_threads // (nsamp * MAXOPT))) if search.max() else B
    out = np.zeros((B, nteams), np.int64)
    d_team = cuda.to_device(team)
    d_rules = cuda.to_device(np.ascontiguousarray(rules, np.int64))
    d_search = cuda.to_device(search)
    for a in range(0, B, chunk):
        b = min(B, a + chunk)
        G = b - a
        d_hands = cuda.to_device(hands0[a:b].copy())
        d_trumps = cuda.to_device(np.ascontiguousarray(trumps[a:b], np.int64))
        d_gone = cuda.to_device(np.zeros(G, np.int64))
        d_won = cuda.to_device(np.zeros((G, nteams), np.int64))
        d_voids = cuda.to_device(np.zeros((G, n), np.int64))
        d_tp = cuda.to_device(np.zeros((G, n), np.int64))
        d_tc = cuda.to_device(np.zeros((G, n), np.int64))
        d_leader = cuda.to_device(np.full(G, bidder, np.int64))
        if search.max():
            d_worlds = cuda.device_array((G, nsamp, n), np.int64)
            d_vals = cuda.device_array((G, nsamp, MAXOPT), np.float64)
            rng = create_xoroshiro128p_states(G * nsamp, seed=seed * 1_000_003 + a)
        else:
            d_vals = cuda.device_array((1, 1, MAXOPT), np.float64)
        for step in range(6 * n):
            tlen = step % n
            if search.max():
                k_sample[_blocks(G * nsamp), TPB](
                    d_hands, d_gone, d_voids, d_tp, d_tc, tlen, d_leader, d_trumps, d_team, n,
                    d_search, nsamp, d_worlds, rng)
                k_rollout[_blocks(G * nsamp * MAXOPT), TPB](
                    d_hands, d_gone, d_won, d_voids, d_tp, d_tc, tlen, d_leader, d_trumps, d_team,
                    n, nteams, d_search, nsamp, d_rules, track, d_worlds, d_vals)
            k_advance[_blocks(G), TPB](
                d_hands, d_gone, d_won, d_voids, d_tp, d_tc, tlen, d_leader, d_trumps, d_team, n,
                d_search, d_rules, lowcap, track, d_vals)
        d_out = cuda.device_array((G, nteams), np.int64)
        d_lo = cuda.to_device(np.ascontiguousarray(lo_team[a:b], np.int64))
        k_score[_blocks(G), TPB](d_won, d_trumps, lowcap, d_lo, d_out)
        out[a:b] = d_out.copy_to_host()
    return out


def survey(nhands, nplayers, team, rules, search, lowcap=True, track=True, seed=1,
           batch=200_000):
    """GPU version of setback_fast.survey: same deals for the same seed, same output
    columns (features, bidder-team points, everyone else's points)."""
    import setback_fast as fast
    team = np.asarray(team, np.int64)
    out = np.zeros((nhands, 7), np.int64)
    for a in range(0, nhands, batch):
        cnt = min(batch, nhands - a)
        hands, trumps, lo_owner = fast.deal_batch(a, cnt, nplayers, seed)
        sc = play_batch(hands, trumps, team[lo_owner], team, rules, search, lowcap, track,
                        seed=seed + a)
        out[a:a + cnt, :5] = fast.features_batch(hands, trumps)
        out[a:a + cnt, fast.F_OWN] = sc[:, team[0]]
        out[a:a + cnt, fast.F_OPP] = sc.sum(axis=1) - sc[:, team[0]]
    return out


def sim_hand(hand, trump, nhands, nplayers, team, rules, search, lowcap=True, track=True, seed=1):
    """GPU version of setback_fast.sim_hand (same deals for the same seed).
    Returns (own, opp) points per deal."""
    import setback_fast as fast
    team = np.asarray(team, np.int64)
    hands, lo_owner = fast.deal_hand_batch(np.int64(hand), trump, nhands, nplayers, seed)
    sc = play_batch(hands, np.full(nhands, trump, np.int64), team[lo_owner], team, rules, search,
                    lowcap, track, seed=seed)
    own = sc[:, team[0]]
    return np.stack([own, sc.sum(axis=1) - own], axis=1)
