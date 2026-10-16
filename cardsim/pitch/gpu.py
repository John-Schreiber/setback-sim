"""CUDA engine for Pitch under any house rules: many games in lockstep on the GPU.

Same design as setback_gpu.py, on the generalized core (core.py compiled as CUDA
device functions). The CPU deals (fast.deal_batch, same seeds as the CPU engine,
including any discard and refill); then every game advances one card at a time:

  1. k_sample   one thread per (game, sample): a searching player guesses the hidden cards
  2. k_rollout  one thread per (game, sample, legal card): finish the hand with the rules.
                Only real work runs: k_count appends each searching game's legal cards to a
                work list (atomic slots, on the GPU, so the host never waits), and a fixed
                grid of threads strides over (work item, sample), the samples of one
                (game, card) side by side so neighbouring threads follow similar paths.
                Each result lands in its own vals[g, s, k], so the list order doesn't matter
  3. k_advance  one thread per game: pick the card (search total or the rules) and play it

Rule-based games match the CPU engine exactly; search matches statistically
(tests/test_pitch_gpu.py). Kernels stay short for the Windows display watchdog.
"""
import importlib.util
import math
import os

import numpy as np
from numba import cuda

from cardsim.pitch.tables import F_ESUIT, F_POWER, G_DECK

_spec = importlib.util.spec_from_file_location(
    "cardsim_pitch_core_gpu", os.path.join(os.path.dirname(os.path.abspath(__file__)), "core.py"))
core = importlib.util.module_from_spec(_spec)
core.TARGET = "cuda"
_spec.loader.exec_module(core)

bit, lowest, popcount = core.bit, core.lowest, core.popcount
options, trick_winner, note_void = core.options, core.trick_winner, core.note_void
heur_pick, rollout = core.heur_pick, core.rollout
search_value, score_into = core.search_value, core.score_into
sample_world, rng_seed = core.sample_world, core.rng_seed

MAXP, MAXT, MAXOPT, TPB = 6, 3, 9, 128  # players, teams, legal cards per decision, threads/block
ROLLOUT_WAVES = 16  # play-out grid = this many full loads of the GPU (threads stride over the rest)
ROLLOUT_TPB = TPB   # threads/block for k_rollout; configure() can change it and its register cap
ROLLOUT_REGS = None  # register cap the play-out kernel was compiled with
ENGINE = "numba"     # play-out kernel: "numba" (k_rollout) or "cpp" (rollout.cu via gpu_cpp.py)


@cuda.jit(device=True)
def kth_card(m, k):
    x = m  # k is small (a legal-card index), so a short loop beats select_bit
    for _ in range(k):
        x &= x - 1
    return lowest(x)


@cuda.jit
def k_sample(hands, gone, voids, tp, tc, tlen, leader, trumps, team, n, search, nsamp, worlds, rng,
             tab, gf):
    """One thread per (game, sample): the shared core.sample_world on stream g*nsamp + s."""
    idx = cuda.grid(1)
    g = idx // nsamp
    s = idx % nsamp
    if g >= hands.shape[0]:
        return
    p = (leader[g] + tlen) % n
    if s >= search[team[p]]:
        return
    t = trumps[g]
    hand = hands[g, p]
    if popcount(options(hand, gone[g], tc[g], tlen, t, tab, gf)) <= 1:
        return
    seen = gone[g] | hand
    for i in range(tlen):
        seen |= bit(tc[g, i])
    order = cuda.local.array(MAXP, np.int64)
    sample_world(p, hands[g], gf[G_DECK] & ~seen, voids[g], n, t, tab, worlds[g, s], order, rng, idx)


@cuda.jit
def k_count(hands, gone, tc, tlen, leader, trumps, team, n, search, work_g, work_k, nwork, tab, gf):
    """Append (game, k) for each legal card of every game whose player searches this step
    (nothing for the rules or a forced play). nwork[0] must be 0 on entry."""
    g = cuda.grid(1)
    if g >= hands.shape[0]:
        return
    p = (leader[g] + tlen) % n
    if search[team[p]] == 0:
        return
    no = popcount(options(hands[g, p], gone[g], tc[g], tlen, trumps[g], tab, gf))
    if no <= 1:
        return
    at = cuda.atomic.add(nwork, 0, no)
    for k in range(no):
        work_g[at + k] = g
        work_k[at + k] = k


@cuda.jit
def k_reset(nwork):
    nwork[0] = 0


@cuda.jit
def k_rollout(hands, gone, won, voids, tp, tc, tlen, leader, trumps, team, n, nteams, search, nsamp,
              rules, track, worlds, vals, work_g, work_k, nwork, tab, gf):
    """Grid-stride over (work item, sample); work item = (game, k-th legal card)."""
    total = nwork[0] * nsamp
    for idx in range(cuda.grid(1), total, cuda.gridsize(1)):
        rollout_one(idx // nsamp, idx % nsamp, hands, gone, won, voids, tp, tc, tlen, leader, trumps, team,
                    n, nteams, search, rules, track, worlds, vals, work_g, work_k, tab, gf)


@cuda.jit(device=True)
def rollout_one(wi, s, hands, gone, won, voids, tp, tc, tlen, leader, trumps, team, n, nteams, search,
                rules, track, worlds, vals, work_g, work_k, tab, gf):
    g = work_g[wi]
    k = work_k[wi]
    p = (leader[g] + tlen) % n
    if s >= search[team[p]]:
        return
    t = trumps[g]
    hand = hands[g, p]
    c = kth_card(options(hand, gone[g], tc[g], tlen, t, tab, gf), k)
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
    for tt in range(MAXT):
        w[tt] = won[g, tt] if tt < nteams else 0
    hs[p] = hand & ~bit(c)
    if track:
        note_void(vs, p, tc2, tlen, c, t, tab, gf)
    tp2[tlen] = p
    tc2[tlen] = c
    lead_p = tp[g, 0] if tlen else p
    last = rollout(hs, tp2, tc2, tlen + 1, lead_p, gone[g], w, vs, t, team, n, track, rules, tab, gf)
    vals[g, s, k] = search_value(w, t, team[p], last, tab, gf)


@cuda.jit
def k_advance(hands, gone, won, voids, tp, tc, tlen, leader, last_team, trumps, team, n, search,
              rules, track, vals, tab, gf):
    g = cuda.grid(1)
    if g >= hands.shape[0]:
        return
    p = (leader[g] + tlen) % n
    t = trumps[g]
    hand = hands[g, p]
    opts = options(hand, gone[g], tc[g], tlen, t, tab, gf)
    ns = search[team[p]]
    if ns > 0 and popcount(opts) > 1:
        best, bt, bp = -1, -1e300, 1 << 30
        x = opts
        k = 0
        while x:  # ascending card order, as on the CPU
            c = lowest(x)
            x &= x - 1
            tot = 0.0
            for s in range(ns):
                tot += vals[g, s, k]
            pw = tab[t, F_POWER, c]
            if tot > bt or (tot == bt and pw < bp):
                best, bt, bp = c, tot, pw
            k += 1
        c = best
    else:
        c = heur_pick(p, hands[g], tp[g], tc[g], tlen, gone[g], voids[g], t, team, n, rules, tab, gf)
    hands[g, p] = hand & ~bit(c)
    if track:
        note_void(voids[g], p, tc[g], tlen, c, t, tab, gf)
    tp[g, tlen] = p
    tc[g, tlen] = c
    if tlen + 1 == n:  # trick complete
        wp, _ = trick_winner(tp[g], tc[g], n, t, tab)
        for i in range(n):
            gone[g] |= bit(tc[g, i])
            won[g, team[wp]] |= bit(tc[g, i])
        leader[g] = wp
        last_team[g] = team[wp]


@cuda.jit
def k_score(won, trumps, lo_team, last_team, out, tab, gf):
    g = cuda.grid(1)
    if g < won.shape[0]:
        score_into(won[g], trumps[g], lo_team[g], last_team[g], out[g], tab, gf)


@cuda.jit
def k_init_states(states, seed, base):
    """Stream i for search = core.rng_seed(seed, base + i): each game's random numbers
    depend only on its global index, as on the CPU, never on how work was batched."""
    i = cuda.grid(1)
    if i < states.shape[0]:
        rng_seed(states, i, seed, base + i)


_ROLLOUT_PY = k_rollout.py_func


def configure(max_registers=None, rollout_tpb=TPB, engine="numba"):
    """Compile the play-out kernel with a register cap and launch it with this many threads
    per block. engine: "numba" (k_rollout) or "cpp" (the same play in CUDA C++, rollout.cu).
    Play-outs take ~85-90% of search time and the kernel is register-bound (134 registers:
    a quarter of an RTX 3070's threads in flight), so the best setting depends on the GPU
    (gpu_tune.py finds it). Results are identical for every setting: each (game, sample,
    card) play-out is independent and lands in its own slot."""
    global k_rollout, ROLLOUT_TPB, ROLLOUT_REGS, ENGINE
    if engine not in ("numba", "cpp"):
        raise ValueError(f"unknown play-out engine {engine!r}")
    k_rollout = cuda.jit(max_registers=max_registers)(_ROLLOUT_PY) if max_registers else cuda.jit(_ROLLOUT_PY)
    ROLLOUT_TPB, ROLLOUT_REGS, ENGINE = int(rollout_tpb), max_registers, engine


def _blocks(nthreads):
    return max(1, math.ceil(nthreads / TPB))


def play_batch(hands0, trumps, lo_team, team, rules, search, tab, gf, track=True, bidder=0, seed=0,
               max_rollout_threads=8_000_000, return_won=False, game_offset=0):
    """Play every deal in hands0 (games x players masks). rules: the rules array of the
    bidding team; search: samples per team (0 = rule-based). game_offset: global index
    of hands0[0]; search's random numbers depend only on (seed, global index), so
    results don't depend on how a study is split into batches. Returns points per team
    (and, with return_won, the cards each team captured)."""
    hands0 = np.ascontiguousarray(hands0, np.int64)
    B, n = hands0.shape
    hand_size = bin(int(hands0[0, 0])).count("1")
    team = np.asarray(team, np.int64)
    nteams = int(team.max()) + 1
    search = np.asarray(search, np.int64)
    nsamp = max(1, int(search.max()))
    chunk = max(1, min(B, max_rollout_threads // (nsamp * MAXOPT))) if search.max() else B
    # fixed grid for the strided play-outs: enough threads to fill the GPU several times
    sms = cuda.get_current_device().MULTIPROCESSOR_COUNT
    rollout_grid = sms * 2048 * ROLLOUT_WAVES // ROLLOUT_TPB
    out = np.zeros((B, nteams), np.int64)
    won_all = np.zeros((B, nteams), np.int64)
    d_team, d_rules = cuda.to_device(team), cuda.to_device(np.ascontiguousarray(rules, np.int64))
    d_search, d_tab, d_gf = cuda.to_device(search), cuda.to_device(tab), cuda.to_device(gf)
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
        d_last = cuda.to_device(np.full(G, -1, np.int64))
        if search.max():
            d_worlds = cuda.device_array((G, nsamp, n), np.int64)
            d_vals = cuda.device_array((G, nsamp, MAXOPT), np.float64)
            d_work_g = cuda.device_array(G * MAXOPT, np.int32)
            d_work_k = cuda.device_array(G * MAXOPT, np.int32)
            d_nwork = cuda.device_array(1, np.int32)
            rng = cuda.device_array((G * nsamp, 2), dtype=np.uint64)
            k_init_states[_blocks(G * nsamp), TPB](rng, seed, (game_offset + a) * nsamp)
        else:
            d_vals = cuda.device_array((1, 1, MAXOPT), np.float64)
        for step in range(hand_size * n):
            tlen = step % n
            if search.max():
                k_sample[_blocks(G * nsamp), TPB](d_hands, d_gone, d_voids, d_tp, d_tc, tlen, d_leader,
                                                  d_trumps, d_team, n, d_search, nsamp, d_worlds, rng,
                                                  d_tab, d_gf)
                k_reset[1, 1](d_nwork)
                k_count[_blocks(G), TPB](d_hands, d_gone, d_tc, tlen, d_leader, d_trumps, d_team, n,
                                         d_search, d_work_g, d_work_k, d_nwork, d_tab, d_gf)
                rargs = (d_hands, d_gone, d_won, d_voids, d_tp, d_tc, tlen, d_leader, d_trumps, d_team, n, nteams,
                         d_search, nsamp, d_rules, track, d_worlds, d_vals, d_work_g, d_work_k, d_nwork, d_tab, d_gf)
                if ENGINE == "cpp":
                    from cardsim.pitch import gpu_cpp
                    gpu_cpp.launch(rollout_grid, ROLLOUT_TPB, ROLLOUT_REGS, *rargs)
                else:
                    k_rollout[rollout_grid, ROLLOUT_TPB](*rargs)
            k_advance[_blocks(G), TPB](d_hands, d_gone, d_won, d_voids, d_tp, d_tc, tlen, d_leader, d_last,
                                       d_trumps, d_team, n, d_search, d_rules, track, d_vals, d_tab, d_gf)
        d_out = cuda.device_array((G, nteams), np.int64)
        d_lo = cuda.to_device(np.ascontiguousarray(lo_team[a:b], np.int64))
        k_score[_blocks(G), TPB](d_won, d_trumps, d_lo, d_last, d_out, d_tab, d_gf)
        out[a:b] = d_out.copy_to_host()
        if return_won:
            won_all[a:b] = d_won.copy_to_host()
    return (out, won_all) if return_won else out


def survey(nhands, game, search, seed=1, batch=200_000, start=0):
    """GPU version of fast.survey for a fast.Game: same deals for the same seed, same
    columns (fast.S_*). start: global index of the first deal (seed + index picks each
    deal, so a study can be split into pieces that reproduce the whole)."""
    from cardsim.pitch import fast
    r = game.rules
    out = np.zeros((nhands, 6), np.int64)
    for a in range(0, nhands, batch):
        cnt = min(batch, nhands - a)
        hands, trumps, lo_team, dealt0 = fast.deal_batch(start + a, cnt, game.nplayers, r.deal_size, r.hand_size, r.refill,
                                                 game.team, game.tab, game.gf, game.tw, game.tl, game.tx, seed)
        sc, won = play_batch(hands, trumps, lo_team, game.team, game.rules_bt[game.team[0]], search, game.tab,
                             game.gf, seed=seed, return_won=True, game_offset=start + a)
        own = sc[:, game.team[0]]
        others = np.delete(won, game.team[0], axis=1)
        sweep = (others == 0).all(axis=1).astype(np.int64)
        out[a:a + cnt] = np.stack([trumps, hands[:, 0], own, sc.sum(axis=1) - own, dealt0, sweep], axis=1)
    return out
