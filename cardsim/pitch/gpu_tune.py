"""Pick the play-out kernel, its register cap and its block size for this GPU (gpu.configure).

Two kernels play the same games: Numba's k_rollout and the CUDA C++ one (rollout.cu,
2.3-2.6x faster on an RTX 3070). Play-outs are register-bound, and the best cap depends
on the card. Every setting gives identical results, so each worker may pick its own.
`tune()` times the candidates on short setback4 searches (4- and 6-handed), checks the
results match, and caches the fastest per GPU model and kernel source in
~/.cache/setback-sim/gpu-tuning.json, so it runs once per GPU and code change.

    .venv/Scripts/python -m cardsim.pitch.gpu_tune          # tune (or show the cached choice)
    .venv/Scripts/python -m cardsim.pitch.gpu_tune --force  # tune again
"""
import argparse
import hashlib
import json
import os
import time

import numpy as np

CACHE = os.path.join(os.path.expanduser("~"), ".cache", "setback-sim", "gpu-tuning.json")
# (engine, register cap or None for the compiler's choice, threads per block)
CANDIDATES = [("numba", None, 128), ("numba", 128, 64), ("cpp", None, 128), ("cpp", None, 64),
              ("cpp", None, 256), ("cpp", 64, 128), ("cpp", 72, 128), ("cpp", 96, 128)]
DEFAULT = ("numba", None, 128)
SECONDS = 2.0  # target time per measurement at the default setting


def _key():
    """GPU model | hash of the kernel source: a new card or new kernels tune again."""
    from numba import cuda
    name = cuda.get_current_device().name
    name = name.decode() if isinstance(name, bytes) else name
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for fn in ("gpu.py", "core.py", "rollout.cu", "gpu_cpp.py"):
        with open(os.path.join(here, fn), "rb") as f:
            h.update(f.read().replace(b"\r\n", b"\n"))
    return f"{name}|{h.hexdigest()[:12]}"


def _load():
    try:
        with open(CACHE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def parse(spec):
    """'cpp', 'cpp:64', 'cpp:64:128', or Numba's '128' / '128:64' -> (engine, regs, threads);
    'default' -> Numba with the compiler's register count, 128 threads."""
    if spec in (None, "", "default"):
        return DEFAULT
    parts = spec.split(":")
    engine = parts.pop(0) if parts[0] in ("numba", "cpp") else "numba"
    regs = parts[0] if parts else ""
    tpb = parts[1] if len(parts) > 1 else ""
    return engine, (int(regs) if regs and regs != "none" else None), int(tpb or 128)


def _label(engine, regs, tpb):
    return f"{engine}:{regs if regs else 'none'}:{tpb}"


def tune(force=False, log=print):
    """Measure the candidates on this GPU (or reuse the cached result) and apply the
    fastest. Returns (max_registers, threads per block)."""
    from cardsim.pitch import PitchRules, fast, gpu
    key = _key()
    cache = _load()
    if key in cache and not force:
        engine, regs, tpb = cache[key]["best"]
        gpu.configure(regs, tpb, engine)
        log(f"gpu tuning: {key.split('|')[0]}: {engine} play-outs, max_registers {regs}, {tpb} threads/block (cached)")
        return engine, regs, tpb
    cases = []
    for n in (4, 6):
        g = fast.Game(PitchRules.preset("setback4"), n)
        cases.append((g, np.full(g.nteams, 16, np.int64)))
    gpu.configure(*DEFAULT[1:], DEFAULT[0])
    sizes = []
    for g, s in cases:  # size each case to about SECONDS at the default setting
        gpu.survey(2_000, g, s, seed=11)
        t = time.time()
        gpu.survey(20_000, g, s, seed=11)
        sizes.append(max(20_000, int(20_000 * SECONDS / max(time.time() - t, 1e-3))))
    results, ref = {}, None
    for engine, regs, tpb in CANDIDATES:
        try:
            gpu.configure(regs, tpb, engine)
            gpu.survey(500, *cases[0], seed=11)
        except Exception as e:  # e.g. no NVRTC for this card: skip the C++ kernel
            log(f"gpu tuning: {engine} unavailable ({type(e).__name__}: {e})")
            continue
        total, rows = 0.0, []
        for (g, s), nh in zip(cases, sizes):
            gpu.survey(2_000, g, s, seed=11)  # compile outside the timing
            t = time.time()
            rows.append(gpu.survey(nh, g, s, seed=11))
            total += time.time() - t
        digest = hashlib.sha256(b"".join(r.tobytes() for r in rows)).hexdigest()
        ref = ref or digest
        if digest != ref:
            raise RuntimeError(f"gpu tuning: {engine}, max_registers {regs}, {tpb} threads changed the results")
        results[_label(engine, regs, tpb)] = total
        log(f"gpu tuning: {engine:5} max_registers {str(regs):4} {tpb:3} threads/block: {total:.2f}s")
    best = min(results, key=results.get)
    engine, regs, tpb = parse(best)
    base = results[_label(*DEFAULT)]
    log(f"gpu tuning: {key.split('|')[0]}: {engine} play-outs, max_registers {regs}, {tpb} threads/block, "
        f"{100 * (base / results[best] - 1):+.0f}% vs default")
    cache[key] = {"best": [engine, regs, tpb], "seconds": results, "when": time.strftime("%Y-%m-%d %H:%M")}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=1)
    gpu.configure(regs, tpb, engine)
    return engine, regs, tpb


def apply(spec, log=print):
    """Worker setting: 'auto' (tune(), cached), 'default', or [ENGINE:]REGS[:THREADS]."""
    if spec == "auto":
        return tune(log=log)
    from cardsim.pitch import gpu
    engine, regs, tpb = parse(spec)
    gpu.configure(regs, tpb, engine)
    log(f"gpu tuning: {engine} play-outs, max_registers {regs}, {tpb} threads/block (set)")
    return engine, regs, tpb


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="measure again even if a result is cached")
    tune(force=ap.parse_args().force)
