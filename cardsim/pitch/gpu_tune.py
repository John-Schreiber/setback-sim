"""Pick the play-out kernel's register cap and block size for this GPU (gpu.configure).

k_rollout is register-bound, and the best cap depends on the card: on an RTX 3070,
capping it at 128 registers lets more threads run and speeds search up 3-15%, while
lower caps spill and slow it down. Every setting gives identical results, so each worker
may pick its own. `tune()` times a few candidates on short setback4 searches (4- and
6-handed), checks the results match, and caches the fastest per GPU model and kernel
source in ~/.cache/setback-sim/gpu-tuning.json, so it runs once per GPU and code change.

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
# (register cap or None for the compiler's choice, threads per block)
CANDIDATES = [(None, 128), (128, 64), (128, 128), (120, 128), (120, 256), (112, 128), (104, 128)]
SECONDS = 2.0  # target time per measurement at the default setting


def _key():
    """GPU model | hash of the kernel source: a new card or new kernels tune again."""
    from numba import cuda
    name = cuda.get_current_device().name
    name = name.decode() if isinstance(name, bytes) else name
    here = os.path.dirname(os.path.abspath(__file__))
    h = hashlib.sha256()
    for fn in ("gpu.py", "core.py"):
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
    """'128' or '128:64' -> (128, 64); 'default' -> (None, 128)."""
    if spec in (None, "", "default"):
        return None, 128
    regs, _, tpb = spec.partition(":")
    return (int(regs) if regs and regs != "none" else None), int(tpb or 128)


def tune(force=False, log=print):
    """Measure the candidates on this GPU (or reuse the cached result) and apply the
    fastest. Returns (max_registers, threads per block)."""
    from cardsim.pitch import PitchRules, fast, gpu
    key = _key()
    cache = _load()
    if key in cache and not force:
        regs, tpb = cache[key]["best"]
        gpu.configure(regs, tpb)
        log(f"gpu tuning: {key.split('|')[0]}: max_registers {regs}, {tpb} threads/block (cached)")
        return regs, tpb
    cases = []
    for n in (4, 6):
        g = fast.Game(PitchRules.preset("setback4"), n)
        cases.append((g, np.full(g.nteams, 16, np.int64)))
    gpu.configure(None, 128)
    sizes = []
    for g, s in cases:  # size each case to about SECONDS at the default setting
        gpu.survey(2_000, g, s, seed=11)
        t = time.time()
        gpu.survey(20_000, g, s, seed=11)
        sizes.append(max(20_000, int(20_000 * SECONDS / max(time.time() - t, 1e-3))))
    results, ref = {}, None
    for regs, tpb in CANDIDATES:
        gpu.configure(regs, tpb)
        total, rows = 0.0, []
        for (g, s), nh in zip(cases, sizes):
            gpu.survey(2_000, g, s, seed=11)  # compile outside the timing
            t = time.time()
            rows.append(gpu.survey(nh, g, s, seed=11))
            total += time.time() - t
        digest = hashlib.sha256(b"".join(r.tobytes() for r in rows)).hexdigest()
        ref = ref or digest
        if digest != ref:
            raise RuntimeError(f"gpu tuning: max_registers {regs}, {tpb} threads changed the results")
        results[f"{regs}:{tpb}"] = total
        log(f"gpu tuning: max_registers {str(regs):4} {tpb:3} threads/block: {total:.2f}s")
    best = min(results, key=results.get)
    regs, tpb = parse(best)
    base = results["None:128"]
    log(f"gpu tuning: {key.split('|')[0]}: max_registers {regs}, {tpb} threads/block, "
        f"{100 * (base / results[best] - 1):+.0f}% vs default")
    cache[key] = {"best": [regs, tpb], "seconds": results, "when": time.strftime("%Y-%m-%d %H:%M")}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=1)
    gpu.configure(regs, tpb)
    return regs, tpb


def apply(spec, log=print):
    """Worker setting: 'auto' (tune(), cached), 'default', or 'REGS[:THREADS]'."""
    if spec == "auto":
        return tune(log=log)
    from cardsim.pitch import gpu
    regs, tpb = parse(spec)
    gpu.configure(regs, tpb)
    log(f"gpu tuning: max_registers {regs}, {tpb} threads/block (set)")
    return regs, tpb


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true", help="measure again even if a result is cached")
    tune(force=ap.parse_args().force)
