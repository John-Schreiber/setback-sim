"""Engine speed and exactness, for comparing optimizations.

Runs fixed surveys (same seeds) on CPU and GPU (search on both GPU play-out kernels,
Numba's and the C++ one) and prints deals per second plus a checksum of every result row. An optimization must leave the checksums unchanged
(the games are identical) and should raise the speeds. Pass a label to tell runs apart.

    .venv/Scripts/python tests/bench_perf.py baseline
"""
import hashlib
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from cardsim.pitch import PitchRules, fast, gpu  # noqa: E402

CASES = [("setback4", 4, 0, 400_000), ("setback4", 4, 16, 4_000), ("setback4", 6, 16, 2_000),
         ("ten_point", 4, 16, 4_000), ("nine_five", 4, 16, 2_000)]
GPU_SCALE = 50  # GPU runs need ~200k games to fill the card; smaller batches understate its speed


def run(label):
    print(f"[{label}]")
    for preset, n, s, nh in CASES:
        g = fast.Game(PitchRules.preset(preset), n)
        r = g.rules
        srch = np.full(g.nteams, s, np.int64)
        args = (n, r.deal_size, r.hand_size, r.refill, g.team, g.nteams, True, g.rules_bt, srch, g.tab, g.gf,
                g.tw, g.tl, g.tx, 7)
        fast.survey(200, *args)  # compile
        t = time.time()
        cpu = fast.survey(nh, *args)
        tc = time.time() - t
        ng = nh * (GPU_SCALE if s else 5)
        speeds, digests = [], set()
        for engine in (("numba", "cpp") if s else ("numba",)):  # play-out kernels matter only for search
            gpu.configure(engine=engine)
            gpu.survey(200, g, srch, seed=7)
            t = time.time()
            gp = gpu.survey(ng, g, srch, seed=7)
            speeds.append(f"{engine} {ng / (time.time() - t):9,.0f}/s")
            assert np.array_equal(cpu, gp[:nh]), f"CPU and GPU ({engine}) differ"
            digests.add(hashlib.sha256(gp.tobytes()).hexdigest()[:12])
        gpu.configure()
        assert len(digests) == 1, "the GPU play-out kernels differ"
        print(f"  {preset:10} {n}p search {s:2}: CPU {nh / tc:9,.0f}/s   GPU {'   '.join(speeds)}   rows {digests.pop()}")


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "run")
