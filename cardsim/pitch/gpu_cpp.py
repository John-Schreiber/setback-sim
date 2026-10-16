"""The C++ play-out kernel (rollout.cu): compiled with NVRTC for this GPU, launched on the
stream Numba uses, as a drop-in for gpu.k_rollout (gpu.configure(engine="cpp")).

The kernel's constants (table fields, flags, rule columns, sizes) are generated from
tables.py, core.py and gpu.py, so the C++ can't drift from the Python layout.
"""
import os

import numpy as np

from cardsim.pitch import core, tables

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rollout.cu")
_cache = {}


def _header():
    from cardsim.pitch import gpu
    names = {}
    for mod in (tables, core):
        for k, v in vars(mod).items():
            if k[:2] in ("F_", "M_", "G_", "R_") and isinstance(v, int) and k.isupper():
                names[k] = v
    names.update(N_FIELDS=tables.N_FIELDS, N_CARDS=54, N_RULES=core.N_RULES, MAXP=gpu.MAXP, MAXT=gpu.MAXT,
                 MAXOPT=gpu.MAXOPT)
    return "".join(f"#define {k} {v}\n" for k, v in sorted(names.items()))


def kernel(max_registers=None):
    """The compiled k_rollout_cpp for the current device (cached per register cap)."""
    from cuda.core import Device, Program, ProgramOptions
    dev = Device()
    dev.set_current()
    key = (dev.device_id, max_registers)
    if key not in _cache:
        with open(_SRC, encoding="utf-8") as f:
            src = _header() + f.read()
        name = "rollout.cu"
        dump = os.environ.get("CARDSIM_CPP_SOURCE")  # profiling: write the source there, so the
        if dump:                                      # line info points at a real file
            with open(dump, "w", encoding="utf-8") as f:
                f.write(src)
            name = dump
        cc = dev.compute_capability
        opts = ProgramOptions(name=name, arch=f"sm_{cc.major}{cc.minor}", std="c++17", lineinfo=True,
                              max_register_count=max_registers)
        mod = Program(src, code_type="c++", options=opts).compile("cubin")
        _cache[key] = mod.get_kernel("k_rollout_cpp")
    return _cache[key]


def _ptr(a):
    return np.uintp(a.__cuda_array_interface__["data"][0])


def launch(grid, block, max_registers, hands, gone, won, voids, tp, tc, tlen, leader, trumps, team, n, nteams,
           search, nsamp, rules, track, worlds, vals, work_g, work_k, nwork, tab, gf):
    """Same arguments as gpu.k_rollout (Numba device arrays)."""
    from cuda.core import LEGACY_DEFAULT_STREAM, LaunchConfig, launch as cu_launch
    k = kernel(max_registers)
    cu_launch(LEGACY_DEFAULT_STREAM, LaunchConfig(grid=int(grid), block=int(block)), k,
              _ptr(hands), _ptr(gone), _ptr(won), _ptr(voids), _ptr(tp), _ptr(tc), np.int32(tlen), _ptr(leader),
              _ptr(trumps), _ptr(team), np.int32(n), np.int32(nteams), _ptr(search), np.int32(nsamp), _ptr(rules),
              np.int32(bool(track)), _ptr(worlds), _ptr(vals), _ptr(work_g), _ptr(work_k), _ptr(nwork), _ptr(tab),
              _ptr(gf))
