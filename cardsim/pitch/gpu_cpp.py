"""The CUDA C++ kernels (rollout.cu): compiled with NVRTC for this GPU and launched on the
stream Numba uses, in place of gpu.py's Numba kernels when gpu.configure(engine="cpp").

The kernels' constants (table fields, flags, rule columns, sizes) are generated from
tables.py, core.py and gpu.py, so the C++ can't drift from the Python layout. Arguments
are Numba device arrays (passed as pointers) and Python ints.
"""
import os

import numpy as np

from cardsim.pitch import core, tables

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rollout.cu")
KERNELS = ("k_rollout_cpp", "k_sample_cpp", "k_count_cpp", "k_advance_cpp", "k_init_states_cpp")
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


def kernels(max_registers=None):
    """{name: compiled kernel} for the current device (cached per register cap)."""
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
        _cache[key] = {k: mod.get_kernel(k) for k in KERNELS}
    return _cache[key]


def kernel(max_registers=None):
    """The play-out kernel (for inspection: registers etc.)."""
    return kernels(max_registers)["k_rollout_cpp"]


def _arg(a):
    """Device array -> pointer; bool/int -> int32 (int64 for the 64-bit parameters below)."""
    if hasattr(a, "__cuda_array_interface__"):
        return np.uintp(a.__cuda_array_interface__["data"][0])
    if isinstance(a, np.integer) and a.dtype.itemsize == 8:
        return a
    return np.int32(int(a))


def _launch(name, grid, block, max_registers, *args):
    from cuda.core import LEGACY_DEFAULT_STREAM, LaunchConfig, launch
    launch(LEGACY_DEFAULT_STREAM, LaunchConfig(grid=int(grid), block=int(block)),
           kernels(max_registers)[name], *(_arg(a) for a in args))


def slot(counter, i):
    """Pointer to element i of an int32 device array (a step's work-counter slot)."""
    return np.uintp(counter.__cuda_array_interface__["data"][0] + 4 * i)


def rollout(grid, block, max_registers, hands, gone, won, voids, tp, tc, tlen, leader, trumps, team, n, nteams,
            search, nsamp, rules, track, worlds, vals, work_g, work_k, nwork, tab, gf):
    """Same arguments as gpu.k_rollout; nwork may be a slot() pointer."""
    _launch("k_rollout_cpp", grid, block, max_registers, hands, gone, won, voids, tp, tc, tlen, leader, trumps, team,
            n, nteams, search, nsamp, rules, track, worlds, vals, work_g, work_k, nwork, tab, gf)


def init_states(grid, block, max_registers, states, count, seed, base):
    _launch("k_init_states_cpp", grid, block, max_registers, states, np.int64(count), np.int64(seed), np.int64(base))


def sample(grid, block, max_registers, hands, gone, voids, tc, tlen, leader, trumps, team, n, search, nsamp, worlds,
           rng, ngames, tab, gf):
    _launch("k_sample_cpp", grid, block, max_registers, hands, gone, voids, tc, tlen, leader, trumps, team, n, search,
            nsamp, worlds, rng, ngames, tab, gf)


def count(grid, block, max_registers, hands, gone, tc, tlen, leader, trumps, team, n, search, work_g, work_k, nwork,
          ngames, tab, gf):
    _launch("k_count_cpp", grid, block, max_registers, hands, gone, tc, tlen, leader, trumps, team, n, search, work_g,
            work_k, nwork, ngames, tab, gf)


def advance(grid, block, max_registers, hands, gone, won, voids, tp, tc, tlen, leader, last_team, trumps, team, n,
            nteams, search, rules, track, vals, nsamp, ngames, tab, gf):
    _launch("k_advance_cpp", grid, block, max_registers, hands, gone, won, voids, tp, tc, tlen, leader, last_team,
            trumps, team, n, nteams, search, rules, track, vals, nsamp, ngames, tab, gf)
