"""Distributed studies must reproduce single-machine runs exactly.

Starts a coordinator in-process, runs several workers with tiny slices (so each study
is split many ways), plus a rogue worker that leases a slice and vanishes; its slice
must time out and be finished by someone else. The merged results are compared with
running each whole study in one go. Also checks that GPU search surveys don't depend
on how they are batched.

    .venv/Scripts/python tests/test_dist.py
"""
import io
import os
import socket
import sys
import tempfile
import threading
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import uvicorn  # noqa: E402

from cardsim.dist import tasks  # noqa: E402
from cardsim.dist.coordinator import create_app  # noqa: E402
from cardsim.dist.worker import Client, work  # noqa: E402


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main():
    data = tempfile.mkdtemp(prefix="dist-test-")
    port = free_port()
    app = create_app(data, lease_timeout=3.0, token="secret")
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    while not server.started:
        time.sleep(0.05)
    url = f"http://127.0.0.1:{port}"
    c = Client(url, "secret")

    survey = tasks.study_params("survey", "ten_point", 4, 0, 5, hands=30000)
    allh = tasks.study_params("allhands", "setback4", 4, 0, 9, deals=32, limit=3000)
    s1 = c.post("/studies", {"kind": "survey", "params": survey, "engine": "cpu", "calib": 1000})["id"]
    s2 = c.post("/studies", {"kind": "allhands", "params": allh, "engine": "cpu", "calib": 100})["id"]

    # a rogue worker leases one slice and disappears
    rogue = c.post("/lease", {"worker": "rogue", "devices": ["cpu"], "rates": {}, "target_seconds": 1})
    assert rogue, "rogue got no slice"

    quiet = []
    threads = [threading.Thread(target=work, kwargs=dict(url=url, token="secret", device="cpu", target_minutes=0.02,
                                                         exit_when_idle=True, name=f"w{i}", idle_sleep=1.0,
                                                         heartbeat_every=0.5, log=quiet.append))
               for i in range(3)]
    for t in threads:
        t.start()
    # workers exit when idle; the rogue's slice reappears after the lease timeout, so
    # keep a worker around until both studies are complete
    deadline = time.time() + 600
    while not all(s["complete"] for s in c.get("/studies")):
        if not any(t.is_alive() for t in threads):
            t = threading.Thread(target=work, kwargs=dict(url=url, token="secret", device="cpu", target_minutes=0.02,
                                                          exit_when_idle=True, name="late", idle_sleep=1.0,
                                                          heartbeat_every=0.5, log=quiet.append))
            t.start()
            threads.append(t)
        assert time.time() < deadline, "studies did not complete"
        time.sleep(0.5)
    jobs = sum(1 for line in quiet if line.startswith("job "))
    print(f"  {jobs} slices run by {len({line.split()[1] for line in quiet if line.startswith('worker')})} workers "
          f"(plus a rogue lease that timed out)")

    for sid, kind, params in ((s1, "survey", survey), (s2, "allhands", allh)):
        merged = dict(np.load(io.BytesIO(c.get(f"/studies/{sid}/result"))))
        whole = tasks.run(kind, params, 0, tasks.total_items(kind, params), "cpu")
        for k in whole:
            assert np.array_equal(merged[k], whole[k]), (kind, k)
        print(f"  {kind}: distributed result identical to one run ({tasks.total_items(kind, params):,} items)")

    # a slice the rogue held was re-run by a real worker
    print("  rogue's slice", rogue["start"], "-", rogue["end"], "completed after its lease expired")

    # a search study shared by a CPU-only worker and a GPU worker equals a single run
    try:
        from numba import cuda
        has_gpu = cuda.is_available()
    except ImportError:
        has_gpu = False
    if has_gpu:
        mixed = tasks.study_params("survey", "ten_point", 4, 4, 77, hands=3000)
        s3 = c.post("/studies", {"kind": "survey", "params": mixed, "engine": "any", "calib": 300})["id"]
        ws = [threading.Thread(target=work, kwargs=dict(url=url, token="secret", device=d, target_minutes=0.01,
                                                        exit_when_idle=True, name=f"mixed-{d}", idle_sleep=1.0,
                                                        heartbeat_every=0.5, log=quiet.append))
              for d in ("cpu", "auto")]
        for t in ws:
            t.start()
        for t in ws:
            t.join()
        devs = {sl["device"] for sl in c.get(f"/studies/{s3}/slices")}
        merged = dict(np.load(io.BytesIO(c.get(f"/studies/{s3}/result"))))
        whole = tasks.run("survey", mixed, 0, 3000, "cpu")
        assert np.array_equal(merged["rows"], whole["rows"])
        print(f"  search study split across {sorted(devs)} workers: identical to one CPU run")

    try:
        from numba import cuda
        if cuda.is_available():
            from cardsim.pitch import PitchRules, fast, gpu
            g = fast.Game(PitchRules.preset("ten_point"), 4)
            s = np.full(g.nteams, 4, np.int64)
            whole = gpu.survey(4000, g, s, seed=3)
            parts = np.concatenate([gpu.survey(1500, g, s, seed=3, start=0, batch=700),
                                    gpu.survey(2500, g, s, seed=3, start=1500, batch=1100)])
            assert np.array_equal(whole, parts)
            print("  GPU search survey: identical whether run whole or in pieces")
    except ImportError:
        pass
    server.should_exit = True


if __name__ == "__main__":
    print("distributed studies vs single runs:")
    main()
