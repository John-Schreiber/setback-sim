"""Workers that stop early (spot interruption, shutdown) lose no work.

A worker runs its slices in short pieces; told to stop mid-slice, it uploads the pieces
it finished and exits, the coordinator queues the rest of the slice, another worker
finishes it, and the merged result equals one run of the whole study. A partial result
from a worker that doesn't hold the lease is refused.

    .venv/Scripts/python tests/test_spot.py
"""
import io
import os
import sys
import tempfile
import threading
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
warnings.filterwarnings("ignore")

import uvicorn  # noqa: E402

from cardsim.dist import tasks  # noqa: E402
from cardsim.dist.coordinator import create_app  # noqa: E402
from cardsim.dist import worker as wk  # noqa: E402
from cardsim.dist.worker import Client, work  # noqa: E402
from test_dist import free_port  # noqa: E402


def main():
    data = tempfile.mkdtemp()
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(data, lease_timeout=30.0, token="secret"), host="127.0.0.1",
                                           port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    time.sleep(1.5)
    url = f"http://127.0.0.1:{port}"
    c = Client(url, "secret")
    nh = 30_000
    params = tasks.study_params("survey", "setback4", 4, 16, 3, hands=nh)  # search: slow enough to stop
    sid = c.post("/studies", {"kind": "survey", "params": params, "engine": "cpu", "calib": 200})["id"]

    # worker 1: slices of ~4 s in pieces of ~0.2 s; stopped once it is partway into a slice
    shutdown, logs = threading.Event(), []
    w1 = threading.Thread(target=work, kwargs=dict(url=url, token="secret", device="cpu", target_minutes=4 / 60,
                                                   name="spot", idle_sleep=0.5, heartbeat_every=0.5,
                                                   log=logs.append, piece_seconds=0.2, shutdown=shutdown))
    w1.start()
    deadline = time.time() + 120
    while time.time() < deadline and not any(s["state"] == "done" for s in c.get(f"/studies/{sid}/slices")):
        time.sleep(0.2)
    time.sleep(1.5)  # into the next (full-size) slice
    shutdown.set()
    w1.join(60)
    assert not w1.is_alive(), "worker didn't stop"
    stopped = [m for m in logs if "stopping early" in m]
    assert stopped, logs[-5:]
    print(f"  worker stopped mid-slice: {stopped[0].split(': ', 1)[1]}")

    sl = c.get(f"/studies/{sid}/slices")
    pending = [s for s in sl if s["state"] == "pending"]
    assert pending, sl
    print(f"  the rest of the slice, {pending[0]['start']:,}-{pending[0]['end']:,}, is back in the queue")

    # worker 2 finishes the study
    work(url, "secret", "cpu", target_minutes=4 / 60, exit_when_idle=True, name="w2", idle_sleep=0.5,
         heartbeat_every=0.5, log=lambda *_: None)
    merged = dict(np.load(io.BytesIO(c.get(f"/studies/{sid}/result"))))
    whole = tasks.run("survey", params, 0, nh, "cpu")
    assert np.array_equal(merged["rows"], whole["rows"])
    print(f"  merged result identical to one run ({nh:,} deals, {len(c.get(f'/studies/{sid}/slices'))} slices)")

    # refusal: lease a slice as "a", then "b" tries to send part of it
    sid2 = c.post("/studies", {"kind": "survey", "params": params, "engine": "cpu", "calib": 1000})["id"]
    job = c.post("/lease", {"worker": "a", "devices": ["cpu"], "rates": {}, "target_seconds": 1})
    part = tasks.run("survey", params, job["start"], job["start"] + 10, "cpu")
    buf = io.BytesIO()
    np.savez(buf, **part)
    r = c.post(f"/jobs/{job['job']}/result?worker=b&seconds=1&end={job['start'] + 10}", buf.getvalue(),
               "application/octet-stream")
    assert r == {"ok": False}, r
    assert all(s["state"] == "leased" for s in c.get(f"/studies/{sid2}/slices"))
    print("  partial result from a worker without the lease: refused")
    server.should_exit = True


def test_spot_notice():
    """watch_spot: no notice (404) leaves the worker running; a notice stops it."""
    import http.server
    state = {"notice": False}

    class IMDS(http.server.BaseHTTPRequestHandler):
        def do_PUT(self):
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b"token")

        def do_GET(self):
            ok = self.headers.get("X-aws-ec2-metadata-token") == "token" and state["notice"]
            self.send_response(200 if ok else 404)
            self.end_headers()
            if ok:
                self.wfile.write(b'{"action": "terminate", "time": "2026-10-15T20:00:00Z"}')

        def log_message(self, *a):
            pass

    port = free_port()
    srv = http.server.HTTPServer(("127.0.0.1", port), IMDS)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    wk.IMDS = f"http://127.0.0.1:{port}/latest"
    stop, logs = threading.Event(), []
    threading.Thread(target=wk.watch_spot, args=(stop, logs.append, 0.2), daemon=True).start()
    time.sleep(1.0)
    assert not stop.is_set(), logs
    state["notice"] = True
    assert stop.wait(5), "no stop after the notice"
    print(f"  spot notice seen: {logs[-1]}")
    srv.shutdown()


if __name__ == "__main__":
    main()
    test_spot_notice()
