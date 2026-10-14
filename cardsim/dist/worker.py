"""Worker: lease a slice from the coordinator, run it, upload the result, repeat.

Each slice is sized by the coordinator to take about --target-minutes at this worker's
measured speed (the first slice of each study is a short calibration run, which also
pays for compiling). A background thread sends heartbeats while a slice runs.

    python -m cardsim.dist.worker --url http://192.168.1.10:8700
Options: --device auto|gpu|cpu, --target-minutes 7.5, --exit-when-idle.
Set COORDINATOR_TOKEN if the coordinator requires one.
"""
import argparse
import io
import json
import os
import socket
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid

import numpy as np

from cardsim.dist import tasks


class Client:
    def __init__(self, url, token=None, timeout=600):
        self.url, self.token, self.timeout = url.rstrip("/"), token, timeout

    def _req(self, method, path, body=None, ctype="application/json"):
        data = json.dumps(body).encode() if isinstance(body, dict) else body
        req = urllib.request.Request(self.url + path, data=data, method=method)
        if data is not None:
            req.add_header("Content-Type", ctype)
        if self.token:
            req.add_header("X-Token", self.token)
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            raw = r.read()
            if r.headers.get("Content-Type", "").startswith("application/json"):
                return json.loads(raw)
            return raw

    def post(self, path, body=None, ctype="application/json"):
        return self._req("POST", path, body, ctype)

    def get(self, path):
        return self._req("GET", path)


def detect_devices(choice):
    if choice in ("gpu", "cpu"):
        return [choice] if choice == "cpu" else ["gpu", "cpu"]
    try:
        from numba import cuda
        return ["gpu", "cpu"] if cuda.is_available() else ["cpu"]
    except Exception:
        return ["cpu"]


def work(url, token=None, device="auto", target_minutes=7.5, exit_when_idle=False, name=None,
         idle_sleep=15.0, heartbeat_every=20.0, log=print):
    client = Client(url, token)
    worker = name or f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
    devices = detect_devices(device)
    rates = {}  # study -> items per second measured on this worker
    log(f"worker {worker}: devices {devices}, slices of about {target_minutes} min, coordinator {url}")
    while True:
        try:
            job = client.post("/lease", {"worker": worker, "devices": devices, "rates": rates,
                                         "target_seconds": target_minutes * 60})
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            log(f"coordinator unreachable ({e}); retrying")
            time.sleep(idle_sleep)
            continue
        if not job:
            if exit_when_idle:
                log("no work left; exiting")
                return
            time.sleep(idle_sleep)
            continue
        jid, n = job["job"], job["end"] - job["start"]
        stop = threading.Event()
        lost = threading.Event()

        def beat():
            while not stop.wait(heartbeat_every):
                try:
                    client.post(f"/jobs/{jid}/heartbeat?worker={worker}")
                except urllib.error.HTTPError as e:
                    if e.code == 409:
                        lost.set()
                        return
                except Exception:
                    pass

        hb = threading.Thread(target=beat, daemon=True)
        hb.start()
        t0 = time.time()
        try:
            arrays = tasks.run(job["kind"], job["params"], job["start"], job["end"], job["device"])
        except Exception:
            stop.set()
            err = traceback.format_exc()
            log(f"job {jid} failed:\n{err}")
            try:
                client.post(f"/jobs/{jid}/fail?worker={worker}", err.encode(), "text/plain")
            except Exception:
                pass
            time.sleep(idle_sleep)
            continue
        secs = time.time() - t0
        stop.set()
        if lost.is_set():
            log(f"job {jid}: lease was lost while running; dropping the result")
            continue
        rates[job["study"]] = n / max(secs, 1e-3)
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        client.post(f"/jobs/{jid}/result?worker={worker}&seconds={secs:.3f}", buf.getvalue(),
                    "application/octet-stream")
        log(f"job {jid}: {job['kind']} items {job['start']:,}-{job['end']:,} on {job['device']} in {secs:.1f}s "
            f"({rates[job['study']]:,.0f}/s)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("COORDINATOR_URL", "http://localhost:8700"))
    ap.add_argument("--device", default=os.environ.get("WORKER_DEVICE", "auto"), choices=["auto", "gpu", "cpu"])
    ap.add_argument("--target-minutes", type=float, default=float(os.environ.get("TARGET_MINUTES", 7.5)))
    ap.add_argument("--exit-when-idle", action="store_true")
    ap.add_argument("--name", default=os.environ.get("WORKER_NAME"))
    a = ap.parse_args()
    work(a.url, os.environ.get("COORDINATOR_TOKEN"), a.device, a.target_minutes, a.exit_when_idle, a.name)


if __name__ == "__main__":
    main()
