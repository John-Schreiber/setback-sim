"""Worker: lease a slice from the coordinator, run it, upload the result, repeat.

Each slice is sized by the coordinator to take about --target-minutes at this worker's
measured speed (the first slice of each study is a short calibration run, which also
pays for compiling). A background thread sends heartbeats while a slice runs.

    python -m cardsim.dist.worker --url http://192.168.1.10:8700
Options: --device auto|gpu|cpu, --target-minutes 7.5, --exit-when-idle, --auto-update.
Set COORDINATOR_TOKEN if the coordinator requires one.

Each slice runs in pieces of about --piece-seconds (results are seeded by global index,
so the pieces join up exactly). On SIGTERM, or with --spot on an EC2 spot instance's
two-minute interruption notice, the worker finishes the piece it is on, uploads what it
has done, and exits; the coordinator queues the rest of the slice for another worker.

GPU workers pick the play-out kernel's register cap and block size for their card
(--gpu-tuning auto, the default: measured once and cached; see cardsim/pitch/gpu_tune.py).

With --auto-update (or WORKER_AUTO_UPDATE=1), a slice of a study submitted with
different code makes the worker fetch that code from GitHub and restart on it
(update.py). Without it, the worker refuses such slices.
"""
import argparse
import io
import json
import os
import signal
import socket
import threading
import time
import traceback
import urllib.error
import urllib.request
import uuid

import numpy as np

from cardsim.dist import tasks, update


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


IMDS = "http://169.254.169.254/latest"


def watch_spot(stop, log=print, every=5.0):
    """Set `stop` when EC2 schedules this spot instance's interruption (about two minutes'
    notice, from the instance metadata service, IMDSv2)."""
    while not stop.wait(every):
        try:
            tok = urllib.request.urlopen(urllib.request.Request(
                f"{IMDS}/api/token", method="PUT", headers={"X-aws-ec2-metadata-token-ttl-seconds": "300"}),
                timeout=2).read().decode()
            with urllib.request.urlopen(urllib.request.Request(
                    f"{IMDS}/meta-data/spot/instance-action", headers={"X-aws-ec2-metadata-token": tok}),
                    timeout=2) as r:
                log(f"spot interruption notice: {r.read().decode()}; stopping after this piece")
                stop.set()
        except urllib.error.HTTPError as e:
            if e.code != 404:  # 404 = no interruption scheduled
                log(f"spot watch: {e}")
        except Exception:
            pass


def work(url, token=None, device="auto", target_minutes=7.5, exit_when_idle=False, name=None,
         idle_sleep=15.0, heartbeat_every=20.0, log=print, auto_update=False, piece_seconds=20.0,
         shutdown=None, gpu_tuning="default"):
    """shutdown: an Event that, once set, makes the worker upload the part of its slice it
    has finished and return (set by SIGTERM / a spot notice in main())."""
    client = Client(url, token)
    worker = name or f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
    devices = detect_devices(device)
    shutdown = shutdown or threading.Event()
    rates = {}  # study -> items per second measured on this worker
    log(f"worker {worker}: devices {devices}, slices of about {target_minutes} min, coordinator {url}, "
        f"code {tasks.code_version()}{', auto-update' if auto_update else ''}")
    if "gpu" in devices and gpu_tuning != "default":
        from cardsim.pitch import gpu_tune
        gpu_tune.apply(gpu_tuning, log)
    while not shutdown.is_set():
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
            shutdown.wait(idle_sleep)
            continue
        jid, n = job["job"], job["end"] - job["start"]
        need = job["params"].get("code")
        if need and need != tasks.code_version() and auto_update:
            root = update.fetch(need, log)
            if root:
                try:  # hand the slice straight back rather than let its lease time out
                    client.post(f"/jobs/{jid}/fail?worker={worker}", f"restarting on code {need}".encode(),
                                "text/plain")
                except Exception:
                    pass
                log(f"restarting on code {need} from {root}")
                update.restart(root)
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
        start, end = job["start"], job["end"]
        # pieces of about piece_seconds at this worker's measured speed (the whole slice
        # while it has none, as for a short calibration slice)
        piece = max(1, int(rates[job["study"]] * piece_seconds)) if job["study"] in rates else n
        try:
            parts, k = [], start
            while k < end:
                e = min(end, k + piece)
                parts.append((k, tasks.run(job["kind"], job["params"], k, e, job["device"])))
                k = e
                if shutdown.is_set() or lost.is_set():
                    break
            arrays = tasks.merge(parts)
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
        rates[job["study"]] = (k - start) / max(secs, 1e-3)
        buf = io.BytesIO()
        np.savez(buf, **arrays)
        client.post(f"/jobs/{jid}/result?worker={worker}&seconds={secs:.3f}" + (f"&end={k}" if k < end else ""),
                    buf.getvalue(), "application/octet-stream")
        log(f"job {jid}: {job['kind']} items {start:,}-{k:,} on {job['device']} in {secs:.1f}s "
            f"({rates[job['study']]:,.0f}/s)" + (f"; stopping early, {end - k:,} items handed back" if k < end else ""))
    log("shutting down")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("COORDINATOR_URL", "http://localhost:8700"))
    ap.add_argument("--device", default=os.environ.get("WORKER_DEVICE", "auto"), choices=["auto", "gpu", "cpu"])
    ap.add_argument("--target-minutes", type=float, default=float(os.environ.get("TARGET_MINUTES", 7.5)))
    ap.add_argument("--exit-when-idle", action="store_true")
    ap.add_argument("--name", default=os.environ.get("WORKER_NAME"))
    ap.add_argument("--auto-update", action="store_true", default=os.environ.get("WORKER_AUTO_UPDATE") == "1",
                    help="fetch the code a study needs from GitHub and restart on it (update.py)")
    ap.add_argument("--spot", action="store_true", default=os.environ.get("WORKER_SPOT") == "1",
                    help="on an EC2 spot instance: stop cleanly on the interruption notice")
    ap.add_argument("--piece-seconds", type=float, default=float(os.environ.get("WORKER_PIECE_SECONDS", 20)),
                    help="run slices in pieces of about this long, the most a stop can lose")
    ap.add_argument("--gpu-tuning", default=os.environ.get("WORKER_GPU_TUNING", "auto"),
                    help="play-out kernel settings: auto (measure once per GPU, cached), default, or "
                         "[ENGINE:]REGS[:THREADS] such as cpp:72:128 (ENGINE numba or cpp)")
    a = ap.parse_args()
    shutdown = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: shutdown.set())
    if a.spot:
        threading.Thread(target=watch_spot, args=(shutdown,), daemon=True).start()
    work(a.url, os.environ.get("COORDINATOR_TOKEN"), a.device, a.target_minutes, a.exit_when_idle, a.name,
         auto_update=a.auto_update, piece_seconds=a.piece_seconds, shutdown=shutdown, gpu_tuning=a.gpu_tuning)


if __name__ == "__main__":
    main()
