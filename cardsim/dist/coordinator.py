"""Coordinator: hands out slices of studies to workers and collects the results.

State lives in one SQLite file plus one .npz per finished slice, under --data. A
worker leases a slice sized to take about `target_seconds` at its own measured rate
(the first slice of a study is a small calibration slice), sends heartbeats while it
works, and uploads the result. A slice whose worker goes quiet for `lease_timeout`
seconds goes back in the queue, so a restarted worker or coordinator loses at most the
slices in flight.

A study's `engine` is "gpu", "cpu" or "any". Rule-based play is identical on both. Search
draws its samples differently on CPU and GPU, so a search study with engine "any" mixes
two (equally valid) sample streams; every slice is still exactly reproducible on the
device it ran on, which /studies/{id}/slices records.

    python -m cardsim.dist.coordinator --data ./dist-data --host 0.0.0.0 --port 8700
Set COORDINATOR_TOKEN to require a matching X-Token header from workers and clients.
"""
import argparse
import io
import json
import os
import sqlite3
import threading
import time
import uuid

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Request, Response

from cardsim.dist import tasks

SCHEMA = """
create table if not exists studies (
  id text primary key, kind text, params text, total integer, cursor integer default 0,
  engine text, calib integer, created real, name text);
create table if not exists jobs (
  id integer primary key autoincrement, study text, start integer, end integer,
  state text, worker text, device text, leased_at real, heartbeat_at real,
  finished_at real, seconds real, attempts integer default 0, error text);
create index if not exists jobs_study on jobs(study, state);
"""


class Store:
    def __init__(self, data, lease_timeout=120.0):
        os.makedirs(os.path.join(data, "results"), exist_ok=True)
        self.data = data
        self.lease_timeout = lease_timeout
        self.db = sqlite3.connect(os.path.join(data, "coordinator.db"), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self.lock = threading.Lock()

    # ---------------------------------------------------------------- studies
    def add_study(self, kind, params, engine="gpu", calib=None, name=None):
        if kind not in tasks.KINDS:
            raise ValueError(f"unknown kind {kind}")
        total = tasks.total_items(kind, params)
        sid = uuid.uuid4().hex[:12]
        with self.lock, self.db:
            self.db.execute("insert into studies values (?,?,?,?,0,?,?,?,?)",
                            (sid, kind, json.dumps(params), total, engine,
                             calib or max(1, min(total, 2000 if kind == "survey" else 64)), time.time(),
                             name or kind))
        return sid

    def studies(self):
        out = []
        for s in self.db.execute("select * from studies order by created").fetchall():
            c = {r["state"]: r["n"] for r in self.db.execute(
                "select state, sum(end - start) as n from jobs where study=? group by state", (s["id"],))}
            done = c.get("done", 0)
            out.append({"id": s["id"], "name": s["name"], "kind": s["kind"], "engine": s["engine"],
                        "params": json.loads(s["params"]), "total": s["total"], "done": done,
                        "leased": c.get("leased", 0), "pending": c.get("pending", 0),
                        "complete": done == s["total"]})
        return out

    # ---------------------------------------------------------------- leasing
    def _expire(self, now):
        self.db.execute("update jobs set state='pending', worker=null where state='leased' and heartbeat_at < ?",
                        (now - self.lease_timeout,))

    def lease(self, worker, devices, rates, target_seconds):
        now = time.time()
        with self.lock, self.db:
            self._expire(now)
            for s in self.db.execute("select * from studies order by created").fetchall():
                engine = s["engine"]
                device = next((d for d in ("gpu", "cpu") if d in devices and engine in (d, "any")), None)
                if device is None:
                    continue
                job = self.db.execute("select * from jobs where study=? and state='pending' order by start limit 1",
                                      (s["id"],)).fetchone()
                if job is not None:
                    self.db.execute("update jobs set state='leased', worker=?, device=?, leased_at=?, heartbeat_at=?,"
                                    " attempts=attempts+1 where id=?", (worker, device, now, now, job["id"]))
                    start, end, jid = job["start"], job["end"], job["id"]
                elif s["cursor"] < s["total"]:
                    rate = rates.get(s["id"])
                    size = s["calib"] if not rate else int(rate * target_seconds)
                    size = max(1, min(size, s["total"] - s["cursor"]))
                    start, end = s["cursor"], s["cursor"] + size
                    self.db.execute("update studies set cursor=? where id=?", (end, s["id"]))
                    cur = self.db.execute("insert into jobs (study,start,end,state,worker,device,leased_at,"
                                          "heartbeat_at,attempts) values (?,?,?,'leased',?,?,?,?,1)",
                                          (s["id"], start, end, worker, device, now, now))
                    jid = cur.lastrowid
                else:
                    continue
                return {"job": jid, "study": s["id"], "kind": s["kind"], "params": json.loads(s["params"]),
                        "start": start, "end": end, "device": device}
        return None

    def heartbeat(self, jid, worker):
        with self.lock, self.db:
            n = self.db.execute("update jobs set heartbeat_at=? where id=? and state='leased' and worker=?",
                                (time.time(), jid, worker)).rowcount
        return n == 1

    def finish(self, jid, worker, payload, seconds):
        with self.lock:
            job = self.db.execute("select * from jobs where id=?", (jid,)).fetchone()
            if job is None or job["state"] == "done":
                return False
            arrays = dict(np.load(io.BytesIO(payload)))  # validate before accepting
            d = os.path.join(self.data, "results", job["study"])
            os.makedirs(d, exist_ok=True)
            fn = os.path.join(d, f"{job['start']:012d}_{job['end']:012d}.npz")
            np.savez(fn + ".tmp.npz", **arrays)
            os.replace(fn + ".tmp.npz", fn)
            with self.db:
                self.db.execute("update jobs set state='done', worker=?, finished_at=?, seconds=? where id=?",
                                (worker, time.time(), seconds, jid))
        return True

    def fail(self, jid, worker, error):
        with self.lock, self.db:
            self.db.execute("update jobs set state='pending', worker=null, error=? where id=? and worker=?",
                            (error[:2000], jid, worker))

    def slices(self, sid):
        return [dict(r) for r in self.db.execute(
            "select start, end, state, worker, device, seconds, attempts from jobs where study=? order by start",
            (sid,)).fetchall()]

    def result(self, sid):
        s = self.db.execute("select * from studies where id=?", (sid,)).fetchone()
        if s is None:
            raise KeyError(sid)
        jobs = self.db.execute("select start, end from jobs where study=? and state='done'", (sid,)).fetchall()
        covered = sum(j["end"] - j["start"] for j in jobs)
        if covered != s["total"]:
            raise ValueError(f"study {sid} is {covered}/{s['total']} done")
        d = os.path.join(self.data, "results", sid)
        parts = [(j["start"], dict(np.load(os.path.join(d, f"{j['start']:012d}_{j['end']:012d}.npz"))))
                 for j in jobs]
        return tasks.merge(parts)


def create_app(data, lease_timeout=120.0, token=None):
    store = Store(data, lease_timeout)
    app = FastAPI(title="setback-sim coordinator")
    app.state.store = store

    def check(x_token):
        if token and x_token != token:
            raise HTTPException(401, "bad token")

    @app.post("/studies")
    async def add_study(req: Request, x_token: str = Header(None)):
        check(x_token)
        b = await req.json()
        try:
            sid = store.add_study(b["kind"], b["params"], b.get("engine", "gpu"), b.get("calib"), b.get("name"))
        except (ValueError, KeyError) as e:
            raise HTTPException(400, str(e))
        return {"id": sid}

    @app.get("/studies")
    def list_studies(x_token: str = Header(None)):
        check(x_token)
        return store.studies()

    @app.post("/lease")
    async def lease(req: Request, x_token: str = Header(None)):
        check(x_token)
        b = await req.json()
        job = store.lease(b["worker"], b.get("devices", ["cpu"]), b.get("rates", {}),
                          float(b.get("target_seconds", 450)))
        return job or {}

    @app.post("/jobs/{jid}/heartbeat")
    def heartbeat(jid: int, worker: str, x_token: str = Header(None)):
        check(x_token)
        if not store.heartbeat(jid, worker):
            raise HTTPException(409, "lease lost")
        return {"ok": True}

    @app.post("/jobs/{jid}/result")
    async def result(jid: int, worker: str, seconds: float, req: Request, x_token: str = Header(None)):
        check(x_token)
        ok = store.finish(jid, worker, await req.body(), seconds)
        return {"ok": ok}

    @app.post("/jobs/{jid}/fail")
    async def fail(jid: int, worker: str, req: Request, x_token: str = Header(None)):
        check(x_token)
        store.fail(jid, worker, (await req.body()).decode("utf-8", "replace"))
        return {"ok": True}

    @app.get("/studies/{sid}/slices")
    def study_slices(sid: str, x_token: str = Header(None)):
        check(x_token)
        return store.slices(sid)

    @app.get("/studies/{sid}/result")
    def study_result(sid: str, x_token: str = Header(None)):
        check(x_token)
        try:
            arrays = store.result(sid)
        except KeyError:
            raise HTTPException(404, "no such study")
        except ValueError as e:
            raise HTTPException(409, str(e))
        buf = io.BytesIO()
        np.savez_compressed(buf, **arrays)
        return Response(buf.getvalue(), media_type="application/octet-stream")

    return app


def main():
    import uvicorn
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="dist-data")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8700)
    ap.add_argument("--lease-timeout", type=float, default=120.0)
    a = ap.parse_args()
    app = create_app(a.data, a.lease_timeout, os.environ.get("COORDINATOR_TOKEN"))
    uvicorn.run(app, host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()
