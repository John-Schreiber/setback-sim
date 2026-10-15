"""Worker self-update (cardsim/dist/update.py).

Offline: update.version_of() reads a tree's code version without importing it and must
agree with tasks.code_version(). Online (needs GitHub): a study pinned to an older code
version is finished by an --auto-update worker, which fetches that code, restarts on it
and gives the same result as running the study here (the survey's play is unchanged
between the two versions).

    .venv/Scripts/python tests/test_update.py [--offline]
"""
import io
import os
import subprocess
import sys
import tempfile
import threading
import time
import warnings

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
warnings.filterwarnings("ignore")

import uvicorn  # noqa: E402

from cardsim.dist import tasks, update  # noqa: E402
from cardsim.dist.coordinator import create_app  # noqa: E402
from cardsim.dist.worker import Client  # noqa: E402
from test_dist import free_port  # noqa: E402

OLD_CODE = "cefa7a341a58f0ae"  # the bid-chart surveys' code (commit 2031e305 on main)


def test_version_of():
    assert update.version_of(tasks.ROOT) == tasks.code_version()
    print(f"  version_of(repo) == code_version() == {tasks.code_version()}")


def test_restart_on_old_code():
    data = tempfile.mkdtemp()
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(data, lease_timeout=5.0, token="secret"), host="127.0.0.1",
                                           port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    time.sleep(1.5)
    url = f"http://127.0.0.1:{port}"
    c = Client(url, "secret")
    params = dict(tasks.study_params("survey", "setback4", 4, 0, 5, hands=2000), code=OLD_CODE)
    sid = c.post("/studies", {"kind": "survey", "params": params, "engine": "cpu", "calib": 500})["id"]
    env = dict(os.environ, COORDINATOR_TOKEN="secret", WORKER_CODE_DIR=os.path.join(data, "code"))
    out = subprocess.run([sys.executable, "-m", "cardsim.dist.worker", "--url", url, "--device", "cpu",
                          "--exit-when-idle", "--auto-update", "--name", "updater"],
                         cwd=tasks.ROOT, env=env, capture_output=True, text=True, timeout=900)
    log = out.stdout + out.stderr
    assert out.returncode == 0, log[-3000:]
    assert f"restarting on code {OLD_CODE}" in log, log[-3000:]
    merged = dict(np.load(io.BytesIO(c.get(f"/studies/{sid}/result"))))
    whole = tasks.run("survey", dict(params, code=tasks.code_version()), 0, 2000, "cpu")
    assert np.array_equal(merged["rows"], whole["rows"])
    print(f"  worker fetched code {OLD_CODE}, restarted on it and finished the study; "
          "result identical to a run on current code")
    server.should_exit = True


if __name__ == "__main__":
    test_version_of()
    if "--offline" not in sys.argv:
        test_restart_on_old_code()
