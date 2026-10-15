"""Worker self-update: fetch the code a study needs from GitHub and restart on it.

A study pins the code version it was submitted with (tasks.code_version: a hash of
COMPUTE_FILES). When a worker leases a slice of a study whose version differs from its
own, it looks for that version in the public repo: the tip of main first, then recent
commits on main. A snapshot is used only if its own files hash to exactly the version
the study asks for, worked out from the files without running any of them. It is
unpacked under the code cache (WORKER_CODE_DIR, default ~/.cache/setback-sim/code/<version>)
and the worker restarts from there with the same Python environment, so a native
worker's git checkout is never touched.

Off unless the worker is started with --auto-update (or WORKER_AUTO_UPDATE=1). The repo
is fixed here, never taken from the coordinator. A new dependency still needs a rebuilt
image / `uv sync`.
"""
import ast
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
import urllib.request

REPO = "John-Schreiber/setback-sim"
CODE_DIR = os.environ.get("WORKER_CODE_DIR", os.path.join(os.path.expanduser("~"), ".cache", "setback-sim", "code"))
MAX_COMMITS = 30  # how far back on main to look
KEEP = 3          # code versions kept in the cache
RETRY_AFTER = 600.0  # seconds before looking again for a version that wasn't found

_missing = {}  # version -> when it was last looked for in vain


def version_of(root):
    """tasks.code_version() of the tree at root, reading its own COMPUTE_FILES list
    (parsed, not imported) so a snapshot is checked without running it."""
    with open(os.path.join(root, "cardsim", "dist", "tasks.py"), encoding="utf-8") as f:
        mod = ast.parse(f.read())
    files = next((ast.literal_eval(n.value) for n in mod.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "COMPUTE_FILES" for t in n.targets)), None)
    if files is None:
        raise ValueError("no COMPUTE_FILES in cardsim/dist/tasks.py (older layout)")
    h = hashlib.sha256()
    for rel in files:
        with open(os.path.join(root, rel), "rb") as f:
            source = f.read().replace(b"\r\n", b"\n")
        h.update(rel.encode() + b"\n" + source)
    return h.hexdigest()[:16]


def _get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "setback-sim-worker"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def _unpack(ref, dest):
    """Download the repo at ref (branch or commit) into dest. Returns dest."""
    data = _get(f"https://codeload.github.com/{REPO}/tar.gz/{ref}", timeout=120)
    tmp = dest + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tf:
        tf.extractall(tmp, filter="data")
    (top,) = os.listdir(tmp)  # GitHub wraps the tree in one folder
    shutil.rmtree(dest, ignore_errors=True)
    os.replace(os.path.join(tmp, top), dest)
    shutil.rmtree(tmp, ignore_errors=True)
    return dest


def fetch(version, log=print):
    """Path of a verified tree with this code version (cached, or fetched from GitHub),
    or None if neither main nor its recent commits have it."""
    final = os.path.join(CODE_DIR, version)
    if os.path.exists(os.path.join(final, ".verified")):
        return final
    if time.time() - _missing.get(version, 0) < RETRY_AFTER:
        return None
    os.makedirs(CODE_DIR, exist_ok=True)
    work = os.path.join(CODE_DIR, f"_candidate-{os.getpid()}")  # workers on one machine share CODE_DIR
    refs = ["main"]
    try:
        commits = json.loads(_get(f"https://api.github.com/repos/{REPO}/commits?sha=main&per_page={MAX_COMMITS}"))
        refs += [c["sha"] for c in commits[1:]]
    except Exception as e:
        log(f"update: couldn't list commits ({e}); trying main only")
    for ref in refs:
        try:
            got = version_of(_unpack(ref, work))
        except FileNotFoundError:
            continue  # a commit from before the distributed code existed
        except Exception as e:
            log(f"update: {ref[:12]}: {type(e).__name__}: {e}")
            continue
        if got == version:
            open(os.path.join(work, ".verified"), "w").close()
            if os.path.exists(os.path.join(final, ".verified")):  # another worker got there first
                shutil.rmtree(work, ignore_errors=True)
                return final
            shutil.rmtree(final, ignore_errors=True)  # an unverified leftover
            os.replace(work, final)
            log(f"update: code {version} is {REPO}@{ref[:12]}")
            _prune(keep=final)
            return final
    shutil.rmtree(work, ignore_errors=True)
    _missing[version] = time.time()
    log(f"update: code {version} not found on main or its last {MAX_COMMITS} commits (pushed yet?)")
    return None


def _prune(keep):
    dirs = [os.path.join(CODE_DIR, d) for d in os.listdir(CODE_DIR) if not d.startswith("_")]
    dirs = sorted((d for d in dirs if d != keep), key=os.path.getmtime, reverse=True)
    for d in dirs[KEEP - 1:]:
        shutil.rmtree(d, ignore_errors=True)


def restart(root):
    """Run this worker again from root, with the same Python and arguments."""
    # older code may not know newer options: carry them in the environment instead
    env = dict(os.environ, PYTHONPATH=root, WORKER_AUTO_UPDATE="1")
    flags = {"--spot": "WORKER_SPOT"}
    valued = {"--piece-seconds": "WORKER_PIECE_SECONDS", "--gpu-tuning": "WORKER_GPU_TUNING"}
    rest, argv, i = [], sys.argv[1:], 0
    while i < len(argv):
        name, eq, val = argv[i].partition("=")
        if name == "--auto-update":
            pass
        elif name in flags:
            env[flags[name]] = "1"
        elif name in valued:
            if not eq:
                i += 1
                val = argv[i]
            env[valued[name]] = val
        else:
            rest.append(argv[i])
        i += 1
    args = [sys.executable, "-m", "cardsim.dist.worker"] + rest
    os.chdir(root)
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == "nt":  # no real exec on Windows: run it and pass its exit code on
        sys.exit(subprocess.call(args, env=env))
    os.execve(sys.executable, args, env)
