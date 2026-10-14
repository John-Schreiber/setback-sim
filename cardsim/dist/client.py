"""Submit studies to the coordinator, watch progress, and fetch merged results.

    python -m cardsim.dist.client --url http://HOST:8700 submit survey --preset ten_point -p 4 \
        --hands 10000000 --search 16
    python -m cardsim.dist.client --url http://HOST:8700 status
    python -m cardsim.dist.client --url http://HOST:8700 fetch STUDY_ID -o ten_point_4p.npz
"""
import argparse
import os

from cardsim.dist.worker import Client


def token_from_env():
    """COORDINATOR_TOKEN from the environment, else from docker/.env (git-ignored)."""
    tok = os.environ.get("COORDINATOR_TOKEN")
    if tok:
        return tok
    fn = os.path.join(os.path.dirname(__file__), "..", "..", "docker", ".env")
    if os.path.exists(fn):
        for line in open(fn):
            if line.startswith("COORDINATOR_TOKEN="):
                return line.split("=", 1)[1].strip()
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default=os.environ.get("COORDINATOR_URL", "http://localhost:8700"))
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("kind", choices=["survey", "allhands"])
    s.add_argument("--preset", default="setback4")
    s.add_argument("-p", "--players", type=int, default=4)
    s.add_argument("--hands", type=int, default=1_000_000, help="survey: number of deals")
    s.add_argument("--deals", type=int, default=224, help="allhands: deals per hand class")
    s.add_argument("--search", type=int, default=16)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--engine", default=None,
                   help="gpu, cpu or any (default: gpu with search, any without). 'any' lets CPU "
                        "workers share a search study too (CPU and GPU draw search samples differently; "
                        "each slice stays reproducible on its recorded device)")
    s.add_argument("--name")
    sub.add_parser("status")
    f = sub.add_parser("fetch")
    f.add_argument("study")
    f.add_argument("-o", "--out")
    a = ap.parse_args()
    c = Client(a.url, token_from_env())
    if a.cmd == "submit":
        from cardsim.dist.tasks import study_params
        extra = {"hands": a.hands} if a.kind == "survey" else {"deals": a.deals}
        params = study_params(a.kind, a.preset, a.players, a.search, a.seed, **extra)
        engine = a.engine or ("gpu" if a.search else "any")
        r = c.post("/studies", {"kind": a.kind, "params": params, "engine": engine,
                                "name": a.name or f"{a.kind} {a.preset} {a.players}p search {a.search}"})
        print(r["id"])
    elif a.cmd == "status":
        for s in c.get("/studies"):
            print(f"{s['id']}  {s['name']:40} {s['done']:>12,}/{s['total']:<12,} "
                  f"({100 * s['done'] / s['total']:5.1f}%)  leased {s['leased']:,}  {'complete' if s['complete'] else ''}")
    else:
        data = c.get(f"/studies/{a.study}/result")
        out = a.out or f"{a.study}.npz"
        with open(out, "wb") as fh:
            fh.write(data)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
