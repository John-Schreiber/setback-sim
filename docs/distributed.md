# Running studies across several machines

One machine runs the **coordinator** (a job queue); any number of machines run
**workers**. A study, for example "10 million ten-point deals with search players", is
split into slices of about 5–10 minutes each. Each worker leases a slice sized to its
own measured speed, runs it on its GPU (or CPU), and uploads the result. The merged
result is **identical** to running the whole study on one machine: every deal and every
search decision is seeded by its global index, and each study pins the code version and
trump-choice weights it was submitted with.

If a worker dies, its slice goes back in the queue after 2 minutes without a heartbeat.
If the coordinator restarts, its queue and finished slices are on disk (`dist-data/`).

## 1. Coordinator

Pick a token, so only your machines can use the queue, and start the coordinator with
Docker:

```bash
git clone https://github.com/John-Schreiber/setback-sim && cd setback-sim
export COORDINATOR_TOKEN=pick-a-long-random-string
docker compose -f docker/compose.coordinator.yml up -d --build
```

Other machines on your network need to reach TCP port 8700 on this machine; if it runs a
firewall, allow inbound connections to that port from your local network only. Note this
machine's address on the network (for example `192.168.1.10`).

## 2. GPU workers

Needs Docker and the NVIDIA Container Toolkit. To check that containers see the GPU:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

Then:

```bash
git clone https://github.com/John-Schreiber/setback-sim && cd setback-sim
export COORDINATOR_URL=http://192.168.1.10:8700 COORDINATOR_TOKEN=pick-a-long-random-string
docker compose -f docker/compose.worker.yml up -d --build
docker compose -f docker/compose.worker.yml logs -f
```

The image uses CUDA 13 (driver 580 or newer). For older drivers, such as on Pascal cards
(GTX 10xx), build it with CUDA 12 instead: `GPU_EXTRA=gpu-cu12 docker compose ...`.

On the coordinator's own machine, point the worker at `http://host.docker.internal:8700`.

The first slice of each study is a short calibration run; compiling for the GPU takes a
minute or two of it. After that, slices take about `TARGET_MINUTES` (default 7.5).

### CPU workers

Any machine can also lend its CPU. A CPU worker can share a machine with a GPU worker;
cap its threads so the GPU worker keeps a couple of cores for dealing cards:

```bash
CPU_THREADS=10 docker compose -f docker/compose.cpu-worker.yml up -d --build
```

For 4-handed search with 16 samples per decision, a 6-core Ryzen 5600X does about 4,600
deals per second, a GTX 1060 about 22,000 and an RTX 3070 about 240,000.

### Without Docker

A worker can also run straight from a checkout, which avoids any container overhead:

```bash
uv sync --extra gpu        # --extra gpu-cu12 for older drivers; plain `uv sync` for CPU only
export COORDINATOR_URL=http://192.168.1.10:8700 COORDINATOR_TOKEN=pick-a-long-random-string
uv run python -m cardsim.dist.worker --device auto --name my-gpu    # or --device cpu
```

## 3. Submitting studies and getting results

From any machine with the repo and `uv sync`:

```bash
export COORDINATOR_URL=http://192.168.1.10:8700 COORDINATOR_TOKEN=...
uv run python -m cardsim.dist.client submit survey --preset ten_point -p 4 --hands 10000000 --search 16
uv run python -m cardsim.dist.client submit allhands --preset setback4 -p 4 --deals 224 --search 16
uv run python -m cardsim.dist.client status
uv run python -m cardsim.dist.client fetch STUDY_ID -o ten_point_4p.npz
```

| Kind | Items | Result |
|---|---|---|
| `survey` | random deals | `rows`: trump, hand, bidding team's points, others' points, dealt hand, swept all tricks |
| `allhands` | the 962,988 distinct 6-card hands (52-card, 6-card presets) | `pts`: every team's points (bidder's team first) and `sweep`: bidder's team took every trick, per hand and deal; `trump` per hand |

Search studies default to GPU workers only (`--engine gpu`); rule-based studies run on any
worker. The CPU and GPU engines play bit-identical games, search included, so submitting a
search study with `--engine any` lets CPU workers share the work and still gives exactly
the same result. `GET /studies/{id}/slices` shows which worker and device ran each slice.

## Updating the code

Rebuild the images on every machine (`docker compose ... up -d --build`) after pulling
new code. A study records the code version it was submitted with; a worker running
different code refuses its slices and logs why, so results never mix versions.
