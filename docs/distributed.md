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

## 1. Coordinator (on the Windows machine)

Pick a token, so only your machines can use the queue, and start the coordinator with
Docker Desktop:

```powershell
cd setback-sim
$env:COORDINATOR_TOKEN = "pick-a-long-random-string"
docker compose -f docker/compose.coordinator.yml up -d --build
```

Allow other machines on your LAN to reach port 8700. Run this once, in an administrator
PowerShell; it opens the port on private networks only:

```powershell
New-NetFirewallRule -DisplayName "setback-sim coordinator" -Direction Inbound -Protocol TCP `
  -LocalPort 8700 -Action Allow -Profile Private
```

Find this machine's LAN address with `ipconfig` (for example `192.168.1.10`).

## 2. Worker (on the Linux GPU machine)

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

The first slice of each study is a short calibration run; compiling for the GPU takes a
minute or two of it. After that, slices take about `TARGET_MINUTES` (default 7.5).

The Windows machine can run a worker too, through Docker Desktop's WSL2 GPU support:

```powershell
$env:COORDINATOR_URL = "http://host.docker.internal:8700"
docker compose -f docker/compose.worker.yml up -d --build
```

### CPU workers

Any machine can also lend its CPU. A CPU worker can share a machine with a GPU worker;
cap its threads so the GPU worker keeps a couple of cores for dealing cards:

```bash
CPU_THREADS=10 docker compose -f docker/compose.cpu-worker.yml up -d --build
```

A CPU worker does about as much search work as a GTX 1060 (2,000-3,000 4-handed search
deals per second on a 6-8 core Ryzen; an RTX 3070 does about 29,000). It takes studies
submitted with `--engine cpu` or `--engine any`.

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
| `allhands` | the 962,988 distinct 6-card hands (52-card, 6-card presets) | `own`: bidding team's points, per hand and deal |

Search studies default to GPU workers only (`--engine gpu`), so the whole result can be
reproduced bit for bit on one GPU. Submit with `--engine any` to let CPU workers share
the work: the CPU and GPU draw search samples differently, but both are valid, and each
slice is still exactly reproducible on the device it ran on (see
`GET /studies/{id}/slices`). Rule-based studies run anywhere and give identical results.

## Updating the code

Rebuild the images on every machine (`docker compose ... up -d --build`) after pulling
new code. A study records the code version it was submitted with; a worker running
different code refuses its slices and logs why, so results never mix versions.
