# Workers on AWS

Studies here run on a home cluster (an RTX 3070 and a GTX 1060 plus their CPUs). AWS adds
GPUs for a few hours when a study is big or wanted soon. The workers are ready for it:
`--spot` stops a worker cleanly on a spot interruption notice, losing nothing (it hands
back the rest of its slice), `--gpu-tuning auto` fits the play-out kernel to whatever GPU
it lands on, and `--auto-update` fetches the code each study needs.

## Which instances

Search (look-ahead) play is all GPU integer work, one GPU per worker. Measured at home for
4-handed setback4 with 16 samples per decision, with the Numba play-out kernel the
estimates below were scaled from:

| Machine | Games per second |
|---|---|
| RTX 3070 | about 225,000 (6-handed: 95,000) |
| GTX 1060 | about 22,000 |
| Ryzen 5600X, 6 cores | about 4,500 |

The C++ play-out kernel (2026-10-15) runs the 3070 at about 400,000 (6-handed 180,000),
roughly doubling every GPU figure below and halving its cost per game; the ranking of the
instance types should hold, but measure before relying on it.

The AWS figures below are **estimates** scaled from those by each GPU's multiprocessors,
sustained clocks and generation (the 1060 shows older cards do worse than their specs
suggest), so expect ±30% until measured. Prices are spot, Linux, per hour on 2026-10-15:
the first from instances.vantage.sh (us-east-1), the second the lowest US region in AWS's
spot price feed, which can lag. Check current prices before launching
(`aws ec2 describe-spot-price-history --instance-types g5.xlarge ...`).

| Instance | GPU | Est. games/s | Spot $/h | $ per billion games |
|---|---|---|---|---|
| g4dn.xlarge | T4 (Turing, 70 W) | 70k–110k | 0.33 (0.12) | about 1.00 |
| **g5.xlarge** | **A10G (Ampere)** | **250k–330k** | **0.46 (0.42)** | **about 0.45** |
| g6.xlarge | L4 (Ada, 72 W) | 170k–240k | 0.62 (0.10?) | about 0.85 |
| g6e.xlarge | L40S (Ada) | 600k–800k | 1.83 | about 0.75 |
| c7a.48xlarge | 192 CPU cores | about 150k | 1.70 | about 3 |
| home RTX 3070 | | 225k | electricity ~0.03 | about 0.04 |

- **g5.xlarge is the default pick**: the lowest cost per game, available in most regions,
  and an Ampere card like the 3070, so the estimate is the firmest.
- **g6e.xlarge** when time matters more than cost: about three g5s in one machine.
- **g6.xlarge** only if its spot price really is near the feed's $0.10; at $0.62 it loses to g5.
- **Skip CPU instances** (several times the cost per game) and **g4dn** (an older, power-capped card).
- **Use single-GPU sizes** (`xlarge`). A worker drives one GPU; on a multi-GPU instance run
  one worker per GPU with `CUDA_VISIBLE_DEVICES`.
- The 4 vCPUs of an xlarge are enough: dealing the cards takes about 6% of a GPU worker's time.

**What it costs.** Everything published so far (every preset's bid charts plus the four
setback4 every-hand tables) is under a billion 4-handed-equivalent games: about $0.45 of
g5 spot, or an hour on five g5s. The reason to use AWS is wall-clock time, or studies the
home cluster would take days on (more deals, more samples per decision, Spades and Hearts),
not cost. Uploading results back is small (an every-hand table is about 650 MB, about $0.06
of egress).

**Measure before a big run.** An hour each of g5, g6 and g6e (about $3 in all) running
`python -m cardsim.pitch.gpu_tune` and `python tests/bench_perf.py` replaces the estimates
above with real numbers.

## Spot interruptions

With `--spot`, a worker polls the instance metadata service (IMDSv2) every 5 seconds. On an
interruption notice (about two minutes before the instance goes) it finishes the piece of
its slice it is on (pieces are about 20 s, `--piece-seconds`), uploads the part of the
slice it has done, and exits; the coordinator queues the rest for another worker. With no
notice (a hard kill), the slice goes back in the queue after the 2-minute lease timeout,
so at most one slice of work (about 7.5 minutes; less with `--target-minutes`) is repeated.
SIGTERM (a normal shutdown, `docker stop`) does the same as a notice.

Run workers natively on EC2 (as below) rather than in Docker: with IMDSv2, a container
reaches the metadata service only if the instance's metadata hop limit is 2 or more, and
many AMIs default to 1.

## Launching workers

The coordinator stays at home. EC2 workers reach it over [Tailscale](https://tailscale.com),
which keeps port 8700 off the internet and encrypts the traffic: install Tailscale on the
coordinator's machine, note its tailnet name, and create a reusable, ephemeral auth key
(ephemeral nodes disappear from the tailnet when the instance goes).

[`scripts/aws-worker-userdata.sh`](../scripts/aws-worker-userdata.sh) is user data for an
Ubuntu GPU instance with the NVIDIA driver installed, such as AWS's "Deep Learning Base OSS
Nvidia Driver GPU AMI (Ubuntu 24.04)". Fill in the three values at its top, then launch
spot instances with it, for example:

```bash
aws ec2 run-instances --image-id <DLAMI id> --instance-type g5.xlarge --count 4 \
  --instance-market-options 'MarketType=spot,SpotOptions={SpotInstanceType=one-time}' \
  --instance-initiated-shutdown-behavior terminate \
  --metadata-options 'HttpTokens=required' \
  --user-data file://scripts/aws-worker-userdata.sh
```

Each instance joins the tailnet, installs the code, tunes its GPU, and works with
`--spot --auto-update --exit-when-idle`; when the queue is empty it shuts down, and with
`--instance-initiated-shutdown-behavior terminate` that ends the billing. So: push, submit
the studies, then launch. The user data (including the token and auth key) is readable by
anyone who can describe the instance, so keep the AWS account to yourself.

The script has not yet been run on AWS; treat the first launch as a test.
