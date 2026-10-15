#!/bin/bash
# EC2 user data: a setback-sim GPU worker on a spot instance (see docs/aws.md).
# For an Ubuntu GPU AMI with the NVIDIA driver installed (e.g. AWS's "Deep Learning Base
# OSS Nvidia Driver GPU AMI (Ubuntu 24.04)"). Joins the tailnet, installs the code, and
# runs one worker per GPU with --spot --auto-update --exit-when-idle; when every worker has
# run out of work the instance shuts down (launch with
# --instance-initiated-shutdown-behavior terminate so that ends the billing).
# Not yet tested on AWS: treat the first launch as a test (log: /var/log/setback-worker.log).
set -euo pipefail

# ---- fill these in
TS_AUTHKEY="tskey-auth-..."              # Tailscale: reusable, ephemeral auth key
COORDINATOR_URL="http://my-pc:8700"      # the coordinator's tailnet name or address
COORDINATOR_TOKEN="..."                  # the coordinator's token (docker/.env at home)
# ----

exec > >(tee -a /var/log/setback-worker.log) 2>&1
IID=$(TOKEN=$(curl -sX PUT http://169.254.169.254/latest/api/token -H "X-aws-ec2-metadata-token-ttl-seconds: 300") &&
      curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/instance-id)

curl -fsSL https://tailscale.com/install.sh | sh
tailscale up --authkey "$TS_AUTHKEY" --hostname "setback-$IID"

export HOME=/root
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone --depth 1 https://github.com/John-Schreiber/setback-sim /opt/setback-sim
cd /opt/setback-sim
# CUDA 13 wheels need driver 580 or newer; older drivers get the CUDA 12 build
DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
EXTRA=$([ "$DRIVER" -ge 580 ] && echo gpu || echo gpu-cu12)
uv sync --extra "$EXTRA" --extra dist

NGPU=$(nvidia-smi -L | wc -l)
for i in $(seq 0 $((NGPU - 1))); do
  systemd-run --unit "setback-worker-$i" --property=KillSignal=SIGTERM --property=TimeoutStopSec=120 \
    --setenv=HOME=/root --setenv=CUDA_VISIBLE_DEVICES="$i" --setenv=PYTHONUNBUFFERED=1 \
    --setenv=COORDINATOR_URL="$COORDINATOR_URL" --setenv=COORDINATOR_TOKEN="$COORDINATOR_TOKEN" \
    --working-directory=/opt/setback-sim \
    /opt/setback-sim/.venv/bin/python -m cardsim.dist.worker --device gpu --name "aws-$IID-$i" \
      --spot --auto-update --exit-when-idle
done

# shut down once every worker has exited (queue empty, or a spot notice)
while systemctl is-active --quiet 'setback-worker-*'; do sleep 30; done
shutdown -h now
