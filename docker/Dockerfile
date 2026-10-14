# syntax=docker/dockerfile:1
# One image for both roles; the command picks coordinator or worker.
# Build from the repo root:  docker build -f docker/Dockerfile -t setback-sim .
FROM python:3.14-slim
# GPU_EXTRA=gpu (CUDA 13: RTX 20xx and newer, driver 580+) or gpu-cu12 (CUDA 12: older
# cards such as a GTX 10xx, driver 525+)
ARG GPU_EXTRA=gpu
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /bin/
ENV PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never \
    PATH="/app/.venv/bin:$PATH"
WORKDIR /app
# Exact versions from uv.lock. numba-cuda's packages bring the CUDA compiler libraries;
# the NVIDIA container runtime (docker run --gpus all) provides the driver, so no CUDA
# base image is needed.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-install-project --extra ${GPU_EXTRA} --extra dist
COPY setback.py trump_weights.py README.md LICENSE ./
COPY cardsim ./cardsim
EXPOSE 8700
CMD ["python", "-m", "cardsim.dist.worker"]
