#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
model_root=${UMA_QMOE_MODEL_ROOT:-/home/amd/work/models}
container_image=${UMA_QMOE_CONTAINER_IMAGE:-nzhangnju/llama_factory_workshop@sha256:675c93a6149f58cebe3f8689d938973496b2fd265219d1bbd5314844e4088abe}
prompt=${1:-The capital of France is}
max_new_tokens=${2:-32}

exec docker run --rm --network none \
  --device=/dev/kfd \
  --device=/dev/dri \
  --group-add 44 \
  --group-add 992 \
  --ipc=host \
  --security-opt seccomp=unconfined \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -e PYTHONPATH=/workspace/src:/workspace/benchmarks/runners \
  -v "${repo_root}:/workspace:ro" \
  -v "${model_root}:/models:rw" \
  -w /workspace \
  "${container_image}" \
  /opt/venv/bin/python benchmarks/runners/qwen_local_halo_generate.py \
  --model /models/Qwen1.5-MoE-A2.7B \
  --expert-pack /models/UMA-QMoE/local-halo/qwen-bf16-prefix16-q8-tail8.uqtp \
  --target-policy-id qwen-bf16-prefix16-q8-tail8-v1 \
  --model-manifest-sha256 87667d3fb147eb692748ce33d0e570919281b264a3cc950b2e2e9df4dc12b13f \
  --native-build-dir /models/UMA-QMoE/local-halo/native-build-v9 \
  --prompt "${prompt}" \
  --max-new-tokens "${max_new_tokens}"
