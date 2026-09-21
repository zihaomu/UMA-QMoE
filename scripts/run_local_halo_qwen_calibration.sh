#!/usr/bin/env bash
set -euo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
model_root=${UMA_QMOE_MODEL_ROOT:-/home/amd/work/models}
container_image=${UMA_QMOE_CONTAINER_IMAGE:-nzhangnju/llama_factory_workshop@sha256:675c93a6149f58cebe3f8689d938973496b2fd265219d1bbd5314844e4088abe}
output_directory=${1:?usage: run_local_halo_qwen_calibration.sh OUTPUT_DIRECTORY [FIXTURE] [runner args...]}
fixture=${2:-benchmarks/fixtures/qwen1_5_moe_smoke_v1.jsonl}
if [[ $# -ge 2 ]]; then
  shift 2
else
  shift 1
fi

if [[ "${fixture}" = /* || "${fixture}" == *..* || ! -f "${repo_root}/${fixture}" ]]; then
  echo "fixture must be an existing project-relative path without traversal: ${fixture}" >&2
  exit 2
fi
if [[ "${output_directory}" = /* || "${output_directory}" == *..* ]]; then
  echo "output directory must be a project-relative path without traversal" >&2
  exit 2
fi

source_name=${UMA_QMOE_DATASET_NAME:-$(basename -- "${fixture}" .jsonl)}
source_uri=${UMA_QMOE_DATASET_URI:-repo://${fixture}}
source_revision=${UMA_QMOE_DATASET_REVISION:-$(sha256sum "${repo_root}/${fixture}" | awk '{print $1}')}
source_license=${UMA_QMOE_DATASET_LICENSE:-Apache-2.0}

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
  -v "${repo_root}:/workspace:rw" \
  -v "${model_root}:/models:ro" \
  -w /workspace \
  "${container_image}" \
  /opt/venv/bin/python benchmarks/runners/capture_qwen_calibration.py \
  --target-id local-halo \
  --backend hip \
  --model /models/Qwen1.5-MoE-A2.7B \
  --fixture "${fixture}" \
  --source-name "${source_name}" \
  --source-uri "${source_uri}" \
  --source-revision "${source_revision}" \
  --license "${source_license}" \
  --output-directory "${output_directory}" \
  "$@"
