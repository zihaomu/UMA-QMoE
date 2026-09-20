# UMA-QMoE

UMA-QMoE is a quantization-native MoE inference project for bandwidth-constrained
unified-memory systems. The first targets are NVIDIA DGX Spark / GB10 and AMD
Strix Halo / Radeon 8060S.

The public control plane and the OLMoE M0-M1 foundation are established. Active
work is closing the Spark-only M2 path and preparing Qwen M3-M5 target evidence.
See the [project handoff and recovery status](doc/UMA_QMOE_PROJECT_HANDOFF.md) for
the authoritative checkpoint, MVP distance, blockers, and restore order.

## Runtime boundary

UMA-QMoE does not embed vLLM. The core path is a fixed PyTorch/Hugging Face
model host that loads dense BF16 tensors, maps one Q4 ExpertPack, and invokes
the project-owned `uma_qmoe::moe_forward` operator. vLLM remains an isolated
external comparison under `benchmarks/external/vllm/`; CI rejects imports in
either direction between that external runner and the core runtime.

The operator includes the CPU correctness implementation, guarded SM121/gfx1151
dispatch, and a target-compiled W4A16 backend that reads canonical packed Q4
bytes and FP32 group-128 scales directly. Call
`uma_qmoe.native_backend.install_packed_q4_backend()` before selecting
performance mode. Performance mode still fails closed when the backend is not
explicitly installed, so benchmark runs cannot silently fall back to the
correctness implementation. The first native backend is a correctness-first
microkernel baseline; its single-layer timing is not an end-to-end TPS claim.

## Bootstrap development

The Python control plane uses [uv](https://docs.astral.sh/uv/):

```bash
uv sync --extra dev
uv run pytest -q
```

Validate the pinned OLMoE manifest and its M0–M2 benchmark contract:

```bash
uv run umaq validate models/manifests/olmoe_1b_7b_0125.yaml
uv run umaq validate benchmarks/contracts/olmoe_1b_7b_0125.yaml
```

Rebuild the draft manifest from immutable Hugging Face metadata without
downloading weight payloads:

```bash
uv run umaq import-hf-manifest \
  --repo-id allenai/OLMoE-1B-7B-0125 \
  --revision 9b0c1aa87e34a20052389dce1f0cf01da783f654 \
  --variant base \
  --model-card-weight-dtype bfloat16 \
  --oracle-dtype bfloat16 \
  --declared-total-parameters 7B \
  --declared-active-parameters 1.3B \
  --output /tmp/olmoe-manifest.yaml
```

The pinned repository metadata currently reports F32 tensors even though the
model card says BF16. The draft records both facts and stays unfrozen until a
local Safetensors scan resolves the conflict.

Download or resume the exact pinned files into the workspace-level source
cache, then emit verification evidence. The downloader writes only `.part`
files until both the byte count and SHA-256 match the manifest:

```bash
REVISION=9b0c1aa87e34a20052389dce1f0cf01da783f654
SOURCE_ROOT=/absolute/path/to/workspace/models/source/OLMoE-1B-7B-0125/$REVISION

uv run umaq fetch-model \
  models/manifests/olmoe_1b_7b_0125.yaml \
  "$SOURCE_ROOT" \
  --jobs 2 \
  --output /absolute/private/path/olmoe-source-acquisition.json
uv run umaq verify-model \
  models/manifests/olmoe_1b_7b_0125.yaml \
  "$SOURCE_ROOT" \
  --output /absolute/private/path/olmoe-source-verification.json
```

The contract is intentionally still `draft`. The release gate must reject it
until all recorded blockers are resolved:

```bash
uv run umaq validate --require-frozen \
  benchmarks/contracts/olmoe_1b_7b_0125.yaml
```

Collect a non-sensitive machine baseline as JSON:

```bash
uv run umaq collect-machine --target-id local-test --output machine-baseline.json
uv run umaq validate machine-baseline.json
```

Capture a read-only procfs/cgroup memory snapshot. This is input to the Safe
UMA Budget; it is not itself a capacity claim:

```bash
uv run umaq probe-memory --target-id local-test --output memory-snapshot.json
uv run umaq validate memory-snapshot.json
```

Build a draft Safe UMA Budget only after every reserve has measured provenance.
The source snapshot hash is derived automatically. `MemAvailable` is recorded
as a health signal but never used as the physical ceiling:

```bash
uv run umaq build-safe-budget memory-snapshot.json \
  --os-daemon-reserve-bytes "$OS_DAEMON_RESERVE_BYTES" \
  --runtime-reserve-bytes "$RUNTIME_RESERVE_BYTES" \
  --kv-budget-bytes "$KV_BUDGET_BYTES" \
  --workspace-budget-bytes "$WORKSPACE_BUDGET_BYTES" \
  --safety-margin-bytes "$SAFETY_MARGIN_BYTES" \
  --output safe-uma-budget.json
uv run umaq validate safe-uma-budget.json
```

Do not substitute guessed values for those variables. A frozen budget also
requires `--status frozen`, `--policy-provenance`, and `--decision-record`, and
is rejected when the cgroup ceiling is unknown. A frozen BenchmarkContract
must then bind one hash-verified Safe UMA Budget file for every target.

After the pinned source shards have been downloaded once, stream-verify every
file and tensor payload without loading a shard into memory:

```bash
uv run umaq inventory-safetensors \
  models/manifests/olmoe_1b_7b_0125.yaml \
  "$SOURCE_ROOT" \
  --output models/inventories/olmoe_1b_7b_0125_f32.json
uv run umaq validate models/inventories/olmoe_1b_7b_0125_f32.json
uv run umaq freeze-model-manifest \
  models/manifests/olmoe_1b_7b_0125.yaml \
  models/inventories/olmoe_1b_7b_0125_f32.json \
  --output models/manifests/olmoe_1b_7b_0125.yaml
```

The ModelManifest remains `draft` until the generated TensorInventory is
hash-bound to it and the local shard, dtype, and tensor-payload checks pass.

After freezing that binding, estimate Q4 expert storage and the batch-1 cold
weight bytes read per decode token. The estimate includes packed payload,
per-group scale/zero-point bytes, and per-tensor alignment; it deliberately
excludes KV, activations, cache effects, and hardware traffic amplification:

```bash
uv run umaq estimate-weight-traffic \
  models/manifests/olmoe_1b_7b_0125.yaml \
  models/inventories/olmoe_1b_7b_0125_f32.json \
  --dense-bits 16 \
  --expert-bits 4 \
  --group-size 128 \
  --scale-bytes 2 \
  --tensor-alignment 128 \
  --output models/estimates/olmoe_q4_group128.json
```

Spark deliberately does not require privileged performance counters. Build the
versioned traffic sensitivity model from the frozen source ledger, canonical Q4
estimate, RouteTrace, unprivileged bandwidth soak, and local BF16 config:

```bash
uv run umaq build-spark-traffic-model \
  benchmarks/sources/spark1_traffic_source_ledger_v1.json \
  models/estimates/olmoe_q4_group128.json \
  benchmarks/traces/olmoe_1b_7b_0125_128x32_greedy_v1.json \
  /absolute/path/to/spark1-bandwidth-soak-v1.json \
  /absolute/path/to/bf16-rne-v1/config.json \
  --output /absolute/path/to/spark1-traffic-model.json
```

The report is restricted to `modeled_estimated`: its token rates are bandwidth
ceilings over explicit amplification factors, not measured DRAM traffic or
end-to-end throughput.

Once that manifest is frozen, derive the deterministic BF16 Oracle without
materializing a whole shard in memory. This path uses explicit
round-to-nearest-even, preserves NaNs, resumes compatible shard `.part` files,
and verifies the source SHA-256 while converting:

```bash
BF16_ROOT=/absolute/path/to/workspace/models/derived/OLMoE-1B-7B-0125/$REVISION/bf16-rne-v1

uv sync --extra conversion
uv run umaq derive-bf16 \
  models/manifests/olmoe_1b_7b_0125.yaml \
  "$SOURCE_ROOT" \
  "$BF16_ROOT" \
  --artifact-root "models/derived/OLMoE-1B-7B-0125/$REVISION/bf16-rne-v1" \
  --output models/manifests/olmoe_1b_7b_0125_bf16_oracle.yaml
uv run umaq validate models/manifests/olmoe_1b_7b_0125_bf16_oracle.yaml
uv run umaq verify-derivation \
  models/manifests/olmoe_1b_7b_0125_bf16_oracle.yaml \
  "$BF16_ROOT" \
  --target-id local \
  --output /absolute/private/path/olmoe-bf16-verification.json
```

Inside each target's pinned Torch/Transformers container, load that target's
local BF16 derivation with networking disabled and run one fixed full-model
forward pass. The command binds the result to the derivation and prompt fixture,
and records finite logits, top tokens, runtime versions, and peak accelerator
memory. It is a load/compute smoke gate, not yet a quality-equivalence claim:

```bash
PYTHONPATH=src python -m uma_qmoe.cli oracle-smoke \
  models/manifests/olmoe_1b_7b_0125_bf16_oracle.yaml \
  "$BF16_ROOT" \
  benchmarks/fixtures/olmoe_smoke_v1.jsonl \
  --prompt-id general-001 \
  --target-id "$TARGET_ID" \
  --output /absolute/private/path/olmoe-bf16-oracle-smoke.json
```

The initial UMA memory microbenchmark also runs inside the pinned target
container. It measures synchronized PyTorch read-reduce, write-fill, and copy
operations. Reported GB/s uses algorithmic bytes, with both read and write
bytes counted for copy; the evidence remains explicitly uncalibrated until a
target-native streaming kernel and DRAM counters are added:

```bash
PYTHONPATH=src python -m uma_qmoe.cli benchmark-memory-bandwidth \
  --target-id "$TARGET_ID" \
  --buffer-mib 512 \
  --warmup 3 \
  --iterations 10 \
  --output /absolute/private/path/memory-bandwidth-v1.json
```

For stability and thermal diagnostics, add `--inner-loops 64 --telemetry`.
Management-tool samples are taken before and after each timed sample, never
inside the timed interval. Evidence generated before `inner_loops` was added
continues to validate with an implicit value of one.

Run the M0 continuous thermal-stability gate with the same buffer and inner
loop count. A duration below 1800 seconds is emitted as `diagnostic`; a full
run passes only when the requested duration is reached, the workload cgroup v2
has swap disabled (`memory.swap.max=0`) with zero current swap and no OOM event,
every operation has CV at most 3%, and first-to-last bandwidth drift stays
within 5%. Host-global `/proc/vmstat` swap counters remain in the evidence as
an interference diagnostic, but they are not attributed to the workload.
Management telemetry is sampled outside timed regions. When Docker launches
the benchmark, set equal `--memory` and `--memory-swap` limits to disable swap
for the container (for example, `--memory 8g --memory-swap 8g`):

```bash
PYTHONPATH=src python -m uma_qmoe.cli benchmark-bandwidth-soak \
  --target-id "$TARGET_ID" \
  --buffer-mib 512 \
  --duration-seconds 1800 \
  --warmup-cycles 3 \
  --inner-loops 64 \
  --telemetry-interval-seconds 5 \
  --output /absolute/private/path/bandwidth-soak-v1.json
```

Build the first diagnostic Allocation Matrix before choosing an arena. The v1
command measures runtime-device copies, pageable and pinned staging transfers,
and a pre-touched file mapping. It records native Managed/Unified and VMM/HMM
as not measured until target-specific allocator probes exist; a normal device
allocation is never relabeled as a managed allocation:

```bash
PYTHONPATH=src python -m uma_qmoe.cli benchmark-allocation-matrix \
  --target-id "$TARGET_ID" \
  --buffer-mib 256 \
  --warmup 3 \
  --iterations 10 \
  --output /absolute/private/path/allocation-matrix-v1.json
```

Before adding those native cases, bind the target's allocator capabilities to
the checked-in CUDA/HIP source and compiler architecture. Capability flags are
only admission checks; they are not performance results and do not turn an
unmeasured Matrix row into a measured one:

```bash
PYTHONPATH=src python -m uma_qmoe.cli probe-native-allocation-capabilities \
  benchmarks/native/allocation_capabilities.cu \
  --target-id "$TARGET_ID" \
  --backend cuda \
  --arch sm_121 \
  --output /absolute/private/path/native-allocation-capabilities-v1.json
```

Compile and run the shared known-byte streaming source with the target's native
compiler. CUDA builds require an SM architecture such as `sm_121`; HIP builds
require a GFX architecture such as `gfx1151`. The command uses GPU events and
records every raw sample plus algorithmic read/write bytes. Its output remains
`counter_calibrated=false` until a trusted hardware DRAM counter is compared
against the known byte count:

```bash
PYTHONPATH=src python -m uma_qmoe.cli benchmark-native-stream \
  benchmarks/native/native_stream.cu \
  --target-id "$TARGET_ID" \
  --backend "$BACKEND" \
  --arch "$ARCH" \
  --buffer-mib 512 \
  --warmup 3 \
  --iterations 10 \
  --inner-loops 64 \
  --output /absolute/private/path/native-stream-v1.json
```

On gfx1151, normalize the filtered `rocprofv3` counter CSV against one known
byte count per dispatch. The parser excludes the source-initialization write,
retains every profiled dispatch and fails the aggregate gate if any selected
read/write mapping exceeds the error threshold:

```bash
PYTHONPATH=src python -m uma_qmoe.cli calibrate-rocprof-counters \
  benchmarks/native/native_stream.cu \
  /absolute/private/path/counters_counter_collection.csv \
  --target-id "$TARGET_ID" \
  --arch gfx1151 \
  --profiler-version "rocprofv3 1.3.3" \
  --known-bytes 536870912 \
  --maximum-relative-error 0.10 \
  --output /absolute/private/path/hardware-counter-calibration-v1.json
```

The current mappings are intentionally architecture- and profiler-stack-
specific: `GL2C_EA_RDREQ_DRAM_sum × 128 B` for reads and
`GCEA_WDRAM_SIZE_REQ_sum × 32 B` for writes. Do not reuse them on another GPU
or tool version without a new calibration.

Build the pinned Python package artifacts with:

```bash
uv build
```

Large model files, generated benchmark runs, and private machine inventory are
kept outside the public repository.
