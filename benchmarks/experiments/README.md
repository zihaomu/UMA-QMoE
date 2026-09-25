# Exploratory experiments (L0)

Scripts in this directory emit resumable exploratory evidence only.  Their JSONL
records are not formal policy, TargetPack, correctness, quality, or Host evidence.
Promotion into those existing L1 gates is always a separate explicit operation.

The first vertical slice scans all 72 Qwen `layer x {gate, up, down}` units while
loading the model once and restoring every unit after measurement:

```bash
python3 benchmarks/experiments/qwen_projection_rtn.py \
  --model /path/to/Qwen1.5-MoE-A2.7B \
  --model-manifest-sha256 <sha256> \
  --prompt-fixture benchmarks/fixtures/qwen1_5_moe_completion_quality_v1.jsonl \
  --backend hip \
  --ledger /path/to/qwen-rtn-q4.jsonl
```

Use `--maximum-units 1` for a wiring smoke test.  The default records each result
as `diagnostic_only`; pass `--full-quality` only when the supplied fixture really is
the frozen full-quality set.  Re-running the same command skips `passed`, `rejected`,
`error`, and `oom` identities.  An `interrupted` attempt remains in the ledger and is
retried on the next run.

## Faster Flash Decoding (FFD)

FFD is an independent, conditional research backend.  Its policy, page-wise Q2/Q4
oracle, compressed cache, fail-closed Qwen integration, and gfx1151 Triton backend
live under `src/uma_qmoe/ffd/`.  The official package is audited but is not installed
as a dependency and the Transformers Qwen model is not forked.

Freeze upstream and local source identities:

```bash
python benchmarks/experiments/audit_ffd_source.py \
  --official-checkout /path/to/faster-flash-decoding-at-ca09458 \
  --output artifacts/local-halo/ffd/source-audit.json
```

Capture selector fidelity from real Qwen projections.  This is a quality/oracle run,
not a performance benchmark:

```bash
python benchmarks/experiments/qwen_ffd_selector.py \
  --model /models/Qwen1.5-MoE-A2.7B \
  --expert-pack /models/UMA-QMoE/local-halo/qwen-bf16-prefix16-q8-tail8.uqtp \
  --target-policy-id qwen-bf16-prefix16-q8-tail8-v1 \
  --model-manifest-sha256 87667d3fb147eb692748ce33d0e570919281b264a3cc950b2e2e9df4dc12b13f \
  --native-build-dir /models/UMA-QMoE/local-halo/native-build-v9 \
  --prompt-fixture benchmarks/fixtures/qwen1_5_moe_completion_quality_v1.jsonl \
  --context-tokens 1024 --delta 7 --block-size 128 \
  --output artifacts/local-halo/ffd/selector-fidelity.jsonl
```

Run the synthetic matrix with the exact Qwen attention shape.  Unsupported Q4 and
block-256 candidates are emitted as explicit rejected rows rather than falling back:

```bash
python benchmarks/experiments/qwen_ffd_kernel_matrix.py \
  --output artifacts/local-halo/ffd/kernel-matrix.jsonl \
  --verify-oracle
```

`qwen_ffd_dense_baseline.py` captures the actual dense backend and component timing.
`qwen_ffd_host_compare.py` performs a same-process, same-shape warmed comparison of
dense versus FFD decode and records device/wall TPOT, token agreement, cache memory,
and audited backend calls.  It is explicitly relaxed-accuracy L0 evidence.
`benchmarks/runners/qwen_ffd_generate.py` is a diagnostic end-to-end wiring check:
prefill remains dense, every selected decode layer must call the fused FFD backend,
and unsupported inputs fail closed.  Neither runner authorizes a deployment claim;
the promotion order and stop rules are frozen in
`doc/UMA_QMOE_FASTER_FLASH_DECODING_EXPERIMENT_PLAN.md`.
