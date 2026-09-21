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
