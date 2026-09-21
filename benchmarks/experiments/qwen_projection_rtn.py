#!/usr/bin/env python3
"""L0 scan of the 72 Qwen layer x projection RTN units."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from uma_qmoe.experiments.evaluate import (
    file_sha256,
    load_completion_samples,
    QualityThresholds,
    RelativeQualityEvaluator,
)
from uma_qmoe.experiments.ledger import JsonlLedger
from uma_qmoe.experiments.quantizers import RTNQuantizer
from uma_qmoe.experiments.search import collect_provenance, run_trial
from uma_qmoe.experiments.session import load_qwen_experiment
from uma_qmoe.experiments.types import CalibrationView, Encoding, TrialSpec, Unit
from uma_qmoe.fixed_models import QWEN1_5_MOE


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--model-manifest-sha256", required=True)
    parser.add_argument("--prompt-fixture", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--backend", choices=("cuda", "hip"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--storage", choices=("q4", "q8", "bf16"), default="q4")
    parser.add_argument("--group-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--maximum-units", type=int)
    parser.add_argument("--full-quality", action="store_true")
    return parser


def _units(limit: int | None) -> tuple[Unit, ...]:
    units = tuple(
        Unit(layer=layer, projection=projection)
        for layer in range(QWEN1_5_MOE.num_layers)
        for projection in ("gate", "up", "down")
    )
    return units if limit is None else units[:limit]


def main() -> int:
    args = _parser().parse_args()
    if args.maximum_units is not None and not 1 <= args.maximum_units <= 72:
        raise SystemExit("--maximum-units must be in [1, 72]")
    if len(args.model_manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in args.model_manifest_sha256
    ):
        raise SystemExit("--model-manifest-sha256 must be a SHA-256 digest")
    samples = load_completion_samples(args.prompt_fixture)
    fixture_identity = {
        "fixture_sha256": file_sha256(args.prompt_fixture),
        "sample_ids": [sample["id"] for sample in samples],
    }
    encoding = Encoding(
        storage=args.storage,
        group_size=None if args.storage == "bf16" else args.group_size,
        method="rtn",
        parameters={"symmetric": True, "rounding": "nearest_even"},
    )
    thresholds = QualityThresholds()
    specs = [
        TrialSpec(
            model_identity={
                "model_id": QWEN1_5_MOE.model_id,
                "model_revision": QWEN1_5_MOE.model_revision,
                "model_manifest_sha256": args.model_manifest_sha256,
            },
            data_identity=fixture_identity,
            unit=unit,
            encoding=encoding,
            quantizer_version=RTNQuantizer.version,
            evaluation=thresholds.to_dict(),
            seed=args.seed,
            diagnostic_only=not args.full_quality,
        )
        for unit in _units(args.maximum_units)
    ]
    ledger = JsonlLedger(args.ledger)
    pending = [spec for spec in specs if not ledger.is_complete(spec)]
    if not pending:
        print(f"all {len(specs)} RTN trials are already complete", flush=True)
        return 0

    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    runtime = load_qwen_experiment(
        args.model, backend=args.backend, device=args.device, seed=args.seed
    )
    if runtime.session.discover_units() != _units(None):
        raise RuntimeError("Qwen unit discovery order changed")
    evaluator = RelativeQualityEvaluator(
        runtime.model,
        samples,
        runtime.tokenizer,
        num_layers=QWEN1_5_MOE.num_layers,
        device=args.device,
        torch_module=runtime.torch,
        thresholds=thresholds,
    )
    calibration = CalibrationView(identity={"kind": "none", "reason": "RTN"})
    quantizer = RTNQuantizer()
    provenance = collect_provenance(Path(__file__).resolve().parents[2])
    provenance.update(
        {
            "backend": runtime.backend,
            "device": args.device,
            "gpu_name": runtime.torch.cuda.get_device_name(args.device),
            "torch": runtime.torch.__version__,
            "torch_hip": runtime.torch.version.hip,
        }
    )

    for index, spec in enumerate(pending, start=1):
        print(f"[{index}/{len(pending)}] {spec.unit}", flush=True)
        result = run_trial(
            spec=spec,
            ledger=ledger,
            session=runtime.session,
            quantizer=quantizer,
            calibration=calibration,
            evaluate=evaluator.evaluate,
            gate=evaluator.gate,
            memory_reset=runtime.reset_peak_memory,
            memory_read=runtime.memory,
            provenance=provenance,
        )
        assert result is not None
        print(f"{result.trial_id[:12]} {result.status.value}", flush=True)
        runtime.torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
