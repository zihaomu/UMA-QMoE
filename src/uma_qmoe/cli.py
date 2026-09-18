"""Command-line interface for UMA-QMoE bootstrap tooling."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml

from .contracts import (
    ContractError,
    canonical_sha256,
    file_sha256,
    find_project_root,
    load_document,
    validate_document,
    validate_file,
)


def _write_document(document: dict[str, Any], output: str) -> None:
    # Generated evidence must never be emitted in a form that this same
    # control plane would later reject.  Validate before writing so stdout and
    # destination files remain all-or-nothing evidence boundaries.
    validate_document(document)

    if output == "-":
        json.dump(document, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
        return

    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    suffix = destination.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        rendered = yaml.safe_dump(document, sort_keys=False, allow_unicode=True)
    else:
        rendered = json.dumps(document, indent=2, sort_keys=True) + "\n"

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(rendered)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="umaq")
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate_parser = subparsers.add_parser("validate", help="validate a YAML/JSON contract")
    validate_parser.add_argument("path", type=Path)
    validate_parser.add_argument(
        "--require-frozen",
        action="store_true",
        help="reject explicit draft or pending state",
    )
    validate_parser.add_argument(
        "--no-reference-check",
        action="store_true",
        help="skip referenced model/fixture hash checks",
    )

    hash_parser = subparsers.add_parser("hash", help="print the canonical semantic SHA-256")
    hash_parser.add_argument("path", type=Path)

    collect_parser = subparsers.add_parser(
        "collect-machine", help="collect a non-sensitive machine baseline"
    )
    collect_parser.add_argument("--target-id", required=True)
    collect_parser.add_argument("--output", default="-")

    memory_parser = subparsers.add_parser(
        "probe-memory", help="collect a read-only UMA memory snapshot"
    )
    memory_parser.add_argument("--target-id", required=True)
    memory_parser.add_argument("--output", default="-")

    import_parser = subparsers.add_parser(
        "import-hf-manifest",
        help="build a draft manifest from pinned Hugging Face metadata without downloading weights",
    )
    import_parser.add_argument("--repo-id", required=True)
    import_parser.add_argument("--revision", required=True)
    import_parser.add_argument("--variant", choices=("base", "sft", "instruct"), default="base")
    import_parser.add_argument("--model-card-weight-dtype", required=True)
    import_parser.add_argument("--oracle-dtype", required=True)
    import_parser.add_argument("--declared-total-parameters")
    import_parser.add_argument("--declared-active-parameters")
    import_parser.add_argument("--output", default="-")

    fetch_parser = subparsers.add_parser(
        "fetch-model",
        help="resume-download and verify every file pinned by a ModelManifest",
    )
    fetch_parser.add_argument("manifest", type=Path)
    fetch_parser.add_argument("model_directory", type=Path)
    fetch_parser.add_argument("--jobs", type=int, default=1)
    fetch_parser.add_argument("--output", default="-")

    verify_model_parser = subparsers.add_parser(
        "verify-model",
        help="verify all local files pinned by a ModelManifest",
    )
    verify_model_parser.add_argument("manifest", type=Path)
    verify_model_parser.add_argument("model_directory", type=Path)
    verify_model_parser.add_argument("--output", default="-")

    freeze_model_parser = subparsers.add_parser(
        "freeze-model-manifest",
        help="bind a TensorInventory and transition a draft ModelManifest to frozen",
    )
    freeze_model_parser.add_argument("manifest", type=Path)
    freeze_model_parser.add_argument("inventory", type=Path)
    freeze_model_parser.add_argument("--output", required=True)

    derive_parser = subparsers.add_parser(
        "derive-bf16",
        help="derive a resumable, verified BF16 Oracle from a frozen all-F32 model",
    )
    derive_parser.add_argument("manifest", type=Path)
    derive_parser.add_argument("source_directory", type=Path)
    derive_parser.add_argument("destination_directory", type=Path)
    derive_parser.add_argument("--artifact-root", required=True)
    derive_parser.add_argument("--chunk-mib", type=int, default=64)
    derive_parser.add_argument("--output", required=True)

    verify_derivation_parser = subparsers.add_parser(
        "verify-derivation",
        help="verify every artifact pinned by a ModelDerivation on one target",
    )
    verify_derivation_parser.add_argument("derivation", type=Path)
    verify_derivation_parser.add_argument("artifact_directory", type=Path)
    verify_derivation_parser.add_argument("--target-id", required=True)
    verify_derivation_parser.add_argument("--output", default="-")

    oracle_smoke_parser = subparsers.add_parser(
        "oracle-smoke",
        help="run one offline BF16 full-model forward pass and emit bound evidence",
    )
    oracle_smoke_parser.add_argument("derivation", type=Path)
    oracle_smoke_parser.add_argument("model_directory", type=Path)
    oracle_smoke_parser.add_argument("prompt_fixture", type=Path)
    oracle_smoke_parser.add_argument("--prompt-id", required=True)
    oracle_smoke_parser.add_argument("--target-id", required=True)
    oracle_smoke_parser.add_argument("--output", default="-")

    reference_oracle_parser = subparsers.add_parser(
        "compare-reference-oracle",
        help="compare fixed single-expert, MoE-layer, or full-model tensor streams",
    )
    reference_oracle_parser.add_argument("reference_stream", type=Path)
    reference_oracle_parser.add_argument("candidate_stream", type=Path)
    reference_oracle_parser.add_argument("--oracle-id", required=True)
    reference_oracle_parser.add_argument("--model-revision", required=True)
    reference_oracle_parser.add_argument(
        "--level",
        choices=("single_expert", "single_moe_layer", "full_model"),
        required=True,
    )
    reference_oracle_parser.add_argument("--layer-index", type=int)
    reference_oracle_parser.add_argument("--expert-index", type=int)
    reference_oracle_parser.add_argument("--fixture-id", required=True)
    reference_oracle_parser.add_argument("--fixture-sha256", required=True)
    reference_oracle_parser.add_argument("--reference-id", required=True)
    reference_oracle_parser.add_argument("--reference-precision", required=True)
    reference_oracle_parser.add_argument("--reference-artifact-sha256", required=True)
    reference_oracle_parser.add_argument("--candidate-id", required=True)
    reference_oracle_parser.add_argument("--candidate-precision", required=True)
    reference_oracle_parser.add_argument("--candidate-artifact-sha256", required=True)
    reference_oracle_parser.add_argument(
        "--quality-policy",
        type=Path,
        help="optional JSON/YAML draft or frozen numerical threshold policy",
    )
    reference_oracle_parser.add_argument("--output", default="-")

    bandwidth_parser = subparsers.add_parser(
        "benchmark-memory-bandwidth",
        help="measure provisional GPU read/write/copy bandwidth on local UMA memory",
    )
    bandwidth_parser.add_argument("--target-id", required=True)
    bandwidth_parser.add_argument("--buffer-mib", type=int, default=512)
    bandwidth_parser.add_argument("--warmup", type=int, default=3)
    bandwidth_parser.add_argument("--iterations", type=int, default=10)
    bandwidth_parser.add_argument("--inner-loops", type=int, default=1)
    bandwidth_parser.add_argument(
        "--telemetry",
        action="store_true",
        help="sample target management telemetry outside each timed sample",
    )
    bandwidth_parser.add_argument("--output", default="-")

    soak_parser = subparsers.add_parser(
        "benchmark-bandwidth-soak",
        help="run a continuous read/write/copy thermal-stability soak",
    )
    soak_parser.add_argument("--target-id", required=True)
    soak_parser.add_argument("--buffer-mib", type=int, default=512)
    soak_parser.add_argument("--duration-seconds", type=int, default=1800)
    soak_parser.add_argument("--warmup-cycles", type=int, default=3)
    soak_parser.add_argument("--inner-loops", type=int, default=64)
    soak_parser.add_argument("--telemetry-interval-seconds", type=float, default=5.0)
    soak_parser.add_argument("--output", default="-")

    allocation_parser = subparsers.add_parser(
        "benchmark-allocation-matrix",
        help="measure first-touch and steady transfers across UMA allocation paths",
    )
    allocation_parser.add_argument("--target-id", required=True)
    allocation_parser.add_argument("--buffer-mib", type=int, default=256)
    allocation_parser.add_argument("--warmup", type=int, default=3)
    allocation_parser.add_argument("--iterations", type=int, default=10)
    allocation_parser.add_argument("--output", default="-")

    allocation_v2_parser = subparsers.add_parser(
        "benchmark-allocation-matrix-v2",
        help="compile and measure native UMA paths at progressive pressure points",
    )
    allocation_v2_parser.add_argument("source", type=Path)
    allocation_v2_parser.add_argument("--target-id", required=True)
    allocation_v2_parser.add_argument(
        "--backend", choices=("cuda", "hip"), required=True
    )
    allocation_v2_parser.add_argument("--arch", required=True)
    allocation_v2_parser.add_argument(
        "--buffer-mib",
        type=int,
        action="append",
        dest="buffer_mib",
        help="pressure point in MiB; repeat for a strictly increasing sequence",
    )
    allocation_v2_parser.add_argument("--warmup", type=int, default=3)
    allocation_v2_parser.add_argument("--iterations", type=int, default=10)
    allocation_v2_parser.add_argument("--inner-loops", type=int, default=1)
    allocation_v2_parser.add_argument("--maximum-cv", type=float, default=0.03)
    allocation_v2_parser.add_argument(
        "--native-timeout-seconds", type=float, default=1800.0
    )
    allocation_v2_parser.add_argument("--output", default="-")

    allocation_capability_parser = subparsers.add_parser(
        "probe-native-allocation-capabilities",
        help="compile and query native CUDA/HIP UMA allocation capabilities",
    )
    allocation_capability_parser.add_argument("source", type=Path)
    allocation_capability_parser.add_argument("--target-id", required=True)
    allocation_capability_parser.add_argument(
        "--backend", choices=("cuda", "hip"), required=True
    )
    allocation_capability_parser.add_argument("--arch", required=True)
    allocation_capability_parser.add_argument("--output", default="-")

    native_stream_parser = subparsers.add_parser(
        "benchmark-native-stream",
        help="compile and run the known-byte native CUDA/HIP streaming kernels",
    )
    native_stream_parser.add_argument("source", type=Path)
    native_stream_parser.add_argument("--target-id", required=True)
    native_stream_parser.add_argument("--backend", choices=("cuda", "hip"), required=True)
    native_stream_parser.add_argument("--arch", required=True)
    native_stream_parser.add_argument("--buffer-mib", type=int, default=512)
    native_stream_parser.add_argument("--warmup", type=int, default=3)
    native_stream_parser.add_argument("--iterations", type=int, default=10)
    native_stream_parser.add_argument("--inner-loops", type=int, default=64)
    native_stream_parser.add_argument("--output", default="-")

    counter_parser = subparsers.add_parser(
        "calibrate-rocprof-counters",
        help="normalize gfx1151 rocprofv3 counters against a known-byte native stream",
    )
    counter_parser.add_argument("source", type=Path)
    counter_parser.add_argument("profile_csv", type=Path)
    counter_parser.add_argument("--target-id", required=True)
    counter_parser.add_argument("--arch", required=True)
    counter_parser.add_argument("--profiler-version", required=True)
    counter_parser.add_argument("--known-bytes", type=int, required=True)
    counter_parser.add_argument("--maximum-relative-error", type=float, default=0.10)
    counter_parser.add_argument("--output", default="-")

    budget_parser = subparsers.add_parser(
        "build-safe-budget",
        help="calculate a Safe UMA Budget from a memory snapshot and explicit reserves",
    )
    budget_parser.add_argument("snapshot", type=Path)
    budget_parser.add_argument("--os-daemon-reserve-bytes", type=int, required=True)
    budget_parser.add_argument("--runtime-reserve-bytes", type=int, required=True)
    budget_parser.add_argument("--kv-budget-bytes", type=int, required=True)
    budget_parser.add_argument("--workspace-budget-bytes", type=int, required=True)
    budget_parser.add_argument("--safety-margin-bytes", type=int, required=True)
    budget_parser.add_argument("--status", choices=("draft", "frozen"), default="draft")
    budget_parser.add_argument("--policy-provenance", default="")
    budget_parser.add_argument("--decision-record", default="")
    budget_parser.add_argument("--output", default="-")

    public_baseline_parser = subparsers.add_parser(
        "normalize-public-baseline",
        help="normalize one offline vLLM serving run into baseline evidence",
    )
    public_baseline_parser.add_argument("run_directory", type=Path)
    public_baseline_parser.add_argument("--target-id", required=True)
    public_baseline_parser.add_argument("--implementation-version", required=True)
    public_baseline_parser.add_argument("--source-commit", required=True)
    public_baseline_parser.add_argument(
        "--backend", choices=("cuda", "hip"), required=True
    )
    public_baseline_parser.add_argument("--container-image", required=True)
    public_baseline_parser.add_argument("--model-id", required=True)
    public_baseline_parser.add_argument("--model-revision", required=True)
    public_baseline_parser.add_argument("--derivation-semantic-sha256", required=True)
    public_baseline_parser.add_argument("--input-tokens", type=int, required=True)
    public_baseline_parser.add_argument("--output-tokens", type=int, required=True)
    public_baseline_parser.add_argument("--warmup-requests", type=int, required=True)
    public_baseline_parser.add_argument("--measured-requests", type=int, required=True)
    public_baseline_parser.add_argument("--max-concurrency", type=int, required=True)
    public_baseline_parser.add_argument("--output", default="-")

    baseline_run_parser = subparsers.add_parser(
        "build-public-baseline-run-manifest",
        help="bind normalized baseline evidence and raw files into RunManifest v1",
    )
    baseline_run_parser.add_argument("baseline", type=Path)
    baseline_run_parser.add_argument("run_directory", type=Path)
    baseline_run_parser.add_argument("--run-id", required=True)
    baseline_run_parser.add_argument("--git-commit", required=True)
    baseline_run_parser.add_argument("--git-dirty", action="store_true")
    baseline_run_parser.add_argument("--dirty-patch-sha256")
    baseline_run_parser.add_argument("--machine-baseline-sha256", required=True)
    baseline_run_parser.add_argument("--benchmark-contract-sha256", required=True)
    baseline_run_parser.add_argument("--model-manifest-sha256", required=True)
    baseline_run_parser.add_argument("--output", default="-")

    external_baseline_parser = subparsers.add_parser(
        "normalize-external-baseline",
        help="normalize an isolated third-party run into ExternalBaseline v1",
    )
    external_baseline_parser.add_argument("run_directory", type=Path)
    external_baseline_parser.add_argument("--target-id", required=True)
    external_baseline_parser.add_argument("--implementation-name", required=True)
    external_baseline_parser.add_argument("--implementation-version", required=True)
    external_baseline_parser.add_argument("--source-commit", required=True)
    external_baseline_parser.add_argument(
        "--backend", choices=("cuda", "hip", "cpu"), required=True
    )
    external_baseline_parser.add_argument("--container-image", required=True)
    external_baseline_parser.add_argument("--model-id", required=True)
    external_baseline_parser.add_argument("--model-revision", required=True)
    external_baseline_parser.add_argument(
        "--derivation-semantic-sha256", required=True
    )
    external_baseline_parser.add_argument("--input-tokens", type=int, required=True)
    external_baseline_parser.add_argument("--output-tokens", type=int, required=True)
    external_baseline_parser.add_argument(
        "--warmup-requests", type=int, required=True
    )
    external_baseline_parser.add_argument(
        "--measured-requests", type=int, required=True
    )
    external_baseline_parser.add_argument(
        "--max-concurrency", type=int, required=True
    )
    external_baseline_parser.add_argument("--output", default="-")

    external_run_parser = subparsers.add_parser(
        "build-external-baseline-run-manifest",
        help="bind ExternalBaseline evidence and raw files into RunManifest v1",
    )
    external_run_parser.add_argument("baseline", type=Path)
    external_run_parser.add_argument("run_directory", type=Path)
    external_run_parser.add_argument("--run-id", required=True)
    external_run_parser.add_argument("--git-commit", required=True)
    external_run_parser.add_argument("--git-dirty", action="store_true")
    external_run_parser.add_argument("--dirty-patch-sha256")
    external_run_parser.add_argument("--machine-baseline-sha256", required=True)
    external_run_parser.add_argument("--benchmark-contract-sha256", required=True)
    external_run_parser.add_argument("--model-manifest-sha256", required=True)
    external_run_parser.add_argument("--output", default="-")

    reference_host_parser = subparsers.add_parser(
        "normalize-reference-host-baseline",
        help="normalize the fixed offline PyTorch/HF host baseline",
    )
    reference_host_parser.add_argument("run_directory", type=Path)
    reference_host_parser.add_argument("--target-id", required=True)
    reference_host_parser.add_argument("--source-commit", required=True)
    reference_host_parser.add_argument(
        "--backend", choices=("cuda", "hip"), required=True
    )
    reference_host_parser.add_argument("--container-image", required=True)
    reference_host_parser.add_argument("--model-id", required=True)
    reference_host_parser.add_argument("--model-revision", required=True)
    reference_host_parser.add_argument(
        "--derivation-semantic-sha256", required=True
    )
    reference_host_parser.add_argument("--input-tokens", type=int, required=True)
    reference_host_parser.add_argument("--output-tokens", type=int, required=True)
    reference_host_parser.add_argument("--warmup-requests", type=int, required=True)
    reference_host_parser.add_argument("--measured-requests", type=int, required=True)
    reference_host_parser.add_argument("--output", default="-")

    reference_host_run_parser = subparsers.add_parser(
        "build-reference-host-run-manifest",
        help="bind ReferenceHostBaseline evidence and raw files into RunManifest v1",
    )
    reference_host_run_parser.add_argument("baseline", type=Path)
    reference_host_run_parser.add_argument("run_directory", type=Path)
    reference_host_run_parser.add_argument("--run-id", required=True)
    reference_host_run_parser.add_argument("--git-commit", required=True)
    reference_host_run_parser.add_argument("--git-dirty", action="store_true")
    reference_host_run_parser.add_argument("--dirty-patch-sha256")
    reference_host_run_parser.add_argument("--machine-baseline-sha256", required=True)
    reference_host_run_parser.add_argument("--benchmark-contract-sha256", required=True)
    reference_host_run_parser.add_argument("--model-manifest-sha256", required=True)
    reference_host_run_parser.add_argument("--output", default="-")

    route_trace_parser = subparsers.add_parser(
        "build-route-trace",
        help="normalize a fixed OLMoE capture into frozen RouteTrace v1",
    )
    route_trace_parser.add_argument("capture", type=Path)
    route_trace_parser.add_argument("--trace-id", required=True)
    route_trace_parser.add_argument("--output", default="-")

    replay_trace_parser = subparsers.add_parser(
        "replay-route-trace",
        help="validate and fully traverse a RouteTrace in replay order",
    )
    replay_trace_parser.add_argument("trace", type=Path)
    replay_trace_parser.add_argument("--output", default="-")

    inspect_pack_parser = subparsers.add_parser(
        "inspect-expert-pack",
        help="verify every ExpertPack header, identity, layout, and payload hash",
    )
    inspect_pack_parser.add_argument("pack", type=Path)
    inspect_pack_parser.add_argument("--output", default="-")

    inventory_parser = subparsers.add_parser(
        "inventory-safetensors",
        help="stream-verify local Safetensors shards and hash every tensor payload",
    )
    inventory_parser.add_argument("manifest", type=Path)
    inventory_parser.add_argument("model_directory", type=Path)
    inventory_parser.add_argument("--allow-unlisted", action="store_true")
    inventory_parser.add_argument("--output", default="-")

    traffic_parser = subparsers.add_parser(
        "estimate-weight-traffic",
        help="estimate packed model storage and batch-1 cold weight bytes per token",
    )
    traffic_parser.add_argument("manifest", type=Path)
    traffic_parser.add_argument("inventory", type=Path)
    traffic_parser.add_argument("--dense-bits", type=int, default=16)
    traffic_parser.add_argument("--expert-bits", type=int, default=4)
    traffic_parser.add_argument("--group-size", type=int, default=128)
    traffic_parser.add_argument("--scale-bytes", type=int, default=2)
    traffic_parser.add_argument("--zero-point-bytes", type=int, default=0)
    traffic_parser.add_argument("--tensor-alignment", type=int, default=128)
    traffic_parser.add_argument("--output", default="-")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    arguments = parser.parse_args(argv)
    try:
        if arguments.command == "validate":
            document = validate_file(
                arguments.path,
                require_frozen=arguments.require_frozen,
                check_references=not arguments.no_reference_check,
            )
            print(
                json.dumps(
                    {
                        "kind": document["kind"],
                        "path": str(arguments.path),
                        "semantic_sha256": canonical_sha256(document),
                        "status": "valid",
                    },
                    sort_keys=True,
                )
            )
            return 0
        if arguments.command == "hash":
            document = load_document(arguments.path)
            print(canonical_sha256(document))
            return 0
        if arguments.command == "collect-machine":
            from .machine import collect_machine_baseline

            document = collect_machine_baseline(arguments.target_id)
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "probe-memory":
            from .memory import collect_memory_snapshot

            document = collect_memory_snapshot(arguments.target_id)
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "import-hf-manifest":
            from .huggingface import build_model_manifest_draft

            document = build_model_manifest_draft(
                arguments.repo_id,
                arguments.revision,
                variant=arguments.variant,
                model_card_uploaded_dtype=arguments.model_card_weight_dtype,
                oracle_dtype=arguments.oracle_dtype,
                declared_total_parameters=arguments.declared_total_parameters,
                declared_active_parameters=arguments.declared_active_parameters,
            )
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "fetch-model":
            from .model_store import fetch_model_files

            manifest = validate_file(arguments.manifest)
            document = fetch_model_files(
                manifest,
                arguments.model_directory,
                jobs=arguments.jobs,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
            )
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "verify-model":
            from .model_store import verify_model_files

            manifest = validate_file(arguments.manifest)
            document = verify_model_files(manifest, arguments.model_directory)
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "freeze-model-manifest":
            from .model_manifest import freeze_model_manifest

            if arguments.output == "-":
                raise ContractError(
                    "freeze-model-manifest requires a file --output for reference checks"
                )
            manifest_path = arguments.manifest.resolve()
            inventory_path = arguments.inventory.resolve()
            project_root = find_project_root(manifest_path)
            try:
                inventory_relative = inventory_path.relative_to(project_root).as_posix()
            except ValueError as exc:
                raise ContractError("TensorInventory must be inside the project root") from exc
            output_path = Path(arguments.output).resolve()
            try:
                output_path.relative_to(project_root)
            except ValueError as exc:
                raise ContractError(
                    "frozen ModelManifest output must be inside the project root"
                ) from exc
            manifest = validate_file(manifest_path)
            inventory = validate_file(inventory_path)
            document = freeze_model_manifest(
                manifest,
                inventory,
                inventory_path=inventory_relative,
                inventory_file_sha256=file_sha256(inventory_path),
            )
            _write_document(document, str(output_path))
            validate_file(output_path, require_frozen=True)
            return 0
        if arguments.command == "derive-bf16":
            from .bf16 import derive_bf16_oracle

            if arguments.output == "-":
                raise ContractError(
                    "derive-bf16 requires a file --output so references can be validated"
                )
            manifest_path = arguments.manifest.resolve()
            manifest = validate_file(manifest_path, require_frozen=True)
            project_root = find_project_root(manifest_path)
            try:
                manifest_relative = manifest_path.relative_to(project_root).as_posix()
            except ValueError as exc:
                raise ContractError("model manifest must be inside the project root") from exc
            output_path = Path(arguments.output).resolve()
            try:
                output_path.relative_to(project_root)
            except ValueError as exc:
                raise ContractError(
                    "ModelDerivation output must be inside the project root"
                ) from exc
            inventory_reference = manifest["weights"]["tensor_inventory"]
            inventory_path = project_root.joinpath(
                *inventory_reference["path"].split("/")
            )
            inventory = validate_file(inventory_path)
            existing_derivation = (
                validate_file(output_path) if output_path.is_file() else None
            )
            document = derive_bf16_oracle(
                manifest,
                inventory,
                arguments.source_directory,
                arguments.destination_directory,
                source_manifest_path=manifest_relative,
                source_manifest_file_sha256=file_sha256(manifest_path),
                tensor_inventory_path=inventory_reference["path"],
                tensor_inventory_file_sha256=file_sha256(inventory_path),
                artifact_root=arguments.artifact_root,
                chunk_bytes=arguments.chunk_mib * 1024 * 1024,
                progress=lambda message: print(message, file=sys.stderr, flush=True),
                existing_derivation=existing_derivation,
            )
            _write_document(document, str(output_path))
            validate_file(output_path)
            return 0
        if arguments.command == "verify-derivation":
            from .model_store import verify_derivation_files

            derivation_path = arguments.derivation.resolve()
            project_root = find_project_root(derivation_path)
            try:
                derivation_relative = derivation_path.relative_to(project_root).as_posix()
            except ValueError as exc:
                raise ContractError("ModelDerivation must be inside the project root") from exc
            derivation = validate_file(derivation_path)
            document = verify_derivation_files(
                derivation,
                arguments.artifact_directory,
                target_id=arguments.target_id,
                source_contract_path=derivation_relative,
                source_contract_sha256=file_sha256(derivation_path),
            )
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "oracle-smoke":
            from .oracle_smoke import run_oracle_smoke

            derivation_path = arguments.derivation.resolve()
            prompt_fixture_path = arguments.prompt_fixture.resolve()
            project_root = find_project_root(derivation_path)
            try:
                derivation_relative = derivation_path.relative_to(project_root).as_posix()
                prompt_fixture_relative = prompt_fixture_path.relative_to(
                    project_root
                ).as_posix()
            except ValueError as exc:
                raise ContractError(
                    "ModelDerivation and prompt fixture must be inside one project root"
                ) from exc
            derivation = validate_file(derivation_path)
            document = run_oracle_smoke(
                derivation,
                arguments.model_directory,
                prompt_fixture_path,
                target_id=arguments.target_id,
                prompt_id=arguments.prompt_id,
                derivation_path=derivation_relative,
                derivation_file_sha256=file_sha256(derivation_path),
                prompt_fixture_path=prompt_fixture_relative,
                prompt_fixture_file_sha256=file_sha256(prompt_fixture_path),
            )
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "compare-reference-oracle":
            from .reference_oracle import build_reference_oracle_comparison

            quality_policy = None
            if arguments.quality_policy is not None:
                policy_document = load_document(arguments.quality_policy)
                if policy_document.get("kind") == "reference_oracle_policy":
                    validate_document(policy_document)
                    requested_scope = {
                        "level": arguments.level,
                        "layer_index": arguments.layer_index,
                        "expert_index": arguments.expert_index,
                    }
                    if policy_document["scope"] != requested_scope:
                        raise ContractError(
                            "ReferenceOraclePolicy scope does not match comparison"
                        )
                    if policy_document["model_revision"] != arguments.model_revision:
                        raise ContractError(
                            "ReferenceOraclePolicy model revision does not match comparison"
                        )
                    quality_policy = policy_document["policy"]
                else:
                    quality_policy = policy_document
            document = build_reference_oracle_comparison(
                arguments.reference_stream,
                arguments.candidate_stream,
                oracle_id=arguments.oracle_id,
                model_revision=arguments.model_revision,
                scope={
                    "level": arguments.level,
                    "layer_index": arguments.layer_index,
                    "expert_index": arguments.expert_index,
                },
                fixture_id=arguments.fixture_id,
                fixture_sha256=arguments.fixture_sha256,
                reference_implementation={
                    "id": arguments.reference_id,
                    "precision": arguments.reference_precision,
                    "artifact_sha256": arguments.reference_artifact_sha256,
                },
                candidate_implementation={
                    "id": arguments.candidate_id,
                    "precision": arguments.candidate_precision,
                    "artifact_sha256": arguments.candidate_artifact_sha256,
                },
                quality_policy=quality_policy,
            )
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "benchmark-memory-bandwidth":
            from .memory_bandwidth import benchmark_memory_bandwidth

            document = benchmark_memory_bandwidth(
                target_id=arguments.target_id,
                requested_buffer_bytes=arguments.buffer_mib * 1024 * 1024,
                warmup_iterations=arguments.warmup,
                measured_iterations=arguments.iterations,
                inner_iterations=arguments.inner_loops,
                collect_telemetry=arguments.telemetry,
            )
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "benchmark-bandwidth-soak":
            from .bandwidth_soak import benchmark_bandwidth_soak

            document = benchmark_bandwidth_soak(
                target_id=arguments.target_id,
                requested_buffer_bytes=arguments.buffer_mib * 1024 * 1024,
                requested_duration_seconds=arguments.duration_seconds,
                warmup_cycles=arguments.warmup_cycles,
                inner_iterations=arguments.inner_loops,
                telemetry_interval_seconds=arguments.telemetry_interval_seconds,
            )
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "benchmark-allocation-matrix":
            from .allocation_matrix import benchmark_allocation_matrix

            document = benchmark_allocation_matrix(
                target_id=arguments.target_id,
                requested_buffer_bytes=arguments.buffer_mib * 1024 * 1024,
                warmup_iterations=arguments.warmup,
                measured_iterations=arguments.iterations,
            )
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "benchmark-allocation-matrix-v2":
            from .allocation_matrix_v2 import benchmark_allocation_matrix_v2

            source_path = arguments.source.resolve()
            project_root = find_project_root(source_path)
            try:
                source_relative = source_path.relative_to(project_root).as_posix()
            except ValueError as exc:
                raise ContractError(
                    "Allocation Matrix v2 source must be inside project root"
                ) from exc
            buffer_mib = arguments.buffer_mib or [256, 1024, 4096, 8192]
            document = benchmark_allocation_matrix_v2(
                source_path,
                source_relative_path=source_relative,
                source_file_sha256=file_sha256(source_path),
                target_id=arguments.target_id,
                backend=arguments.backend,
                architecture=arguments.arch,
                buffer_sizes_bytes=[value * 1024 * 1024 for value in buffer_mib],
                warmup_iterations=arguments.warmup,
                measured_iterations=arguments.iterations,
                inner_iterations=arguments.inner_loops,
                maximum_coefficient_of_variation=arguments.maximum_cv,
                native_timeout_seconds=arguments.native_timeout_seconds,
            )
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "probe-native-allocation-capabilities":
            from .native_allocation import probe_native_allocation_capabilities

            source_path = arguments.source.resolve()
            project_root = find_project_root(source_path)
            try:
                source_relative = source_path.relative_to(project_root).as_posix()
            except ValueError as exc:
                raise ContractError(
                    "native allocation source must be inside project root"
                ) from exc
            document = probe_native_allocation_capabilities(
                source_path,
                source_relative_path=source_relative,
                source_file_sha256=file_sha256(source_path),
                target_id=arguments.target_id,
                backend=arguments.backend,
                architecture=arguments.arch,
            )
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "benchmark-native-stream":
            from .native_stream import benchmark_native_stream

            source_path = arguments.source.resolve()
            project_root = find_project_root(source_path)
            try:
                source_relative = source_path.relative_to(project_root).as_posix()
            except ValueError as exc:
                raise ContractError("native stream source must be inside project root") from exc
            document = benchmark_native_stream(
                source_path,
                source_relative_path=source_relative,
                source_file_sha256=file_sha256(source_path),
                target_id=arguments.target_id,
                backend=arguments.backend,
                architecture=arguments.arch,
                requested_buffer_bytes=arguments.buffer_mib * 1024 * 1024,
                warmup_iterations=arguments.warmup,
                measured_iterations=arguments.iterations,
                inner_iterations=arguments.inner_loops,
            )
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "calibrate-rocprof-counters":
            from .counter_calibration import calibrate_rocprof_counters

            source_path = arguments.source.resolve()
            project_root = find_project_root(source_path)
            try:
                source_relative = source_path.relative_to(project_root).as_posix()
            except ValueError as exc:
                raise ContractError("native stream source must be inside project root") from exc
            document = calibrate_rocprof_counters(
                arguments.profile_csv,
                source_relative_path=source_relative,
                source_file_sha256=file_sha256(source_path),
                target_id=arguments.target_id,
                architecture=arguments.arch,
                profiler_version=arguments.profiler_version,
                known_bytes_per_dispatch=arguments.known_bytes,
                maximum_relative_error=arguments.maximum_relative_error,
            )
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "build-safe-budget":
            from .safe_budget import build_safe_uma_budget

            snapshot = validate_file(arguments.snapshot)
            document = build_safe_uma_budget(
                snapshot,
                policy={
                    "os_daemon_reserve_bytes": arguments.os_daemon_reserve_bytes,
                    "runtime_reserve_bytes": arguments.runtime_reserve_bytes,
                    "kv_budget_bytes": arguments.kv_budget_bytes,
                    "workspace_budget_bytes": arguments.workspace_budget_bytes,
                    "safety_margin_bytes": arguments.safety_margin_bytes,
                },
                status=arguments.status,
                policy_provenance=arguments.policy_provenance,
                decision_record=arguments.decision_record,
            )
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "normalize-public-baseline":
            from .public_baseline import build_public_baseline

            document = build_public_baseline(
                arguments.run_directory,
                target_id=arguments.target_id,
                implementation_version=arguments.implementation_version,
                source_commit=arguments.source_commit,
                backend=arguments.backend,
                container_image=arguments.container_image,
                model_id=arguments.model_id,
                model_revision=arguments.model_revision,
                derivation_semantic_sha256=arguments.derivation_semantic_sha256,
                input_tokens=arguments.input_tokens,
                output_tokens=arguments.output_tokens,
                warmup_requests=arguments.warmup_requests,
                measured_requests=arguments.measured_requests,
                max_concurrency=arguments.max_concurrency,
            )
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "build-public-baseline-run-manifest":
            from .public_baseline import build_public_baseline_run_manifest

            document = build_public_baseline_run_manifest(
                arguments.baseline,
                arguments.run_directory,
                run_id=arguments.run_id,
                git_commit=arguments.git_commit,
                git_dirty=arguments.git_dirty,
                dirty_patch_sha256=arguments.dirty_patch_sha256,
                machine_baseline_sha256=arguments.machine_baseline_sha256,
                benchmark_contract_sha256=arguments.benchmark_contract_sha256,
                model_manifest_sha256=arguments.model_manifest_sha256,
            )
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "normalize-external-baseline":
            from .external_baseline import build_external_baseline

            document = build_external_baseline(
                arguments.run_directory,
                target_id=arguments.target_id,
                implementation_name=arguments.implementation_name,
                implementation_version=arguments.implementation_version,
                source_commit=arguments.source_commit,
                backend=arguments.backend,
                container_image=arguments.container_image,
                model_id=arguments.model_id,
                model_revision=arguments.model_revision,
                derivation_semantic_sha256=arguments.derivation_semantic_sha256,
                input_tokens=arguments.input_tokens,
                output_tokens=arguments.output_tokens,
                warmup_requests=arguments.warmup_requests,
                measured_requests=arguments.measured_requests,
                max_concurrency=arguments.max_concurrency,
            )
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "build-external-baseline-run-manifest":
            from .external_baseline import build_external_baseline_run_manifest

            document = build_external_baseline_run_manifest(
                arguments.baseline,
                arguments.run_directory,
                run_id=arguments.run_id,
                git_commit=arguments.git_commit,
                git_dirty=arguments.git_dirty,
                dirty_patch_sha256=arguments.dirty_patch_sha256,
                machine_baseline_sha256=arguments.machine_baseline_sha256,
                benchmark_contract_sha256=arguments.benchmark_contract_sha256,
                model_manifest_sha256=arguments.model_manifest_sha256,
            )
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "normalize-reference-host-baseline":
            from .reference_host_baseline import build_reference_host_baseline

            document = build_reference_host_baseline(
                arguments.run_directory,
                target_id=arguments.target_id,
                source_commit=arguments.source_commit,
                backend=arguments.backend,
                container_image=arguments.container_image,
                model_id=arguments.model_id,
                model_revision=arguments.model_revision,
                derivation_semantic_sha256=arguments.derivation_semantic_sha256,
                input_tokens=arguments.input_tokens,
                output_tokens=arguments.output_tokens,
                warmup_requests=arguments.warmup_requests,
                measured_requests=arguments.measured_requests,
            )
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "build-reference-host-run-manifest":
            from .reference_host_baseline import build_reference_host_run_manifest

            document = build_reference_host_run_manifest(
                arguments.baseline,
                arguments.run_directory,
                run_id=arguments.run_id,
                git_commit=arguments.git_commit,
                git_dirty=arguments.git_dirty,
                dirty_patch_sha256=arguments.dirty_patch_sha256,
                machine_baseline_sha256=arguments.machine_baseline_sha256,
                benchmark_contract_sha256=arguments.benchmark_contract_sha256,
                model_manifest_sha256=arguments.model_manifest_sha256,
            )
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "build-route-trace":
            from .route_trace import build_route_trace

            document = build_route_trace(
                arguments.capture,
                trace_id=arguments.trace_id,
            )
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "replay-route-trace":
            from .route_trace import build_replay_summary

            trace = validate_file(arguments.trace, require_frozen=True)
            document = build_replay_summary(trace)
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "inspect-expert-pack":
            from .expert_pack import inspect_expert_pack

            document = inspect_expert_pack(arguments.pack)
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "inventory-safetensors":
            from .safetensors_inventory import verify_model_manifest_artifacts

            manifest = validate_file(arguments.manifest)
            result = verify_model_manifest_artifacts(
                manifest,
                arguments.model_directory,
                reject_unlisted=not arguments.allow_unlisted,
            )
            document = {
                "schema_version": 1,
                "kind": "tensor_inventory",
                "generated_at": datetime.now(timezone.utc)
                .isoformat(timespec="seconds")
                .replace("+00:00", "Z"),
                "model_manifest_sha256": canonical_sha256(manifest),
                "tensor_payload_hash_algorithm": "sha256",
                "shards": result["shards"],
                "tensor_count": result["tensor_count"],
                "observed_dtypes": result["observed_dtypes"],
            }
            _write_document(document, arguments.output)
            return 0
        if arguments.command == "estimate-weight-traffic":
            from .weight_traffic import build_weight_traffic_estimate

            manifest_path = arguments.manifest.resolve()
            inventory_path = arguments.inventory.resolve()
            project_root = find_project_root(manifest_path)
            try:
                manifest_relative = manifest_path.relative_to(project_root).as_posix()
                inventory_relative = inventory_path.relative_to(project_root).as_posix()
            except ValueError as exc:
                raise ContractError(
                    "ModelManifest and TensorInventory must be inside one project root"
                ) from exc
            manifest = validate_file(manifest_path, require_frozen=True)
            inventory = validate_file(inventory_path)
            document = build_weight_traffic_estimate(
                manifest,
                inventory,
                model_manifest_path=manifest_relative,
                tensor_inventory_path=inventory_relative,
                tensor_inventory_sha256=file_sha256(inventory_path),
                dense_bits_per_element=arguments.dense_bits,
                expert_bits_per_element=arguments.expert_bits,
                group_size=arguments.group_size,
                scale_bytes_per_group=arguments.scale_bytes,
                zero_point_bytes_per_group=arguments.zero_point_bytes,
                tensor_alignment_bytes=arguments.tensor_alignment,
            )
            output_path: Path | None = None
            if arguments.output != "-":
                output_path = Path(arguments.output).resolve()
                try:
                    output_path.relative_to(project_root)
                except ValueError as exc:
                    raise ContractError(
                        "WeightTrafficEstimate file output must be inside the project root"
                    ) from exc
            _write_document(document, arguments.output)
            if output_path is not None:
                validate_file(output_path)
            return 0
    except (ContractError, OSError, ValueError) as exc:
        print(f"umaq: error: {exc}", file=sys.stderr)
        return 2
    parser.error(f"unknown command {arguments.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
