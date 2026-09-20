from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from uma_qmoe.contracts import (
    ContractError,
    _validate_benchmark_references,
    canonical_sha256,
    load_document,
    validate_document,
    validate_file,
)
from uma_qmoe.machine import collect_machine_baseline


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models/manifests/olmoe_1b_7b_0125.yaml"
QWEN_MODEL_PATH = PROJECT_ROOT / "models/manifests/qwen1_5_moe_a2_7b.yaml"
CONTRACT_PATH = PROJECT_ROOT / "benchmarks/contracts/olmoe_1b_7b_0125.yaml"
QWEN_CONTRACT_PATH = PROJECT_ROOT / "benchmarks/contracts/qwen1_5_moe_a2_7b.yaml"


def _draft_model_fixture() -> dict:
    manifest = load_document(MODEL_PATH)
    manifest["status"] = "draft"
    manifest["dtypes"]["local_tensor_scan"] = "pending_download"
    manifest["dtypes"].pop("observed_tensor_dtypes", None)
    manifest["weights"]["hash_source"] = "huggingface_lfs_metadata"
    manifest["weights"]["local_verification"] = "pending_download"
    manifest["weights"]["tensor_hashes_status"] = "pending_local_scan"
    manifest["weights"].pop("tensor_inventory", None)
    return manifest


def test_pinned_olmoe_manifest_matches_architecture_golden_values() -> None:
    manifest = validate_file(MODEL_PATH)

    assert manifest["status"] == "frozen"
    assert manifest["model_revision"] == "9b0c1aa87e34a20052389dce1f0cf01da783f654"
    assert manifest["variant"] == "base"
    assert manifest["architecture"] == {
        "class_name": "OlmoeForCausalLM",
        "model_type": "olmoe",
        "num_layers": 16,
        "hidden_size": 2048,
        "expert_intermediate_size": 1024,
        "num_experts": 64,
        "top_k": 8,
        "max_position_embeddings": 4096,
        "normalize_top_k_probability": False,
        "declared_total_parameters": "7B",
        "declared_active_parameters": "1.3B",
    }
    assert manifest["dtypes"]["config_declared"] == "float32"
    assert manifest["dtypes"]["uploaded_weights"] == "float32"
    assert manifest["dtypes"]["api_reported_parameter_counts"] == {"F32": 6919161856}
    assert manifest["dtypes"]["model_card_uploaded_weights_claim"] == "bfloat16"
    assert manifest["dtypes"]["evidence_conflict"] is True


def test_pinned_qwen_moe_manifest_matches_architecture_golden_values() -> None:
    manifest = validate_file(QWEN_MODEL_PATH, require_frozen=True)

    assert manifest["status"] == "frozen"
    assert manifest["model_revision"] == "1a758c50ecb6350748b9ce0a99d2352fd9fc11c9"
    assert manifest["architecture"] == {
        "class_name": "Qwen2MoeForCausalLM",
        "model_type": "qwen2_moe",
        "num_layers": 24,
        "hidden_size": 2048,
        "expert_intermediate_size": 1408,
        "shared_expert_intermediate_size": 5632,
        "num_experts": 60,
        "top_k": 4,
        "max_position_embeddings": 8192,
        "normalize_top_k_probability": False,
        "declared_total_parameters": "14.3B",
        "declared_active_parameters": "2.7B",
    }
    assert manifest["dtypes"]["uploaded_weights"] == "bfloat16"
    assert manifest["dtypes"]["api_reported_parameter_counts"] == {
        "BF16": 14_315_784_192
    }
    assert manifest["dtypes"]["observed_tensor_dtypes"] == ["BF16"]
    assert manifest["weights"]["tensor_inventory"] == {
        "path": "models/inventories/qwen1_5_moe_a2_7b_bf16.json",
        "sha256": "24d6ea5f730c73376103cda89057230f4851c0df0b5e5b140d263f295c8b6f3d",
        "source_manifest_sha256": (
            "4177deb25880512317f919596e66107acb7b3d5ced49788c2f885ecb75553fdc"
        ),
    }
    assert manifest["dtypes"]["evidence_conflict"] is False
    assert len(manifest["weights"]["artifacts"]) == 9


def test_semantic_hash_is_stable_across_key_order_and_resolution_time() -> None:
    manifest = load_document(MODEL_PATH)
    reordered = dict(reversed(list(manifest.items())))
    reordered["provenance"] = dict(manifest["provenance"])
    reordered["provenance"]["resolved_at"] = "2030-01-01T00:00:00Z"

    assert canonical_sha256(reordered) == canonical_sha256(manifest)

    changed = copy.deepcopy(manifest)
    changed["architecture"]["top_k"] = 4
    assert canonical_sha256(changed) != canonical_sha256(manifest)


def test_draft_contract_validates_all_references_but_not_frozen_gate() -> None:
    contract = validate_file(CONTRACT_PATH)

    assert contract["status"] == "draft"
    assert contract["quality_gates"][0]["max_relative_nll_ppl_increase"] == 0.01
    assert contract["quality_gates"][1]["max_normalized_task_score_drop_points"] == 0.5
    assert contract["quality_gates"][2]["min_router_top_k_set_agreement"] == 0.99
    with pytest.raises(ContractError, match="draft"):
        validate_file(CONTRACT_PATH, require_frozen=True)


def test_qwen_draft_contract_binds_uploaded_bf16_model_manifest() -> None:
    contract = validate_file(QWEN_CONTRACT_PATH)

    assert contract["schema_version"] == 2
    assert contract["status"] == "draft"
    assert [target["id"] for target in contract["targets"]] == ["spark1"]
    assert contract["resource_gates"]["safe_uma_budget"]["status"] == "frozen"
    assert contract["oracle"]["weight_source"] == {
        "kind": "model_manifest",
        "path": "models/manifests/qwen1_5_moe_a2_7b.yaml",
        "sha256": "bfec79ba70374fc738221e70dd89e2f9bbab8c4305487835ea2e89ba35130038",
    }
    assert "derivation" not in contract["oracle"]
    with pytest.raises(ContractError, match="draft"):
        validate_file(QWEN_CONTRACT_PATH, require_frozen=True)


def test_qwen_contract_rejects_ambiguous_or_wrong_weight_source() -> None:
    ambiguous = load_document(QWEN_CONTRACT_PATH)
    ambiguous["oracle"]["derivation"] = {
        "path": "models/manifests/olmoe_1b_7b_0125_bf16_oracle.yaml",
        "sha256": "d" * 64,
    }
    with pytest.raises(ContractError):
        validate_document(ambiguous)

    wrong_kind = load_document(QWEN_CONTRACT_PATH)
    wrong_kind["oracle"]["weight_source"]["kind"] = "model_derivation"
    with pytest.raises(ContractError, match="must reference a ModelDerivation"):
        _validate_benchmark_references(
            wrong_kind, QWEN_CONTRACT_PATH, require_frozen=False
        )


def _contract_ready_for_frozen_gate_tests() -> dict:
    contract = load_document(CONTRACT_PATH)
    contract["status"] = "frozen"
    contract["blocking_items"] = []
    contract["oracle"]["derivation"] = {
        "path": "models/manifests/oracle-derivation.yaml",
        "sha256": "d" * 64,
    }
    contract["resource_gates"]["safe_uma_budget"] = {
        "status": "frozen",
        "references": [
            {
                "target_id": target["id"],
                "path": f"benchmarks/budgets/{target['id']}.json",
                "sha256": str(index + 1) * 64,
            }
            for index, target in enumerate(contract["targets"])
        ],
    }
    for gate in contract["quality_gates"]:
        gate["applicability"] = "not_applicable"
        gate["not_applicable_reason"] = "test fixture"
    return contract


def test_benchmark_contract_rejects_duplicate_quality_gate_ids() -> None:
    contract = load_document(CONTRACT_PATH)
    contract["quality_gates"].append(copy.deepcopy(contract["quality_gates"][0]))

    with pytest.raises(ContractError, match="duplicate gate ids"):
        validate_document(contract)


def test_frozen_benchmark_requires_frozen_safe_uma_budget() -> None:
    contract = _contract_ready_for_frozen_gate_tests()
    contract["resource_gates"]["safe_uma_budget"] = {
        "status": "pending_measurement",
        "references": [],
    }

    with pytest.raises(ContractError, match="frozen Safe UMA Budget"):
        validate_document(contract, require_frozen=True)

    missing_target = _contract_ready_for_frozen_gate_tests()
    missing_target["resource_gates"]["safe_uma_budget"]["references"].pop()
    with pytest.raises(ContractError, match="one Safe UMA Budget reference per target"):
        validate_document(missing_target, require_frozen=True)


def test_frozen_benchmark_requires_hash_bound_oracle_derivation() -> None:
    contract = _contract_ready_for_frozen_gate_tests()
    contract["oracle"].pop("derivation")

    with pytest.raises(ContractError, match="derivation"):
        validate_document(contract, require_frozen=True)


def test_frozen_benchmark_semantics_accept_complete_budget_references() -> None:
    validate_document(_contract_ready_for_frozen_gate_tests(), require_frozen=True)


def test_required_quality_gate_requires_threshold_and_evidence() -> None:
    no_threshold = _contract_ready_for_frozen_gate_tests()
    gate = no_threshold["quality_gates"][3]
    gate["applicability"] = "required"
    with pytest.raises(ContractError, match="no quantitative threshold"):
        validate_document(no_threshold, require_frozen=True)

    no_dataset = _contract_ready_for_frozen_gate_tests()
    gate = no_dataset["quality_gates"][0]
    gate["applicability"] = "required"
    with pytest.raises(ContractError, match="pinned dataset"):
        validate_document(no_dataset, require_frozen=True)

    no_trace = _contract_ready_for_frozen_gate_tests()
    gate = no_trace["quality_gates"][2]
    gate["applicability"] = "required"
    gate.pop("trace_reference", None)
    with pytest.raises(ContractError, match="pinned trace reference"):
        validate_document(no_trace, require_frozen=True)


def test_active_workload_cannot_exceed_model_native_context() -> None:
    contract = load_document(CONTRACT_PATH)
    workload = next(item for item in contract["workloads"] if item["id"] == "smoke_b1")
    workload["prompt_tokens"] = 4096
    workload["max_new_tokens"] = 1

    with pytest.raises(ContractError, match="exceeding model-native context 4096"):
        _validate_benchmark_references(contract, CONTRACT_PATH, require_frozen=False)


def test_schema_rejects_floating_model_revision_and_unknown_fields() -> None:
    manifest = load_document(MODEL_PATH)
    manifest["model_revision"] = "main"
    manifest["unexpected"] = True

    with pytest.raises(ContractError) as captured:
        validate_document(manifest)
    assert "does not match" in str(captured.value)
    assert "Additional properties" in str(captured.value)


def test_model_manifest_requires_unique_safe_paths_and_weight_shards() -> None:
    no_shards = load_document(MODEL_PATH)
    no_shards["weights"]["artifacts"] = [
        artifact
        for artifact in no_shards["weights"]["artifacts"]
        if not artifact["path"].endswith(".safetensors")
    ]
    with pytest.raises(ContractError, match="at least one Safetensors shard"):
        validate_document(no_shards)

    duplicate = load_document(MODEL_PATH)
    duplicate_artifact = copy.deepcopy(duplicate["weights"]["artifacts"][0])
    duplicate_artifact["sha256"] = "f" * 64
    duplicate["weights"]["artifacts"].append(duplicate_artifact)
    with pytest.raises(ContractError, match="duplicate file identity path"):
        validate_document(duplicate)

    unsafe = load_document(MODEL_PATH)
    unsafe["config"]["path"] = "../config.json"
    with pytest.raises(ContractError, match="safe project-relative"):
        validate_document(unsafe)


def test_frozen_model_manifest_requires_bound_tensor_inventory() -> None:
    manifest = load_document(MODEL_PATH)
    manifest["weights"].pop("tensor_inventory")

    with pytest.raises(ContractError, match="bound TensorInventory"):
        validate_document(manifest, require_frozen=True)


def test_frozen_model_manifest_validates_bound_tensor_inventory(
    tmp_path: Path,
) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='fixture'\n", encoding="utf-8")
    evidence_directory = tmp_path / "evidence"
    evidence_directory.mkdir()

    manifest = _draft_model_fixture()
    manifest["status"] = "frozen"
    manifest["weights"]["local_verification"] = "verified"
    manifest["weights"]["tensor_hashes_status"] = "verified"
    manifest["dtypes"]["local_tensor_scan"] = "verified"
    manifest["dtypes"]["observed_tensor_dtypes"] = ["F32"]
    source_manifest_sha256 = canonical_sha256(manifest)

    shards = []
    for position, artifact in enumerate(
        item
        for item in manifest["weights"]["artifacts"]
        if item["path"].endswith(".safetensors")
    ):
        tensors = []
        if position == 0:
            tensors.append(
                {
                    "name": "fixture.weight",
                    "dtype": "F32",
                    "shape": [1],
                    "offset_bytes": 0,
                    "size_bytes": 4,
                    "payload_sha256": "e" * 64,
                }
            )
        shards.append(
            {
                "path": artifact["path"],
                "size_bytes": artifact["size_bytes"],
                "file_sha256": artifact["sha256"],
                "tensors": tensors,
            }
        )
    inventory = {
        "schema_version": 1,
        "kind": "tensor_inventory",
        "generated_at": "2026-09-17T09:00:00Z",
        "model_manifest_sha256": source_manifest_sha256,
        "tensor_payload_hash_algorithm": "sha256",
        "shards": shards,
        "tensor_count": 1,
        "observed_dtypes": ["F32"],
    }
    inventory_path = evidence_directory / "tensor-inventory.json"
    inventory_bytes = (json.dumps(inventory, sort_keys=True) + "\n").encode()
    inventory_path.write_bytes(inventory_bytes)
    manifest["weights"]["tensor_inventory"] = {
        "path": "evidence/tensor-inventory.json",
        "sha256": hashlib.sha256(inventory_bytes).hexdigest(),
        "source_manifest_sha256": source_manifest_sha256,
    }
    manifest_path = tmp_path / "model-manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    validate_file(manifest_path, require_frozen=True)

    inventory["shards"][0]["file_sha256"] = "d" * 64
    inventory_bytes = (json.dumps(inventory, sort_keys=True) + "\n").encode()
    inventory_path.write_bytes(inventory_bytes)
    manifest["weights"]["tensor_inventory"]["sha256"] = hashlib.sha256(
        inventory_bytes
    ).hexdigest()
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ContractError, match="shard identities"):
        validate_file(manifest_path, require_frozen=True)


def test_machine_collector_output_conforms_to_schema(monkeypatch) -> None:
    monkeypatch.setattr("uma_qmoe.machine.shutil.which", lambda _command: None)
    baseline = collect_machine_baseline("local-test")

    validate_document(baseline)


def test_run_manifest_rejects_secret_like_environment_names() -> None:
    run = {
        "schema_version": 1,
        "kind": "run_manifest",
        "run_id": "test-001",
        "status": "planned",
        "created_at": "2026-09-17T00:00:00Z",
        "git": {
            "commit": "0" * 40,
            "dirty": False,
            "dirty_patch_sha256": None,
        },
        "target": {
            "id": "halo3",
            "machine_baseline_sha256": "1" * 64,
            "container_image": "example.invalid/image@sha256:" + "2" * 64,
        },
        "inputs": {
            "benchmark_contract_sha256": "3" * 64,
            "model_manifest_sha256": "4" * 64,
        },
        "command": {
            "argv": ["umaq", "validate"],
            "environment_allowlist": {"API_TOKEN": "must-not-be-recorded"},
        },
        "conditions": {
            "power_mode": None,
            "temperature_celsius": {"start": None, "end": None},
            "memory_available_bytes": {"start": None, "end": None},
        },
        "raw_artifacts": [],
        "aggregation": {
            "method": "none",
            "sample_count": 0,
            "report_quantiles": [],
        },
        "result_class": "pending",
    }

    with pytest.raises(ContractError, match="secret-like"):
        validate_document(run)

    run["command"]["environment_allowlist"] = {}
    artifact = {"path": "result.json", "sha256": "5" * 64, "size_bytes": 1}
    run["raw_artifacts"] = [artifact, copy.deepcopy(artifact)]
    with pytest.raises(ContractError, match="duplicate paths"):
        validate_document(run)


def _minimal_target_inventory() -> dict:
    return {
        "schema_version": 1,
        "kind": "target_inventory",
        "workspace": {
            "local_root": "/work/uma",
            "project_dir": "./UMA-QMoE",
            "models_dir": "./models",
            "datasets_dir": "./datasets",
            "artifacts_dir": "./artifacts",
            "private_dir": "./lab-private",
            "known_hosts_file": "./lab-private/state/known_hosts",
        },
        "defaults": {
            "gpus_per_job": 1,
            "max_parallel_jobs": 1,
            "busy_policy": "wait",
            "busy_timeout_seconds": 30,
        },
        "targets": [
            {
                "id": "spark-test",
                "ssh_host": "spark-test-alias",
                "remote_root": "/srv/uma",
                "expected": {
                    "hostname": "spark-test",
                    "host_architecture": "aarch64",
                    "platform": "nvidia-dgx-spark",
                    "accelerator": "NVIDIA GB10",
                    "accelerator_architecture": "sm_121a",
                    "compute_capability": "12.1",
                    "compute_backend": "cuda",
                    "unified_memory": True,
                    "nominal_memory_gb": 128,
                    "nominal_memory_bandwidth_gbps": 273,
                },
                "gpu_ids": [0],
                "gpus_per_job": 1,
                "max_parallel_jobs": 1,
                "busy_policy": "wait",
                "busy_timeout_seconds": 30,
                "container": {
                    "runtime": "docker",
                    "image": "example.invalid/uma@sha256:" + "a" * 64,
                    "source_repository": "https://example.invalid/runtime",
                    "source_commit": "b" * 40,
                    "source_ref": "test-ref",
                    "workdir": "/workspace/uma",
                    "gpu_access": "nvidia_all",
                    "devices": [],
                    "security_options": [],
                    "mounts": [
                        {"source": "/srv/uma", "target": "/workspace/uma", "read_only": False}
                    ],
                    "environment_from_host": [],
                },
            }
        ],
    }


def test_target_inventory_requires_digest_pinned_container() -> None:
    inventory = _minimal_target_inventory()
    validate_document(inventory)

    inventory["targets"][0]["container"]["image"] = "example.invalid/uma:latest"
    with pytest.raises(ContractError, match="does not match"):
        validate_document(inventory)


def test_target_inventory_rejects_duplicate_ids() -> None:
    inventory = _minimal_target_inventory()
    inventory["targets"].append(copy.deepcopy(inventory["targets"][0]))
    inventory["targets"][1]["ssh_host"] = "different-alias"

    with pytest.raises(ContractError, match="duplicate target ids"):
        validate_document(inventory)
