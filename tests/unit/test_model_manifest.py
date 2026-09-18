from __future__ import annotations

import copy
from pathlib import Path

import pytest

from uma_qmoe.contracts import canonical_sha256, load_document, validate_document
from uma_qmoe.model_manifest import ModelManifestError, freeze_model_manifest


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models/manifests/olmoe_1b_7b_0125.yaml"


def _draft_manifest() -> dict:
    manifest = load_document(MODEL_PATH)
    manifest["status"] = "draft"
    manifest["dtypes"]["local_tensor_scan"] = "pending_download"
    manifest["dtypes"].pop("observed_tensor_dtypes", None)
    manifest["weights"]["hash_source"] = "huggingface_lfs_metadata"
    manifest["weights"]["local_verification"] = "pending_download"
    manifest["weights"]["tensor_hashes_status"] = "pending_local_scan"
    manifest["weights"].pop("tensor_inventory", None)
    return manifest


def _inventory(manifest: dict) -> dict:
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
    return {
        "schema_version": 1,
        "kind": "tensor_inventory",
        "generated_at": "2026-09-17T10:00:00Z",
        "model_manifest_sha256": canonical_sha256(manifest),
        "tensor_payload_hash_algorithm": "sha256",
        "shards": shards,
        "tensor_count": 1,
        "observed_dtypes": ["F32"],
    }


def test_freezes_exact_draft_and_binds_tensor_inventory() -> None:
    manifest = _draft_manifest()
    inventory = _inventory(manifest)

    frozen = freeze_model_manifest(
        manifest,
        inventory,
        inventory_path="models/inventories/olmoe.json",
        inventory_file_sha256="a" * 64,
    )

    assert manifest["status"] == "draft"
    assert frozen["status"] == "frozen"
    assert frozen["dtypes"]["observed_tensor_dtypes"] == ["F32"]
    assert frozen["weights"]["hash_source"] == "local_sha256"
    assert frozen["weights"]["tensor_inventory"]["source_manifest_sha256"] == canonical_sha256(
        manifest
    )
    validate_document(frozen, require_frozen=True)


def test_rejects_inventory_from_a_different_draft_or_shard_set() -> None:
    manifest = _draft_manifest()
    inventory = _inventory(manifest)
    inventory["model_manifest_sha256"] = "b" * 64
    with pytest.raises(ModelManifestError, match="exact draft"):
        freeze_model_manifest(
            manifest,
            inventory,
            inventory_path="models/inventories/olmoe.json",
            inventory_file_sha256="a" * 64,
        )

    inventory = _inventory(manifest)
    tampered = copy.deepcopy(inventory)
    tampered["shards"][0]["file_sha256"] = "c" * 64
    with pytest.raises(ModelManifestError, match="shard identities"):
        freeze_model_manifest(
            manifest,
            tampered,
            inventory_path="models/inventories/olmoe.json",
            inventory_file_sha256="a" * 64,
        )
