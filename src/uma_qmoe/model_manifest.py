"""State transitions for immutable model evidence."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

from .contracts import canonical_sha256, validate_document


class ModelManifestError(ValueError):
    """Raised when a draft cannot be frozen from the supplied evidence."""


def freeze_model_manifest(
    manifest: Mapping[str, Any],
    inventory: Mapping[str, Any],
    *,
    inventory_path: str,
    inventory_file_sha256: str,
) -> dict[str, Any]:
    """Bind a verified TensorInventory and transition a draft to frozen."""

    validate_document(manifest)
    validate_document(inventory)
    if manifest.get("kind") != "model_manifest":
        raise ModelManifestError("manifest kind must be 'model_manifest'")
    if manifest["status"] != "draft":
        raise ModelManifestError("only a draft ModelManifest can be frozen")
    if inventory.get("kind") != "tensor_inventory":
        raise ModelManifestError("inventory kind must be 'tensor_inventory'")

    source_manifest_sha256 = canonical_sha256(manifest)
    if inventory["model_manifest_sha256"] != source_manifest_sha256:
        raise ModelManifestError(
            "TensorInventory was not generated from this exact draft ModelManifest"
        )
    expected_shards = {
        artifact["path"]: (artifact["size_bytes"], artifact["sha256"])
        for artifact in manifest["weights"]["artifacts"]
        if artifact["path"].endswith(".safetensors")
    }
    observed_shards = {
        shard["path"]: (shard["size_bytes"], shard["file_sha256"])
        for shard in inventory["shards"]
    }
    if observed_shards != expected_shards:
        raise ModelManifestError(
            "TensorInventory shard identities do not match the ModelManifest"
        )
    if not inventory["observed_dtypes"]:
        raise ModelManifestError("TensorInventory observed_dtypes must not be empty")

    frozen = copy.deepcopy(dict(manifest))
    frozen["status"] = "frozen"
    frozen["dtypes"]["local_tensor_scan"] = "verified"
    frozen["dtypes"]["observed_tensor_dtypes"] = inventory["observed_dtypes"]
    frozen["weights"]["hash_source"] = "local_sha256"
    frozen["weights"]["local_verification"] = "verified"
    frozen["weights"]["tensor_hashes_status"] = "verified"
    frozen["weights"]["tensor_inventory"] = {
        "path": inventory_path,
        "sha256": inventory_file_sha256,
        "source_manifest_sha256": source_manifest_sha256,
    }
    validate_document(frozen, require_frozen=True)
    return frozen


__all__ = ["ModelManifestError", "freeze_model_manifest"]
