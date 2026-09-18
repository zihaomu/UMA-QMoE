from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import pytest

from uma_qmoe.contracts import ContractError, validate_document
from uma_qmoe.safetensors_inventory import (
    SafetensorsInventoryError,
    inventory_safetensors,
    verify_model_manifest_artifacts,
)


def _shard_bytes(header: dict[str, Any], payload: bytes) -> bytes:
    encoded_header = json.dumps(
        header, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return len(encoded_header).to_bytes(8, "little") + encoded_header + payload


def _valid_shard() -> tuple[bytes, bytes, bytes]:
    first_payload = struct.pack("<ff", 1.25, -2.5)
    second_payload = bytes((3, 5, 8))
    payload = first_payload + second_payload
    shard = _shard_bytes(
        {
            "__metadata__": {"format": "pt"},
            "layer.float_weight": {
                "dtype": "F32",
                "shape": [2],
                "data_offsets": [0, len(first_payload)],
            },
            "layer.byte_weight": {
                "dtype": "U8",
                "shape": [3],
                "data_offsets": [len(first_payload), len(payload)],
            },
        },
        payload,
    )
    return shard, first_payload, second_payload


def _manifest_for(path: str, payload: bytes) -> dict[str, Any]:
    return {
        "kind": "model_manifest",
        "weights": {
            "format": "safetensors",
            "artifacts": [
                {
                    "path": path,
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                },
                {
                    "path": "model.safetensors.index.json",
                    "size_bytes": 1,
                    "sha256": "0" * 64,
                },
            ],
        },
    }


def test_inventories_file_and_tensor_hashes_from_handmade_shard(tmp_path: Path) -> None:
    shard, first_payload, second_payload = _valid_shard()
    path = tmp_path / "model.safetensors"
    path.write_bytes(shard)

    inventory = inventory_safetensors(
        path, shard_path="model.safetensors", chunk_size=3
    )

    assert inventory == {
        "path": "model.safetensors",
        "size_bytes": len(shard),
        "file_sha256": hashlib.sha256(shard).hexdigest(),
        "tensors": [
            {
                "name": "layer.float_weight",
                "dtype": "F32",
                "shape": [2],
                "offset_bytes": 0,
                "size_bytes": 8,
                "payload_sha256": hashlib.sha256(first_payload).hexdigest(),
            },
            {
                "name": "layer.byte_weight",
                "dtype": "U8",
                "shape": [3],
                "offset_bytes": 8,
                "size_bytes": 3,
                "payload_sha256": hashlib.sha256(second_payload).hexdigest(),
            },
        ],
    }


def test_verifies_every_manifest_shard_and_reports_observed_dtypes(tmp_path: Path) -> None:
    shard, _first_payload, _second_payload = _valid_shard()
    (tmp_path / "model.safetensors").write_bytes(shard)
    manifest = _manifest_for("model.safetensors", shard)

    result = verify_model_manifest_artifacts(manifest, tmp_path, chunk_size=2)

    assert result["tensor_count"] == 2
    assert result["observed_dtypes"] == ["F32", "U8"]
    assert result["shards"][0]["path"] == "model.safetensors"

    document = {
        "schema_version": 1,
        "kind": "tensor_inventory",
        "generated_at": "2026-09-17T09:00:00Z",
        "model_manifest_sha256": "a" * 64,
        "tensor_payload_hash_algorithm": "sha256",
        "shards": result["shards"],
        "tensor_count": result["tensor_count"],
        "observed_dtypes": result["observed_dtypes"],
    }
    validate_document(document)

    duplicate = dict(document)
    duplicate["shards"] = [result["shards"][0], result["shards"][0]]
    duplicate["tensor_count"] = 4
    with pytest.raises(ContractError, match="duplicate shard paths"):
        validate_document(duplicate)


def test_manifest_verification_rejects_same_size_payload_tampering(tmp_path: Path) -> None:
    shard, _first_payload, _second_payload = _valid_shard()
    manifest = _manifest_for("model.safetensors", shard)
    tampered = bytearray(shard)
    tampered[-1] ^= 0xFF
    (tmp_path / "model.safetensors").write_bytes(tampered)

    with pytest.raises(SafetensorsInventoryError, match="SHA-256 mismatch"):
        verify_model_manifest_artifacts(manifest, tmp_path)


def test_manifest_verification_reports_missing_shard(tmp_path: Path) -> None:
    shard, _first_payload, _second_payload = _valid_shard()
    manifest = _manifest_for("missing.safetensors", shard)

    with pytest.raises(SafetensorsInventoryError, match="missing Safetensors artifact"):
        verify_model_manifest_artifacts(manifest, tmp_path)


def test_manifest_verification_rejects_symlink_parent_escape(tmp_path: Path) -> None:
    shard, _first_payload, _second_payload = _valid_shard()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "model.safetensors").write_bytes(shard)
    root = tmp_path / "root"
    root.mkdir()
    (root / "nested").symlink_to(outside, target_is_directory=True)
    manifest = _manifest_for("nested/model.safetensors", shard)

    with pytest.raises(SafetensorsInventoryError, match="escapes through a symlink"):
        verify_model_manifest_artifacts(manifest, root)


def test_rejects_truncated_header_before_json_decode(tmp_path: Path) -> None:
    path = tmp_path / "truncated.safetensors"
    path.write_bytes((20).to_bytes(8, "little") + b"{}")

    with pytest.raises(SafetensorsInventoryError, match="truncated Safetensors header"):
        inventory_safetensors(path)


def test_rejects_tensor_offsets_outside_data_section(tmp_path: Path) -> None:
    path = tmp_path / "out-of-bounds.safetensors"
    path.write_bytes(
        _shard_bytes(
            {
                "weight": {
                    "dtype": "F32",
                    "shape": [2],
                    "data_offsets": [0, 8],
                }
            },
            bytes(4),
        )
    )

    with pytest.raises(SafetensorsInventoryError, match="exceed the 4-byte data section"):
        inventory_safetensors(path)


def test_rejects_overlapping_tensor_payloads(tmp_path: Path) -> None:
    path = tmp_path / "overlap.safetensors"
    path.write_bytes(
        _shard_bytes(
            {
                "first": {
                    "dtype": "U8",
                    "shape": [4],
                    "data_offsets": [0, 4],
                },
                "second": {
                    "dtype": "U8",
                    "shape": [3],
                    "data_offsets": [3, 6],
                },
            },
            bytes(6),
        )
    )

    with pytest.raises(SafetensorsInventoryError, match="payloads overlap"):
        inventory_safetensors(path)


def test_rejects_unindexed_gap_in_data_section(tmp_path: Path) -> None:
    path = tmp_path / "gap.safetensors"
    path.write_bytes(
        _shard_bytes(
            {
                "weight": {
                    "dtype": "U8",
                    "shape": [2],
                    "data_offsets": [1, 3],
                }
            },
            bytes(3),
        )
    )

    with pytest.raises(SafetensorsInventoryError, match="unindexed gap"):
        inventory_safetensors(path)


def test_zero_size_tensor_must_sit_on_a_payload_boundary(tmp_path: Path) -> None:
    valid = tmp_path / "valid-empty.safetensors"
    valid.write_bytes(
        _shard_bytes(
            {
                "empty": {
                    "dtype": "U8",
                    "shape": [0],
                    "data_offsets": [0, 0],
                },
                "weight": {
                    "dtype": "U8",
                    "shape": [4],
                    "data_offsets": [0, 4],
                },
            },
            bytes(4),
        )
    )
    assert len(inventory_safetensors(valid)["tensors"]) == 2

    invalid = tmp_path / "invalid-empty.safetensors"
    invalid.write_bytes(
        _shard_bytes(
            {
                "weight": {
                    "dtype": "U8",
                    "shape": [4],
                    "data_offsets": [0, 4],
                },
                "empty": {
                    "dtype": "U8",
                    "shape": [0],
                    "data_offsets": [2, 2],
                },
            },
            bytes(4),
        )
    )
    with pytest.raises(SafetensorsInventoryError, match="overlap"):
        inventory_safetensors(invalid)


def test_unknown_or_unsupported_dtype_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "unsupported.safetensors"
    path.write_bytes(
        _shard_bytes(
            {
                "weight": {
                    "dtype": "C128",
                    "shape": [1],
                    "data_offsets": [0, 16],
                }
            },
            bytes(16),
        )
    )

    with pytest.raises(SafetensorsInventoryError, match="unsupported or invalid dtype"):
        inventory_safetensors(path)


def test_rejects_header_with_leading_whitespace(tmp_path: Path) -> None:
    header = b' {"weight":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}'
    path = tmp_path / "leading-space.safetensors"
    path.write_bytes(len(header).to_bytes(8, "little") + header + b"x")

    with pytest.raises(SafetensorsInventoryError, match="must begin"):
        inventory_safetensors(path)


def test_rejects_unexpected_tensor_metadata_field(tmp_path: Path) -> None:
    path = tmp_path / "unexpected-field.safetensors"
    path.write_bytes(
        _shard_bytes(
            {
                "weight": {
                    "dtype": "U8",
                    "shape": [1],
                    "data_offsets": [0, 1],
                    "compression": "none",
                }
            },
            b"x",
        )
    )

    with pytest.raises(SafetensorsInventoryError, match="unexpected fields"):
        inventory_safetensors(path)


def test_rejects_header_larger_than_configured_limit(tmp_path: Path) -> None:
    shard, _first_payload, _second_payload = _valid_shard()
    path = tmp_path / "large-header.safetensors"
    path.write_bytes(shard)

    with pytest.raises(SafetensorsInventoryError, match="exceeding the 16-byte limit"):
        inventory_safetensors(path, max_header_bytes=16)


def test_rejects_unlisted_safetensors_in_closed_model_directory(tmp_path: Path) -> None:
    shard, _first_payload, _second_payload = _valid_shard()
    (tmp_path / "model.safetensors").write_bytes(shard)
    (tmp_path / "other.safetensors").write_bytes(shard)
    manifest = _manifest_for("model.safetensors", shard)

    with pytest.raises(SafetensorsInventoryError, match="unlisted Safetensors artifacts"):
        verify_model_manifest_artifacts(manifest, tmp_path)
