from __future__ import annotations

import hashlib
import json
from pathlib import Path
import struct

import numpy as np
import pytest

from uma_qmoe.contracts import ContractError, validate_document
from uma_qmoe.expert_pack import (
    PREFIX,
    ExpertPackReader,
    inspect_expert_pack,
    write_expert_pack,
)
from uma_qmoe.q4 import dequantize_q4


def _write(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    tensors = {
        "model.layers.2.experts.7.gate_up_proj": np.linspace(
            -2, 2, 257, dtype=np.float32
        ).reshape(1, 257),
        "model.layers.2.experts.7.down_proj": np.linspace(
            1, -1, 129, dtype=np.float32
        ),
    }
    manifest = write_expert_pack(
        path,
        tensors.items(),
        model_id="allenai/OLMoE-1B-7B-0125",
        model_revision="a" * 40,
        model_manifest_sha256="b" * 64,
    )
    return manifest, tensors


def _rewrite_canonical_payload(path: Path, mutate: object) -> Path:
    payload = bytearray(path.read_bytes())
    prefix = list(PREFIX.unpack_from(payload))
    header_length = prefix[2]
    payload_offset = prefix[3]
    header = json.loads(payload[PREFIX.size : PREFIX.size + header_length])
    tensor = header["tensors"][0]
    raw = bytearray(payload[payload_offset:])
    mutate(raw, tensor)
    packed = raw[tensor["packed_offset"] : tensor["packed_offset"] + tensor["packed_bytes"]]
    scales = raw[tensor["scale_offset"] : tensor["scale_offset"] + tensor["scale_bytes"]]
    tensor["tensor_payload_sha256"] = hashlib.sha256(packed + scales).hexdigest()
    header_bytes = json.dumps(
        header, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    assert len(header_bytes) == header_length
    prefix[5] = hashlib.sha256(header_bytes).digest()
    prefix[6] = hashlib.sha256(raw).digest()
    payload[: PREFIX.size] = PREFIX.pack(*prefix)
    payload[PREFIX.size : PREFIX.size + header_length] = header_bytes
    payload[payload_offset:] = raw
    corrupted = path.with_name("semantic-corruption.uqep")
    corrupted.write_bytes(payload)
    return corrupted


def test_expert_pack_header_alignment_identity_hash_and_single_mapping(
    tmp_path: Path,
) -> None:
    path = tmp_path / "experts.uqep"
    manifest, tensors = _write(path)

    validate_document(manifest)
    assert manifest["payload_offset"] % 4096 == 0
    assert manifest["header"]["tensor_count"] == 2
    assert all(
        item["packed_offset"] % 64 == 0
        for item in manifest["header"]["tensors"]
    )
    with ExpertPackReader(
        path,
        expected_model_id="allenai/OLMoE-1B-7B-0125",
        expected_model_revision="a" * 40,
        expected_model_manifest_sha256="b" * 64,
    ) as reader:
        assert reader.mapping_count == 1
        restored = dequantize_q4(
            reader.tensor_q4("model.layers.2.experts.7.down_proj")
        )
        assert restored.shape == tensors["model.layers.2.experts.7.down_proj"].shape
        packed, scales, metadata = reader.tensor_views(
            "model.layers.2.experts.7.gate_up_proj"
        )
        assert len(packed) == metadata["packed_bytes"]
        assert len(scales) == metadata["scale_bytes"]
        packed.release()
        scales.release()


def test_expert_pack_rejects_payload_corruption_and_model_mismatch(
    tmp_path: Path,
) -> None:
    path = tmp_path / "experts.uqep"
    _write(path)
    corrupted = tmp_path / "corrupted.uqep"
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 0x01
    corrupted.write_bytes(payload)

    with pytest.raises(ContractError, match="payload hash"):
        ExpertPackReader(corrupted)
    with pytest.raises(ContractError, match="model revision"):
        ExpertPackReader(path, expected_model_revision="c" * 40)


def test_expert_pack_manifest_rejects_tampered_tensor_count(tmp_path: Path) -> None:
    path = tmp_path / "experts.uqep"
    manifest, _ = _write(path)
    manifest["header"]["tensor_count"] += 1

    with pytest.raises(ContractError, match="tensor count"):
        validate_document(manifest)


def test_expert_pack_inspection_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "experts.uqep"
    manifest, _ = _write(path)

    assert inspect_expert_pack(path) == manifest
    json.dumps(manifest, allow_nan=False)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda raw, tensor: raw.__setitem__(tensor["packed_offset"], 0x08),
            "reserved Q4 -8",
        ),
        (
            lambda raw, tensor: struct.pack_into(
                "<f", raw, tensor["scale_offset"], float("nan")
            ),
            "finite and positive",
        ),
        (
            lambda raw, tensor: raw.__setitem__(
                tensor["packed_offset"] + tensor["packed_bytes"] - 1,
                raw[tensor["packed_offset"] + tensor["packed_bytes"] - 1] | 0x10,
            ),
            "tail padding",
        ),
        (
            lambda raw, tensor: raw.__setitem__(
                tensor["packed_offset"] + tensor["packed_bytes"], 1
            ),
            "padding must be zero",
        ),
    ],
)
def test_expert_pack_rejects_semantically_invalid_rehashed_payload(
    tmp_path: Path, mutate: object, message: str
) -> None:
    path = tmp_path / "experts.uqep"
    _write(path)
    corrupted = _rewrite_canonical_payload(path, mutate)

    with pytest.raises(ContractError, match=message):
        ExpertPackReader(corrupted)
