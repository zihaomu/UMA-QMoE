from __future__ import annotations

import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest

from uma_qmoe.contracts import ContractError, validate_document
from uma_qmoe.target_pack import (
    TargetPackReader,
    TargetTensor,
    decode_target_tensor,
    inspect_target_pack,
    write_target_pack,
)


POLICY_ID = "olmoe-layer15-awq-q4-q8-bf16-v1"
LAYER_ENCODINGS = {
    layer: (
        "q4_group128"
        if layer == 15
        else "q8_group128"
        if layer in {8, 11, 12, 13, 14}
        else "bf16_le"
    )
    for layer in range(16)
}


def _write(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    tensors = {
        "model.layers.0.mlp.experts.0.gate_proj.weight": np.linspace(
            -2.0, 2.0, 258, dtype=np.float32
        ).reshape(2, 129),
        "model.layers.8.mlp.experts.1.up_proj.weight": np.linspace(
            -1.0, 1.0, 257, dtype=np.float32
        ).reshape(1, 257),
        "model.layers.15.mlp.experts.2.down_proj.weight": np.linspace(
            1.0, -1.0, 129, dtype=np.float32
        ).reshape(1, 129),
    }
    manifest = write_target_pack(
        path,
        tensors.items(),
        model_id="allenai/OLMoE-1B-7B-0125",
        model_revision="a" * 40,
        model_manifest_sha256="b" * 64,
        policy_id=POLICY_ID,
        layer_encodings=LAYER_ENCODINGS,
        policy_evidence_sha256="c" * 64,
    )
    return manifest, tensors


def test_target_pack_mixed_encodings_identity_and_single_mapping(tmp_path: Path) -> None:
    path = tmp_path / "experts.uqtp"
    manifest, tensors = _write(path)

    validate_document(manifest)
    assert manifest["header"]["encoding_tensor_counts"] == {
        "bf16_le": 1,
        "q4_group128": 1,
        "q8_group128": 1,
    }
    with TargetPackReader(
        path,
        expected_model_id="allenai/OLMoE-1B-7B-0125",
        expected_model_revision="a" * 40,
        expected_model_manifest_sha256="b" * 64,
        expected_policy_id=POLICY_ID,
    ) as reader:
        assert reader.mapping_count == 1
        assert reader.layer_encoding(0) == "bf16_le"
        assert reader.layer_encoding(8) == "q8_group128"
        assert reader.layer_encoding(15) == "q4_group128"
        for name, source in tensors.items():
            restored = decode_target_tensor(reader.tensor(name))
            assert restored.shape == source.shape
            assert np.all(np.isfinite(restored))
        bf16_name = "model.layers.0.mlp.experts.0.gate_proj.weight"
        assert np.max(
            np.abs(decode_target_tensor(reader.tensor(bf16_name)) - tensors[bf16_name])
        ) < 0.01


def test_target_pack_rejects_policy_and_payload_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "experts.uqtp"
    _write(path)
    with pytest.raises(ContractError, match="policy id"):
        TargetPackReader(path, expected_policy_id="different")

    corrupted = tmp_path / "corrupted.uqtp"
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    corrupted.write_bytes(payload)
    with pytest.raises(ContractError, match="payload hash"):
        TargetPackReader(corrupted)


def test_target_pack_manifest_rejects_tampered_encoding_counts(tmp_path: Path) -> None:
    path = tmp_path / "experts.uqtp"
    manifest, _ = _write(path)
    tampered = copy.deepcopy(manifest)
    tampered["header"]["encoding_tensor_counts"]["q4_group128"] += 1
    with pytest.raises(ContractError, match="encoding tensor counts"):
        validate_document(tampered)


def test_target_pack_inspection_is_deterministic(tmp_path: Path) -> None:
    path = tmp_path / "experts.uqtp"
    manifest, _ = _write(path)
    assert inspect_target_pack(path) == manifest
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["artifact"]["sha256"]


def test_target_pack_requires_complete_layer_policy(tmp_path: Path) -> None:
    policy = dict(LAYER_ENCODINGS)
    policy.pop(0)
    with pytest.raises(ContractError, match="every OLMoE layer"):
        write_target_pack(
            tmp_path / "invalid.uqtp",
            [
                (
                    "model.layers.15.mlp.experts.0.down_proj.weight",
                    np.ones((1, 128), dtype=np.float32),
                )
            ],
            model_id="allenai/OLMoE-1B-7B-0125",
            model_revision="a" * 40,
            model_manifest_sha256="b" * 64,
            policy_id=POLICY_ID,
            layer_encodings=policy,
            policy_evidence_sha256="c" * 64,
        )


def test_target_pack_preserves_validated_preencoded_tensor(tmp_path: Path) -> None:
    path = tmp_path / "preencoded.uqtp"
    name = "model.layers.8.mlp.experts.0.gate_proj.weight"
    tensor = TargetTensor(
        shape=(1, 2),
        encoding="q8_group128",
        data=bytes((1, 255)),
        scales=np.asarray([0.25], dtype="<f4").tobytes(),
        group_size=128,
    )
    write_target_pack(
        path,
        [(name, tensor)],
        model_id="allenai/OLMoE-1B-7B-0125",
        model_revision="a" * 40,
        model_manifest_sha256="b" * 64,
        policy_id=POLICY_ID,
        layer_encodings=LAYER_ENCODINGS,
        policy_evidence_sha256="c" * 64,
    )
    with TargetPackReader(path) as reader:
        observed = reader.tensor(name)
        assert observed == tensor
        assert np.array_equal(
            decode_target_tensor(observed),
            np.asarray([[0.25, -0.25]], dtype=np.float32),
        )


def test_target_pack_rejects_preencoded_tensor_outside_layer_policy(
    tmp_path: Path,
) -> None:
    tensor = TargetTensor(
        shape=(1, 2),
        encoding="q8_group128",
        data=bytes((1, 255)),
        scales=np.asarray([0.25], dtype="<f4").tobytes(),
        group_size=128,
    )
    with pytest.raises(ContractError, match="does not match its layer policy"):
        write_target_pack(
            tmp_path / "mismatch.uqtp",
            [("model.layers.0.mlp.experts.0.gate_proj.weight", tensor)],
            model_id="allenai/OLMoE-1B-7B-0125",
            model_revision="a" * 40,
            model_manifest_sha256="b" * 64,
            policy_id=POLICY_ID,
            layer_encodings=LAYER_ENCODINGS,
            policy_evidence_sha256="c" * 64,
        )
