from __future__ import annotations

import hashlib
import json
import struct
from datetime import datetime, timezone
from pathlib import Path

import pytest

from uma_qmoe.bf16 import (
    Bf16DerivationError,
    convert_safetensors_f32_to_bf16,
    derive_bf16_oracle,
    f32_bytes_to_bf16_rne,
)
from uma_qmoe.contracts import canonical_sha256, validate_document
from uma_qmoe.safetensors_inventory import inventory_safetensors


def _f32_shard(words: list[int]) -> bytes:
    payload = struct.pack(f"<{len(words)}I", *words)
    header = json.dumps(
        {
            "weight": {
                "dtype": "F32",
                "shape": [len(words)],
                "data_offsets": [0, len(payload)],
            }
        },
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * ((-len(header)) % 8)
    return len(header).to_bytes(8, "little") + header + payload


def _payload(path: Path) -> bytes:
    with path.open("rb") as stream:
        header_size = int.from_bytes(stream.read(8), "little")
        stream.seek(header_size, 1)
        return stream.read()


def test_f32_words_round_to_bf16_nearest_even_and_preserve_nan() -> None:
    words = [
        0x3F808000,  # exact tie, even high word stays even
        0x3F818000,  # exact tie, odd high word increments
        0x7F800000,  # positive infinity
        0xFF800000,  # negative infinity
        0x7F800001,  # NaN whose low payload would otherwise disappear
        0xFF800001,  # negative NaN preserves sign and NaN class
    ]
    raw = struct.pack("<6I", *words)

    assert struct.unpack("<6H", f32_bytes_to_bf16_rne(raw)) == (
        0x3F80,
        0x3F82,
        0x7F80,
        0xFF80,
        0x7FC0,
        0xFFC0,
    )


def test_converts_and_inventories_a_safetensors_shard(tmp_path: Path) -> None:
    source_bytes = _f32_shard([0x3F800000, 0xC0200000, 0x00000000])
    source = tmp_path / "source.safetensors"
    destination = tmp_path / "derived.safetensors"
    source.write_bytes(source_bytes)

    identity = convert_safetensors_f32_to_bf16(
        source,
        destination,
        expected_source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        chunk_bytes=8,
    )

    assert identity["path"] == "derived.safetensors"
    assert identity["sha256"] == hashlib.sha256(destination.read_bytes()).hexdigest()
    assert struct.unpack("<3H", _payload(destination)) == (0x3F80, 0xC020, 0)
    inventory = inventory_safetensors(destination)
    assert inventory["tensors"][0]["dtype"] == "BF16"
    assert inventory["tensors"][0]["shape"] == [3]


def test_resumes_a_compatible_partial_and_is_idempotent(tmp_path: Path) -> None:
    source_bytes = _f32_shard([0x3F800000, 0x40000000, 0x40400000, 0x40800000])
    source = tmp_path / "source.safetensors"
    destination = tmp_path / "derived.safetensors"
    source.write_bytes(source_bytes)
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    first = convert_safetensors_f32_to_bf16(
        source,
        destination,
        expected_source_sha256=source_sha256,
        chunk_bytes=8,
    )
    expected = destination.read_bytes()
    header_size = int.from_bytes(expected[:8], "little")
    partial_size = 8 + header_size + 4
    destination.rename(destination.with_name(destination.name + ".part"))
    destination.with_name(destination.name + ".part").write_bytes(expected[:partial_size])

    resumed = convert_safetensors_f32_to_bf16(
        source,
        destination,
        expected_source_sha256=source_sha256,
        chunk_bytes=8,
    )
    repeated = convert_safetensors_f32_to_bf16(
        source,
        destination,
        expected_source_sha256=source_sha256,
        chunk_bytes=8,
    )

    assert destination.read_bytes() == expected
    assert resumed == first
    assert repeated == first


def test_rejects_non_f32_source(tmp_path: Path) -> None:
    header = json.dumps(
        {"weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}},
        separators=(",", ":"),
    ).encode("utf-8")
    header += b" " * ((-len(header)) % 8)
    source_bytes = len(header).to_bytes(8, "little") + header + bytes(2)
    source = tmp_path / "source.safetensors"
    source.write_bytes(source_bytes)

    with pytest.raises(Bf16DerivationError, match="all-F32"):
        convert_safetensors_f32_to_bf16(
            source,
            tmp_path / "derived.safetensors",
            expected_source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        )


def test_derives_complete_idempotent_oracle_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_root = tmp_path / "source"
    destination_root = tmp_path / "derived"
    source_root.mkdir()
    shard_bytes = _f32_shard([0x3F800000, 0x40000000])
    (source_root / "model.safetensors").write_bytes(shard_bytes)
    config_bytes = b'{"model_type":"fixture","torch_dtype":"float32"}\n'
    tokenizer_bytes = b'{"version":"1"}\n'
    index_bytes = (
        json.dumps(
            {
                "metadata": {"total_size": 8},
                "weight_map": {"weight": "model.safetensors"},
            }
        )
        + "\n"
    ).encode()
    (source_root / "config.json").write_bytes(config_bytes)
    (source_root / "tokenizer.json").write_bytes(tokenizer_bytes)
    (source_root / "model.safetensors.index.json").write_bytes(index_bytes)

    def identity(name: str, payload: bytes) -> dict[str, object]:
        return {
            "path": name,
            "size_bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    manifest = {
        "model_id": "example/fixture",
        "model_revision": "a" * 40,
        "config": identity("config.json", config_bytes),
        "tokenizer": {"files": [identity("tokenizer.json", tokenizer_bytes)]},
        "weights": {
            "artifacts": [
                identity("model.safetensors", shard_bytes),
                identity("model.safetensors.index.json", index_bytes),
            ]
        },
    }
    inventory = {
        "observed_dtypes": ["F32"],
        "shards": [{"path": "model.safetensors"}],
    }
    keyword_arguments = {
        "source_manifest_path": "models/manifests/source.yaml",
        "source_manifest_file_sha256": "b" * 64,
        "tensor_inventory_path": "models/inventories/source.json",
        "tensor_inventory_file_sha256": "c" * 64,
        "artifact_root": "models/derived/fixture/bf16-rne-v1",
        "chunk_bytes": 8,
    }

    first = derive_bf16_oracle(
        manifest, source_directory=source_root, destination_directory=destination_root,
        inventory=inventory, **keyword_arguments
    )

    class LaterDatetime:
        @classmethod
        def now(cls, tz: object) -> datetime:
            assert tz is timezone.utc
            return datetime(2030, 1, 2, 3, 4, 5, tzinfo=timezone.utc)

    monkeypatch.setattr("uma_qmoe.bf16.datetime", LaterDatetime)
    second = derive_bf16_oracle(
        manifest, source_directory=source_root, destination_directory=destination_root,
        inventory=inventory, **keyword_arguments
    )
    reused = derive_bf16_oracle(
        manifest, source_directory=source_root, destination_directory=destination_root,
        inventory=inventory, existing_derivation=first, **keyword_arguments
    )

    validate_document(first)
    assert second["generated_at"] != first["generated_at"]
    assert canonical_sha256(second) == canonical_sha256(first)
    assert reused == first
    assert json.loads((destination_root / "config.json").read_text())["torch_dtype"] == "bfloat16"
    derived_index = json.loads(
        (destination_root / "model.safetensors.index.json").read_text()
    )
    assert derived_index["metadata"]["total_size"] == 4
