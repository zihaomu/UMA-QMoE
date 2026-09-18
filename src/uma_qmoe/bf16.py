"""Deterministic, resumable F32-to-BF16 Safetensors derivation.

The converter works on raw IEEE-754 words in bounded chunks.  It implements
round-to-nearest-even explicitly instead of relying on a framework cast, and
keeps NaNs as NaNs even when truncation would otherwise produce an infinity.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, BinaryIO

try:
    import numpy as np
except ImportError:  # pragma: no cover - exercised by packaging, not dev tests
    np = None  # type: ignore[assignment]

from .contracts import canonical_sha256, file_sha256, validate_document
from .safetensors_inventory import (
    DEFAULT_MAX_HEADER_BYTES,
    SafetensorsInventoryError,
    _parse_header,
    _read_exact,
    _tensor_layouts,
)


DEFAULT_CONVERSION_CHUNK_BYTES = 64 * 1024 * 1024


class Bf16DerivationError(ValueError):
    """Raised when a source or partially derived artifact is not trustworthy."""


def f32_bytes_to_bf16_rne(raw: bytes) -> bytes:
    """Convert little-endian F32 bytes to BF16 using round-to-nearest-even."""

    if np is None:
        raise Bf16DerivationError(
            "BF16 derivation requires the 'conversion' extra: "
            "install with `uv sync --extra conversion`"
        )
    if len(raw) % 4:
        raise Bf16DerivationError("F32 input byte count must be divisible by four")
    words = np.frombuffer(raw, dtype="<u4")
    low = words & np.uint32(0xFFFF)
    high = words >> np.uint32(16)
    increment = (low > np.uint32(0x8000)) | (
        (low == np.uint32(0x8000)) & ((high & np.uint32(1)) != 0)
    )
    result = (high + increment.astype(np.uint32)).astype("<u2")

    # Any F32 NaN must remain a BF16 NaN.  Very small F32 NaN payloads can be
    # lost when rounded; set the BF16 quiet-NaN bit only in that case.
    source_nan = (words & np.uint32(0x7FFFFFFF)) > np.uint32(0x7F800000)
    if np.any(source_nan):
        result = result.copy()
        result[source_nan] = high[source_nan].astype("<u2") | np.uint16(0x0040)
    return result.tobytes()


def _positive_aligned_chunk_size(chunk_bytes: int) -> int:
    if isinstance(chunk_bytes, bool) or not isinstance(chunk_bytes, int) or chunk_bytes < 4:
        raise Bf16DerivationError("chunk_bytes must be an integer of at least four")
    return chunk_bytes - (chunk_bytes % 4)


def _read_source_header(
    stream: BinaryIO, file_size: int
) -> tuple[bytes, bytes, dict[str, Any], int]:
    if file_size < 8:
        raise Bf16DerivationError("source has a truncated Safetensors prefix")
    prefix = _read_exact(stream, 8, "Safetensors length prefix")
    header_size = int.from_bytes(prefix, "little")
    if header_size == 0 or header_size > DEFAULT_MAX_HEADER_BYTES:
        raise Bf16DerivationError(f"invalid Safetensors header length {header_size}")
    if header_size > file_size - 8:
        raise Bf16DerivationError("source has a truncated Safetensors header")
    raw_header = _read_exact(stream, header_size, "Safetensors header")
    header = _parse_header(raw_header)
    data_size = file_size - 8 - header_size
    layouts = _tensor_layouts(header, data_size)
    unexpected = sorted({layout.dtype for layout in layouts}.difference({"F32"}))
    if unexpected:
        raise Bf16DerivationError(
            "BF16 Oracle derivation requires an all-F32 source shard; observed "
            + ", ".join(unexpected)
        )
    return prefix, raw_header, header, data_size


def _derived_header(header: Mapping[str, Any]) -> bytes:
    derived: dict[str, Any] = {}
    for name, entry in header.items():
        if name == "__metadata__":
            derived[name] = dict(entry)
            continue
        begin, end = entry["data_offsets"]
        if begin % 4 or end % 4:
            raise Bf16DerivationError(
                f"tensor {name!r} F32 offsets must be divisible by four"
            )
        derived[name] = {
            "dtype": "BF16",
            "shape": list(entry["shape"]),
            "data_offsets": [begin // 2, end // 2],
        }
    encoded = json.dumps(
        derived, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    padding = (-len(encoded)) % 8
    return encoded + (b" " * padding)


def _hash_existing_prefix(path: Path, expected_header: bytes) -> tuple[Any, int]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        prefix = _read_exact(stream, 8, "partial BF16 length prefix")
        header_size = int.from_bytes(prefix, "little")
        raw_header = _read_exact(stream, header_size, "partial BF16 header")
        if raw_header != expected_header:
            raise Bf16DerivationError(
                f"partial artifact header does not match this transform: {path}"
            )
        digest.update(prefix)
        digest.update(raw_header)
        total = 8 + header_size
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
            total += len(chunk)
    return digest, total


def convert_safetensors_f32_to_bf16(
    source: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    expected_source_sha256: str,
    chunk_bytes: int = DEFAULT_CONVERSION_CHUNK_BYTES,
) -> dict[str, Any]:
    """Derive one shard, resuming a compatible ``.part`` file if present."""

    chunk_bytes = _positive_aligned_chunk_size(chunk_bytes)
    source_path = Path(source)
    destination_path = Path(destination)
    partial_path = destination_path.with_name(destination_path.name + ".part")
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    if destination_path.is_symlink() or partial_path.is_symlink():
        raise Bf16DerivationError(
            f"refusing symlink destination or partial artifact: {destination_path}"
        )

    try:
        with source_path.open("rb") as source_stream:
            source_stat = os.fstat(source_stream.fileno())
            if not stat.S_ISREG(source_stat.st_mode):
                raise Bf16DerivationError(f"source is not a regular file: {source_path}")
            source_prefix, source_header, header, source_data_size = _read_source_header(
                source_stream, source_stat.st_size
            )
            if source_data_size % 4:
                raise Bf16DerivationError("F32 data section size is not divisible by four")
            target_header = _derived_header(header)
            target_prefix = len(target_header).to_bytes(8, "little")
            target_header_bytes = target_prefix + target_header
            target_data_size = source_data_size // 2

            source_digest = hashlib.sha256()
            source_digest.update(source_prefix)
            source_digest.update(source_header)

            if destination_path.exists():
                if partial_path.exists():
                    raise Bf16DerivationError(
                        f"both final and partial artifacts exist: {destination_path}"
                    )
                target_digest, observed_size = _hash_existing_prefix(
                    destination_path, target_header
                )
                if observed_size != len(target_header_bytes) + target_data_size:
                    raise Bf16DerivationError(
                        f"existing destination has the wrong size: {destination_path}"
                    )
                with destination_path.open("rb") as target_stream:
                    target_stream.seek(len(target_header_bytes))
                    remaining = source_data_size
                    while remaining:
                        chunk = source_stream.read(min(remaining, chunk_bytes))
                        if not chunk:
                            raise Bf16DerivationError(
                                "source truncated while validating existing derivation"
                            )
                        source_digest.update(chunk)
                        expected = f32_bytes_to_bf16_rne(chunk)
                        observed = _read_exact(
                            target_stream,
                            len(expected),
                            "existing BF16 tensor data",
                        )
                        if observed != expected:
                            raise Bf16DerivationError(
                                f"existing destination is not the deterministic BF16 "
                                f"derivation: {destination_path}"
                            )
                        remaining -= len(chunk)
                after = os.fstat(source_stream.fileno())
                before_identity = (
                    source_stat.st_dev,
                    source_stat.st_ino,
                    source_stat.st_size,
                    source_stat.st_mtime_ns,
                )
                after_identity = (
                    after.st_dev,
                    after.st_ino,
                    after.st_size,
                    after.st_mtime_ns,
                )
                if before_identity != after_identity:
                    raise Bf16DerivationError(
                        f"source changed during conversion: {source_path}"
                    )
                observed_source_sha256 = source_digest.hexdigest()
                if observed_source_sha256 != expected_source_sha256:
                    raise Bf16DerivationError(
                        f"source SHA-256 mismatch for {source_path.name}: expected "
                        f"{expected_source_sha256}, observed {observed_source_sha256}"
                    )
                return {
                    "path": destination_path.name,
                    "size_bytes": observed_size,
                    "sha256": target_digest.hexdigest(),
                }
            if partial_path.exists():
                if not partial_path.is_file():
                    raise Bf16DerivationError(
                        f"partial artifact is not a regular file: {partial_path}"
                    )
                target_digest, partial_size = _hash_existing_prefix(
                    partial_path, target_header
                )
                completed_target_bytes = partial_size - len(target_header_bytes)
                if (
                    completed_target_bytes < 0
                    or completed_target_bytes % 2
                    or completed_target_bytes > target_data_size
                ):
                    raise Bf16DerivationError(
                        f"partial artifact has an invalid data length: {partial_path}"
                    )
                completed_source_bytes = completed_target_bytes * 2
                remaining = completed_source_bytes
                while remaining:
                    chunk = source_stream.read(min(remaining, chunk_bytes))
                    if not chunk:
                        raise Bf16DerivationError("source truncated while validating resume point")
                    source_digest.update(chunk)
                    remaining -= len(chunk)
                output_mode = "ab"
            else:
                target_digest = hashlib.sha256(target_header_bytes)
                completed_source_bytes = 0
                output_mode = "xb"

            with partial_path.open(output_mode) as target_stream:
                if output_mode == "xb":
                    target_stream.write(target_header_bytes)
                remaining = source_data_size - completed_source_bytes
                while remaining:
                    chunk = source_stream.read(min(remaining, chunk_bytes))
                    if not chunk:
                        raise Bf16DerivationError("source truncated during BF16 conversion")
                    source_digest.update(chunk)
                    converted = f32_bytes_to_bf16_rne(chunk)
                    target_stream.write(converted)
                    target_digest.update(converted)
                    remaining -= len(chunk)
                target_stream.flush()
                os.fsync(target_stream.fileno())

            after = os.fstat(source_stream.fileno())
            before_identity = (
                source_stat.st_dev,
                source_stat.st_ino,
                source_stat.st_size,
                source_stat.st_mtime_ns,
            )
            after_identity = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            if before_identity != after_identity:
                raise Bf16DerivationError(f"source changed during conversion: {source_path}")
    except SafetensorsInventoryError as exc:
        raise Bf16DerivationError(str(exc)) from exc

    observed_source_sha256 = source_digest.hexdigest()
    if observed_source_sha256 != expected_source_sha256:
        try:
            partial_path.unlink()
        except FileNotFoundError:
            pass
        raise Bf16DerivationError(
            f"source SHA-256 mismatch for {source_path.name}: expected "
            f"{expected_source_sha256}, observed {observed_source_sha256}"
        )
    expected_size = len(target_header_bytes) + target_data_size
    if partial_path.stat().st_size != expected_size:
        raise Bf16DerivationError(
            f"derived size mismatch for {source_path.name}: expected {expected_size}, "
            f"observed {partial_path.stat().st_size}"
        )
    os.replace(partial_path, destination_path)
    return {
        "path": destination_path.name,
        "size_bytes": expected_size,
        "sha256": target_digest.hexdigest(),
    }


def _verified_copy(source: Path, destination: Path, expected_sha256: str) -> dict[str, Any]:
    observed = file_sha256(source)
    if observed != expected_sha256:
        raise Bf16DerivationError(
            f"source SHA-256 mismatch for {source.name}: expected {expected_sha256}, "
            f"observed {observed}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    if destination.is_symlink() or partial.is_symlink():
        raise Bf16DerivationError(
            f"refusing symlink destination or partial artifact: {destination}"
        )
    if destination.exists():
        destination_sha256 = file_sha256(destination)
        if destination_sha256 != expected_sha256:
            raise Bf16DerivationError(
                f"existing destination SHA-256 mismatch: {destination}"
            )
        return {
            "path": destination.name,
            "size_bytes": destination.stat().st_size,
            "sha256": destination_sha256,
        }
    if partial.exists():
        if partial.is_file() and file_sha256(partial) == expected_sha256:
            os.replace(partial, destination)
            return {
                "path": destination.name,
                "size_bytes": destination.stat().st_size,
                "sha256": expected_sha256,
            }
        raise Bf16DerivationError(f"unrecognized partial non-shard artifact: {partial}")
    with source.open("rb") as source_stream, partial.open("xb") as destination_stream:
        shutil.copyfileobj(source_stream, destination_stream, length=1024 * 1024)
        destination_stream.flush()
        os.fsync(destination_stream.fileno())
    os.replace(partial, destination)
    return {
        "path": destination.name,
        "size_bytes": destination.stat().st_size,
        "sha256": file_sha256(destination),
    }


def _write_json_artifact(destination: Path, document: Mapping[str, Any]) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(document, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    expected_sha256 = hashlib.sha256(encoded).hexdigest()
    partial = destination.with_name(destination.name + ".part")
    if destination.is_symlink() or partial.is_symlink():
        raise Bf16DerivationError(
            f"refusing symlink destination or partial artifact: {destination}"
        )
    if destination.exists():
        if file_sha256(destination) != expected_sha256:
            raise Bf16DerivationError(
                f"existing JSON destination does not match this derivation: {destination}"
            )
        return {
            "path": destination.name,
            "size_bytes": len(encoded),
            "sha256": expected_sha256,
        }
    if partial.exists():
        if partial.is_file() and file_sha256(partial) == expected_sha256:
            os.replace(partial, destination)
            return {
                "path": destination.name,
                "size_bytes": len(encoded),
                "sha256": expected_sha256,
            }
        raise Bf16DerivationError(f"unrecognized partial JSON artifact: {partial}")
    with partial.open("xb") as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, destination)
    return {
        "path": destination.name,
        "size_bytes": len(encoded),
        "sha256": expected_sha256,
    }


def _safetensors_data_size(path: Path) -> int:
    with path.open("rb") as stream:
        header_size = int.from_bytes(
            _read_exact(stream, 8, "Safetensors length prefix"), "little"
        )
    data_size = path.stat().st_size - 8 - header_size
    if data_size < 0:
        raise Bf16DerivationError(f"invalid derived Safetensors size: {path}")
    return data_size


def _artifact_path(root: Path, logical_path: str) -> Path:
    segments = logical_path.split("/")
    if (
        not logical_path
        or logical_path.startswith("/")
        or "\\" in logical_path
        or any(segment in {"", ".", ".."} for segment in segments)
    ):
        raise Bf16DerivationError(f"unsafe artifact path {logical_path!r}")
    path = root.joinpath(*segments)
    try:
        path.resolve(strict=False).relative_to(root.resolve())
    except ValueError as exc:
        raise Bf16DerivationError(
            f"artifact path escapes through a symlink: {logical_path!r}"
        ) from exc
    return path


def derive_bf16_oracle(
    manifest: Mapping[str, Any],
    inventory: Mapping[str, Any],
    source_directory: str | os.PathLike[str],
    destination_directory: str | os.PathLike[str],
    *,
    source_manifest_path: str,
    source_manifest_file_sha256: str,
    tensor_inventory_path: str,
    tensor_inventory_file_sha256: str,
    artifact_root: str,
    chunk_bytes: int = DEFAULT_CONVERSION_CHUNK_BYTES,
    progress: Callable[[str], None] | None = None,
    existing_derivation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Derive the complete BF16 Oracle and return its evidence contract."""

    source_root = Path(source_directory)
    destination_root = Path(destination_directory)
    destination_root.mkdir(parents=True, exist_ok=True)
    shard_inventory = {shard["path"]: shard for shard in inventory["shards"]}
    safetensor_identities = [
        artifact
        for artifact in manifest["weights"]["artifacts"]
        if artifact["path"].endswith(".safetensors")
    ]
    if set(shard_inventory) != {artifact["path"] for artifact in safetensor_identities}:
        raise Bf16DerivationError("TensorInventory shards do not match ModelManifest shards")
    if inventory["observed_dtypes"] != ["F32"]:
        raise Bf16DerivationError(
            "BF16 Oracle derivation requires TensorInventory observed_dtypes == ['F32']"
        )

    weight_artifacts: list[dict[str, Any]] = []
    for identity in safetensor_identities:
        logical_path = identity["path"]
        if progress is not None:
            progress(f"derive {logical_path}")
        derived_identity = convert_safetensors_f32_to_bf16(
            _artifact_path(source_root, logical_path),
            _artifact_path(destination_root, logical_path),
            expected_source_sha256=identity["sha256"],
            chunk_bytes=chunk_bytes,
        )
        derived_identity["path"] = logical_path
        weight_artifacts.append(derived_identity)

    config_identity = manifest["config"]
    config_source = _artifact_path(source_root, config_identity["path"])
    if file_sha256(config_source) != config_identity["sha256"]:
        raise Bf16DerivationError("source config SHA-256 mismatch")
    try:
        config = json.loads(config_source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise Bf16DerivationError(f"cannot parse source config: {exc}") from exc
    if not isinstance(config, dict):
        raise Bf16DerivationError("source config must contain a JSON object")
    config["torch_dtype"] = "bfloat16"
    derived_config = _write_json_artifact(
        _artifact_path(destination_root, config_identity["path"]), config
    )
    derived_config["path"] = config_identity["path"]

    tokenizer_files: list[dict[str, Any]] = []
    for identity in manifest["tokenizer"].get("files", []):
        derived_identity = _verified_copy(
            _artifact_path(source_root, identity["path"]),
            _artifact_path(destination_root, identity["path"]),
            identity["sha256"],
        )
        derived_identity["path"] = identity["path"]
        tokenizer_files.append(derived_identity)

    for identity in manifest["weights"]["artifacts"]:
        if identity["path"].endswith(".safetensors"):
            continue
        source_path = _artifact_path(source_root, identity["path"])
        if identity["path"].endswith(".index.json"):
            if file_sha256(source_path) != identity["sha256"]:
                raise Bf16DerivationError("source Safetensors index SHA-256 mismatch")
            try:
                index = json.loads(source_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise Bf16DerivationError(f"cannot parse Safetensors index: {exc}") from exc
            if not isinstance(index, dict):
                raise Bf16DerivationError("Safetensors index must contain a JSON object")
            metadata = index.get("metadata")
            if isinstance(metadata, dict) and "total_size" in metadata:
                metadata["total_size"] = sum(
                    _safetensors_data_size(
                        _artifact_path(destination_root, artifact["path"])
                    )
                    for artifact in weight_artifacts
                    if artifact["path"].endswith(".safetensors")
                )
            derived_identity = _write_json_artifact(
                _artifact_path(destination_root, identity["path"]), index
            )
            derived_identity["path"] = identity["path"]
            weight_artifacts.append(derived_identity)
        else:
            derived_identity = _verified_copy(
                source_path,
                _artifact_path(destination_root, identity["path"]),
                identity["sha256"],
            )
            derived_identity["path"] = identity["path"]
            weight_artifacts.append(derived_identity)

    document = {
        "schema_version": 1,
        "kind": "model_derivation",
        "generated_at": datetime.now(timezone.utc)
        .isoformat(timespec="seconds")
        .replace("+00:00", "Z"),
        "derivation_id": "olmoe-1b-7b-0125-bf16-rne-v1",
        "status": "verified",
        "source": {
            "model_id": manifest["model_id"],
            "model_revision": manifest["model_revision"],
            "model_manifest_path": source_manifest_path,
            "model_manifest_sha256": source_manifest_file_sha256,
            "tensor_inventory_path": tensor_inventory_path,
            "tensor_inventory_sha256": tensor_inventory_file_sha256,
        },
        "transform": {
            "id": "f32_to_bf16_rne_v1",
            "source_dtype": "F32",
            "target_dtype": "BF16",
            "rounding": "round_to_nearest_even",
            "nan_policy": "preserve_nan",
            "implementation": "uma_qmoe.bf16",
            "chunk_bytes": _positive_aligned_chunk_size(chunk_bytes),
        },
        "artifact_root": artifact_root,
        "config": derived_config,
        "tokenizer_files": tokenizer_files,
        "weights": {
            "format": "safetensors",
            "observed_dtypes": ["BF16"],
            "artifacts": weight_artifacts,
            "total_artifact_bytes": sum(
                artifact["size_bytes"] for artifact in weight_artifacts
            ),
        },
    }
    # A repeated CLI run against the same output path must not change that
    # file's SHA merely because the wall clock advanced. Downstream evidence
    # binds this file by SHA-256, so preserve the original operational
    # timestamp when the validated semantic identity is unchanged.
    document_identity = canonical_sha256(document)
    if existing_derivation is not None:
        validate_document(existing_derivation)
        if canonical_sha256(existing_derivation) == document_identity:
            document["generated_at"] = existing_derivation["generated_at"]
    return document
