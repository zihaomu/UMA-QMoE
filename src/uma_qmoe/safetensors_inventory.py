"""Validate and inventory local Safetensors shards without third-party code.

Safetensors offsets are relative to the beginning of the data section (the
first byte after the eight-byte length prefix and JSON header).  This module
keeps that convention in the ``offset_bytes`` field it returns.

The scanner deliberately opens a shard once and reads it only in the forward
direction.  The whole-file SHA-256 and each tensor payload SHA-256 are updated
from that same stream, so even multi-gigabyte tensors are never materialized in
memory.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO


DEFAULT_MAX_HEADER_BYTES = 100_000_000
DEFAULT_CHUNK_BYTES = 1024 * 1024

# This scanner deliberately supports the byte-aligned subset of the current
# Safetensors dtype enum.  Sub-byte F4/F6 values fail closed until bit packing
# and alignment are implemented and tested here.
_DTYPE_BYTES = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "F8_E8M0": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2FNUZ": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "C64": 8,
    "I64": 8,
    "U64": 8,
    "F64": 8,
}
_LOWER_HEX = frozenset("0123456789abcdef")


class SafetensorsInventoryError(ValueError):
    """Raised when a shard or its ModelManifest binding cannot be verified."""


@dataclass(frozen=True)
class _TensorLayout:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    size: int


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise SafetensorsInventoryError(f"{name} must be a positive integer")
    return value


def _read_exact(stream: BinaryIO, count: int, description: str) -> bytes:
    chunks: list[bytes] = []
    remaining = count
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            received = count - remaining
            raise SafetensorsInventoryError(
                f"truncated {description}: expected {count} bytes, received {received}"
            )
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _reject_json_constant(value: str) -> None:
    raise SafetensorsInventoryError(f"header contains invalid JSON constant {value!r}")


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise SafetensorsInventoryError(f"header contains duplicate JSON key {key!r}")
        result[key] = value
    return result


def _parse_header(raw_header: bytes) -> dict[str, Any]:
    if not raw_header.startswith(b"{"):
        raise SafetensorsInventoryError(
            "Safetensors header must begin with the JSON object byte '{'"
        )
    try:
        text = raw_header.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise SafetensorsInventoryError(f"header is not valid UTF-8: {exc}") from exc

    try:
        value = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_json_constant,
        )
    except SafetensorsInventoryError:
        raise
    except json.JSONDecodeError as exc:
        raise SafetensorsInventoryError(f"header is not valid JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise SafetensorsInventoryError("Safetensors header must be a JSON object")
    return value


def _tensor_layouts(header: Mapping[str, Any], data_size: int) -> list[_TensorLayout]:
    metadata = header.get("__metadata__")
    if metadata is not None:
        if not isinstance(metadata, Mapping) or any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in metadata.items()
        ):
            raise SafetensorsInventoryError(
                "header __metadata__ must be an object with string keys and values"
            )

    layouts: list[_TensorLayout] = []
    for name, raw_entry in header.items():
        if name == "__metadata__":
            continue
        if not name:
            raise SafetensorsInventoryError("tensor names must be non-empty")
        if not isinstance(raw_entry, Mapping):
            raise SafetensorsInventoryError(f"tensor {name!r} metadata must be an object")
        unexpected_fields = sorted(set(raw_entry).difference({"dtype", "shape", "data_offsets"}))
        if unexpected_fields:
            raise SafetensorsInventoryError(
                f"tensor {name!r} metadata contains unexpected fields: "
                + ", ".join(repr(field) for field in unexpected_fields)
            )

        dtype = raw_entry.get("dtype")
        if not isinstance(dtype, str) or dtype not in _DTYPE_BYTES:
            rendered = repr(dtype) if isinstance(dtype, str) else type(dtype).__name__
            raise SafetensorsInventoryError(
                f"tensor {name!r} has unsupported or invalid dtype {rendered}"
            )

        raw_shape = raw_entry.get("shape")
        if not isinstance(raw_shape, list):
            raise SafetensorsInventoryError(f"tensor {name!r} shape must be an array")
        shape: list[int] = []
        element_count = 1
        for index, dimension in enumerate(raw_shape):
            if (
                isinstance(dimension, bool)
                or not isinstance(dimension, int)
                or dimension < 0
            ):
                raise SafetensorsInventoryError(
                    f"tensor {name!r} shape[{index}] must be a non-negative integer"
                )
            shape.append(dimension)
            element_count *= dimension

        raw_offsets = raw_entry.get("data_offsets")
        if not isinstance(raw_offsets, list) or len(raw_offsets) != 2:
            raise SafetensorsInventoryError(
                f"tensor {name!r} data_offsets must be a two-integer array"
            )
        if any(
            isinstance(offset, bool) or not isinstance(offset, int)
            for offset in raw_offsets
        ):
            raise SafetensorsInventoryError(
                f"tensor {name!r} data_offsets must be a two-integer array"
            )
        begin, end = raw_offsets
        if begin < 0 or end < begin:
            raise SafetensorsInventoryError(
                f"tensor {name!r} has invalid data_offsets [{begin}, {end}]"
            )

        size = end - begin
        expected_size = element_count * _DTYPE_BYTES[dtype]
        if size != expected_size:
            raise SafetensorsInventoryError(
                f"tensor {name!r} shape/dtype requires {expected_size} bytes, "
                f"but data_offsets describe {size}"
            )
        if end > data_size:
            raise SafetensorsInventoryError(
                f"tensor {name!r} data_offsets [{begin}, {end}] exceed the "
                f"{data_size}-byte data section"
            )
        layouts.append(
            _TensorLayout(
                name=name,
                dtype=dtype,
                shape=tuple(shape),
                offset=begin,
                size=size,
            )
        )

    ordered = sorted(
        layouts,
        key=lambda layout: (layout.offset, layout.offset + layout.size, layout.name),
    )
    cursor = 0
    for layout in ordered:
        if layout.offset < cursor:
            raise SafetensorsInventoryError(
                f"tensor payloads overlap before {layout.name!r}: "
                f"offset {layout.offset} is below prior end {cursor}"
            )
        if layout.offset > cursor:
            raise SafetensorsInventoryError(
                f"Safetensors data section contains an unindexed gap "
                f"[{cursor}, {layout.offset}] before tensor {layout.name!r}"
            )
        cursor = layout.offset + layout.size
    if cursor != data_size:
        raise SafetensorsInventoryError(
            f"Safetensors data section contains an unindexed trailing gap "
            f"[{cursor}, {data_size}]"
        )
    return ordered


def _stream_bytes(
    stream: BinaryIO,
    count: int,
    *,
    file_digest: Any,
    payload_digest: Any | None,
    chunk_size: int,
    description: str,
) -> None:
    remaining = count
    while remaining:
        chunk = stream.read(min(remaining, chunk_size))
        if not chunk:
            received = count - remaining
            raise SafetensorsInventoryError(
                f"truncated {description}: expected {count} bytes, received {received}"
            )
        file_digest.update(chunk)
        if payload_digest is not None:
            payload_digest.update(chunk)
        remaining -= len(chunk)


def inventory_safetensors(
    path: str | os.PathLike[str],
    *,
    shard_path: str | None = None,
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
    chunk_size: int = DEFAULT_CHUNK_BYTES,
) -> dict[str, Any]:
    """Validate a shard and return deterministic file/tensor SHA-256 inventory.

    ``offset_bytes`` in each tensor result is relative to the data section, as
    defined by the Safetensors file format.  ``shard_path`` can provide the
    logical repository-relative path while ``path`` points at the local file.
    """

    max_header_bytes = _positive_integer(max_header_bytes, "max_header_bytes")
    chunk_size = _positive_integer(chunk_size, "chunk_size")
    local_path = Path(path)
    logical_path = str(local_path) if shard_path is None else shard_path
    if not isinstance(logical_path, str) or not logical_path:
        raise SafetensorsInventoryError("shard_path must be a non-empty string")

    file_digest = hashlib.sha256()
    try:
        with local_path.open("rb") as stream:
            before = os.fstat(stream.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise SafetensorsInventoryError(f"shard is not a regular file: {local_path}")
            file_size = before.st_size
            if file_size < 8:
                raise SafetensorsInventoryError(
                    f"truncated Safetensors prefix in {local_path}: file is {file_size} bytes"
                )

            prefix = _read_exact(stream, 8, "Safetensors length prefix")
            file_digest.update(prefix)
            header_size = int.from_bytes(prefix, byteorder="little", signed=False)
            if header_size == 0:
                raise SafetensorsInventoryError("Safetensors header length must be non-zero")
            if header_size > max_header_bytes:
                raise SafetensorsInventoryError(
                    f"Safetensors header declares {header_size} bytes, exceeding the "
                    f"{max_header_bytes}-byte limit"
                )
            if header_size > file_size - 8:
                raise SafetensorsInventoryError(
                    f"truncated Safetensors header: declares {header_size} bytes, "
                    f"but only {file_size - 8} remain"
                )

            raw_header = _read_exact(stream, header_size, "Safetensors header")
            file_digest.update(raw_header)
            header = _parse_header(raw_header)
            data_size = file_size - 8 - header_size
            layouts = _tensor_layouts(header, data_size)

            payload_digests = {
                layout.name: hashlib.sha256() for layout in layouts
            }
            cursor = 0
            occupied = [layout for layout in layouts if layout.size]
            for layout in occupied:
                gap = layout.offset - cursor
                _stream_bytes(
                    stream,
                    gap,
                    file_digest=file_digest,
                    payload_digest=None,
                    chunk_size=chunk_size,
                    description="Safetensors data gap",
                )
                _stream_bytes(
                    stream,
                    layout.size,
                    file_digest=file_digest,
                    payload_digest=payload_digests[layout.name],
                    chunk_size=chunk_size,
                    description=f"tensor {layout.name!r} payload",
                )
                cursor = layout.offset + layout.size
            _stream_bytes(
                stream,
                data_size - cursor,
                file_digest=file_digest,
                payload_digest=None,
                chunk_size=chunk_size,
                description="Safetensors trailing data",
            )

            after = os.fstat(stream.fileno())
            before_identity = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            )
            after_identity = (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            if after_identity != before_identity:
                raise SafetensorsInventoryError(
                    f"shard changed while it was being inventoried: {local_path}"
                )
    except SafetensorsInventoryError:
        raise
    except OSError as exc:
        raise SafetensorsInventoryError(f"cannot read shard {local_path}: {exc}") from exc

    return {
        "path": logical_path,
        "size_bytes": file_size,
        "file_sha256": file_digest.hexdigest(),
        "tensors": [
            {
                "name": layout.name,
                "dtype": layout.dtype,
                "shape": list(layout.shape),
                "offset_bytes": layout.offset,
                "size_bytes": layout.size,
                "payload_sha256": payload_digests[layout.name].hexdigest(),
            }
            for layout in layouts
        ],
    }


def _manifest_shards(manifest: Mapping[str, Any]) -> list[tuple[str, int, str]]:
    if manifest.get("kind") != "model_manifest":
        raise SafetensorsInventoryError("document kind must be 'model_manifest'")
    weights = manifest.get("weights")
    if not isinstance(weights, Mapping):
        raise SafetensorsInventoryError("ModelManifest weights must be an object")
    if weights.get("format") != "safetensors":
        raise SafetensorsInventoryError("ModelManifest weights.format must be 'safetensors'")
    artifacts = weights.get("artifacts")
    if not isinstance(artifacts, list):
        raise SafetensorsInventoryError("ModelManifest weights.artifacts must be an array")

    shards: list[tuple[str, int, str]] = []
    seen: set[str] = set()
    for position, artifact in enumerate(artifacts):
        if not isinstance(artifact, Mapping):
            raise SafetensorsInventoryError(
                f"ModelManifest weights.artifacts[{position}] must be an object"
            )
        artifact_path = artifact.get("path")
        if not isinstance(artifact_path, str) or not artifact_path:
            raise SafetensorsInventoryError(
                f"ModelManifest weights.artifacts[{position}].path must be non-empty"
            )
        if not artifact_path.endswith(".safetensors"):
            continue
        segments = artifact_path.split("/")
        if (
            artifact_path.startswith("/")
            or artifact_path.endswith("/")
            or "\\" in artifact_path
            or any(segment in {"", ".", ".."} for segment in segments)
        ):
            raise SafetensorsInventoryError(
                f"unsafe Safetensors artifact path {artifact_path!r}"
            )
        if artifact_path in seen:
            raise SafetensorsInventoryError(
                f"duplicate Safetensors artifact path {artifact_path!r}"
            )
        seen.add(artifact_path)

        expected_size = artifact.get("size_bytes")
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size <= 0
        ):
            raise SafetensorsInventoryError(
                f"artifact {artifact_path!r} has invalid size_bytes"
            )
        expected_sha = artifact.get("sha256")
        if not (
            isinstance(expected_sha, str)
            and len(expected_sha) == 64
            and all(character in _LOWER_HEX for character in expected_sha)
        ):
            raise SafetensorsInventoryError(
                f"artifact {artifact_path!r} has invalid SHA-256"
            )
        shards.append((artifact_path, expected_size, expected_sha))
    if not shards:
        raise SafetensorsInventoryError(
            "ModelManifest contains no .safetensors weight artifacts"
        )
    return shards


def verify_model_manifest_artifacts(
    manifest: Mapping[str, Any],
    model_directory: str | os.PathLike[str],
    *,
    reject_unlisted: bool = True,
    max_header_bytes: int = DEFAULT_MAX_HEADER_BYTES,
    chunk_size: int = DEFAULT_CHUNK_BYTES,
) -> dict[str, Any]:
    """Verify every Safetensors artifact bound by a ModelManifest.

    By default the model directory is a closed world: an unlisted
    ``*.safetensors`` file is rejected instead of silently mixing revisions.
    Set ``reject_unlisted=False`` only when the caller deliberately stores
    multiple model revisions in the same directory.
    """

    if not isinstance(manifest, Mapping):
        raise SafetensorsInventoryError("ModelManifest must be a mapping")
    if not isinstance(reject_unlisted, bool):
        raise SafetensorsInventoryError("reject_unlisted must be a boolean")
    directory = Path(model_directory)
    if not directory.is_dir():
        raise SafetensorsInventoryError(
            f"model directory does not exist or is not a directory: {directory}"
        )
    shards = _manifest_shards(manifest)

    local_paths: dict[str, Path] = {}
    resolved_directory = directory.resolve()
    for artifact_path, expected_size, _expected_sha in shards:
        local_path = directory.joinpath(*artifact_path.split("/"))
        try:
            local_path.resolve(strict=False).relative_to(resolved_directory)
        except ValueError as exc:
            raise SafetensorsInventoryError(
                f"Safetensors artifact escapes through a symlink: {artifact_path!r}"
            ) from exc
        if local_path.is_symlink():
            raise SafetensorsInventoryError(
                f"Safetensors artifact must not be a symlink: {artifact_path!r}"
            )
        if not local_path.exists():
            raise SafetensorsInventoryError(
                f"missing Safetensors artifact {artifact_path!r} in {directory}"
            )
        if not local_path.is_file():
            raise SafetensorsInventoryError(
                f"Safetensors artifact is not a regular file: {artifact_path!r}"
            )
        try:
            observed_size = local_path.stat().st_size
        except OSError as exc:
            raise SafetensorsInventoryError(
                f"cannot stat Safetensors artifact {artifact_path!r}: {exc}"
            ) from exc
        if observed_size != expected_size:
            raise SafetensorsInventoryError(
                f"size mismatch for Safetensors artifact {artifact_path!r}: "
                f"expected {expected_size}, observed {observed_size}"
            )
        local_paths[artifact_path] = local_path

    if reject_unlisted:
        try:
            discovered = {
                candidate.relative_to(directory).as_posix()
                for candidate in directory.rglob("*.safetensors")
                if candidate.is_file()
            }
        except OSError as exc:
            raise SafetensorsInventoryError(
                f"cannot enumerate Safetensors artifacts in {directory}: {exc}"
            ) from exc
        unexpected = sorted(discovered.difference(local_paths))
        if unexpected:
            raise SafetensorsInventoryError(
                "unlisted Safetensors artifacts in model directory: "
                + ", ".join(repr(path) for path in unexpected)
            )

    inventories: list[dict[str, Any]] = []
    observed_dtypes: set[str] = set()
    for artifact_path, expected_size, expected_sha in shards:
        inventory = inventory_safetensors(
            local_paths[artifact_path],
            shard_path=artifact_path,
            max_header_bytes=max_header_bytes,
            chunk_size=chunk_size,
        )
        if inventory["size_bytes"] != expected_size:
            raise SafetensorsInventoryError(
                f"size mismatch for Safetensors artifact {artifact_path!r}: "
                f"expected {expected_size}, observed {inventory['size_bytes']}"
            )
        observed_sha = inventory["file_sha256"]
        if not hmac.compare_digest(observed_sha, expected_sha):
            raise SafetensorsInventoryError(
                f"SHA-256 mismatch for Safetensors artifact {artifact_path!r}: "
                f"expected {expected_sha}, observed {observed_sha}"
            )
        inventories.append(inventory)
        observed_dtypes.update(tensor["dtype"] for tensor in inventory["tensors"])

    return {
        "model_directory": str(directory),
        "shards": inventories,
        "tensor_count": sum(len(inventory["tensors"]) for inventory in inventories),
        "observed_dtypes": sorted(observed_dtypes),
    }


verify_manifest_safetensors = verify_model_manifest_artifacts


__all__ = [
    "DEFAULT_CHUNK_BYTES",
    "DEFAULT_MAX_HEADER_BYTES",
    "SafetensorsInventoryError",
    "inventory_safetensors",
    "verify_manifest_safetensors",
    "verify_model_manifest_artifacts",
]
