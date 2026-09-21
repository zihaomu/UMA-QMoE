"""Canonical, mmap-friendly ExpertPack v1 binary container."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import hashlib
import json
import math
import mmap
from pathlib import Path
import re
import struct
import tempfile
from typing import Any, BinaryIO

from .contracts import ContractError, validate_document
from .q4 import DEFAULT_GROUP_SIZE, Q4Tensor, align_up, q4_layout, quantize_q4


MAGIC = b"UQMOEPK1"
FORMAT_VERSION = 1
HEADER_ALIGNMENT = 4096
TENSOR_ALIGNMENT = 64
PREFIX = struct.Struct("<8sIIQQ32s32s")
_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_OLMOE_EXPERT = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(down_proj|gate_proj|up_proj)\.weight$"
)
_QWEN_EXPERT = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(down_proj|gate_proj|up_proj)\.weight$"
)
_RESERVED_NIBBLE_TABLE = bytes(
    1 if (value & 0x0F) == 8 or (value >> 4) == 8 else 0 for value in range(256)
)
_SCAN_CHUNK_BYTES = 8 * 1024 * 1024


def _ordered_olmoe_experts(weight_map: Mapping[str, str]) -> list[tuple[str, str]]:
    projection_order = ("gate_proj", "up_proj", "down_proj")
    expected = [
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
        for layer in range(16)
        for expert in range(64)
        for projection in projection_order
    ]
    observed = {name for name in weight_map if ".mlp.experts." in name}
    if observed != set(expected):
        missing = sorted(set(expected) - observed)
        unexpected = sorted(observed - set(expected))
        raise ContractError(
            "OLMoE expert tensor set mismatch: "
            f"missing={missing[:8]!r}, unexpected={unexpected[:8]!r}"
        )
    return [(name, weight_map[name]) for name in expected]


def _validate_olmoe_expert_tensor(name: str, tensor: Any) -> None:
    match = _OLMOE_EXPERT.fullmatch(name)
    if match is None:
        raise ContractError(f"unexpected OLMoE expert tensor name {name!r}")
    projection = match.group(3)
    expected_shape = (2048, 1024) if projection == "down_proj" else (1024, 2048)
    if tuple(tensor.shape) != expected_shape:
        raise ContractError(
            f"OLMoE expert tensor {name!r} shape mismatch: "
            f"expected {expected_shape}, observed {tuple(tensor.shape)}"
        )
    if str(tensor.dtype) != "torch.bfloat16":
        raise ContractError(
            f"OLMoE expert tensor {name!r} must be BF16, observed {tensor.dtype}"
        )


def _ordered_qwen_experts(weight_map: Mapping[str, str]) -> list[tuple[str, str]]:
    projection_order = ("gate_proj", "up_proj", "down_proj")
    expected = [
        f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight"
        for layer in range(24)
        for expert in range(60)
        for projection in projection_order
    ]
    observed = {name for name in weight_map if ".mlp.experts." in name}
    if observed != set(expected):
        missing = sorted(set(expected) - observed)
        unexpected = sorted(observed - set(expected))
        raise ContractError(
            "Qwen expert tensor set mismatch: "
            f"missing={missing[:8]!r}, unexpected={unexpected[:8]!r}"
        )
    return [(name, weight_map[name]) for name in expected]


def _validate_qwen_expert_tensor(name: str, tensor: Any) -> None:
    match = _QWEN_EXPERT.fullmatch(name)
    if match is None:
        raise ContractError(f"unexpected Qwen expert tensor name {name!r}")
    projection = match.group(3)
    expected_shape = (2048, 1408) if projection == "down_proj" else (1408, 2048)
    if tuple(tensor.shape) != expected_shape:
        raise ContractError(
            f"Qwen expert tensor {name!r} shape mismatch: "
            f"expected {expected_shape}, observed {tuple(tensor.shape)}"
        )
    if str(tensor.dtype) != "torch.bfloat16":
        raise ContractError(
            f"Qwen expert tensor {name!r} must be BF16, observed {tensor.dtype}"
        )


def _sha256_bytes(payload: bytes | memoryview) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_regions(*regions: memoryview) -> str:
    digest = hashlib.sha256()
    for region in regions:
        digest.update(region)
    return digest.hexdigest()


def _require_zero_region(mapping: mmap.mmap, start: int, stop: int, label: str) -> None:
    for offset in range(start, stop, _SCAN_CHUNK_BYTES):
        chunk = mapping[offset : min(stop, offset + _SCAN_CHUNK_BYTES)]
        if chunk.strip(b"\0"):
            raise ContractError(f"ExpertPack {label} padding must be zero")


def _validate_q4_semantics(
    packed: memoryview, scales: memoryview, element_count: int, name: str
) -> None:
    for offset in range(0, len(packed), _SCAN_CHUNK_BYTES):
        chunk = bytes(packed[offset : offset + _SCAN_CHUNK_BYTES])
        if chunk.translate(_RESERVED_NIBBLE_TABLE).find(b"\x01") != -1:
            raise ContractError(f"ExpertPack tensor {name!r} contains reserved Q4 -8")
    if element_count % 2 and (packed[-1] >> 4) != 0:
        raise ContractError(f"ExpertPack tensor {name!r} has non-zero Q4 tail padding")
    if len(scales) % 4:
        raise ContractError(f"ExpertPack tensor {name!r} scale payload is misaligned")
    for (scale,) in struct.iter_unpack("<f", scales):
        if not math.isfinite(scale) or scale <= 0:
            raise ContractError(
                f"ExpertPack tensor {name!r} scales must be finite and positive"
            )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _model_identity(model_id: str, model_revision: str, manifest_sha256: str) -> str:
    value = {
        "model_id": model_id,
        "model_revision": model_revision,
        "model_manifest_sha256": manifest_sha256,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _write_zeros(stream: BinaryIO, count: int) -> None:
    block = bytes(64 * 1024)
    while count:
        size = min(count, len(block))
        stream.write(block[:size])
        count -= size


def write_expert_pack(
    destination: str | Path,
    tensors: Iterable[tuple[str, Any]],
    *,
    model_id: str,
    model_revision: str,
    model_manifest_sha256: str,
    group_size: int = DEFAULT_GROUP_SIZE,
    tensor_alignment_bytes: int = TENSOR_ALIGNMENT,
) -> dict[str, Any]:
    """Quantize and stream tensors into a new ExpertPack without a second pack copy."""

    if len(model_revision) != 40 or any(c not in "0123456789abcdef" for c in model_revision):
        raise ContractError("ExpertPack model revision must be a lowercase git SHA")
    if len(model_manifest_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in model_manifest_sha256
    ):
        raise ContractError("ExpertPack model manifest identity must be SHA-256")
    # Reuse the shared alignment validator even when the iterable is empty.
    align_up(0, tensor_alignment_bytes)
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ContractError(f"refusing to overwrite existing ExpertPack {target}")

    tensor_headers: list[dict[str, Any]] = []
    seen: set[str] = set()
    payload_hasher = hashlib.sha256()
    payload_size = 0
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.", suffix=".payload", dir=target.parent, delete=False
        ) as payload_stream:
            temporary = Path(payload_stream.name)

            def payload_write(data: bytes) -> None:
                nonlocal payload_size
                payload_stream.write(data)
                payload_hasher.update(data)
                payload_size += len(data)

            for name, values in tensors:
                if not isinstance(name, str) or not _NAME.fullmatch(name):
                    raise ContractError(f"invalid ExpertPack tensor name {name!r}")
                if name in seen:
                    raise ContractError(f"duplicate ExpertPack tensor name {name!r}")
                seen.add(name)
                tensor = quantize_q4(values, group_size=group_size)
                tensor_start = align_up(payload_size, tensor_alignment_bytes)
                payload_write(bytes(tensor_start - payload_size))
                packed_offset = payload_size
                payload_write(tensor.packed)
                scale_offset = align_up(payload_size, 4)
                payload_write(bytes(scale_offset - payload_size))
                payload_write(tensor.scales)
                storage_end = align_up(payload_size, tensor_alignment_bytes)
                payload_write(bytes(storage_end - payload_size))
                layout = q4_layout(
                    tensor.element_count, group_size, tensor_alignment_bytes
                )
                tensor_payload_sha256 = hashlib.sha256(
                    tensor.packed + tensor.scales
                ).hexdigest()
                tensor_headers.append(
                    {
                        "name": name,
                        "shape": list(tensor.shape),
                        "element_count": tensor.element_count,
                        "group_count": layout.group_count,
                        "packed_offset": packed_offset,
                        "packed_bytes": len(tensor.packed),
                        "scale_offset": scale_offset,
                        "scale_bytes": len(tensor.scales),
                        "storage_end_offset": storage_end,
                        "effective_bits_per_weight": (
                            (storage_end - tensor_start) * 8 / tensor.element_count
                        ),
                        "tensor_payload_sha256": tensor_payload_sha256,
                    }
                )
            payload_stream.flush()
        if not tensor_headers:
            raise ContractError("ExpertPack must contain at least one tensor")

        header = {
            "schema_version": 1,
            "format": "canonical_q4_symmetric_grouped",
            "quantization": {
                "bits": 4,
                "signed_range": [-7, 7],
                "group_size": group_size,
                "scale_dtype": "float32_le",
                "rounding": "nearest_even",
                "zero_point": None,
                "nibble_order": "low_first_twos_complement",
            },
            "model": {
                "model_id": model_id,
                "model_revision": model_revision,
                "model_manifest_sha256": model_manifest_sha256,
                "model_identity_sha256": _model_identity(
                    model_id, model_revision, model_manifest_sha256
                ),
            },
            "tensor_alignment_bytes": tensor_alignment_bytes,
            "tensor_count": len(tensor_headers),
            "tensors": tensor_headers,
        }
        header_bytes = json.dumps(
            header, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        payload_offset = align_up(PREFIX.size + len(header_bytes), HEADER_ALIGNMENT)
        prefix = PREFIX.pack(
            MAGIC,
            FORMAT_VERSION,
            len(header_bytes),
            payload_offset,
            payload_size,
            hashlib.sha256(header_bytes).digest(),
            payload_hasher.digest(),
        )
        with target.open("xb") as output, temporary.open("rb") as payload_stream:
            output.write(prefix)
            output.write(header_bytes)
            _write_zeros(output, payload_offset - PREFIX.size - len(header_bytes))
            while chunk := payload_stream.read(8 * 1024 * 1024):
                output.write(chunk)
            output.flush()
        return inspect_expert_pack(target)
    except BaseException:
        if target.exists():
            target.unlink()
        raise
    finally:
        if temporary is not None:
            try:
                temporary.unlink()
            except FileNotFoundError:
                pass


class ExpertPackReader:
    """One read-only file descriptor and exactly one mmap per pack."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_model_id: str | None = None,
        expected_model_revision: str | None = None,
        expected_model_manifest_sha256: str | None = None,
    ) -> None:
        self.path = Path(path)
        try:
            self._file = self.path.open("rb")
            self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except OSError as exc:
            raise ContractError(f"cannot map ExpertPack {self.path}: {exc}") from exc
        self.mapping_count = 1
        try:
            self.header, self.payload_offset, self.payload_length = self._validate(
                expected_model_id=expected_model_id,
                expected_model_revision=expected_model_revision,
                expected_model_manifest_sha256=expected_model_manifest_sha256,
            )
        except BaseException:
            self.close()
            raise
        self._tensors = {item["name"]: item for item in self.header["tensors"]}

    def _validate(
        self,
        *,
        expected_model_id: str | None,
        expected_model_revision: str | None,
        expected_model_manifest_sha256: str | None,
    ) -> tuple[dict[str, Any], int, int]:
        if len(self._mapping) < PREFIX.size:
            raise ContractError("ExpertPack is shorter than its fixed prefix")
        (
            magic,
            version,
            header_length,
            payload_offset,
            payload_length,
            header_digest,
            payload_digest,
        ) = PREFIX.unpack_from(self._mapping)
        if magic != MAGIC or version != FORMAT_VERSION:
            raise ContractError("ExpertPack magic/version mismatch")
        if payload_offset % HEADER_ALIGNMENT or payload_offset < PREFIX.size + header_length:
            raise ContractError("ExpertPack payload offset/alignment is invalid")
        if payload_offset + payload_length != len(self._mapping):
            raise ContractError("ExpertPack payload length does not match file size")
        header_bytes = self._mapping[PREFIX.size : PREFIX.size + header_length]
        if hashlib.sha256(header_bytes).digest() != header_digest:
            raise ContractError("ExpertPack header hash mismatch")
        padding = self._mapping[PREFIX.size + header_length : payload_offset]
        if any(padding):
            raise ContractError("ExpertPack header padding must be zero")
        payload = memoryview(self._mapping)[payload_offset : payload_offset + payload_length]
        try:
            if hashlib.sha256(payload).digest() != payload_digest:
                raise ContractError("ExpertPack payload hash mismatch")
        finally:
            payload.release()
        try:
            header = json.loads(header_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError("ExpertPack header is not canonical JSON") from exc
        if not isinstance(header, dict) or header.get("schema_version") != 1:
            raise ContractError("ExpertPack header schema is unsupported")
        canonical = json.dumps(
            header, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        if canonical != header_bytes:
            raise ContractError("ExpertPack header JSON is not canonical")
        model = header.get("model")
        quantization = header.get("quantization")
        tensors = header.get("tensors")
        if not isinstance(model, dict) or not isinstance(quantization, dict) or not isinstance(tensors, list):
            raise ContractError("ExpertPack header is incomplete")
        expected_identity = _model_identity(
            model.get("model_id", ""),
            model.get("model_revision", ""),
            model.get("model_manifest_sha256", ""),
        )
        if model.get("model_identity_sha256") != expected_identity:
            raise ContractError("ExpertPack model identity hash mismatch")
        for expected, observed, label in (
            (expected_model_id, model.get("model_id"), "model id"),
            (expected_model_revision, model.get("model_revision"), "model revision"),
            (
                expected_model_manifest_sha256,
                model.get("model_manifest_sha256"),
                "model manifest",
            ),
        ):
            if expected is not None and expected != observed:
                raise ContractError(f"ExpertPack {label} mismatch")
        if quantization != {
            "bits": 4,
            "group_size": quantization.get("group_size"),
            "nibble_order": "low_first_twos_complement",
            "rounding": "nearest_even",
            "scale_dtype": "float32_le",
            "signed_range": [-7, 7],
            "zero_point": None,
        }:
            raise ContractError("ExpertPack quantization contract mismatch")
        group_size = quantization["group_size"]
        alignment = header.get("tensor_alignment_bytes")
        align_up(0, alignment)
        if header.get("tensor_count") != len(tensors) or not tensors:
            raise ContractError("ExpertPack tensor count is inconsistent")
        names: set[str] = set()
        previous_end = 0
        for item in tensors:
            if not isinstance(item, dict):
                raise ContractError("ExpertPack tensor header must be an object")
            name = item.get("name")
            if not isinstance(name, str) or not _NAME.fullmatch(name) or name in names:
                raise ContractError("ExpertPack tensor names must be unique and safe")
            names.add(name)
            shape = item.get("shape")
            if not isinstance(shape, list) or not shape or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in shape
            ):
                raise ContractError(f"ExpertPack tensor {name!r} has invalid shape")
            element_count = 1
            for value in shape:
                element_count *= value
            layout = q4_layout(element_count, group_size, alignment)
            packed_offset = item.get("packed_offset")
            packed_bytes = item.get("packed_bytes")
            scale_offset = item.get("scale_offset")
            scale_bytes = item.get("scale_bytes")
            storage_end = item.get("storage_end_offset")
            if (
                item.get("element_count") != element_count
                or item.get("group_count") != layout.group_count
                or packed_bytes != layout.packed_bytes
                or scale_bytes != layout.scale_bytes
                or not isinstance(packed_offset, int)
                or packed_offset % alignment
                or packed_offset < previous_end
                or scale_offset != align_up(packed_offset + packed_bytes, 4)
                or storage_end != align_up(scale_offset + scale_bytes, alignment)
                or storage_end > payload_length
            ):
                raise ContractError(f"ExpertPack tensor {name!r} layout is invalid")
            expected_bpw = (storage_end - packed_offset) * 8 / element_count
            if item.get("effective_bits_per_weight") != expected_bpw:
                raise ContractError(f"ExpertPack tensor {name!r} effective bpw is invalid")
            _require_zero_region(
                self._mapping,
                payload_offset + previous_end,
                payload_offset + packed_offset,
                f"tensor {name!r} leading",
            )
            _require_zero_region(
                self._mapping,
                payload_offset + packed_offset + packed_bytes,
                payload_offset + scale_offset,
                f"tensor {name!r} packed-to-scale",
            )
            _require_zero_region(
                self._mapping,
                payload_offset + scale_offset + scale_bytes,
                payload_offset + storage_end,
                f"tensor {name!r} trailing",
            )
            packed = memoryview(self._mapping)[
                payload_offset
                + packed_offset : payload_offset
                + packed_offset
                + packed_bytes
            ]
            scales = memoryview(self._mapping)[
                payload_offset
                + scale_offset : payload_offset
                + scale_offset
                + scale_bytes
            ]
            try:
                if _sha256_regions(packed, scales) != item.get(
                    "tensor_payload_sha256"
                ):
                    raise ContractError(f"ExpertPack tensor {name!r} hash mismatch")
                _validate_q4_semantics(packed, scales, element_count, name)
            finally:
                packed.release()
                scales.release()
            previous_end = storage_end
        if previous_end != payload_length:
            raise ContractError("ExpertPack payload has unclaimed trailing bytes")
        return header, payload_offset, payload_length

    def tensor_q4(self, name: str) -> Q4Tensor:
        try:
            item = self._tensors[name]
        except KeyError as exc:
            raise ContractError(f"ExpertPack has no tensor {name!r}") from exc
        packed_start = self.payload_offset + item["packed_offset"]
        scale_start = self.payload_offset + item["scale_offset"]
        return Q4Tensor(
            shape=tuple(item["shape"]),
            group_size=self.header["quantization"]["group_size"],
            packed=bytes(self._mapping[packed_start : packed_start + item["packed_bytes"]]),
            scales=bytes(self._mapping[scale_start : scale_start + item["scale_bytes"]]),
        )

    def tensor_views(self, name: str) -> tuple[memoryview, memoryview, Mapping[str, Any]]:
        try:
            item = self._tensors[name]
        except KeyError as exc:
            raise ContractError(f"ExpertPack has no tensor {name!r}") from exc
        packed_start = self.payload_offset + item["packed_offset"]
        scale_start = self.payload_offset + item["scale_offset"]
        return (
            memoryview(self._mapping)[packed_start : packed_start + item["packed_bytes"]],
            memoryview(self._mapping)[scale_start : scale_start + item["scale_bytes"]],
            item,
        )

    def close(self) -> None:
        mapping = getattr(self, "_mapping", None)
        if mapping is not None:
            mapping.close()
            self._mapping = None
        file = getattr(self, "_file", None)
        if file is not None:
            file.close()
            self._file = None

    def __enter__(self) -> "ExpertPackReader":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def inspect_expert_pack(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with ExpertPackReader(source) as reader:
        prefix = PREFIX.unpack_from(reader._mapping)
        document = {
            "schema_version": 1,
            "kind": "expert_pack_manifest",
            "status": "verified",
            "artifact": {
                "path": source.name,
                "size_bytes": source.stat().st_size,
                "sha256": _sha256_file(source),
            },
            "header_sha256": prefix[5].hex(),
            "payload_sha256": prefix[6].hex(),
            "payload_offset": reader.payload_offset,
            "payload_length": reader.payload_length,
            "header": reader.header,
        }
    validate_document(document)
    return document


__all__ = [
    "ExpertPackReader",
    "FORMAT_VERSION",
    "HEADER_ALIGNMENT",
    "MAGIC",
    "TENSOR_ALIGNMENT",
    "inspect_expert_pack",
    "write_expert_pack",
]
