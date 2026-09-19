"""Mixed-precision, mmap-friendly TargetPack v1 container.

TargetPack complements the canonical all-Q4 ExpertPack.  It preserves a
single immutable mapping while allowing each fixed OLMoE layer to select one
of three explicit encodings: canonical Q4, symmetric group-wise Q8, or raw
little-endian BF16.  The format never infers an encoding from byte lengths.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
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
from .q4 import DEFAULT_GROUP_SIZE, Q4Tensor, align_up, dequantize_q4, quantize_q4


MAGIC = b"UQMOTPK1"
FORMAT_VERSION = 1
HEADER_ALIGNMENT = 4096
TENSOR_ALIGNMENT = 64
PREFIX = struct.Struct("<8sIIQQ32s32s")
ENCODINGS = frozenset({"q4_group128", "q8_group128", "bf16_le"})
_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")
_OLMOE_EXPERT = re.compile(
    r"^model\.layers\.(\d+)\.mlp\.experts\.(\d+)\."
    r"(down_proj|gate_proj|up_proj)\.weight$"
)
_RESERVED_NIBBLE_TABLE = bytes(
    1 if (value & 0x0F) == 8 or (value >> 4) == 8 else 0 for value in range(256)
)
_SCAN_CHUNK_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class TargetTensor:
    shape: tuple[int, ...]
    encoding: str
    data: bytes
    scales: bytes = b""
    group_size: int | None = None

    @property
    def element_count(self) -> int:
        return math.prod(self.shape)


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:
        raise ContractError("TargetPack conversion requires the conversion extra") from exc
    return np


def _source_array(values: Any) -> Any:
    np = _numpy()
    if hasattr(values, "detach"):
        values = values.detach().float().cpu().numpy()
    source = np.asarray(values)
    if source.size == 0 or not np.issubdtype(source.dtype, np.number):
        raise ContractError("TargetPack source must be a non-empty numeric tensor")
    source = source.astype(np.float32, copy=False)
    if not np.all(np.isfinite(source)):
        raise ContractError("TargetPack source must contain only finite values")
    return source


def _float32_to_bf16_bytes(source: Any) -> bytes:
    np = _numpy()
    words = np.asarray(source, dtype="<f4").view("<u4")
    upper = words >> 16
    rounding = np.uint32(0x7FFF) + (upper & 1)
    return ((words + rounding) >> 16).astype("<u2").tobytes()


def _bf16_bytes_to_float32(payload: bytes | memoryview, shape: tuple[int, ...]) -> Any:
    np = _numpy()
    words = np.frombuffer(payload, dtype="<u2").astype("<u4") << 16
    return words.view("<f4").reshape(shape)


def _quantize_q8(values: Any, group_size: int) -> TargetTensor:
    np = _numpy()
    source = _source_array(values)
    shape = tuple(int(value) for value in source.shape)
    flat = source.reshape(-1)
    group_count = math.ceil(flat.size / group_size)
    padded = np.zeros(group_count * group_size, dtype=np.float32)
    padded[: flat.size] = flat
    groups = padded.reshape(group_count, group_size)
    scales = (np.max(np.abs(groups), axis=1) / 127.0).astype("<f4")
    scales[scales == 0] = 1.0
    quantized = np.clip(
        np.rint(groups / scales[:, None]), -127, 127
    ).astype(np.int8).reshape(-1)[: flat.size]
    return TargetTensor(
        shape=shape,
        encoding="q8_group128",
        data=quantized.tobytes(),
        scales=scales.tobytes(),
        group_size=group_size,
    )


def encode_target_tensor(
    values: Any, encoding: str, *, group_size: int = DEFAULT_GROUP_SIZE
) -> TargetTensor:
    """Encode one finite tensor using an explicit TargetPack encoding."""

    if encoding not in ENCODINGS:
        raise ContractError(f"unsupported TargetPack encoding {encoding!r}")
    if group_size != DEFAULT_GROUP_SIZE:
        raise ContractError("TargetPack v1 requires group size 128")
    source = _source_array(values)
    shape = tuple(int(value) for value in source.shape)
    if encoding == "bf16_le":
        return TargetTensor(
            shape=shape,
            encoding=encoding,
            data=_float32_to_bf16_bytes(source),
        )
    if encoding == "q8_group128":
        return _quantize_q8(source, group_size)
    q4 = quantize_q4(source, group_size=group_size)
    return TargetTensor(
        shape=q4.shape,
        encoding=encoding,
        data=q4.packed,
        scales=q4.scales,
        group_size=group_size,
    )


def decode_target_tensor(tensor: TargetTensor) -> Any:
    """Decode one TargetTensor to float32 for the correctness backend."""

    np = _numpy()
    if tensor.encoding == "bf16_le":
        expected = tensor.element_count * 2
        if len(tensor.data) != expected or tensor.scales or tensor.group_size is not None:
            raise ContractError("BF16 TargetTensor layout is invalid")
        return _bf16_bytes_to_float32(tensor.data, tensor.shape)
    if tensor.encoding == "q4_group128":
        if tensor.group_size != DEFAULT_GROUP_SIZE:
            raise ContractError("Q4 TargetTensor group size is invalid")
        return dequantize_q4(
            Q4Tensor(
                shape=tensor.shape,
                group_size=tensor.group_size,
                packed=tensor.data,
                scales=tensor.scales,
            )
        )
    if tensor.encoding == "q8_group128":
        if tensor.group_size != DEFAULT_GROUP_SIZE:
            raise ContractError("Q8 TargetTensor group size is invalid")
        group_count = math.ceil(tensor.element_count / tensor.group_size)
        if len(tensor.data) != tensor.element_count or len(tensor.scales) != group_count * 4:
            raise ContractError("Q8 TargetTensor payload lengths are invalid")
        quantized = np.frombuffer(tensor.data, dtype=np.int8).astype(np.float32)
        scales = np.frombuffer(tensor.scales, dtype="<f4")
        if not np.all(np.isfinite(scales)) or np.any(scales <= 0):
            raise ContractError("Q8 TargetTensor scales must be finite and positive")
        expanded = np.repeat(scales, tensor.group_size)[: tensor.element_count]
        return (quantized * expanded).reshape(tensor.shape)
    raise ContractError(f"unsupported TargetTensor encoding {tensor.encoding!r}")


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


def _policy_identity(
    policy_id: str,
    layer_encodings: Mapping[int, str],
    policy_evidence_sha256: str,
) -> str:
    value = {
        "policy_id": policy_id,
        "layer_encodings": {str(index): layer_encodings[index] for index in range(16)},
        "policy_evidence_sha256": policy_evidence_sha256,
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validated_policy(
    policy_id: str,
    layer_encodings: Mapping[int, str],
    policy_evidence_sha256: str,
) -> dict[str, Any]:
    if not isinstance(policy_id, str) or not policy_id:
        raise ContractError("TargetPack policy id must be non-empty")
    if set(layer_encodings) != set(range(16)):
        raise ContractError("TargetPack policy must assign every OLMoE layer exactly once")
    if any(value not in ENCODINGS for value in layer_encodings.values()):
        raise ContractError("TargetPack policy contains an unsupported encoding")
    if len(policy_evidence_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in policy_evidence_sha256
    ):
        raise ContractError("TargetPack policy evidence identity must be SHA-256")
    normalized = {str(index): layer_encodings[index] for index in range(16)}
    return {
        "policy_id": policy_id,
        "layer_encodings": normalized,
        "policy_evidence_sha256": policy_evidence_sha256,
        "policy_identity_sha256": _policy_identity(
            policy_id, layer_encodings, policy_evidence_sha256
        ),
    }


def _write_zeros(stream: BinaryIO, count: int) -> None:
    block = bytes(64 * 1024)
    while count:
        size = min(count, len(block))
        stream.write(block[:size])
        count -= size


def write_target_pack(
    destination: str | Path,
    tensors: Iterable[tuple[str, Any]],
    *,
    model_id: str,
    model_revision: str,
    model_manifest_sha256: str,
    policy_id: str,
    layer_encodings: Mapping[int, str],
    policy_evidence_sha256: str,
    tensor_alignment_bytes: int = TENSOR_ALIGNMENT,
) -> dict[str, Any]:
    """Encode and stream a fixed mixed policy into one immutable TargetPack."""

    if len(model_revision) != 40 or any(c not in "0123456789abcdef" for c in model_revision):
        raise ContractError("TargetPack model revision must be a lowercase git SHA")
    if len(model_manifest_sha256) != 64 or any(
        c not in "0123456789abcdef" for c in model_manifest_sha256
    ):
        raise ContractError("TargetPack model manifest identity must be SHA-256")
    align_up(0, tensor_alignment_bytes)
    policy = _validated_policy(
        policy_id, layer_encodings, policy_evidence_sha256
    )
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise ContractError(f"refusing to overwrite existing TargetPack {target}")

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
                    raise ContractError(f"invalid TargetPack tensor name {name!r}")
                if name in seen:
                    raise ContractError(f"duplicate TargetPack tensor name {name!r}")
                match = _OLMOE_EXPERT.fullmatch(name)
                if match is None:
                    raise ContractError(f"unexpected TargetPack tensor name {name!r}")
                layer_index = int(match.group(1))
                expert_index = int(match.group(2))
                if layer_index >= 16 or expert_index >= 64:
                    raise ContractError(f"TargetPack tensor identity is out of range: {name!r}")
                encoding = layer_encodings[layer_index]
                tensor = encode_target_tensor(values, encoding)
                tensor_start = align_up(payload_size, tensor_alignment_bytes)
                payload_write(bytes(tensor_start - payload_size))
                data_offset = payload_size
                payload_write(tensor.data)
                scale_offset = align_up(payload_size, 4)
                payload_write(bytes(scale_offset - payload_size))
                payload_write(tensor.scales)
                storage_end = align_up(payload_size, tensor_alignment_bytes)
                payload_write(bytes(storage_end - payload_size))
                group_count = (
                    math.ceil(tensor.element_count / DEFAULT_GROUP_SIZE)
                    if encoding != "bf16_le"
                    else 0
                )
                tensor_headers.append(
                    {
                        "name": name,
                        "layer_index": layer_index,
                        "expert_index": expert_index,
                        "projection": match.group(3),
                        "encoding": encoding,
                        "shape": list(tensor.shape),
                        "element_count": tensor.element_count,
                        "group_size": tensor.group_size,
                        "group_count": group_count,
                        "data_offset": data_offset,
                        "data_bytes": len(tensor.data),
                        "scale_offset": scale_offset,
                        "scale_bytes": len(tensor.scales),
                        "storage_end_offset": storage_end,
                        "effective_bits_per_weight": (
                            (storage_end - tensor_start) * 8 / tensor.element_count
                        ),
                        "tensor_payload_sha256": hashlib.sha256(
                            tensor.data + tensor.scales
                        ).hexdigest(),
                    }
                )
                seen.add(name)
            payload_stream.flush()
        if not tensor_headers:
            raise ContractError("TargetPack must contain at least one tensor")

        encoding_counts = {
            encoding: sum(item["encoding"] == encoding for item in tensor_headers)
            for encoding in sorted(ENCODINGS)
        }
        header = {
            "schema_version": 1,
            "format": "mixed_precision_target_pack",
            "model": {
                "model_id": model_id,
                "model_revision": model_revision,
                "model_manifest_sha256": model_manifest_sha256,
                "model_identity_sha256": _model_identity(
                    model_id, model_revision, model_manifest_sha256
                ),
            },
            "policy": policy,
            "encodings": {
                "q4_group128": {
                    "bits": 4,
                    "signed_range": [-7, 7],
                    "group_size": 128,
                    "scale_dtype": "float32_le",
                    "rounding": "nearest_even",
                    "zero_point": None,
                    "nibble_order": "low_first_twos_complement",
                },
                "q8_group128": {
                    "bits": 8,
                    "signed_range": [-127, 127],
                    "group_size": 128,
                    "scale_dtype": "float32_le",
                    "rounding": "nearest_even",
                    "zero_point": None,
                },
                "bf16_le": {"bits": 16, "byte_order": "little"},
            },
            "tensor_alignment_bytes": tensor_alignment_bytes,
            "tensor_count": len(tensor_headers),
            "encoding_tensor_counts": encoding_counts,
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
        return inspect_target_pack(target)
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


def _require_zero(mapping: mmap.mmap, start: int, stop: int, label: str) -> None:
    for offset in range(start, stop, _SCAN_CHUNK_BYTES):
        if mapping[offset : min(stop, offset + _SCAN_CHUNK_BYTES)].strip(b"\0"):
            raise ContractError(f"TargetPack {label} padding must be zero")


class TargetPackReader:
    """One read-only file descriptor and exactly one mmap per TargetPack."""

    def __init__(
        self,
        path: str | Path,
        *,
        expected_model_id: str | None = None,
        expected_model_revision: str | None = None,
        expected_model_manifest_sha256: str | None = None,
        expected_policy_id: str | None = None,
    ) -> None:
        self.path = Path(path)
        try:
            self._file = self.path.open("rb")
            self._mapping = mmap.mmap(self._file.fileno(), 0, access=mmap.ACCESS_READ)
        except OSError as exc:
            raise ContractError(f"cannot map TargetPack {self.path}: {exc}") from exc
        self.mapping_count = 1
        try:
            self.header, self.payload_offset, self.payload_length = self._validate(
                expected_model_id,
                expected_model_revision,
                expected_model_manifest_sha256,
                expected_policy_id,
            )
        except BaseException:
            self.close()
            raise
        self._tensors = {item["name"]: item for item in self.header["tensors"]}

    def _validate(
        self,
        expected_model_id: str | None,
        expected_model_revision: str | None,
        expected_model_manifest_sha256: str | None,
        expected_policy_id: str | None,
    ) -> tuple[dict[str, Any], int, int]:
        if len(self._mapping) < PREFIX.size:
            raise ContractError("TargetPack is shorter than its fixed prefix")
        prefix = PREFIX.unpack_from(self._mapping)
        magic, version, header_length, payload_offset, payload_length = prefix[:5]
        if magic != MAGIC or version != FORMAT_VERSION:
            raise ContractError("TargetPack magic/version mismatch")
        if payload_offset % HEADER_ALIGNMENT or payload_offset < PREFIX.size + header_length:
            raise ContractError("TargetPack payload offset/alignment is invalid")
        if payload_offset + payload_length != len(self._mapping):
            raise ContractError("TargetPack payload length does not match file size")
        header_bytes = self._mapping[PREFIX.size : PREFIX.size + header_length]
        if hashlib.sha256(header_bytes).digest() != prefix[5]:
            raise ContractError("TargetPack header hash mismatch")
        _require_zero(
            self._mapping,
            PREFIX.size + header_length,
            payload_offset,
            "header",
        )
        payload = memoryview(self._mapping)[payload_offset : payload_offset + payload_length]
        try:
            if hashlib.sha256(payload).digest() != prefix[6]:
                raise ContractError("TargetPack payload hash mismatch")
        finally:
            payload.release()
        try:
            header = json.loads(header_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ContractError("TargetPack header is not canonical JSON") from exc
        if json.dumps(
            header, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8") != header_bytes:
            raise ContractError("TargetPack header JSON is not canonical")
        if header.get("schema_version") != 1 or header.get("format") != "mixed_precision_target_pack":
            raise ContractError("TargetPack header schema is unsupported")
        model = header.get("model", {})
        if model.get("model_identity_sha256") != _model_identity(
            model.get("model_id", ""),
            model.get("model_revision", ""),
            model.get("model_manifest_sha256", ""),
        ):
            raise ContractError("TargetPack model identity hash mismatch")
        for expected, observed, label in (
            (expected_model_id, model.get("model_id"), "model id"),
            (expected_model_revision, model.get("model_revision"), "model revision"),
            (expected_model_manifest_sha256, model.get("model_manifest_sha256"), "model manifest"),
            (expected_policy_id, header.get("policy", {}).get("policy_id"), "policy id"),
        ):
            if expected is not None and expected != observed:
                raise ContractError(f"TargetPack {label} mismatch")
        raw_layers = header.get("policy", {}).get("layer_encodings", {})
        try:
            layer_encodings = {int(key): value for key, value in raw_layers.items()}
        except (AttributeError, TypeError, ValueError) as exc:
            raise ContractError("TargetPack layer policy is invalid") from exc
        expected_policy = _validated_policy(
            header.get("policy", {}).get("policy_id", ""),
            layer_encodings,
            header.get("policy", {}).get("policy_evidence_sha256", ""),
        )
        if header.get("policy") != expected_policy:
            raise ContractError("TargetPack policy identity hash mismatch")
        alignment = header.get("tensor_alignment_bytes")
        align_up(0, alignment)
        tensors = header.get("tensors")
        if not isinstance(tensors, list) or not tensors or header.get("tensor_count") != len(tensors):
            raise ContractError("TargetPack tensor count is inconsistent")
        names: set[str] = set()
        observed_counts = {encoding: 0 for encoding in sorted(ENCODINGS)}
        previous_end = 0
        for item in tensors:
            if not isinstance(item, dict):
                raise ContractError("TargetPack tensor header must be an object")
            name = item.get("name")
            match = _OLMOE_EXPERT.fullmatch(name) if isinstance(name, str) else None
            if match is None or name in names:
                raise ContractError("TargetPack tensor names must be unique fixed OLMoE experts")
            names.add(name)
            layer_index = int(match.group(1))
            expert_index = int(match.group(2))
            encoding = item.get("encoding")
            if (
                layer_index >= 16
                or expert_index >= 64
                or item.get("layer_index") != layer_index
                or item.get("expert_index") != expert_index
                or item.get("projection") != match.group(3)
                or encoding != layer_encodings[layer_index]
            ):
                raise ContractError(f"TargetPack tensor {name!r} policy identity is invalid")
            shape = item.get("shape")
            if not isinstance(shape, list) or not shape or any(
                isinstance(value, bool) or not isinstance(value, int) or value <= 0
                for value in shape
            ):
                raise ContractError(f"TargetPack tensor {name!r} shape is invalid")
            element_count = math.prod(shape)
            data_offset = item.get("data_offset")
            data_bytes = item.get("data_bytes")
            scale_offset = item.get("scale_offset")
            scale_bytes = item.get("scale_bytes")
            storage_end = item.get("storage_end_offset")
            group_size = item.get("group_size")
            group_count = item.get("group_count")
            expected_data = {
                "q4_group128": math.ceil(element_count / 2),
                "q8_group128": element_count,
                "bf16_le": element_count * 2,
            }.get(encoding)
            expected_groups = 0 if encoding == "bf16_le" else math.ceil(element_count / 128)
            expected_scales = expected_groups * 4
            if (
                item.get("element_count") != element_count
                or expected_data is None
                or data_bytes != expected_data
                or scale_bytes != expected_scales
                or group_size != (None if encoding == "bf16_le" else 128)
                or group_count != expected_groups
                or not isinstance(data_offset, int)
                or data_offset % alignment
                or data_offset < previous_end
                or scale_offset != align_up(data_offset + data_bytes, 4)
                or storage_end != align_up(scale_offset + scale_bytes, alignment)
                or storage_end > payload_length
            ):
                raise ContractError(f"TargetPack tensor {name!r} layout is invalid")
            expected_bpw = (storage_end - data_offset) * 8 / element_count
            if item.get("effective_bits_per_weight") != expected_bpw:
                raise ContractError(f"TargetPack tensor {name!r} effective bpw is invalid")
            _require_zero(self._mapping, payload_offset + previous_end, payload_offset + data_offset, f"tensor {name!r} leading")
            _require_zero(self._mapping, payload_offset + data_offset + data_bytes, payload_offset + scale_offset, f"tensor {name!r} data-to-scale")
            _require_zero(self._mapping, payload_offset + scale_offset + scale_bytes, payload_offset + storage_end, f"tensor {name!r} trailing")
            data = memoryview(self._mapping)[payload_offset + data_offset : payload_offset + data_offset + data_bytes]
            scales = memoryview(self._mapping)[payload_offset + scale_offset : payload_offset + scale_offset + scale_bytes]
            try:
                if hashlib.sha256(data.tobytes() + scales.tobytes()).hexdigest() != item.get("tensor_payload_sha256"):
                    raise ContractError(f"TargetPack tensor {name!r} hash mismatch")
                if encoding == "q4_group128":
                    for offset in range(0, len(data), _SCAN_CHUNK_BYTES):
                        if bytes(data[offset : offset + _SCAN_CHUNK_BYTES]).translate(_RESERVED_NIBBLE_TABLE).find(b"\x01") != -1:
                            raise ContractError(f"TargetPack tensor {name!r} contains reserved Q4 -8")
                    if element_count % 2 and (data[-1] >> 4) != 0:
                        raise ContractError(f"TargetPack tensor {name!r} has non-zero Q4 tail padding")
                if scales:
                    for (scale,) in struct.iter_unpack("<f", scales):
                        if not math.isfinite(scale) or scale <= 0:
                            raise ContractError(f"TargetPack tensor {name!r} scales must be finite and positive")
            finally:
                data.release()
                scales.release()
            observed_counts[encoding] += 1
            previous_end = storage_end
        if previous_end != payload_length:
            raise ContractError("TargetPack payload has unclaimed trailing bytes")
        if header.get("encoding_tensor_counts") != observed_counts:
            raise ContractError("TargetPack encoding tensor counts are inconsistent")
        return header, payload_offset, payload_length

    def layer_encoding(self, layer_index: int) -> str:
        if not 0 <= layer_index < 16:
            raise ContractError("TargetPack layer index must be in [0, 15]")
        return self.header["policy"]["layer_encodings"][str(layer_index)]

    def validate_fixed_olmoe_complete(self) -> None:
        """Require the exact 16x64x3 expert tensor set used by the fixed Host."""

        expected = {
            f"model.layers.{layer}.mlp.experts.{expert}.{projection}.weight": (
                (2048, 1024) if projection == "down_proj" else (1024, 2048)
            )
            for layer in range(16)
            for expert in range(64)
            for projection in ("gate_proj", "up_proj", "down_proj")
        }
        observed = {name: tuple(item["shape"]) for name, item in self._tensors.items()}
        if observed != expected:
            missing = sorted(set(expected) - set(observed))
            unexpected = sorted(set(observed) - set(expected))
            wrong_shape = sorted(
                name
                for name in set(observed) & set(expected)
                if observed[name] != expected[name]
            )
            raise ContractError(
                "TargetPack fixed OLMoE tensor set mismatch: "
                f"missing={missing[:8]!r}, unexpected={unexpected[:8]!r}, "
                f"wrong_shape={wrong_shape[:8]!r}"
            )
        expected_counts = {encoding: 0 for encoding in sorted(ENCODINGS)}
        for layer in range(16):
            expected_counts[self.layer_encoding(layer)] += 64 * 3
        if self.header["encoding_tensor_counts"] != expected_counts:
            raise ContractError("TargetPack fixed OLMoE policy counts are inconsistent")

    def tensor_views(self, name: str) -> tuple[memoryview, memoryview, Mapping[str, Any]]:
        try:
            item = self._tensors[name]
        except KeyError as exc:
            raise ContractError(f"TargetPack has no tensor {name!r}") from exc
        data_start = self.payload_offset + item["data_offset"]
        scale_start = self.payload_offset + item["scale_offset"]
        return (
            memoryview(self._mapping)[data_start : data_start + item["data_bytes"]],
            memoryview(self._mapping)[scale_start : scale_start + item["scale_bytes"]],
            item,
        )

    def tensor(self, name: str) -> TargetTensor:
        data, scales, item = self.tensor_views(name)
        try:
            return TargetTensor(
                shape=tuple(item["shape"]),
                encoding=item["encoding"],
                data=bytes(data),
                scales=bytes(scales),
                group_size=item["group_size"],
            )
        finally:
            data.release()
            scales.release()

    def close(self) -> None:
        mapping = getattr(self, "_mapping", None)
        if mapping is not None:
            mapping.close()
            self._mapping = None
        file = getattr(self, "_file", None)
        if file is not None:
            file.close()
            self._file = None

    def __enter__(self) -> "TargetPackReader":
        return self

    def __exit__(self, _type: Any, _value: Any, _traceback: Any) -> None:
        self.close()


def inspect_target_pack(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with TargetPackReader(source) as reader:
        prefix = PREFIX.unpack_from(reader._mapping)
        document = {
            "schema_version": 1,
            "kind": "target_pack_manifest",
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
    "ENCODINGS",
    "FORMAT_VERSION",
    "HEADER_ALIGNMENT",
    "MAGIC",
    "PREFIX",
    "TENSOR_ALIGNMENT",
    "TargetPackReader",
    "TargetTensor",
    "decode_target_tensor",
    "encode_target_tensor",
    "inspect_target_pack",
    "write_target_pack",
]
