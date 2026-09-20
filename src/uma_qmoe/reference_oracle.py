"""Versioned, streaming numerical comparison contracts for fixed-model Oracles.

The comparison core deliberately does not load Transformers or a model.  It
consumes small tensor-record streams captured by a reference implementation
and a candidate implementation, which keeps the numerical contract testable
on CPU-only hosts and reusable by later Q4 tooling.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
import hashlib
from importlib import resources
import json
import math
from pathlib import Path
import struct
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker

from .contracts import ContractError
from .fixed_models import OLMOE, FixedModelSpec, fixed_model_spec


# Backward-compatible constants for callers that still import the canonical
# OLMoE Oracle dimensions directly.
MODEL_ID = OLMOE.model_id
MODEL_TYPE = OLMOE.model_type
NUM_LAYERS = OLMOE.num_layers
NUM_EXPERTS = OLMOE.num_experts
TOP_K = OLMOE.top_k

_NUMERIC_DTYPES = frozenset({"float16", "bfloat16", "float32", "float64"})
_ROUTER_DTYPES = frozenset({"int16", "int32", "int64", "uint16", "uint32"})
_LEVELS = frozenset({"single_expert", "single_moe_layer", "full_model"})
_RECORD_FIELDS = frozenset({"identity", "role", "dtype", "shape", "values"})
_NPY_RECORD_FIELDS = frozenset({"identity", "role", "npy_path"})
_THRESHOLD_FIELDS = (
    "max_absolute_error",
    "max_relative_error",
    "max_p99_absolute_error",
    "max_p99_relative_error",
    "min_cosine_similarity",
)
_DEFAULT_POLICY: dict[str, Any] = {
    "status": "draft",
    "percentile_method": "nearest_rank",
    "relative_error_epsilon": 1e-12,
    "max_absolute_error": None,
    "max_relative_error": None,
    "max_p99_absolute_error": None,
    "max_p99_relative_error": None,
    "min_cosine_similarity": None,
    "min_router_top_k_set_agreement": None,
}


class ReferenceOracleError(ValueError):
    """Raised when a comparison cannot produce trustworthy evidence."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _schema() -> dict[str, Any]:
    resource = resources.files("uma_qmoe.schemas").joinpath(
        "reference_oracle_comparison.schema.json"
    )
    return json.loads(resource.read_text(encoding="utf-8"))


def _format_path(parts: Sequence[Any]) -> str:
    rendered = "$"
    for part in parts:
        rendered += f"[{part}]" if isinstance(part, int) else f".{part}"
    return rendered


def _validate_schema(document: Mapping[str, Any]) -> None:
    validator = Draft202012Validator(_schema(), format_checker=FormatChecker())
    errors = sorted(
        validator.iter_errors(document), key=lambda error: list(error.absolute_path)
    )
    if errors:
        details = "; ".join(
            f"{_format_path(list(error.absolute_path))}: {error.message}"
            for error in errors
        )
        raise ReferenceOracleError(details)


def _reject_non_finite_numbers(value: Any, *, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ReferenceOracleError(f"{path} contains NaN or Inf")
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_non_finite_numbers(item, path=f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _reject_non_finite_numbers(item, path=f"{path}[{index}]")


def _validate_scope(scope: Mapping[str, Any], spec: FixedModelSpec) -> None:
    level = scope.get("level")
    if level not in _LEVELS:
        raise ReferenceOracleError(f"unsupported Oracle scope {level!r}")
    layer = scope.get("layer_index")
    expert = scope.get("expert_index")
    if level == "single_expert":
        if (
            not isinstance(layer, int)
            or isinstance(layer, bool)
            or not 0 <= layer < spec.num_layers
        ):
            raise ReferenceOracleError(
                f"single_expert requires layer_index in [0, {spec.num_layers - 1}]"
            )
        if (
            not isinstance(expert, int)
            or isinstance(expert, bool)
            or not 0 <= expert < spec.num_experts
        ):
            raise ReferenceOracleError(
                f"single_expert requires expert_index in [0, {spec.num_experts - 1}]"
            )
    elif level == "single_moe_layer":
        if (
            not isinstance(layer, int)
            or isinstance(layer, bool)
            or not 0 <= layer < spec.num_layers
        ):
            raise ReferenceOracleError(
                f"single_moe_layer requires layer_index in [0, {spec.num_layers - 1}]"
            )
        if expert is not None:
            raise ReferenceOracleError("single_moe_layer forbids expert_index")
    elif layer is not None or expert is not None:
        raise ReferenceOracleError("full_model forbids layer_index and expert_index")


def _expected_identities(
    scope: Mapping[str, Any], spec: FixedModelSpec
) -> tuple[dict[str, str], set[str]]:
    level = scope["level"]
    layer = scope.get("layer_index")
    expert = scope.get("expert_index")
    if level == "single_expert":
        return (
            {
                "expert_output": (
                    f"model.layers.{layer}.mlp.experts.{expert}.output"
                )
            },
            set(),
        )
    if level == "single_moe_layer":
        return (
            {
                "router_logits": f"model.layers.{layer}.mlp.router_logits",
                "moe_layer_output": f"model.layers.{layer}.mlp.output",
            },
            {f"model.layers.{layer}.mlp.router_topk_indices"},
        )
    return (
        {"final_logits": "model.final_logits"},
        {
            f"model.layers.{index}.mlp.router_topk_indices"
            for index in range(spec.num_layers)
        },
    )


def _shape(value: Any, *, identity: str) -> list[int]:
    if (
        not isinstance(value, list)
        or not value
        or any(
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            for dimension in value
        )
    ):
        raise ReferenceOracleError(f"tensor {identity!r} has an invalid shape")
    return value


def _validate_record(record: Mapping[str, Any]) -> None:
    fields = frozenset(record)
    if fields != _RECORD_FIELDS:
        raise ReferenceOracleError(
            "tensor record must contain exactly identity, role, dtype, shape, values"
        )
    identity = record["identity"]
    role = record["role"]
    dtype = record["dtype"]
    if not isinstance(identity, str) or not identity:
        raise ReferenceOracleError("tensor identity must be a non-empty string")
    if not isinstance(role, str) or not role:
        raise ReferenceOracleError(f"tensor {identity!r} has an invalid role")
    allowed_dtypes = _ROUTER_DTYPES if role == "router_topk_indices" else _NUMERIC_DTYPES
    if dtype not in allowed_dtypes:
        raise ReferenceOracleError(
            f"tensor {identity!r} has unsupported dtype {dtype!r} for role {role!r}"
        )
    shape = _shape(record["shape"], identity=identity)
    values = record["values"]
    if isinstance(values, (str, bytes, Mapping)) or not hasattr(values, "__iter__"):
        raise ReferenceOracleError(f"tensor {identity!r} values must be iterable")
    if math.prod(shape) <= 0:
        raise ReferenceOracleError(f"tensor {identity!r} must not be empty")


def _record_from_mapping(
    value: Mapping[str, Any], *, base_directory: Path | None
) -> dict[str, Any]:
    fields = frozenset(value)
    if fields == _RECORD_FIELDS:
        record = dict(value)
        _validate_record(record)
        return record
    if fields != _NPY_RECORD_FIELDS:
        raise ReferenceOracleError(
            "record must contain inline values or exactly identity, role, npy_path"
        )
    npy_path = value["npy_path"]
    if (
        not isinstance(npy_path, str)
        or not npy_path
        or "\\" in npy_path
        or Path(npy_path).is_absolute()
    ):
        raise ReferenceOracleError("npy_path must be a non-empty relative path")
    if any(part in {"", ".", ".."} for part in Path(npy_path).parts):
        raise ReferenceOracleError("npy_path must not contain traversal components")
    if base_directory is None:
        raise ReferenceOracleError("npy_path records require a file-backed source")
    return tensor_record_from_npy(
        base_directory / npy_path,
        identity=value["identity"],
        role=value["role"],
    )


def iter_tensor_records(
    source: str | Path | Iterable[Mapping[str, Any]],
) -> Iterator[dict[str, Any]]:
    """Yield fail-closed tensor records from JSON, JSONL, or an iterable.

    A JSON/JSONL record may contain inline ``values`` or a relative ``npy_path``.
    NPY payloads are opened with NumPy memory mapping, so a record can be
    compared without loading the complete tensor into process memory.
    """

    if isinstance(source, (str, Path)):
        path = Path(source)
        suffix = path.suffix.lower()
        if suffix not in {".json", ".jsonl"}:
            raise ReferenceOracleError("tensor stream path must end in .json or .jsonl")
        try:
            stream = path.open("r", encoding="utf-8")
        except OSError as exc:
            raise ReferenceOracleError(f"cannot read tensor stream {path}: {exc}") from exc
        with stream:
            if suffix == ".json":
                try:
                    root = json.load(stream)
                except json.JSONDecodeError as exc:
                    raise ReferenceOracleError(f"invalid JSON in {path}: {exc}") from exc
                items = root if isinstance(root, list) else [root]
                for item in items:
                    if not isinstance(item, Mapping):
                        raise ReferenceOracleError("tensor stream entries must be objects")
                    yield _record_from_mapping(item, base_directory=path.parent)
                return
            for line_number, line in enumerate(stream, start=1):
                if not line.strip():
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ReferenceOracleError(
                        f"invalid JSON at {path}:{line_number}: {exc}"
                    ) from exc
                if not isinstance(item, Mapping):
                    raise ReferenceOracleError(
                        f"tensor stream entry at line {line_number} must be an object"
                    )
                yield _record_from_mapping(item, base_directory=path.parent)
        return

    for item in source:
        if not isinstance(item, Mapping):
            raise ReferenceOracleError("tensor stream entries must be objects")
        yield _record_from_mapping(item, base_directory=None)


def tensor_record_from_npy(
    path: str | Path, *, identity: str, role: str
) -> dict[str, Any]:
    """Build a memory-mapped tensor record from a non-object NumPy array."""

    try:
        import numpy as np
    except ImportError as exc:
        raise ReferenceOracleError("NPY records require NumPy") from exc
    try:
        array = np.load(Path(path), mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ReferenceOracleError(f"cannot load NPY tensor {path}: {exc}") from exc
    if not isinstance(array, np.ndarray) or array.dtype.hasobject:
        raise ReferenceOracleError("NPY tensor must be a non-object ndarray")
    dtype = array.dtype.name
    record = {
        "identity": identity,
        "role": role,
        "dtype": dtype,
        "shape": list(array.shape),
        "values": array.flat,
    }
    _validate_record(record)
    return record


def _next_pair(
    reference: Iterator[Any], candidate: Iterator[Any], *, identity: str
) -> tuple[Any, Any] | None:
    sentinel = object()
    left = next(reference, sentinel)
    right = next(candidate, sentinel)
    if left is sentinel and right is sentinel:
        return None
    if left is sentinel or right is sentinel:
        raise ReferenceOracleError(f"tensor {identity!r} has inconsistent value counts")
    return left, right


def _finite_float(value: Any, *, identity: str, side: str) -> float:
    if isinstance(value, bool):
        raise ReferenceOracleError(f"{side} tensor {identity!r} contains a boolean")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ReferenceOracleError(
            f"{side} tensor {identity!r} contains a non-numeric value"
        ) from exc
    if not math.isfinite(result):
        raise ReferenceOracleError(
            f"{side} tensor {identity!r} contains NaN or Inf"
        )
    return result


def _integer(
    value: Any, *, identity: str, side: str, num_experts: int
) -> int:
    if isinstance(value, bool):
        raise ReferenceOracleError(f"{side} router tensor {identity!r} contains a boolean")
    try:
        result = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReferenceOracleError(
            f"{side} router tensor {identity!r} contains a non-integer"
        ) from exc
    try:
        exact = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ReferenceOracleError(
            f"{side} router tensor {identity!r} contains a non-integer"
        ) from exc
    if not math.isfinite(exact) or exact != result or not 0 <= result < num_experts:
        raise ReferenceOracleError(
            f"{side} router tensor {identity!r} contains an invalid expert id"
        )
    return result


def _hash_start(record: Mapping[str, Any]) -> hashlib._Hash:
    digest = hashlib.sha256()
    header = {
        "identity": record["identity"],
        "role": record["role"],
        "dtype": record["dtype"],
        "shape": record["shape"],
    }
    digest.update(
        json.dumps(header, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    digest.update(b"\0")
    return digest


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(quantile * len(ordered)) - 1)]


def _distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": math.fsum(values) / len(values),
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "max": max(values),
    }


def _compare_numeric(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    *,
    epsilon: float,
) -> tuple[dict[str, Any], list[float], list[float]]:
    identity = reference["identity"]
    expected_count = math.prod(reference["shape"])
    reference_hash = _hash_start(reference)
    candidate_hash = _hash_start(candidate)
    absolute_errors: list[float] = []
    relative_errors: list[float] = []
    dot = reference_norm = candidate_norm = 0.0
    reference_values = iter(reference["values"])
    candidate_values = iter(candidate["values"])
    count = 0
    while True:
        pair = _next_pair(reference_values, candidate_values, identity=identity)
        if pair is None:
            break
        left = _finite_float(pair[0], identity=identity, side="reference")
        right = _finite_float(pair[1], identity=identity, side="candidate")
        reference_hash.update(struct.pack("<d", left))
        candidate_hash.update(struct.pack("<d", right))
        absolute = abs(left - right)
        relative = absolute / max(abs(left), epsilon)
        if not math.isfinite(absolute) or not math.isfinite(relative):
            raise ReferenceOracleError(
                f"tensor {identity!r} error calculation overflowed"
            )
        absolute_errors.append(absolute)
        relative_errors.append(relative)
        dot += left * right
        reference_norm += left * left
        candidate_norm += right * right
        count += 1
    if count != expected_count:
        raise ReferenceOracleError(
            f"tensor {identity!r} shape declares {expected_count} values, observed {count}"
        )
    if reference_norm == 0.0 and candidate_norm == 0.0:
        cosine = 1.0
    elif reference_norm == 0.0 or candidate_norm == 0.0:
        cosine = 0.0
    else:
        cosine = dot / math.sqrt(reference_norm * candidate_norm)
        if not math.isfinite(cosine):
            raise ReferenceOracleError(
                f"tensor {identity!r} cosine calculation overflowed"
            )
        cosine = min(1.0, max(-1.0, cosine))
    return (
        {
            "identity": identity,
            "role": reference["role"],
            "shape": reference["shape"],
            "reference": {
                "dtype": reference["dtype"],
                "sha256": reference_hash.hexdigest(),
            },
            "candidate": {
                "dtype": candidate["dtype"],
                "sha256": candidate_hash.hexdigest(),
            },
            "metrics": {
                "element_count": count,
                "finite": True,
                "absolute_error": _distribution(absolute_errors),
                "relative_error": _distribution(relative_errors),
                "cosine_similarity": cosine,
            },
        },
        absolute_errors,
        relative_errors,
    )


def _compare_router(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
    spec: FixedModelSpec,
) -> dict[str, Any]:
    identity = reference["identity"]
    expected_count = math.prod(reference["shape"])
    if reference["shape"][-1] != spec.top_k:
        raise ReferenceOracleError(
            f"router tensor {identity!r} must have trailing dimension {spec.top_k}"
        )
    reference_hash = _hash_start(reference)
    candidate_hash = _hash_start(candidate)
    left_values: list[int] = []
    right_values: list[int] = []
    left_iterator = iter(reference["values"])
    right_iterator = iter(candidate["values"])
    while True:
        pair = _next_pair(left_iterator, right_iterator, identity=identity)
        if pair is None:
            break
        left = _integer(
            pair[0],
            identity=identity,
            side="reference",
            num_experts=spec.num_experts,
        )
        right = _integer(
            pair[1],
            identity=identity,
            side="candidate",
            num_experts=spec.num_experts,
        )
        reference_hash.update(struct.pack("<q", left))
        candidate_hash.update(struct.pack("<q", right))
        left_values.append(left)
        right_values.append(right)
    if len(left_values) != expected_count:
        raise ReferenceOracleError(
            f"router tensor {identity!r} shape declares {expected_count} values, "
            f"observed {len(left_values)}"
        )

    exact = 0
    overlap_total = 0.0
    decisions = expected_count // spec.top_k
    for offset in range(0, expected_count, spec.top_k):
        left_set = set(left_values[offset : offset + spec.top_k])
        right_set = set(right_values[offset : offset + spec.top_k])
        if len(left_set) != spec.top_k or len(right_set) != spec.top_k:
            raise ReferenceOracleError(
                f"router tensor {identity!r} contains duplicate expert ids in a Top-K row"
            )
        if left_set == right_set:
            exact += 1
        overlap_total += len(left_set.intersection(right_set)) / spec.top_k
    return {
        "identity": identity,
        "shape": reference["shape"],
        "top_k": spec.top_k,
        "reference": {
            "dtype": reference["dtype"],
            "sha256": reference_hash.hexdigest(),
        },
        "candidate": {
            "dtype": candidate["dtype"],
            "sha256": candidate_hash.hexdigest(),
        },
        "decision_count": decisions,
        "exact_set_match_count": exact,
        "top_k_set_agreement": exact / decisions,
        "mean_set_overlap": overlap_total / decisions,
    }


def _normalise_policy(value: Mapping[str, Any] | None, *, level: str) -> dict[str, Any]:
    policy = dict(_DEFAULT_POLICY if value is None else value)
    if set(policy) != set(_DEFAULT_POLICY):
        raise ReferenceOracleError(
            "quality policy fields do not match the version-1 contract"
        )
    status = policy["status"]
    if status not in {"draft", "frozen"}:
        raise ReferenceOracleError("quality policy status must be draft or frozen")
    if policy["percentile_method"] != "nearest_rank":
        raise ReferenceOracleError("percentile_method must be nearest_rank")
    epsilon = policy["relative_error_epsilon"]
    if (
        isinstance(epsilon, bool)
        or not isinstance(epsilon, (int, float))
        or not math.isfinite(epsilon)
        or epsilon <= 0
    ):
        raise ReferenceOracleError("relative_error_epsilon must be finite and positive")
    for field in (*_THRESHOLD_FIELDS, "min_router_top_k_set_agreement"):
        threshold = policy[field]
        if threshold is not None and (
            isinstance(threshold, bool)
            or not isinstance(threshold, (int, float))
            or not math.isfinite(threshold)
            or threshold < 0
        ):
            raise ReferenceOracleError(f"{field} must be null or finite and non-negative")
    for field in ("min_cosine_similarity", "min_router_top_k_set_agreement"):
        threshold = policy[field]
        if threshold is not None and threshold > 1:
            raise ReferenceOracleError(f"{field} must not exceed 1")
    if status == "draft" and any(
        policy[field] is not None
        for field in (*_THRESHOLD_FIELDS, "min_router_top_k_set_agreement")
    ):
        raise ReferenceOracleError("draft quality policy must not set thresholds")
    if status == "frozen":
        missing = [field for field in _THRESHOLD_FIELDS if policy[field] is None]
        if level != "single_expert" and policy["min_router_top_k_set_agreement"] is None:
            missing.append("min_router_top_k_set_agreement")
        if missing:
            raise ReferenceOracleError(
                "frozen quality policy is missing thresholds: " + ", ".join(missing)
            )
        if level == "single_expert" and policy["min_router_top_k_set_agreement"] is not None:
            raise ReferenceOracleError(
                "single_expert policy cannot set a router agreement threshold"
            )
    return policy


def _evaluate(
    aggregate: Mapping[str, Any],
    router: Mapping[str, Any] | None,
    policy: Mapping[str, Any],
) -> list[dict[str, Any]]:
    if policy["status"] == "draft":
        return []
    observed = {
        "max_absolute_error": aggregate["absolute_error"]["max"],
        "max_relative_error": aggregate["relative_error"]["max"],
        "max_p99_absolute_error": aggregate["absolute_error"]["p99"],
        "max_p99_relative_error": aggregate["relative_error"]["p99"],
        "min_cosine_similarity": aggregate["minimum_tensor_cosine_similarity"],
    }
    evaluations: list[dict[str, Any]] = []
    for metric in _THRESHOLD_FIELDS:
        comparator = ">=" if metric == "min_cosine_similarity" else "<="
        threshold = policy[metric]
        actual = observed[metric]
        passed = actual >= threshold if comparator == ">=" else actual <= threshold
        evaluations.append(
            {
                "metric": metric,
                "observed": actual,
                "comparator": comparator,
                "threshold": threshold,
                "passed": passed,
            }
        )
    if router is not None:
        actual = router["top_k_set_agreement"]
        threshold = policy["min_router_top_k_set_agreement"]
        evaluations.append(
            {
                "metric": "min_router_top_k_set_agreement",
                "observed": actual,
                "comparator": ">=",
                "threshold": threshold,
                "passed": actual >= threshold,
            }
        )
    return evaluations


def build_reference_oracle_comparison(
    reference_source: str | Path | Iterable[Mapping[str, Any]],
    candidate_source: str | Path | Iterable[Mapping[str, Any]],
    *,
    oracle_id: str,
    model_id: str = MODEL_ID,
    model_revision: str,
    scope: Mapping[str, Any],
    fixture_id: str,
    fixture_sha256: str,
    reference_implementation: Mapping[str, Any],
    candidate_implementation: Mapping[str, Any],
    quality_policy: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compare ordered streams and build fixed-model Oracle evidence."""

    if not isinstance(oracle_id, str) or not oracle_id:
        raise ReferenceOracleError("oracle_id must be non-empty")
    if not isinstance(model_revision, str) or len(model_revision) != 40 or any(
        character not in "0123456789abcdef" for character in model_revision
    ):
        raise ReferenceOracleError("model_revision must be a lowercase 40-character SHA")
    try:
        spec = fixed_model_spec(model_id, model_revision)
    except ContractError as exc:
        raise ReferenceOracleError(str(exc)) from exc
    if not isinstance(fixture_id, str) or not fixture_id:
        raise ReferenceOracleError("fixture_id must be non-empty")
    if not isinstance(fixture_sha256, str) or len(fixture_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in fixture_sha256
    ):
        raise ReferenceOracleError("fixture_sha256 must be a lowercase SHA-256")

    normalized_scope = {
        "level": scope.get("level"),
        "layer_index": scope.get("layer_index"),
        "expert_index": scope.get("expert_index"),
    }
    if set(scope) != set(normalized_scope):
        raise ReferenceOracleError(
            "scope must contain exactly level, layer_index, and expert_index"
        )
    _validate_scope(normalized_scope, spec)
    expected_numeric, expected_router = _expected_identities(normalized_scope, spec)
    policy = _normalise_policy(quality_policy, level=normalized_scope["level"])

    reference_stream = iter(iter_tensor_records(reference_source))
    candidate_stream = iter(iter_tensor_records(candidate_source))
    tensors: list[dict[str, Any]] = []
    router_records: list[dict[str, Any]] = []
    absolute_errors: list[float] = []
    relative_errors: list[float] = []
    seen_identities: set[str] = set()
    while True:
        sentinel = object()
        reference = next(reference_stream, sentinel)
        candidate = next(candidate_stream, sentinel)
        if reference is sentinel and candidate is sentinel:
            break
        if reference is sentinel or candidate is sentinel:
            raise ReferenceOracleError("reference and candidate tensor counts differ")
        assert isinstance(reference, Mapping) and isinstance(candidate, Mapping)
        for field in ("identity", "role", "shape"):
            if reference[field] != candidate[field]:
                raise ReferenceOracleError(
                    f"reference/candidate {field} mismatch: "
                    f"{reference[field]!r} != {candidate[field]!r}"
                )
        identity = reference["identity"]
        if identity in seen_identities:
            raise ReferenceOracleError(f"duplicate tensor identity {identity!r}")
        seen_identities.add(identity)
        role = reference["role"]
        if role == "router_topk_indices":
            if identity not in expected_router:
                raise ReferenceOracleError(
                    f"unexpected router tensor identity {identity!r} for scope"
                )
            router_records.append(_compare_router(reference, candidate, spec))
            continue
        if expected_numeric.get(role) != identity:
            raise ReferenceOracleError(
                f"unexpected role/identity pair {role!r}/{identity!r} for scope"
            )
        result, absolute, relative = _compare_numeric(
            reference,
            candidate,
            epsilon=policy["relative_error_epsilon"],
        )
        tensors.append(result)
        absolute_errors.extend(absolute)
        relative_errors.extend(relative)

    observed_numeric = {tensor["role"]: tensor["identity"] for tensor in tensors}
    if observed_numeric != expected_numeric:
        raise ReferenceOracleError(
            "numeric tensor coverage does not match the fixed scope contract"
        )
    observed_router = {record["identity"] for record in router_records}
    if observed_router != expected_router:
        raise ReferenceOracleError(
            "router tensor coverage does not match the fixed scope contract"
        )
    if not absolute_errors:
        raise ReferenceOracleError("Oracle comparison contains no numeric values")

    aggregate = {
        "tensor_count": len(tensors),
        "element_count": sum(
            tensor["metrics"]["element_count"] for tensor in tensors
        ),
        "absolute_error": _distribution(absolute_errors),
        "relative_error": _distribution(relative_errors),
        "minimum_tensor_cosine_similarity": min(
            tensor["metrics"]["cosine_similarity"] for tensor in tensors
        ),
    }
    if router_records:
        decision_count = sum(record["decision_count"] for record in router_records)
        exact = sum(record["exact_set_match_count"] for record in router_records)
        router = {
            "record_count": len(router_records),
            "decision_count": decision_count,
            "exact_set_match_count": exact,
            "top_k_set_agreement": exact / decision_count,
            "mean_set_overlap": sum(
                record["mean_set_overlap"] * record["decision_count"]
                for record in router_records
            )
            / decision_count,
            "records": router_records,
        }
    else:
        router = None
    evaluations = _evaluate(aggregate, router, policy)
    status = (
        "measured"
        if policy["status"] == "draft"
        else ("passed" if all(item["passed"] for item in evaluations) else "failed")
    )
    document = {
        "schema_version": 1,
        "kind": "reference_oracle_comparison",
        "generated_at": _utc_now(),
        "oracle_id": oracle_id,
        "status": status,
        "model": {
            "model_id": spec.model_id,
            "model_revision": model_revision,
            "model_type": spec.model_type,
            "num_layers": spec.num_layers,
            "num_experts": spec.num_experts,
            "top_k": spec.top_k,
        },
        "scope": normalized_scope,
        "fixture": {"id": fixture_id, "sha256": fixture_sha256},
        "implementations": {
            "reference": dict(reference_implementation),
            "candidate": dict(candidate_implementation),
        },
        "quality_policy": policy,
        "tensors": tensors,
        "router": router,
        "aggregate": aggregate,
        "evaluations": evaluations,
    }
    validate_reference_oracle_comparison(document)
    return document


def validate_reference_oracle_comparison(document: Mapping[str, Any]) -> None:
    """Validate schema plus fail-closed policy and aggregate invariants."""

    _reject_non_finite_numbers(document)
    _validate_schema(document)
    model = document["model"]
    try:
        spec = fixed_model_spec(model["model_id"], model["model_revision"])
    except ContractError as exc:
        raise ReferenceOracleError(str(exc)) from exc
    if model != {
        "model_id": spec.model_id,
        "model_revision": spec.model_revision,
        "model_type": spec.model_type,
        "num_layers": spec.num_layers,
        "num_experts": spec.num_experts,
        "top_k": spec.top_k,
    }:
        raise ReferenceOracleError("Oracle model architecture is inconsistent")
    scope = document["scope"]
    _validate_scope(scope, spec)
    expected_numeric, expected_router = _expected_identities(scope, spec)
    observed_numeric = {
        tensor["role"]: tensor["identity"] for tensor in document["tensors"]
    }
    if observed_numeric != expected_numeric:
        raise ReferenceOracleError("numeric tensor coverage violates the scope contract")
    router = document["router"]
    observed_router = (
        set() if router is None else {record["identity"] for record in router["records"]}
    )
    if observed_router != expected_router:
        raise ReferenceOracleError("router tensor coverage violates the scope contract")
    policy = _normalise_policy(document["quality_policy"], level=scope["level"])
    aggregate = document["aggregate"]
    for tensor in document["tensors"]:
        if any(
            tensor[side]["dtype"] not in _NUMERIC_DTYPES
            for side in ("reference", "candidate")
        ):
            raise ReferenceOracleError(
                f"tensor {tensor['identity']!r} has an unsupported output dtype"
            )
        metrics = tensor["metrics"]
        if metrics["element_count"] != math.prod(tensor["shape"]):
            raise ReferenceOracleError(
                f"tensor {tensor['identity']!r} element_count does not match shape"
            )
        for field in ("absolute_error", "relative_error"):
            distribution = metrics[field]
            if not (
                distribution["p50"]
                <= distribution["p95"]
                <= distribution["p99"]
                <= distribution["max"]
            ) or distribution["mean"] > distribution["max"]:
                raise ReferenceOracleError(
                    f"tensor {tensor['identity']!r} has an invalid {field} distribution"
                )
    if aggregate["tensor_count"] != len(document["tensors"]):
        raise ReferenceOracleError("aggregate tensor_count does not match tensors")
    if aggregate["element_count"] != sum(
        tensor["metrics"]["element_count"] for tensor in document["tensors"]
    ):
        raise ReferenceOracleError("aggregate element_count does not match tensors")
    if not math.isclose(
        aggregate["minimum_tensor_cosine_similarity"],
        min(
            tensor["metrics"]["cosine_similarity"]
            for tensor in document["tensors"]
        ),
    ):
        raise ReferenceOracleError(
            "aggregate minimum cosine similarity does not match tensors"
        )
    for field in ("absolute_error", "relative_error"):
        distribution = aggregate[field]
        tensor_max = max(
            tensor["metrics"][field]["max"] for tensor in document["tensors"]
        )
        if not math.isclose(distribution["max"], tensor_max) or not (
            distribution["p50"]
            <= distribution["p95"]
            <= distribution["p99"]
            <= distribution["max"]
        ) or distribution["mean"] > distribution["max"]:
            raise ReferenceOracleError(f"aggregate {field} distribution is invalid")
    if router is not None:
        records = router["records"]
        if len({record["identity"] for record in records}) != len(records):
            raise ReferenceOracleError("router records contain duplicate identities")
        for record in records:
            if any(
                record[side]["dtype"] not in _ROUTER_DTYPES
                for side in ("reference", "candidate")
            ):
                raise ReferenceOracleError(
                    f"router record {record['identity']!r} has an unsupported dtype"
                )
            expected_decisions = math.prod(record["shape"]) // spec.top_k
            if (
                record["shape"][-1] != spec.top_k
                or record["top_k"] != spec.top_k
                or record["decision_count"] != expected_decisions
            ):
                raise ReferenceOracleError(
                    f"router record {record['identity']!r} count does not match shape"
                )
            if record["exact_set_match_count"] > record["decision_count"] or not math.isclose(
                record["top_k_set_agreement"],
                record["exact_set_match_count"] / record["decision_count"],
            ):
                raise ReferenceOracleError(
                    f"router record {record['identity']!r} agreement is inconsistent"
                )
        decision_count = sum(record["decision_count"] for record in records)
        exact = sum(record["exact_set_match_count"] for record in records)
        if router["record_count"] != len(records) or router["decision_count"] != decision_count:
            raise ReferenceOracleError("router aggregate counts do not match records")
        if router["exact_set_match_count"] != exact or not math.isclose(
            router["top_k_set_agreement"], exact / decision_count
        ):
            raise ReferenceOracleError("router agreement does not match record counts")
        expected_overlap = sum(
            record["mean_set_overlap"] * record["decision_count"]
            for record in records
        ) / decision_count
        if not math.isclose(router["mean_set_overlap"], expected_overlap):
            raise ReferenceOracleError("router overlap does not match records")
    expected_evaluations = _evaluate(aggregate, router, policy)
    if document["evaluations"] != expected_evaluations:
        raise ReferenceOracleError("quality evaluations do not match observations")
    expected_status = (
        "measured"
        if policy["status"] == "draft"
        else (
            "passed"
            if all(item["passed"] for item in expected_evaluations)
            else "failed"
        )
    )
    if document["status"] != expected_status:
        raise ReferenceOracleError("comparison status does not match quality evaluations")


__all__ = [
    "ReferenceOracleError",
    "build_reference_oracle_comparison",
    "iter_tensor_records",
    "tensor_record_from_npy",
    "validate_reference_oracle_comparison",
]
