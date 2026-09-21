"""Torch-free data types shared by exploratory quantization experiments."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import math
from types import MappingProxyType
from typing import Any, TypeAlias


JsonScalar: TypeAlias = None | bool | int | float | str
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
FrozenJsonValue: TypeAlias = JsonScalar | tuple["FrozenJsonValue", ...] | Mapping[
    str, "FrozenJsonValue"
]


class ExperimentTypeError(ValueError):
    """Raised when an experiment identity contains an unstable value."""


def _freeze_json(value: Any, *, path: str = "$") -> FrozenJsonValue:
    if value is None or isinstance(value, (bool, str)):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExperimentTypeError(f"{path} must not contain NaN or infinity")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, FrozenJsonValue] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ExperimentTypeError(f"{path} has a non-string object key")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(frozen)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(
            _freeze_json(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    raise ExperimentTypeError(f"{path} contains non-JSON value {type(value).__name__}")


def thaw_json(value: FrozenJsonValue) -> JsonValue:
    """Return an ordinary JSON-compatible copy of a frozen value."""

    if isinstance(value, Mapping):
        return {key: thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [thaw_json(item) for item in value]
    return value


def _nonempty(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentTypeError(f"{field_name} must be a non-empty string")
    return value


@dataclass(frozen=True, order=True)
class Unit:
    layer: int
    projection: str
    expert: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.layer, bool) or not isinstance(self.layer, int) or self.layer < 0:
            raise ExperimentTypeError("unit layer must be a non-negative integer")
        _nonempty(self.projection, "unit projection")
        if self.expert is not None and (
            isinstance(self.expert, bool)
            or not isinstance(self.expert, int)
            or self.expert < 0
        ):
            raise ExperimentTypeError("unit expert must be a non-negative integer")

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "layer": self.layer,
            "projection": self.projection,
            "expert": self.expert,
        }


@dataclass(frozen=True)
class Encoding:
    storage: str
    group_size: int | None
    method: str
    parameters: Mapping[str, FrozenJsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "storage", _nonempty(self.storage, "storage").lower())
        object.__setattr__(self, "method", _nonempty(self.method, "method").lower())
        if self.group_size is not None and (
            isinstance(self.group_size, bool)
            or not isinstance(self.group_size, int)
            or self.group_size <= 0
        ):
            raise ExperimentTypeError("group_size must be a positive integer or null")
        frozen = _freeze_json(self.parameters, path="$.parameters")
        assert isinstance(frozen, Mapping)
        object.__setattr__(self, "parameters", frozen)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "storage": self.storage,
            "group_size": self.group_size,
            "method": self.method,
            "parameters": thaw_json(self.parameters),
        }


@dataclass(frozen=True)
class CalibrationView:
    identity: Mapping[str, FrozenJsonValue]
    statistics: Mapping[str, FrozenJsonValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        identity = _freeze_json(self.identity, path="$.calibration.identity")
        statistics = _freeze_json(self.statistics, path="$.calibration.statistics")
        assert isinstance(identity, Mapping) and isinstance(statistics, Mapping)
        object.__setattr__(self, "identity", identity)
        object.__setattr__(self, "statistics", statistics)


@dataclass(frozen=True)
class TrialSpec:
    model_identity: Mapping[str, FrozenJsonValue]
    data_identity: Mapping[str, FrozenJsonValue]
    unit: Unit
    encoding: Encoding
    quantizer_version: str
    evaluation: Mapping[str, FrozenJsonValue]
    seed: int = 0
    diagnostic_only: bool = True
    schema_version: int = 1

    def __post_init__(self) -> None:
        model = _freeze_json(self.model_identity, path="$.model_identity")
        data = _freeze_json(self.data_identity, path="$.data_identity")
        evaluation = _freeze_json(self.evaluation, path="$.evaluation")
        if not isinstance(model, Mapping) or not model:
            raise ExperimentTypeError("model_identity must be a non-empty object")
        if not isinstance(data, Mapping) or not data:
            raise ExperimentTypeError("data_identity must be a non-empty object")
        if not isinstance(evaluation, Mapping):
            raise ExperimentTypeError("evaluation must be an object")
        _nonempty(self.quantizer_version, "quantizer_version")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ExperimentTypeError("seed must be a non-negative integer")
        if not isinstance(self.diagnostic_only, bool):
            raise ExperimentTypeError("diagnostic_only must be boolean")
        if self.schema_version != 1:
            raise ExperimentTypeError("unsupported trial spec schema version")
        object.__setattr__(self, "model_identity", model)
        object.__setattr__(self, "data_identity", data)
        object.__setattr__(self, "evaluation", evaluation)

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "model_identity": thaw_json(self.model_identity),
            "data_identity": thaw_json(self.data_identity),
            "unit": self.unit.to_dict(),
            "encoding": self.encoding.to_dict(),
            "quantizer_version": self.quantizer_version,
            "evaluation": thaw_json(self.evaluation),
            "seed": self.seed,
            "diagnostic_only": self.diagnostic_only,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TrialSpec":
        unit = value["unit"]
        encoding = value["encoding"]
        return cls(
            schema_version=int(value.get("schema_version", 1)),
            model_identity=value["model_identity"],
            data_identity=value["data_identity"],
            unit=Unit(
                layer=unit["layer"],
                projection=unit["projection"],
                expert=unit.get("expert"),
            ),
            encoding=Encoding(
                storage=encoding["storage"],
                group_size=encoding.get("group_size"),
                method=encoding["method"],
                parameters=encoding.get("parameters", {}),
            ),
            quantizer_version=value["quantizer_version"],
            evaluation=value.get("evaluation", {}),
            seed=value.get("seed", 0),
            diagnostic_only=value.get("diagnostic_only", True),
        )


class EvidenceLevel(str, Enum):
    EXPLORATORY = "L0"
    FORMAL = "L1"


class TrialStatus(str, Enum):
    PASSED = "passed"
    REJECTED = "rejected"
    ERROR = "error"
    OOM = "oom"
    INTERRUPTED = "interrupted"


COMPLETED_STATUSES = frozenset(
    {TrialStatus.PASSED, TrialStatus.REJECTED, TrialStatus.ERROR, TrialStatus.OOM}
)


@dataclass(frozen=True)
class TrialResult:
    trial_id: str
    spec: TrialSpec
    status: TrialStatus
    started_at: str
    finished_at: str
    wall_time_seconds: float
    offline_metrics: Mapping[str, FrozenJsonValue] = field(default_factory=dict)
    quality_metrics: Mapping[str, FrozenJsonValue] = field(default_factory=dict)
    gate: Mapping[str, FrozenJsonValue] = field(default_factory=dict)
    memory: Mapping[str, FrozenJsonValue] = field(default_factory=dict)
    kernel_metrics: Mapping[str, FrozenJsonValue] = field(default_factory=dict)
    diagnostics: Mapping[str, FrozenJsonValue] = field(default_factory=dict)
    error: Mapping[str, FrozenJsonValue] | None = None
    provenance: Mapping[str, FrozenJsonValue] = field(default_factory=dict)
    evidence_level: EvidenceLevel = EvidenceLevel.EXPLORATORY
    schema_version: int = 1

    def __post_init__(self) -> None:
        _nonempty(self.trial_id, "trial_id")
        if len(self.trial_id) != 64 or any(c not in "0123456789abcdef" for c in self.trial_id):
            raise ExperimentTypeError("trial_id must be a lowercase SHA-256 digest")
        if not isinstance(self.status, TrialStatus):
            object.__setattr__(self, "status", TrialStatus(self.status))
        if not isinstance(self.evidence_level, EvidenceLevel):
            object.__setattr__(self, "evidence_level", EvidenceLevel(self.evidence_level))
        if self.evidence_level != EvidenceLevel.EXPLORATORY:
            raise ExperimentTypeError("TrialResult may only contain L0 evidence")
        if self.schema_version != 1:
            raise ExperimentTypeError("unsupported trial result schema version")
        _nonempty(self.started_at, "started_at")
        _nonempty(self.finished_at, "finished_at")
        if not math.isfinite(self.wall_time_seconds) or self.wall_time_seconds < 0:
            raise ExperimentTypeError("wall_time_seconds must be finite and non-negative")
        for name in (
            "offline_metrics",
            "quality_metrics",
            "gate",
            "memory",
            "kernel_metrics",
            "diagnostics",
            "provenance",
        ):
            frozen = _freeze_json(getattr(self, name), path=f"$.{name}")
            assert isinstance(frozen, Mapping)
            object.__setattr__(self, name, frozen)
        if self.error is not None:
            error = _freeze_json(self.error, path="$.error")
            assert isinstance(error, Mapping)
            object.__setattr__(self, "error", error)

    @property
    def complete(self) -> bool:
        return self.status in COMPLETED_STATUSES

    def to_dict(self) -> dict[str, JsonValue]:
        return {
            "schema_version": self.schema_version,
            "kind": "exploratory_trial_result",
            "evidence_level": self.evidence_level.value,
            "trial_id": self.trial_id,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "wall_time_seconds": self.wall_time_seconds,
            "spec": self.spec.to_dict(),
            "offline_metrics": thaw_json(self.offline_metrics),
            "quality_metrics": thaw_json(self.quality_metrics),
            "gate": thaw_json(self.gate),
            "memory": thaw_json(self.memory),
            "kernel_metrics": thaw_json(self.kernel_metrics),
            "diagnostics": thaw_json(self.diagnostics),
            "error": None if self.error is None else thaw_json(self.error),
            "provenance": thaw_json(self.provenance),
        }


__all__ = [
    "COMPLETED_STATUSES",
    "CalibrationView",
    "Encoding",
    "EvidenceLevel",
    "ExperimentTypeError",
    "FrozenJsonValue",
    "JsonValue",
    "TrialResult",
    "TrialSpec",
    "TrialStatus",
    "Unit",
    "thaw_json",
]
