"""Pure Safe UMA Budget calculation from a captured memory snapshot.

The calculation deliberately uses installed memory, optionally constrained by
a finite cgroup v2 limit, as its physical ceiling.  ``MemAvailable`` is a
volatile observation and therefore never changes the resulting budget.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contracts import canonical_sha256


_POLICY_COMPONENTS = (
    "os_daemon_reserve_bytes",
    "runtime_reserve_bytes",
    "kv_budget_bytes",
    "workspace_budget_bytes",
    "safety_margin_bytes",
)
_STATUSES = frozenset({"draft", "frozen"})


class SafeBudgetError(ValueError):
    """Raised when a Safe UMA Budget input is missing or inconsistent."""


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise SafeBudgetError(f"{name} must be a mapping")
    return value


def _require_nonnegative_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise SafeBudgetError(f"{name} must be a non-negative integer")
    if value < 0:
        raise SafeBudgetError(f"{name} must be non-negative")
    return value


def _snapshot_limits(
    memory_snapshot: Mapping[str, Any],
) -> tuple[str, int, int | None, int]:
    if memory_snapshot.get("schema_version") != 1:
        raise SafeBudgetError("memory_snapshot.schema_version must be 1")
    if memory_snapshot.get("kind") != "memory_snapshot":
        raise SafeBudgetError("memory_snapshot.kind must be 'memory_snapshot'")

    target_id = memory_snapshot.get("target_id")
    if not isinstance(target_id, str) or not target_id.strip():
        raise SafeBudgetError("memory_snapshot.target_id must be a non-empty string")

    memory = _require_mapping(memory_snapshot.get("memory"), "memory_snapshot.memory")
    mem_total_bytes = _require_nonnegative_int(
        memory.get("mem_total_bytes"), "memory_snapshot.memory.mem_total_bytes"
    )

    cgroup = _require_mapping(
        memory_snapshot.get("cgroup_v2"), "memory_snapshot.cgroup_v2"
    )
    raw_cgroup_max = cgroup.get("memory_max_bytes")
    maximum_is_unlimited = cgroup.get("memory_max_is_unlimited")
    if (
        maximum_is_unlimited is not True
        and maximum_is_unlimited is not False
        and maximum_is_unlimited is not None
    ):
        raise SafeBudgetError(
            "memory_snapshot.cgroup_v2.memory_max_is_unlimited must be a boolean or null"
        )

    cgroup_max_bytes: int | None
    if raw_cgroup_max is None:
        if maximum_is_unlimited is False:
            raise SafeBudgetError(
                "finite cgroup memory maximum is missing memory_max_bytes"
            )
        cgroup_max_bytes = None
    else:
        cgroup_max_bytes = _require_nonnegative_int(
            raw_cgroup_max, "memory_snapshot.cgroup_v2.memory_max_bytes"
        )
        if maximum_is_unlimited is True:
            raise SafeBudgetError(
                "cgroup memory maximum cannot be both finite and unlimited"
            )

    physical_limit_bytes = (
        mem_total_bytes
        if cgroup_max_bytes is None
        else min(mem_total_bytes, cgroup_max_bytes)
    )
    return target_id.strip(), mem_total_bytes, cgroup_max_bytes, physical_limit_bytes


def _validated_policy(policy: Mapping[str, Any]) -> dict[str, int]:
    supplied = set(policy)
    expected = set(_POLICY_COMPONENTS)
    missing = sorted(expected - supplied)
    unexpected = sorted((repr(key) for key in supplied - expected))
    if missing:
        raise SafeBudgetError(
            "policy is missing required components: " + ", ".join(missing)
        )
    if unexpected:
        raise SafeBudgetError(
            "policy contains unexpected components: " + ", ".join(unexpected)
        )
    return {
        component: _require_nonnegative_int(policy[component], f"policy.{component}")
        for component in _POLICY_COMPONENTS
    }


def build_safe_uma_budget(
    memory_snapshot: Mapping[str, Any],
    *,
    policy: Mapping[str, Any],
    status: str = "draft",
    policy_provenance: str = "",
    decision_record: str = "",
) -> dict[str, Any]:
    """Build a deterministic Safe UMA Budget document.

    Every reserve must be explicitly present in ``policy``.  The physical
    limit is ``min(MemTotal, finite cgroup memory.max)``.  In particular,
    ``MemAvailable`` is intentionally ignored because it is a transient
    measurement rather than stable machine capacity.

    A frozen budget additionally requires non-empty provenance and a decision
    record so measured policy choices cannot silently become release gates.

    Raises:
        SafeBudgetError: If an input is missing, malformed, or overcommitted.
    """

    snapshot = _require_mapping(memory_snapshot, "memory_snapshot")
    explicit_policy = _require_mapping(policy, "policy")
    source_hash = canonical_sha256(snapshot)

    if not isinstance(status, str) or status not in _STATUSES:
        raise SafeBudgetError("status must be 'draft' or 'frozen'")
    if not isinstance(policy_provenance, str):
        raise SafeBudgetError("policy_provenance must be a string")
    if not isinstance(decision_record, str):
        raise SafeBudgetError("decision_record must be a string")

    provenance = policy_provenance.strip()
    decision = decision_record.strip()
    if status == "frozen" and not provenance:
        raise SafeBudgetError("frozen budget requires non-empty policy_provenance")
    if status == "frozen" and not decision:
        raise SafeBudgetError("frozen budget requires non-empty decision_record")
    target_id, mem_total_bytes, cgroup_max_bytes, physical_limit_bytes = (
        _snapshot_limits(snapshot)
    )
    if (
        status == "frozen"
        and snapshot["cgroup_v2"].get("memory_max_is_unlimited") is None
    ):
        raise SafeBudgetError(
            "frozen budget requires an explicitly finite or unlimited cgroup memory maximum"
        )
    components = _validated_policy(explicit_policy)
    total_reserved_bytes = sum(components.values())
    if total_reserved_bytes > physical_limit_bytes:
        raise SafeBudgetError(
            "total policy reserve exceeds the physical memory limit "
            f"({total_reserved_bytes} > {physical_limit_bytes})"
        )

    return {
        "schema_version": 1,
        "kind": "safe_uma_budget",
        "target_id": target_id,
        "source_snapshot_sha256": source_hash,
        "status": status,
        "policy_provenance": provenance,
        "decision_record": decision,
        "mem_total_bytes": mem_total_bytes,
        "cgroup_memory_max_bytes": cgroup_max_bytes,
        "physical_limit_bytes": physical_limit_bytes,
        **components,
        "total_reserved_bytes": total_reserved_bytes,
        "safe_budget_bytes": physical_limit_bytes - total_reserved_bytes,
    }


__all__ = ["SafeBudgetError", "build_safe_uma_budget"]
