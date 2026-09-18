from __future__ import annotations

from copy import deepcopy
from typing import Any

import pytest

from uma_qmoe.safe_budget import SafeBudgetError, build_safe_uma_budget
from uma_qmoe.contracts import ContractError, canonical_sha256, validate_document


def _snapshot(
    *,
    mem_total_bytes: int = 1_000,
    mem_available_bytes: int = 1,
    cgroup_max_bytes: int | None = None,
    cgroup_is_unlimited: bool | None = True,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "memory_snapshot",
        "target_id": "halo-test",
        "memory": {
            "mem_total_bytes": mem_total_bytes,
            "mem_available_bytes": mem_available_bytes,
        },
        "cgroup_v2": {
            "memory_max_bytes": cgroup_max_bytes,
            "memory_max_is_unlimited": cgroup_is_unlimited,
        },
    }


def _policy(**overrides: int) -> dict[str, int]:
    values = {
        "os_daemon_reserve_bytes": 100,
        "runtime_reserve_bytes": 50,
        "kv_budget_bytes": 200,
        "workspace_budget_bytes": 75,
        "safety_margin_bytes": 25,
    }
    values.update(overrides)
    return values


def test_draft_uses_physical_memory_and_never_transient_mem_available() -> None:
    snapshot = _snapshot(mem_total_bytes=1_000, mem_available_bytes=1)
    policy = _policy()
    original_snapshot = deepcopy(snapshot)
    original_policy = deepcopy(policy)

    budget = build_safe_uma_budget(
        snapshot,
        policy=policy,
    )

    assert budget == {
        "schema_version": 1,
        "kind": "safe_uma_budget",
        "target_id": "halo-test",
        "source_snapshot_sha256": canonical_sha256(snapshot),
        "status": "draft",
        "policy_provenance": "",
        "decision_record": "",
        "mem_total_bytes": 1_000,
        "cgroup_memory_max_bytes": None,
        "physical_limit_bytes": 1_000,
        "os_daemon_reserve_bytes": 100,
        "runtime_reserve_bytes": 50,
        "kv_budget_bytes": 200,
        "workspace_budget_bytes": 75,
        "safety_margin_bytes": 25,
        "total_reserved_bytes": 450,
        "safe_budget_bytes": 550,
    }
    assert snapshot == original_snapshot
    assert policy == original_policy
    validate_document(budget)


def test_finite_cgroup_limit_caps_budget() -> None:
    budget = build_safe_uma_budget(
        _snapshot(
            mem_total_bytes=1_000,
            mem_available_bytes=900,
            cgroup_max_bytes=600,
            cgroup_is_unlimited=False,
        ),
        policy=_policy(),
    )

    assert budget["mem_total_bytes"] == 1_000
    assert budget["cgroup_memory_max_bytes"] == 600
    assert budget["physical_limit_bytes"] == 600
    assert budget["safe_budget_bytes"] == 150


def test_schema_validation_rejects_forged_physical_limit() -> None:
    budget = build_safe_uma_budget(
        _snapshot(
            mem_total_bytes=1_000,
            cgroup_max_bytes=600,
            cgroup_is_unlimited=False,
        ),
        policy=_policy(),
    )
    budget["physical_limit_bytes"] = 900
    budget["safe_budget_bytes"] = 450

    with pytest.raises(ContractError, match="physical_limit_bytes"):
        validate_document(budget)


def test_cgroup_limit_above_installed_memory_does_not_raise_ceiling() -> None:
    budget = build_safe_uma_budget(
        _snapshot(
            mem_total_bytes=1_000,
            cgroup_max_bytes=2_000,
            cgroup_is_unlimited=False,
        ),
        policy=_policy(),
    )

    assert budget["physical_limit_bytes"] == 1_000
    assert budget["safe_budget_bytes"] == 550


def test_reserve_equal_to_limit_is_valid_zero_budget_boundary() -> None:
    budget = build_safe_uma_budget(
        _snapshot(mem_total_bytes=10),
        policy=_policy(
            os_daemon_reserve_bytes=2,
            runtime_reserve_bytes=2,
            kv_budget_bytes=2,
            workspace_budget_bytes=2,
            safety_margin_bytes=2,
        ),
    )

    assert budget["total_reserved_bytes"] == 10
    assert budget["safe_budget_bytes"] == 0


def test_total_reserve_cannot_exceed_physical_limit() -> None:
    with pytest.raises(SafeBudgetError, match="exceeds the physical memory limit"):
        build_safe_uma_budget(
            _snapshot(
                mem_total_bytes=1_000,
                cgroup_max_bytes=400,
                cgroup_is_unlimited=False,
            ),
            policy=_policy(),
        )


@pytest.mark.parametrize(
    "component",
    [
        "os_daemon_reserve_bytes",
        "runtime_reserve_bytes",
        "kv_budget_bytes",
        "workspace_budget_bytes",
        "safety_margin_bytes",
    ],
)
def test_each_policy_component_must_be_nonnegative(component: str) -> None:
    with pytest.raises(SafeBudgetError, match=component):
        build_safe_uma_budget(
            _snapshot(),
            policy=_policy(**{component: -1}),
        )


@pytest.mark.parametrize("invalid", [True, 1.5, "1"])
def test_policy_components_must_be_integers_not_coercions(invalid: object) -> None:
    policy: dict[str, Any] = _policy()
    policy["runtime_reserve_bytes"] = invalid

    with pytest.raises(SafeBudgetError, match="runtime_reserve_bytes"):
        build_safe_uma_budget(
            _snapshot(),
            policy=policy,
        )


def test_policy_requires_exact_explicit_component_set() -> None:
    missing = _policy()
    del missing["kv_budget_bytes"]
    with pytest.raises(SafeBudgetError, match="missing.*kv_budget_bytes"):
        build_safe_uma_budget(
            _snapshot(),
            policy=missing,
        )

    extra: dict[str, Any] = _policy()
    extra["mem_available_bytes"] = 999
    with pytest.raises(SafeBudgetError, match="unexpected.*mem_available_bytes"):
        build_safe_uma_budget(
            _snapshot(),
            policy=extra,
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("mem_total_bytes", -1),
        ("cgroup_max_bytes", -1),
    ],
)
def test_physical_limits_must_be_nonnegative(field: str, value: int) -> None:
    arguments: dict[str, Any] = {field: value}
    if field == "cgroup_max_bytes":
        arguments["cgroup_is_unlimited"] = False

    with pytest.raises(SafeBudgetError, match="must be non-negative"):
        build_safe_uma_budget(
            _snapshot(**arguments),
            policy=_policy(),
        )


def test_contradictory_cgroup_limit_is_rejected() -> None:
    with pytest.raises(SafeBudgetError, match="both finite and unlimited"):
        build_safe_uma_budget(
            _snapshot(cgroup_max_bytes=600, cgroup_is_unlimited=True),
            policy=_policy(),
        )


def test_finite_cgroup_marker_requires_a_limit() -> None:
    with pytest.raises(SafeBudgetError, match="finite.*missing"):
        build_safe_uma_budget(
            _snapshot(cgroup_max_bytes=None, cgroup_is_unlimited=False),
            policy=_policy(),
        )


@pytest.mark.parametrize("invalid_flag", [0, 1, "false", {}])
def test_cgroup_unlimited_marker_must_be_boolean_or_null(
    invalid_flag: object,
) -> None:
    snapshot = _snapshot()
    snapshot["cgroup_v2"]["memory_max_is_unlimited"] = invalid_flag

    with pytest.raises(SafeBudgetError, match="boolean or null"):
        build_safe_uma_budget(
            snapshot,
            policy=_policy(),
        )


def test_source_snapshot_hash_is_derived_not_caller_supplied() -> None:
    snapshot = _snapshot(mem_available_bytes=777)
    budget = build_safe_uma_budget(snapshot, policy=_policy())

    assert budget["source_snapshot_sha256"] == canonical_sha256(snapshot)


@pytest.mark.parametrize("status", ["", "validated", "FROZEN"])
def test_status_is_draft_or_frozen(status: str) -> None:
    with pytest.raises(SafeBudgetError, match="status"):
        build_safe_uma_budget(
            _snapshot(),
            policy=_policy(),
            status=status,
        )


@pytest.mark.parametrize(
    ("provenance", "decision", "message"),
    [
        ("", "allocation-matrix/halo.json", "policy_provenance"),
        ("   ", "allocation-matrix/halo.json", "policy_provenance"),
        ("stress-and-allocation-matrix-v1", "", "decision_record"),
        ("stress-and-allocation-matrix-v1", "  ", "decision_record"),
    ],
)
def test_frozen_requires_policy_evidence(
    provenance: str, decision: str, message: str
) -> None:
    with pytest.raises(SafeBudgetError, match=message):
        build_safe_uma_budget(
            _snapshot(),
            policy=_policy(),
            status="frozen",
            policy_provenance=provenance,
            decision_record=decision,
        )


def test_frozen_budget_records_trimmed_policy_evidence() -> None:
    budget = build_safe_uma_budget(
        _snapshot(),
        policy=_policy(),
        status="frozen",
        policy_provenance="  stress-and-allocation-matrix-v1  ",
        decision_record="  decisions/halo-safe-budget.md  ",
    )

    assert budget["status"] == "frozen"
    assert budget["policy_provenance"] == "stress-and-allocation-matrix-v1"
    assert budget["decision_record"] == "decisions/halo-safe-budget.md"


def test_frozen_budget_rejects_unknown_cgroup_ceiling() -> None:
    with pytest.raises(SafeBudgetError, match="explicitly finite or unlimited"):
        build_safe_uma_budget(
            _snapshot(cgroup_max_bytes=None, cgroup_is_unlimited=None),
            policy=_policy(),
            status="frozen",
            policy_provenance="stress-and-allocation-matrix-v1",
            decision_record="decisions/halo-safe-budget.md",
        )


def test_require_frozen_rejects_draft_budget() -> None:
    budget = build_safe_uma_budget(_snapshot(), policy=_policy())

    with pytest.raises(ContractError, match="draft"):
        validate_document(budget, require_frozen=True)
