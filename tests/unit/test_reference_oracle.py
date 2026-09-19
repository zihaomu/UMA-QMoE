from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest

from uma_qmoe.reference_oracle import (
    ReferenceOracleError,
    build_reference_oracle_comparison,
    iter_tensor_records,
    tensor_record_from_npy,
    validate_reference_oracle_comparison,
)


REVISION = "9b0c1aa87e34a20052389dce1f0cf01da783f654"
SHA256 = "a" * 64
REFERENCE = {"id": "torch-f32", "precision": "F32", "artifact_sha256": "b" * 64}
CANDIDATE = {"id": "torch-bf16", "precision": "BF16", "artifact_sha256": "c" * 64}


def _record(
    identity: str,
    role: str,
    values: list[float] | list[int],
    *,
    shape: list[int] | None = None,
    dtype: str = "float32",
) -> dict:
    return {
        "identity": identity,
        "role": role,
        "dtype": dtype,
        "shape": [len(values)] if shape is None else shape,
        "values": values,
    }


def _build(
    reference: list[dict],
    candidate: list[dict],
    *,
    level: str = "single_expert",
    layer_index: int | None = 2,
    expert_index: int | None = 7,
    policy: dict | None = None,
) -> dict:
    return build_reference_oracle_comparison(
        reference,
        candidate,
        oracle_id="olmoe-oracle-v1",
        model_revision=REVISION,
        scope={
            "level": level,
            "layer_index": layer_index,
            "expert_index": expert_index,
        },
        fixture_id="fixture-001",
        fixture_sha256=SHA256,
        reference_implementation=REFERENCE,
        candidate_implementation=CANDIDATE,
        quality_policy=policy,
    )


def _frozen_policy(*, absolute: float = 0.2, router: float | None = None) -> dict:
    return {
        "status": "frozen",
        "percentile_method": "nearest_rank",
        "relative_error_epsilon": 1e-12,
        "max_absolute_error": absolute,
        "max_relative_error": 0.2,
        "max_p99_absolute_error": absolute,
        "max_p99_relative_error": 0.2,
        "min_cosine_similarity": 0.99,
        "min_router_top_k_set_agreement": router,
    }


def test_single_expert_draft_records_identity_hash_and_error_distributions() -> None:
    identity = "model.layers.2.mlp.experts.7.output"
    reference = [_record(identity, "expert_output", [1.0, -2.0, 4.0, 0.0])]
    candidate = [_record(identity, "expert_output", [1.1, -2.0, 3.8, 0.0])]

    document = _build(reference, candidate)

    assert document["kind"] == "reference_oracle_comparison"
    assert document["schema_version"] == 1
    assert document["status"] == "measured"
    assert document["evaluations"] == []
    assert document["router"] is None
    tensor = document["tensors"][0]
    assert tensor["identity"] == identity
    assert tensor["shape"] == [4]
    assert tensor["reference"]["dtype"] == "float32"
    assert len(tensor["reference"]["sha256"]) == 64
    assert tensor["metrics"]["finite"] is True
    assert tensor["metrics"]["absolute_error"]["max"] == pytest.approx(0.2)
    assert tensor["metrics"]["relative_error"]["max"] == pytest.approx(0.1)
    assert tensor["metrics"]["cosine_similarity"] > 0.99
    validate_reference_oracle_comparison(document)


def test_frozen_single_expert_policy_is_fail_closed() -> None:
    identity = "model.layers.2.mlp.experts.7.output"
    reference = [_record(identity, "expert_output", [1.0, 2.0])]
    candidate = [_record(identity, "expert_output", [1.01, 1.99])]

    passed = _build(reference, candidate, policy=_frozen_policy(absolute=0.02))
    assert passed["status"] == "passed"
    assert len(passed["evaluations"]) == 5

    failed = _build(reference, candidate, policy=_frozen_policy(absolute=0.001))
    assert failed["status"] == "failed"
    assert any(not evaluation["passed"] for evaluation in failed["evaluations"])

    missing = _frozen_policy()
    missing["min_cosine_similarity"] = None
    with pytest.raises(ReferenceOracleError, match="missing thresholds"):
        _build(reference, candidate, policy=missing)


def test_single_moe_layer_compares_router_sets_independent_of_order() -> None:
    layer = 3
    router_logits = f"model.layers.{layer}.mlp.router_logits"
    output = f"model.layers.{layer}.mlp.output"
    topk = f"model.layers.{layer}.mlp.router_topk_indices"
    reference = [
        _record(router_logits, "router_logits", [0.0, 1.0, 2.0]),
        _record(output, "moe_layer_output", [1.0, 2.0, 3.0]),
        _record(topk, "router_topk_indices", list(range(8)), shape=[1, 8], dtype="int64"),
    ]
    candidate = [
        _record(router_logits, "router_logits", [0.0, 1.0, 2.0]),
        _record(output, "moe_layer_output", [1.0, 2.0, 3.0]),
        _record(
            topk,
            "router_topk_indices",
            list(reversed(range(8))),
            shape=[1, 8],
            dtype="int32",
        ),
    ]

    document = _build(
        reference,
        candidate,
        level="single_moe_layer",
        layer_index=layer,
        expert_index=None,
        policy=_frozen_policy(router=1.0),
    )

    assert document["status"] == "passed"
    assert document["router"]["top_k_set_agreement"] == 1.0
    assert document["router"]["mean_set_overlap"] == 1.0


def test_full_model_requires_all_sixteen_router_layers() -> None:
    reference = [_record("model.final_logits", "final_logits", [1.0, 2.0])]
    candidate = [_record("model.final_logits", "final_logits", [1.0, 2.0])]
    for layer in range(16):
        identity = f"model.layers.{layer}.mlp.router_topk_indices"
        reference.append(
            _record(identity, "router_topk_indices", list(range(8)), shape=[1, 8], dtype="int64")
        )
        candidate.append(
            _record(identity, "router_topk_indices", list(range(8)), shape=[1, 8], dtype="int64")
        )

    document = _build(
        reference,
        candidate,
        level="full_model",
        layer_index=None,
        expert_index=None,
    )
    assert document["router"]["record_count"] == 16
    assert document["router"]["decision_count"] == 16

    with pytest.raises(ReferenceOracleError, match="coverage"):
        _build(
            reference[:-1],
            candidate[:-1],
            level="full_model",
            layer_index=None,
            expert_index=None,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda records: records[0]["values"].__setitem__(0, float("nan")), "NaN or Inf"),
        (lambda records: records[0].__setitem__("identity", "wrong"), "mismatch"),
        (lambda records: records[0].__setitem__("shape", [3]), "shape"),
    ],
)
def test_rejects_nonfinite_identity_and_shape_mismatches(mutation, message: str) -> None:
    identity = "model.layers.2.mlp.experts.7.output"
    reference = [_record(identity, "expert_output", [1.0, 2.0])]
    candidate = [_record(identity, "expert_output", [1.0, 2.0])]
    mutation(candidate)
    with pytest.raises(ReferenceOracleError, match=message):
        _build(reference, candidate)


def test_rejects_invalid_router_rows_and_stream_length() -> None:
    layer = 3
    router_logits = f"model.layers.{layer}.mlp.router_logits"
    output = f"model.layers.{layer}.mlp.output"
    topk = f"model.layers.{layer}.mlp.router_topk_indices"
    reference = [
        _record(router_logits, "router_logits", [0.0]),
        _record(output, "moe_layer_output", [0.0]),
        _record(topk, "router_topk_indices", [0] * 8, shape=[1, 8], dtype="int64"),
    ]
    candidate = copy.deepcopy(reference)
    with pytest.raises(ReferenceOracleError, match="duplicate expert ids"):
        _build(
            reference,
            candidate,
            level="single_moe_layer",
            layer_index=layer,
            expert_index=None,
        )

    identity = "model.layers.2.mlp.experts.7.output"
    with pytest.raises(ReferenceOracleError, match="value counts"):
        _build(
            [_record(identity, "expert_output", [1.0, 2.0])],
            [_record(identity, "expert_output", [1.0], shape=[2])],
        )


def test_jsonl_and_memory_mapped_npy_sources(tmp_path: Path) -> None:
    identity = "model.layers.2.mlp.experts.7.output"
    np.save(tmp_path / "candidate.npy", np.asarray([1.0, 2.0], dtype=np.float32))
    stream = tmp_path / "candidate.jsonl"
    stream.write_text(
        json.dumps(
            {
                "identity": identity,
                "role": "expert_output",
                "npy_path": "candidate.npy",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    records = list(iter_tensor_records(stream))
    assert records[0]["dtype"] == "float32"
    assert list(records[0]["values"]) == [1.0, 2.0]
    direct = tensor_record_from_npy(
        tmp_path / "candidate.npy", identity=identity, role="expert_output"
    )
    document = _build([_record(identity, "expert_output", [1.0, 2.0])], [direct])
    assert document["aggregate"]["absolute_error"]["max"] == 0.0


def test_output_validator_rejects_unknown_nonfinite_and_tampered_fields() -> None:
    identity = "model.layers.2.mlp.experts.7.output"
    document = _build(
        [_record(identity, "expert_output", [1.0, 2.0])],
        [_record(identity, "expert_output", [1.0, 2.0])],
    )

    unknown = copy.deepcopy(document)
    unknown["unexpected"] = True
    with pytest.raises(ReferenceOracleError, match="Additional properties"):
        validate_reference_oracle_comparison(unknown)

    tampered = copy.deepcopy(document)
    tampered["aggregate"]["element_count"] += 1
    with pytest.raises(ReferenceOracleError, match="element_count"):
        validate_reference_oracle_comparison(tampered)

    nonfinite = copy.deepcopy(document)
    nonfinite["aggregate"]["absolute_error"]["max"] = float("inf")
    with pytest.raises(ReferenceOracleError):
        validate_reference_oracle_comparison(nonfinite)
