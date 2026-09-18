from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import pytest

from uma_qmoe.contracts import validate_document
from uma_qmoe.huggingface import HuggingFaceImportError, build_model_manifest_draft


REPO_ID = "allenai/OLMoE-1B-7B-0125"
REVISION = "9b0c1aa87e34a20052389dce1f0cf01da783f654"


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _olmoe_fixture() -> tuple[dict[str, Any], dict[str, bytes]]:
    files = {
        "config.json": _json_bytes(
            {
                "architectures": ["OlmoeForCausalLM"],
                "hidden_size": 2048,
                "intermediate_size": 1024,
                "max_position_embeddings": 4096,
                "model_type": "olmoe",
                "norm_topk_prob": False,
                "num_experts": 64,
                "num_experts_per_tok": 8,
                "num_hidden_layers": 16,
                "torch_dtype": "float32",
            }
        ),
        "tokenizer.json": b'{"version":"1.0"}',
        "tokenizer_config.json": _json_bytes({"chat_template": None}),
        "special_tokens_map.json": _json_bytes({"eos_token": "<|endoftext|>"}),
        "model.safetensors.index.json": _json_bytes(
            {
                "metadata": {"total_size": 300},
                "weight_map": {
                    "model.layers.0.mlp.experts.0.down_proj.weight": (
                        "model-00001-of-00002.safetensors"
                    ),
                    "model.layers.15.mlp.experts.63.up_proj.weight": (
                        "model-00002-of-00002.safetensors"
                    ),
                },
            }
        ),
    }
    siblings: list[dict[str, Any]] = [
        {"rfilename": path, "size": len(payload)} for path, payload in files.items()
    ]
    siblings.extend(
        [
            {
                "rfilename": "model-00001-of-00002.safetensors",
                "size": 100,
                "lfs": {"sha256": "a" * 64, "size": 100},
            },
            {
                "rfilename": "model-00002-of-00002.safetensors",
                "size": 200,
                "lfs": {"sha256": "b" * 64, "size": 200},
            },
        ]
    )
    metadata = {
        "id": REPO_ID,
        "sha": REVISION,
        "cardData": {"license": "apache-2.0"},
        "safetensors": {"parameters": {"F32": 6_919_161_856}},
        "siblings": siblings,
    }
    return metadata, files


class OfflineFetcher:
    def __init__(self, metadata: Mapping[str, Any], files: Mapping[str, bytes]) -> None:
        self._metadata = metadata
        self._files = files
        self.urls: list[str] = []

    def __call__(self, url: str) -> bytes:
        self.urls.append(url)
        if "/api/models/" in url:
            return _json_bytes(self._metadata)
        marker = f"/resolve/{REVISION}/"
        if marker not in url:
            raise AssertionError(f"unexpected URL: {url}")
        path = url.split(marker, maxsplit=1)[1]
        if path.endswith(".safetensors"):
            raise AssertionError("the importer attempted to download a weight payload")
        return self._files[path]


def _build(fetcher: OfflineFetcher) -> dict[str, Any]:
    return build_model_manifest_draft(
        REPO_ID,
        REVISION,
        model_card_uploaded_dtype="bfloat16",
        oracle_dtype="float16",
        variant="base",
        fetcher=fetcher,
        resolved_at="2026-09-17T00:00:00Z",
        declared_total_parameters="7B",
        declared_active_parameters="1.3B",
    )


def test_imports_pinned_olmoe_metadata_without_downloading_weights() -> None:
    metadata, files = _olmoe_fixture()
    fetcher = OfflineFetcher(metadata, files)

    manifest = _build(fetcher)

    validate_document(manifest)
    assert manifest["status"] == "draft"
    assert manifest["architecture"] == {
        "class_name": "OlmoeForCausalLM",
        "model_type": "olmoe",
        "num_layers": 16,
        "hidden_size": 2048,
        "expert_intermediate_size": 1024,
        "num_experts": 64,
        "top_k": 8,
        "max_position_embeddings": 4096,
        "normalize_top_k_probability": False,
        "declared_total_parameters": "7B",
        "declared_active_parameters": "1.3B",
    }
    assert manifest["dtypes"] == {
        "config_declared": "float32",
        "uploaded_weights": "float32",
        "uploaded_weights_evidence": (
            "Hugging Face API safetensors.parameters at pinned revision"
        ),
        "api_reported_parameter_counts": {"F32": 6_919_161_856},
        "model_card_uploaded_weights_claim": "bfloat16",
        "evidence_conflict": True,
        "oracle_compute": "float16",
        "local_tensor_scan": "pending_download",
    }
    assert manifest["config"] == {
        "path": "config.json",
        "size_bytes": len(files["config.json"]),
        "sha256": hashlib.sha256(files["config.json"]).hexdigest(),
    }
    assert [artifact["path"] for artifact in manifest["weights"]["artifacts"]] == [
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
        "model.safetensors.index.json",
    ]
    assert manifest["weights"]["artifacts"][0] == {
        "path": "model-00001-of-00002.safetensors",
        "size_bytes": 100,
        "sha256": "a" * 64,
    }
    assert not any(
        url.split("?", maxsplit=1)[0].endswith(".safetensors") for url in fetcher.urls
    )


@pytest.mark.parametrize("revision", ["main", "v1.0", "A" * 40, "0" * 39, "0" * 41])
def test_rejects_floating_or_noncanonical_revision_before_fetch(revision: str) -> None:
    def unexpected_fetch(_url: str) -> bytes:
        raise AssertionError("invalid revision must be rejected before network access")

    with pytest.raises(HuggingFaceImportError, match="immutable 40-character"):
        build_model_manifest_draft(
            REPO_ID,
            revision,
            model_card_uploaded_dtype="bfloat16",
            oracle_dtype="bfloat16",
            fetcher=unexpected_fetch,
        )


def test_rejects_weight_without_lfs_sha256() -> None:
    metadata, files = _olmoe_fixture()
    first_weight = next(
        sibling
        for sibling in metadata["siblings"]
        if sibling["rfilename"] == "model-00001-of-00002.safetensors"
    )
    del first_weight["lfs"]["sha256"]
    fetcher = OfflineFetcher(metadata, files)

    with pytest.raises(HuggingFaceImportError, match="missing a valid LFS SHA-256"):
        _build(fetcher)


def test_accepts_injected_api_metadata_and_fetches_only_control_files() -> None:
    metadata, files = _olmoe_fixture()
    fetcher = OfflineFetcher(metadata, files)

    manifest = build_model_manifest_draft(
        REPO_ID,
        REVISION,
        model_card_uploaded_dtype="bfloat16",
        oracle_dtype="bfloat16",
        api_metadata=metadata,
        fetcher=fetcher,
        resolved_at="2026-09-17T00:00:00Z",
    )

    validate_document(manifest)
    assert fetcher.urls
    assert all("/api/models/" not in url for url in fetcher.urls)


def test_missing_api_dtype_fails_closed_instead_of_using_model_card_claim() -> None:
    metadata, files = _olmoe_fixture()
    del metadata["safetensors"]
    fetcher = OfflineFetcher(metadata, files)

    with pytest.raises(HuggingFaceImportError, match="missing safetensors parameter dtype/counts"):
        _build(fetcher)
