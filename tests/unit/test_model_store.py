from __future__ import annotations

import hashlib
import http.client
import io
from pathlib import Path
from typing import Any

import pytest

from uma_qmoe.contracts import canonical_sha256, validate_document
from uma_qmoe.model_store import (
    ModelStoreError,
    derivation_file_identities,
    fetch_model_files,
    manifest_file_identities,
    verify_derivation_files,
    verify_model_files,
)


def _identity(path: str, payload: bytes) -> dict[str, Any]:
    return {
        "path": path,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _manifest(config: bytes, tokenizer: bytes, weights: bytes) -> dict[str, Any]:
    return {
        "kind": "model_manifest",
        "model_revision": "a" * 40,
        "config": _identity("config.json", config),
        "tokenizer": {"files": [_identity("tokenizer.json", tokenizer)]},
        "weights": {
            "artifacts": [_identity("model.safetensors", weights)],
        },
        "provenance": {
            "provider": "huggingface",
            "repository": "example/model",
        },
    }


def test_verifies_every_manifest_file_and_builds_schema_evidence(tmp_path: Path) -> None:
    payloads = {
        "config.json": b"config",
        "tokenizer.json": b"tokenizer",
        "model.safetensors": b"weights",
    }
    manifest = _manifest(
        payloads["config.json"],
        payloads["tokenizer.json"],
        payloads["model.safetensors"],
    )
    for path, payload in payloads.items():
        (tmp_path / path).write_bytes(payload)

    evidence = verify_model_files(manifest, tmp_path, chunk_size=2)

    assert evidence["model_manifest_sha256"] == canonical_sha256(manifest)
    assert evidence["file_count"] == 3
    assert evidence["total_bytes"] == sum(map(len, payloads.values()))
    assert [entry["path"] for entry in evidence["files"]] == list(payloads)
    validate_document(evidence)


def test_verification_rejects_missing_and_tampered_files(tmp_path: Path) -> None:
    manifest = _manifest(b"config", b"tokenizer", b"weights")
    (tmp_path / "config.json").write_bytes(b"config")
    (tmp_path / "tokenizer.json").write_bytes(b"tokenizer")

    with pytest.raises(ModelStoreError, match="missing regular"):
        verify_model_files(manifest, tmp_path)

    (tmp_path / "model.safetensors").write_bytes(b"Weights")
    with pytest.raises(ModelStoreError, match="SHA-256 mismatch"):
        verify_model_files(manifest, tmp_path)


def test_verifies_complete_model_derivation_replica(tmp_path: Path) -> None:
    payloads = {
        "config.json": b"config",
        "tokenizer.json": b"tokenizer",
        "model.safetensors": b"derived-weights",
        "model.safetensors.index.json": b"index",
    }
    derivation = {
        "kind": "model_derivation",
        "config": _identity("config.json", payloads["config.json"]),
        "tokenizer_files": [
            _identity("tokenizer.json", payloads["tokenizer.json"])
        ],
        "weights": {
            "artifacts": [
                _identity("model.safetensors", payloads["model.safetensors"]),
                _identity(
                    "model.safetensors.index.json",
                    payloads["model.safetensors.index.json"],
                ),
            ]
        },
    }
    for name, payload in payloads.items():
        (tmp_path / name).write_bytes(payload)

    assert [
        identity["path"] for identity in derivation_file_identities(derivation)
    ] == list(payloads)
    evidence = verify_derivation_files(
        derivation,
        tmp_path,
        target_id="halo3",
        source_contract_path="models/manifests/derived.yaml",
        source_contract_sha256="a" * 64,
        chunk_size=2,
    )

    assert evidence["file_count"] == 4
    assert evidence["total_bytes"] == sum(map(len, payloads.values()))
    validate_document(evidence)

def test_manifest_file_identities_reject_unsafe_or_duplicate_paths() -> None:
    manifest = _manifest(b"config", b"tokenizer", b"weights")
    manifest["config"]["path"] = "../config.json"
    with pytest.raises(ModelStoreError, match="unsafe"):
        manifest_file_identities(manifest)

    manifest = _manifest(b"config", b"tokenizer", b"weights")
    manifest["tokenizer"]["files"][0]["path"] = "config.json"
    with pytest.raises(ModelStoreError, match="duplicate"):
        manifest_file_identities(manifest)


def test_verification_rejects_symlink_parent_escape(tmp_path: Path) -> None:
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "root"
    root.mkdir()
    (root / "nested").symlink_to(outside, target_is_directory=True)
    manifest = _manifest(b"config", b"tokenizer", b"weights")
    manifest["config"] = _identity("nested/config.json", b"config")

    with pytest.raises(ModelStoreError, match="escapes through a symlink"):
        verify_model_files(manifest, root)

class _FakeResponse:
    def __init__(self, payload: bytes, *, status: int, content_range: str = "") -> None:
        self._stream = io.BytesIO(payload)
        self._status = status
        self.headers = {"Content-Range": content_range} if content_range else {}

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def getcode(self) -> int:
        return self._status

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)


class _InterruptingResponse(_FakeResponse):
    def __init__(self, payload: bytes) -> None:
        super().__init__(payload, status=200)
        self._reads = 0

    def read(self, size: int = -1) -> bytes:
        self._reads += 1
        if self._reads == 2:
            raise http.client.IncompleteRead(b"")
        return super().read(size)


def test_fetch_resumes_partial_file_and_atomically_finalizes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = {
        "config.json": b"config",
        "tokenizer.json": b"tokenizer",
        "model.safetensors": b"weights",
    }
    manifest = _manifest(
        payloads["config.json"],
        payloads["tokenizer.json"],
        payloads["model.safetensors"],
    )
    (tmp_path / "model.safetensors.part").write_bytes(b"wei")
    observed_ranges: dict[str, str | None] = {}

    def fake_urlopen(request: Any, *, timeout: int) -> _FakeResponse:
        assert timeout == 60
        name = request.full_url.rsplit("/", 1)[-1]
        payload = payloads[name]
        requested_range = request.headers.get("Range")
        observed_ranges[name] = requested_range
        if requested_range:
            start = int(requested_range.removeprefix("bytes=").removesuffix("-"))
            return _FakeResponse(
                payload[start:],
                status=206,
                content_range=f"bytes {start}-{len(payload) - 1}/{len(payload)}",
            )
        return _FakeResponse(payload, status=200)

    monkeypatch.setattr("uma_qmoe.model_store.urllib.request.urlopen", fake_urlopen)

    evidence = fetch_model_files(manifest, tmp_path, jobs=1, chunk_size=2)

    assert observed_ranges["model.safetensors"] == "bytes=3-"
    assert not (tmp_path / "model.safetensors.part").exists()
    for path, payload in payloads.items():
        assert (tmp_path / path).read_bytes() == payload
    validate_document(evidence)


def test_fetch_retries_and_resumes_an_interrupted_http_body(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payloads = {
        "config.json": b"config",
        "tokenizer.json": b"tokenizer",
        "model.safetensors": b"weights",
    }
    manifest = _manifest(
        payloads["config.json"],
        payloads["tokenizer.json"],
        payloads["model.safetensors"],
    )
    attempts: dict[str, int] = {}

    def fake_urlopen(request: Any, *, timeout: int) -> _FakeResponse:
        name = request.full_url.rsplit("/", 1)[-1]
        attempts[name] = attempts.get(name, 0) + 1
        payload = payloads[name]
        requested_range = request.headers.get("Range")
        if name == "model.safetensors" and attempts[name] == 1:
            return _InterruptingResponse(payload)
        if requested_range:
            start = int(requested_range.removeprefix("bytes=").removesuffix("-"))
            return _FakeResponse(
                payload[start:],
                status=206,
                content_range=f"bytes {start}-{len(payload) - 1}/{len(payload)}",
            )
        return _FakeResponse(payload, status=200)

    monkeypatch.setattr("uma_qmoe.model_store.urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("uma_qmoe.model_store.time.sleep", lambda _seconds: None)

    fetch_model_files(manifest, tmp_path, jobs=1, chunk_size=2, retries=1)

    assert attempts["model.safetensors"] == 2
    assert (tmp_path / "model.safetensors").read_bytes() == b"weights"
