"""Pinned, resumable Hugging Face model acquisition and local verification."""

from __future__ import annotations

import hashlib
import http.client
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import canonical_sha256


DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024
DEFAULT_TIMEOUT_SECONDS = 60


class ModelStoreError(ValueError):
    """Raised when a pinned model file cannot be acquired or verified."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def _safe_relative_path(path: object) -> str:
    if not isinstance(path, str) or not path:
        raise ModelStoreError("model artifact path must be a non-empty string")
    segments = path.split("/")
    if (
        path.startswith("/")
        or "\\" in path
        or any(segment in {"", ".", ".."} for segment in segments)
    ):
        raise ModelStoreError(f"unsafe model artifact path {path!r}")
    return path


def manifest_file_identities(manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return every locally required file identity in deterministic order."""

    if manifest.get("kind") != "model_manifest":
        raise ModelStoreError("document kind must be 'model_manifest'")
    try:
        raw_identities = [manifest["config"]]
        raw_identities.extend(manifest["tokenizer"].get("files", []))
        raw_identities.extend(manifest["weights"]["artifacts"])
    except (KeyError, TypeError, AttributeError) as exc:
        raise ModelStoreError("ModelManifest file identities are incomplete") from exc

    identities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_identity in raw_identities:
        if not isinstance(raw_identity, Mapping):
            raise ModelStoreError("ModelManifest file identity must be an object")
        path = _safe_relative_path(raw_identity.get("path"))
        if path in seen:
            raise ModelStoreError(f"duplicate model artifact path {path!r}")
        seen.add(path)
        size = raw_identity.get("size_bytes")
        sha256 = raw_identity.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ModelStoreError(f"artifact {path!r} has invalid size_bytes")
        if not (
            isinstance(sha256, str)
            and len(sha256) == 64
            and all(character in "0123456789abcdef" for character in sha256)
        ):
            raise ModelStoreError(f"artifact {path!r} has invalid SHA-256")
        identities.append({"path": path, "size_bytes": size, "sha256": sha256})
    if not identities:
        raise ModelStoreError("ModelManifest contains no file identities")
    return identities


def derivation_file_identities(
    derivation: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Return every file pinned by a ModelDerivation."""

    if derivation.get("kind") != "model_derivation":
        raise ModelStoreError("document kind must be 'model_derivation'")
    try:
        raw_identities = [derivation["config"]]
        raw_identities.extend(derivation["tokenizer_files"])
        raw_identities.extend(derivation["weights"]["artifacts"])
    except (KeyError, TypeError, AttributeError) as exc:
        raise ModelStoreError("ModelDerivation file identities are incomplete") from exc

    identities: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_identity in raw_identities:
        if not isinstance(raw_identity, Mapping):
            raise ModelStoreError("ModelDerivation file identity must be an object")
        path = _safe_relative_path(raw_identity.get("path"))
        if path in seen:
            raise ModelStoreError(f"duplicate derived artifact path {path!r}")
        seen.add(path)
        size = raw_identity.get("size_bytes")
        sha256 = raw_identity.get("sha256")
        if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
            raise ModelStoreError(f"derived artifact {path!r} has invalid size_bytes")
        if not (
            isinstance(sha256, str)
            and len(sha256) == 64
            and all(character in "0123456789abcdef" for character in sha256)
        ):
            raise ModelStoreError(f"derived artifact {path!r} has invalid SHA-256")
        identities.append({"path": path, "size_bytes": size, "sha256": sha256})
    if not identities:
        raise ModelStoreError("ModelDerivation contains no file identities")
    return identities


def _file_sha256(path: Path, chunk_size: int) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(chunk_size), b""):
                digest.update(chunk)
    except OSError as exc:
        raise ModelStoreError(f"cannot hash {path}: {exc}") from exc
    return digest.hexdigest()


def _artifact_path(root: Path, relative_path: str) -> Path:
    """Resolve a manifest path while rejecting symlink-parent escapes."""

    path = root.joinpath(*relative_path.split("/"))
    try:
        path.resolve(strict=False).relative_to(root.resolve())
    except ValueError as exc:
        raise ModelStoreError(
            f"model artifact path escapes through a symlink: {relative_path!r}"
        ) from exc
    return path


def _verify_file(
    root: Path, identity: Mapping[str, Any], *, chunk_size: int
) -> dict[str, Any]:
    relative_path = identity["path"]
    path = _artifact_path(root, relative_path)
    if path.is_symlink() or not path.is_file():
        raise ModelStoreError(f"missing regular model artifact {relative_path!r}")
    try:
        observed_size = path.stat().st_size
    except OSError as exc:
        raise ModelStoreError(f"cannot stat model artifact {relative_path!r}: {exc}") from exc
    if observed_size != identity["size_bytes"]:
        raise ModelStoreError(
            f"size mismatch for model artifact {relative_path!r}: expected "
            f"{identity['size_bytes']}, observed {observed_size}"
        )
    observed_sha256 = _file_sha256(path, chunk_size)
    if observed_sha256 != identity["sha256"]:
        raise ModelStoreError(
            f"SHA-256 mismatch for model artifact {relative_path!r}: expected "
            f"{identity['sha256']}, observed {observed_sha256}"
        )
    return {
        "path": relative_path,
        "size_bytes": observed_size,
        "sha256": observed_sha256,
    }


def _acquisition_document(
    manifest: Mapping[str, Any], root: Path, files: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "kind": "model_acquisition",
        "completed_at": _utc_now(),
        "model_manifest_sha256": canonical_sha256(manifest),
        "model_directory": str(root.resolve()),
        "file_count": len(files),
        "total_bytes": sum(file["size_bytes"] for file in files),
        "files": files,
    }


def verify_model_files(
    manifest: Mapping[str, Any],
    model_directory: str | os.PathLike[str],
    *,
    chunk_size: int = DEFAULT_CHUNK_BYTES,
) -> dict[str, Any]:
    """Verify all config, tokenizer, index, and weight files in a manifest."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ModelStoreError("chunk_size must be a positive integer")
    root = Path(model_directory)
    if not root.is_dir():
        raise ModelStoreError(f"model directory does not exist: {root}")
    identities = manifest_file_identities(manifest)
    files = [_verify_file(root, identity, chunk_size=chunk_size) for identity in identities]
    return _acquisition_document(manifest, root, files)


def verify_derivation_files(
    derivation: Mapping[str, Any],
    artifact_directory: str | os.PathLike[str],
    *,
    target_id: str,
    source_contract_path: str,
    source_contract_sha256: str,
    chunk_size: int = DEFAULT_CHUNK_BYTES,
) -> dict[str, Any]:
    """Verify a complete derived artifact replica and emit target evidence."""

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ModelStoreError("chunk_size must be a positive integer")
    if not target_id:
        raise ModelStoreError("target_id must be non-empty")
    root = Path(artifact_directory)
    if not root.is_dir():
        raise ModelStoreError(f"artifact directory does not exist: {root}")
    identities = derivation_file_identities(derivation)
    files = [_verify_file(root, identity, chunk_size=chunk_size) for identity in identities]
    return {
        "schema_version": 1,
        "kind": "artifact_verification",
        "verified_at": _utc_now(),
        "target_id": target_id,
        "source_contract_kind": "model_derivation",
        "source_contract_path": _safe_relative_path(source_contract_path),
        "source_contract_sha256": source_contract_sha256,
        "artifact_directory": str(root.resolve()),
        "file_count": len(files),
        "total_bytes": sum(file["size_bytes"] for file in files),
        "files": files,
    }


def _artifact_url(manifest: Mapping[str, Any], relative_path: str) -> str:
    try:
        repository = manifest["provenance"]["repository"]
        revision = manifest["model_revision"]
    except (KeyError, TypeError) as exc:
        raise ModelStoreError("ModelManifest Hugging Face provenance is incomplete") from exc
    if manifest.get("provenance", {}).get("provider") != "huggingface":
        raise ModelStoreError("only pinned Hugging Face ModelManifests can be fetched")
    if not isinstance(repository, str) or not repository:
        raise ModelStoreError("Hugging Face repository must be non-empty")
    if not isinstance(revision, str) or len(revision) != 40:
        raise ModelStoreError("Hugging Face revision must be a pinned 40-character commit")
    quoted_repository = urllib.parse.quote(repository, safe="/")
    quoted_revision = urllib.parse.quote(revision, safe="")
    quoted_path = urllib.parse.quote(relative_path, safe="/")
    return (
        f"https://huggingface.co/{quoted_repository}/resolve/"
        f"{quoted_revision}/{quoted_path}"
    )


def _transfer_once(
    url: str,
    partial: Path,
    *,
    expected_size: int,
    chunk_size: int,
    timeout_seconds: int,
) -> None:
    start = partial.stat().st_size if partial.exists() else 0
    headers = {
        "Accept-Encoding": "identity",
        "User-Agent": "uma-qmoe-model-store/0.1",
    }
    if start:
        headers["Range"] = f"bytes={start}-"
    request = urllib.request.Request(url, headers=headers)
    try:
        response = urllib.request.urlopen(request, timeout=timeout_seconds)
    except urllib.error.HTTPError as exc:
        if exc.code == 416 and start == expected_size:
            return
        raise

    with response:
        status = response.getcode()
        if start and status == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith(f"bytes {start}-"):
                raise ModelStoreError(
                    f"resume response has unexpected Content-Range {content_range!r}"
                )
            mode = "ab"
        elif status == 200:
            mode = "wb"
            start = 0
        else:
            raise ModelStoreError(f"unexpected HTTP status {status} for {url}")

        try:
            with partial.open(mode) as stream:
                written = start
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    stream.write(chunk)
                    written += len(chunk)
                    if written > expected_size:
                        raise ModelStoreError(
                            f"download exceeded expected size {expected_size} for {url}"
                        )
                stream.flush()
                os.fsync(stream.fileno())
        except (OSError, http.client.HTTPException) as exc:
            raise ModelStoreError(
                f"cannot stream partial model artifact {partial}: {exc}"
            ) from exc


def _fetch_one(
    manifest: Mapping[str, Any],
    root: Path,
    identity: Mapping[str, Any],
    *,
    chunk_size: int,
    timeout_seconds: int,
    retries: int,
    progress: Callable[[str], None] | None,
) -> dict[str, Any]:
    relative_path = identity["path"]
    final = _artifact_path(root, relative_path)
    partial = final.with_name(final.name + ".part")
    final.parent.mkdir(parents=True, exist_ok=True)
    # A concurrent filesystem change could have introduced a symlink while
    # parents were created, so enforce the containment boundary again.
    final = _artifact_path(root, relative_path)
    partial = final.with_name(final.name + ".part")
    if final.is_symlink() or partial.is_symlink():
        raise ModelStoreError(f"refusing symlink model artifact {relative_path!r}")
    if final.exists():
        verified = _verify_file(root, identity, chunk_size=chunk_size)
        if progress is not None:
            progress(f"verified-existing {relative_path}")
        return verified
    if partial.exists() and (
        not partial.is_file() or partial.stat().st_size > identity["size_bytes"]
    ):
        raise ModelStoreError(f"invalid partial model artifact {partial}")

    url = _artifact_url(manifest, relative_path)
    if progress is not None:
        progress(f"downloading {relative_path}")
    last_error: BaseException | None = None
    for attempt in range(retries + 1):
        try:
            _transfer_once(
                url,
                partial,
                expected_size=identity["size_bytes"],
                chunk_size=chunk_size,
                timeout_seconds=timeout_seconds,
            )
            if partial.stat().st_size == identity["size_bytes"]:
                break
            raise ModelStoreError(
                f"incomplete model artifact {relative_path!r}: observed "
                f"{partial.stat().st_size} of {identity['size_bytes']} bytes"
            )
        except (OSError, urllib.error.URLError, ModelStoreError) as exc:
            last_error = exc
            if attempt == retries:
                raise ModelStoreError(
                    f"failed to download {relative_path!r} after {retries + 1} attempts: {exc}"
                ) from exc
            time.sleep(min(2**attempt, 10))
    else:  # pragma: no cover - loop either breaks or raises
        raise ModelStoreError(f"failed to download {relative_path!r}: {last_error}")

    observed_sha256 = _file_sha256(partial, chunk_size)
    if observed_sha256 != identity["sha256"]:
        raise ModelStoreError(
            f"SHA-256 mismatch for downloaded artifact {relative_path!r}: expected "
            f"{identity['sha256']}, observed {observed_sha256}"
        )
    try:
        os.replace(partial, final)
    except OSError as exc:
        raise ModelStoreError(f"cannot finalize model artifact {relative_path!r}: {exc}") from exc
    if progress is not None:
        progress(f"verified {relative_path}")
    return {
        "path": relative_path,
        "size_bytes": identity["size_bytes"],
        "sha256": observed_sha256,
    }


def fetch_model_files(
    manifest: Mapping[str, Any],
    model_directory: str | os.PathLike[str],
    *,
    jobs: int = 1,
    chunk_size: int = DEFAULT_CHUNK_BYTES,
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS,
    retries: int = 5,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Fetch every pinned file with resume support, then return acquisition evidence."""

    for value, name, maximum in (
        (jobs, "jobs", 8),
        (chunk_size, "chunk_size", None),
        (timeout_seconds, "timeout_seconds", None),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ModelStoreError(f"{name} must be a positive integer")
        if maximum is not None and value > maximum:
            raise ModelStoreError(f"{name} must not exceed {maximum}")
    if isinstance(retries, bool) or not isinstance(retries, int) or retries < 0:
        raise ModelStoreError("retries must be a non-negative integer")

    root = Path(model_directory)
    root.mkdir(parents=True, exist_ok=True)
    identities = manifest_file_identities(manifest)

    def fetch(identity: Mapping[str, Any]) -> dict[str, Any]:
        return _fetch_one(
            manifest,
            root,
            identity,
            chunk_size=chunk_size,
            timeout_seconds=timeout_seconds,
            retries=retries,
            progress=progress,
        )

    if jobs == 1:
        files = [fetch(identity) for identity in identities]
    else:
        with ThreadPoolExecutor(max_workers=jobs, thread_name_prefix="umaq-fetch") as executor:
            files = list(executor.map(fetch, identities))
    return _acquisition_document(manifest, root, files)


__all__ = [
    "ModelStoreError",
    "derivation_file_identities",
    "fetch_model_files",
    "manifest_file_identities",
    "verify_derivation_files",
    "verify_model_files",
]
