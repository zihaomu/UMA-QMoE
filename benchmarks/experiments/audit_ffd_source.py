#!/usr/bin/env python3
"""Freeze upstream FFD identity and bind it to the local implementation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from uma_qmoe.ffd import backend_source_sha256


OFFICIAL_REPOSITORY = "https://github.com/qluoluo/faster-flash-decoding"
OFFICIAL_REVISION = "ca09458ab1536328e5f46502b27891412733cd6d"
PAPER = "https://arxiv.org/abs/2609.00097v1"


def _run(*args: str, cwd: Path) -> str:
    result = subprocess.run(
        args,
        cwd=cwd,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _dirty_hash(root: Path) -> tuple[str, str]:
    status = _run("git", "status", "--porcelain=v1", "--untracked-files=all", cwd=root)
    diff = subprocess.run(
        ["git", "diff", "--binary", "HEAD"],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    digest = hashlib.sha256(status.encode("utf-8") + b"\0" + diff)
    untracked = _run(
        "git", "ls-files", "--others", "--exclude-standard", "-z", cwd=root
    )
    for relative in sorted(item for item in untracked.split("\0") if item):
        path = root / relative
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return status, digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-checkout", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    checkout = args.official_checkout.resolve()
    revision = _run("git", "rev-parse", "HEAD", cwd=checkout)
    if revision != OFFICIAL_REVISION:
        raise RuntimeError(
            f"official checkout is {revision}, expected {OFFICIAL_REVISION}"
        )
    license_path = checkout / "LICENSE"
    if "Apache License" not in license_path.read_text(encoding="utf-8"):
        raise RuntimeError(
            "official checkout does not contain the expected Apache license"
        )
    status, dirty_hash = _dirty_hash(root)
    upstream_files = {}
    for relative in (
        "LICENSE",
        "pyproject.toml",
        "ffd_core/modeling/quantized_cache.py",
        "ffd_core/kernels/paged_decode_kernel.py",
    ):
        path = checkout / relative
        upstream_files[relative] = _sha256(path)
    evidence: dict[str, Any] = {
        "schema_version": 1,
        "kind": "ffd_source_audit",
        "paper": PAPER,
        "official_repository": OFFICIAL_REPOSITORY,
        "official_revision": revision,
        "official_license": "Apache-2.0",
        "official_files_sha256": upstream_files,
        "project_commit": _run("git", "rev-parse", "HEAD", cwd=root),
        "project_dirty": bool(status),
        "project_dirty_state_sha256": dirty_hash,
        "project_backend_source_sha256": backend_source_sha256(),
        "porting_boundary": {
            "official_package_dependency": False,
            "model_forked": False,
            "algorithm_reimplemented_in_project": True,
            "target": "hip_gfx1151",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
