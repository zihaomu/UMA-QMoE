"""Static dependency-boundary checks for the UMA-QMoE core runtime."""

from __future__ import annotations

import ast
from pathlib import Path
import re
from typing import Iterable


CORE_DIRECTORIES = ("src", "bindings", "kernels")
EXTERNAL_DIRECTORIES = ("benchmarks/external",)
FORBIDDEN_PYTHON_ROOTS = frozenset({"vllm"})
FORBIDDEN_EXTERNAL_ROOTS = frozenset({"uma_qmoe"})
NATIVE_SUFFIXES = frozenset({".c", ".cc", ".cpp", ".cu", ".cuh", ".h", ".hpp", ".hip"})
_NATIVE_REFERENCE = re.compile(r"(?i)(?:#\s*include\s*[<\"]|\b(?:link|library)\b).*vllm")
_DEPENDENCY = re.compile(r"(?im)^\s*[\"']?vllm(?:\[[^]]+\])?\s*(?:[<>=!~]|[\"'])")


def _python_violations(
    path: Path,
    project_root: Path,
    *,
    forbidden_roots: frozenset[str] = FORBIDDEN_PYTHON_ROOTS,
) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except (OSError, SyntaxError) as exc:
        return [f"{path.relative_to(project_root)}: cannot parse: {exc}"]
    violations: list[str] = []
    for node in ast.walk(tree):
        names: Iterable[str]
        if isinstance(node, ast.Import):
            names = (alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names = (node.module,)
        else:
            continue
        for name in names:
            if name.split(".", 1)[0].lower() in forbidden_roots:
                relative = path.relative_to(project_root)
                violations.append(f"{relative}:{node.lineno}: forbidden import {name}")
    return violations


def check_dependency_boundaries(project_root: str | Path) -> list[str]:
    """Return deterministic violations; an empty list means the boundary holds."""

    root = Path(project_root).resolve()
    violations: list[str] = []
    for directory_name in CORE_DIRECTORIES:
        directory = root / directory_name
        if not directory.is_dir():
            continue
        for path in sorted(item for item in directory.rglob("*") if item.is_file()):
            if path.suffix == ".py":
                violations.extend(_python_violations(path, root))
            elif path.suffix.lower() in NATIVE_SUFFIXES:
                try:
                    text = path.read_text(encoding="utf-8", errors="replace")
                except OSError as exc:
                    violations.append(f"{path.relative_to(root)}: cannot read: {exc}")
                    continue
                for line_number, line in enumerate(text.splitlines(), start=1):
                    if _NATIVE_REFERENCE.search(line):
                        violations.append(
                            f"{path.relative_to(root)}:{line_number}: forbidden native reference"
                        )
    for directory_name in EXTERNAL_DIRECTORIES:
        directory = root / directory_name
        if not directory.is_dir():
            continue
        for path in sorted(directory.rglob("*.py")):
            violations.extend(
                _python_violations(
                    path,
                    root,
                    forbidden_roots=FORBIDDEN_EXTERNAL_ROOTS,
                )
            )
    pyproject = root / "pyproject.toml"
    if pyproject.is_file():
        text = pyproject.read_text(encoding="utf-8")
        in_project_dependencies = False
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped.startswith("["):
                in_project_dependencies = stripped == "[project]"
            if in_project_dependencies and _DEPENDENCY.search(line):
                violations.append(
                    f"pyproject.toml:{line_number}: forbidden core dependency vllm"
                )
    return violations


__all__ = ["check_dependency_boundaries"]
