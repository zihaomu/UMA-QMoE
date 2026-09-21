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

# These edges predate the experiment framework.  They are frozen rather than
# expanded; F3 removes them only after their runners have real replacements.
LEGACY_RUNNER_IMPORTS = frozenset(
    {
        ("run_activation_aware_mixed_policy", "run_mixed_precision_policy_search"),
        ("run_activation_aware_mixed_policy", "run_mixed_precision_sensitivity"),
        ("run_compressed_olmoe_baseline", "pytorch_reference_host"),
        ("run_compressed_qwen_baseline", "pytorch_reference_host"),
        ("run_layer_precision_search", "run_mixed_precision_policy_search"),
        ("run_layer_precision_search", "run_mixed_precision_sensitivity"),
        ("run_layer_precision_search", "run_quantization_compensation_search"),
        ("run_mixed_precision_policy_search", "run_mixed_precision_refinement"),
        ("run_mixed_precision_policy_search", "run_mixed_precision_sensitivity"),
        ("run_mixed_precision_refinement", "run_mixed_precision_sensitivity"),
        ("run_quantization_compensation_search", "run_mixed_precision_policy_search"),
        ("run_quantization_compensation_search", "run_mixed_precision_sensitivity"),
        ("run_reverse_layer_quantization_search", "run_layer_precision_search"),
        ("run_reverse_layer_quantization_search", "run_mixed_precision_policy_search"),
        ("run_reverse_layer_quantization_search", "run_mixed_precision_sensitivity"),
        ("run_reverse_layer_quantization_search", "run_quantization_compensation_search"),
        ("run_route_coverage_policy_search", "run_mixed_precision_policy_search"),
        ("run_route_coverage_policy_search", "run_mixed_precision_refinement"),
        ("run_route_coverage_policy_search", "run_mixed_precision_sensitivity"),
        ("run_router_logit_compensation_search", "run_mixed_precision_policy_search"),
        ("run_router_logit_compensation_search", "run_mixed_precision_sensitivity"),
        ("validate_olmoe_target_pack_host", "run_mixed_precision_policy_search"),
        ("validate_olmoe_target_pack_host", "run_mixed_precision_sensitivity"),
    }
)


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


def _local_import_edges(directory: Path, project_root: Path) -> list[tuple[str, str, int]]:
    files = sorted(directory.glob("*.py"))
    local_modules = {path.stem for path in files}
    edges: list[tuple[str, str, int]] = []
    for path in files:
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue  # the general Python check reports parse failures
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name.split(".", 1)[0] for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = (
                    [node.module.split(".", 1)[0]]
                    if node.module
                    else [alias.name.split(".", 1)[0] for alias in node.names]
                )
            else:
                continue
            edges.extend(
                (path.stem, name, node.lineno)
                for name in names
                if name in local_modules and name != path.stem
            )
    return edges


def _experiment_boundary_violations(root: Path) -> list[str]:
    violations: list[str] = []
    for relative in ("benchmarks/runners", "benchmarks/experiments"):
        directory = root / relative
        if not directory.is_dir():
            continue
        for source, target, line_number in _local_import_edges(directory, root):
            edge = (source, target)
            if relative == "benchmarks/runners" and edge in LEGACY_RUNNER_IMPORTS:
                continue
            violations.append(
                f"{relative}/{source}.py:{line_number}: forbidden sibling import {target}"
            )

    experiment_root = root / "src" / "uma_qmoe" / "experiments"
    no_torch = {"types.py", "ledger.py", "search.py"}
    forbidden_quantizer_modules = {"cli", "contracts", "target_pack"}
    for path in sorted(experiment_root.glob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError):
            continue
        for node in ast.walk(tree):
            modules: list[str]
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = ([node.module] if node.module else []) + [
                    alias.name for alias in node.names
                ]
            else:
                continue
            for module in modules:
                root_name = module.split(".", 1)[0]
                leaf_name = module.rsplit(".", 1)[-1]
                relative_path = path.relative_to(root)
                if path.name in no_torch and root_name == "torch":
                    violations.append(
                        f"{relative_path}:{node.lineno}: control-plane module imports torch"
                    )
                if path.name == "quantizers.py" and (
                    root_name == "benchmarks" or leaf_name in forbidden_quantizer_modules
                ):
                    violations.append(
                        f"{relative_path}:{node.lineno}: quantizer crosses framework boundary"
                    )
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
    violations.extend(_experiment_boundary_violations(root))
    return violations


__all__ = ["LEGACY_RUNNER_IMPORTS", "check_dependency_boundaries"]
