#!/usr/bin/env python3
"""Fail CI when the core runtime imports an external serving runtime."""

from __future__ import annotations

from pathlib import Path
import sys

from uma_qmoe.dependency_boundary import check_dependency_boundaries


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    violations = check_dependency_boundaries(root)
    if violations:
        print("UMA-QMoE dependency boundary violations:", file=sys.stderr)
        for violation in violations:
            print(f"- {violation}", file=sys.stderr)
        return 1
    print("UMA-QMoE dependency boundaries: passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
