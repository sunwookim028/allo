#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import difflib
from pathlib import Path

CPP_EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx"}


def clang_format_version() -> tuple[int, ...]:
    result = subprocess.run(
        ["clang-format", "--version"], capture_output=True, text=True
    )
    # "clang-format version 14.0.0 ..."
    for token in result.stdout.split():
        if token[0].isdigit():
            return tuple(int(x) for x in token.split(".")[:2])
    return (0, 0)


def collect_sources(directory: Path) -> list[Path]:
    return [
        p for p in directory.rglob("*") if p.suffix in CPP_EXTENSIONS and p.is_file()
    ]


def check_with_dry_run(files: list[Path]) -> list[str]:
    """LLVM >= 11: --dry-run --Werror exits 1 and emits diagnostics for non-conforming files."""
    bad = []
    for f in files:
        result = subprocess.run(
            ["clang-format", "--dry-run", "--Werror", str(f)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            bad.append(f"  {f}\n{result.stderr.rstrip()}")
    return bad


def check_with_diff(files: list[Path]) -> list[str]:
    """Fallback for older clang-format: format to stdout and diff against original."""
    bad = []
    for f in files:
        original = f.read_text(encoding="utf-8", errors="replace")
        result = subprocess.run(
            ["clang-format", str(f)],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            bad.append(f"  {f}: clang-format error: {result.stderr.rstrip()}")
            continue
        formatted = result.stdout
        if original != formatted:
            diff = difflib.unified_diff(
                original.splitlines(keepends=True),
                formatted.splitlines(keepends=True),
                fromfile=f"a/{f}",
                tofile=f"b/{f}",
                n=3,
            )
            bad.append(f"{''.join(diff).rstrip()}")
    return bad


def check_clang_format(directory: str | Path) -> bool:
    target = Path(directory).resolve()
    if not target.is_dir():
        print(f"Error: {target} is not a valid directory", file=sys.stderr)
        sys.exit(1)

    try:
        version = clang_format_version()
    except FileNotFoundError:
        print("Error: clang-format not found on PATH", file=sys.stderr)
        sys.exit(1)

    files = collect_sources(target)
    if not files:
        print(f"No C/C++ source files found in {target}")
        return True

    use_dry_run = version >= (11, 0)
    checker = check_with_dry_run if use_dry_run else check_with_diff

    violations = checker(files)
    if not violations:
        print(f"✅ All files in {target} are properly formatted.")
        return True

    print(f"✗ {len(violations)} file(s) should be reformatted:\n")
    print("\n".join(violations))
    return False


SOURCE_PATHS = ["mlir"]

if __name__ == "__main__":

    ok = True
    for path in SOURCE_PATHS:
        if not check_clang_format(path):
            ok = False

    sys.exit(0 if ok else 1)
