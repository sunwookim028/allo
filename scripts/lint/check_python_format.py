#!/usr/bin/env python3
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
from pathlib import Path


def check_black_formatting(dir: str | Path) -> bool:
    """Run ``black --check`` on the given directory."""
    target = Path(dir).resolve()
    if not target.is_dir():
        print(f"Error: {target} is not a directory.", file=sys.stderr)
        sys.exit(1)

    try:
        cmd = [sys.executable, "-m", "black", "--check", "--diff", str(target)]
        result = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        print(
            "Error: 'black' is not installed. Please install it with 'pip install black'.",
            file=sys.stderr,
        )
        sys.exit(1)

    if result.returncode == 0:
        print(f"✅ All files in {target} are properly formatted.")
        return True
    elif result.returncode == 1:
        print(f"❌ The following files in {target} should be reformatted:\n")
        if result.stderr:
            print(result.stderr, end="")  # Print stderr if available
        if result.stdout:
            print(result.stdout, end="")  # Print stdout if available
        return False
    else:
        print(
            f"Error: 'black' encountered an unexpected error:\n{result.stderr}",
            file=sys.stderr,
        )
        print(result.stderr, file=sys.stderr)
        sys.exit(1)


SOURCE_PATHS = ["allo", "tests", "examples", "scripts"]

if __name__ == "__main__":
    ok = True
    for path in SOURCE_PATHS:
        if not check_black_formatting(path):
            ok = False

    sys.exit(0 if ok else 1)
