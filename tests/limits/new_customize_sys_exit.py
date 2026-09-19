# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""NEW (not in ALLO_SHORTCOMINGS.md): any frontend error inside
`allo.customize` (type inference or IR building) prints a traceback and calls
`sys.exit(1)`. A library that exits the interpreter cannot be handled by
`except Exception`: a sweep, a notebook, or a test harness that expects to
catch a build failure and move on is terminated instead."""
import _worktree
from _worktree import verdict

ITEM = "new-sys-exit"
import allo
from allo.ir.types import int32


def bad(a: int32[4]) -> int32[4]:
    b: int32[4]
    for i in range(4):
        b[i] = undefined_name[i]  # noqa: F821
    return b


def main():
    try:
        allo.customize(bad)
        outcome = "built?!"
    except Exception as e:  # noqa: BLE001  -- what a caller would write
        outcome = f"catchable {type(e).__name__}"
    except SystemExit as e:
        outcome = f"SystemExit({e.code}) -- escapes `except Exception`"
    print(outcome)
    verdict(ITEM, outcome.startswith("SystemExit"), outcome)


if __name__ == "__main__":
    _worktree.run_guarded(ITEM, main)
