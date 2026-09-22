# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item P4: a bit-slice `x[0:NAME]` whose bound is a GLOBAL integer name
silently becomes UInt(32).

Original claim (branch `tinytpu-align`, increment 2): `infer.py`
`visit_Subscript` infers a slice's width from `upper - lower` with every
`ast.Name` turned into a free sympy symbol (`visit_symbol`), never resolved
against `ctx.global_vars`. `0 : VW` is therefore the symbol `VW`, not an
integer, so the inferer warns "Cannot infer the bitwidth of the slice, use
UInt(32) as default" and the assignment writes only 32 bits. It was right at
T=4 (VW = 32) and wrong at T=8 (VW = 64).

This repro is deliberately arithmetic, not a warning check: the same slice is
written twice, once with the global name as the bound and once with the
literal it equals. If the widths agree the two results agree.
"""
import traceback
import warnings

import numpy as np
import _worktree
from _worktree import verdict

ITEM = "P4"
import allo  # noqa: E402
from allo.ir.types import UInt  # noqa: E402

VW = 64  # a global int, exactly as microarch_isa.py's VW is


def main():
    def kernel(src: UInt(64)[1], by_name: UInt(64)[1], by_literal: UInt(64)[1]):
        a: UInt(64) = 0
        a[0:VW] = src[0]
        by_name[0] = a
        b: UInt(64) = 0
        b[0:64] = src[0]
        by_literal[0] = b

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            s = allo.customize(kernel)
            mod = s.build()
        warned = [
            str(w.message) for w in caught if "bitwidth of the slice" in str(w.message)
        ]
        src = np.array([0xDEADBEEFCAFEF00D], dtype=np.uint64)
        name = np.zeros(1, dtype=np.uint64)
        lit = np.zeros(1, dtype=np.uint64)
        mod(src, name, lit)
        bad = int(name[0]) != int(src[0])
        verdict(
            ITEM,
            bad,
            f"x[0:VW] (VW={VW}) wrote 0x{int(name[0]):016x}, x[0:64] wrote "
            f"0x{int(lit[0]):016x}, source 0x{int(src[0]):016x}"
            + (f"; warning: {warned[0]!r}" if warned else "; no warning was raised"),
        )
    except (Exception, SystemExit) as e:  # noqa: BLE001
        traceback.print_exc()
        verdict(ITEM, True, f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    _worktree.run_guarded(ITEM, main)
