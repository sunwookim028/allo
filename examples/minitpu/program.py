# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The assembler for the MiniTPU model: lay out VMEM, emit a tape, check it.

MiniTPU has no interlocks.  Its README's line is that "a legal schedule is a
correctness argument, and the assembler carries it", and ``board_package/asm.py``
refuses a program it cannot prove legal.  This is the same idea at the model's
granularity: ``assemble()`` emits the tape and ``check()`` refuses the three
rules that the hardware does not enforce and that fail silently when broken.

The rules, from ``~/core/minitpu``:

* **Port C exclusivity** (``vpu.sv:429-431``, ``asm.py:1052-1063``) -- one
  physical VREG read shared by the matrix stream engine and the store path.
  The matrix wins the mux; the store reads the matrix's register instead of its
  own, with no interlock and no fault.
* **Output FIFO capacity** (``mxu.sv:44-47``, ``asm.py:933-955``) -- 64 result
  rows per lane, four per ``vmatpush``.  Overflow drops results; there is no
  replay and nothing stalls.
* **The weight-bank contract** (``mxu.sv:246-262``, ``asm.py:1313-1330``) --
  two pending banks, so a ``vmatload`` refills the bank of the load two before
  it.  Issue it before that tile has drained and it overwrites weights a PE has
  not switched to yet; the tile comes back wrong without a fault.  On the
  machine the window is 75 cycles (``WEIGHT_SWITCH_SPAN``); this model has no
  cycles, so the rule it can check is the ordering one: the bank's previous
  tile must be fully popped.
"""

import numpy as np
import ml_dtypes

from .microarch import (
    DIM,
    SUB,
    NVREG,
    WEIGHT_BANKS,
    FIFO_ENTRIES,
    OP_VLD,
    OP_VST,
    OP_MATLOAD,
    OP_MATPUSH,
    OP_MATPOP,
    OP_VADD,
    OP_VMOV,
    OP_FENCE,
)

bf16 = ml_dtypes.bfloat16

#: ``MXU_OUTPUT_FIFO_DEPTH`` (``vpu_pkg.sv:39``), counted in result ROWS.
OUTPUT_FIFO_ROWS = FIFO_ENTRIES * SUB

# VREG allocation.  ``DIM // SUB`` for the weight tile (a ``vmatload`` reads
# base..base+3), then the token tile, the pop destination and the BF16
# accumulator -- the last three all ``M // SUB`` wide.  ``Layout`` computes the
# bases and checks they fit the 32-entry file.
V_W = 0


class Layout:
    """Where the operands sit in VMEM, in whole 4x16 words."""

    def __init__(self, m, k, n, dim=DIM, sub=SUB):
        assert m % sub == 0, "a vmatpush moves SUB token rows"
        assert k % dim == 0 and n % dim == 0
        self.m, self.k, self.n = m, k, n
        self.dim, self.sub = dim, sub
        self.mw = m // sub  # words per token tile
        self.kt, self.nt = k // dim, n // dim
        self.a0 = 0
        self.b0 = self.a0 + self.kt * self.mw
        self.c0 = self.b0 + self.kt * self.nt * (dim // sub)
        self.n_words = self.c0 + self.nt * self.mw
        # VREG bases, and the budget check that replaces an M <= DIM rule.
        self.v_w = V_W
        self.v_a = self.v_w + dim // sub
        self.v_p = self.v_a + self.mw
        self.v_c = self.v_p + self.mw
        need = self.v_c + self.mw
        assert need <= NVREG, (
            f"M = {m} needs {need} VREGs (weights {dim // sub} + token tile, "
            f"pop destination and accumulator {self.mw} each); the file has "
            f"{NVREG}"
        )

    def a_word(self, kt, j):
        return self.a0 + kt * self.mw + j

    def b_word(self, kt, nt, j):
        return self.b0 + (nt * self.kt + kt) * (self.dim // self.sub) + j

    def c_word(self, nt, j):
        return self.c0 + nt * self.mw + j

    def pack(self, a, b):
        """Lay A, B and a zeroed C into one DRAM image of whole words."""
        a = np.asarray(a, dtype=bf16)
        b = np.asarray(b, dtype=bf16)
        img = np.zeros((self.n_words, self.sub, self.dim), dtype=bf16)
        for kt in range(self.kt):
            for j in range(self.mw):
                r0 = j * self.sub
                img[self.a_word(kt, j)] = a[
                    r0 : r0 + self.sub, kt * self.dim : (kt + 1) * self.dim
                ]
            for nt in range(self.nt):
                for j in range(self.dim // self.sub):
                    r0 = kt * self.dim + j * self.sub
                    img[self.b_word(kt, nt, j)] = b[
                        r0 : r0 + self.sub, nt * self.dim : (nt + 1) * self.dim
                    ]
        return img

    def unpack(self, img):
        """Read C back out of a drained VMEM image."""
        out = np.zeros((self.m, self.n), dtype=bf16)
        for nt in range(self.nt):
            for j in range(self.mw):
                r0 = j * self.sub
                out[r0 : r0 + self.sub, nt * self.dim : (nt + 1) * self.dim] = img[
                    self.c_word(nt, j)
                ]
        return out


def assemble(layout):
    """Emit the tape for one GEMM.

    One weight tile at a time: load it, push the token tile against it, pop the
    results, fold them into the BF16 accumulator.  That last fold is where a
    contraction deeper than 16 loses its accuracy, and it is deliberate -- it
    is what the machine does.
    """
    p = []
    mw = layout.mw
    wv = layout.dim // layout.sub
    V_A, V_P, V_C = layout.v_a, layout.v_p, layout.v_c
    for nt in range(layout.nt):
        for kt in range(layout.kt):
            for j in range(wv):
                p.append((OP_VLD, V_W + j, layout.b_word(kt, nt, j), 0))
            for j in range(mw):
                p.append((OP_VLD, V_A + j, layout.a_word(kt, j), 0))
            p.append((OP_FENCE, wv + mw, 0, 0))
            p.append((OP_MATLOAD, V_W, 0, 0))
            for j in range(mw):
                p.append((OP_MATPUSH, V_A + j, 0, 0))
            for j in range(mw):
                p.append((OP_MATPOP, V_P + j, 0, 0))
            p.append((OP_FENCE, mw, 0, 0))
            for j in range(mw):
                if kt == 0:
                    p.append((OP_VMOV, V_C + j, V_P + j, 0))
                else:
                    p.append((OP_VADD, V_C + j, V_C + j, V_P + j))
            p.append((OP_FENCE, mw, 0, 0))
        for j in range(mw):
            p.append((OP_VST, V_C + j, layout.c_word(nt, j), 0))
    return p


class ProgramError(Exception):
    """A rule the hardware does not enforce, broken."""


def check(prog):
    """Refuse a tape that MiniTPU would run wrong without saying so."""
    # Port C: a store and a matrix stream command may not be in flight at once.
    # At this model's granularity "in flight" is "between the command and the
    # next fence", so the check is that no vst sits inside a matrix burst.
    streaming = False
    for op, a, b, c in prog:
        if op in (OP_MATLOAD, OP_MATPUSH):
            streaming = True
        elif op == OP_FENCE:
            streaming = False
        elif op == OP_VST and streaming:
            raise ProgramError(
                "vst inside a matrix burst: both read VREG port C, the matrix "
                "wins the mux, and the store silently reads the matrix's "
                "register (vpu.sv:429-431)"
            )

    # Output FIFO: four result rows per push, 64 rows per lane, no replay.
    rows = 0
    for op, a, b, c in prog:
        if op == OP_MATPUSH:
            rows += SUB
            if rows > OUTPUT_FIFO_ROWS:
                raise ProgramError(
                    f"{rows} result rows outstanding in a lane's output FIFO, "
                    f"capacity {OUTPUT_FIFO_ROWS}: overflow drops results and "
                    "nothing stalls (mxu.sv:44-47)"
                )
        elif op == OP_MATPOP:
            rows -= SUB

    # Weight banks: a vmatload refills the bank of the load two before it.
    bank = 0
    tiles = []  # (bank, pushes, pops) per tile, oldest first
    for op, a, b, c in prog:
        if op == OP_MATLOAD:
            for t in tiles:
                if t[0] == bank and t[1] != t[2]:
                    raise ProgramError(
                        "vmatload refills a weight bank whose previous tile has "
                        f"{t[1] - t[2]} pushes still undrained: it overwrites "
                        "weights a PE has not switched to and the tile comes "
                        "back wrong without a fault (mxu.sv:246-262)"
                    )
            tiles.append([bank, 0, 0])
            bank = (bank + 1) % WEIGHT_BANKS
        elif op == OP_MATPUSH:
            tiles[-1][1] += 1
        elif op == OP_MATPOP:
            for t in tiles:
                if t[2] < t[1]:
                    t[2] += 1
                    break

    # Every vmatload reads base..base+3, so the base must not wrap.
    for op, a, b, c in prog:
        if op == OP_MATLOAD and a > NVREG - DIM // SUB:
            raise ProgramError(f"vmatload base {a} wraps the VREG file")
    return True
