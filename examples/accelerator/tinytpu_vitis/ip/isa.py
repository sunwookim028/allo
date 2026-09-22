# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The instruction set: one 64-bit control word, one 64-bit address word.

    op [0:6]  f0 [6:18]  f1 [18:30]  f2 [30:42]  f3 [42:54]  nr [54:62]

Every field carries one more bit than its value range needs, and `enc` asserts
it: a bit-slice used to be extracted into a *signed* ``ap_int<N>``, so a field
whose top bit was set read back negative. The spare-bit rule, the bug it cost
and the fix are on ``docs/source/designs/tinytpu_isa.rst``.
"""

OP_NOP = 0
OP_DMA_LD = 1     # f0=src|dst f1=dram_row0 f2=col_block f3=spad0|vr0   nr=rows
OP_DMA_ST = 2     # (retired: results leave via OP_MVOUT)
OP_VLD = 3        # f0=vr0  f1=spad0                            nr=rows
OP_MM = 4         # f0=vr_a f1=ar0 f2=acc f3=spad_w     nr=rows
OP_VADD = 5       # f0=ar_d f1=ar_s1 f2=ar_s2            nr=rows
OP_VRELU = 6      # f0=ar_d f1=ar_s                      nr=rows
OP_MVOUT = 7      # f0=ar0 f1=dram_row0 f2=col_block     nr=rows  acc -> DRAM
OP_LOOP = 8       # open a loop, body is the next instruction   nr=trip count
OP_ENDLOOP = 9    # close the innermost loop

OPCODE_NAMES = {OP_NOP: "nop", OP_DMA_LD: "dma_ld", OP_DMA_ST: "dma_st",
                OP_VLD: "vld", OP_MM: "mm", OP_VADD: "vadd", OP_VRELU: "vrelu",
                OP_MVOUT: "mvout", OP_LOOP: "loop", OP_ENDLOOP: "endloop"}

# `dma_ld`'s f0: bit 0 is the SOURCE matrix, bit 1 the DESTINATION memory.
#   0: A -> spad    1: B -> spad    2: A -> vr    3: B -> vr
DMA_SRC_B = 1
DMA_TO_VR = 2

LOOP_DEPTH = 4                 # nesting levels, as MiniTPU's loop stack
IWORDS = 2                     # an instruction is two 64-bit words
NHDR = 8                       # imem[0:NHDR] is the header, instructions follow
MAXROWS = 127                  # `nr` is 8 bits, top bit spare

AGU_TERMS = 3                  # address terms per instruction
AGU_F0, AGU_F1, AGU_F2, AGU_F3 = 1, 2, 3, 4   # term targets (0 = unused)

# The names a unit may decode. `isa=()` on a unit is the claim that it decodes
# nothing, and `Unit.check` enforces it against the body.
ISA_NAMESPACE = {name: value for name, value in list(globals().items())
                 if name.startswith(("OP_", "DMA_", "AGU_"))
                 or name in ("LOOP_DEPTH", "IWORDS", "NHDR", "MAXROWS")}


def enc_agu(*terms):
    """The second instruction word: up to `AGU_TERMS` address terms.

    Each term is `(target, level, stride)` and resolves to
    `field[target] += iv[level] * stride`, so an address can be relative to any
    enclosing loop's induction variable. Terms name their target rather than
    being fixed one-per-field, because a single field often needs two.
    """
    assert len(terms) <= AGU_TERMS, f"at most {AGU_TERMS} address terms"
    w = 0
    for i, (target, level, stride) in enumerate(terms):
        assert 0 <= target <= 4 and 0 <= level < LOOP_DEPTH
        assert 0 <= stride < (1 << 11), f"stride {stride} does not fit"
        base = 19 * i
        w |= (target << base) | (level << (base + 4)) | (stride << (base + 7))
    return w


def enc(op, f0=0, f1=0, f2=0, f3=0, nr=0):
    """Assemble one instruction word."""
    for v, w in ((f0, 12), (f1, 12), (f2, 12), (f3, 12), (nr, 8)):
        assert 0 <= v < (1 << (w - 1)), (
            f"field {v} does not fit in {w - 1} usable bits "
            f"(bit {w - 1} is the sign bit after extraction)")
    return (
        (op & 0x3F)
        | (f0 << 6)
        | (f1 << 18)
        | (f2 << 30)
        | (f3 << 42)
        | (nr << 54)
    )
