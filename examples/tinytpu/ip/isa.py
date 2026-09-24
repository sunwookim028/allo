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
OP_VADDRELU = 10  # f0=ar_d f1=ar_s1 f2=ar_s2            nr=rows

OPCODE_NAMES = {OP_NOP: "nop", OP_DMA_LD: "dma_ld", OP_DMA_ST: "dma_st",
                OP_VLD: "vld", OP_MM: "mm", OP_VADD: "vadd", OP_VRELU: "vrelu",
                OP_MVOUT: "mvout", OP_LOOP: "loop", OP_ENDLOOP: "endloop",
                OP_VADDRELU: "vaddrelu"}

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

# ---- THE LAYOUT, ONCE ----
# The two Python sites that take an instruction apart read these: `enc` /
# `enc_agu` below, and `Assembler.trace`'s decoder. `sequencer`'s slices are
# written out as literals instead, because Allo cannot infer a slice's width
# from symbolic bounds and a symbolic bound widens the slice to i32
# (`gen_isa.py`).
#
# `gen_isa.py` generates this layout from `isa_spec.json`, and `--check` holds
# the encoder, `expand`, the header, the sequencer's literal slices and the
# emitted HLS to it. That check is the guarantee these names do not provide:
# one definition stops the sites DRIFTING, it does not prove any of them
# matches an ISA written down independently.
OP_LO, OP_HI = 0, 6
F0_LO, F0_HI = 6, 18
F1_LO, F1_HI = 18, 30
F2_LO, F2_HI = 30, 42
F3_LO, F3_HI = 42, 54
NR_LO, NR_HI = 54, 62
FIELD_LO = (F0_LO, F1_LO, F2_LO, F3_LO)

#: One AGU term: target, then level, then stride, packed back to back.
AGU_TERM_BITS = 19
AGU_TARGET_BITS = 4
AGU_LEVEL_BITS = 3

OP_MASK = (1 << (OP_HI - OP_LO)) - 1
FIELD_MASK = (1 << (F0_HI - F0_LO)) - 1
NR_MASK = (1 << (NR_HI - NR_LO)) - 1
AGU_TARGET_MASK = (1 << AGU_TARGET_BITS) - 1
AGU_LEVEL_MASK = (1 << AGU_LEVEL_BITS) - 1
AGU_STRIDE_MASK = (1 << (AGU_TERM_BITS - AGU_TARGET_BITS - AGU_LEVEL_BITS)) - 1

# The names a unit may decode, spelled out rather than swept out of the module:
# `isa=()` on a unit is the claim that it decodes nothing, `Unit.check`
# enforces it against the body, and a claim is only as good as the list it is
# checked against.
ISA_NAMESPACE = {
    "OP_NOP": OP_NOP, "OP_DMA_LD": OP_DMA_LD, "OP_DMA_ST": OP_DMA_ST,
    "OP_VLD": OP_VLD, "OP_MM": OP_MM, "OP_VADD": OP_VADD,
    "OP_VRELU": OP_VRELU, "OP_MVOUT": OP_MVOUT, "OP_LOOP": OP_LOOP,
    "OP_ENDLOOP": OP_ENDLOOP, "OP_VADDRELU": OP_VADDRELU,
    "DMA_SRC_B": DMA_SRC_B, "DMA_TO_VR": DMA_TO_VR,
    "AGU_TERMS": AGU_TERMS, "AGU_F0": AGU_F0, "AGU_F1": AGU_F1,
    "AGU_F2": AGU_F2, "AGU_F3": AGU_F3,
    "LOOP_DEPTH": LOOP_DEPTH, "IWORDS": IWORDS, "NHDR": NHDR,
    "MAXROWS": MAXROWS,
    "OP_LO": OP_LO, "OP_HI": OP_HI, "F0_LO": F0_LO, "F0_HI": F0_HI,
    "F1_LO": F1_LO, "F1_HI": F1_HI, "F2_LO": F2_LO, "F2_HI": F2_HI,
    "F3_LO": F3_LO, "F3_HI": F3_HI, "NR_LO": NR_LO, "NR_HI": NR_HI,
    "AGU_TERM_BITS": AGU_TERM_BITS, "AGU_TARGET_BITS": AGU_TARGET_BITS,
    "AGU_LEVEL_BITS": AGU_LEVEL_BITS,
}


def enc_agu(*terms):
    """The second instruction word: up to `AGU_TERMS` address terms.

    Each term is `(target, level, stride)` and resolves to
    `field[target] += iv[level] * stride`, so an address can be relative to any
    enclosing loop's induction variable. Terms name their target rather than
    being fixed one-per-field, because a single field often needs two.
    """
    assert len(terms) <= AGU_TERMS, f"at most {AGU_TERMS} address terms"
    word = 0
    for index, (target, level, stride) in enumerate(terms):
        assert 0 <= target <= 4 and 0 <= level < LOOP_DEPTH
        assert 0 <= stride < (1 << 11), f"stride {stride} does not fit"
        base = AGU_TERM_BITS * index
        word |= ((target << base)
                 | (level << (base + AGU_TARGET_BITS))
                 | (stride << (base + AGU_TARGET_BITS + AGU_LEVEL_BITS)))
    return word


def enc(op, f0=0, f1=0, f2=0, f3=0, nr=0):
    """Assemble one instruction word."""
    widths = ((f0, F0_HI - F0_LO), (f1, F1_HI - F1_LO), (f2, F2_HI - F2_LO),
              (f3, F3_HI - F3_LO), (nr, NR_HI - NR_LO))
    for value, width in widths:
        assert 0 <= value < (1 << (width - 1)), (
            f"field {value} does not fit in {width - 1} usable bits "
            f"(bit {width - 1} is the sign bit after extraction)")
    word = (op & OP_MASK) << OP_LO
    for value, lo in zip((f0, f1, f2, f3), FIELD_LO):
        word |= value << lo
    return word | (nr << NR_LO)
