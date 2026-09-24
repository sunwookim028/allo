# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The hardware parameter set the units are instantiated at.

Every number here is fixed at build time and independent of the workload: one
RTL build runs every shape, with M, K and N arriving as instruction fields. An
architecture passes this namespace to its units; nothing is read from the
environment or from a module global, so two architectures can instantiate the
same unit at different sizes in one process.

The memory sizes are DERIVED from T and MAXDIM rather than typed in. They used
to be three independent literals that happened to be big enough at MAXDIM=16
and silently were not above it: the shipped GEMM lays A out at
`A_VR + kb*MAXDIM + m` and B at `B_SP + nb*MAXDIM + k`, so the highest operand
row either names is `(MAXDIM/T - 1)*MAXDIM + MAXDIM == MAXDIM*MAXDIM/T`, which
at MAXDIM=64 is 1024 rows against a 256-entry vreg file -- a build that
assembles and gives wrong answers. `None` on any of them means "derive"; an
explicit value overrides, which is how a deliberately-undersized build is
probed.
"""

from dataclasses import dataclass

PARAMETER_NAMES = ("T", "VW", "AW", "QD", "MAXDIM", "WPR", "OPERAND_ROWS",
                   "DMA_WORDS", "SPAD_ROWS", "NVR", "NAR", "IMEM_SIZE")

#: The fixed window `stress_isa.random_program` addresses in each memory,
#: independently of MAXDIM: a memory smaller than this cannot hold a fuzz
#: program at all, so it is a floor under the derived size. Sizing purely to
#: `OPERAND_ROWS` gave 16 rows at MAXDIM=8 and took `param_check.py` from
#: `PARAM OK` to "0 of 24 random programs could be generated".
TEST_WINDOW = 64

#: Widest beat the operand ports can move once `align_value(64)` is emitted.
BUS_BYTES = 64

#: An address field carries 11 usable bits (`enc`'s spare-sign-bit rule), so the
#: highest operand address is 2047 and the layout may name 2048 rows. At T=4
#: that is MAXDIM <= 88 (90.5 unrounded, and MAXDIM is a multiple of T), at T=8
#: 128; MAXDIM=96 at T=4 fails in `Assembler.check` with "AGU-resolved f3=2112
#: is outside the 0..2047 range". isa_spec.json computes both.
ADDRESS_FIELD_MAX = (1 << 11)


@dataclass(frozen=True)
class TpuParams:
    """T is the only parameter that changes the *shape* of the region: the
    array is T*T instances and the chains are T and T*T stream arrays."""

    T: int = 4                  # SIMD width == array dimension
    MAXDIM: int = 64            # largest M, K, N supported
    SPAD_ROWS: int = None       # scratchpad rows, each one packed word
    NVR: int = None             # operand vector registers
    NAR: int = None             # accumulator vector registers
    QD: int = 16                # stream depth (16 since item 24: depth 8 deadlocks legal programs)
    IMEM_SIZE: int = 56         # instruction memory words, header included
    DMA_WORDS: int = 1          # packed words per operand-burst iteration

    def __post_init__(self):
        # A packed word must hold the two 16-bit counts the scratchpad sends
        # down the weight chain to the array.
        assert self.T >= 4, "a packed operand word must be at least 32 bits"
        assert self.MAXDIM % self.T == 0, (
            "a DRAM row must be a whole number of packed words")
        assert self.IMEM_SIZE % 8 == 0, (
            "the program prefetch moves 8 words per iteration")
        assert self.OPERAND_ROWS <= ADDRESS_FIELD_MAX, (
            f"MAXDIM={self.MAXDIM} at T={self.T} needs {self.OPERAND_ROWS} "
            f"operand rows, past the {ADDRESS_FIELD_MAX - 1} an 11-bit address "
            f"field carries")
        assert self.DMA_WORDS >= 1
        if self.SPAD_ROWS is None:
            object.__setattr__(self, "SPAD_ROWS",
                               max(TEST_WINDOW, self.OPERAND_ROWS))
        if self.NVR is None:
            object.__setattr__(self, "NVR", max(TEST_WINDOW, self.OPERAND_ROWS))
        if self.NAR is None:
            # `AR_C` is MAXDIM rows and `AR_P` another MAXDIM starting at
            # MAXDIM+1, so the GEMM and vector programs need 2*MAXDIM+2; the
            # 128 floor keeps the fixed-address test programs
            # (`isa_dsl.vector_program` reaches row 112) legal at small MAXDIM.
            object.__setattr__(self, "NAR",
                               max(128, TEST_WINDOW, 2 * self.MAXDIM + 8))

    @property
    def VW(self) -> int:
        """Packed operand word: T int8 lanes."""
        return self.T * 8

    @property
    def AW(self) -> int:
        """Packed accumulator word: T int32 lanes."""
        return self.T * 32

    @property
    def WPR(self) -> int:
        """Packed words per DRAM row."""
        return self.MAXDIM // self.T

    @property
    def OPERAND_ROWS(self) -> int:
        """Rows of a memory the shipped GEMM's operand layout names."""
        return (self.MAXDIM // self.T) * self.MAXDIM

    @staticmethod
    def widest_burst(T: int, MAXDIM: int) -> int:
        """The widest operand burst the 64-byte bus holds, capped at one whole
        DRAM row: rounding the row span up to a multiple of it never leaves the
        operand, since the extra words land in burst-buffer rows no instruction
        names."""
        return min(MAXDIM // T, max(1, BUS_BYTES // T))

    def namespace(self) -> dict:
        return {name: getattr(self, name) for name in PARAMETER_NAMES}
