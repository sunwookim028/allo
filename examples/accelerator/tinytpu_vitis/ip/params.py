# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The hardware parameter set the units are instantiated at.

Every number here is fixed at build time and independent of the workload: one
RTL build runs every shape, with M, K and N arriving as instruction fields. An
architecture passes this namespace to its units; nothing is read from the
environment or from a module global, so two architectures can instantiate the
same unit at different sizes in one process.
"""

from dataclasses import dataclass

PARAMETER_NAMES = ("T", "VW", "AW", "QD", "MAXDIM", "WPR",
                   "SPAD_ROWS", "NVR", "NAR", "IMEM_SIZE")


@dataclass(frozen=True)
class TpuParams:
    """T is the only parameter that changes the *shape* of the region: the
    array is T*T instances and the chains are T and T*T stream arrays."""

    T: int = 4                  # SIMD width == array dimension
    MAXDIM: int = 16            # largest M, K, N supported
    SPAD_ROWS: int = 512        # scratchpad rows, each one packed word
    NVR: int = 256              # operand vector registers
    NAR: int = 128              # accumulator vector registers
    QD: int = 8                 # stream depth
    IMEM_SIZE: int = 56         # instruction memory words, header included

    def __post_init__(self):
        # A packed word must hold the two 16-bit counts the scratchpad sends
        # down the weight chain to the array.
        assert self.T >= 4, "a packed operand word must be at least 32 bits"
        assert self.MAXDIM % self.T == 0, (
            "a DRAM row must be a whole number of packed words")
        assert self.IMEM_SIZE % 8 == 0, (
            "the program prefetch moves 8 words per iteration")

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

    def namespace(self) -> dict:
        return {name: getattr(self, name) for name in PARAMETER_NAMES}
