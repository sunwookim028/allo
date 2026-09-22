# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Where the shipped programs put their operands, and the programs themselves.

The memory map is an ABI between a program and the machine, not a hardware
parameter: nothing in `ip/units` knows these offsets. `isa_dsl.py` generates
the shipped GEMM against the same map, and `GemmPrograms` is the reference it
is checked against word for word.
"""

from dataclasses import dataclass

from .isa import (AGU_F0, AGU_F2, AGU_F3, MAXROWS, OP_DMA_LD, OP_ENDLOOP,
                  OP_LOOP, OP_MM, OP_MVOUT, OP_VADD, OP_VRELU, DMA_SRC_B,
                  DMA_TO_VR, enc, enc_agu)


@dataclass(frozen=True)
class MemoryMap:
    """Operand and result locations for the programs below."""

    A_VR: int = 0               # A vregs:  kb * MAXDIM + m  (dma_ld'd direct)
    B_SP: int = 0               # B words:  nb * MAXDIM + k  (mm's weights)
    AR_C: int = 0               # the accumulator, up to MAXDIM words
    AR_P: int = 0               # scratch region for vector-unit programs

    @staticmethod
    def of(params):
        return MemoryMap(A_VR=0, B_SP=0, AR_C=0, AR_P=params.MAXDIM + 1)


class GemmPrograms:
    """Tiled GEMM in three forms, all at one parameter set and memory map."""

    def __init__(self, params, memory_map=None):
        self.p = params
        self.m = memory_map or MemoryMap.of(params)

    def _tiles(self, M, K, N):
        p = self.p
        assert M <= p.MAXDIM and K <= p.MAXDIM and N <= p.MAXDIM, (
            f"{M}x{K}x{N} exceeds the built MAXDIM={p.MAXDIM}")
        assert M % p.T == 0 and K % p.T == 0 and N % p.T == 0
        assert M <= MAXROWS and K <= MAXROWS
        return K // p.T, N // p.T

    def looped(self, M, K, N, relu=False):
        """Tiled GEMM with **control flow**, hand-emitted.

        The static program is O(nesting), not O(tiles): the n and k loops are
        `loop`/`endloop` pairs and the addresses are AGU terms against the
        induction variables. The first k-tile is peeled out deliberately -- it
        carries f2=0 (overwrite) where the looped tiles carry f2=1
        (accumulate), and peeling is how you say that without a predicate.

        The AGU *levels* below are hand-typed integers that have to agree with
        where the `loop`/`endloop` pairs sit, and nothing here checks that they
        do; `isa_dsl.gemm_program` derives them from nesting instead, and ships.
        """
        T, MAXDIM, m = self.p.T, self.p.MAXDIM, self.m
        Kt, Nt = self._tiles(M, K, N)
        p = []
        ins = lambda w, agu=0: p.append((w, agu))

        # --- A: one dma_ld per column block, straight into the vregs ---
        ins(enc(OP_LOOP, nr=Kt))
        ins(enc(OP_DMA_LD, f0=DMA_TO_VR, f1=0, f2=0, f3=m.A_VR, nr=M),
            enc_agu((AGU_F2, 0, 1), (AGU_F3, 0, MAXDIM)))
        ins(enc(OP_ENDLOOP))
        # --- B: one per column block, into the scratchpad ---
        ins(enc(OP_LOOP, nr=Nt))
        ins(enc(OP_DMA_LD, f0=DMA_SRC_B, f1=0, f2=0, f3=m.B_SP, nr=K),
            enc_agu((AGU_F2, 0, 1), (AGU_F3, 0, MAXDIM)))
        ins(enc(OP_ENDLOOP))

        # --- the output loop: level 0 is nb, level 1 is kb ---
        ins(enc(OP_LOOP, nr=Nt))
        #   peeled first k-tile: overwrite the accumulator. Weights by
        #   scratchpad address: B_SP + nb*MAXDIM
        ins(enc(OP_MM, f0=m.A_VR, f1=m.AR_C, f2=0, f3=m.B_SP, nr=M),
            enc_agu((AGU_F3, 0, MAXDIM)))
        if Kt > 1:
            ins(enc(OP_LOOP, nr=Kt - 1))
            #   f3 needs BOTH tiles: B_SP + nb*MAXDIM + (kb+1)*T
            ins(enc(OP_MM, f0=m.A_VR + MAXDIM, f1=m.AR_C, f2=1,
                    f3=m.B_SP + T, nr=M),
                enc_agu((AGU_F0, 1, MAXDIM), (AGU_F3, 0, MAXDIM),
                        (AGU_F3, 1, T)))
            ins(enc(OP_ENDLOOP))
        if relu:
            ins(enc(OP_VRELU, f0=m.AR_C, f1=m.AR_C, nr=M))
        ins(enc(OP_MVOUT, f0=m.AR_C, f1=0, f2=0, nr=M),
            enc_agu((AGU_F2, 0, 1)))
        ins(enc(OP_ENDLOOP))
        return p

    def flat(self, M, K, N, relu=False):
        """The fully unrolled form, kept as the differential reference: every
        instruction carries absolute addresses and there is no control flow, so
        this is what the looped program must reproduce exactly."""
        T, MAXDIM, m = self.p.T, self.p.MAXDIM, self.m
        Kt, Nt = self._tiles(M, K, N)
        p = []
        for kb in range(Kt):
            p.append((enc(OP_DMA_LD, f0=DMA_TO_VR, f1=0, f2=kb,
                          f3=m.A_VR + kb * MAXDIM, nr=M), 0))
        for nb in range(Nt):
            p.append((enc(OP_DMA_LD, f0=DMA_SRC_B, f1=0, f2=nb,
                          f3=m.B_SP + nb * MAXDIM, nr=K), 0))
        for nb in range(Nt):
            for kb in range(Kt):
                p.append((enc(OP_MM, f0=m.A_VR + kb * MAXDIM, f1=m.AR_C,
                              f2=(1 if kb else 0),
                              f3=m.B_SP + nb * MAXDIM + kb * T, nr=M), 0))
            if relu:
                p.append((enc(OP_VRELU, f0=m.AR_C, f1=m.AR_C, nr=M), 0))
            p.append((enc(OP_MVOUT, f0=m.AR_C, f1=0, f2=nb, nr=M), 0))
        return p

    def vector(self, M, K, N):
        """`relu(2 * (A @ B))` on the first output tile: a program for the
        vector unit itself, so `vadd`/`vrelu` stay exercised now that tiled
        GEMM no longer needs them on its inner loop."""
        T, m = self.p.T, self.m
        return [(w, 0) for w in [
            enc(OP_DMA_LD, f0=DMA_TO_VR, f1=0, f2=0, f3=m.A_VR, nr=M),
            enc(OP_DMA_LD, f0=DMA_SRC_B, f1=0, f2=0, f3=m.B_SP, nr=T),
            enc(OP_MM, f0=m.A_VR, f1=m.AR_C, f2=0, f3=m.B_SP, nr=M),
            enc(OP_MM, f0=m.A_VR, f1=m.AR_P, f2=0, f3=m.B_SP, nr=M),
            enc(OP_VADD, f0=m.AR_C, f1=m.AR_C, f2=m.AR_P, nr=M),
            enc(OP_VRELU, f0=m.AR_C, f1=m.AR_C, nr=M),
            enc(OP_MVOUT, f0=m.AR_C, f1=0, f2=0, nr=M)]]
