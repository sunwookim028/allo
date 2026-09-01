# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""CornellTPU microarchitecture: a real instruction-driven accelerator.

This is the synthesizable hardware that *runs* the CornellTPU ISA (defined in
``isa.py``). It is plain Allo HLS, but the hardware blocks use ``@tpu.unit`` and
the top uses ``@tpu.entry`` (so each ``Kernel`` registers on the ISA).

Shape of the machine (host talks to it over MMIO):

- Two memory ports + one scalar control port are the *only* top-level interfaces:
  ``dmem`` (the DRAM data bus, AXI master), ``imem`` (the instruction stream, AXI
  master), and ``n_instr`` (how many instructions to run, AXI-lite). The host
  writes a program into the imem region, sets ``n_instr``, and pulses start; the
  ap_ctrl/return handshake is the Vitis s_axilite state machine (runtime, not
  modeled here).
- The on-chip state -- the ``bram`` scratchpad and the ``vreg`` vector file --
  is declared *inside the top* (local arrays), so it is private on-chip memory
  passed by reference into the units.
- Instructions are fixed ``IWIDTH``-word records ``[opcode, a0, a1, a2]``. The
  top is a fetch-decode-dispatch loop: ``for pc in range(n_instr)`` reads the
  next record, decodes ``opcode``, and calls the matching unit. (A ``while`` +
  HALT state machine would be the textbook form, but the Allo frontend has no
  ``break`` and the HLS emitter cannot yet render ``scf.while`` loop-carried
  values, so the host-supplied ``n_instr`` bounds a plain ``for`` instead.)

Build the Vitis HLS project (no Vitis needed to scaffold the files):

    python -m example.accelerator.cornell_tpu.microarch --out ./cornell_prj
"""

import argparse
from pathlib import Path

from allo.lang.core import f32, i32, range as arange

from .isa import (
    tpu,
    DRAM_SIZE,
    BRAM_SIZE,
    IMEM_SIZE,
    VEC_LANES,
    VEC_REGS,
)
from . import isa as _isa

IWIDTH = 4  # words per instruction record: [opcode, a0, a1, a2]
SYS_DIM = 4  # the systolic array is SYS_DIM x SYS_DIM

# Opcodes. The 4 VPU ops are 0..3 so the decoder recognizes the whole VPU group
# with a single `opcode <= OP_VRELU` compare; the rest follow.
OP_VADD = 0
OP_VSUB = 1
OP_VMUL = 2
OP_VRELU = 3
OP_VLOAD = 4
OP_VSTORE = 5
OP_DMA_LOAD = 6
OP_DMA_STORE = 7
OP_MATMUL = 8


# --- DRAM <-> BRAM block movers (dma) -------------------------------------- #
@tpu.unit
def dma_load(
    dmem: f32[DRAM_SIZE],
    bram: f32[BRAM_SIZE],
    dram_addr: i32,
    bram_addr: i32,
    length: i32,
):
    """Copy ``length`` words DRAM -> on-chip BRAM."""
    for i in arange(length, name="i"):
        bram[bram_addr + i] = dmem[dram_addr + i]


@tpu.unit
def dma_store(
    dmem: f32[DRAM_SIZE],
    bram: f32[BRAM_SIZE],
    dram_addr: i32,
    bram_addr: i32,
    length: i32,
):
    """Copy ``length`` words on-chip BRAM -> DRAM."""
    for i in arange(length, name="i"):
        dmem[dram_addr + i] = bram[bram_addr + i]


# --- BRAM <-> VREG movers -------------------------------------------------- #
@tpu.unit
def vload(
    bram: f32[BRAM_SIZE], vreg: f32[VEC_REGS, VEC_LANES], bram_addr: i32, slot: i32
):
    """Load one 8-lane vector BRAM -> vreg[slot]."""
    for i in arange(VEC_LANES, name="lane"):
        vreg[slot, i] = bram[bram_addr + i]


@tpu.unit
def vstore(
    vreg: f32[VEC_REGS, VEC_LANES], bram: f32[BRAM_SIZE], slot: i32, bram_addr: i32
):
    """Store vreg[slot] -> BRAM (8 lanes)."""
    for i in arange(VEC_LANES, name="lane"):
        bram[bram_addr + i] = vreg[slot, i]


# --- VPU: one 8-lane SIMD datapath, opcode-multiplexed --------------------- #
@tpu.unit
def vpu(opcode: i32, vreg: f32[VEC_REGS, VEC_LANES], a: i32, b: i32, d: i32):
    """``vreg[d] = op(vreg[a], vreg[b])`` lanewise; ``opcode`` picks the op.
    The unary vrelu rectifies ``vreg[a]`` and ignores ``b``."""
    for i in arange(VEC_LANES, name="lane"):
        x: f32 = vreg[a, i]
        y: f32 = vreg[b, i]
        if opcode == OP_VADD:
            vreg[d, i] = x + y
        elif opcode == OP_VSUB:
            vreg[d, i] = x - y
        elif opcode == OP_VMUL:
            vreg[d, i] = x * y
        else:  # OP_VRELU
            vreg[d, i] = max(x, 0.0)


# --- Systolic array: Z = X @ W^T over 4x4 tiles in BRAM -------------------- #
@tpu.unit
def mxu(bram: f32[BRAM_SIZE], w_addr: i32, x_addr: i32, z_addr: i32):
    """4x4 matmul reading row-major tiles X@``x_addr``, W@``w_addr`` and writing
    Z@``z_addr``. The weight is consumed transposed (Z[i,j] = sum_k X[i,k]*W[j,k]).

    The X and W tiles are first staged into small on-chip register files (``Xt`` /
    ``Wt``); the dot product then reads them in parallel instead of hammering the
    shared ``bram`` 8x per column, so ``bram`` needs no partitioning of its own."""
    Xt: f32[SYS_DIM, SYS_DIM]
    Wt: f32[SYS_DIM, SYS_DIM]
    for li in arange(SYS_DIM, name="li"):
        for lk in arange(SYS_DIM, name="lk"):
            Xt[li, lk] = bram[x_addr + li * SYS_DIM + lk]
            Wt[li, lk] = bram[w_addr + li * SYS_DIM + lk]
    for i in arange(SYS_DIM, name="i"):
        for j in arange(SYS_DIM, name="j"):
            acc: f32 = 0.0
            for k in arange(SYS_DIM, name="k"):
                acc += Xt[i, k] * Wt[j, k]
            bram[z_addr + i * SYS_DIM + j] = acc


# --- Top: fetch-decode-dispatch -------------------------------------------- #
from .isa import bram
@tpu.unit(extends=bram)
def bram_controller():
    bram_hw = ...
    def crossbar(...):
        pass
    crossbar(bram_hw) # define crossbar logic here
    pass


@tpu.entry
def cornell_tpu(dmem: f32[DRAM_SIZE], imem: i32[IMEM_SIZE], n_instr: i32):
    """The accelerator. ``bram``/``vreg`` are on-chip; the loop fetches each
    4-word instruction from ``imem``, decodes the opcode, and dispatches to the
    matching unit."""
    bram: f32[BRAM_SIZE] = bram
    vreg: f32[VEC_REGS, VEC_LANES]
    for pc in arange(n_instr, name="pc"):
        base: i32 = pc * IWIDTH
        opcode: i32 = imem[base]
        a0: i32 = imem[base + 1]
        a1: i32 = imem[base + 2]
        a2: i32 = imem[base + 3]
        if opcode <= OP_VRELU:
            vpu(opcode, vreg, a0, a1, a2)
        elif opcode == OP_VLOAD:
            vload(bram, vreg, a0, a1)
        elif opcode == OP_VSTORE:
            vstore(vreg, bram, a0, a1)
        elif opcode == OP_DMA_LOAD:
            dma_load(dmem, bram, a0, a1, a2)
        elif opcode == OP_DMA_STORE:
            dma_store(dmem, bram, a0, a1, a2)
        elif opcode == OP_MATMUL:
            mxu(bram, a0, a1, a2)


# ==========================================================================#
# Scheduling. The optimizations are modest -- this is a simple machine -- but
# each unit pipelines its inner loop and the systolic dot-product unrolls. The
# fetch-decode loop itself stays SEQUENTIAL: consecutive instructions carry
# dependencies through bram/vreg, so it must not be pipelined or made a dataflow
# region. Every unit schedule is then composed onto its copy inside the top.
# ==========================================================================#
dl_s = dma_load.schedule()
dl_s.pipeline("i")  # burst-copy DRAM -> BRAM at II=1

ds_s = dma_store.schedule()
ds_s.pipeline("i")  # burst-copy BRAM -> DRAM at II=1

# vreg is the vector register file: partition dim=1 (the slot dim) completely so it
# becomes 8 independently-ported registers. vpu reads two operands and writes one
# result per cycle (3 accesses) -- only with this partition can its lane loop hold
# II=1. The arg is partitioned on every unit that takes vreg (and on the top-local
# below) so the banks line up across the (non-inlined) calls.
vl_s = vload.schedule()
vl_s.pipeline("lane")  # 8-lane BRAM -> vreg load

vs_s = vstore.schedule()
vs_s.pipeline("lane")  # 8-lane vreg -> BRAM store

vpu_s = vpu.schedule()
vpu_s.pipeline("lane")  # the 8 SIMD lanes at II=1

# mxu stages X/W into completely-partitioned register tiles, so the unrolled dot
# product reads them in parallel; bram then sees only the 1 result write per column.
mxu_s = mxu.schedule()
mxu_s.partition(mxu_s.buffer("Xt"), kind=mxu_s.Complete)  # -> 16 registers
mxu_s.partition(mxu_s.buffer("Wt"), kind=mxu_s.Complete)
mxu_s.pipeline("lk")  # stage both tiles (2 bram reads/cycle)
mxu_s.unroll("k")  # fully unroll the 4-deep dot product (multiply + adder tree)
mxu_s.pipeline("j")  # then pipeline the 4 output columns

top_s = cornell_tpu.schedule()
top_s.partition(top_s.buffer("vreg"), dim=1, kind=top_s.Complete)  # on-chip reg file
top_s.compose(dl_s, ds_s, vl_s, vs_s, vpu_s, mxu_s)


# ==========================================================================#
# Binding: which unit runs each instruction, and what that unit costs.
#
# This is the ISA <-> microarchitecture link as *data*. Without it the connection
# exists only as the opcode convention the decoder above is written against, and
# the compiler cannot attribute a cycle to anything.
#
# Latency is stated per UNIT, not per instruction, because units are shared: the
# four VPU ops are one opcode-multiplexed datapath and cost the same. `trips` is
# per instruction -- how many times it occupies its unit -- which is what makes a
# burst mover's cost scale with the block it copies (`trips=lambda n: n`) while a
# fixed-size op stays at 1. An instruction's cost is then `depth + ii * trips`.
#
# The numbers below are AUTHORED, matching the schedules above (every inner loop
# is pipelined at II=1, so a unit's II is per element and its depth is the
# pipeline fill + call overhead). They are the (ii, depth) pair a synthesis report
# yields, so a measured table can replace this block without an API change.
# ==========================================================================#
tpu.latency(dma_load, ii=1, depth=8)  # DRAM burst: 1 word/cycle + AXI latency
tpu.latency(dma_store, ii=1, depth=8)
tpu.latency(vload, ii=1, depth=3)  # on-chip BRAM <-> vreg
tpu.latency(vstore, ii=1, depth=3)
tpu.latency(vpu, ii=1, depth=5)  # 8 lanes at II=1 + fmul/fadd depth
tpu.latency(mxu, ii=1, depth=20)  # stage 2 tiles, then 4 unrolled columns

tpu.bind(_isa.dma_load, dma_load, trips=lambda n: n)  # cost scales with the block
tpu.bind(_isa.dma_store, dma_store, trips=lambda n: n)
tpu.bind(_isa.vload, vload, trips=lambda: VEC_LANES)
tpu.bind(_isa.vstore, vstore, trips=lambda: VEC_LANES)
for _op in (_isa.vadd, _isa.vsub, _isa.vmul, _isa.vrelu):
    tpu.bind(_op, vpu, trips=lambda: VEC_LANES)  # one datapath, four opcodes
tpu.bind(_isa.matmul, mxu, trips=lambda: SYS_DIM * SYS_DIM)


# ==========================================================================#
# Vitis HLS project scaffolding
# ==========================================================================#
def scaffold(out_dir: str, part: str | None = None, freq_mhz: float = 300.0) -> Path:
    """Lower the optimized, composed ``top_s`` and emit a Vitis HLS project at
    ``out_dir``. ``dmem``/``imem`` become AXI-master ports; ``n_instr`` and the
    ap_ctrl return are AXI-lite."""
    kwargs: dict = {"freq_mhz": freq_mhz}
    if part:
        kwargs["part"] = part
    backend = top_s.export("vitis", **kwargs)
    backend.set_axi(0, offset="slave", bundle="dmem")  # dmem  -> DRAM
    backend.set_axi(1, offset="slave", bundle="imem")  # imem  -> DRAM
    backend.set_axilite(2)  # n_instr (scalar control)
    backend.set_axilite(-1)  # ap_ctrl on return
    return backend.scaffold_project(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default="cornell_tpu_prj", help="output project dir")
    ap.add_argument("--part", default=None, help="target FPGA part")
    ap.add_argument("--freq", type=float, default=300.0, help="clock target in MHz")
    ap.add_argument("--print-hls", action="store_true", help="print the HLS C++")
    args = ap.parse_args()

    if args.print_hls:
        print(top_s.export("vitis").hls_code)

    proj = scaffold(args.out, part=args.part, freq_mhz=args.freq)
    print(f"Scaffolded top '{tpu.top.func_name}' -> {proj}")  # type: ignore[union-attr]
    for f in sorted(Path(proj).iterdir()):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
