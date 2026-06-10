# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""QKV microarchitecture: the synthesizable hardware that runs the QKV ISA.

This is the instruction-driven accelerator that *executes* the QKV ISA (defined
in ``isa.py``). Like CornellTPU it is plain Allo HLS -- the hardware blocks use
``@qkv.unit`` and the top uses ``@qkv.entry`` -- but the datapath is bf16
(``ap_float<16,8>`` in HLS), matching the ISA, with f32 accumulation inside the
compute units.

Shape of the machine (host talks to it over MMIO):

- Two memory ports + one scalar control port are the *only* top-level interfaces:
  ``d0`` (the off-chip I/O pool, an AXI master carrying bf16 words), ``imem`` (the
  instruction stream, AXI master), and ``n_instr`` (how many instructions to run,
  AXI-lite). The host marshals Q/K/V into ``d0``, writes a program into ``imem``,
  sets ``n_instr``, and pulses start; the ap_ctrl/return handshake is the Vitis
  s_axilite state machine (runtime, not modeled here).
- The on-chip scratchpads -- ``d1`` (the 128x64 GEMM operand staging) and ``d2``
  (the 64x64 GEMM output / softmax buffer) -- are declared *inside the top* (local
  arrays), so they are private on-chip memory passed by reference into the units.
- Instructions are fixed ``IWIDTH``-word records ``[opcode, a0, a1, a2]``. The top
  is a fetch-decode-dispatch loop: ``for pc in range(n_instr)`` reads the next
  record, decodes ``opcode``, and calls the matching unit. (As in CornellTPU, the
  host-supplied ``n_instr`` bounds a plain ``for`` rather than a HALT state
  machine, which the Allo frontend cannot yet render.)

The two compute units follow the brief:

- ``mxu`` is a *traditional adder-tree* matmul ``Z = A @ B``. Each operand tile is
  staged into a one-dimension-partitioned register file (``At`` along its column
  dim, ``Bt`` along its row dim) so the fully unrolled 64-deep dot product reads
  both in parallel -- 64 multipliers feeding a 64-input adder tree, pipelined over
  the output columns. ``d1`` itself then needs no partitioning of its own.
- ``smx`` is the cosim-validated row softmax from ``allo/library/transformer``:
  three passes over the row block (max-reduce, ``exp(x-max)`` + sum-reduce,
  normalize), with the 64 columns unrolled into max / adder trees and the rows
  pipelined. The max-subtraction is numerically stable and yields the same
  probabilities as the ISA's naive ``exp / sum`` form.

Build the Vitis HLS project (no Vitis needed to scaffold the files):

    python -m example.accelerator.qkv.microarch --out ./qkv_prj
"""

import argparse
from pathlib import Path

import allo
from allo.lang.core import bf16, f16, f32, i32, range as arange
from allo.operators import math as m

from .isa import qkv, N, D0_SIZE

D1_SLOTS = qkv.buffers["d1"].size  # 128 GEMM-operand staging rows
D2_SLOTS = qkv.buffers["d2"].size  # 64 GEMM-output / softmax rows
IMEM_SIZE = 4096  # instruction-stream words (IWIDTH words per instruction)
IWIDTH = 4  # words per instruction record: [opcode, a0, a1, a2]
NEG = -1e30  # additive "-inf" seed for the stable-softmax row max

# Opcodes, one per QKV mnemonic (see isa.py).
OP_LOAD_RM = 0
OP_STORE_RM = 1
OP_LOAD_CM = 2
OP_STORE_CM = 3
OP_MOV = 4
OP_GEMM = 5
OP_SOFTMAX = 6

bf16 = f16  # alias to f16


# --- d0 <-> d1 row-major moves (n rows, plain bf16 copies) ----------------- #
@qkv.unit
def load_rm(
    d0: bf16[D0_SIZE], d1: bf16[D1_SLOTS, N], addr_in: i32, addr_out: i32, n: i32
):
    """Copy ``n`` rows d0 -> d1 row-major: ``d1[addr_out+i, j] = d0[addr_in+i*N+j]``."""
    for i in arange(n, name="i"):
        for j in arange(N, name="j"):
            d1[addr_out + i, j] = d0[addr_in + i * N + j]


@qkv.unit
def store_rm(
    d1: bf16[D1_SLOTS, N], d0: bf16[D0_SIZE], addr_in: i32, addr_out: i32, n: i32
):
    """Copy ``n`` rows d1 -> d0 row-major: ``d0[addr_out+i*N+j] = d1[addr_in+i, j]``."""
    for i in arange(n, name="i"):
        for j in arange(N, name="j"):
            d0[addr_out + i * N + j] = d1[addr_in + i, j]


# --- d0 <-> d1 column-major moves (64x64, transposing) --------------------- #
@qkv.unit
def load_cm(d0: bf16[D0_SIZE], d1: bf16[D1_SLOTS, N], addr_in: i32, addr_out: i32):
    """Load a 64x64 block d0 -> d1 transposed: ``d1[addr_out+i, j] = d0[addr_in+j*N+i]``."""
    for i in arange(N, name="i"):
        for j in arange(N, name="j"):
            d1[addr_out + i, j] = d0[addr_in + j * N + i]


@qkv.unit
def store_cm(d1: bf16[D1_SLOTS, N], d0: bf16[D0_SIZE], addr_in: i32, addr_out: i32):
    """Store a 64x64 block d1 -> d0 transposed: ``d0[addr_out+i*N+j] = d1[addr_in+j, i]``."""
    for i in arange(N, name="i"):
        for j in arange(N, name="j"):
            d0[addr_out + i * N + j] = d1[addr_in + j, i]


# --- mov: copy n rows d2 -> d1 (stage a GEMM result back as a GEMM operand) - #
@qkv.unit
def mov(
    d2: bf16[D2_SLOTS, N], d1: bf16[D1_SLOTS, N], addr_in: i32, addr_out: i32, n: i32
):
    """Copy ``n`` rows d2 -> d1: ``d1[addr_out+i, j] = d2[addr_in+i, j]``."""
    for i in arange(n, name="i"):
        for j in arange(N, name="j"):
            d1[addr_out + i, j] = d2[addr_in + i, j]


# --- mxu: traditional adder-tree matmul Z = A @ B over 64x64 tiles --------- #
@qkv.unit
def mxu(
    d1: bf16[D1_SLOTS, N], d2: bf16[D2_SLOTS, N], a_addr: i32, b_addr: i32, z_addr: i32
):
    """``Z[i,j] = sum_k A[i,k] * B[k,j]`` reading A@``a_addr`` / B@``b_addr`` from d1
    and writing Z@``z_addr`` to d2 (plain matmul -- the column-major loads already
    transpose where needed).

    Each operand is staged into a register file partitioned on its contraction dim
    (``At`` on its columns, ``Bt`` on its rows), so the fully unrolled 64-deep dot
    product reads both in parallel: 64 multipliers + a 64-input adder tree,
    pipelined over the output columns. Accumulation is in f32; the result is cast
    back to bf16 on the d2 store."""
    At: f32[N, N]
    Bt: f32[N, N]
    for ai in arange(N, name="ai"):
        for ak in arange(N, name="ak"):
            At[ai, ak] = d1[a_addr + ai, ak]
    for bk in arange(N, name="bk"):
        for bj in arange(N, name="bj"):
            Bt[bk, bj] = d1[b_addr + bk, bj]
    for i in arange(N, name="i"):
        for j in arange(N, name="j"):
            acc: f32 = 0.0
            for k in arange(N, name="k"):  # unrolled -> multiply + adder tree
                acc += At[i, k] * Bt[k, j]
            d2[z_addr + i, j] = acc


# --- smx: stable row softmax of an n x 64 block in d2 (in place) ----------- #
@qkv.unit
def smx(d2: bf16[D2_SLOTS, N], addr: i32, n: i32):
    """Row softmax of ``d2[addr:addr+n]`` in place, after the cosim-validated
    ``allo/library/transformer`` design: three passes over an on-chip f32 row
    buffer -- (1) max-reduce each row, (2) ``exp(x-max)`` -> buf with a sum-reduce,
    (3) multiply by ``1/sum`` -- with the 64 columns unrolled into max / adder
    trees and the rows pipelined. ``softmax(x) == softmax(x-max(x))``, so the
    stable form matches the ISA's naive ``exp / sum`` semantics."""
    buf: f32[D2_SLOTS, N]
    mx: f32[D2_SLOTS]
    iv: f32[D2_SLOTS]
    for r in arange(n, name="m_r"):  # pass 1: read row -> buf, max-reduce
        mr: f32 = NEG
        for j in arange(N, name="m_j"):  # unrolled -> max tree
            v: f32 = d2[addr + r, j]
            buf[r, j] = v
            mr = allo.max(mr, v)
        mx[r] = mr
    for r in arange(n, name="e_r"):  # pass 2: exp(x-max) -> buf, sum-reduce
        sr: f32 = 0.0
        for j in arange(N, name="e_j"):  # unrolled -> exp + adder tree
            e: f32 = m.exp(buf[r, j] - mx[r])
            buf[r, j] = e
            sr = sr + e
        iv[r] = 1.0 / sr
    for r in arange(n, name="n_r"):  # pass 3: normalize
        for j in arange(N, name="n_j"):  # unrolled
            d2[addr + r, j] = buf[r, j] * iv[r]


# --- Top: fetch-decode-dispatch -------------------------------------------- #
@qkv.entry
def qkv_top(d0: bf16[D0_SIZE], imem: i32[IMEM_SIZE], n_instr: i32):
    """The accelerator. ``d1``/``d2`` are on-chip; the loop fetches each 4-word
    instruction from ``imem``, decodes the opcode, and dispatches to the matching
    unit."""
    d1: bf16[D1_SLOTS, N]
    d2: bf16[D2_SLOTS, N]
    for pc in arange(n_instr, name="pc"):
        base: i32 = pc * IWIDTH
        opcode: i32 = imem[base]
        a0: i32 = imem[base + 1]
        a1: i32 = imem[base + 2]
        a2: i32 = imem[base + 3]
        if opcode == OP_LOAD_RM:
            load_rm(d0, d1, a0, a1, a2)
        elif opcode == OP_STORE_RM:
            store_rm(d1, d0, a0, a1, a2)
        elif opcode == OP_LOAD_CM:
            load_cm(d0, d1, a0, a1)
        elif opcode == OP_STORE_CM:
            store_cm(d1, d0, a0, a1)
        elif opcode == OP_MOV:
            mov(d2, d1, a0, a1, a2)
        elif opcode == OP_GEMM:
            mxu(d1, d2, a0, a1, a2)
        elif opcode == OP_SOFTMAX:
            smx(d2, a0, a1)


# ==========================================================================#
# Scheduling. Every mover pipelines its inner lane copy; the matmul stages both
# operands then unrolls the dot product and pipelines the output columns; the
# softmax unrolls each 64-wide column pass and pipelines the rows. The
# fetch-decode loop itself stays SEQUENTIAL: consecutive instructions carry
# dependencies through d1/d2, so it must not be pipelined or made a dataflow
# region. Each unit schedule is then composed onto its copy inside the top.
# ==========================================================================#
lr_s = load_rm.schedule()
lr_s.pipeline("j")  # 64-lane row copy at II=1

sr_s = store_rm.schedule()
sr_s.pipeline("j")

lc_s = load_cm.schedule()
lc_s.pipeline("j")  # transposing copy (strided d0 read)

sc_s = store_cm.schedule()
sc_s.pipeline("j")

mv_s = mov.schedule()
mv_s.pipeline("j")

# mxu stages A/B into register files partitioned on the contraction dim (At on its
# columns, Bt on its rows), so the unrolled dot product reads both in parallel; d2
# then sees only the 1 result write per column.
mxu_s = mxu.schedule()
mxu_s.partition(mxu_s.buffer("At"), dim=2, kind=mxu_s.Complete)  # 64 banks (col dim)
mxu_s.partition(mxu_s.buffer("Bt"), dim=1, kind=mxu_s.Complete)  # 64 banks (row dim)
mxu_s.pipeline("ak")  # stage A row at II=1
mxu_s.pipeline("bj")  # stage B row at II=1
mxu_s.unroll("k")  # fully unroll the 64-deep dot product (multiply + adder tree)
mxu_s.pipeline("j")  # then pipeline the 64 output columns

# smx unrolls each 64-wide column pass (into max / adder trees) over the
# completely-partitioned row buffer and pipelines the rows.
smx_s = smx.schedule()
smx_s.partition(smx_s.buffer("buf"), dim=2, kind=smx_s.Complete)  # 64 banks (col dim)
smx_s.unroll("m_j")
smx_s.pipeline("m_r")  # pass 1: max-reduce
smx_s.unroll("e_j")
smx_s.pipeline("e_r")  # pass 2: exp + sum-reduce
smx_s.unroll("n_j")
smx_s.pipeline("n_r")  # pass 3: normalize

top_s = qkv_top.schedule()
top_s.partition(top_s.buffer("d2"), dim=2, kind=top_s.Complete)  # 64 lane banks
top_s.compose(lr_s, sr_s, lc_s, sc_s, mv_s, mxu_s, smx_s)


# ==========================================================================#
# Vitis HLS project scaffolding
# ==========================================================================#
def scaffold(out_dir: str, part: str | None = None, freq_mhz: float = 300.0) -> Path:
    """Lower the optimized, composed ``top_s`` and emit a Vitis HLS project at
    ``out_dir``. ``d0``/``imem`` become AXI-master ports; ``n_instr`` and the
    ap_ctrl return are AXI-lite."""
    kwargs: dict = {"freq_mhz": freq_mhz}
    if part:
        kwargs["part"] = part
    backend = top_s.export("vitis", **kwargs)
    backend.set_axi(0, offset="slave", bundle="d0")  # d0   -> off-chip pool
    backend.set_axi(1, offset="slave", bundle="imem")  # imem -> instruction stream
    backend.set_axilite(2)  # n_instr (scalar control)
    backend.set_axilite(-1)  # ap_ctrl on return
    return backend.scaffold_project(out_dir)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", default="qkv_prj", help="output project dir")
    ap.add_argument("--part", default=None, help="target FPGA part")
    ap.add_argument("--freq", type=float, default=300.0, help="clock target in MHz")
    ap.add_argument("--print-hls", action="store_true", help="print the HLS C++")
    args = ap.parse_args()

    if args.print_hls:
        print(top_s.export("vitis").hls_code)

    proj = scaffold(args.out, part=args.part, freq_mhz=args.freq)
    print(f"Scaffolded top '{qkv.top.func_name}' -> {proj}")  # type: ignore[union-attr]
    for f in sorted(Path(proj).iterdir()):
        print(f"    {f.name}")


if __name__ == "__main__":
    main()
