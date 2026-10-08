# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""M-R1: MiniTPU's whole ``minitpu_core`` as one Allo ``RTLModule``, inside a ``@df.region``.

The region (``target="simulator"``) has three kernels and five kinds of boundary memory:

* ``feed``  -- reads the program ``P`` (command words, ``gen_shim.CMD``) and puts it on the ``cmd`` stream;
* ``core``  -- the host driver: one call of the IP (``minitpu_core_shim`` around the unmodified core,
  Verilated by PR #48's ``RTLModule``) with the eight DDR banks ``B0..B7`` as ``MemPort`` RAMs, then copies
  the drain window (``W`` = [first DM word, DM words]) out of the banks into ``R``;
* ``sink``  -- takes the ``NSTAT`` status words off the ``st`` stream into ``S``.

One call of the built module is one MiniTPU launch from reset (the IP is not persistent), as one run of
MiniTPU's ``tb_kernel_image`` is. The host-side helpers below only lay bytes into those arrays and read them
back; ``run_kernel.py`` drives them.
"""

from __future__ import annotations

import os
import pathlib

import numpy as np

import allo.dataflow as df
from allo import MemPort, Port, RTLModule
from allo.compose import Architecture, Channel, Memory, unit
from allo.ir.types import Stream, int32

try:  # a package import (examples.minitpu_rtl...) or a script beside it
    from . import gen_shim
except ImportError:  # pragma: no cover
    import gen_shim  # type: ignore

HERE = pathlib.Path(__file__).resolve().parent
SHIM = HERE / "rtl" / "minitpu_core_shim.sv"

BANKS = gen_shim.BANKS
BANK_WORDS = 1 << gen_shim.BANK_ADDR_W          # DM words per bank = 32 MiB / 32 B
MEM_BASE = gen_shim.MEM_BASE
NSTAT = gen_shim.NSTAT
MAX_BUNDLES = 1280                              # the program array's room (largest oracle image: 1208)
NCMD = 4 * MAX_BUNDLES + 32                     # IRAM header + 4 words/bundle + CSRs + START + END
DRAIN_BEATS = 1 << 15                           # 1 MiB drain room (largest oracle drain: 550 KiB)
DRAIN_WORDS = DRAIN_BEATS * BANKS
OP = gen_shim.CMD
STATUS_NAMES = [name for name, _ in gen_shim.STATUS]

# tb_kernel_image's layout (tb/tb_kernel_image.sv: MEM_BASE, KBIN_ADDR, ARG_BASE), so arguments bind to the
# same DM-word pointers and the DMA splits bursts at the same 4 KiB pages as on the testbench.
ARG_BASE = (MEM_BASE + 0x0010_0000, MEM_BASE + 0x0040_0000, MEM_BASE + 0x0180_0000, MEM_BASE + 0x0100_0000)
KBIN_ADDR = MEM_BASE


def core_files(tree: pathlib.Path) -> list[pathlib.Path]:
    """``src/core/core.f`` of the MiniTPU export, expanded (``+incdir+`` lines become ``include_paths``)."""
    lines = (tree / "src" / "core" / "core.f").read_text().split()
    return [tree / line for line in lines if not line.startswith("+")]


def make_ip(tree: pathlib.Path, fifo_depth: int, *, build_jobs: int = 16, max_stall: int = 20_000_000):
    ports = [Port("cmd", "cmd_data", "cmd_valid", "cmd_ready", size=NCMD),
             Port("st", "st_data", "st_valid", "st_ready", dir="out")]
    ports += [MemPort(f"m{k}", BANK_WORDS, "int32_t", f"m{k}_addr", f"m{k}_ce", q=f"m{k}_q", we=f"m{k}_we",
                      d=f"m{k}_d") for k in range(BANKS)]
    return RTLModule(
        "minitpu_core_shim", [*core_files(tree), SHIM], ports=ports, name="minitpu_core_shim_sim",
        clock="clk", reset="rst_n", reset_active_high=False, start=None, done="done", persistent=False,
        include_paths=[tree / "src" / "pkg"],
        defines={"MINITPU_MXU_OUTPUT_FIFO_DEPTH": fifo_depth},
        verilator_args=["--build-jobs", str(build_jobs)], max_stall=max_stall)


def make_region(ip):
    """The Allo design: program -> cmd stream -> IP (DDR banks as RAMs) -> status stream; drain copy-out."""

    @df.region()
    def minitpu_rtl(P: int32[NCMD], B0: int32[BANK_WORDS], B1: int32[BANK_WORDS], B2: int32[BANK_WORDS],
                    B3: int32[BANK_WORDS], B4: int32[BANK_WORDS], B5: int32[BANK_WORDS], B6: int32[BANK_WORDS],
                    B7: int32[BANK_WORDS], W: int32[2], R: int32[DRAIN_WORDS], S: int32[NSTAT]):
        cmd: Stream[int32, 4]
        st: Stream[int32, 4]

        @df.kernel(mapping=[1], args=[P])
        def feed(p: int32[NCMD]):
            for i in range(NCMD):
                cmd.put(p[i])

        @df.kernel(mapping=[1], args=[B0, B1, B2, B3, B4, B5, B6, B7, W, R])
        def core(b0: int32[BANK_WORDS], b1: int32[BANK_WORDS], b2: int32[BANK_WORDS], b3: int32[BANK_WORDS],
                 b4: int32[BANK_WORDS], b5: int32[BANK_WORDS], b6: int32[BANK_WORDS], b7: int32[BANK_WORDS],
                 w: int32[2], r: int32[DRAIN_WORDS]):
            ip(cmd, st, b0, b1, b2, b3, b4, b5, b6, b7)
            first: int32 = w[0]
            count: int32 = w[1]
            for j in range(DRAIN_BEATS):
                if j < count:
                    r[j * 8 + 0] = b0[first + j]
                    r[j * 8 + 1] = b1[first + j]
                    r[j * 8 + 2] = b2[first + j]
                    r[j * 8 + 3] = b3[first + j]
                    r[j * 8 + 4] = b4[first + j]
                    r[j * 8 + 5] = b5[first + j]
                    r[j * 8 + 6] = b6[first + j]
                    r[j * 8 + 7] = b7[first + j]

        @df.kernel(mapping=[1], args=[S])
        def sink(s: int32[NSTAT]):
            for i in range(NSTAT):
                s[i] = st.get()

    return minitpu_rtl


# ---------------------------------------------------------------------------- the same region, composed
# The same three kernels as compose units: compose checks each body against its declaration (channels,
# memories, parameters -- the IP object is bound as the parameter CORE) and emits the region text.


@unit(memories=("P",), writes=("cmd",), parameters=("NCMD",))
def feed(p: int32[NCMD]):
    for i in range(NCMD):
        cmd.put(p[i])


@unit(memories=("B0", "B1", "B2", "B3", "B4", "B5", "B6", "B7", "W", "R"), reads=("cmd",), writes=("st",),
      parameters=("CORE", "BANK_WORDS", "DRAIN_BEATS", "DRAIN_WORDS"))
def core(b0: int32[BANK_WORDS], b1: int32[BANK_WORDS], b2: int32[BANK_WORDS], b3: int32[BANK_WORDS],
         b4: int32[BANK_WORDS], b5: int32[BANK_WORDS], b6: int32[BANK_WORDS], b7: int32[BANK_WORDS],
         w: int32[2], r: int32[DRAIN_WORDS]):
    CORE(cmd, st, b0, b1, b2, b3, b4, b5, b6, b7)
    first: int32 = w[0]
    count: int32 = w[1]
    for j in range(DRAIN_BEATS):
        if j < count:
            r[j * 8 + 0] = b0[first + j]
            r[j * 8 + 1] = b1[first + j]
            r[j * 8 + 2] = b2[first + j]
            r[j * 8 + 3] = b3[first + j]
            r[j * 8 + 4] = b4[first + j]
            r[j * 8 + 5] = b5[first + j]
            r[j * 8 + 6] = b6[first + j]
            r[j * 8 + 7] = b7[first + j]


@unit(memories=("S",), reads=("st",), parameters=("NSTAT",))
def sink(s: int32[NSTAT]):
    for i in range(NSTAT):
        s[i] = st.get()


def architecture(ip, name="minitpu_rtl"):
    banks = tuple(Memory(f"B{k}", "int32[BANK_WORDS]") for k in range(BANKS))
    return Architecture(
        name=name,
        parameters={"CORE": ip, "NCMD": NCMD, "NSTAT": NSTAT, "BANK_WORDS": BANK_WORDS,
                    "DRAIN_BEATS": DRAIN_BEATS, "DRAIN_WORDS": DRAIN_WORDS},
        memories=(Memory("P", "int32[NCMD]"), *banks, Memory("W", "int32[2]"),
                  Memory("R", "int32[DRAIN_WORDS]"), Memory("S", "int32[NSTAT]")),
        channels=(Channel("cmd", "int32", "4", carries="feed -> core: the program, gen_shim.CMD words"),
                  Channel("st", "int32", "4", carries="core -> sink: NSTAT status words per run")),
        units=(feed, core, sink))


def build(tree: pathlib.Path, fifo_depth: int, *, composed: bool = True, **ip_options):
    """The region built for the simulator: composed by ``compose.Architecture`` (default) or the plain
    ``@df.region`` above. Both are the same three kernels."""
    ip = make_ip(tree, fifo_depth, **ip_options)
    if composed:
        return architecture(ip).build(target="simulator"), ip
    return df.build(make_region(ip), target="simulator"), ip


# ---------------------------------------------------------------------------- host-side layout helpers


def program(image: bytes, csrs: dict[int, int], max_cycles: int) -> np.ndarray:
    """The command words for one launch: load IRAM, write CSRs, set the cycle limit, START, pad, END."""
    if len(image) % 16:
        raise ValueError("an asm.py image is a whole number of 16-byte bundles")
    bundles = len(image) // 16
    if bundles > MAX_BUNDLES:
        raise ValueError(f"{bundles} bundles exceed the program array's room ({MAX_BUNDLES})")
    words = [(OP["IRAM"] << 28) | bundles, *np.frombuffer(image, dtype="<u4").tolist()]
    for k, value in sorted(csrs.items()):
        words += [(OP["CSR"] << 28) | k, value & 0xFFFF_FFFF]
    words += [OP["MAXCYC"] << 28, max_cycles & 0xFFFF_FFFF, OP["START"] << 28]
    if len(words) >= NCMD:
        raise ValueError("program does not fit")
    out = np.zeros(NCMD, dtype=np.uint32)       # zero = NOP
    out[:len(words)] = words
    out[-1] = OP["END"] << 28
    return out.view(np.int32)


def ddr_image(image: bytes, windows: list[bytes]) -> np.ndarray:
    """tb_kernel_image's DDR: the kernel binary at KBIN_ADDR, argument window n at ARG_BASE[n], zero elsewhere."""
    ddr = np.zeros(BANK_WORDS * 32, dtype=np.uint8)
    ddr[KBIN_ADDR - MEM_BASE:KBIN_ADDR - MEM_BASE + len(image)] = np.frombuffer(image, dtype=np.uint8)
    for base, window in zip(ARG_BASE, windows):
        if window:
            ddr[base - MEM_BASE:base - MEM_BASE + len(window)] = np.frombuffer(window, dtype=np.uint8)
    return ddr


def banks_of(ddr: np.ndarray) -> list[np.ndarray]:
    """Bank k holds bytes [4k, 4k + 4) of every 32-byte DM word."""
    words = ddr.view("<u4").reshape(BANK_WORDS, BANKS)
    return [np.ascontiguousarray(words[:, k]).view(np.int32) for k in range(BANKS)]


def csr_values(offsets, values) -> dict[int, int]:
    """kernel_arg_csr words as tb_kernel_image writes them: a DM-word pointer, or +ARG_VALUEn as is."""
    csrs = {}
    for a in range(4):
        if values[a] is not None:
            csrs[a] = values[a]
        else:
            if offsets[a] % 32:
                raise ValueError("argument offsets are whole DM words")
            csrs[a] = (ARG_BASE[a] + offsets[a]) >> 5
    return csrs


def drain_window(argument: int, nbytes: int) -> np.ndarray:
    first = (ARG_BASE[argument] - MEM_BASE) // 32
    beats = -(-nbytes // 32)
    if beats > DRAIN_BEATS:
        raise ValueError(f"drain of {nbytes} bytes exceeds the drain room")
    return np.array([first, beats], dtype=np.int32)


def run(mod, image: bytes, windows, offsets, values, drain_argument: int, drain_bytes: int, max_cycles: int,
        cwd: pathlib.Path):
    """One launch. ``cwd`` must be the MiniTPU tree: the SFU ROMs are $readmemh'd by a cwd-relative path."""
    prog = program(image, csr_values(offsets, values), max_cycles)
    banks = banks_of(ddr_image(image, windows))
    window = drain_window(drain_argument, drain_bytes)
    drained = np.zeros(DRAIN_WORDS, dtype=np.int32)
    status = np.zeros(NSTAT, dtype=np.int32)
    here = os.getcwd()
    os.chdir(cwd)
    try:
        mod(prog, *banks, window, drained, status)
    finally:
        os.chdir(here)
    drain = drained.view(np.uint32).astype("<u4").tobytes()[:drain_bytes]
    return dict(zip(STATUS_NAMES, status.view(np.uint32).tolist())), drain, banks
