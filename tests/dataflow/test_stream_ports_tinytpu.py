# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The fork's own accelerator, wired through declared ports.

``examples/tinytpu/units_isa.py`` holds TinyTPU-isa's eight
kernel bodies unchanged at module level, each naming its streams in its own
signature, and an architecture that wires them under channel names the design
never uses. The design itself (``microarch_isa.py``) is untouched and remains
what ``reproduce.sh`` measures; this runs both on the same programs.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
os.environ.setdefault("TPU_MAXDIM", "16")

import allo.dataflow as df  # noqa: E402
from allo.netlist import netlist_of  # noqa: E402
from examples.tinytpu.microarch_isa import (  # noqa: E402
    tinytpu_isa,
    gemm_program_flat,
    assemble,
    MAXDIM,
    IMEM_SIZE,
    T,
)
from examples.tinytpu.isa_dsl import Program, gemm_program  # noqa: E402
from examples.tinytpu.units_isa import (  # noqa: E402
    tinytpu_ports,
)

SHAPES = [(4, 4, 4), (8, 8, 8), (16, 16, 16)]

# The census in docs/source/designs/tinytpu_library.rst, now read off the
# declared interfaces instead of counted by hand.
PORTS = {
    "sequencer": 5,
    "dma_ld": 3,
    "spm": 4,
    "vru": 4,
    "wld": 3,
    "pe": 5,
    "accu": 3,
    "dma_st": 2,
}


#: `vaddrelu` is the one opcode no GEMM program issues, so nothing here
#: reached the sequencer's dispatch arm for it -- which is how `units_isa.py`
#: came to be missing that arm entirely with every test still green.
#: `lift_units.py --check` is the other half of that repair.
VECTOR_M = 2 * T


def fused_program(fused: bool):
    """`relu(a @ W1 + b @ W2)` written two ways: as `vadd` then `vrelu`, and
    as the single fused `vaddrelu`. The two must agree bit for bit, and the
    fused one is the only program in this file that issues `OP_VADDRELU`.

    Region bases are derived, not typed in, for the reason
    `isa_dsl.vector_program` gives: they only have to be distinct, non-zero and
    in range, and `STRIDE` is the widest region one instruction here touches.
    """
    M = VECTOR_M
    stride = max(M, 2 * T)
    sp_a, sp_w = 1, 1 + stride
    vr_1, vr_2 = 1, 1 + stride
    ar_1, ar_2, ar_3, ar_4 = (1 + i * stride for i in range(4))
    k = Program(f"fused {M}" if fused else f"two-step {M}")
    k.dma_ld(src=0, dram_row=3, col_block=1, spad=sp_a, rows=M)
    k.dma_ld(src=1, dram_row=2, col_block=2, vr=vr_2, rows=M)
    k.dma_ld(src=1, dram_row=5, col_block=3, spad=sp_w, rows=2 * T)
    k.vld(vr_1, sp_a, rows=M)
    k.mm(vr_1, ar_1, sp_w, rows=M)
    k.mm(vr_2, ar_2, sp_w + T, rows=M)
    if fused:
        k.vaddrelu(ar_3, ar_1, ar_2, rows=M)
        out = ar_3
    else:
        k.vadd(ar_3, ar_1, ar_2, rows=M)
        k.vrelu(ar_4, ar_3, rows=M)
        out = ar_4
    k.mvout(out, dram_row=0, col_block=0, rows=M)
    return k.emit()


def operands():
    rng = np.random.default_rng(0)
    A = rng.integers(-4, 5, (MAXDIM, MAXDIM)).astype(np.int8)
    B = rng.integers(-4, 5, (MAXDIM, MAXDIM)).astype(np.int8)
    return A, B


def image(program):
    words = assemble(program)
    memory = np.zeros(IMEM_SIZE, np.uint64)
    memory[: len(words)] = np.array(words, np.uint64)
    return memory


def gold(A, B, M, K, N, relu):
    product = A[:M, :K].astype(np.int64) @ B[:K, :N].astype(np.int64)
    if relu:
        product = np.maximum(product, 0)
    return np.clip(product, -128, 127).astype(np.int8)


def run(module, program):
    A, B = operands()
    C = np.zeros(MAXDIM * MAXDIM, np.int8)
    module(image(program), A.reshape(-1), B.reshape(-1), C)
    return C.reshape(MAXDIM, MAXDIM)


@pytest.fixture(scope="module")
def ported():
    return df.build(tinytpu_ports, target="simulator")


@pytest.fixture(scope="module")
def design():
    return df.build(tinytpu_isa, target="simulator")


def test_every_kernel_declares_the_ports_the_census_counted():
    netlist = netlist_of("tinytpu_ports")
    counted = {
        instance.unit.name: len(instance.unit.ports) for instance in netlist.instances
    }
    assert counted == PORTS
    assert sum(counted.values()) == 29
    assert len(netlist.channels) == 16


def test_no_channel_keeps_the_name_the_design_gave_it():
    netlist = netlist_of("tinytpu_ports")
    for instance in netlist.instances:
        for port in instance.unit.ports:
            assert instance.bindings[port.name] != port.name


def test_the_ported_architecture_computes_the_same_results(ported):
    A, B = operands()
    for M, K, N in SHAPES:
        for relu in (False, True):
            program = (
                gemm_program if (M, K, N) == (16, 16, 16) else gemm_program_flat
            )(M, K, N, relu)
            result = run(ported, program)
            np.testing.assert_array_equal(result[:M, :N], gold(A, B, M, K, N, relu))


def test_the_ported_architecture_agrees_with_the_design(ported, design):
    for M, K, N in SHAPES:
        for relu in (False, True):
            program = (
                gemm_program if (M, K, N) == (16, 16, 16) else gemm_program_flat
            )(M, K, N, relu)
            np.testing.assert_array_equal(run(ported, program), run(design, program))


@pytest.mark.skipif(MAXDIM // T < 4 or 5 + 2 * T > MAXDIM or 2 * VECTOR_M > MAXDIM,
                    reason="the fused vector program's fixed shape needs a "
                           "larger MAXDIM at this T")
def test_the_fused_vaddrelu_arm_is_reached_and_agrees_with_the_two_step(
        ported, design):
    """The only test here that issues `OP_VADDRELU`.

    It fails if the sequencer's dispatch arm for the opcode is missing from
    the lifted units -- which it silently was, because every other program in
    this file is a GEMM and no GEMM issues it.
    """
    fused, two_step = fused_program(True), fused_program(False)
    expected = run(design, two_step)
    np.testing.assert_array_equal(run(design, fused), expected)
    np.testing.assert_array_equal(run(ported, fused), expected)
    np.testing.assert_array_equal(run(ported, two_step), expected)


def test_units_isa_is_what_the_ip_library_composes_to():
    """`units_isa.py` says GENERATED -- do not edit. This is what makes that
    true; without it the file drifted and no test could see it."""
    from examples.tinytpu.lift_units import main as lift_main  # noqa: PLC0415

    assert lift_main(["--check"]) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
