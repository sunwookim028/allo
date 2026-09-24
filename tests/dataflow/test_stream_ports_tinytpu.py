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
)
from examples.tinytpu.isa_dsl import gemm_program  # noqa: E402
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


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
