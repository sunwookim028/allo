# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The error taxonomy, and the reason it exists.

Every check that depends on user input — an ISA description, a hand-written
instruction stream, a source program — raises rather than asserting. That is not
style: these were all ``assert``s, so ``python -O`` deleted the entire validation
layer in one go, and because ``search._fit`` selects candidate instructions by
*catching* the rejection, an ISA that fit nothing would report that everything fit.
The last test here runs the compiler in a real ``-O`` subprocess to keep that shut.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from allo.exp.dsa import primitive
from allo.exp.dsa.access import contiguous
from allo.exp.dsa.core import ISA
from allo.exp.dsa.errors import (
    AcceleratorDescriptionError,
    AllocationError,
    AssemblyError,
    CompileError,
    DSAError,
    NoMatchError,
    ShapeError,
)
from allo.lang.core import f32

REPO_ROOT = Path(__file__).resolve().parents[2]


def _isa(name="errs", slots=4):
    isa = ISA(name)
    dram = isa.global_("dram", shape=(256,), dtype=f32)
    vreg = isa.vector("vreg", slots=slots, shape=(8,), dtype=f32)

    @isa.instruction(src=dram, dst=vreg)
    def load(I):
        @I.access
        def _(s, d):
            return (contiguous(dram, s, 8), contiguous(vreg, d, 1))

        @I.compute
        def _(a, o):
            return primitive.identity(a)

    @isa.instruction(src=vreg, dst=dram)
    def store(I):
        @I.access
        def _(s, d):
            return (contiguous(vreg, s, 1), contiguous(dram, d, 8))

        @I.compute
        def _(a, o):
            return primitive.identity(a)

    @isa.instruction(src=[vreg, vreg], dst=vreg)
    def vadd(I):
        @I.access
        def _(x, y, d):
            return (
                contiguous(vreg, x, 1),
                contiguous(vreg, y, 1),
                contiguous(vreg, d, 1),
            )

        @I.compute
        def _(x, y, d):
            return primitive.add(x, y)

    return isa, dram, vreg, load, store, vadd


def _add_src(n=8) -> str:
    t = f"tensor<{n}xf32>"
    return f"""
func.func @main(%a: {t}, %b: {t}) -> {t} {{
  %r = tosa.add %a, %b : ({t}, {t}) -> {t}
  return %r : {t}
}}
"""


# --- the taxonomy: three sources of error, three families ------------------------


def test_every_error_shares_one_base():
    """One base class, so a caller can wrap the whole frontend in a single except."""
    for cls in (AcceleratorDescriptionError, AssemblyError, CompileError):
        assert issubclass(cls, DSAError)
    for cls in (NoMatchError, ShapeError, AllocationError):
        assert issubclass(cls, CompileError)


def test_a_bad_isa_description_is_an_accelerator_description_error():
    """The ISA author's mistake, raised at declaration/trace time — never while
    compiling a program."""
    isa = ISA("bad")
    isa.global_("mem", shape=(64,), dtype=f32)
    with pytest.raises(AcceleratorDescriptionError, match="duplicate buffer"):
        isa.global_("mem", shape=(64,), dtype=f32)

    isa2, _dram, vreg, *_ = _isa("bad2")
    with pytest.raises(AcceleratorDescriptionError, match="missing @I.compute"):

        @isa2.instruction(src=vreg, dst=vreg)
        def no_compute(I):
            @I.access
            def _(a, d):
                return (contiguous(vreg, a, 1), contiguous(vreg, d, 1))


def test_a_bad_instruction_call_is_an_assembly_error():
    """Hand-written assembly: wrong operands, or a call outside an @oracle body."""
    isa, dram, _vreg, load, _store, _vadd = _isa("asm")

    with pytest.raises(AssemblyError, match="only be called inside @oracle"):
        load(s=0, d=0)

    @isa.oracle
    def bad_kwarg():
        load(s=0, nope=1)

    with pytest.raises(AssemblyError, match="unknown parameter 'nope'"):
        bad_kwarg()

    @isa.oracle
    def missing():
        load(s=0)

    with pytest.raises(AssemblyError, match="missing parameters"):
        missing()


def test_an_unmatched_source_op_is_a_no_match_error():
    isa, *_ = _isa("nomatch")
    src = """
func.func @main(%a: tensor<8xf32>) -> tensor<8xf32> {
  %r = tosa.exp %a : (tensor<8xf32>) -> tensor<8xf32>
  return %r : tensor<8xf32>
}
"""
    with pytest.raises(NoMatchError, match="no instruction matches"):
        isa.compile_program(src)


def test_a_shape_that_does_not_fit_is_a_shape_error():
    """Stage 2, and distinguishable from Stage 1: the structure matched, the size
    did not."""
    isa, *_ = _isa("shape")
    with pytest.raises(ShapeError, match="expects 8 but source is 16"):
        isa.compile_program(_add_src(16))


def test_running_out_of_buffer_is_an_allocation_error():
    """Stage 3. One vreg slot cannot hold a two-operand add's operands."""
    isa, *_ = _isa("alloc", slots=1)
    with pytest.raises(AllocationError, match="capacity too small"):
        isa.compile_program(_add_src())


def test_calling_a_compiled_program_with_the_wrong_arity():
    isa, *_ = _isa("arity")
    prog = isa.compile_program(_add_src())
    a = np.zeros(8, np.float32)
    with pytest.raises(AssemblyError, match="expected 2 inputs, got 1"):
        prog(a)


# --- the reason the taxonomy exists ----------------------------------------------


# An 8x8 matmul in the systolic-native `a @ b^T` form: it matches CornellTPU's
# `matmul` structurally, and only the exact-fit check refuses it. `__ROOT__` is
# substituted rather than formatted, so the MLIR attribute braces stay literal.
_O_SMOKE = '''
import sys
sys.path.insert(0, "__ROOT__")
from allo.exp.dsa.errors import ShapeError
from examples.accelerator.cornell_tpu.isa import tpu

if __debug__:
    raise SystemExit("subprocess is not running under -O")

SRC = """
func.func @main(%a: tensor<1x8x8xf32>, %b: tensor<1x8x8xf32>) -> tensor<1x8x8xf32> {
  %z = "tosa.const"() {values = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
  %t = tosa.transpose %b {perms = array<i32: 0, 2, 1>} : (tensor<1x8x8xf32>) -> tensor<1x8x8xf32>
  %r = tosa.matmul %a, %t, %z, %z : (tensor<1x8x8xf32>, tensor<1x8x8xf32>, tensor<1xf32>, tensor<1xf32>) -> tensor<1x8x8xf32>
  return %r : tensor<1x8x8xf32>
}
"""

try:
    tpu.compile_program(SRC)
except ShapeError as e:
    if "expects 4 but source is 8" not in str(e):
        raise SystemExit("wrong rejection: " + str(e))
    print("REJECTED-UNDER-O")
else:
    raise SystemExit("an 8x8 matmul compiled onto a 4x4-only ISA under -O")
'''


def test_validation_survives_python_dash_O():
    """The whole point. Under ``-O`` every ``assert`` in the frontend disappears, so
    when these checks were asserts an 8x8 matmul on CornellTPU's 4x4-only systolic
    stopped being rejected on shape and blew up later somewhere unrelated
    (``ValueError: list.remove(x): x not in list``). Run the real compiler in a real
    ``-O`` interpreter and require the same, specific rejection."""
    out = subprocess.run(
        [sys.executable, "-O", "-c", _O_SMOKE.replace("__ROOT__", str(REPO_ROOT))],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert out.returncode == 0, out.stdout + out.stderr
    assert "REJECTED-UNDER-O" in out.stdout, out.stdout + out.stderr
