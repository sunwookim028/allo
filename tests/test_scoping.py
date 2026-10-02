# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Name resolution in the front end follows Python's scoping.

A called function's free names resolve in its own module's globals and its
closure, never in the caller's; a kernel's parameters and locals shadow
globals of the same name. Before the fix, ``get_global_vars`` merged every
reachable module's globals into one dict (first name wins) and each callee was
built with the caller's dict, so a reused function silently read the caller's
global of the same name, and a module-level numpy array silently replaced a
kernel parameter of the same name in an assignment.
"""

import importlib.util
import itertools
import textwrap

import numpy as np
import pytest

import allo
from allo.ir.types import int32, uint16

_counter = itertools.count()


def _module(tmp_path, body):
    """Import ``body`` as a fresh module from a real file (Allo reads the
    source back with ``inspect``)."""
    name = f"_scoping_lib_{next(_counter)}"
    path = tmp_path / f"{name}.py"
    path.write_text("from allo.ir.types import int32, uint16\n" + textwrap.dedent(body))
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run(kernel, *inputs, out_shape=(4,)):
    mod = allo.customize(kernel).build()
    out = np.zeros(out_shape, np.int32)
    mod(*inputs, out)
    return out


A4 = np.arange(4, dtype=np.int32)

# Module-level names a kernel in this file sees; a callee must not.
K = 5
N = 8
# A module-level array with the name of a kernel parameter below.
a = np.full(4, 7, np.int32)
b = np.full((2, 4), 7, np.int32)


def test_reused_function_reads_its_own_global(tmp_path):
    # C3: lib.K = 3, this module's K = 5.
    lib = _module(
        tmp_path,
        """
        K = 3
        def scale(x: int32) -> int32:
            return x * K
        """,
    )
    scale = lib.scale

    def kernel(A: int32[4], C: int32[4]):
        for i in range(4):
            C[i] = scale(A[i]) + K

    np.testing.assert_array_equal(_run(kernel, A4), [scale(x) + K for x in A4])


def test_reused_function_caller_local_does_not_leak(tmp_path):
    # The caller's K is a local of the enclosing (test) function this time.
    lib = _module(
        tmp_path,
        """
        K = 3
        def scale(x: int32) -> int32:
            return x * K
        """,
    )
    scale = lib.scale
    K = 11  # pylint: disable=redefined-outer-name

    def kernel(A: int32[4], C: int32[4]):
        for i in range(4):
            C[i] = scale(A[i]) + K

    np.testing.assert_array_equal(_run(kernel, A4), 3 * A4 + 11)


def test_two_level_reuse_and_one_def_name_twice(tmp_path):
    # lib.outer -> lib.inner reads lib.K = 7. The caller also has a function
    # named ``inner`` of its own: two functions with one def name (C2).
    lib = _module(
        tmp_path,
        """
        K = 7
        def inner(x: int32) -> int32:
            return x * K
        def outer(x: int32) -> int32:
            return inner(x) + 1
        """,
    )
    outer = lib.outer

    def inner(x: int32) -> int32:
        return x * 100

    def kernel(A: int32[4], C: int32[4]):
        for i in range(4):
            C[i] = outer(A[i]) + inner(A[i])

    np.testing.assert_array_equal(_run(kernel, A4), 7 * A4 + 1 + 100 * A4)


def test_attribute_call_resolves_the_module_function(tmp_path):
    lib = _module(
        tmp_path,
        """
        K = 3
        def scale(x: int32) -> int32:
            return x * K
        """,
    )

    def scale(x: int32) -> int32:  # same name, different function
        return x - 1

    def kernel(A: int32[4], C: int32[4]):
        for i in range(4):
            C[i] = lib.scale(A[i]) + scale(A[i])

    np.testing.assert_array_equal(_run(kernel, A4), 3 * A4 + A4 - 1)


def test_closure_wins_over_caller_global():
    def make(K):  # pylint: disable=redefined-outer-name
        def scale(x: int32) -> int32:
            return x * K

        return scale

    scale = make(3)

    def kernel(A: int32[4], C: int32[4]):
        for i in range(4):
            C[i] = scale(A[i])

    np.testing.assert_array_equal(_run(kernel, A4), 3 * A4)


def test_nested_helper_sees_enclosing_locals():
    # A helper nested in this file reads the enclosing function's locals,
    # through its closure (body) and the stack (annotation), as before.
    M = 4
    off = 2

    def helper(x: int32[M]) -> int32:
        s: int32 = off
        for i in range(M):
            s += x[i]
        return s

    def kernel(A: int32[4], C: int32[1]):
        C[0] = helper(A)

    np.testing.assert_array_equal(_run(kernel, A4, out_shape=(1,)), [6 + 2])


def test_callee_constant_used_as_shape(tmp_path):
    # lib.N = 4 sizes the parameter; this module's N = 8.
    lib = _module(
        tmp_path,
        """
        N = 4
        def total(x: int32[N]) -> int32:
            s: int32 = 0
            for i in range(N):
                s += x[i]
            return s
        """,
    )
    total = lib.total

    def kernel(A: int32[4], C: int32[1]):
        C[0] = total(A)

    np.testing.assert_array_equal(_run(kernel, A4, out_shape=(1,)), [6])


def test_function_passed_under_another_name(tmp_path):
    # C1: called by the name it is bound to, emitted under its def name.
    lib = _module(
        tmp_path,
        """
        def twice(x: int32) -> int32:
            return x + x
        """,
    )
    eng = lib.twice

    def kernel(A: int32[4], C: int32[4]):
        for i in range(4):
            C[i] = eng(A[i])

    np.testing.assert_array_equal(_run(kernel, A4), 2 * A4)


def test_two_modules_one_def_name(tmp_path):
    # C2: two functions both named ``add`` coexist.
    m1 = _module(tmp_path, "def add(x: int32) -> int32:\n    return x + 1\n")
    m2 = _module(tmp_path, "def add(x: int32) -> int32:\n    return x + 100\n")
    add1, add2 = m1.add, m2.add

    def kernel(A: int32[4], C: int32[4]):
        for i in range(4):
            C[i] = add1(A[i]) * 1000 + add2(A[i])

    np.testing.assert_array_equal(_run(kernel, A4), (A4 + 1) * 1000 + A4 + 100)


def test_call_argument_converted_to_parameter_type():
    # C5: ``x ^ 0x8000`` is wider than the uint16 parameter.
    def f(x: uint16) -> uint16:
        return x

    def kernel(A: uint16[4], C: uint16[4]):
        for i in range(4):
            C[i] = f(A[i] ^ 0x8000)

    mod = allo.customize(kernel).build()
    inp = np.array([0, 1, 0x8000, 0xFFFF], np.uint16)
    out = np.zeros(4, np.uint16)
    mod(inp, out)
    np.testing.assert_array_equal(out, inp ^ 0x8000)


def test_parameter_shadows_global_array():
    # C4, whole-array form: ``a`` is both a module global and a parameter.
    def kernel(a: int32[4], C: int32[4]):  # pylint: disable=redefined-outer-name
        x: int32[4] = a
        for i in range(4):
            C[i] = x[i]

    np.testing.assert_array_equal(_run(kernel, A4), A4)


def test_parameter_shadows_global_array_sliced():
    # C4, sliced form.
    def kernel(b: int32[2, 4], C: int32[4]):  # pylint: disable=redefined-outer-name
        x: int32[4] = b[1]
        for i in range(4):
            C[i] = x[i]

    B = np.arange(8, dtype=np.int32).reshape(2, 4)
    np.testing.assert_array_equal(_run(kernel, B), B[1])


def test_global_array_still_a_constant():
    # Without a parameter of that name the global array is a constant.
    def kernel(C: int32[4]):
        x: int32[4] = a
        for i in range(4):
            C[i] = x[i]

    mod = allo.customize(kernel).build()
    out = np.zeros(4, np.int32)
    mod(out)
    np.testing.assert_array_equal(out, a)


def test_local_slice_bound_is_not_the_global():
    # ``n`` is a local; the module has an ``n`` too. The slice is not the
    # constant ``A[2:4]``: refused, not silently built from the global.
    g = {"int32": int32, "n": 2}
    src = (
        "def kernel(A: int32[4], C: int32[2]):\n"
        "    n: int32 = 0\n"
        "    t: int32[2] = A[n:n + 2]\n"
        "    for j in range(2):\n"
        "        C[j] = t[j]\n"
    )
    with pytest.raises(SystemExit):
        allo.customize(src, global_vars=g)


if __name__ == "__main__":
    pytest.main([__file__])
