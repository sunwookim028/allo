# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Name resolution in ``@df.region`` / ``@df.kernel`` follows Python's
scoping (see also ``tests/test_scoping.py``): a function or region reused
from another module reads its own module's globals, and a kernel parameter
shadows a module-level array of the same name."""

import importlib.util
import itertools
import textwrap

import numpy as np

import allo.dataflow as df
from allo.ir.types import int32

_counter = itertools.count()

# This module's K and a; the reused code below has its own K.
K = 5
a = np.full(4, 7, np.int32)


def _module(tmp_path, body):
    name = f"_df_scoping_lib_{next(_counter)}"
    path = tmp_path / f"{name}.py"
    path.write_text(
        "import allo.dataflow as df\nfrom allo.ir.types import int32\n"
        + textwrap.dedent(body)
    )
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sim(region, *inputs):
    mod = df.build(region, target="simulator")
    out = np.zeros(4, np.int32)
    mod(*inputs, out)
    return out


A4 = np.arange(4, dtype=np.int32)


def test_kernel_calls_reused_function(tmp_path):
    # C3, as found: [0 5 10 15] before the fix.
    lib = _module(
        tmp_path,
        """
        K = 3
        def scale(x: int32) -> int32:
            return x * K
        """,
    )
    scale = lib.scale

    @df.region()
    def top(A: int32[4], C: int32[4]):
        @df.kernel(mapping=[1], args=[A, C])
        def k(a: int32[4], c: int32[4]):  # pylint: disable=redefined-outer-name
            for i in range(4):
                c[i] = scale(a[i]) + K

    np.testing.assert_array_equal(_sim(top, A4), 3 * A4 + 5)


def test_kernel_parameter_shadows_global_array():
    # C4: the parameter ``a``, not the module's ``a`` (all 7).
    @df.region()
    def top(A: int32[4], C: int32[4]):
        @df.kernel(mapping=[1], args=[A, C])
        def k(a: int32[4], c: int32[4]):  # pylint: disable=redefined-outer-name
            x: int32[4] = a
            for i in range(4):
                c[i] = x[i]

    np.testing.assert_array_equal(_sim(top, A4), A4)


def test_reused_region_reads_its_own_global(tmp_path):
    # A region from another module, called from a kernel here.
    lib = _module(
        tmp_path,
        """
        K = 3
        @df.region()
        def sub(A: int32[4], C: int32[4]):
            @df.kernel(mapping=[1], args=[A, C])
            def inner_k(a: int32[4], c: int32[4]):
                for i in range(4):
                    c[i] = a[i] * K
        """,
    )
    sub = lib.sub

    @df.region()
    def top(A: int32[4], C: int32[4]):
        @df.kernel(mapping=[1], args=[A, C])
        def outer_k(a: int32[4], c: int32[4]):  # pylint: disable=redefined-outer-name
            sub(a, c)

    np.testing.assert_array_equal(_sim(top, A4), 3 * A4)


if __name__ == "__main__":
    import pytest

    pytest.main([__file__])
