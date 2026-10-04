# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Constant arrays read by SystemC helper functions (A2, A4).

A2: the SystemC emitter declared a baked-in constant array only inside the
kernel thread that reads it; a helper function (a plain C++ function, not a
``df.kernel``) that reads one referred to an undeclared name (``'rom' was
not declared in this scope``). It is now declared once at file scope.

A4: passing the constant array to a helper made the caller's argument a
``const int32_t[]`` and the callee's parameter ``int32_t*`` (``invalid
conversion``). See ``dev/records/limitations/u3_fixes_2026-10-04.rst``.
"""
import os
import re
import tempfile

import numpy as np
import pytest

import allo.dataflow as df
from allo.ir.types import int32

needs_csim = pytest.mark.skipif(
    not (os.environ.get("MGC_HOME") and os.environ.get("SYSTEMC_HOME")),
    reason="SystemC csim needs Catapult (MGC_HOME) and SYSTEMC_HOME",
)

ROM = (np.arange(64, dtype=np.int32) * 3) % 17
N = 8


def lut(i: int32) -> int32:
    rom: int32[64] = ROM
    return rom[i]


def _helper_region():
    @df.region()
    def top(X: int32[N], R: int32[N]):
        @df.kernel(mapping=[1], args=[X, R])
        def k(x: int32[N], r: int32[N]):
            for i in range(N):
                r[i] = lut(x[i])

    return top


def _shared_region():
    """The kernel reads the same table directly as well as through the helper."""

    @df.region()
    def top(X: int32[N], R: int32[N]):
        @df.kernel(mapping=[1], args=[X, R])
        def k(x: int32[N], r: int32[N]):
            rom: int32[64] = ROM
            for i in range(N):
                r[i] = lut(x[i]) + rom[x[i] + 1]

    return top


def _inputs():
    return np.array([0, 1, 5, 17, 30, 41, 62, 7], dtype=np.int32)


def test_helper_constant_declared_at_file_scope():
    code = df.customize(_helper_region()).build(target="systemc").hls_code
    decls = re.findall(r"^( *)static const int32_t (\w+)\[64\]", code, re.M)
    assert len(decls) == 1 and decls[0][0] == "", decls
    assert code.index(f"static const int32_t {decls[0][1]}[64]") < code.index(
        "void lut("
    )


def test_shared_constant_declared_once():
    code = df.customize(_shared_region()).build(target="systemc").hls_code
    decls = re.findall(r"^( *)static const int32_t \w+\[64\]", code, re.M)
    # the helper's copy at file scope; the kernel's own table is its own
    # global (one per declaration) and stays in the thread
    assert decls.count("") == 1, decls


@needs_csim
@pytest.mark.parametrize("which", ["helper", "shared"])
def test_helper_constant_csim(which):
    region = _helper_region() if which == "helper" else _shared_region()
    x = _inputs()
    r = np.zeros(N, dtype=np.int32)
    with tempfile.TemporaryDirectory() as tmp:
        df.build(region, target="systemc", mode="csim", project=tmp)(x, r)
    want = ROM[x] if which == "helper" else ROM[x] + ROM[x + 1]
    np.testing.assert_array_equal(r, want)


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
