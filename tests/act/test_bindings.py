# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which compiled bindings did this checkout actually load.

The editable install maps the name ``allo`` to one checkout, so a worktree whose
own ``mlir/`` has not been built still imports *somebody else's* ``allo._mlir``
and their ``_allo`` extension, silently. Nothing looks stale, because cmake
preserves mtimes. See ``docs/source/developer/toolchains.rst``.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..")))
import allo  # noqa: E402

CHECKOUT = os.path.dirname(os.path.dirname(os.path.realpath(allo.__file__)))


def inside(path, checkout):
    return os.path.realpath(path).startswith(os.path.realpath(checkout)
                                             + os.sep)


def test_inside_accepts_a_build_of_the_same_checkout():
    assert inside("/w/tree/mlir/build/tools/allo/_mlir/_mlir_libs/_allo.so",
                  "/w/tree")


def test_inside_rejects_another_checkout_and_a_prefix_lookalike():
    assert not inside("/w/other/mlir/build/tools/allo/_mlir/x.so", "/w/tree")
    assert not inside("/w/tree-bench/mlir/build/x.so", "/w/tree")


def extension():
    libs = pytest.importorskip(
        "allo._mlir._mlir_libs._allo",
        reason="this checkout has no compiled bindings, so every test that "
               "needs the target is skipped rather than run against another "
               "checkout's")
    return os.path.realpath(libs.__file__)


def test_the_compiled_extension_comes_from_this_checkout():
    where = extension()
    assert inside(where, CHECKOUT), (
        f"`allo` is {CHECKOUT}/allo but the `_allo` extension is {where}, "
        f"which belongs to a different checkout. Build this one's bindings "
        f"(examples/tinytpu/reproduce.sh, or ninja -C "
        f"mlir/build) so the tracked `allo/_mlir` symlink resolves inside it. "
        f"See docs/source/developer/toolchains.rst.")


def test_the_python_bindings_and_the_extension_agree():
    import allo._mlir

    where = os.path.realpath(allo._mlir.__file__)
    assert inside(where, CHECKOUT), (
        f"`allo._mlir` is {where}, outside {CHECKOUT}")
