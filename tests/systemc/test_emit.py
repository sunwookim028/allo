# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The claim each SystemC demonstration in this directory makes, under pytest.

Every `.py` beside this file ends in a `__main__` block that emits SystemC and
asserts something about the text -- that a random-access boundary array became
an `AlloMemPins` instance, that a stream-only kernel became its own
`SC_MODULE`, that a `mapping=[P]` grid unrolled into P kernel modules. Those
assertions are the point of the files, and until this test existed **nothing
ran them**: they are scripts, not tests, so `tiled_systolic.py` sat asserting
`AlloMem<`/`AlloMemW<` counts for a boundary the emitter stopped producing.

So the claims live here, collected, and the `__main__` blocks stay for reading
and for the `MGC_HOME` csim path that this cannot run (neither Catapult nor
Xcelium is installed on this host -- `dev/toolchains.rst`).

Emission only: `df.build(..., target="systemc")` produces text and needs no
Catapult. It does need this checkout's MLIR bindings.
"""

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

df = pytest.importorskip(
    "allo.dataflow",
    reason="this checkout has no compiled bindings; see tests/act/test_bindings.py")


def emit(module_name, region="top"):
    import importlib

    mod = importlib.import_module(module_name)
    return df.build(getattr(mod, region), target="systemc").hls_code


# (module, region, [(substring, expected count or None for "at least one")])
CASES = [
    # A producer and a consumer over a buffer-free valid_ready Channel: one
    # SC_MODULE each, and a Combinational between them.
    ("pc_channel", "pc_channel", [
        ("SC_MODULE(producer_0)", 1),
        ("SC_MODULE(consumer_0)", 1),
    ]),
    # A kernel whose whole interface is streams is its own synthesizable
    # module, while the region's array boundary stays a member.
    ("stream_boundary", "top", [
        ("SC_MODULE(compute_0)", 1),
    ]),
    # mapping=[P] unrolls into P kernel modules with no emitter special case.
    ("systolic_chain", "top", [
        ("SC_MODULE(pe_0)", 1),
        ("SC_MODULE(pe_3)", 1),
    ]),
    # A boundary array read at arbitrary indices becomes a RAM-pins memory.
    ("mem_port_reverse", "top", [
        ("AlloMemPins<", 1),
        ("_rd(", None),
    ]),
    # ... and one written at arbitrary indices becomes one too, with stores.
    ("mem_port_scatter", "top", [
        ("AlloMemPins<", 1),
        ("_wr(", None),
    ]),
    # Three boundary arrays, three memories, both directions in one design.
    ("tiled_systolic", "top", [
        ("AlloMemPins<", 3),
        ("_rd(", None),
        ("_wr(", None),
    ]),
]


@pytest.mark.parametrize("module,region,claims",
                         CASES, ids=[c[0] for c in CASES])
def test_the_emitted_systemc_matches_the_file_s_own_claim(module, region,
                                                          claims):
    code = emit(module, region)
    for needle, count in claims:
        got = code.count(needle)
        if count is None:
            assert got >= 1, f"{module}: no {needle!r} in the emitted SystemC"
        else:
            assert got == count, (
                f"{module}: expected {count} x {needle!r}, emitted {got}. "
                f"If the emitter changed deliberately, change the number here "
                f"AND in {module}.py's __main__ block, which asserts the same "
                f"thing when run standalone.")


def test_the_four_link_variants_all_emit():
    """`dot_product_four_links.py` is a comparison, so all four must build."""
    import importlib

    mod = importlib.import_module("dot_product_four_links")
    for name, region in mod.VARIANTS.items():
        code = df.build(region, target="systemc").hls_code
        assert "SC_MODULE" in code, f"{name} emitted no module"
