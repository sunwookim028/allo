# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``s.partition`` on a ``@ Stateful`` array (F5).

It crashed in ``find_buffer`` (``'NoneType' object has no attribute
'operations'``: the kernel's ``if we: mem[wa] = wd`` is an ``scf.if`` with
no else block) and, past that, could not find the array: a Stateful's
``memref.global`` is named ``__stateful_<func>_<name>_<n>``, not ``<name>``.
Now the partition applies to the Stateful's storage like any array: the
global and its ``get_global`` take the layout; the SystemC emitter spells a
complete partition as Catapult's ``hls_resource [Register]`` on the module
member; the LLVM path strips the layout as it does for an alloc. See
``dev/records/limitations/uint_index_2026-10-02.rst``.
"""
import numpy as np
import pytest

import allo
import allo.dataflow as df
from allo.customize import Partition
from allo.ir.types import UInt, uint1, uint16, Stateful

N = 8
A5 = UInt(5)

RA = np.array([0, 1, 2, 3, 0, 1, 2, 3], np.uint8)
WA = np.array([0, 1, 2, 3, 0, 0, 0, 0], np.uint8)
WD = np.array([10, 11, 12, 13, 0, 0, 0, 0], np.uint16)
WE = np.array([1, 1, 1, 1, 0, 0, 0, 0], np.uint8)
WANT = [0, 0, 0, 0, 10, 11, 12, 13]  # read-old-on-write; a write is visible next


def rf(ra: A5[N], wa: A5[N], wd: uint16[N], we: uint1[N], qa: uint16[N]):
    mem: uint16[32] @ Stateful = 0
    for t in range(N):
        qa[t] = mem[ra[t]]
        if we[t]:
            mem[wa[t]] = wd[t]


def _region():
    @df.region()
    def top(RAa: A5[N], WAa: A5[N], WDa: uint16[N], WEa: uint1[N], QAa: uint16[N]):
        @df.kernel(mapping=[1], args=[RAa, WAa, WDa, WEa, QAa])
        def rf(ra: A5[N], wa: A5[N], wd: uint16[N], we: uint1[N], qa: uint16[N]):
            mem: uint16[32] @ Stateful = 0
            for t in range(N):
                qa[t] = mem[ra[t]]
                if we[t]:
                    mem[wa[t]] = wd[t]

    return top


def test_partition_stateful_applies_to_the_global():
    s = allo.customize(rf)
    s.partition("rf:mem", Partition.Complete)  # crashed
    ir = str(s.module)
    glob = [l for l in ir.splitlines() if "memref.global" in l and "__stateful_rf_mem" in l]
    getg = [l for l in ir.splitlines() if "memref.get_global @__stateful_rf_mem" in l]
    assert glob and "#map" in glob[0], glob
    assert getg and "#map" in getg[0], getg
    assert 'stateful_name = "mem"' in getg[0]


def test_partition_stateful_builds_and_runs_on_llvm():
    s = allo.customize(rf)
    s.partition("rf:mem", Partition.Complete)
    qa = np.zeros(N, np.uint16)
    s.build()(RA, WA, WD, WE, qa)  # was: failed to legalize memref.global
    assert qa.tolist() == WANT


def test_block_partition_stateful_builds_on_llvm():
    s = allo.customize(rf)
    s.partition("rf:mem", Partition.Block, factor=2)
    qa = np.zeros(N, np.uint16)
    s.build()(RA, WA, WD, WE, qa)
    assert qa.tolist() == WANT


def test_partition_stateful_hls_emission():
    for tgt in ("vhls", "catapult"):
        s = allo.customize(rf)
        s.partition("rf:mem", Partition.Complete)
        code = str(s.build(target=tgt))
        assert "__stateful_rf_mem" in code and "[32]" in code


def test_partition_stateful_systemc_register_pragma():
    s = df.customize(_region())
    s.partition("rf_0:mem", Partition.Complete)
    code = s.build(target="systemc").hls_code
    member = [l for l in code.splitlines() if "__stateful_rf_0_mem" in l and "@ Stateful" in l]
    assert member, code
    assert (
        '#pragma hls_resource __stateful_rf_0_mem_1_rsc variables="__stateful_rf_0_mem_1" '
        'map_to_module="[Register]"'
    ) in code, code


def test_unpartitioned_stateful_has_no_register_pragma():
    code = df.build(_region(), target="systemc").hls_code
    assert "hls_resource __stateful" not in code


def test_unknown_stateful_name_is_refused():
    s = allo.customize(rf)
    with pytest.raises(RuntimeError, match="not found"):
        s.partition("rf:nosuch", Partition.Complete)


if __name__ == "__main__":
    pytest.main([__file__])
