# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The Catapult latency manifest and the latency pin (allo/backend/catapult.py).

Fixtures are trimmed from two real Catapult 2024.2 runs of MiniTPU's acc24
adder (dev/records/minitpu/latency_report_2026-10-02.rst): pinned to 3 at
3.33 ns (measured 3 in Verilator), and with the leading-zero loop left rolled
at 5 ns (measured 19 cycles per vector, while the schedule says II=1).
"""
import os

import pytest

from allo.backend.catapult import catapult_latency_manifest, io_latency_tcl

PINNED = (
    "Processes/Blocks in Design\n  Process        Real Operation(s) count Latency Throughput Reset Length II Comments\n  -------------- ----------------------- ------- ---------- ------------ -- --------\n  /top/add_0/run                     256      -1          1       352120  0\n  Design Total:                      256      -1          1       352120  0\n\nClock Information\n  Clock Signal Edge   Period Sharing Alloc (%) Uncertainty Used by Processes/Blocks\n  ------------ ------ ------ ----------------- ----------- ------------------------\n  clk          rising  3.330             20.00    0.000000 /top/add_0/run\n\nLoops\n  Process        Loop             Iterations C-Steps Total Cycles  Duration  Unroll Init     Comments\n  -------------- ---------------- ---------- ------- ------------- --------- ------ ---- ------------\n  /top/add_0/run run:rlp            Infinite       1       352121   1.17 ms\n  /top/add_0/run  l_S_i_0_i           352116       4      (352119) (1.17 ms)           1 reset action\n  /top/add_0/run  while             Infinite       1            1   3.33 ns\n\n",
    "directive set /top/add_0/run/run:rlp/l_S_i_0_i/v10.Pop() CSTEPS_FROM {{.. == 0}}\ndirective set /top/add_0/run/run:rlp/l_S_i_0_i/v11.Pop() CSTEPS_FROM {{.. == 0}}\ndirective set /top/add_0/run/run:rlp/l_S_i_0_i/v12.Push() CSTEPS_FROM {{.. == 3}}\n",
    "# $MGC_HOME/shared/include/ac_sc.h(105): Loop '/Connections::OutBlocking<ac_int<32,false>,Connections::SYN_PORT>::Push/core/to_sc<32>:for' iterated at most 1 times. (LOOP-2)\n# $MGC_HOME/shared/include/ac_sc.h(75): Loop '/Connections::InBlocking<ac_int<32,false>,Connections::SYN_PORT>::Pop/core/to_ac<32>:for' iterated at most 1 times. (LOOP-2)\n# $PROJECT_HOME/../kernel.cpp(1239): Loop '/top/add_0/run/while' is left rolled. (LOOP-4)\n# $PROJECT_HOME/../kernel.cpp(477): Loop '/add_0/run/l_S_offset_0_offset' iterated at most 19 times. (LOOP-2)\n# $PROJECT_HOME/../kernel.cpp(516): Loop '/add_0/run/l_S_i_0_i' iterated at most 352116 times. (LOOP-2)\n# $PROJECT_HOME/../kernel.cpp(516): Loop '/top/add_0/run/l_S_i_0_i' is left rolled. (LOOP-4)\n",
)
ROLLED = (
    "Processes/Blocks in Design\n  Process        Real Operation(s) count Latency Throughput Reset Length II Comments\n  -------------- ----------------------- ------- ---------- ------------ -- --------\n  /top/add_0/run                     190      -1          1      6690206  0\n  Design Total:                      190      -1          1      6690206  0\n\nClock Information\n  Clock Signal Edge   Period Sharing Alloc (%) Uncertainty Used by Processes/Blocks\n  ------------ ------ ------ ----------------- ----------- ------------------------\n  clk          rising  5.000             20.00    0.000000 /top/add_0/run\n\nLoops\n  Process        Loop             Iterations C-Steps Total Cycles   Duration  Unroll Init     Comments\n  -------------- ---------------- ---------- ------- ------------- ---------- ------ ---- ------------\n  /top/add_0/run run:rlp            Infinite       1      6690207   33.45 ms\n  /top/add_0/run  l_S_i_0_i          6690204       2     (6690205) (33.45 ms)           1 reset action\n  /top/add_0/run  while             Infinite       1            1    5.00 ns\n\n",
    "directive set /top/add_0/run/run:rlp/l_S_i_0_i/v10.Pop() CSTEPS_FROM {{.. == 0}}\ndirective set /top/add_0/run/run:rlp/l_S_i_0_i/v11.Pop() CSTEPS_FROM {{.. == 0}}\ndirective set /top/add_0/run/run:rlp/l_S_i_0_i/v12.Push() CSTEPS_FROM {{.. == 1}}\n",
    "# $MGC_HOME/shared/include/ac_sc.h(105): Loop '/Connections::OutBlocking<ac_int<32,false>,Connections::SYN_PORT>::Push/core/to_sc<32>:for' iterated at most 1 times. (LOOP-2)\n# $MGC_HOME/shared/include/ac_sc.h(75): Loop '/Connections::InBlocking<ac_int<32,false>,Connections::SYN_PORT>::Pop/core/to_ac<32>:for' iterated at most 1 times. (LOOP-2)\n# $PROJECT_HOME/../kernel.cpp(1238): Loop '/top/add_0/run/while' is left rolled. (LOOP-4)\n# $PROJECT_HOME/../kernel.cpp(476): Loop '/add_0/run/l_S_offset_0_offset' iterated at most 19 times. (LOOP-2)\n# $PROJECT_HOME/../kernel.cpp(476): Loop '/top/add_0/run/l_S_offset_0_offset' is left rolled. (LOOP-4)\n# $PROJECT_HOME/../kernel.cpp(515): Loop '/add_0/run/l_S_i_0_i' iterated at most 352116 times. (LOOP-2)\n# $PROJECT_HOME/../kernel.cpp(515): Loop '/top/add_0/run/l_S_i_0_i' is left rolled. (LOOP-4)\n",
)


# A Wire-port kernel (MiniTPU's vpu_word_array, `wire` form, both shift loops
# unrolled; dev/records/minitpu/u2_word_array_2026-10-02/catapult/wire_unr_n64_3p33):
# the process latency counts the reset c-step (2), the RTL measures 1.
WIRE_UNROLLED = (
    "Processes/Blocks in Design\n  Process       Real Operation(s) count Latency Throughput Reset Length II Comments\n  ------------- ----------------------- ------- ---------- ------------ -- --------\n  /wa_0/run                          76       2          1            1  0\n  Design Total:                      76       2          1            1  1\n\nClock Information\n  Clock Signal Edge   Period Sharing Alloc (%) Uncertainty Used by Processes/Blocks\n  ------------ ------ ------ ----------------- ----------- ------------------------\n  clk          rising  3.330             20.00    0.000000 /wa_0/run\n\nLoops\n  Process   Loop             Iterations C-Steps Total Cycles  Duration  Unroll Init Comments\n  --------- ---------------- ---------- ------- ------------- --------- ------ ---- --------\n  /wa_0/run run:rlp            Infinite       1            2   6.66 ns\n  /wa_0/run  while             Infinite       1            1   3.33 ns            1\n\n",
    # a Wire kernel's cycle_set.tcl has no Pop/Push: sc_in reads are not scheduled ops
    "directive set /wa_0/run/while/mem:rsc.@ CSTEPS_FROM {{.. == 0}}\ndirective set /wa_0/run/while/v40.write() CSTEPS_FROM {{.. == 0}}\n",
    "# $PROJECT_HOME/../kernel.cpp(575): Loop '/wa_0/run/l_S_k_0_k' iterated at most 2 times. (LOOP-2)\n# $PROJECT_HOME/../kernel.cpp(570): Loop '/wa_0/run/while' is left rolled. (LOOP-4)\n# Info: $PROJECT_HOME/../kernel.cpp(570): Loop '/wa_0/run/while' is pipelined with initiation interval 1 and no flushing (SCHD-43)\n",
)
# The same kernel with the shift loop left rolled (wire_n64_3p33): Catapult
# merges it into the steady-state loop ("Prescheduled LOOP '/wa_0/run/while'
# (2 c-steps) (SCHD-7)"), so the number is flagged, not reported.
WIRE_ROLLED = (
    WIRE_UNROLLED[0].replace("/wa_0/run                          76       2          1            1  0", "/wa_0/run                          70       3          2            1  0").replace("while             Infinite       1            1   3.33 ns            1", "while             Infinite       2            3   9.99 ns            1"),
    WIRE_UNROLLED[1],
    WIRE_UNROLLED[2] + "# $PROJECT_HOME/../kernel.cpp(575): Loop '/wa_0/run/l_S_k_0_k' is left rolled. (LOOP-4)\n",
)


def _sol(tmp_path, fx):
    sol = tmp_path / "top.v1"
    sol.mkdir()
    (sol / "cycle.rpt").write_text(fx[0])
    (sol / "cycle_set.tcl").write_text(fx[1])
    (tmp_path / "catapult.log").write_text(fx[2])
    return str(sol), str(tmp_path / "catapult.log")


def test_pinned_latency_is_read_back(tmp_path):
    sol, log = _sol(tmp_path, PINNED)
    u = catapult_latency_manifest(sol, log, declared={"add_0": 3})["units"]["add_0"]
    assert (u["latency"], u["ii"], u["status"]) == (3, 1, "scheduled")
    assert u["io"] == {"v10.Pop()": 0, "v11.Pop()": 0, "v12.Push()": 3}
    assert u["declared"] == 3


def test_rolled_inner_loop_is_flagged_not_reported(tmp_path):
    sol, log = _sol(tmp_path, ROLLED)
    u = catapult_latency_manifest(sol, log)["units"]["add_0"]
    assert u["status"] == "unreliable"
    assert "l_S_offset_0_offset" in u["reason"]
    assert u["cycles_per_vector_max"] == 19


def test_wire_kernel_latency_excludes_the_reset_step(tmp_path):
    sol, log = _sol(tmp_path, WIRE_UNROLLED)
    u = catapult_latency_manifest(sol, log)["units"]["wa_0"]
    assert u["port_style"] == "wire"
    assert (u["latency"], u["ii"], u["status"]) == (1, 1, "scheduled")  # was 2
    assert u["process_latency"] == 2 and u["loop_c_steps"] == 1


def test_wire_kernel_with_a_merged_rolled_loop_is_flagged(tmp_path):
    sol, log = _sol(tmp_path, WIRE_ROLLED)
    u = catapult_latency_manifest(sol, log)["units"]["wa_0"]
    assert u["status"] == "unreliable" and "l_S_k_0_k" in u["reason"]
    assert u["latency"] == 2  # the merged loop's 2 c-steps, not the process's 3


KERNEL = """SC_MODULE(add_0) {
  Connections::In< ac_int<32, false> > v10;
  Connections::In< ac_int<32, false> > v11;
  Connections::Out< ac_int<32, false> > v12;
};
SC_MODULE(w_0) {
  sc_in< ac_int<32, false> > a;
  sc_out< ac_int<32, false> > c;
};
"""


def test_pin_emits_io_constraints():
    lines = io_latency_tcl(KERNEL, {"add_0": 3})
    assert [l.split("  ;#")[0] for l in lines] == [
        "cycle set {v12.Push()} -from {v10.Pop()} -equal 3",
        "cycle set {v12.Push()} -from {v11.Pop()} -equal 3",
    ]


@pytest.mark.parametrize(
    "pins, msg", [({"add_0": 0}, "least is 1"), ({"w_0": 2}, "Wire")]
)
def test_pin_refuses_what_catapult_cannot_honour(pins, msg):
    with pytest.raises(ValueError, match=msg):
        io_latency_tcl(KERNEL, pins)
