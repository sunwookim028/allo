..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

##########################################################################
MiniTPU FPGA, inverse angle: the Allo MXU through Vitis HLS (D-21 probe)
##########################################################################

.. note::

   **Dated measurement record, 2026-10-08.** zhang-21, branch
   ``minitpu-fpga-mxu`` from ``u1-pilot`` at ``20937d9e`` (worktree
   ``scratch/wt-fpga``, its own bindings: Ninja, gcc-toolset-13 13.3.1,
   ``LLVM_DIR``/``MLIR_DIR`` from ``/work/shared/common/llvm-project-main/build-rhel8``,
   ``nice ninja -j16``, 336 steps). Vitis HLS 2023.2 (Build 4023990),
   part ``xczu7ev-ffvc1156-2-e`` (ZCU104), clock 5 ns (200 MHz, MiniTPU's
   ``PL_CLK_MHZ`` default); Verilator 5.052 (``--binary --timing``, the unit
   suite's flags), g++ 13.3.1; Vivado 2023.2. MiniTPU ``b3ba0a4d`` read only
   (its tbs run from ``/work/shared/users/phd/sk3463/minitpu`` unchanged);
   the bitstream builds run in a local clone (``scratch/minitpu-fpga``, no
   remote, never pushed). The Catapult half of this angle is
   ``minitpu_rtl_mxu_2026-10-08.rst``; its form and wrapper are the starting
   point here. Nothing under ``allo/`` or ``mlir/`` changed. Owner away:
   every call is provisional (D-9).

.. contents::
   :local:
   :depth: 1

Reproduce (``source examples/minitpu/harness/env-zhang21.sh``; worktree root;
``export PYTHONPATH=$PWD``; ``R=dev/records/minitpu/minitpu_fpga_mxu_2026-10-08``,
``S`` a scratch directory, ``T`` the read-only MiniTPU clone at ``b3ba0a4d``,
``P`` the partition flags in ``$R/logs/vitis/partitions.txt``)::

   # pass 1: emit + csynth (schedules); ~75 s at DIM 2
   $ALLO_PYTHON examples/minitpu/fpga/vhls_build.py $R/forms/mxu_wide.py $S/v2q8 --inst dim2 --depth 8 \
       --partition mxu_back_w_0:mem
   # link depths from the pass-1 schedule (section 2), then pass 2 with them and the partitions
   python3 examples/minitpu/fpga/balance_depths.py $S/v2q8 --out $S/v2_bal.tcl
   $ALLO_PYTHON examples/minitpu/fpga/vhls_build.py $R/forms/mxu_wide.py $S/v2c --inst dim2 $P --tcl-file $S/v2_bal.tcl
   # MiniTPU's tb unchanged, then the measuring copy (Verilator 5.052, under scl enable gcc-toolset-13)
   $R/scripts/run_tb.sh $T $S/v2c $T/tb/tb_mxu_single_port.sv $S/tb_v2c
   $R/scripts/run_tb.sh $T $S/v2c $R/tb/tb_mxu_single_port_lat.sv $S/tb_v2c +define+MRTL_DIM=2 +define+MRTL_ALLO

Files: ``examples/minitpu/fpga/`` holds the flow (``vhls_build.py``: Allo
``target="vhls"`` emission + the flow's own ``run.tcl``; ``balance_depths.py``;
``ooc_synth.tcl``; ``board_build.sh``). The record directory holds
``forms/mxu_wide.py`` (copied unchanged from the Catapult record),
``scripts/`` (``gen_mxu_vhls.py``, the wrapper generator adapted to Vitis
handshakes; ``run_tb.sh``; ``gen_isa_delta.py``), ``tb/`` (the measuring tb
copy), ``rtl/`` (generated wrappers), ``logs/`` (csynth reports, the
flow's ``run.tcl`` per build, balance tables, tb outputs, Vivado reports).

1. Vitis HLS of the wide form
=============================

``s.build(target="vhls", mode="csyn", configs={"device": "zcu104",
"frequency": 200})`` emits ``kernel.cpp`` (6,449 lines at DIM 2) with a
``#pragma HLS dataflow`` top whose ten region arrays (``UInt(1)[N]`` ...
``UInt(16 D SUB)[N]``) are plain C arrays and whose grid links are
``hls::stream`` with the ``Channel``'s depth. Allo's own ``run.tcl`` (kept as
``run.tcl.allo``) has no interface directives; for ``vitis_hls`` Allo would
wrap every argument in a local buffer (``wrap_io``), i.e. a batch kernel. So
the flow writes its own ``run.tcl``: the emitted C++ verbatim, and per region
array ``set_directive_interface -mode ap_fifo`` (each is read or written once
per iteration, in order: Vitis implements it as a stream with HLS 214-142's
"may cause mismatch if ... not in sequential order" warning, which holds here)
plus ``ap_ctrl_none`` on the top -- a free-running kernel, no
``ap_start``/``ap_done``. Ports come out ``<v>_dout/_empty_n/_read`` and
``<v>_din/_full_n/_write``, ``ap_clk``, ``ap_rst`` (active high). The
schedule is the Catapult record's (``--pipeline-all --unroll-inner``): every
kernel's iteration loop ``s.pipeline``, every inner loop ``s.unroll``.
Vitis ran in 61-69 s per DIM 2 build.

.. list-table:: DIM 2, 5 ns, every kernel's iteration loop (``csynth.rpt``)
   :header-rows: 1
   :widths: 30 14 14 14 28

   * - partitions
     - front II (it. lat.)
     - PE II (it. lat.)
     - back II (it. lat.)
     - notes
   * - none (``v2``)
     - 1 (2)
     - 1 (7 row 0, 9 row 1)
     - **3** (4)
     - HLS 200-880/885: ``mem`` carried dependence and "limited memory
       ports (II = 2)"; five front and six back initializer loops run as
       sequential pipelines before the main loop (43 and 22 cycles)
   * - ``mxu_back_w_0:mem`` (``v2q*``, ``v2b``)
     - 1 (2)
     - 1 (7 / 9)
     - **1** (4)
     - initializer loops still sequential (the startup backlog of section 2)
   * - ``mem`` + front ``skd skv cbs wss hdata`` + back ``gidx gq rd wr
       cnt`` (``v2c``, the build kept)
     - 1 (2)
     - 1 (7 / 9)
     - 1 (4)
     - no initializer loops left; HLS estimate 6,639 FF / 16,548 LUT, 0
       BRAM, 0 DSP; estimated slack -0.35 (PE) / -1.55 ns (back) against
       Vitis's 3.65 ns budget (5 ns less its default 1.35 ns uncertainty)

**Vitis refuses II 1 exactly where Catapult did** (the back's ``mem`` ring,
one array for every lane's 16 entries, two accesses per lane per iteration),
and needs the same ``s.partition``; it did **not** need the front's arrays
partitioned for II 1 at DIM 2 (Catapult did at DIM 4), but without them each
array initializer (``hdata: UInt(16)[D] = 0`` emits a C loop) becomes its own
sequential pipeline before the iteration loop, which section 2 shows is a
permanent token offset. Partitioning them is a schedule on the unit, as in
the Catapult record's finding 1. Vivado out-of-context synthesis of ``v2c``
(``ooc_synth.tcl``, 2 min) at 5 ns: **WNS +2.401 ns** (the HLS slack
estimate is pessimistic), 7,864 LUT / 5,311 FF / 0 DSP in all; per PE
**411-419 LUT** in row 0 (no psum input, ``ROW0_PSUM_ZERO``) and **677-684
LUT** in row 1, ~645 FF, 0 DSP; the back 5,032 LUT; the front 124.

The DIM 16 build is in section 3.

2. The token rate on Vitis RTL: link depths from the schedule
=============================================================

The wrapper (``gen_mxu_vhls.py``) is the Catapult record's ``mxu_allo.sv``
with the core's ports changed: an In port's ``_dout`` is the wrapper FIFO's
head, ``_empty_n`` its non-empty, ``_read`` its pop; an Out port's ``_full_n``
is tied high (always ready) and ``_write`` is the token's valid;
``ap_rst = !rst_ni``; no ``ap_start``/``ap_done`` (``ap_ctrl_none``). The token
FIFOs, the pop mask, the lag probes and the lockstep-violation flag are
unchanged. ``tb_mxu_single_port`` (MiniTPU's, unchanged, DIM 2) **passed on
the first build** against it.

The rate is the composition's, as on Catapult, but the cause differs.
Catapult made each kernel latency 1, so only the link depth mattered (depth 4
gave one token per cycle at DIM 2). Vitis's PE is a 7-9-stage pipeline that
reads its links in state 2 and writes them in its last state; the front
writes ``lhsx[1, 0]`` in the same cycle it writes ``wx[0, 0]``, but
``PE(1, 0)`` cannot read token *t* of it until ``PE(0, 0)``'s south output
for *t* arrives 7 cycles later. Every reconvergent pair of paths (front ->
``PE(r, 0)`` against the grid; ``px[D, 0]`` against ``px[D, D-1]`` into the
back; ``ctl`` against the whole grid) needs depth for the difference, and a
uniform depth is the wrong lever:

.. list-table:: DIM 2, ``tb_mxu_single_port_lat`` (``logs/tb/*.txt``)
   :header-rows: 1
   :widths: 30 12 12 14 32

   * - links
     - push->valid
     - pop->next valid
     - lag (first tile -> +2,000 idle)
     - verdict
   * - uniform depth 2 (the form's ``pe_channels``; ``v2q2``)
     - 145
     - 730
     - 133 -> wrapper FIFO full
     - **FAIL by the wrapper** (lockstep violation), tiles exact
   * - uniform depth 4 (``v2q4``)
     - 97
     - 303
     - 85 -> 981 at +1,000 -> full
     - **FAIL by the wrapper**: 0.36 token/cycle
   * - uniform depth 8 (``v2q8``)
     - 67
     - 107
     - 55 -> 491 -> 851
     - PASS on the check, **0.64 token/cycle** (violation by rate)
   * - per-link (``balance_depths.py``), arrays not partitioned (``v2b``)
     - 55
     - 44
     - 43, 43, 43, 43, 43 (``acc_lag`` 17)
     - PASS, **one token per cycle**; the 17 is the front's initializer
       pipelines, a backlog an II-1 front never recovers
   * - per-link + every array partitioned (``v2c``, kept)
     - **39**
     - **28**
     - **27**, 27, 27, 27, 27 (``acc_lag`` **2**)
     - PASS, **one token per cycle**, stationary
   * - ``mxu.sv`` (MiniTPU's own, same tb copy)
     - 11
     - 0
     - --
     - PASS (DIM 16: push->valid 81, pop->next valid 0)

``balance_depths.py`` reads each process's ``ap_fifo`` read and write states
from Vitis's own ``*.verbose.sched.rpt`` (the reports name the top's streams
directly), propagates each process's steady-state start
``A_dst = max(A_src + w_src + 1 - r_dst)`` through the link graph, and sizes
each link at the tokens it holds, ``(A_dst + r_dst) - (A_src + w_src)``, plus
one for ``full_n`` and a margin of one; it writes
``set_directive_stream -depth`` lines (a tcl directive overrides the emitted
``#pragma HLS stream depth``: the RTMG line says ``fifo_w32_d8`` for a
directive of 8 on a pragma of 2). At DIM 2: 13 links, depths 3-25, 3,360
FIFO bits; ``ctl`` needs **25** where the form's ``CTLD`` is 16. The model
predicts the back's start at A = 23 and its writes at state 4, i.e. the
measured lag of 27. The six unused streams Allo declares (the grid's edge
links with ``EDGE_OUT`` 0) have no endpoint and are dropped by Vitis.

**What the version says at DIM 2**: push->valid 39 (``mxu.sv`` 11, +28),
pop->next valid 28 (``mxu.sv`` 0, +28; the pop mask is the lag plus one),
``acc_lag`` 2 (``input_accept_o``'s lag; it feeds only
``mxu_stream_engine``'s ``ifndef SYNTHESIS`` assertion, so it cannot change
hardware behaviour), FIFO bits as above.

.. In progress: section 3 (DIM 16, the bitstream) and section 4 (the ISA version) follow in later commits.
