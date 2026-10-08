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

**The form variant ``forms/mxu_wide_v.py``.** ``mxu_wide.py``'s back reads
``mem[lane * ENTRIES + rd[lane]]`` with an ``int32`` pointer; Vitis cannot
bound that index, so with ``mem`` partitioned complete each lane's head read
is a ``D * ENTRIES``:1 mux of 64-bit words (``sparsemux_65_5_64`` at DIM 2,
256:1 at DIM 16, where the back alone had not finished scheduling after more
than an hour). The variant types the rings for the mux the RTL means --
``mem: UInt(64)[D, ENTRIES]``, ``rd``/``wr: UInt(4)[D]``, ``cnt: UInt(5)[D]``
-- and is otherwise ``mxu_wide.py`` verbatim (same function: the pointers
wrap at ``ENTRIES - 1`` as before). At DIM 2 (``v2w``): every kernel II 1,
the back's iteration latency 3 (was 4); Vivado OOC: the back **1,497 LUT**
(was 5,032), PEs unchanged (413-422 / 688-689), WNS +2.401 ns. It passes
both tbs with one token per cycle (section 2). The DIM 16 builds use it.

**Area levers tried at DIM 2** (tcl only, ``logs/vitis/run_v2a*.tcl``,
Vivado OOC per PE, row 0 / row 1): ``config_op mul -impl dsp`` 355-359 /
606-613 LUT and 1 DSP; plus ``config_compile -pipeline_style stp``: no
change; ``config_op add -impl dsp`` (every adder) 349-350 / 574-580 LUT and
10-12 DSP. MiniTPU's own PE (``mxu_pe.sv``) is 392 LUT and 2 DSP
(``use_dsp`` on the 8 x 8 mantissa product and the 20-bit magnitude add;
section 3's baseline). The Vitis PE costs 1.55-1.75 x the LUTs whatever
these levers do.

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
   * - the variant ``mxu_wide_v`` (``v2w``), per-link
     - **38**
     - **27**
     - **26**, 26, 26, 26, 26 (``acc_lag`` **2**)
     - PASS, **one token per cycle**, stationary (the back is one state
       shorter); ``tb_mxu_single_port`` PASS
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

3. DIM 16: the build, the rate, the area
=========================================

**Allo.** ``mxu_wide_v`` at DIM 16 (258 processes, 769 links, 48 unused
streams): customize 86 s, 260 ``s.unroll`` 69-85 s, 258 ``s.pipeline``
67-86 s, emission 22 s -- **244-280 s**, ``kernel.cpp`` 362,861 lines. The
same with the eleven partitions as ``s.partition`` had not left the first
partition's use-def walk after 49 min (finding 4); the build uses the tcl
partitions.

**Vitis HLS.** 3,073 s (pass 1, ``v16v``) and 3,965 s (``v16w``, with the
depths) on one core, ~11 GB. **Every kernel II 1**: front depth 2, 16 row-0
PEs depth 7, 240 PEs depth 9, back depth 3. HLS estimate 339,319 FF /
752,620 LUT (326 %; the estimate runs ~3 x Vivado's here). The original form
(``mxu_wide.py``, ``v16t``) scheduled the front and all 256 PEs at II 1 in
under 45 min and then spent **over 2 h 14 min** in the back's scheduling
(its 256:1 head muxes, section 1) without finishing; it was stopped
(``logs/vitis/v16t_csynth_stopped.log.gz``).

**Depths.** ``balance_depths.py --stages-from`` the DIM 2 build (process
kinds: front, back, PE row 0, PE) predicted the DIM 16 depths before any DIM
16 csynth; pass 2 (``v16w``) used them and pass 1's own schedule gives
**DEPTHS-MATCH** on all 769 links. Depths 3-249 (``ctl`` 249; the front's
links to ``PE(r, 0)`` 8 r + 1; ``px[16, c]`` into the back 123 - 8 c),
5,643 slots, **235,872 FIFO bits**; the back starts at A = 247 cycles.

**Vivado out-of-context synthesis of the DIM 16 core at 5 ns** (``v16w``,
``ooc_synth.tcl``, 14 min, 6.1 GB, 8 threads):

.. list-table:: the MXU alone, Vivado 2023.2 synthesis, xczu7ev-ffvc1156-2-e
   :header-rows: 1
   :widths: 30 14 14 10 10 22

   * - MXU
     - LUT
     - FF
     - DSP
     - BRAM
     - timing at 5 ns
   * - MiniTPU's ``mxu.sv`` (``i_mxu`` in the b3ba0a4d synth-check)
     - 100,261
     - 48,386
     - 512
     - 0
     - (the shipped bitstream: WNS +0.054 post-route, whole design)
   * - Allo/Vitis ``mxu_widev_dim16`` (OOC)
     - **213,429** (192,180 logic, 21,249 LUTRAM/SRL)
     - 256,332
     - 0
     - 0.5
     - **WNS +2.332 ns**, 0 failing endpoints
   * - of which the 256 PEs / back / front
     - ~174,000 (max 704 per PE) / 5,675 / 662
     - -- / 18,009 / 1,091
     - 0
     - 0
     -

The rest of MiniTPU synthesizes to **100,961 LUT** (201,222 for the whole
b3ba0a4d design less its ``i_mxu``; ``logs/vivado/base_b3ba0a4d_*``), so the
MXU's room on the XCZU7EV is at most **129,439 LUT at 100 %**; the Allo MXU
needs 213,429, i.e. the design would be **~314 K LUT, 136 % of the part**.
The ZCU102's XCZU9EG (274,080 LUT; MiniTPU's flow also accepts ``zcu102``)
would be at ~115 %. The DSP lever (section 1) saves ~17 K LUT at DIM 16 and
does not change the verdict. **Timing closes; area does not.**

.. In progress: the DIM 16 rate (initializer fix), the board flow, section 4 (the ISA version).
