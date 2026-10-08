..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

############################################################################
MiniTPU-rtl M-R2b: MiniTPU's own DDR bridge inside the shim; cycles equal
############################################################################

.. note::

   **Dated measurement record, 2026-10-08.** zhang-21, branch
   ``minitpu-rtl-m1`` (worktree ``scratch/wt-mr1``) after merging
   ``origin/u1-pilot`` at ``bed18ed0``. Follows
   ``minitpu_rtl_m1_2026-10-08.rst`` section 5, which named the memory
   model as the whole cycle difference and proposed this. Same pins as M-R1
   (MiniTPU ``b3ba0a4d`` exported read-only, Verilator 5.052, g++ 13.3.1,
   ``oracle.json`` at ``round_trip_cycles`` 0). RTL-only: nothing under
   ``allo/``, ``mlir/`` or the MiniTPU tree changed.

**Verdict.** With MiniTPU's own ``uncore_io_tile`` (``dm_axi_bridge`` +
landing FIFO, unmodified) between the core and the banks, and an AXI4 slave
answering as the testbench's ``axi4_mem_model`` does at
``ROUND_TRIP_CYCLES=0``, **all 52 launches drain bit-identical (52/52) and
report exactly the testbench's ``perf_cnt_cycles`` (52/52, difference 0 on
every launch)**, issued bundles equal on 52/52. There is no residual
difference to explain. M-R1's direct memory, kept as ``--memory direct``,
still gives M-R1's cycles and digests exactly (regression of the port change
below).

Reproduce (env of ``reproduce.sh``)::

   examples/minitpu/rtl/reproduce.sh                                    # bridge (default): 52/52 bits, 52/52 cycles
   $ALLO_PYTHON examples/minitpu/rtl/run_kernel.py --all --memory direct   # M-R1's memory, for the "before" column

Wall: bridge 2 min 32 s (build 65 s, the 52 launches 81 s), direct 2 min 33 s
(build 67 s). Results: ``minitpu_rtl_m2b_2026-10-08/{results,run}_{bridge,direct}.{json,log}``.

1. What changed
===============

``gen_shim.py`` now emits two shims from one command/status/core section:

- ``rtl/minitpu_core_shim.sv`` (``memory="bridge"``, the default): the
  core's credit pipe goes to ``src/ddr/uncore_io_tile.sv`` (MiniTPU's, with
  ``dm_axi_bridge.sv`` and ``dma_landing_fifo.sv`` added to the file list;
  ``OUTSTANDING`` 2 and ``LANDING_DEPTH`` 8 by its defaults, as
  ``src/minitpu.sv`` instantiates it). Its AXI4 master meets a slave written
  in the shim after ``tb/minitpu_axi_mem_model.svh``: ``arready`` only with
  no read burst pending, the first R beat the cycle after the address, a beat
  per cycle, ``rlast`` on beat ``arlen``; a single-outstanding AW/W/B machine
  with ``bvalid`` the cycle after ``wlast``; ``in_range`` on
  ``(len + 1) << size`` with DECERR (``2'b11``) and zero data outside the
  32 MiB window; the address's low 32 bits, as the TB's 32-bit ``m_araddr``.
- ``rtl/minitpu_core_shim_direct.sv`` (``memory="direct"``): M-R1's memory.

**Where the bridge sits relative to the seam.** The seam is the core's
device-memory credit pipe (``dm_req_*``/``dm_rsp_*``), U4 track C's
memory-side contract (``u4_track_c_2026-10-08.rst`` section 10), unchanged
in fields, widths and order. The bridge sits **below** it: on the memory side
of the seam, between the core's DMA and the memory, exactly where
``src/minitpu.sv`` puts it (its loader/core phase mux is pass-through once
the IRAM loader is idle, which is the whole of ``perf_cnt_cycles``). So the
Allo DMA (track C) and the wrapped core still share one contract: either
can stand on the shim's side of the credit pipe.

**Two MemPorts per bank.** AXI reads and writes run concurrently (the bridge
can stream a load's R beats while a later store's W beats arrive), so each
bank is a write ``MemPort`` and a read ``MemPort`` bound to the *same* Allo
array (``CORE(cmd, st, b0, b0, b1, b1, ...)``), write port listed first. The
transactor commits ports in list order, so a read issued in the cycle of a
write to the same word returns the new data, as the TB model's next-cycle
combinational read would. Passing one boundary array twice to an
``RTLModule`` call works in the simulator (finding R1). Both shims use the
same 16 ports, so the region is the same for either memory.

**Read-ahead.** The TB model's ``rdata`` is combinational from its array in
the cycle a beat is shown; a ``MemPort`` answers a cycle after the request.
The slave therefore fetches beat 0 in the cycle ``AR`` is accepted (from
``araddr``) and beat k+1 in the cycle beat k is taken, which shows each beat
in the same cycle the model would. The one case where this could differ: a
write to the word *being shown* while ``rready`` is low (the model would show
the new value, the shim the old). No launch does that (a load and a store of
the same word in flight together is a program hazard); recorded, not
exercised.

2. Cycle table: testbench vs shim, before and after the bridge
==============================================================

``tb cycles`` and the digest are ``oracle.json``'s (``tb_kernel_image``);
``direct`` is M-R1's memory, ``bridge`` this one; every digest is the
testbench's in both forms::

   launch                     tb cycles  direct (M-R1)   diff  bridge (M-R2b)  diff  digest    verdict
   gemm_structured                 1350           1348     -2            1350    +0  886e377e  identical
   gemm_varying                    1350           1348     -2            1350    +0  aab61c80  identical
   gemm_varying_gelu               2542           2540     -2            2542    +0  d94353e8  identical
   gemm_c4g2_fuse                  6992           6987     -5            6992    +0  5f0e471e  identical
   gemm_c4g2                       8270           8253    -17            8270    +0  54fb566a  identical
   softmax_8                        734            731     -3             734    +0  f2ed46f3  identical
   layernorm_4                     3801           3789    -12            3801    +0  c9f5b422  identical
   layernorm_32_packed            31069          32444  +1375           31069    +0  b2215a09  identical
   add                            10975          10951    -24           10975    +0  48f27937  identical
   add_bias                        9626           9588    -38            9626    +0  f627505f  identical
   add_bias_gelu_3072             40386          40236   -150           40386    +0  4858ad7c  identical
   softmax_packed_12              18918          18882    -36           18918    +0  a9af09ae  identical
   softmax_packed_2_span128        5354           5350     -4            5354    +0  ddcf96c4  identical
   rmsnorm_4_w896                  3073           3061    -12            3073    +0  6ea18856  identical
   rmsnorm_32_w896_packed         25781          27412  +1631           25781    +0  a07a1071  identical
   rope_2                          2125           2119     -6            2125    +0  7acc77b0  identical
   swiglu_2                        2739           2735     -4            2739    +0  1c8d2101  identical
   gqa_2_2.0                       4400           4381    -19            4400    +0  b22cd595  identical
   gqa_2_2.1                       4402           4383    -19            4402    +0  3c0eeb26  identical
   head_scatter.0                  1973           1925    -48            1973    +0  9cd1c2da  identical
   head_scatter.1                  2914           2842    -72            2914    +0  e45dfc07  identical
   head_scatter.2                  1973           1925    -48            1973    +0  ac3db020  identical
   head_scatter.3                  1459           1435    -24            1459    +0  6b84c21d  identical
   head_scatter.4                  2065           2047    -18            2065    +0  35692694  identical
   head_scatter.5                  1461           1437    -24            1461    +0  8df3ea60  identical
   loop_begin_r.0                    23             22     -1              23    +0  7bef7079  identical
   loop_begin_r.1                    25             24     -1              25    +0  7bef7079  identical
   loop_begin_r.2                    47             44     -3              47    +0  417ce8b3  identical
   loop_begin_r.3                    49             46     -3              49    +0  417ce8b3  identical
   loop_begin_r.4                   153            142    -11             153    +0  19a93954  identical
   loop_begin_r.5                   155            144    -11             155    +0  19a93954  identical
   loop_begin_r.6                   465            430    -35             465    +0  5d408de1  identical
   loop_begin_r.7                   467            432    -35             467    +0  5d408de1  identical
   loop_begin_r.8                   350            325    -25             350    +0  0badf96c  identical
   loop_begin_r.9                    32             31     -1              32    +0  7bef7079  identical
   loop_begin_r.10                   24             23     -1              24    +0  7bef7079  identical
   loop_begin_r.11                  980            907    -73             980    +0  375007ec  identical
   loop_begin_r.12                   32             31     -1              32    +0  7bef7079  identical
   loop_begin_r.13                   56             53     -3              56    +0  417ce8b3  identical
   loop_begin_r.14                  104             97     -7             104    +0  c9d5b0c5  identical
   loop_begin_r.15                  128            119     -9             128    +0  0d9620e5  identical
   loop_begin_r.16                  128            119     -9             128    +0  0d9620e5  identical
   loop_begin_r.17               113617         104878  -8739          113617    +0  e7d966e9  identical
   loop_begin_r.18               113617         104878  -8739          113617    +0  e7d966e9  identical
   flash_attention.0              10081          10091    +10           10081    +0  5a961446  identical
   flash_attention.1              23892          23904    +12           23892    +0  910f3b03  identical
   flash_attention.2              30473          30479     +6           30473    +0  ffa7692e  identical
   flash_attention.3              22380          22390    +10           22380    +0  1e92158d  identical
   flash_attention.4              13949          13967    +18           13949    +0  97ea2f6b  identical
   flash_attention.5               3941           3951    +10            3941    +0  75c96a6c  identical
   flash_attention.6              23441          23459    +18           23441    +0  74cd6fcc  identical
   flash_attention.7              29597          29615    +18           29597    +0  03e5e469  identical

   bridge: 52/52 drains bit-identical; cycles equal to the TB on 52/52
   direct: 52/52 drains bit-identical; cycles equal to the TB on 0/52 (-8,739 .. +1,631; M-R1 section 5)

Status words on the bridge run: ``done`` seen and ``dma_rsp_seen`` on all 52,
``dma_err`` 0, no DECERR, no partial ``wstrb``, no ``$readmem`` warning,
``bridge_to_seen`` (the bridge watchdog) 0.

**Residual differences: none.** What makes that possible, stated so a future
difference can be placed: ``perf_cnt_cycles`` counts from ``start`` to
``done`` inside the core, so the IRAM load path (the TB's ``iram_loader``
over DDR vs the shim's ``IRAM`` command words) and the AXI-Lite command path
are outside it; everything inside it that is not the core -- the bridge, the
landing FIFO, the AXI memory -- is now MiniTPU's own RTL or a cycle-exact
transcription of the TB's model at round trip 0.

3. Findings
===========

R1. **match** -- one boundary array bound to two ``MemPort``\ s of the same
    ``RTLModule`` call (``b0, b0``) runs in the simulator and gives a
    dual-ported RAM whose same-cycle order is the port order. This is
    undocumented behaviour of the transactor (``rtl.py`` ``_emit`` commits
    memory ports in list order); it is what makes a write-first true dual
    port expressible today. **Missing abstraction**: a ``MemPort`` with
    separate read and write ports (or an explicit ``same_array=`` /
    ``order=``) so this does not rest on argument aliasing.
R2. **match** -- MiniTPU's DDR path (three more RTL files, unmodified) drops
    into the ``RTLModule`` file list with no flag change; the build stays
    warning-free and ``--lint-only`` clean.
R3. Not modelled: ``ROUND_TRIP_CYCLES > 0`` (the TB model's ``due``
    queue of up to 8 read bursts). The oracle was taken at 0; a round-trip
    knob in the slave (a delay line on ``rvalid``/``bvalid`` and an 8-deep AR
    queue) is the next step if the board-like setting (about 40) is wanted.

4. The fold into ``examples/minitpu/rtl/``
==========================================

After the run above, ``examples/minitpu-rtl/`` was moved to
``examples/minitpu/rtl/`` (one design tree, checkpoint 24's layout call; a
``git mv``, the generated shims' header line and ``reproduce.sh``'s root
depth the only content changes; paths in README D-21, ``rtl_module.rst`` and
the ``minitpu_rtl_*`` records rewritten; ``oracle.json``'s ``schema`` id
``allo/minitpu-rtl/oracle/1`` kept, being an identifier and not a path).
``examples/minitpu/rtl/reproduce.sh --all``, run from ``/tmp``: shims up to
date, **52/52 bit-identical, cycles equal on 52/52**, 2 min 45 s (build 73 s).
