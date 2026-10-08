..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

#################################################################################
MiniTPU-rtl M-R0: the oracle, the standalone core build, MemPort in the simulator
#################################################################################

.. note::

   **Dated measurement record, 2026-10-08.** zhang-21, branch
   ``minitpu-rtl-m0`` (worktree ``scratch/wt-mr0``) from ``origin/u1-pilot`` at
   ``32448516``. Plan: ``minitpu_rtl_plan_2026-10-08.rst`` section 4, M-R0.
   MiniTPU read-only at ``/work/shared/users/phd/sk3463/minitpu`` @
   ``b3ba0a4d4fb69d39091c55f5f00d1f237082a4f1`` (clean); nothing was written
   into it: its tracked tree was exported with ``git archive`` into
   ``scratch/mr0/minitpu-b3ba0a4d`` and every run used that export. Host load
   80-135 on 8 cores throughout (other sessions), so wall times are upper
   bounds. Nothing under ``allo/`` or ``mlir/`` changed on this branch.

Pins
====

.. list-table::
   :header-rows: 1

   * - What
     - Pin
   * - MiniTPU
     - ``b3ba0a4d4fb69d39091c55f5f00d1f237082a4f1``; ``git archive --format=tar
       b3ba0a4d`` sha256 ``931257f59d7e20f3863f7fcae10e7be67c8b46a11ab1ae4e68a540c62e095124``;
       ``src/core/core.f`` sha256 ``9f133018...52e8``
   * - Verilator
     - 5.052 ``2026-09-05 rev conda-forge build``, conda
       ``verilator-5.052-py312pl5321h9d6c286_0``, prefix
       ``/work/shared/users/phd/sk3463/tools/verilator`` (``dev/toolchains.rst``)
   * - C++ compiler
     - g++ (GCC) 13.3.1 20240611 (Red Hat 13.3.1-2), ``/opt/rh/gcc-toolset-13``,
       put first on ``PATH`` by ``oracle.py`` and ``probe_core.sh``.
       ``env-zhang21.sh`` alone resolves ``g++`` to Catapult's 10.3.0 because
       it puts ``$MGC_HOME/bin`` first; the Verilator runs here never use that
       compiler
   * - Python / numpy
     - 3.12.12 / 2.4.0 (``allo`` conda env, ``$ALLO_PYTHON``)
   * - Allo
     - this branch (``u1-pilot`` ``32448516`` + this record); bindings are the
       main checkout's ``mlir/build``, symlinked in (``u1-pilot`` and ``main``
       ``9b33ee03`` have no difference under ``mlir/``)
   * - PR #48
     - ``sunwookim028/allo#48`` head ``d83a988786b51ca305f19b7ebdd4134dd7f4ed18``,
       merged into ``origin/main`` ``9b33ee03`` in a throwaway worktree
       (``scratch/mr0/wt-pr48``, not committed anywhere), the two conflicts
       resolved as ``pr48_probe_2026-10-08.md`` section 2 says, plus
       ``minitpu_rtl_m0_2026-10-08/pr48_rtl_validate_json.patch`` (the
       ``--json-only`` port of ``validate_rtl``; sha256 ``4a5fc526...d264``)

Reproduce::

   source examples/minitpu/harness/env-zhang21.sh
   $ALLO_PYTHON examples/minitpu-rtl/oracle.py            # 1. oracle.json + the table (~45 s once the TB is built)
   dev/records/minitpu/minitpu_rtl_m0_2026-10-08/probe_core.sh \
       scratch/mr0/minitpu-b3ba0a4d scratch/mr0/probe2   # 2. standalone core build, lint, smoke
   $ALLO_PYTHON dev/records/minitpu/minitpu_rtl_m0_2026-10-08/boundary_array_sim.py   # 3a. this branch
   # 3b. in a PR #48 tree (above), from its root, with gcc-toolset-13 first on PATH,
   #     CXX=.../gcc-toolset-13/root/usr/bin/g++ and VERILATOR=<pinned verilator>:
   $ALLO_PYTHON <this dir>/memport_sim.py

``oracle.py`` exports the clone itself (refusing a clone that is not at the
pin or has modified tracked files) and builds ``tb_kernel_image`` into
``scratch/mr0/build/kernel_image_fifo64`` on first use, with MiniTPU's own
``tb/build_kernel_image.sh``. That build took **3 min 48 s** wall, 627 MB peak
RSS (``/usr/bin/time``; ``--binary --timing -Wno-fatal``, ``VERILATOR_JOBS``
from MiniTPU's own load-aware ``tb/verilator_jobs.sh``).

1. The oracle
=============

**What it runs.** The kernel set is ``tb/run_all.sh``'s default
``sim_kernel.py`` lines (71-137) minus the 2-round and
``MINITPU_GEMM_STAGING`` variants and minus the long tier (``rope 14``,
``gqa 2 7``, ``ffn``, ``attention``, ``flash --full``): 21 groups, **52
launches**. The images and operand blobs are not re-derived: ``oracle.py``
imports the clone's ``tools/sim_kernel.py``, replaces its one launch function
``run_kernel_image`` with a recorder, and calls sim_kernel's own ``run_*``
functions, so it records exactly the image (``asm.py`` via the clone's
``board_package/kernels.py`` builders), blobs, offsets, register values and
drain window MiniTPU's own gate stages. The looped GEMM and the row softmax
bypass that function in sim_kernel, so those two are staged by copying
sim_kernel's lines (``build_operands``/``write_operands``, ``softmax_in_sim``).
Each launch then goes to

- **RTL**: ``tb/run_kernel_image.sh`` with the plusargs
  ``compiler/harness.py:run`` writes (``+KERNEL_BIN +ARGn +ARG_OFFSETn
  +ARG_VALUEn +DUMP +DUMP_BYTES +DUMP_ARG +MAX_CYCLES``),
  ``MINITPU_ROUND_TRIP_CYCLES=0`` (recorded in the manifest as
  ``round_trip_cycles``), ``MINITPU_FIFO_DEPTH=64`` (``asm.mxu_output_fifo()``);
  cycles and bundles are the TB's ``KERNEL_IMAGE_ROUND`` line, i.e. the
  device's own ``perf_cnt_cycles``/``perf_cnt_instrs``;
- **emulator**: ``tools/emulate_image.py`` through the clone's own
  ``compiler/backends.py:emulate`` (same windows, offsets and values; its own
  address layout).

``oracle.json`` (committed beside ``oracle.py``) holds per launch: image,
operand and both drain sha256s, bit identity, words differing, NaN-only
differences, max ULP and relative error over finite words, RTL cycles,
issued bundles (RTL and emulator), ``launch_status``, any assertion lines. It
holds no paths and no wall times: **two full runs gave a byte-identical
``oracle.json``** (sha256 ``4fd68404...bfde8``).

**Cross-check against MiniTPU's own driver.** ``MINITPU_RESULT_DIGEST=1
tools/sim_kernel.py`` in the export (same TB build) prints the first 16 hex of
each drain's sha256. For ``--operands structured``, ``--operands varying``,
``--layernorm 4``, ``--head-scatter`` (6 launches) and ``--rope 2`` all ten
digests equal ``oracle.json``'s RTL digests (``886e377e5b74a3c8``,
``aab61c80c6af6451``, ``c9f5b422e9fb3c0f``, ``9cd1c2da``/``e45dfc07``/``ac3db020``/
``6b84c21d``/``35692694``/``8df3ea60``, ``7acc77b06adeef6a``), and sim_kernel
judged each PASS against numpy. So the recorder reproduces MiniTPU's staging
bit for bit.

**Result** (stdout of the second identical run;
digests are the first 8 hex of sha256, wall is per launch with 4 in
parallel; whole run 44 s wall)::

   kernel                     image     rtl drain emu drain agree  differ  rel err  cycles bundles  emu  wall
   gemm_structured            d7b1a21e  886e377e  886e377e  yes         0  0.0e+00    1350     349    =    1s
   gemm_varying               d7b1a21e  aab61c80  f30ad0e7  NO          6  1.0e-03    1350     349    =    1s
   gemm_varying_gelu          128f9167  d94353e8  8489987b  NO        365  7.0e-03    2542     572    =    1s
   gemm_c4g2_fuse             bc28ecf8  5f0e471e  ba5ae232  NO         33  7.1e-04    6992    2461    =    2s
   gemm_c4g2                  7472829f  54fb566a  29134d69  NO         61  6.6e-04    8270    2791    =    2s
   softmax_8                  b532580b  f2ed46f3  f1cb978b  NO        378  4.3e-03     734      77    =    1s
   layernorm_4                84fcf2f1  c9f5b422  54a8f492  NO        931  2.6e-03    3801     797    =    1s
   layernorm_32_packed        eb0bf837  b2215a09  76444c51  NO       8549  4.3e-03   31069    6314    =    6s
   add                        ffc54559  48f27937  48f27937  yes         0  0.0e+00   10975    1494    =    2s
   add_bias                   709ba36f  f627505f  f627505f  yes         0  0.0e+00    9626    1477    =    2s
   add_bias_gelu_3072         139a4049  4858ad7c  3032eec5  NO      57426  3.5e-03   40386    7805    =    8s
   softmax_packed_12          0d6c77cc  a9af09ae  5e2c2643  NO      17749  4.5e-03   18918    3245    =    4s
   softmax_packed_2_span128   a3daf98b  ddcf96c4  120cc58d  NO       5817  4.7e-03    5354     961    =    2s
   rmsnorm_4_w896             779da9ae  6ea18856  63258350  NO        878  3.9e-03    3073     565    =    2s
   rmsnorm_32_w896_packed     5c1218e3  a07a1071  faff129c  NO      12243  5.5e-03   25781    4458    =    5s
   rope_2                     ad3da3b9  7acc77b0  7acc77b0  yes         0  0.0e+00    2125     460    =    1s
   swiglu_2                   deb85b5b  1c8d2101  8df7f3e8  NO       2200  4.7e-03    2739     854    =    1s
   gqa_2_2.0                  f35babc3  b22cd595  a4e83610  NO         22  3.9e-04    4400    1299    =    2s
   gqa_2_2.1                  af94f284  3c0eeb26  a090c32d  NO         28  5.7e-04    4402    1301    =    2s
   head_scatter.0             7eaff0cf  9cd1c2da  9cd1c2da  yes         0  0.0e+00    1973     420    =    1s
   head_scatter.1             436cbf60  e45dfc07  e45dfc07  yes         0  0.0e+00    2914     609    =    1s
   head_scatter.2             91b2c62d  ac3db020  ac3db020  yes         0  0.0e+00    1973     420    =    1s
   head_scatter.3             d9d944ab  6b84c21d  6b84c21d  yes         0  0.0e+00    1459      74    =    1s
   head_scatter.4             5b30c0c6  35692694  35692694  yes         0  0.0e+00    2065      60    =    1s
   head_scatter.5             437801ef  8df3ea60  8df3ea60  yes         0  0.0e+00    1461      76    =    1s
   loop_begin_r.0             f7ea0a44  7bef7079  7bef7079  yes         0  0.0e+00      23       8    =    1s
   loop_begin_r.1             83819c6d  7bef7079  7bef7079  yes         0  0.0e+00      25      10    =    1s
   loop_begin_r.2             f7ea0a44  417ce8b3  417ce8b3  yes         0  0.0e+00      47      12    =    1s
   loop_begin_r.3             83819c6d  417ce8b3  417ce8b3  yes         0  0.0e+00      49      14    =    1s
   loop_begin_r.4             f7ea0a44  19a93954  19a93954  yes         0  0.0e+00     153      28    =    1s
   loop_begin_r.5             83819c6d  19a93954  19a93954  yes         0  0.0e+00     155      30    =    1s
   loop_begin_r.6             f7ea0a44  5d408de1  5d408de1  yes         0  0.0e+00     465      76    =    1s
   loop_begin_r.7             83819c6d  5d408de1  5d408de1  yes         0  0.0e+00     467      78    =    1s
   loop_begin_r.8             6256b7da  0badf96c  0badf96c  yes         0  0.0e+00     350      63    =    1s
   loop_begin_r.9             6256b7da  7bef7079  7bef7079  yes         0  0.0e+00      32      13    =    1s
   loop_begin_r.10            6256b7da  7bef7079  7bef7079  yes         0  0.0e+00      24       9    =    1s
   loop_begin_r.11            6256b7da  375007ec  375007ec  yes         0  0.0e+00     980     161    =    1s
   loop_begin_r.12            f63469e6  7bef7079  7bef7079  yes         0  0.0e+00      32      11    =    1s
   loop_begin_r.13            f63469e6  417ce8b3  417ce8b3  yes         0  0.0e+00      56      15    =    1s
   loop_begin_r.14            f63469e6  c9d5b0c5  c9d5b0c5  yes         0  0.0e+00     104      23    =    1s
   loop_begin_r.15            f63469e6  0d9620e5  0d9620e5  yes         0  0.0e+00     128      27    =    1s
   loop_begin_r.16            f63469e6  0d9620e5  0d9620e5  yes         0  0.0e+00     128      27    =    1s
   loop_begin_r.17            57915317  e7d966e9  e7d966e9  yes         0  0.0e+00  113617   17484    =   17s
   loop_begin_r.18            57915317  e7d966e9  e7d966e9  yes         0  0.0e+00  113617   17484    =   17s
   flash_attention.0          4287d6fc  5a961446  91e6c8e4  NO       2706  1.1e-32   10081    2377    =    2s
   flash_attention.1          2e5ba528  910f3b03  b0623cbb  NO       4127  1.6e-32   23892    6495    =    5s
   flash_attention.2          398a166c  ffa7692e  2c13e789  NO       5674  4.1e-32   30473    8639    =    6s
   flash_attention.3          bd85d239  1e92158d  43e5f4eb  NO       2735  3.0e-33   22380    6284    =    5s
   flash_attention.4          4287d6fc  97ea2f6b  968e7f9c  NO       1521  5.5e-33   13949    3205    =    3s
   flash_attention.5          4287d6fc  75c96a6c  a049d6c8  nan      4096  0.0e+00    3941     421    =    1s
   flash_attention.6          4287d6fc  74cd6fcc  ea361ca2  NO       2810  4.2e-33   23441    6297    =    4s
   flash_attention.7          4287d6fc  03e5e469  f94be81a  NO       3095  5.5e-33   29597    8253    =    5s
   52 launches; RTL halted on all; RTL == emulator drain bit-identical on 29, equal up to NaN sign/payload on 1 more; issued bundles equal on 52/52
   agree: yes = bit-identical, nan = only NaN encodings differ, NO = values differ (emulator is functional: float64 MXU, exact SFU functions)
   
   real	0m44.270s
   user	4m39.066s
   sys	0m4.247s

- **The RTL halts on all 52**, ``launch_status`` 0, no assertion line.
- **Issued bundles agree on 52/52** (RTL ``perf_cnt_instrs`` == emulator
  ``bundles``): the control flow (loops, ``loop.begin.r`` skips, guards,
  saturation) is the same in both.
- **Drains are bit-identical on 29/52**: every data-movement launch
  (``head_scatter`` x6), ``add``, ``add_bias``, ``rope_2``, the structured
  GEMM, and all 19 ``loop_begin_r`` launches. One more (``flash_attention.5``)
  differs only in NaN encoding: the RTL writes ``+qNaN`` (``0x7FC0``), the
  emulator ``-qNaN`` (``0xFFC0``), on 4,096 words.
- **22 differ in value**, as the emulator's own docstring says they must: its
  MXU accumulates in float64 where the RTL's ``mxu_acc24`` chain rounds
  (GEMM, gqa, flash: a few to a few thousand words, relative error <= 1e-3
  on GEMM), and its SFU is the exact function where the RTL interpolates
  tables (softmax, layernorm, rmsnorm, gelu, swiglu: relative error 2e-3 to
  7e-3). Max ULP (in the JSON) is large wherever a near-zero value changes
  sign and is not a useful tolerance; flash attention's relative error is
  ~1e-32 only because its window holds large mask constants that dominate the
  norm.

2. ``minitpu_core`` standalone under PR #48's build flags
=========================================================

PR #48 builds an RTL IP with ``_rtl_command()`` (``verilator --top-module
<top> <rtl files> -I<incdirs> -G.. -D..``) plus ``--cc --build --Mdir vgen
-CFLAGS "-fPIC -fvisibility=hidden"`` (``rtl.py:732-743, 802-813`` at
``d83a9887``); no ``-Wno-fatal``, ``--timing``, ``--assert`` or
``--build-jobs`` can be passed. ``validate_rtl`` runs first with
``-Wno-fatal``. ``probe_core.sh`` runs exactly that on ``src/core/core.f``
expanded into a file list (``+incdir+src/pkg`` becomes ``-I``), top
``minitpu_core``:

.. list-table:: ``probe_core.sh`` (second run; ``scratch/mr0/probe2.log``)
   :header-rows: 1

   * - Step
     - Flags beyond the file list
     - rc
     - Warnings / errors
     - Wall
   * - **#48 build step, exact**
     - ``--cc --build --Mdir vgen -CFLAGS "-fPIC -fvisibility=hidden"``
     - 0
     - **0 / 0**
     - 253 s (first run 430 s; serial, #48 passes no ``--build-jobs``;
       Verilator's own report: 25 s convert, the rest the C++ build)
   * - lint, default warnings
     - ``--lint-only``
     - 0
     - 0 / 0
     - 15 s
   * - lint, ``-Wall``
     - ``--lint-only -Wall``
     - 1
     - 1 / 1: ``SYNCASYNCNET`` on ``rst_n`` (``minitpu_core.sv:20``; flopped
       both sync and async), fatal only because ``-Wall`` enables it
     - 16 s
   * - lint, ``--timing`` / ``--no-timing``
     -
     - 0 / 0
     - 0 / 0
     - 14 / 16 s
   * - #48 ``validate_rtl`` (JSON port)
     - ``--json-only -Wno-fatal``
     - 0
     - 0 / 0
     - 1 s
   * - contrast: AXI top ``minitpu``
     - ``--lint-only`` (core.f + minitpu.f)
     - 0
     - 0 / 0
     - 17 s
   * - contrast: ``tb_kernel_image``
     - ``--lint-only --timing`` (MiniTPU's TB)
     - 1
     - 21 / 1: 16 ``WIDTHEXPAND`` + 5 ``WIDTHTRUNC``, **all in ``tb/``**
       (15 ``minitpu_axi_mem_model.svh``, 3 ``minitpu_launch_harness.svh``,
       3 ``tb_kernel_image.sv``)
     - 16 s

**Verdict: yes.** ``minitpu_core`` builds standalone with #48's exact flags,
no warning at all; the minimal flag set is #48's set as it is. The
``-Wno-fatal`` in MiniTPU's own builds is needed only for its testbench
files; neither seam K (``minitpu_core``) nor seam T (``minitpu``) needs it,
nor ``--timing`` (the RTL has no timing constructs). The built model
(``libVminitpu_core.a`` 5.0 MB) links into a 20-line C++ driver
(``core_smoke.cpp``) that resets it and clocks 200 cycles.

Two facts M-R1 must design for, found here:

- **The SFU ROMs load by a cwd-relative path.** ``sfu.sv`` reads
  ``src/core/sfu/{gelu,exp}_bf16.mem`` with ``$readmemh`` (or
  ``vpu_*_bf16.mem`` under ``MINITPU_PACKAGED_IP``), relative to the process's
  working directory, once per lane (64 instances). Run from any other
  directory, the model prints ``%Warning: ... $readmem file not found`` 64 x 2
  times **and runs on, with zeroed GELU/exp tables** (smoke above: it does not
  stop). Under #48 the Verilated model runs inside the Python process, so its
  cwd is the Allo program's cwd. M-R1 must make the path absolute (the
  ``GELU_MEM_FILE``/``EXP_MEM_FILE`` parameters belong to ``sfu``, not to the
  top, so ``-G`` cannot reach them; ``-DMINITPU_PACKAGED_IP`` plus copying the
  two files beside the run, or a shim-side ``chdir``, or a ``defparam``-style
  wrapper) and must fail on the warning, since the TB's numbers depend on
  those tables.
- **Every top input must be bound** (``rtl.py:785-791``) and the core's ports
  are 128- and 256-bit wide (``kernel_arg_csr``, ``dma_iram_din``,
  ``dm_req_wdata``, ``dm_rsp_data``): the JSON validation accepts them as
  basic types, but no ``Port``/``MemPort`` can carry them, so the shim of the
  plan's (a-K) is required, as the plan says.

3. ``MemPort``-style boundary array under ``target="simulator"``
================================================================

(a) **On this branch (no #48): yes.** ``boundary_array_sim.py``: a
``@df.region`` with a boundary ``ddr: int32[8]`` passed to one kernel that
reads and writes every word in place; ``df.build(target="simulator")``;
``BOUNDARY_ARRAY_SIM PASS [1, 3, 5, ..., 15]``, 11.6 s wall (build + run).
This is the Allo half of the RAM form and needs nothing new. ``MemPort``
itself does not exist on this branch (it is PR #48's).

(b) **#48's ``MemPort`` bound to a region boundary array, in the dataflow
simulator: yes.** ``memport_sim.py`` on the merged #48 tree (Pins): the PR's
own ``memory.v`` (per call: read ``X[0]``, write ``X[0] + 1``) declared as
``RTLModule(..., ports=[MemPort("X", 4, "int32_t", "addr", "ce", q="q",
we="we", d="d")], done="ap_done")``, called twice by the one kernel of a
``@df.region`` whose boundary array ``X`` it is handed::

   MEMPORT llvm: PASS [11, 2, 3, 4] (want [11, 2, 3, 4])        # the PR's own test_memory_binding
   MEMPORT simulator: PASS [11, 2, 3, 4] (want [11, 2, 3, 4])   # the new case; 57 s wall for both

So the plan's [I] precondition holds: the RAM form of (a-K) (owner decision 2)
is open. What this does **not** show: a ``MemPort`` and stream ``Port``\ s on
one IP in the simulator (#48's stream wrapper path, ``compile_shared_lib
(stream_sim=True)``, is chosen per IP by ``has_stream_args``; a mixed IP was
not tried), a ``MemPort`` larger than 4 words, ``uint32``, or two kernels
touching the same boundary array. The first is the shape the shim needs (RAM
for ``ddr`` + a command stream + a status stream) and is M-R1's first test.

4. Where the plan was wrong or imprecise
========================================

- **"Check: TB and emulator agree on every kernel" (M-R0) cannot be a bit
  check.** ``emulate_image.py`` is a functional model (float64 MXU, exact SFU
  functions; its docstring says so), so its drain is bit-identical to the
  RTL's on 29/52 launches only (section 1). What does hold on all 52: the RTL
  halts, issued bundles are equal, and the value differences are the
  documented arithmetic ones. **The bit oracle for M-R1/M-R2 is the TB's drain
  digest** (``rtl.drain_sha256``), which sim_kernel's own ``RESULT_DIGEST``
  confirms; the emulator is a tolerance and control-flow oracle.
- **"no ``--assert``, so the core's ``ifndef SYNTHESIS`` checks are silent"
  ([P], plan section 3a) is false on Verilator 5.052.** Immediate assertions
  are on by default there (``--no-assert`` disables them; there is no
  ``--assert`` in ``verilator --help``). Both the #48-flag core build and
  MiniTPU's TB build compile the ``Assertion failed`` checks in (e.g. 180 in
  ``Vminitpu_core___024root__0.cpp``, ``mxu_pe`` 11 each). #48 not passing
  ``--assert`` loses nothing.
- **"#48's build step has no ``-Wno-fatal`` (MiniTPU's full-core build uses
  it)" is a risk only for the testbench**, not for the core or the AXI top
  (section 2).
- **"``plan.json`` and ``layout.json`` ... belong to minitpu-comp master /
  ``minitpu-cc``" (plan section 2) is imprecise:** ``minitpu-cc`` is in the
  pinned tree (``compiler/cc.py``, ``compiler/artifacts.py:1-16``) and writes
  both as build outputs; they are not checked-in files, which is why a file
  search finds none. ``isa_version`` and a ``versions`` key in
  ``docs/isa_latency.json`` are indeed absent at ``b3ba0a4d``.
- The plan's paths are otherwise as stated: ``tb/run_kernel_image.sh``,
  ``tools/sim_kernel.py:318-366``-style invocation, ``tools/emulate_image.py``,
  ``src/core/minitpu_core.sv`` with no AXI. Two details it did not say: the
  emulator is reached through ``compiler/backends.py:emulate`` (MiniTPU's own
  Launch/window layout), and ``compiler/harness.py:run`` is MiniTPU's own
  Python wrapper of the same TB, which ``oracle.py`` mirrors rather than calls
  (it hides the transcript and ``launch_status``).

Notes on running it
===================

- ``oracle.py`` prints three ``note: ... sim_kernel's judging raised
  IndexError`` lines (``layernorm_4``, ``rmsnorm_4_w896``, ``swiglu_2``):
  sim_kernel judges the recorder's unwritten drain after the launch was
  recorded and fails indexing rows it prints on failure. These are
  single-launch groups, so nothing is lost; the note is kept so a multi-launch
  group that stopped early would show.
- The export is refreshed only when its ``.allo-export-pin`` marker differs
  from ``PIN``; the TB build is reused by ``tb/build_kernel_image.sh``'s own
  newer-source check.
- ``--round-trip N`` runs the memory model with an N-cycle round trip
  (``round_trip_cycles`` in the manifest); only 0 was run here.
