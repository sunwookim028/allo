MiniTPU-rtl: plan for the D-21 baseline (2026-10-08)
=====================================================

:Host: zhang-21. MiniTPU read-only at ``/work/shared/users/phd/sk3463/minitpu``
       @ ``b3ba0a4d`` (clean). Allo ``main`` @ ``9b33ee03``; D-21 and the U3
       records on ``origin/u1-pilot``; PR #48 read at ``d83a9887`` (ref
       ``pr48-study``, deleted after reading).
:Marks: **[V]** read in the cited file/line or checked on this host;
       **[I]** inferred. No code was written in either tree.
:Inputs: the parallel #48 probe, ``scratch/pr48_probe_2026-10-08.md`` (merged
       tree tested on this host; ``mxu.sv`` shim at DIM=2 run), folded in at
       §2, §3c and §4 as **[P]** (its results, not re-run here).

1. The claim, and what "obtained from Allo" must mean
-----------------------------------------------------

**Claim (proposed wording).** ``examples/minitpu/rtl/`` is an Allo design (one
``@df.region`` composed by ``compose.Architecture``) in which the MiniTPU
compute core at ``b3ba0a4d`` runs as the real RTL, wrapped as IP. An Allo
program -- a MiniTPU kernel image assembled by MiniTPU's own ``asm.py`` plus
operands, written into the region's boundary memories -- is driven end to end
through Allo's dataflow simulator (Verilator inside the IP), and the drain it
reads back is bit-identical to what MiniTPU's own ``tb_kernel_image`` produces
for the same image and operands, with the device cycle count reported beside
MiniTPU's.

**What Allo contributes** (and what the record must say it contributes):

- the *design description*: the region, its boundary memories (program, DDR
  image, drain), the channels and the unit that hosts the IP; declared, checked
  by ``compose`` (one owner per memory/port, declaration == body) [V
  ``allo/compose.py:519-599``];
- the *IP wrapping*: the ``RTLModule`` descriptor (ports, clock, reset,
  parameters) and the shim RTL that adapts MiniTPU's top to a stream/RAM
  boundary. The shim is new RTL written on the Allo side; it is listed as
  Allo's, never as MiniTPU's;
- the *driver/compiler path*: the host sequence (load image, set CSRs, start,
  wait, dump) expressed as an Allo program over the region, not as a Python
  script beside it; MiniTPU's ``asm.py`` is imported from the pinned clone as
  the encoder (as ``harness/rtl.py`` already imports the clone) [V
  ``u1-pilot:examples/minitpu/harness/rtl.py:80-89``];
- the *checks*: a differential gate against MiniTPU's ``tb_kernel_image``
  drain digest and the functional emulator, plus the cycle report.

**What stays MiniTPU's**: every ``.sv`` under ``src/``; the ISA, the bundle
format and ``asm.schedule()``; ``docs/isa_latency.json``/``isa_slots.json``;
the numpy references in ``board_package/{kernels,gpt2_kernels,bf16}.py`` and
``tools/emulate_image.py``; the board flow under ``fpga/``.

**Honesty conditions.** (1) In architecture (a) no Allo-generated logic is on
the data path, so the claim is "obtained *through* Allo's design, wrapping,
driver and check path", and the record says exactly which RTL is MiniTPU's and
which is shim. (2) The core is unmodified (no interlocks, no ``define``
changes beyond MiniTPU's own ``MINITPU_MXU_OUTPUT_FIFO_DEPTH``); D-7's rule
stands. (3) The oracle is MiniTPU's own harness on the same bytes, not numpy
alone: ``tb_kernel_image``'s PASS banner only means "halted" [V
``tb/tb_kernel_image.sv:5``], so the bit comparison is ours to make.
(4) Cycle counts are reported with the memory-latency model named
(``+ROUND_TRIP_CYCLES``), because they depend on it [V ``tb/run_kernel_image.sh``].

2. The core's actual interface
------------------------------

**Two tops.** ``src/core/minitpu_core.sv`` is the compute core and has **no
AXI** [V ``:19-56``]; ``src/minitpu.sv`` is the board/IP top that adds the
AXI-Lite slave, the AXI4 master, the CDMA controller and the IRAM loader [V
``src/minitpu.sv:7-98, 370-689``]. Everything is one clock domain.

**Seam K -- ``minitpu_core`` ports** [V ``src/core/minitpu_core.sv:13-56``]:

- ``clk``; ``rst_n`` active-low, async assert / sync deassert;
- control: ``start`` (in, pulse), ``done`` (out), ``illegal_op_o`` (const 0);
- IRAM load port: ``instr_write_en``, ``iram_addr[11:0]``,
  ``dma_iram_din[127:0]`` -- one 128-bit bundle per write;
- CSRs: ``program_id_csr[31:0]`` (entry PC = ``[3:0] << 8``),
  ``kernel_arg_csr[127:0]`` = {arg3..arg0}, DM-word addresses or loop bounds;
- device-memory "credit pipe" (ready/valid, in-order, not AXI): request
  ``dm_req_valid/ready/we``, ``dm_req_addr[28:0]`` (DM-word = 32 B units),
  ``dm_req_len[7:0]`` (words-1), ``dm_req_wdata[255:0]``, ``dm_req_wstrb[31:0]``;
  response ``dm_rsp_valid/ready``, ``dm_rsp_data[255:0]``, ``dm_rsp_last``,
  ``dm_rsp_resp[1:0]``;
- status: ``dma_err_o`` (+channel/cause/resp), ``dma_rsp_seen_o``,
  ``beat_count_o``, ``overlap_stat_o``, ``perf_cnt_cycles_o``,
  ``perf_cnt_instrs_o``, ``dma_channel_done_o[1:0]``. No interrupt anywhere;
  completion is polled [V].

Parameters: ``INSTR_ADDR_W=12``, ``DRAM_BEAT_ADDR_W=29``, ``DM_DATA_WIDTH=256``;
VPU geometry is ``define``-driven in ``vpu_pkg`` (UNITS.md §3), so an
``RTLModule`` passes it through ``defines=`` not ``parameters=`` [V/I].

**Seam T -- ``minitpu`` (AXI top)** [V ``src/minitpu.sv:20-98``]:
``s00_axi_aclk/aresetn``; AXI4-Lite slave ``s00_axi_*`` (7-bit addr, 32-bit);
AXI4 master ``m00_axi_*`` (34-bit addr, 256-bit data, id 1 bit, INCR bursts up
to 256 beats, no cache/prot/qos); AXI4-Lite master ``cdma_lite_*`` (programs an
external Xilinx CDMA; tied off in the TB); ``ddr4_calib_complete``.

**AXI-Lite command ABI** (ABI v9; word index ``ARADDR[6:2]``) [V
``src/host/minitpu_slave_axi_lite.v:208-436``, ``board_package/minitpu_runtime/minitpu/abi.py:27-61``]:
``0x00`` cmd_w0 = [3:0] OPC (0 COPY, 1 LAUNCH), [4] doorbell (self-clearing),
[7:5] DIR; ``0x04/0x08`` kbin_addr lo/hi; ``0x0C`` kbin_size (bytes);
``0x10-0x1C`` w4-w7 (COPY fields / reserved); ``0x20`` RO instr_ready;
``0x24-0x30`` kernel_arg_0..3 (WO at 0x24-0x2C; reads there return
dma_channel_done / copy_status / copy_cdmasr); ``0x34`` program_id; ``0x3C``
launch_status ([0] sticky launch_error); ``0x40`` perf_cnt_cycles; ``0x44``
perf_cnt_instrs; ``0x48`` build_info; ``0x4C`` dma_err; ``0x50`` beat_count;
``0x54`` dma_overlap; ``0x58`` build_id. Launch sequence [V ``runtime/launch.py:164-227``,
``tb/tb_kernel_image.sv:126-165``]: write args -> poll 0x20 == 1 -> write
w1..w7 -> write w0 = 0x11 last -> poll 0x20 low then high -> read 0x3C, 0x4C.
Hardware: ``command_unit`` latches on the doorbell, ``iram_loader`` bursts
``kbin_size/16`` bundles from DDR into IRAM, ``core_launch_ctrl`` pulses
``start``, ``done`` retires the doorbell [V ``src/host/command_unit.sv:134-192``].

**IRAM loading.** On the AXI top, only the loader writes IRAM, from DDR; there
is no AXI-Lite path into IRAM and no ``$readmemh`` [V ``src/minitpu.sv:676-689``].
The sub-core testbenches drive the core's IRAM port directly [V
``tb/tb_e2e_unified_runner.sv:666-668``]. **VMEM has no host readback**;
results leave by ``vmemst`` descriptors to DDR and the TB backdoor-reads a
``+DUMP`` window [V ``tb/tb_kernel_image.sv:79-99``; ``docs/ISA_AND_INTERFACES.md:205-209``].

**Kernel image.** Raw little-endian 16-byte bundles, no header, no version
word [V ``board_package/asm.py:1693-1694``]. ``isa_version``, ``plan.json`` and
``layout.json`` do **not** exist at ``b3ba0a4d`` (grep over the tree and
``git log -S``); they belong to minitpu-comp master / ``minitpu-cc`` [V].

**What PR #48 carries, and what a shim must add** [V ``pr48-study:allo/backend/rtl.py``]:

- carries: ready/valid or ``ap_fifo`` *scalar* stream ports, payload ≤ 32 bits
  (``_TYPES``, ``rtl.py:24-34``), one clock, one reset with polarity, optional
  ``start``/``done`` pins, ``persistent=True`` state across calls, ``-G``
  parameters and ``-D`` defines, a word-addressed ``MemPort`` RAM backed by a
  host array (simulation only, tested on ``target="llvm"``) [V ``rtl.py:69-81, 936-941``];
- does not carry: AXI-Lite, AXI4, aggregate pins (rejected at ``--xml-only``
  validation, ``rtl.py:773-776``), any payload > 32 bits, a user-triggered
  reset, an interrupt, a second clock; every top-level *input* must be bound
  [V ``rtl.py:785-791``]; transactions are one call = run until ``done`` or
  until every output delivered ``size`` tokens [V ``rtl.py:324-329``]; no
  cycle synchronisation between two ``RTLModule`` objects [V ``docs/RTL_MODULE.md``];
  ``MAX_STALL=500000`` idle cycles aborts the run [V ``rtl.py:167, 333-335``] --
  a long kernel with no stream traffic would trip it [I]; Verilator flags
  are fixed (no ``-Wno-fatal``, ``--assert``, ``--timing`` pass-through) and
  ``validate_rtl`` uses ``--xml-only``, which **Verilator 5.052 no longer
  has**: on the pinned Verilator the PR simulates nothing until its ~50-line
  ``--json-only`` port is applied (``scratch/pr48_rtl_validate_json.patch``;
  with it 60 passed, 1 skipped) [P];
- so the shim must add, for seam K: the 256-bit credit-pipe memory (serving
  reads and writes in order), the IRAM write sequencing, CSR latching,
  ``start``/``done`` and the status dump -- all behind ≤ 32-bit stream or RAM
  pins; for seam T, additionally an AXI-Lite master and an AXI4 slave memory.
  MiniTPU's ``tb/minitpu_axi_mem_model.svh`` is that memory in SV (byte array,
  backdoor tasks, 8-deep read queue, ``+ROUND_TRIP_CYCLES``) but its
  ``axil_write/read`` BFM tasks and the harness use ``--timing`` constructs; #48
  builds ``--cc`` without ``--timing`` [V ``rtl.py:802-813``], so a seam-T
  shim is an FSM rewrite, not an include [I].

3. Three candidate architectures
--------------------------------

(a) Whole core as one ``RTLModule`` behind a shim; Allo program = host driver
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two shim variants, same Allo program:

- **(a-K, RAM form)** ``core_shim.sv`` around ``minitpu_core``: the DDR image
  is a region boundary memory ``ddr: uint32[N]`` exposed to the shim as a #48
  ``MemPort`` (addr/ce/we/d/q, 32-bit words); the shim converts each 256-bit
  credit-pipe beat into eight RAM accesses; a command stream (32-bit words:
  ``LOAD_IRAM n`` + 4n bundle words, ``CSR``, ``START``, ``END``) and a status
  stream out (``done``, cycles, bundles, dma_err, launch flags). The drain is
  read back from ``ddr`` by an Allo kernel -- the host program is Allo code
  over arrays, exactly TinyTPU's shape (``imem/A/B/C`` boundary memories, no
  scalars) [V ``examples/tinytpu/ip/tinytpu.py:56-63``]. *Precondition to
  verify at M-R0:* ``MemPort`` works under ``target="simulator"`` (only
  ``llvm`` is tested in #48) [I].
- **(a-K, stream form)** no ``MemPort``: the shim holds the memory as an SV
  array and the command stream also carries ``LOAD_MEM addr len`` + data and
  ``DUMP addr len`` returning data on the output stream. Fits exactly what #48
  verifies today (stream ports in the dataflow simulator) at the cost of
  streaming operands 32 bits per cycle (a few thousand cycles for one-tile
  kernels; ~3M for a 12 MiB operand window -- acceptable for the gate set) [I].
- **(a-T)** the same around ``src/minitpu.sv``: AXI-Lite master FSM speaking
  ABI v9 + AXI4 slave memory, in the shim. Buys the real command path, the
  IRAM loader and ``launch_status``; costs the AXI FSMs and gets nothing the
  claim needs that (a-K) lacks [I].

Simulation path: Verilator 5.052 through #48's transactor, inside the hosting
kernel's thread; the MiniTPU file list is ``src/minitpu.f`` -> ``src/core/core.f``
expanded into ``rtl=[...]`` [V]. Risks: #48's build step has no ``-Wno-fatal``
(MiniTPU's full-core build uses it; the eight MXU files happened to build
clean [P]) and no ``--assert``, so the core's ``ifndef SYNTHESIS`` checks are
silent [P]; ``MAX_STALL`` must become configurable; ``VERILATOR`` must point
at the pinned prefix (not on PATH) [V]; the ``--json-only`` port of
``validate_rtl`` is a prerequisite [P]. The transactor is untimed with respect
to the rest of the region, so this is a functional check with the core's own
cycle counters read out, not a cycle model of the region [P/V].

HLS/Vitis path: not meaningful for the whole core. #48's black box is
``csyn``-only, ``ap_fifo`` ports, ``ap_ctrl_chain``, needs a self-contained C
model with the ports' signature [V ``rtl.py:906-966``]; the core fills 83 % of
a ZCU104 [V ``fpga/build_bd_bitstream.tcl:542``]. Mark the Vitis cell
*refused* and leave bitstreams to MiniTPU's ``fpga/`` flow (M3).

Checked, against what: (i) drain bytes == ``tb_kernel_image`` ``+DUMP`` for
the same ``kernel.bin`` and operand blobs (``sim_kernel.py``'s ``_digest``
makes that a hash compare) [V ``tools/sim_kernel.py:130-145``]; (ii) drain ==
``tools/emulate_image.py`` (functional oracle); (iii) numpy references via
``sim_kernel.py``'s ``report()``; (iv) ``perf_cnt_cycles``/``instrs`` vs the
TB's ``KERNEL_IMAGE_ROUND`` line under the same ``ROUND_TRIP_CYCLES`` [V
``tb/tb_kernel_image.sv:183``]. Kernel set: ``sim_kernel.py``'s
``run_product``/``run_kernel_image`` callers (looped GEMM, layernorm, softmax,
rope, swiglu, head copy/transpose, flash attention) [V].

Cost: shim RTL 1-2 d; Allo region + driver + ``RTLModule`` descriptor 1 d;
differential gate against the TB 1 d; record 0.5 d -> **3.5-4.5 agent-days**
for (a-K); (a-T) adds 2-3 d. Supports: D-21's claim as written ("most or all
units are the real RTL" -- here all).

(b) Per-unit RTL IPs (vpu, mxu, sequencer, dma) composed in an Allo region
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The seam is ``vpu_ctrl_t`` (``vpu_pkg.sv:164-197``): ~29 decoded fields issued
as a **pulse every cycle with no handshake**, on a fixed write-port calendar
(``sequencer_pkg.sv:559-573``); ``matrix_busy`` is simulation-only [V]. Under
#48 this is blocked three times over: aggregate pins are rejected, there is
no cycle synchronisation between two ``RTLModule`` instances (each runs its
own clock loop in its own kernel thread), and a pulse interface has no
``size``/``done`` transaction to bound a call [V]. The owner's own ``UNITS.md``
§2.2/§3 says ``vpu.sv``, ``sequencer.sv``, ``dma.sv``, ``mxu.sv`` "cannot be
lifted out" [V]. What *is* transactional: ``mxu`` (push/commit/pop),
``dma`` (descriptor valid/accept + credit pipe), ``iram_loader``, the host
blocks [V]. A general (b) needs a cycle-synchronous multi-IP transactor (new
mechanism, ~6-8 d) and then re-derives U4's contract question in RTL; it
duplicates the ladder rather than baselining it. Simulation: Verilator only;
Vitis: refused (same as (a)). Cost **10+ d**; supports the "composability of
RTL modules" sentence of the Direction, but at the ladder's expense.

(c) Hybrid: the ladder's Allo units + RTL IP for the rest
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Two directions with very different maturity:

- **(c1) real ``mxu.sv`` inside the ladder's Allo units** -- shown by the
  probe at DIM=2: ``mxu_shim.sv`` (60 lines) makes one call = DIM RHS rows +
  commit (derived from the accept count) + NS LHS rows, NS result words out;
  3 and 5 tiles bit-identical to ``ref_mxu.mxu_row`` through Verilator in a
  3-kernel region with persistent state and bank alternation [P]. Its
  limits are the adapter's: payloads ≤ 32 bits (DIM 4/16 need a 2-/8-beat
  row serializer and 8-/32-beat output), sidebands only by word order, one
  tile op per call (no ``vmatload``/``vmatpush``/``vmatpop`` as separate
  steps on one MXU: one static call site and one Verilated model per
  object), no overlap (``ii=0``: the double-banked weights are never
  exercised), and no cycle contract (push->valid 12/22/82, span 5/15/75,
  FIFO-full drop are only the trace harness's) [P; V ``u3_phase0:142-155,
  197-214``]. Allo contributes the surrounding units (ladder U1-U3) and the
  composition; cost **2-3 d** to DIM 4/16. Vitis: needs ``HLSBlackBox`` with
  a C++ tile-op model and a clock-enable the core lacks (the generated
  ``ReadyValidAdapter`` wrapper fails elaboration without ``ce``) [P] -- not
  worth it for a baseline.
- **(c2) Allo's MXU in MiniTPU's RTL tree in place of ``mxu.sv``** (D-21's
  second bullet): Catapult RTL of the U3 MXU behind the same
  push/commit/pop ports, held to MiniTPU's own ``tb_mxu_*``/bundle suites.
  Today that RTL is contract-exact at DIM 4 but 74 cyc/row and push->valid
  1,631 vs 22 [V ``u3_track_c:167-171``]; until it reaches II=1 (D-1's
  reversal condition) it is a declared ISA version with a very different
  schedule, not a drop-in. Cost **5-8 d** after the Catapult II work, plus
  the delta generator (§5). Oracle: MiniTPU's tbs with a re-scheduled
  ``isa_latency.json``.
- A forward hybrid in the *Allo simulator* (Allo VPU + RTL sequencer) is a
  semantic mismatch: the simulator is untimed and the RTL sequencer issues on
  a calendar [V ``compose.py:960-961``]; not proposed.

4. Recommendation and order of work
-----------------------------------

**Recommend (a-K)**, RAM form if ``MemPort`` runs in the dataflow simulator,
else stream form; (a-T) only if the owner wants the AXI ABI exercised (§6).
Run (c1) in parallel as the D-21 hybrid; (c2) after Catapult II=1 on the MXU.

Milestones (each with one pass check), on a branch ``minitpu-rtl`` in a
worktree, merged after review:

- **M-R0 -- oracle and fallback (1 d, no #48).** ``examples/minitpu/rtl/``
  gets ``oracle.py``: assemble a kernel with the clone's ``asm.py``, run
  ``tb/run_kernel_image.sh`` (as ``sim_kernel.py`` does, ``:318-366``), run
  ``emulate_image.py``, record the drain digest, cycles and bundles per
  kernel under ``dev/records/minitpu/minitpu_rtl_oracle_<date>/``. *Check:*
  TB and emulator agree on every kernel in the set. Also: build
  ``minitpu_core`` once with #48's exact Verilator flags to learn the lint
  and ``--timing`` facts, and try ``MemPort`` under ``target="simulator"``.
- **M-R1 -- the shim and one kernel (2-2.5 d).** ``rtl/core_shim.sv`` +
  ``design.py`` (region, ``RTLModule``, driver kernels) + ``run.py``. *Check:*
  one looped GEMM's drain digest equals M-R0's, cycles reported beside the
  TB's with ``ROUND_TRIP_CYCLES`` named; the matrix row for "whole core"
  filled (simulator: match; Vitis: refused; Catapult/RTLGen/AMC: n/a).
- **M-R2 -- the gate (1 d).** The ``sim_kernel.py`` kernel set through the
  same path; a ``reproduce.sh``-style script from a clean checkout with the
  pinned clone; record + README D-21 evidence line.
- **M-R3 (optional) -- seam T (2-3 d).** Replace the shim with the AXI
  form; same Allo program, same digests; ``launch_status``/``dma_err``
  surfaced as status words.

Parallel tracks: **T1** (a-K) above; **T2** (c1) from the probe's
``mxu_shim.sv`` to a DIM-16 row-serialized ``RTLModule`` under
``examples/minitpu/rtl/mxu_ip/``, checked against ``ref_mxu`` and the U3
Allo MXU (2-3 d); **T3** #48 merge readiness: the two trivial conflicts in
``simulator.py``/``dataflow.py`` (resolutions in the probe), the
``--json-only`` ``validate_rtl`` port, ``-Wno-fatal``/``--assert``/``MAX_STALL``
/``VERILATOR`` knobs, the ``vitis.py`` ``m_axi depth=`` change (TinyTPU's
``vhls``/``catapult`` emission is byte-identical on the merged tree, but
every ``target="vitis_hls"`` project with static array args changes [P]),
its three Markdown docs moved into ``docs/source/backends/``, a Verilator pin
in ``dev/toolchains.rst``; the probe's merged tree passed ``test_df_unit``,
``test_region_stateful``, ``stress_isa`` 640/640 and ``bench_isa`` ALL EXACT
[P]; **T4** (c2) once the Catapult MXU is at II=1, with the delta generator
of §5 first.

**Fallback if #48 is not merged.** M-R0's driver *is* the fallback: Allo
assembles, calls MiniTPU's harness, checks -- "driven from Allo" but not
"wrapped as IP". A #48-free IP wrap would extend the ladder's own
``harness/rtl.py`` trace driver (C++ generated, ``--cc --exe``) with a credit-
pipe memory model and a whole-core driver (~2-3 d) [V mechanism
``u1-pilot:examples/minitpu/harness/rtl.py:137-232, 486-517``]; it would not
link into the Allo ExecutionEngine, so the Allo program would call it as a
Python-side ``IPModule`` stand-in -- weaker than #48's in-region call.

5. The ISA-version question
---------------------------

- (a) changes no latency: the core is the pinned RTL, so its version *is*
  ``v1-course`` and needs no delta. The only knob that moves its cycle count
  is the memory model (``ROUND_TRIP_CYCLES``, outstanding depth), which is a
  harness parameter, recorded beside the number, not an ISA quantity [V/I].
- (c1) changes nothing in MiniTPU; the Allo-side composition books the real
  MXU's declared numbers (12/22/82, 5/15/75) through D-20's legality
  properties, so a wrong booking is refused at composition [V D-20].
- (c2) is the case the seam exists for: the Allo MXU's push->valid, pop
  interval/beats, switch span and output-FIFO depth become one
  ``versions.list.<name>.deltas`` entry each (``{"what", "set": {dotted path:
  value}, "source"}``, overrides only), mapped to ``matrix.result_latency.vmatpush``,
  ``matrix.issue_interval.vmatpop``, ``rtl_params.WB_W_MPOP_LAST``,
  ``matrix.weight_switch.span``, ``resources.mxu_output_fifo.depth`` [V
  ``u1_matrix.rst:459-486``]. ``asm.schedule()`` reads ``isa_latency.json``,
  so a delta'd JSON yields a correctly scheduled image for the mixed core,
  and MiniTPU's own tbs become the check. Two facts constrain it: at
  ``b3ba0a4d`` the JSON has **no ``versions`` key** and ``gen_isa_delta.py``
  does not exist on ``main`` or ``u1-pilot`` [V]; and ``v1-course`` counts
  clock cycles from issue while ``v1``/E03 count issue cycles, so (c2)'s
  numbers are v1-course-shaped and must be written against the pinned
  commit, as an overlay our generator applies, until the pin moves (D-16).
  Name the version for what it is (e.g. ``allo-mxu-catapult-<allo commit>``)
  and keep the ``source`` field's provenance (allo commit, backend, clock,
  MiniTPU pin) [I, from the agreed schema].

6. Owner decisions needed
-------------------------

1. **Seam K first (core, no AXI) rather than the AXI top?** D-21 says "the
   core's AXI top"; §2 shows the core seam is ready/valid already and the AXI
   top adds only host plumbing. Proposed: K for M-R1/2, T as M-R3.
2. **Shim form:** RAM (``MemPort``; the DDR image is an Allo boundary array)
   vs stream-script (no #48 extension, slower operand transfer). Proposed:
   RAM if M-R0 shows it runs in the simulator, else stream.
3. **Extending #48 is allowed?** Needed either way: ``-Wno-fatal`` on the
   build step, a ``MAX_STALL`` knob, ``VERILATOR`` from ``dev/toolchains.rst``;
   possibly ``MemPort`` in the dataflow simulator.
4. **Acceptance:** bits only, or bits + cycles equal to the TB under a named
   memory model? Proposed: bits are the gate; cycles are reported, and a
   difference is a recorded finding.
5. **Kernel set for the gate:** one looped GEMM (M-R1) then ``sim_kernel.py``'s
   list (M-R2), or a smaller subset.
6. **Wording of the claim:** "all units are the real RTL" for (a); "mostly"
   only becomes true with (c1)/(c2). Which is the headline.
7. **Where the shim RTL lives** (``examples/minitpu/rtl/rtl/``, Allo's) and
   whether it may ever be offered to minitpu-comp as a ``tb/`` harness.
8. **(c2)'s version carrier:** a local overlay generator against the pinned
   JSON now, or wait for minitpu-comp's ``versions`` schema to land and re-pin.
9. **Priority of (c2)** relative to the Catapult II=1 work it depends on.
10. **#48 merge prerequisites** beyond the two conflicts: the ``--json-only``
    port (the PR cannot simulate on the pinned Verilator without it), the
    ``vitis.py`` ``depth=`` change on every ``vitis_hls`` project, Markdown
    docs to Sphinx, a Verilator pin; and whether the fork carries these as
    review requests to the author or as a follow-up commit after merge.
11. **PR #14** (choonsik1's scalar NB-stream guard) is already on ``main``
    as ``dd23b4ee``; close it as merged [P].

Appendix -- verified host facts
-------------------------------

Verilator 5.052 (conda-forge) at ``/work/shared/users/phd/sk3463/tools/verilator/bin``,
not on PATH; g++ 13 (gcc-toolset-13); Vivado 2019.2-2024.2 + 2026.1 and
Vitis HLS 2022.1-2024.2 under ``/opt/xilinx`` (ZCU104 part present; no board
access recorded); Catapult 2024.2 and Xcelium 24.03 per the inventory record;
no ``minitpu-comp`` clone on this host; the fork's bindings build here
(``LLVM_BUILD_DIR`` from ``conda activate allo``). MiniTPU's full-core TB
builds with ``verilator --binary --timing -Wno-fatal --top-module
tb_kernel_image -f src/minitpu.f`` and has no reusable C++ main [V
``tb/build_kernel_image.sh``].
