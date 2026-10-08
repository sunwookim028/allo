U1 findings matrix (README D-9)
===============================

One row per unit, one column per tool. Cells: **match**, **finding** (with its
class: bug, missing abstraction, workaround, semantic mismatch), **blocked**,
or **n/a**. Each finding links to its evidence. Started 2026-10-02 on
zhang-21; MiniTPU at ``b3ba0a4d``; harness on branch ``u1-pilot``.

For the owner's triage (D-9, second checkpoint)
-----------------------------------------------

What the pilot settled, and the calls that wait for the owner. Each call has
the agent's provisional default in brackets.

**What the pilot showed.** Every tool column has been tried on
``vpu_bf16_add``. The integer ``bits`` expression matches the RTL bit for bit
in the Allo simulator, RTLGen and AMC; Catapult's RTL from the ``native``
expression matches except on NaN encoding and one signed zero, and is 8 %
smaller than MiniTPU's RTL under the same DC flow. Nothing that works needed a
change to Allo's programming model. What failed were bugs (Allo core: unsigned
compares; SystemC emitter: seven; AMC: two silent miscompiles) and four gaps
in what Allo can say.

**Owner's decisions, 2026-10-02** (checkpoint 2, by question):

- Items 1, 2, 10: **merged** to ``main`` at ``db184ebc`` (``core-uint-compare``,
  ``core-scoping``, ``systemc-u1-fixes``; regression against a ``main``
  baseline: identical failure sets on every suite, all TinyTPU gates, byte-
  identical emission). Upstream issues filed: cornell-zhang/allo #617
  (unsigned compares), #618 (scoping).
- Item 3: **don't-care** for NaN sign/payload and the zero sign in verdicts,
  classification kept; a D-n proposal for stating a unit's NaN/zero policy
  is owed.
- Item 4: **reframed.** A combinational leaf is only a problem at a
  standalone top level; inside its sequential parent it is combinational
  logic between registers (the ALU already does this). Catapult can also emit
  a clockless CCORE (``latency_report_2026-10-02.rst``). Not an
  expressiveness gap.
- Item 8: **report first, constrain by contract** -- the owner's point that a
  compiler can consume what the HLS tool scheduled. D-10 drafted in the
  README (provisional) from ``latency_report_2026-10-02.rst``.
- Item 7 and 13: **filed.** AMC cornell-zhang/amc-dialect #126; MiniTPU
  sunwookim028/minitpu-tmp #18 (ALU and/or/xor), #19 (FIFO rule ignores loop
  trip counts; push+pop on empty), #20 (unchecked ``vmatpop``).
- U2: undefined behaviour is **masked and counted**; ``vpu_regfile`` pilot
  first (branch ``u2-regfile``), then a checkpoint.

**Owner's decisions, 2026-10-02, checkpoint 3** (after the ``vpu_regfile``
pilot, ``u2_regfile_2026-10-02.rst``):

- Async read: **pursue a combinational (same-cycle) read**, not a recorded
  deviation. Branch ``u2-comb-read``.
- ``Stateful``: **persists across calls on every backend**; sharing one
  ``Stateful`` across kernels is **refused**. README D-11. Implementation on
  a review branch ``core-stateful``.
- Memory ports (G1) and the dropped ``Memory`` fields: **draft a D-n** for
  "``compose`` declares memory ports" (modelled on AMC's port type) for the
  owner's review before code.
- Rest of U2: ``vpu_word_array``, ``vpu_fifo`` and the output FIFO in
  **parallel tracks**, with the pilot's ``trace``, ``ported`` and ``wire``
  expressions. B4/S6 (``core-uint-index``, in review) stay worked around.

**Owner's decisions, 2026-10-02, checkpoint 4** (on the D-12 study,
``d12_memory_ports_2026-10-02.rst``):

- Q1 (one owner per port): **under review in chat**; D-12 stays a proposal
  until the owner's wording.
- Q2: the VREG write port is **one writeback unit** owning ``vreg.w``, fed
  over channels (true to ``vpu.sv``, refusable at composition).
- Q4: the VMEM compute/DMA same-word collision is an **obligation on the
  Allo composition**, checked by stress cosim; reported to MiniTPU as an
  assembler gap (issue #21).
- Q7: the regfile prototype (3 reader + 1 writer units) plus
  ``vpu_word_array`` is the **U2 acceptance** for the memory-port item --
  with the owner's condition that the decomposition be target-neutral: the
  port declaration is the contract, and replica (FPGA LUTRAM-like), register
  array + muxes (ASIC cells) or SRAM macro are per-backend lowerings stated
  in ``memory.json``, never part of the declaration.

**Owner's decisions, 2026-10-02, checkpoint 5** (on the combinational read,
``u2_comb_read_2026-10-02.rst``):

- F2: a combinational output is an **explicit marker** (``Wire[T, comb]``),
  refused where a backend cannot build it. README D-13. Implementation on
  review branch ``backend-comb-wire``.
- F3: **look for an unreset storage form** in Catapult; accept the reset as a
  recorded deviation if none is well supported.
- Form e (thread for the write, combinational process for the reads) is the
  regfile's Catapult cell. F5 (``s.partition`` on a ``Stateful`` crashes)
  goes with the ``core-uint-index`` batch.

**Owner's decisions, 2026-10-02, checkpoint 6** (after ``vpu_word_array`` and
the FIFOs, ``u2_word_array_2026-10-02.rst``, ``u2_fifo_2026-10-02.rst``):

- FIFOs: for the **standalone** verdicts the explicit ring is the unit (the
  self-FIFO Catapult cannot schedule, C1, is a harness artefact: one kernel
  both pushes and pops). For the **composed** MXU at U3, a two-kernel
  ``Stream`` through Catapult is measured against ``vpu_fifo`` before the
  form is chosen, with the reset (pointer-only vs drained) and the illegal-
  program (drop vs stall) differences recorded either way. Branch
  ``u2-fifo-composed``.
- ``Stream.peek()``: decided by whether the MXU controller holds the head
  word across cycles before popping (then peek is demanded) or pops in the
  cycle it consumes (then ``try_get`` suffices). Checked on the RTL first.
- Registered-read storage: **pipe-as-data is the reference form** (the
  shift registers written out, as MiniTPU's RTL writes them; it gave exactly
  3/2/1 on every RTL backend); ``issue`` + ``latency=L`` is the form that
  *asks* a backend for a contract; D-12 lowers a port's declared read
  latency to exactly that pipe.
- Filed: AMC #126 (comment: A1, a failed schedule returns a design);
  MiniTPU #22 (F6, the VMEM sim model is not DC-synthesizable).

**Owner's decisions, 2026-10-02, checkpoint 7:**

- D-11 implementation (``core-stateful`` ``89222da8``, held for review): the
  old whole-core model's genuine sharing of ``vregs``/``vmem`` across kernels
  is kept alive by an explicit, **unchecked, simulator-only premise**
  ``@df.region(shared_stateful={...})``, mirroring ``deadlock_free_because``;
  HLS backends still refuse. **Transitional**: retired when D-12's ported
  memories replace ``microarch.py`` (D-4).
- D-13 implemented (``backend-comb-wire`` ``548ffe3f``, merged here): the
  ``comb`` regfile is bit-exact at read latency 0 / write-visible 1 on
  Catapult with no design patch, w16 and w256; DC area equals the hand
  patch. F3: an **unreset** storage form exists after all (the write as a
  clock-edge method, no reset action, ``-RESET_CLEARS_ALL_REGS no``): 0
  reset flops, bit-exact, DC **4,021.9 um^2 vs MiniTPU 4,021.7** -- parity.
  Not yet emitted (the kernel's writer is its thread); proposed as an
  explicit marker, owner's call below. Gap: ``@df.unit`` ports are
  Stream-typed in the netlist, so a ``Wire[T, comb]`` *unit* port is still
  refused; the marker lives on region-scope links until U3.

**Checkpoint 8, 2026-10-02:**

- Unreset storage: **explicit marker**, README D-14
  (``Stateful(..., reset=False)``); implementation on ``backend-unreset``.
- Composed MXU FIFOs measured (``u3_fifo_composed_2026-10-02.rst``): peek is
  **not** demanded (both consumers pop in the cycle the head is first
  visible, cited and traced); a two-kernel ``Stream`` schedules in Catapult
  at II=1, ``pop_data`` exact on every pop, area **below** MiniTPU's FIFO +
  output register on all three instances and 32-54 % below the ring. Its
  push-to-pop is 3 cycles against MiniTPU's 1 (to the latency contract at
  U3), and the output FIFO must use ``try_put`` (its overflow is a drop).
  Recommendation for U3: Streams for the MXU FIFOs, the ring stays the
  standalone-verdict unit.
- Fix batch ``core-uint-index`` (``18b4fcf8``, 12 fixes: B4-B7, S6-S9, C1,
  C-W1, F5) and ``core-stateful`` (D-11) held for review; ``latency-manifest-
  fix`` (C-M1) merged here.

**Checkpoint 9, 2026-10-02:**

- D-12 **adopted** (README): one owner per port; port kinds ``r``/``w``/``rw``;
  a declared read latency lowers to the pipe written as data; a collision
  rule per memory; AMC's ``count`` kept; Vitis refuses ported memories for
  now. Prototype next: the regfile as three reader units and one writeback
  unit.
- D-14 follow-ups approved: scope ``-RESET_CLEARS_ALL_REGS`` (or refuse
  mixing reset and unreset storage); fix unsigned ``Stateful`` emitted as
  signed ``ac_int``.
- ``core-uint-index`` + ``core-stateful`` **merged to main** (``9d48a90f``,
  one combined regression: identical emission and gates, the same failing
  sets, 63 new tests passing). Upstream issue filed: cornell-zhang/allo
  #619 (unsigned index).

**Owner's mandate for the absence of 2026-10-04** (~10 h):

- **U3 starts fully** on provisional decisions recorded for review (an
  explicit exception to "review each milestone before the next").
- **``u1-pilot`` merges to main when clean** (same impact check as the fix
  batches). ``main`` was synced into ``u1-pilot`` first (``d1e729b4``): one
  semantic clash fixed (D-11 state save/load vs D-13/D-14 signal storage);
  U1 verdicts 56/56 identical; U2 cells changed only where a merged fix
  applies (S6, S8, D-11 refusals), and the B4/S6 workarounds are no longer
  needed in the regfile.
- **If D-12's prototype fails its reverses-if, iterate** on alternative
  lowerings, recording each.

**Checkpoint 10, 2026-10-04** (the owner away; provisional):

- **D-12 prototyped, neither reverses-if fired** (``u2_d12_prototype_2026-10-04.rst``,
  branch ``d12-ports``): the register file as three reader units and one
  writeback unit is bit- and cycle-exact to ``vpu_regfile.sv`` on Catapult
  RTL (180,780/180,780, read 0, write-visible 1) under both lowerings
  (``server``: one storage kernel; ``replica``: a copy per read port), every
  port kernel at II=1; DC 4,061.6 um^2 vs MiniTPU 4,021.7 (+0.99 %). The
  two-port VMEM (narrow) as two units on one declared memory schedules at
  II=1, cycle-exact 67,717/67,717 on both ports. The 4,096-word RAM stays at
  II=2 on Catapult's sync dual-port model, as the one-kernel form did.
  Composition refuses every rule violation naming the port; Vitis refuses
  more than one owner; ``allo.memory.Memory(latency=, depth=)`` refused.
- **D-14 follow-ups** (``u2_d14_followups_2026-10-04.rst``, branch
  ``d14-followups``): ``RESET_CLEARS_ALL_REGS`` is scoped to the write
  process (Catapult allows Solution/Design/Process). The design-wide form
  **had dropped the kernel thread's own reset** -- a real hazard, now
  closed; mixing reset and unreset storage needs no refusal (Verilator
  mid-run reset: unreset kept 32/32, reset cleared). ``UInt`` ``Stateful``
  now emits unsigned on every backend (TinyTPU emission unchanged).
- **B3, a regression on ``main``**: D-11's csim state save in ``sc_main``
  sat outside ``#ifndef __SYNTHESIS__``, so Catapult ``go analyze`` aborted
  (CRD-135) on every ``@ Stateful`` design since ``core-stateful`` merged.
  Fixed on ``d14-followups``; goes to ``main`` with the next ladder sync.
- ``d14-followups`` merged into ``u1-pilot``; ``d12-ports`` conflicts with it
  in ``hls.py`` and ``EmitSystemC.cpp`` -- resolved on ``u1-pilot-sync2``.

**U3 track E landed** (``u3_composition_design_2026-10-04.rst``, branch
``u3-compose``): drafts **D-15..D-19** for the owner (engine interface;
parameters at instantiation; schedules that travel; optional modules;
derived-parameter legality), each with a prototype in
``examples/minitpu/template/`` (gate ``run_u3e``: 27 OK / 1 finding / 0 fail).
Measured: one PE source with two MAC engines matches in one region on the
simulator and csim (H10 holds); systolic and adder-tree engines behind one
declaration, each bit-exact against the contract reference *with its own
order* -- and 74/8,192 differ between orders at bf16 DIM 16 on random data, 0
on exact-sum data, 0 at int8 (H11: the order is part of the function);
twelve derived geometry numbers equal Phase 0's; five wrong declarations
refused. New tool findings E1-E6, the first a **silent bug**: a plain
``@df.kernel`` region builds and runs a ``Stream`` with one or no endpoint
and SystemC emits it (the ``@df.unit`` path refuses). Provisional: O3
answered as "MiniTPU instance ``sequential`` only; DotTree declares ``tree``".

**U3 track A landed** (``u3_track_a_2026-10-04.rst``, branch ``u3-xlu-sfu``):
SFU ``bits``/``staged`` match 462,144/462,144 on simulator and csim; the
reduction tree matches as one unit (N=16 and N=64) **and as 15 / 63 composed
adder units** (T2), with seven wrong latency sets refused by the derived
legality (H5); the transpose matches with a reset tile (recorded deviation;
its unreset form is refused in csim because D-14's lowering is Wire-only,
A6). H1 held: no vector ever differed; every failure was a tool finding.
Findings: **A3 bug** -- ``allo/passes.py:471`` erases any user symbol whose
name starts with ``gelu``/``layernorm``/``tril``; **A5 silent bug** -- csim
drops all but the first of the last iteration's stores to a 2-D output when
they come last; A2 constant globals missing from helper scope; A4 const array
to non-const callee param; **A7 missing abstraction** -- ``compose.Unit``
has no latency, so a composed tree's adder latency is a trusted parameter
(the D-10/D-15 hook). X2 ``ported`` blocked until D-12 is in the tree.

**Owner's decisions, 2026-10-04, checkpoint 11** (the owner back briefly):

- D-15..D-19: **reviewed in chat, one at a time**, like D-12.
- MXU push-to-output latency (O1/O2): **a manifest number** the assembler
  reads (D-10), not a contract; a Stream link's extra cycles are recorded.
- Large memories: **add an SRAM-macro path now** (OpenRAM in the codebase,
  FreePDK45), and **revisit FPGA-flavoured design choices** -- the multi-copy
  LUTRAM-style register file -- replacing them with ASIC-flavoured swap-ins
  (flop array + mux; SRAM macro) implemented and integrated as stated
  lowerings. Branch ``asic-memories``.
- MiniTPU findings from U3 Phase 0 and tracks A/E: **file all** on
  sunwookim028/minitpu-tmp.
- Catapult's target technology, confirmed for the owner: ASIC
  (``nangate-45nm_beh`` cells, generic sync-RAM models, DC on FreePDK45);
  ``ccs_fpga`` never used; the large VMEM's area is not an ASIC number until
  a macro path exists.

**U3 track B landed** (``u3_track_b_2026-10-04.rst``, branch ``u3-mxu``): PE as
one kernel (P1) and as a ``compose.unit`` with stream ports (P2) both match
the cycle model 360k/360k on simulator and csim; the systolic array as a
``mapping=[D,D]`` grid of the PE unit over Streams matches per cycle at DIM
2/4 (both) and 16 (simulator, 256 kernels); the composed MXU (front with skew
lines, grid, back with gather/``pack_bf16``/lane rings/pop) is contract-exact
at DIM 2/4 (both) and 16 (simulator) on legal, illegal (overflow drop, bank
overwrite, early commit) and random programs. Provisional P-11: a
Stream-linked composition is judged in *token time* on the untimed backends,
so cycle questions (H7) belong to track C. **Measured semantic mismatch**:
Stream FIFOs polled with ``try_put``/``empty()`` cannot be held to a cycle
(simulator: the pop engine never sees a group; csim: valid rises at 92 vs
19), so M1 uses explicit rings for the per-cycle verdict (P-12).
Findings: **compose bug** -- ``Unit.check`` reports a nested helper ``def``
and its params as undeclared free names; unit bodies need lazy annotations
(doc gap); ``Architecture._check`` refuses a dangling channel (H15 on this
path). Tracks C and D now take B's units.

**Checkpoint 12, 2026-10-04:** D-15 **approved** (README), with the two
wording edits (engine latency is a D-10 ``latency=``; "an adder-tree instance
declares ``tree``"). Implementation on review branch ``compose-engines``.
D-16..D-19 next, one at a time.

**Coordination with ``minitpu-comp``, 2026-10-04** (the owner's compiler/ISA
session; facts as of minitpu-tmp master ``a9757be``, its decisions in
``docs/DECISIONS.md`` / ``docs/COMPILER_PLAN.md``):

- **ISA versions**: ``v1-course`` = the frozen course ISA (our pin
  ``b3ba0a4d``; MXU push->valid **82**), ``v1`` = master (**85**), ``v2`` =
  ``docs/ISA_V2.md`` (frozen on defaults, §8.1): standard branches with 2 delay
  slots + one zero-overhead loop level replacing the 8-deep ``loop.begin``
  stack, post-increment addressing, counting semaphores with queued DMA
  descriptors, a sticky fault register. v2 RTL not started.
- **Interlocks**: v2 **adds them** (matrix-slot stalls, SREG scoreboard
  default 5, E03 freeze-on-stall); v1 stays assembler-enforced. The one RTL
  change in flight is **E03** (branch ``e03``, unmerged): units freeze on
  every issue stall; new ``en_i`` ports on ``sfu``, ``xlu_reduction_tree``,
  ``vpu_alu``, ``xlu_transpose`` and a 4-entry VMEM load landing ring
  (``docs/E03_PROTOTYPE.md``). Our U1-U3 models are of ``b3ba0a4d`` and would
  need those ports to follow E03.
- **Arithmetic contracts unchanged** in v1/E03/v2 (NaN canon, (+0)+(-0),
  multiplier flush, SFU no-flush, sequential acc24 + one ``pack_bf16``). Open
  there: vrecip wrap (ISA-N03 = our #33) and ``vrsqrt(+0)=NaN``; vmax/vmin
  NaN (ISA-X02, deferred); f32 accumulation across K tiles (deferred). v2
  default 12: writeback collisions become a **fault** via a priority mux.
- **Shared truth stays ``docs/isa_latency.json`` + ``isa_slots.json``**,
  everything generated and ``--check``ed from them (datasheet, asm.py
  constants, tb packages, an LLVM target's TableGen on branch
  ``llvm-spike``). A ``versions`` section is in progress. **Agreed seam**: a
  generator writes a version's deltas from our ``latency.json``/``memory.json``
  per (unit, backend, clock) into ``isa_*.json``, ``--check`` failing on
  disagreement; D-12's ports map onto their LLVM target's FuncUnits (write
  port, port C, matrix engines, weight banks). Schema to be agreed when their
  versioning branch lands (they will message).
- Our #19 and #20 are **fixed on master** as ``asm.py`` refusals ("M6-F3",
  "M6-F5/F6"); #18, #21, #22, #33-#36 open; #21/#22 bear on E03's landing
  ring.
- **Decision needed from the owner**: which version the ladder models
  (stay on ``b3ba0a4d`` v1-course through U5, then re-pin; follow master v1;
  or model v2 for U4's control), and the D-7 interlock note given v2.

**Checkpoint 13, 2026-10-04:** the owner decided the ladder **stays on
``b3ba0a4d`` (v1-course) through U5, then re-pins** (README D-16); **D-7 kept
as is**. E03/v2 become declared variants later.

**Checkpoint 14, 2026-10-04:** D-17 (instantiation binds parameters,
channels, engines) **approved** with the subsumption note; implementation
joins ``compose-engines``; the ``@df.unit`` type-parameter form is queued
front-end work. Drafts 3-5 (schedules that travel, optional modules, derived
legality) will be README D-18..D-20.

**Checkpoint 15, 2026-10-04:** D-18 (schedules travel with the function)
**approved** with the carried-directive edit; implementation joins
``compose-engines``. ``minitpu-comp`` follow-up: the owner decided **for v2
only** -- vrecip saturates the exponent and rounds (ISA-N03; today's worst
case 0.71 %, ``vrecip(1.0) = 0x3F7F``), vmax/vmin follow IEEE
``maxNum``/``minNum`` so a NaN operand loses (ISA-X02; matches
``arith.maxnumf``). v1-course keeps today's behaviour, so our ``b3ba0a4d``
references stay correct. v2's resource budgets are sized against the U280
FPGA (minitpu-tmp PRs #31/#32) -- relevant to RTLGen (FPGA-only), not to the
Catapult/DC ASIC columns.

**Checkpoint 16, 2026-10-04:** D-19 (optional modules as declared deltas)
**approved**; the owner asked for real use cases -- found: MiniTPU's SFU,
transpose, reduction tree, perf_counters; TinyTPU's accumulator file. The
SFU is the first implementation (``compose-engines``), with track A's S1
unit and U1's ALU in a VPU lane. RTL-side ``generate if`` + decoder refusal
to be synced with ``minitpu-comp`` later. One draft left: derived-parameter
legality (README D-20).

**Checkpoint 17, 2026-10-04:** D-20 (derived parameters as properties;
relations as legality; bookings vs manifest) **approved as drafted**. All five
composition drafts are now decisions: D-15 engines, D-17 instantiation, D-18
schedules travel, D-19 optional modules, D-20 derived legality (D-16 is the
ISA-version pin). Implementation: ``compose-engines`` (D-15/17/18/19 and
D-20's ``Architecture.parameters`` record); harness manifest-vs-booking on
``harness-bookings``.

**U3 track D landed** (``u3_track_d_2026-10-04.rst``, branch ``u3-openhls``):
RTLGen matches the SFU (462,144/462,144, II=1) and the PE (360,202/360,202
per cycle, II=1) and the tree at N=16 only in a textually inlined form
(II=4: 16 lane reads on a 4-port memory); AMC matches the PE (II=1 after
rewriting carried scalars as 1-element arrays; Vivado OOC 1,327 LUT, ~174
MHz on the adder recurrence) and is **blocked** on the tree (compiler
assertion on any chain of >= 3 carried registers). H2: neither tool has a
"ROM from file" -- RTLGen keeps table contents as a BRAM-style ``initial``
block widened to 32 b; **AMC emits the ROM empty** (contents dropped). New
**silent miscompiles**: RTLGen D1 -- a loop-carried shift pipe collapses to
one stage whenever the body is decomposed (nested call or inner ``for``),
even at reported II=1; AMC A-D1 -- constant-table contents dropped. Also
AMC A-D2 (crash: >= 3 carried registers), RTLGen D2/D3 (nested kernels are
sequenced instances; lane-array ports become few-port memories). Recorded,
not filed (P-10). All RTLGen/AMC numbers are FPGA estimates (u55c / Vivado
OOC); no ASIC run of their SV yet.

**D-15/17/18/19/20 implemented** (``u3_d15_d19_impl_2026-10-04.rst``, branch
``compose-engines``, ``allo/compose.py`` only): ``Engine`` records and
``Unit(engines=)`` slots with type/order checks; ``Instance(unit, name,
bind)``; engine directives applied once per engine, directives on inlined
functions refused; ``Option`` / ``with_options`` / ``isa_slots`` /
``check_program``; ``Architecture(parameters=<geometry record>)``. First
real optional module: ``template/vpu_lane.py`` = U1's ALU + track A's SFU,
writeback rebound ``alu_out -> sfu_out``; bit-exact with and without the SFU
(simulator 512/512, csim 128/128); ``vgelu`` refused without it; both
malformed options refused naming the channel. TinyTPU emission and gates
unchanged; ``run_u3e`` 26 OK / 1 finding. **Provisional calls for the
owner**: engine latency is a dict keyed by body (``{"mul": 0, "add": 3}``);
an engine or matrix-engine unit declares the order it computes, the
architecture declares the order its reference uses; a new ``Unit(calls=)``
declares a unit's helper functions (a compose unit could not call a helper
before, and D-15 now refuses a bare function as a parameter); memories can
be renamed per instance. Not done: the SystemC emitter half of D-18; the
lane is untimed (SFU latency 5 not modelled); track B's PE/MXU units have
no engine slot yet (the MAC is inline).

**Checkpoint 18, 2026-10-04.** Owner approved the compose implementation's
three calls: engine latency as a dict keyed by body; the engine declares the
order it computes and the architecture the order its reference uses
(mismatch refused unless ``accepts=``); ``Unit(calls=)`` declares a unit's
helpers. Memory renaming per instance follows from D-17.

**U3 track C landed** (``u3_track_c_2026-10-04.rst``, branch ``u3-catapult``):
every U3 unit through Catapult at 3.33/2.0 ns and DC/FreePDK45 beside
MiniTPU: SFU ``bits`` 462,144/462,144 at latency 5 cycle-equal, **0.74x**
MiniTPU's area (Catapult trims the ROMs; H2: four ROM components inferred);
tree N=16 1.63x; transpose 1.48x; PE ``bits`` 1.66x; array DIM 2/4 2.1-2.3x;
composed MXU DIM 4 contract-exact, 1.32x, push->valid 1,631 cycles (vs 22:
the token-time cost of Stream links, H7, reported to the manifest per
checkpoint 11); composed tree T2 2.80x with a tail stall (last 4 rows never
emitted). H4: one ``unroll`` on the shared helper suffices (C10 does not
recur). **C9**: the polled-Stream FIFO MXU **fails the contract on RTL**
(pops 0/2,423; valid from cycle 0; overflow 17 vs 16) -- a bug candidate in
the ``empty()`` sideband; the explicit ring holds. C1-C8: 2-D lane arrays
and conditional port access become RAM pins; manifest latency is Pop->Push
while unit latency is rows + I/O (D-10 gap); free schedules put multi-output
Pushes on different edges; ``Stateful`` arrays are RAMs unless partitioned.

**SRAM-macro path landed** (``asic_memories_2026-10-04.rst``, branch
``asic-memories``): OpenRAM ``b2b069ce`` pinned under ``tools/`` with its
bundled FreePDK45 (own conda env; routers off: 32x64 in 56 s, 512x64 in 19
min, 1024x64 in 53 min, 4096x64 one-macro unfinished), Catapult MemGen libs
(plain licence), ``lc_shell`` for DC. D-12's two-port VMEM on OpenRAM 2RW
macros: **II=1 refused on a RW port** (6 variants; same as the ccs model),
II=2 cycle-exact at stretch 2. DC: 512x64 **macro 105,968 um^2 vs MiniTPU
flop-mapped 262,798** (0.40x); 32x64 macro 30,474 vs 17,792. Lowerings as
stated ``impl=``: ``registers`` (default), ``sram`` (refuses by port name
what the macro cannot honour), ``replica`` FPGA-only. Open: bank the
4,096-word VMEM 8 x w512; a 1R1W macro to reach II=1; the ``registers`` form
carries 1.8x MiniTPU's flops.

**E03 merged on MiniTPU master** (``f9812f3``, RTL ``3c0a5ce``,
``docs/E03_PROTOTYPE.md``, bitstream ``0xB108D8C3`` on 3 boards): ``en_i``
issue-time clock enable on sfu, vpu_alu, xlu_reduction_tree, xlu_transpose,
vpu_bf16_add_pipe, vpu_bf16_mul (reset has priority; pipeline registers
hold when low); VMEM and the MXU engines keep running; a VMEM landing ring
of depth READ_LATENCY+1; latencies now count issue cycles; arithmetic and
``isa_latency.json`` unchanged. Our pin (D-16) unaffected; E03 becomes a
declared variant after U5 (``en_i`` as a declared port on those units).

**Session handoff, 2026-10-04 (session ``allo-minitpu`` ending at its limit).**
State: ``main`` = ``9b33ee03`` (ladder through U3 wave 1, D-12..D-20 decided,
D-12/D-14/B3 code). ``u1-pilot`` = ``8ceb3011`` = main + ``compose-engines``
(D-15..D-20 in ``allo/compose.py``) + ``asic-memories`` (``impl=`` lowerings,
``Sram``) + tracks C/D records; Python-only beyond main (no ``mlir/``
change), so **not yet regressed for main as a whole**. **Done, not yet merged:** ``core-fixes-3`` @ ``39940f4e`` (based on
``87420c94``; applies cleanly to ``9b33ee03``): A3, A2, A4, A5, E1, E2, E3,
E5 (= E4) fixed in eight commits, 36 new tests; TinyTPU emission and gates
identical, same failing sets as main; two existing tests edited under E1;
draft upstream issues for A3/E5/E1 in ``dev/records/limitations/
u3_fixes_2026-10-04.rst``. Merge it into ``u1-pilot`` first. Still in flight: a Sonnet regression of the compose tip (``wt-reg4``, base
``wt-reg4-base``; it does not cover ``asic-memories``), ``asic-memories-2``
(banked 4,096-word VMEM; a 1R1W macro for II=1; ``wt-asicmem2``).
Next session, in order: (1) read this file's checkpoints 11-18 and the
records they cite; (2) when ``core-fixes-3`` and ``reg4`` report, merge the
fix batch into ``u1-pilot``, run ONE regression of ``u1-pilot`` vs ``main``
(the usual: emission hashes, gates, test-set diff) and fast-forward ``main``
(the owner's mandate covers it); (3) merge ``asic-memories-2`` when it
lands; (4) U4 (control: sequencer, DMA) Phase 0 against ``b3ba0a4d`` (D-16),
using D-12 ports for VMEM's DMA side and the SRAM path; (5) the queued
front-end work: ``@df.unit`` type parameters (D-17), the schedule attribute
(D-18), the SystemC half of D-18, E03 as a declared variant after U5.
Open with the owner: push RTLGen's ``bits`` SV through DC for comparable ASIC
numbers (asked, unanswered). Coordination: ``minitpu-comp`` will send the
``versions`` commit; the manifest -> ``isa_*.json`` schema is agreed in
principle. Worktrees not in flight are removed; ``wt-u1`` is the integrator
and has its own build at ``8ceb3011``.

**Agreed seam with ``minitpu-comp``, 2026-10-08** (their master ``7676911``;
the ISA JSON moved from ``docs/`` to a top-level ``isa/`` on master
``f1e978e`` (move commit ``08b19dd``): ``isa/latency.json`` (``versions``
unchanged inside) and ``isa/slots.json``; ``isa/experimental.json`` is new and
not ours to read; a future ``isa/faults.json`` for v2. Their ``make host``
now fails on any code reading under ``docs/``. **Our generator targets
``isa/latency.json`` and pins ``--check`` to ``f1e978e`` or later**):

- Our generator ``gen_isa_delta.py`` emits ``versions.list.<name>.deltas`` as
  a **list** of ``{"what": str, "set": {<dotted path>: value}, "source": str}``,
  one entry per quantity, provenance (allo commit, backend, clock, MiniTPU
  pin) in ``source``; an ``unresolved`` list stays outside the delta.
  **Overrides only**: a path the base lacks is refused; new quantities are
  added to their base first, with a reader.
- Mapping: ``push_to_valid`` -> ``matrix.result_latency.vmatpush`` (85; v1-course
  82); pop interval -> ``matrix.issue_interval.vmatpop`` (1; 4); pop beats ->
  ``rtl_params.WB_W_MPOP_LAST`` = ``WB_W_MPOP_FIRST`` (3) + beats - 1 (no
  separate beats key); ``switch_span`` -> ``matrix.weight_switch.span`` (75);
  output FIFO -> ``resources.mxu_output_fifo.depth`` (64 result rows, one FIFO
  per lane). Per-unit ``rtl_params.WB_W_{ALU 5, SFU 7, REDUCE 15,
  LANE_REDUCE 11, VLD 6, TXOUT 3}`` are **W = unit latency + VPU_WB_STAGES**
  (the cycle the op claims the VREG write port), not unit latencies: our
  manifest latency must add the writeback stage count before mapping.
- Units: cycles from the issuing bundle. v1 (E03) counts ISSUE cycles; v1-course
  counts CLOCK cycles from issue (units run free through stalls; the scheduler
  treats them as lower bounds). Our free-running RTL measurements at
  ``b3ba0a4d`` are directly comparable to v1-course.
- Landing: a PR to minitpu-tmp touching only ``isa/isa_latency.json``'s
  ``versions.list.<name>`` and the regenerated outputs
  (``tools/gen_isa_doc.py --write``); their ``make host`` and ``--check``
  gates run on it; our ``gen_isa_delta.py --check`` re-derives against the
  pinned commit. Nothing blocks before U5.

**Checkpoint 19, 2026-10-08:** README D-21 -- the owner's short-term
baseline: ``examples/minitpu-rtl/`` claiming "MiniTPU RTL obtained from Allo,
mostly as RTL IP" (whole-core shim planned first, the hybrid real-``mxu.sv``
in parallel), and the inverse angle, an Allo-synthesized MXU integrated into
MiniTPU under its own ISA version. PR #48 (``RTLModule``) is being probed
read-only (trial merge, its tests on this host, an ``mxu.sv`` descriptor);
merging it is the owner's review. An audit of the 30 open fork issues
against this month's fixes and decisions is in progress (no posting).
Also started: U4 Phase 0 (prep only) and the TinyTPU-as-instance track.

**Checkpoint 20, 2026-10-08:** README D-22 (TinyTPU = the toy instance for
communication; MiniTPU = the design driver; the instance track is a probe,
not a requirement). Issue audit recorded
(``dev/records/fork_issue_audit_2026-10-08.md``); the owner approved closing
6 issues and PR #14 and posting 12 update comments (delegated). Running:
``u1-pilot-sync3`` (integration), ``tinytpu-example`` (README cleanup),
``u4-phase0``, ``tinytpu-instance``, PR #48 probe, whole-core shim plan,
``minitpu-rtl-mxu`` probe. Next session: merge what landed, regress, ff
``main``, prune the 13 merged branches, add the README naming paragraph
(after the cleanup lands).

**Landing procedure for ISA deltas (minitpu-comp, 2026-10-08).** Target
``sunwookim028/minitpu-tmp`` branch ``master`` (``e5c2222`` at the time; the
only maintained branch), from a pushed feature branch. The PR touches only
``isa/latency.json`` ``versions.list.<name>`` entries plus what ``python3
tools/gen_isa_doc.py --write`` regenerates (the generated docs,
``board_package/asm.py``'s marker table if it changes, ``tb/isa_*.svh``), and
must pass ``make host`` (includes ``gen_isa_doc --check`` and
``check_no_docs_reads``). Each new version carries ``status: experimental``
and no ``bitstreams`` entry until a board build exists (the runtime refuses an
image on an unlisted bitstream). A board_package refactor is landing
concurrently (code generation moving to ``compiler/``): on conflict, rebase
onto master and rerun ``gen_isa_doc.py --write``; never hand-merge generated
files. A ``bitstreams`` entry (BUILD_ID) is added only after the image has
been validated on a board: ``make identity`` reads that id with fault 0 and
the board suite passes twice after programming (MiniTPU's CLAUDE.md rule);
the entry's text cites the board, date and commit like the ``v1`` entries.
Opening the PR needs the owner's approval.

**Checkpoint 25 (2026-10-08, the pin is ``v1``; U4 B and C landed).**
minitpu-comp confirmed track B's measurement: ``b3ba0a4d`` is ISA ``v1``,
not ``v1-course`` (v1-course = Lab 2's tree / minitpu ``613190d`` timing;
``05e1bdf`` one-beat vmatpop + result latency 85, ``49d895d``, ``3bcf0b7``
are all in the pin). D-16 re-keyed in the README; the deltas
(``allo-selftimed``, ``allo-mxu``, later ``allo-mxu-vitis``) are relative to
base ``v1``. Open: the ladder's MXU push->valid 82 is at the MXU port, the
ISA's ``result_latency.vmatpush`` 85 counts from the issue edge -- the offset
is being measured on the pinned RTL before any delta maps one onto the
other; the landing PR (``isa/latency.json`` versions + regenerated outputs)
waits for the owner's approval. Merged into ``u1-pilot``: U4 track C
(``4446f019``: DMA units match; findings C1 ``done`` name clash, C2 unreset
state refused in Stream kernels, C7 local arrays on the SystemC thread
stack) and track B (``f4e13af3``: issue/command/write-back match; D-23 holds
at command-stream depth >= 2, fails at 1; D-24: cycle-locked reproduces W
with no delta, self-timed cannot run the scheduled programs unchanged;
F-B4 simulator hang on an unfinished producer). ``main`` = ``33b7cbf6``
(the TinyTPU demo with cosim working here). Fork issue #51 (harness
``calendar.py`` shadows the stdlib).

**Checkpoint 24 (2026-10-08, D-21 re-scoped; the FPGA route started).**
Owner: the RTL-wrapped baseline is worth doing only if meaningful. Provisional
re-scope (README D-21 to be amended when M-R1 lands): the whole-core wrap is
not a demo but the *substitution spine* for U4/U5 -- once ``minitpu_core.sv``
runs inside an Allo region against the testbench-digest oracle, each
Allo-modelled unit (the MXU first, then U1-U3's VPU units, then U4's control)
replaces its RTL counterpart inside the same region, the same oracle on every
step, until the core is all Allo (U5). Its probe target is the tool gap that
PR #48's transaction seam cannot express a per-cycle sideband (``vpu_ctrl_t``),
so unit-level substitution needs a cycle-level mixed RTL/Allo cosimulation
seam; M-R3 becomes that first swap (the Allo MXU inside the wrapped core),
not the AXI top. Issues filed: minitpu-tmp #41-#47, cornell-zhang/allo #621
#622, fork #49 #50. Started: ``minitpu-fpga-mxu`` -- Allo's MXU through
Vitis HLS into MiniTPU's own ZCU104 bitstream flow (``make bitstream``,
Vivado 2023.2), held to ``tb_mxu_single_port``, published as
``versions.list.allo-mxu-vitis``; the Catapult hold on (c2) does not apply
to the FPGA route. Layout call pending the owner: fold ``examples/minitpu-rtl/``
into ``examples/minitpu/rtl/`` (one design tree, two routes).

**Checkpoint 23 (2026-10-08, U4 wave 1 started; D-23/D-24).** The owner
answered the U4 and merge questions: D-23 (the VPU command as three
valid-qualified slot Streams plus four declared resources, payload gating a
recorded deviation, with the condition that blocking streams cost neither a
deadlock nor issue rate) and D-24 (cycle-locked first, self-timed compared);
both TinyTPU branches merged as they were; all findings filed (MiniTPU to
its issue repo, two Allo core bugs upstream, two fork defects). Merged into
``u1-pilot`` without conflict: ``u4-phase0``, ``minitpu-rtl-m0``,
``minitpu-rtl-mxu``, ``tinytpu-example``, ``tinytpu-instance`` (quick gates:
compose/instance/tutorial tests 23 passed, ``gen_isa --check`` OK); their
branches and worktrees deleted. Running: PR #48 merge + follow-up fixes
(branch ``pr48-merge``), U4 tracks A ``u4-front``, B ``u4-issue``, C
``u4-dma`` (one worktree each, own bindings), the issue filing, and the
``u1-pilot`` -> ``main`` regression for this batch. Held per the owner: Allo's
MXU into MiniTPU's tree (c2) until Catapult II=1 and the ``versions`` seam;
its DIM-4 link-depth sweep is the first step when it resumes.

**Checkpoint 22 (2026-10-08, owner decisions on D-21; U3 on main).**
The owner settled the four D-21 choices (README D-21, last bullet): core seam
first; merge PR #48 with a fork follow-up fixing its debts; the whole-core
claim is the headline, the hybrid in parallel, Allo's MXU into MiniTPU after
Catapult II=1; bits are the gate, cycles reported. ``u1-pilot-sync3``
regressed clean (132 passed, emissions byte-identical, all gates and
``check.py`` verdicts equal to the baseline, pytest 1004/78 vs 924/80 with
only ``tests/act/test_bindings`` changing, fail -> pass); ``main`` and
``u1-pilot`` fast-forwarded to ``552db66a``; sixteen merged branches deleted.
M-R0 landed on ``minitpu-rtl-m0`` (``e25cc1be``): 52 launches, the RTL halts
on all, the emulator's drain is bit-identical on 29 and is a functional model
for the rest (float64 MXU, exact SFU), so the bit oracle is the testbench's
digest; ``minitpu_core`` builds standalone under #48's flags with no
warnings; #48's ``MemPort`` runs under the simulator. Landed for review:
``u4-phase0`` (``c66ac2fa``; eleven control units matched, W = latency +
``VPU_WB_STAGES`` measured three ways, findings F-W1..W4 for MiniTPU),
``tinytpu-instance`` (``12d11cc6``; the instance reproduces the published
cycles 175/265/421/482/674 and the mutant score; findings F1/F3 core bugs,
F8 engine swap reaches one unit, F13 host ``ld`` 2.30 breaks every Vitis
cosim link), ``tinytpu-example`` (``5d74701f``; cosim not reproducible on
this host for the same ``ld`` reason), ``minitpu-rtl-mxu`` (``2aaad780``;
MiniTPU's ``tb_mxu_single_port`` passes against the Allo MXU at DIM 2,
depth-4 links give the first declarable latency: push->valid 20, pop 9).

**Checkpoint 21 (2026-10-08, two D-21 probes landed; owner review pending).**
``pr48_probe_2026-10-08.md``: PR #48 merges onto ``main`` with two one-hunk
conflicts and leaves every existing design's emission byte-identical; as
submitted it simulates nothing on the pinned Verilator 5.052 (``--xml-only``
is gone; a ~50-line ``--json-only`` port gives 60/60); raw ``mxu.sv`` is
refused at the first pin (enum, packed arrays, >32-bit payloads), a 60-line
shim at DIM=2 runs bit-exact through its Verilator transactor as a *tile op*;
vmatload/vmatpush/vmatpop as separate instructions on one MXU are
inexpressible (fixed transfer counts, one instance per object, ``ii=0``).
``minitpu_rtl_plan_2026-10-08.rst``: the whole-core baseline should wrap
``minitpu_core.sv`` (no AXI; a ready/valid credit memory pipe) rather than the
AXI top; M-R0 (oracle digests from MiniTPU's own TB, no #48 needed) then M-R1
(one GEMM bit-identical) / M-R2 (the ``sim_kernel.py`` set); 3.5-4.5
agent-days; eleven owner decisions in its §6, the first being core seam vs
AXI top and whether #48 may be extended. Provisional call while the owner is
away: M-R0 may start (it touches neither tree); nothing else on the
``minitpu-rtl`` track until the owner answers §6 and decides on #48.

**Anchors (for anyone resuming this work), 2026-10-08.** MiniTPU is the
practical design driver; TinyTPU is the toy instance for communicating the
programming model (D-22). The ladder is a probe of the tools (D-9): findings
first, the RTL match as the goal. Decisions D-9..D-22 are in the README;
the owner reviews each programming-model change in chat, one at a time, and
proceeds on recorded provisional calls when away. The checkpoint that closes
this stage: U1-U3 on ``main`` with their decisions implemented, every branch
merged or deleted, the handoff current; then U4 (control) against
``b3ba0a4d`` (D-16), with the ``minitpu-rtl`` baseline (D-21) in parallel.
Progress: U1-U3 done; U4 Phase 0, the TinyTPU-instance probe, the ``main``
integration, the TinyTPU example cleanup, PR #48's probe and the two D-21
studies are in flight (branches listed in the handoff below).

1. **Merge the unsigned-compare fix** (``core-uint-compare``, B1-B3)? Zero
   measured impact on every gate, test and TinyTPU emission. *[merge; file
   the drafted upstream issue]*
   *Checked together, 2026-10-02:* ``u1-pilot`` with both
   ``core-uint-compare`` and ``core-scoping`` merged (branch
   ``u1-with-core-fixes``, ``fd076a86``) gives verdicts identical to
   ``u1-pilot``'s on every unit, variant and backend, and the 66 regression
   tests of all three fix branches pass together
   (``u1_integration_2026-10-02/``).
2. **Merge the SystemC emitter fixes** (``systemc-u1-fixes``, eight bugs,
   each with a compile-and-run regression test)? EVA bit-exact; TinyTPU
   SystemC csim unchanged; TinyTPU's Vitis and Catapult emission is
   byte-identical to ``main`` (sha256 ``6bc774bc...``/``ade1ab5d...``);
   ``tests/test_vhls.py`` gives the same 7 failures (host ``libstdc++``) and
   28 passes on both trees. With them, SystemC csim of ``bf16_add`` runs the
   full stimulus in 5.5 s: ``bits`` matches 251,936/251,936; ``native``
   differs only on NaN encodings (Catapult ``ac::bfloat16`` gives
   ``0x7FFF``/``0xFFFF``) and ``(+0)+(-0)``. *[merge]*
3. **NaN and signed-zero rules.** MiniTPU, Allo's simulator, Catapult and
   RTLGen give four different NaN encodings and two answers to ``(+0)+(-0)``.
   Options: (a) treat NaN payload/sign and the zero sign as don't-care in the
   harness; (b) give Allo a way to state a unit's NaN/zero policy on its float
   type; (c) require ``bits`` expressions wherever the RTL's rules matter.
   *[(a) for U1 verdicts, with the classification kept; (b) as a D-n
   proposal, since every machine has its own policy]*
4. **A combinational unit.** No backend can express "ports a, b, result, no
   clock": Allo's nearest is ``Wire`` ports (still clocked), RTLGen and AMC
   always add start/done. Is a zero-latency unit a programming-model feature
   to add, or is latency >= 1 acceptable as a recorded deviation for leaves?
   *[recorded deviation for U1; revisit at U3, where leaves compose]*
5. **Bit-level notation gaps** (concatenation, reduction-OR, ``[hi:lo]``
   slices; expression widths sized bottom-up; shifts by >= width). Add them,
   or keep them as recorded workarounds? *[proposal only; they cost
   readability, not correctness]*
6. **Pipeline flush mode for Catapult** (default stall leaves the last
   element of a finite stream stuck; ``style=`` refused for SystemC).
   *[emitter fix: emit the flush style for finite streams]*
7. **Report AMC's silent miscompiles** (A5 ``a or b or c`` drops ``c``; A7 a
   two-scalar loop exits with the wrong one) to AMC's author? *[yes, after the
   owner's go-ahead]*
8. **Declared latency on a unit** (``u1_pipe_2026-10-02.rst``). Allo cannot
   state it; the simulator is untimed and csim's cycles are the handshakes'
   (2 per kernel, whatever the unit). Unconstrained, Catapult picks the
   latency from the clock (acc24: 2 at 5 ns, 3 at 2 ns, against MiniTPU's 3);
   a hand-patched I/O cycle constraint pins it exactly (measured = declared
   for 1, 2, 3, 6) and Catapult refuses an infeasible one. Adopt the proposal
   (``latency=L, ii=`` on a kernel/unit; Catapult honours via ``cycle set
   -from``, csim and the simulator report "unchecked"/"untimed", RTLGen/AMC
   refuse; the harness checks it on RTL only)? The multipliers reached the same finding
   independently (``vpu_bf16_mul_pipe``: a ``Stream`` between stage kernels is
   a FIFO, not a register; at depth 1 it halves throughput).
   *[yes, as a D-n proposal;
   csim stays "unchecked" until U3 needs cycle-locked composition]*
   *Followed up 2026-10-02* (``latency_report_2026-10-02.rst``, branch
   ``latency-report``): every RTL backend already *reports* the scheduled
   latency exactly (Catapult 75/75 builds, Vitis 10/10, RTLGen 17/17, AMC
   14/14 against measured RTL), so the proposed D-10 makes report-and-check
   the default and ``latency=`` the exception; Catapult CCORE gives a
   clockless ``bf16_add`` (item 4).
9. **Two more SystemC bugs from the multipliers** (S4: ``bf16 -> f32``
   widening does not compile; S5: a ``UInt(24)`` port cannot be read back).
   *[fix on ``systemc-u1-fixes`` with a regression test each]*

10. **Front-end scoping miscompiles** (``vpu_alu``, ``u1_alu_2026-10-02.rst``;
    both reproduced independently). C3: a function reused from another module
    reads the *caller's* global of the same name (``K = 3`` in the engine's
    module, ``K = 5`` in the caller's: Allo computes ``x * 5``). C4: a
    module-level numpy array silently replaces a kernel parameter of the same
    name. Neither raises. Every cross-module unit reuse -- which composition
    from U3 on rests on -- is exposed. **Fixed** with C1, C2, C5 and one more
    silent case (a slice bound read from a same-named global) on
    ``core-scoping`` (``b8cf732e``, 17 new tests, 15 fail on ``main``):
    TinyTPU emission byte-identical, every gate and test outcome unchanged,
    and a trace of all 825 function builds in the test run shows nothing
    relied on the old resolution. Upstream has the same code
    (``dev/records/limitations/frontend_scoping_2026-10-02.rst`` on that
    branch, with a draft issue). Left open: slicing a kernel *parameter*
    (``a[1:2]``) crashes LLVM in the simulator on ``main`` too; ``Stateful``
    globals are named by variable name (two ``test_stateful`` failures on
    ``main``). *[merge with item 1; file both upstream issues]*
11. **ALU engines, by reuse.** All three compositions match the RTL bit for
    bit, but function-level engine swapping works only by naming convention
    (C1, C2, C11), a ``@df.unit``'s sizes freeze at decoration (C9), and a
    reused function's schedule does not travel with it (C10). *[proposals,
    with item 8, for the U3 composition design review]*
12. **Catapult track, all U1 units** (``u1_catapult_units_2026-10-02/``).
    Every ``bits`` variant is bit-exact in Catapult RTL against MiniTPU on
    the full stimulus at II=1, with no hand-patch to the emitted SystemC
    (the pilot's two emitter bugs and S4 are fixed). The declared latency is
    met on all three pipelined units through Connections ports (item 8's I/O
    constraint). Same-flow DC: Catapult's RTL is 0.96-1.72x MiniTPU's area.
    New: a ``Wire``-port unit's latency **cannot be pinned** (N2), and
    latency 0 is refused (N3). *[amend item 8's proposal: Catapult honours
    ``latency=`` on Connections ports and refuses it on Wire ports]*

13. **MiniTPU findings for its owner** (D-7: changing MiniTPU is the owner's
    call). ``vpu_alu`` does not implement AND/OR/XOR (they return ``a``; the
    decoder never issues them; ``docs/UNITS.md`` lists them). From U2 Phase 0
    (``u2_phase0_2026-10-02.rst``): the assembler's output-FIFO rule counts
    pushes in program order and ignores loop trip counts, so ``loop.begin 17
    { vmatpush }`` passes and the RTL silently drops the 17th push; a
    ``vmatpop`` with nothing pushed is accepted with no assertion;
    ``vpu_fifo`` push+pop on an empty FIFO loses the word (contradicting
    ``vpu_fifo.sv:5``; unreachable from the MXU). *[report to the owner; no
    change to the pin]*

``vpu_bf16_add`` (pilot)
------------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - Allo simulator, ``native``
     - **finding, semantic mismatch**: 250,955/251,936 match; the 981 others
       are the RTL's two non-IEEE rules (NaN always ``+0x7FC0``; ``(+0)+(-0) =
       -0``)
     - ``harness/check.py``; ``docs/source/designs/minitpu.rst`` ("The unit
       ladder"). Allo's ``bfloat16`` has IEEE semantics with no way to state
       a unit's own NaN/zero rules.
   * - Allo simulator, ``bits``
     - **match** 251,936/251,936, **after a workaround for a core bug**:
       **finding, bug B1** -- every ``<``/``<=``/``>``/``>=`` between unsigned
       integers is lowered as a *signed* ``arith.cmpi`` (``allo/ir/builder.py``
       picks the predicate from the MLIR type string, which is signless), so
       ``uint8`` ``200 > 100`` is 0 in the simulator and LLVM backend but 1 in
       HLS C++. Also B2 (bug: comparison typed as its operands, not ``uint1``),
       M1-M3 (missing abstractions: bit concatenation, reduction-OR, ``[lo:hi)``
       slices against SV's ``[hi:lo]``), M4-M5 (latent semantic mismatches:
       expression widths sized bottom-up; shifts by >= width)
     - ``u1_bf16_add_bits_2026-10-02.rst``. B1 confirmed independently. Fixed
       with B2 and a third bug in the same function (B3: every signed
       ``Fixed`` compare took unsigned predicates) on branch
       ``core-uint-compare`` (``716c7baf``, 17 new tests, 12 fail on main):
       no gate, test or TinyTPU emission changes against ``main``. Upstream
       has the same code. Held for the owner's review
       (``dev/records/limitations/uint_compare_2026-10-02.rst`` on that
       branch).
       Everything else in the RTL transcribed directly, including the nested
       ``leading_zeros17`` function.
   * - SystemC csim, ``native``
     - **finding, bug** x3 (emitter): the ``ac::bfloat16`` ``sc_trace``
       overload is in the global namespace, invisible to ADL, so any bf16
       Connections port fails to compile; the testbench feeds all of input 0
       before input 1 from one thread, so any unit with two interleaved
       streamed inputs deadlocks (0 outputs, even at n=200); float values
       cross the testbench as decimal text (``nan``/``inf``). For ``bits``
       (``uint16`` ports): S1, unsigned ports emitted as signed ``ac_int`` (any
       ``uint16`` region fails to compile); S2, a nested function returning
       ``UInt`` gets a signed result buffer
     - Fixes on branch ``systemc-u1-fixes``. The existing bf16 test only
       checked emission, so the compile error was never seen.
   * - Catapult csyn / RTL, ``native``
     - **finding, semantic mismatch** after two emitter bugs worked around:
       RTL 249,907/251,936 vs MiniTPU (249,908 vs IEEE; ties and subnormals
       exact); the rest are NaN encodings (Catapult ``0x7fff``/``0xffff``)
       and ``(+0)+(-0)`` (+0). II=1 latency 2 with ``s.pipeline``; 3 cy/vector
       without. Same-flow DC (FreePDK45, 3.33 ns, output-registered, wire
       ports): Catapult **813.4** vs MiniTPU **883.9** um^2. Bugs: include
       order (CRD-135 ``Marshall``), Wire-only kernel lacks ``wait()``
       (CIN-123). Missing abstractions: a combinational unit (closest is
       ``Wire`` ports, still clocked); pipeline stall/flush mode (default
       leaves the last element stuck; ``style=`` refused for SystemC)
     - ``u1_bf16_add_catapult_2026-10-02/README.md`` (nine findings). Emitter
       bugs passed to ``systemc-u1-fixes``. Catapult's own area score ranks
       the variants differently from DC.
   * - RTLGen, ``native``
     - **finding, semantic mismatch + missing abstraction**: 249,995/251,936
       match. The others are NaN encodings (RTLGen keeps sign and payload)
       and ``(+0)+(-0)``. The adder is an extern Vivado IP with no RTL body,
       so cosim checks a DPI-C model, and ``add_rtl_model`` is unimplemented
     - ``u1_bf16_add_rtlgen_2026-10-02.rst`` (F1, F2)
   * - RTLGen, ``bits``
     - **match**: 251,936/251,936 bit-exact, II=1, N+6 cycles. Findings: no
       combinational kernel (latency >= 1 with start/done); a ``for`` loop
       like the ``.sv``'s gives II=48 unless it is unrolled; 32-bit
       temporaries are not narrowed
     - ``u1_bf16_add_rtlgen_2026-10-02.rst`` (F3-F5)
   * - AMC, ``native``
     - **blocked**: no bf16 in its frontend or its operator library; f32
       needs DesignWare models that are not in the repository
     - ``dev/records/open_hls/amc_exploration_2026-10-02.rst``
   * - AMC, ``bits``
     - **match** 251,936/251,936 bit-exact, through AMC's own frontend and
       through our frontend's ``df.region`` MLIR. II=1 with N+2 cycles needs
       ``s.unroll`` + ``s.pipeline``; as written it takes 21 cycles per
       element (P1). Synthesis (N=16, 10 ns): 268 LUT, 21 FF. This holds
       **only after 7 kernel edits**, which come from these findings:
       **bugs** A1 (no ``not``), A2 (``x[k]`` dead on Python >= 3.9), A3, A4
       (scalar-returning call aborts), **A5 and A7 (silent miscompiles:
       ``a or b or c`` drops ``c``; a two-scalar loop exits with the wrong
       value)**, A8 (slice assignment lowers to a bit-serial loop, then a
       crash); **B1 is present in AMC's frontend too** (A6). Our MLIR also
       needs 4 text edits: R1-R3 are **semantic mismatches** between the
       forks (rank-0 scalars, ``pipeline_ii`` type, ``unroll`` vs
       ``loopschedule.parallel``), and R4 is a ``top`` name clash. One is
       ours: O1, invalid ``trunci`` in plain ``customize`` (B2)
     - ``u1_bf16_add_amc_2026-10-02.rst``; repros in
       ``u1_bf16_add_amc/repros.py``. These are AMC defects, to file with
       AMC (D-2) after triage. Also T1: at a 3.333 ns target AMC's delay
       model misses by 1.24 ns

``vpu_bf16_add_pipe`` (latency 2)
---------------------------------

Same function as ``vpu_bf16_add``, so the same two Allo expressions. The
value cells repeat the pilot's; the latency cells are new. Evidence:
``u1_pipe_2026-10-02.rst`` (values ``check.txt``, csim ``csim_cycles.txt``,
RTL ``rtl_cmp.txt``).

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - Allo simulator, ``native``
     - **finding, semantic mismatch**: 250,955/251,936; the 981 are the RTL's
       NaN (``+0x7FC0``) and ``(+0)+(-0)=-0`` rules. Latency: **untimed**
     - as the pilot
   * - Allo simulator, ``bits``
     - **match** 251,936/251,936 (B1 workaround). Latency: **untimed**
     - ``bf16_add.bits`` reused unchanged
   * - SystemC csim, ``native``
     - **finding, semantic mismatch**: 249,907/251,936 (2,028 ``ac::bfloat16``
       NaN ``0x7FFF``/``0xFFFF``; 1 signed zero). Latency: **finding, semantic
       mismatch** (L3): csim reads 2 for every one-kernel unit -- the
       handshakes' count, equal to 2 here by coincidence
     - ``csim_cycles.py`` stamps the emitted testbench
   * - SystemC csim, ``bits``
     - **match** 251,936/251,936; latency as ``native`` (unchecked)
     - with the merged emitter fixes
   * - Catapult RTL, ``bits``
     - **match** 251,936/251,936 **at latency 2, II=1** (5.0 ns) -- but only
       with ``s.unroll("leading_zeros17:offset")``: rolled, Catapult says
       "II 1" and the RTL takes 2 or 18 cycles per vector, data-dependent
       (1.42 average; **finding L5**), and fails to schedule at 2.0 ns. At
       2.0 ns unrolled: latency **3**, not 2 (**L2**). Pinned by a
       hand-patched I/O cycle constraint: 2 at 5.0 ns; at 2.0 ns Catapult
       **refuses** (SCHD-30) (**L4**)
     - Verilator ``stream`` shape vs MiniTPU ``valid``; full stimulus
   * - Catapult RTL, ``native``
     - **finding, semantic mismatch** (NaN, signed zero, as the pilot);
       latency **1** at 5.0 ns where ``bits`` gives 2 (**L2**)
     - ``bf16_native_ii1_5p0``
   * - RTLGen, AMC
     - **n/a** (not run): the function is the pilot's; neither has a latency
       directive to test (proposal: refuse)
     - pilot rows

``vpu_bf16_mul``
----------------

6,019,104 vectors (``bf16_mul.stimulus()``: corners crossed, ties, random,
every ``a`` against every corner). Evidence for every row:
``u1_mul_2026-10-02.rst``.

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - Allo simulator, ``native``
     - **finding, semantic mismatch**: 5,011,918 match; the 1,007,186 others
       are the RTL's rules: flush-to-zero on input and output (603,316) and
       NaN always ``+0x7FC0`` (403,870). Outside NaN the simulator is IEEE bit
       for bit
     - N2: ``bfloat16`` has no flush-to-zero mode. N1: the simulator keeps
       the NaN operand's sign and gives ``Inf x 0`` the x86 ``-NaN``
   * - Allo simulator, ``bits``
     - **match** 6,019,104/6,019,104, no workaround (the RTL's exponent is
       ``logic signed``, so B1 cannot apply)
     - ``units/bf16_mul.py``; M1 (concat as slice stores) and M3 only
   * - SystemC csim, ``native``
     - **finding, semantic mismatch** x2: 5,011,890 match; same flush classes
       as the simulator; NaN sign is ``a ^ b`` (``ac::bfloat16``), so **the
       simulator and csim disagree on 403,397** of the 1,067,862 NaN/Inf pairs
       of the same Allo program
     - N1 (``u1_mul/repros.py n1``). The NaN a bf16 op returns depends on the
       op and the backend; the simulator is no stand-in for csim on NaNs
   * - SystemC csim, ``bits``
     - **match** 6,019,104/6,019,104 (30 s)
     -
   * - Catapult RTL, ``bits``
     - **match** 6,019,104/6,019,104 in Verilator, II=1, Stream and Wire ports,
       2.0 and 3.33 ns, with no hand-patch to the emitted SystemC. Latency:
       declared 0 is **refused** (I/O constraint ``-equal 0``: SCHD-30, a
       ``Push`` cannot chain after a ``Pop``; N3); unpinned 2, pinned 1.
       Same-flow DC at 3.33 ns, Wire ports against MiniTPU plus an output
       register: **790.8 vs 508.9** um^2 (1.55x; 46 vs 16 flops)
     - ``u1_catapult_units_2026-10-02/README.md``. The latency is
       **finding, semantic mismatch** (N3: no zero-latency unit on
       Connections ports; triage item 4)
   * - Catapult RTL, ``native``
     - **finding, semantic mismatch**: 5,011,890, the same vectors as csim
       (flush rules, NaN). Equal to IEEE on all 6,019,104; NaN is ``0x7fc0``
       with sign ``a ^ b``, where ``+`` gives ``0x7fff`` (N5)
     - Same record
   * - RTLGen, AMC
     - not tried
     - out of this session's scope

``vpu_bf16_mul_pipe``
---------------------

Same function as ``vpu_bf16_mul``, two register banks, latency 2 (RTL
measured 2). The question was whether a pipelined unit differs from the comb
one in Allo at all: on the simulator and SystemC side **it does not** (L1-L3).

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - Allo simulator / SystemC, ``native``, ``bits``
     - as ``vpu_bf16_mul`` (the same Allo programs) + **finding, missing
       abstraction** L1: latency is not expressible. The simulator is
       untimed; ``s.pipeline`` states II, not depth; no primitive says
       "register here" or "latency 2"
     - Recorded deviation for U1, as triage item 4 (latency 0) -- here
       latency 2
   * - Allo simulator, ``bits_pipe`` (the pipe's own stage-2 text)
     - **match** 6,019,104, **after the B1 workaround** at a new site:
       ``exp_sum_s2 <= exp_bias_s2`` on ``logic [8:0]``; without the spare bit
       1,229,630 vectors are wrong. The comb module's text needs none: the two
       RTL forms of one unit differ in whether Allo computes them right
     - ``units/bf16_mul_pipe.py``
   * - SystemC csim, ``bits_pipe``
     - **match** 6,019,104
     -
   * - Allo simulator / SystemC, ``stages`` (two kernels, one ``Stream`` per
       ``*_s1_q`` register)
     - **match** 6,019,104 on both (317 s / 137 s) + **finding, semantic
       mismatch** L2: in csim a depth-1 Stream is not a register -- latency
       4 and **II 2** (depth 2: II 1); comb, pipe and ``s.pipeline``'d comb
       all take 2 cycles in csim. **Missing abstraction** L3: no link type is
       a pipeline register
     - ``u1_mul/latency_probe.py``. A declared latency can be checked only on
       RTL a tool wrote from Allo (Catapult track)
   * - Catapult RTL, ``bits_pipe``
     - **match** 6,019,104/6,019,104 **at latency 2, II=1**, 2.0 and 3.33 ns:
       Stream ports with the hand-added I/O constraint ``-equal 2`` (unpinned:
       3 at 2.0 ns, 1 at 3.33 ns, L2); backpressure loses nothing. Wire ports:
       2 with the loop constraint ``-equal 2``, but ``-equal 3``/``4`` still
       give 2 (**finding, missing abstraction** N2: a Wire unit's latency
       cannot be pinned). DC at 3.33 ns, Wire against ``vpu_bf16_mul_pipe``:
       **681.0 vs 711.8** um^2 (0.96x)
     - ``u1_catapult_units_2026-10-02/README.md``. ``native``: 5,011,890, as
       ``vpu_bf16_mul``

``mxu_bf16_mul_acc24``
----------------------

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - Allo simulator, ``native``
     - **finding, workaround + semantic mismatch**: no acc24 type, so a
       ``float32`` product + bitcast + RNE to 15 bits in a ``uint32`` (W1,
       missing abstraction). 4,668,187 match; the 1,350,917 others are flush
       (674,119) and NaN (676,798; float32 also keeps the payload). W1
       double-rounds float32 subnormals: 31 vectors off IEEE acc24 by an ulp,
       invisible here only because the RTL flushes them
     - ``units/mul_acc24.py``; ``u1_mul_2026-10-02.rst`` (W1, N1, N2)
   * - Allo simulator, ``bits``
     - **match** 6,019,104/6,019,104, no workaround (compares are against
       literals)
     - The ``UInt(24)`` port works in the simulator; committed with ``uint32``
       for S5
   * - SystemC csim, ``native``
     - **finding, bug** S4 (emitter): ``bf16 -> f32`` is emitted as
       copy-initialisation of ``ac_ieee_float<binary32>`` from
       ``ac::bfloat16``, whose constructor is ``explicit``; g++ refuses.
       With the widening done on bit patterns (``native_bitext``,
       **workaround**): 4,941,087 match, same flush classes, NaN 403,898
     - Fix proposed: emit ``T v = T(x);`` for ``ExtFOp``/``TruncFOp``
       between ac floats (``EmitVivadoHLS.cpp:2613`` ``emitCast``)
   * - SystemC csim, ``bits``
     - **match** 6,019,104/6,019,104 (34 s) with a ``uint32`` port; with
       ``UInt(24)``, **finding, bug** S5: csim runs, then reading the output
       raises ``KeyError: 'ui24'`` (``np_supported_types``)
     - ``u1_mul/repros.py s5``
   * - Catapult RTL, ``bits``
     - **match** 6,019,104/6,019,104, II=1, latency 1 (Stream and Wire, 2.0
       and 3.33 ns); declared 0 is not reachable (N3). DC at 3.33 ns, Wire
       against MiniTPU plus an output register: **562.6 vs 517.1** um^2
       (1.09x)
     - ``u1_catapult_units_2026-10-02/README.md``
   * - Catapult RTL, ``native`` (as written)
     - **finding, semantic mismatch**: 4,941,087, the same as csim's
       ``native_bitext``. S4 is fixed, so the unit synthesizes as written. The
       31 float32-subnormal double roundings (W1) are in the RTL (6,019,073 vs
       IEEE acc24), hidden by MiniTPU's flush
     - Same record
   * - RTLGen, AMC
     - not tried
     - out of this session's scope

``mxu_acc24_add_pipe`` (latency 3)
----------------------------------

New variants in ``units/acc24_add_pipe.py``: ``native`` (float32 + bitcast
round, a workaround), ``bits`` (the ``.sv``, three stages in one kernel),
``staged`` (three kernels, one ``Stream`` per stage register bank).
Evidence: ``u1_pipe_2026-10-02.rst``.

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - Allo simulator, ``native``
     - **finding, workaround + semantic mismatch**: 350,021/352,116. No acc24
       type (V2); 113 double roundings (one acc24 ulp, README divergence 5);
       1,981 NaN encodings; 1 signed zero. Latency **untimed**
     - ``check.txt``
   * - Allo simulator, ``bits`` / ``staged``
     - **match** 352,116/352,116 both, first time, after B1 spare bits at 5
       compares (V3). Latency **untimed**; ``staged`` states three stages and
       the simulator cannot see them (L1)
     - 
   * - SystemC csim, ``native``
     - **finding, semantic mismatch** (V1): 349,935/352,116 -- **every NaN
       result (2,067) becomes a zero** (``-0``/``+0``): ``ac_ieee_float<binary32>`` NaN
       is ``0x7FFFFFFF`` and the round step carries it into the sign; plus the
       113 double roundings and 1 signed zero
     - NaN encoding probed in C++
   * - SystemC csim, ``bits`` / ``staged``
     - **match** 352,116/352,116 both. Latency **finding** (L3): csim reads 2
       (``bits``, with or without ``s.pipeline``) and 6 (``staged``) for a unit
       of latency 3
     - ``csim_cycles.txt``
   * - Catapult RTL, ``bits``
     - **match** 352,116/352,116 bit-exact on every build. Latency: unrolled
       LZC + ``s.pipeline``, **2** at 5.0 ns and 3 at 2.0 ns (**L2**, the
       clock decides); rolled (as written), 19 cyc/vector under a
       "pipelined II=1" message (**L5**). **Pinned to 3 at II=1** by the
       hand-patched ``cycle set {v12.Push()} -from {v10.Pop()} -equal 3``,
       at 5.0 and 2.0 ns; -equal 1 and 6 also measured exactly; an
       infeasible loop constraint is refused (SCHD-3) (**L4**). ``cycle set
       <loop> -equal 3`` is not latency (measured 2)
     - ``emit_csyn.py --io``, ``cmp_rtl.py``
   * - Catapult RTL, ``bits`` at 2.0 / 3.33 ns, and DC (Catapult track)
     - **match** 352,116/352,116 **at latency 3, II=1** at both clocks:
       Stream ports with I/O constraint ``-equal 3``, and Wire ports with the
       loop constraint ``-equal 3`` (unpinned 2: it works here by coincidence,
       N2). ``rtl.rpt`` slack is -0.33 to -1.96 ns, but DC closes the same
       RTL (**finding** N4). DC at 3.33 ns, Wire against
       ``mxu_acc24_add_pipe``: **2448.0 vs 1424.4** um^2 (**1.72x**, comb
       1.9x; hypothesis: the M5/B1 workarounds in the text, N7). ``native``:
       349,935, csim's vectors, V1 (NaN -> 0) now in RTL
     - ``u1_catapult_units_2026-10-02/README.md``
   * - Catapult RTL, ``staged``
     - **finding, missing abstraction** (L1): latency **5** at 5.0 ns, II=1
       (unrolled), not 3: stage kernels are not register stages
     - ``acc24_staged_u_ii1_5p0``
   * - RTLGen, AMC
     - **not run** (time bound). Expected to take ``bits``; no latency
       directive in either
     - proposal: refuse a declared latency

``vpu_alu`` (composition probe)
-------------------------------

The ALU is built out of Allo engine units, as ``vpu_alu.sv`` is built out of
its adder and multiplier: the pilot's ``bits`` adder is *reused*, hoisted out
of its kernel into ``bf16_add.add_bits``. 5,079,552 vectors (all 16 op codes).
Composition findings C1-C11 are in ``u1_alu_2026-10-02.rst``, with repros in
``u1_alu/repros.py``. Two of them are **silent miscompiles in the front end**
that any cross-module reuse is exposed to (C3, C4). *[provisional: fix them
before more units are composed; proposals 2-3 in the record]*

.. list-table::
   :header-rows: 1
   :widths: 18 40 42

   * - Tool
     - Cell
     - Evidence / note
   * - Allo simulator, ``bits`` / ``bits_dispatch`` / ``netlist``
     - **match** 5,079,552/5,079,552 each, **after workarounds**: B1 spare
       bits (``mul_bits``, ``bf16_gt``), B2 (C6), engine selection by def-name
       factories (C11, forced by **bugs** C1, C5 and the **missing
       abstractions** C2, C9)
     - Function reuse across modules works; an engine passed as a value does
       not link (C1). ``netlist`` (engines as ``@df.unit`` on streams) swaps
       engines by value, but a unit cannot be sized at instantiation and a
       stale size deadlocks (C9). **C3, C4: bugs, silent** (a reused
       function reads the caller's same-named globals; a module-level numpy
       array shadows a kernel parameter).
   * - Allo simulator, ``native``
     - **finding, semantic mismatch**: 5,051,836 match. The rest: NaN
       encodings 3,810, MUL flush 6,065, ``(+0)±0`` 2, MAX/MIN with NaN 4,859
       (``maximumf``: NaN wins), and **C7**: 12,980 MOV/default-arm NaNs lose
       their payload through a ``phi bfloat`` (x86 ``__truncsfbf2``)
     - Engine swaps isolate each cause: ``engines_native`` 5,069,695,
       ``add_bits_mul_native`` 5,072,213 (MUL only), ``netlist_native``
       5,069,696.
   * - SystemC csim, ``bits`` / ``bits_dispatch`` / ``netlist``
     - **match** 5,079,552/5,079,552 each
     - 24.9-88.3 s. The emitter fixes merged in ``u1-pilot`` are enough;
       the engine hierarchy survives as C++ functions.
   * - SystemC csim, ``native``
     - **finding, bug + semantic mismatch**: 5,064,720 match. NaN 6,249
       (``ac::bfloat16`` ``0x7fff``), MUL flush 6,065, zeros 2, and **C8**:
       Allo ``max``/``min`` (``arith.maximumf``) are emitted as ``std::max``,
       which returns ``a`` on NaN or ``±0``, so 2,516 MAX/MIN vectors differ
       from the simulator's own answer
     - ``engines_native``/``netlist_native`` 5,067,236;
       ``add_bits_mul_native`` 5,072,282. C7 does not occur here.
   * - Catapult csyn, ``bits`` vs ``bits_dispatch``
     - **finding, missing abstraction** (C10): II=1 needs
       ``s.unroll("leading_zeros17:offset")``, the reused adder's internal
       loop, named by the ALU's build (``SCHD-30`` without it). Then both
       pipeline at II=1: area score **4138.1** (compute all, mux) vs
       **5211.9** (dispatch, +26 %: one adder per call site, no sharing
       across exclusive branches). rtl.rpt slack -1.34/-1.28 ns at 2.0 ns
     - ``u1_alu/catapult/``. csyn only; not simulated as RTL, no DC. The
       RTL's own form is the cheaper one to write, and Allo keeps whichever
       is written.
   * - Catapult RTL, ``bits`` (+ unroll), Verilator and DC
     - **match** 5,079,552/5,079,552 **at latency 3, II=1** on Stream ports
       with I/O constraint ``-equal 3``, at 2.0 and 3.33 ns (unpinned: 3 and
       2); backpressure loses nothing. Wire ports: 3 at 2.0 ns, but **2 at
       3.33 ns whatever the loop constraint** (3, 4, 5; **finding, missing
       abstraction** N2). DC at 3.33 ns, Wire (latency 2) against
       ``vpu_alu``: **2826.3 vs 2394.0** um^2 (1.18x). At 2.0 ns, latency 3:
       3947.2 vs 2612.9 (1.51x)
     - ``u1_catapult_units_2026-10-02/README.md``. ``rtl.rpt`` slack about
       -1.4 ns, but DC closes it (N4)
   * - Catapult RTL, ``native``
     - **finding, bug + semantic mismatch**: 5,064,720, csim's exact classes,
       including **C8 in RTL** (``std::max``/``min``: 2,516 MAX/MIN vectors)
     - Same record
   * - RTLGen, AMC
     - **not tried**
     - Out of this session's scope.

Environment findings met on the way
-----------------------------------

* ``LLVM_BUILD_DIR``: CLAUDE.md told sessions on zhang-21 to export a ``/home``
  LLVM build that no longer exists there; the simulator then fails with
  ``Unknown function <top>``. Fixed in ``4904eadc``.
* The emitted SystemC testbench and Catapult's g++ 10.3: the ``allo`` env's
  activate script puts ``gcc-toolset-13`` first and Catapult's module puts its
  own ``python`` first; ``harness/env-zhang21.sh`` fixes the order.
* **Unconfirmed, seen once:** a SystemC kernel that reads back its own
  output-port element (``c[i] = c[i] * 256``) failed g++ with ``'v2' was not
  declared`` (``systemc-u1-fixes``, while writing the S5 test). Not yet
  reproduced on purpose.
