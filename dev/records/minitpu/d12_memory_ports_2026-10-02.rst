D-12 study: ``compose`` declares memory ports
=============================================

2026-10-02, zhang-21. Read-only design study for the owner's review **before any
code** (D-9: a change to the programming model is a D-n entry first). Nothing
was built or run. Sources: README on ``origin/u1-pilot`` (Direction, D-1, D-7,
D-9..D-11, the U2-U5 rows); ``allo/compose.py``, ``allo/memory.py``,
``allo/actions.py``, the fork docs; ``dev/records/minitpu/{u2_plan,
u2_phase0, u2_regfile, latency_report}_2026-10-02.rst`` and
``dev/records/open_hls/amc_exploration_2026-10-02.rst`` (``u1-pilot``); AMC at
``scratch/amc/amc-dialect`` (``fe60c121``); MiniTPU ``b3ba0a4d`` read-only;
Catapult 2024.2's library directory on this host.

**[V]** verified by reading source, a record, or a measured run cited from a
record. **[I]** inferred, still to be tested. Line numbers are against
``main`` at ``db184ebc`` unless a record is named.

1. The problem, with the evidence
---------------------------------

**Nothing in Allo can say "one memory, N ports, used by N state machines", and
nothing can state a port's timing.** [V]

* ``compose.Memory`` is a boundary ``m_axi`` array with two fields, ``name`` and
  ``dtype`` (``compose.py:83-89``). ``Architecture._check`` refuses one memory
  addressed by two units: ``"memory {mem!r} is addressed by both ..."``
  (``:289-296``). A unit's on-chip arrays (TinyTPU's ``ar``/``vr``/``spad``)
  are kernel-local, private by construction, "one read and one write per
  iteration" by comment, never by rule. ``Unit.arrays()`` sees
  ``read``/``write`` flags, not a count (``:160-204``); ``actions.structure``
  derives "one read and one write port" per array and the page says that is
  wrong for a partitioned array (``actions.rst:248-253``);
  ``isa_spec.json``'s ``memories`` carry ``read_ports: 1, write_ports: 1`` as
  data no rule reads.
* ``allo.memory.Memory(resource, storage_type, latency, depth)`` is accepted and
  ``latency``/``depth`` are **dropped by every backend**: ``get_memory_space()``
  encodes ``resource*16 + storage`` only (``memory.py:223-240``); the regfile
  ``annotated`` variant emitted byte-identical SystemC with and without it,
  and Vitis emitted ``bind_storage ... type=ram_1wnr impl=lutram`` with no
  latency (``u2_regfile`` H3, confirmed). A D-1 violation by construction.
* **G1** (``u2_regfile``): "3 reader kernels + 1 writer kernel on one regfile"
  is refused by SystemC (region ``Stateful`` "is used by 4 kernels",
  ``EmitSystemC.cpp:3346-3380``; a region argument with a pure reader and a
  pure writer, ``:2896-2933``), refused by Vitis (``EmitVivadoHLS.cpp:3454``),
  refused by ``compose``, and run **unordered** by the simulator (``shared``
  8,502/180,780, no diagnostic; ``shared_sync`` with explicit token order
  180,780/180,780). D-11 now refuses a shared ``Stateful`` everywhere and
  says "a memory with several ports is a different thing".
* The open backends **infer** ports and neither lets the user state them:
  RTLGen built the one-kernel regfile as **two registered-read 1R1W copies
  with every write sent to both** (``rf_trace_256.sv`` ``mem_c0``/``mem_c1``,
  II=3); AMC declared ``amcMemory0`` with five logical ports
  ``(w, r, r, r, w)``, each ``dyn ... (1)``, and implemented **two physical**
  (``p0`` rw, ``p1`` r) time-multiplexed at II=3 (``amc/*.alloc.txt``,
  ``amcMemory0_impl.sv``), neither refusing nor replicating.
* **G2**: no generated RTL reads in the request cycle (Catapult read 1 /
  write-to-read 2 measured; RTLGen and AMC registered reads). The owner's
  checkpoint-3 call: pursue a combinational read (branch ``u2-comb-read``),
  not a recorded deviation. So a port must be able to say ``latency=0``.
* **What MiniTPU has, which U2-U5 must express** (``u2_phase0``, measured):
  ``vpu_regfile`` 3 asynchronous reads (0) + 1 synchronous write (visible 1),
  never reset, read-old on a same-cycle write; ``vpu_word_array`` 2 symmetric
  read/write ports, registered reads at 3 (compute) and 2 (DMA), write
  visible 1 from either port, a same-word cross-port access with a write
  **undefined** (asserted in simulation only, ``vpu_vmem_simd.sv:121``, no
  assembler rule). The regfile's one write port serves six writers (ALU,
  SFU, reduce, ``vld``, ``vmatpop``, ``vtxout``) through an **OR mux with no
  arbiter** (``vpu.sv:290-304``); MiniTPU keeps them apart only by a
  **calendar in the assembler**: ``_Timeline.write_port: set[int]`` of booked
  cycles, and ``first_free_write`` slides a bundle until none of its write
  offsets (``W`` = 5/7/15/11/6/3/3 per class) is taken (``asm.py:1342-1378``);
  the same calendar exists again in ``sequencer.sv:276-342`` under ``ifndef
  SYNTHESIS``. Port C is a fixed-priority mux the matrix engine wins
  silently (``vpu.sv:429-431``): refused in one bundle, delayed across.
* The fork already names the right form: "a predicate over resources is the
  wrong form ... the sound form is a **calendar**" (``paper.rst:178-182``);
  Actions refuses "two units writing one port in one cycle" with "a cycle
  offset, not a port" as the repair (``:214-218``); README Threads lists "a
  calendar-shaped port rule" as next.
* Vitis is not the wall it was said to be: in 2023.2 two processes on one
  array with ``#pragma HLS stream type=unsync`` are accepted, one BRAM port
  each (``HLS 200-824``/``200-755``); three are refused (``200-780``, "only 2
  ports") (``register_2026-10-01.rst`` probe table). Allo never emits it.
* ``ip_gaps.rst`` rows 3, 4 and 8 all touch memory: per-slice local memory
  (cannot express), explicit placement ("an owning slice on ``Memory``, and a
  reachability rule in ``Architecture._check``"), and "a memory with two
  writers is not refused" (the cheapest unfilled row).

2. The proposal
---------------

**Declaration.** ``compose.Memory`` grows ``rows`` and ``ports``; a ``Memory``
without ``rows`` stays today's boundary ``m_axi`` array, so TinyTPU composes
unchanged. ::

    vreg = Memory("vreg", dtype="UInt(W)", rows="NVREG",
                  ports=(Port("ra", "r", latency=0),
                         Port("rb", "r", latency=0),
                         Port("rc", "r", latency=0),
                         Port("w",  "w", visible=1)),
                  collision="old")   # read of a word written this cycle
    vmem = Memory("vmem", "UInt(VW)", rows="VMEM_WORDS",
                  ports=(Port("compute", "rw", latency=3, visible=1),
                         Port("dma",     "rw", latency=2, visible=1)),
                  collision="undefined")

``Port(name, dir, latency, visible, count=1)``, modelled on AMC's
``!amc.port<shape, T, proto, rw, readLatency, writeLatency> x count``
(``AmcTypes.td:131-168`` [V]): ``dir`` in ``r``/``w``/``rw``; ``latency`` is
read edges from address to data, **0 = asynchronous**, legal at AMC's type
level too (``dyn-error.mlir:8`` [V]); ``visible`` is edges from a write to
the first read on *any* port that sees it (MiniTPU: 1 on every unit);
``count`` is a group of interchangeable ports (AMC's ``count``), so a 2-port
symmetric VMEM is ``Port("p", "rw", 3, count=2)`` when the two are not
distinguished. Width is the memory's ``dtype``, depth is ``rows``; a
per-port width (lane split) is out of scope, as in AMC (``split_port`` is a
separate op). ``collision`` names the one semantic AMC leaves hardcoded
(read-during-write on another port returns old data, in its ``seq.hlmem``
lowering and its legacy RAMs alike [V]): ``old`` (regfile, VMEM simulation
model), ``undefined`` (VMEM on URAM), ``refuse``.
``allo.memory.Memory``'s ``latency``/``depth`` are **refused** from now (D-1;
the owner's bracketed default in ``u2_regfile`` item 6) and return only as
the kernel-local shorthand ``x: T[N] @ Memory(ports=(Port("p", "rw",
latency=L),))``: a one-port memory owned by the declaring kernel. Every
unset field constrains nothing (Encoding invariant 2,
``extending_allo.rst:164-181``).

**Binding.** A unit binds ports, not memories, positionally onto its body
parameters: ``memories=("vreg.ra", "vreg.w")`` (a bare memory name still
means its single ``m_axi`` port). The body parameter's uses are checked
against the port's direction by the AST walk ``Unit.arrays()`` already does.
The regfile's ``trace`` expression (one kernel owning all four ports) is the
degenerate case, one unit binding every port, which is exactly today's
kernel-local array; the two spellings must emit the same region.

**Legality rules, in order of cost** (``extending_allo.rst:37-50``):

1. *one-owner-per-port* (replaces one-owner-per-memory; plan Q8): a port name
   appears in exactly one unit's ``memories``; a port nobody binds is
   refused as dangling, like a channel nobody reads.
2. *port-direction*: a store through an ``r`` port or a load through a ``w``
   port is refused at ``Architecture._check``.
3. *accesses-per-iteration*: the subscript sites through one port in the
   body's steady-state loop number at most ``count``, each executed exactly
   once per iteration (the S6 shape, a conditional access, is refused as
   ``e238e410`` refuses a written-and-read stream). This is the structural
   half of a calendar: one access per port per iteration and one owner per
   port means two *units* cannot touch one port in one cycle.
4. *write-ports*: a memory with more than one ``w``/``rw`` port and
   ``collision="refuse"`` is refused; with ``old`` or ``undefined`` the
   cross-port same-word case is an **obligation** the composition states
   (``Obligation(where, premise)`` as ``s.dependence`` does), because no
   static rule decides it (the assembler cannot see DMA beat timing either,
   conflict row 9). This closes ``ip_gaps`` row 8 for free.

**The calendar MiniTPU enforces only in software** is not a ``compose`` rule;
it is the Actions layer's, and the ports give it its data. An Action already
spends "a named port of that unit" and has ``span`` and ``initiation``
(``actions.rst:27-30, 163-178``). With ``latency`` and ``visible`` declared,
``Machine`` can compute, per instruction, the cycle offset at which each
action occupies each port (MiniTPU's ``W`` per class is exactly "the unit's
own latency plus ``VPU_WB_STAGES``", ``ISA_AND_INTERFACES.md:315``), and the
existing refusal "two actions on one port in one cycle" becomes decidable
with real offsets instead of a count. The OR-mux write port is then
expressed honestly: a *writeback* unit owns ``vreg.w`` and the six producers
reach it over channels; the calendar rule sits on that unit's actions, and
``assemble()``/ACT consume the offsets the assembler's ``_Timeline``
hard-codes today. Program-level checking (across bundles) stays in the
assembler/ACT, instruction-level in Actions, structural in ``compose``:
plan 2c's three tiers, now from one declaration.

**Emission and reporting (D-10's pattern).** ``compose`` lowers a ported
memory to one region-scope array carrying a ``ports`` attribute (as a stream
carries ``stream:`` in its memory space); every backend reads it and does one
of three things: honour, refuse quoting the port and the cause, or lower via
a stated replica. Every RTL backend writes ``memory.json`` beside
``latency.json``: per memory and port the implementation chosen (register,
1R1W, dualport, replica x N), the achieved read latency and write
visibility, and a ``status``. The harness's step probes already measure both
on RTL (``u2_phase0`` latency table), so declared-vs-achieved is checked the
way D-10 checks a unit's latency.

3. Per backend
--------------

.. list-table::
   :header-rows: 1
   :widths: 11 30 59

   * - backend
     - verdict
     - evidence; what is inferred
   * - simulator
     - **honour the structure, record "untimed"**
     - A region-scope buffer shared by kernels already runs (``shared_sync``
       180,780/180,780 [V]); without an order it is racy (M2 [V]). The
       simulator cannot model ``latency``/``visible`` (iteration = cycle is
       an order only). So: build the buffer, print one ``[memory] untimed``
       line per ported memory, and let the unit's token streams or the
       sequencer order it, as ``shared_sync`` does. Reads before writes stay
       undefined, as on the RTL. [I: read-old across kernels needs the order
       to be explicit.]
   * - SystemC csim
     - **lower via write-broadcast replica; latency 0 pending; refuse W > 1**
     - Today: refuses sharing in both forms [V]; ``AlloMemPins`` is 1R1W,
       **synchronous read** (``q`` on the next edge, ``EmitSystemC.cpp:
       3797-3798`` [V]); per-client replicas "summed at readout, assumes
       disjoint writes" (``EmitSystemC.md:238-243`` [V]). Proposed: N read
       ports and one write port become N 1R1W copies whose write pins are
       all driven by the one writer, so every copy holds the same data and
       any copy's read-out is exact; RTLGen's own choice [V] and MiniTPU's
       LUTRAM arithmetic (3 arrays per 3R1W, ``REGISTER_FILE.md:134``) [V].
       Two write ports need N_R x N_W copies and a live-value table
       (``REGISTER_FILE.md:118-124``): refuse for now. ``latency=0`` needs
       the combinational read path (``u2-comb-read``); ``latency>1`` needs
       ``L-1`` hold registers after ``q`` [I]. The register estimates 50-100
       lines for one shared instance (``:1925-1928``).
   * - Catapult
     - **honour via register map or dualport; report; refuse 3+ ports on RAM**
     - ``s.partition(Complete)`` becomes ``hls_resource map_to_module=
       "[Register]"`` (``EmitCatapultHLS.cpp:488-491``), Catapult obeys
       (CIN-341): 3R1W at II=1, 180,780/180,780 per cycle, read 1 /
       write-to-read 2 measured, +38 % area over MiniTPU + output registers
       [V]. Its library on this host is ``ccs_ram_sync_{1R1W, dualport,
       singleport, singleport_wmask}``: **all synchronous, no asynchronous
       RAM** (``$MGC_HOME/pkgs/siflibs`` [V]). So ``latency=0`` = registers
       (Complete); a 2-port ``rw`` memory = ``ccs_ram_sync_dualport`` [I:
       directive and II=1 untested]; ``latency=3`` on a RAM read = pipelined
       read stages [I: the ``-STAGE_REPLICATION``/read-latency directives
       named in ``u2_regfile`` item 6 are unverified]; 3+ ports on a RAM is
       SCHD-30, which Catapult itself refuses [V]. Achieved latency is read
       from ``cycle_set.tcl`` as D-10 does [V].
   * - Vitis
     - **honour 1-2 ports; refuse 3+; measurement only (D-1)**
     - ``bind_storage`` honours ``resource``/``storage_type`` and drops
       latency [V]. Two kernels on one array with ``stream type=unsync`` get
       one BRAM port each; three are refused ``200-780`` [V, 2023.2]. Allo
       never emits the pragma [V]. A 2-port ``Memory`` lowers to
       ``type=unsync`` + ``RAM_T2P`` [I: untested together]; ``latency=0``
       is unreachable on BRAM (refuse), reachable on LUTRAM with
       ``RAM_1WNR`` only [I]; latency is reported from ``csynth.xml`` (D-10)
       and never constrained.
   * - RTLGen
     - **lower-via-replica (its own); refuse latency 0; adapter at M2**
     - Infers ports from accesses: a write-broadcast two-copy replica at
       II=3, registers at II=1 with ``partition(Complete)`` [V]; registered
       reads only (``mem_c0_rd0_reg``) [V]; no user-facing port declaration
       (array arguments become ``A_rd0_*`` external ports) [V]. The adapter
       maps ``Port`` counts onto ``bind_storage``/``assign-banks`` [I] and
       reads ``manifest.json`` for the achieved numbers [V, D-10].
   * - AMC
     - **honour through MLIR; refuse ``w`` latency != 1; adapter at M2**
     - The port type carries direction, read/write latency and ``count``
       [V]; ports are declared in ``amc.memory``'s signature and handed out
       by ``amc.instance`` (``test/simple/conv.mlir:4``: 4 ``static r(1)`` +
       1 ``rw``) [V]; ``r(0)`` lowers to a combinational ``seq.read``
       (``AmcToHW.cpp:290-362`` [V by reading, not run]). The ``ram_*.sv``
       library the plan cited (all registered) is **legacy**: each
       ``amc.memory`` becomes a ``seq.hlmem`` with one physical port per
       ``create_port``, any number [V]. The cap is a policy: allocation is
       "one port per access", grouped and capped at ``bank-ports=2``, the
       rest time-multiplexed through ``dyn`` ports and arbiters, never
       refused, never replicated [V: the regfile's 5-on-2 at II=3]; more
       static ports than the policy allows only warns. Gaps against this
       proposal [V]: no owner (ports are positional), no
       ``visible``/collision attribute, write latency must be 1
       (``LowerSeqHLMem.cpp:90-93`` [I: not run]), no port API in the Python
       frontend, and the vendored Allo lacks ``@df.region`` entirely
       (``amc_exploration``), so the hand-off is textual MLIR: an
       ``amc.memory`` with the declared ports and ``bank-ports`` set to the
       declared count.

4. What it serves
-----------------

* **Parameters.** ``rows``, ``dtype`` and every ``latency`` are expressions
  over the architecture's parameters, like ``Channel.depth``; ``vpu_regfile``
  w16 and w256 are one ``Memory`` at two ``W``, and VMEM's 3/2 are numbers
  ACT reads from ``memory.json``, never constants (D-10).
* **Optional modules.** An accumulator file is a ``Memory`` plus the ports
  its clients bind; leave it out and rule 1 (dangling port) or the unit's
  own ``legality`` says which clients must go with it. TinyTPU as an
  instance (U3 track) adds ``ar`` this way.
* **Swappable engines.** The systolic array and the adder tree bind the
  **same ports** of the same memories; the port contract (direction,
  latency, accesses per iteration) is the interface an engine must honour,
  and ``_check`` refuses an engine that needs a port the memory lacks.
* **NUMA slice memory** (``ip_gaps`` row 3): a ``Memory`` per slice instance
  needs ``@df.unit`` instantiation-site parameters first (the row's own
  "needed"); ports are the per-slice interface, so a slice's memory is
  reachable only through ports bound inside the slice.
* **Explicit placement** (row 4) asks for "an owning slice on ``Memory`` and a
  reachability rule in ``_check``". Port ownership is that edge: a unit may
  bind a port only of a memory placed where the unit is, or across a
  declared channel. The port declaration is the hook; placement is the
  later field. Row 8 closes with rule 4; "two networks" is untouched.

5. Draft README text
--------------------

::

  **D-12 (proposed, 2026-10-02). `compose` declares memory ports; a port
  has one owner.**
  - `compose.Memory(rows=, ports=(Port(name, dir, latency, visible,
    count),), collision=)` declares an on-chip memory with N ports, modelled
    on AMC's port type: `dir` is `r`/`w`/`rw`, `latency` is read edges with
    `0` asynchronous, `visible` is edges until a write is seen on any port.
    A `Memory` without `rows` stays the boundary `m_axi` array it is today.
  - A unit binds ports (`memories=("vreg.ra", "vreg.w")`), not memories.
    Each port has exactly one owner, which replaces one-owner-per-memory.
    The body's uses are held to the port's direction and to `count`
    accesses per iteration, at composition. Two write ports are an
    obligation unless `collision="refuse"`. `Stateful` sharing stays refused
    (D-11): a shared memory is a ported one.
  - The calendar (which cycle each action occupies each port) is the Actions
    layer's rule, computed from `latency`/`visible`; the assembler and ACT
    consume it. `compose` checks structure only.
  - Every backend honours a port, refuses it quoting the port and the cause,
    or lowers it through a stated replica (SystemC: write-broadcast, N read
    copies, one writer). None drops it. RTL backends write `memory.json`
    beside `latency.json`; the harness checks declared against measured.
  - `allo.memory.Memory(latency=, depth=)` is refused until it becomes the
    one-port shorthand of this declaration.
  - Evidence: `u2_regfile_2026-10-02.rst` (G1, H3), `u2_phase0_2026-10-02.rst`.
  - *Reverses if* SystemC's write-broadcast replica cannot match the regfile
    trace per cycle, or Catapult cannot reach II=1 on a 2-port VMEM.

6. What to prototype first, and the cost
----------------------------------------

**First: ``vpu_regfile`` as a ported memory, in ``shared_sync``'s shape.** One
``Memory`` (3 ``r`` at 0, 1 ``w``), three reader units and one writer unit,
driven by the joined Phase 0 trace and compared on the 180,780 defined slots
as ``u2_regfile`` did. Verdicts: simulator match (with the token order),
SystemC csim match through the replica, Catapult per-cycle match at the
register map, ``memory.json`` equal to the step-probe table. Then
``vpu_word_array`` (two ``rw`` at 3/2) is the first real two-port case and
the first ``collision="undefined"`` obligation.

Estimated cost (sizes from the files cited, not measured):

* ``allo/compose.py``: ``Port``, ``Memory.rows/ports/collision``, port-name
  binding, rules 1-4, the region emission of a ported memory: ~150-200
  lines plus a refused/accepted test pair per rule
  (``extending_allo.rst:224-232``). Half a day.
* Front end / IR: a ``ports`` attribute on the region-scope array, the
  one-port shorthand, the ``latency``/``depth`` refusal: ~50 lines. Hours.
* ``EmitSystemC.cpp``: the write-broadcast ``AlloMem`` (N read pin sets, one
  fanned-out write pin set) and the refusal messages: ~200-300 lines on top
  of the register's 50-100 for one shared instance. One to two days, and
  ``latency=0`` depends on ``u2-comb-read``.
* ``allo/backend/catapult.py`` + ``vitis.py``: ``memory.json`` from
  ``cycle.rpt``/``csynth.xml`` and the per-port directives: ~100-150 lines
  each, modelled on ``latency_report``'s manifests. Half a day each;
  Catapult builds the regfile in ~40 s.
* Harness: ``check.py`` reads ``memory.json`` and gives ``PORT-MATCH`` from the
  existing step probes: ~50 lines.
* RTLGen and AMC adapters: at M2; the AMC hand-off is an ``amc.memory``
  signature, a text template.

Nothing above touches ``allo/actions.py`` yet: the calendar rule is phase two,
once two units bind ports of one memory inside a composed instruction.

7. Open questions for the owner
-------------------------------

* **Q1. Port vs. memory ownership.** Approve replacing one-owner-per-memory
  with one-owner-per-port (plan Q8)? It is the programming-model change.
* **Q2. The OR-mux write port.** Express MiniTPU's six writers as one
  *writeback unit* owning ``vreg.w``, fed over channels (true to ``vpu.sv``),
  or allow several owners on one port with the calendar as the only guard
  (true to the assembler)? Only the first is refusable at composition.
* **Q3. Latency 0 on every backend.** Keep ``latency=0`` declarable while
  only Catapult (registers) and AMC (``r(0)``) can produce it, with SystemC
  csim refusing until ``u2-comb-read`` lands, or wait for a measurement?
* **Q4. Where the collision obligation is discharged.** VMEM's compute/DMA
  same-word rule has no assembler rule and only a simulation assertion. Is
  "an obligation on the composition, checked by a stress cosim" enough, or
  should MiniTPU's assembler grow the rule (a change to MiniTPU, D-7)?
* **Q5. ``count`` vs. named ports.** VMEM's two ports differ (3 vs 2), so
  they are named; AMC's ``count`` is for interchangeable ones. Keep both?
  And retire ``latency``/``depth`` from ``allo.memory.Memory`` outright
  (``depth`` is documented "for streams/FIFOs", which ``Stream`` carries)?
* **Q6. Vitis's role.** D-1 makes Vitis measurement-only; is emitting
  ``stream type=unsync`` for a 2-port memory in scope, or does Vitis refuse
  every ported memory with more than one owner?
* **Q7. Checkpoint.** Is the four-unit regfile prototype (section 6) the U2
  acceptance for this item, with ``vpu_word_array`` as the second row?
