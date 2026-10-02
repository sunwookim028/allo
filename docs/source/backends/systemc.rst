..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

..  Licensed to the Apache Software Foundation (ASF) under one
    or more contributor license agreements.  See the NOTICE file
    distributed with this work for additional information
    regarding copyright ownership.  The ASF licenses this file
    to you under the Apache License, Version 2.0 (the
    "License"); you may not use this file except in compliance
    with the License.  You may obtain a copy of the License at

..    http://www.apache.org/licenses/LICENSE-2.0

..  Unless required by applicable law or agreed to in writing,
    software distributed under the License is distributed on an
    "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
    KIND, either express or implied.  See the License for the
    specific language governing permissions and limitations
    under the License.

#################################
SystemC emitter (Catapult, ASIC)
#################################

``target="systemc"`` turns an Allo ``@df.region`` into synthesizable **SystemC**: one
``SC_MODULE`` per ``@df.kernel``, MatchLib ``Connections`` for the links between them, and a
self-contained ``sc_main`` testbench. Siemens Catapult then synthesizes that SystemC to
Verilog, and Cadence Xcelium re-runs the emitted testbench against the synthesized RTL for a
bit-exact check. Emission needs nothing but Allo; everything past it needs Catapult
(``MGC_HOME``) and, for cosim, Xcelium.

This backend came from the ``choonsik1/allo`` fork's ``SystemC-emitter`` branch and was merged
with its history, so ``git log`` on ``mlir/lib/Translation/EmitSystemC.cpp`` attributes it
correctly.

The code is in three places. The emitter is ``mlir/lib/Translation/EmitSystemC.cpp``. The
Python entry point, :mod:`allo.backend.systemc`, holds the SystemC-only steps: argument
directions, testbench data files, and the csim compile command. The flow around them is
shared with Catapult (``allo/backend/hls.py``, ``catapult.py``, ``allo/harness/catapult``).

Quick start
-----------

Emission runs anywhere the ``allo`` env runs:

.. code-block:: bash

   # the env sets neither of these
   source $(conda info --base)/etc/profile.d/conda.sh && conda activate allo
   export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build OMP_NUM_THREADS=8

   cd tests/systemc
   python pc_channel.py systemc      # print the generated SystemC
   python pc_channel.py mlir         # the MLIR it was emitted from

In Python the whole surface is one call:

.. code-block:: python

   import allo.dataflow as df
   from allo.ir.types import int32, Stream

   @df.region()
   def top(A: int32[8], B: int32[8]):
       fifo: Stream[int32, 4][1]

       @df.kernel(mapping=[1], args=[A])
       def producer(a: int32[8]):
           for i in range(8):
               fifo[0].put(a[i])

       @df.kernel(mapping=[1], args=[B])
       def consumer(b: int32[8]):
           for i in range(8):
               b[i] = fifo[0].get() + 1

   print(df.build(top, target="systemc").hls_code)          # emit only
   mod = df.build(top, target="systemc", mode="csim",       # compile + run
                  project="pc_project")                     #   (needs Catapult)

Emitting a design of this size takes under a second. ``csim`` compiles the emitted
``kernel.cpp`` with ``g++`` against Catapult's bundled ``libsystemc`` and runs it; ``csyn``
and ``cosim`` invoke Catapult.

The larger worked example is **EVA** (``examples/eva/``), a mesh of fused router + PE nodes
emitted at 1x1 as 22 ``SC_MODULE``\ s and 32 FIFOs. ``examples/eva/README.md`` has its
environment and the two commands.

How it works
------------

Three emitters form a subclass chain, each overriding only what differs:

.. code-block:: text

   VhlsModuleEmitter        ap_int, #pragma HLS ...          mlir/lib/Translation/EmitVivadoHLS.cpp
     └── CatapultModuleEmitter   ac_int, #pragma hls_...      mlir/lib/Translation/EmitCatapultHLS.cpp
           └── SystemCModuleEmitter   SC_MODULE + Connections mlir/lib/Translation/EmitSystemC.cpp

So arithmetic, control flow and most of a kernel body are the Vitis emitter's; Catapult swaps
the vendor types and pragmas; SystemC replaces the *structure*. Emission proceeds in four
stages: the region hierarchy is flattened (an ``SC_THREAD`` cannot structurally instantiate a
sub-region), each kernel becomes an ``SC_MODULE`` with a clocked ``SC_THREAD run()``, the
region top becomes a wiring module that declares one channel per link and binds every port,
and finally the device header and testbench are emitted.

The three modes exercise **the same emitted source** three ways:

.. list-table::
   :header-rows: 1
   :widths: 12 44 44

   * - mode
     - what runs
     - what it proves
   * - ``csim``
     - ``g++`` compiles and runs the SystemC on the host
     - functional correctness, in seconds
   * - ``csyn``
     - Catapult synthesizes the SystemC to Verilog
     - that it synthesizes; area and Fmax
   * - ``cosim``
     - the synthesized RTL runs in Xcelium against the csim golden
     - the RTL is bit-exact with the C simulation

csim and synthesis do not compile the same code -- the emitter uses several
``#ifdef __SYNTHESIS__`` splits, the largest being that a kernel's outermost repeat loop
becomes ``while (1)`` under synthesis so Catapult pipelines it rather than treating the body
as reset-action setup. **A green csim therefore does not prove the RTL is right.** Run
``cosim``.

Reference
---------

Link primitives
~~~~~~~~~~~~~~~

The link type is a design decision, not a label: each maps to different RTL. ``Wire`` and
``Channel`` are **SystemC-only** -- every other target raises ``NotImplementedError``.

.. list-table::
   :header-rows: 1
   :widths: 26 30 22 22

   * - Allo link
     - RTL
     - handshake
     - buffered
   * - ``Wire[T]``
     - ``sc_signal<T>``
     - none
     - no
   * - ``Wire[T, comb]``
     - ``sc_signal<T>`` driven by an ``SC_METHOD`` (same-cycle output; below)
     - none
     - no
   * - ``Channel[T, valid_ready]``
     - ``Connections::Combinational<T>``
     - full valid/ready
     - no
   * - ``Channel[T, valid_only]``
     - ``_dat`` + ``_vld`` signal pair
     - valid only
     - no
   * - ``Stream[T, depth]``
     - ``Connections::Fifo`` (``AlloFifoC`` when ``empty()``/``full()`` are queried)
     - credit/handshake
     - yes

A ``Wire`` has zero storage **and** zero alignment: it reads garbage unless the two kernels are
cycle-locked. ``valid_only`` drops a datum if the consumer is not looking that cycle. Both are
deliberate performance points -- take them only when you own the timing.

Boundary arrays are not all alike either. A 1-D array scanned strictly in order becomes a
stream port (``a[i]`` lowers to ``.Pop()``); anything 2-D, strided, random or re-read becomes an
addressable memory port behind a request/response channel.

.. _systemc-comb:

Combinational outputs: ``Wire[T, comb]``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every ``sc_out`` an ``SC_THREAD`` writes is a register: the value shows after the
next clock edge, whatever the body computed. A unit whose contract is a same-cycle
output -- a register file's asynchronous read, a decoder, a mux -- declares it
(README D-13; *declared, never inferred*, because an inferred form would move the
port's latency silently when its cone changed shape):

.. code-block:: python

   w_qa: Wire[UInt(16), comb]          # a region-scope link; Wire(T, (), comb=True) is the same

   @df.kernel(mapping=[1], args=[])
   def rf():
       mem: UInt(16)[32]
       for _ in range(n):
           a: int32 = w_ra.get()      # Wire inputs
           ...
           w_qa.put(mem[a])           # the comb output: read storage, THEN store
           if e:
               mem[x] = d

The marker is part of the IR type (``!allo.wire<i16, comb>``) and the emitter builds
the kernel as the form measured in ``dev/records/minitpu/u2_comb_read_2026-10-02.rst``
(form e) and ``u2_comb_wire_impl_2026-10-02.rst``:

- **Storage a comb cone reads becomes signal storage**, ``sc_signal<T> mem[N]`` as a
  module member (a kernel-local array or an ``@ Stateful`` one; multi-dimensional
  arrays are flattened), read with ``.read()`` in both processes and written with
  ``.write()`` in the thread. Catapult requires a signal a thread writes to be set in
  the reset action (CIN-233), so the storage is **zeroed by reset**. That is a
  recorded deviation (MiniTPU's register file is never reset): it costs
  ``DFFR_X1`` in place of ``DFF_X1`` -- +10.7 % area on the w16 register file,
  all of it in the reset flops. Declaring the storage ``@ Stateful(reset=False)``
  removes it (next section). A plain member array cannot feed a combinational
  process (CIN-197), which is why the storage changes form.
- **One ``SC_METHOD(comb)`` per kernel** holds every comb port's cone: the put and,
  backwards from its value, ``Wire`` input reads, storage loads, the iteration's
  scalar temporaries (``a: int32 = a5``) with the store that reaches them, and pure
  arithmetic. It is sensitive to those inputs and to every storage element. The
  thread keeps the stores, the other ports and its ``wait()``, and drops the puts
  and whatever only they consumed. A comb ``sc_out`` has no reset-action write (the
  method is its only driver).
- **The rule.** A comb cone may read storage only *before* any store to it in the
  iteration, in program order (read first, then store -- what the register file
  does); it may read no stream, channel or memory port, contain no control flow,
  depend on nothing computed outside the iteration (the induction variable,
  loop-carried values) and be driven by exactly one unconditional ``put`` per
  iteration. The thread's own reads of that storage must also precede its stores
  (a signal returns the old value until the next delta). Anything else is
  **refused at build**, naming the port and the reason (``comb port `w_qa` (rf_0):
  its value reads mem after a store to it in the same iteration ...``); the
  diagnostic is in the ``RuntimeError``.
- **Other backends refuse** a comb port: Vitis and the Catapult C++ flow because it
  is a ``Wire`` (they have none), the simulator with a clear ``NotImplementedError``
  (it has no wire semantics at all; before, it failed inside the ExecutionEngine).
- **Where it can be written.** On a region-scope link, as above. ``Wire[...]`` now
  also evaluates as a value, so a ``@df.unit`` signature can name it -- but the
  netlist still types every unit port as a ``Stream``
  (:doc:`../developer/stream_ports`), so a ``Wire`` port of a unit is not wired
  yet; a comb output belongs to a ``@df.kernel`` that drives a region-scope link.
- **``latency.json`` reports it as ``comb``**, not ``0``: the kernel entry gains
  ``"ports": {"v24": "comb", ...}`` (from the ``// allo comb ports:`` line the
  emitter leaves in the module) while ``latency``/``ii`` stay the clocked thread's.
  ``comb`` and ``latency=`` are different things (D-10).

Measured (the ``comb`` variant of ``examples/minitpu/units/vpu_regfile.py``, no hand
patch): Catapult's RTL is bit-exact against MiniTPU's ``vpu_regfile.sv`` on every
defined slot at read latency 0 and write visible after 1 edge, w16
(180,780/180,780) and w256 (45,744/45,744); DC same-flow area is the record's
(4,463 um^2 at w16). In csim a ``Wire`` link is still not cycle-locked
(:ref:`limitation-22`): the recorded ports agree with the reference at a constant
offset, one cycle less than the thread form (+3/+2/+2 against +4/+3/+3).

Unreset storage: ``@ Stateful(reset=False)``
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Storage is reset by default. Storage whose contents must survive reset (only the
unit's control is reset) -- MiniTPU's register file and FIFO contents -- is
declared (README D-14; *declared, never inferred*):

.. code-block:: python

   mem: UInt(16)[32] @ Stateful(reset=False)   # contents undefined until written

The global carries ``allo.unreset = "mem"``. Catapult refuses a signal an
``SC_THREAD`` writes unless the reset action sets it (CIN-233, no directive exempts
it), and a thread cannot be reset-less (CIN-194); a signal written by a clock-edge
``SC_METHOD`` has no reset action to satisfy
(``dev/records/minitpu/u2_comb_wire_impl_2026-10-02.rst``, F3). So, for a kernel that
stores to unreset storage:

- **The storage is signal storage** (``sc_signal<T> mem[N]``, as comb storage) and
  the reset action does not touch it.
- **Its stores move to ``SC_METHOD(wr); sensitive << clk.pos();``** with their cone
  (address, data, and the conditions of the ``if``\ s they sit under). The thread
  drops them and whatever only they needed. ``dont_initialize()`` (csim only) keeps
  the method from running once at time 0, which the flop never does.
- **``run.tcl`` gets ``directive set -RESET_CLEARS_ALL_REGS no``** whenever the
  emitted code holds unreset storage (the ``// allo unreset storage:`` marker).
  Without it Catapult adds a reset to every register, the storage included (F3
  form c). With it, a register is reset only when a reset action sets it: the
  kernel's FSM and ``done`` flag still are.
- **The rule** (refused at build, naming the storage: ``unreset storage `mem`
  (rf_0): ...``). ``wr`` runs at *every* clock edge -- under reset, before the
  kernel's first iteration and after its last -- so it may compute only what is a
  function of this cycle's inputs. The writing kernel is **Wire-only** (every
  argument a ``Wire``, no stream or channel op: only then is one iteration one
  clock cycle); each store is in the iteration block under nothing but ``if``\ s
  (no inner loop); its address, data and conditions read only ``Wire`` inputs,
  constants, arithmetic and scalar iteration temporaries written unconditionally
  earlier in the iteration -- **no storage load** (so no read-modify-write), no
  induction variable, nothing loop-carried; and the iteration reads the storage
  only before its stores (signal storage reads old until the next edge). Comb reads
  of it (D-13) are unchanged.
- **Other backends refuse it**, naming the storage: Vitis and the Catapult C++ flow
  raise ``NotImplementedError`` (no measured unreset form; Vitis' default
  ``config_rtl -reset control`` may well leave a static array unreset, but that is
  unmeasured). **The simulator treats it as ordinary storage** (initial value, then
  writes); harness verdicts mask pre-write contents as undefined.
- **``latency.json``** adds ``"storage": {"__stateful_rf_0_mem_1": "unreset"}`` to
  the kernel's entry. ``latency``/``ii`` are still the thread's loop; the write is
  visible one edge after it is presented, by construction.

Measured (the ``comb_unreset`` variant of ``examples/minitpu/units/vpu_regfile.py``,
``dev/records/minitpu/u2_unreset_impl_2026-10-02.rst``): 0 ``if ( rst )`` on the
storage, all 512 storage flops ``DFF_X1``, bit-exact against MiniTPU at read 0 /
write -> read 1 (w16 180,780/180,780, w256 45,744/45,744); DC 4,033.9 um^2 at w16
against MiniTPU's 4,021.7 -- the difference is the kernel's two reset control
flops (FSM state and ``done``), which the hand form of F3 did not have.

Configuration
~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 26 74

   * - key / variable
     - meaning
   * - ``mode``
     - ``csim`` | ``csyn`` | ``cosim`` (see above); ``ppa`` is the Catapult backend's
   * - ``configs["library"]``
     - the Catapult standard-cell library, e.g. ``nangate-45nm_beh``. **Canonical** -- the
       older ``device`` key means an FPGA part to ``vitis_hls`` and a cell library here, and
       passing an FPGA part now raises rather than failing inside Catapult
   * - ``configs["clock_period"]``
     - clock period in ns. Takes precedence over ``frequency`` (MHz). Catapult bakes the
       period into the schedule, so this is the main area/timing knob
   * - ``configs["synth_top"]``
     - synthesize one ``<kernel>_0`` submodule instead of the whole region. **Any area number
       you report needs this**: a region contains its testbench kernels and their memories
   * - ``MGC_HOME``
     - Catapult install. Emission does not need it; ``csim``/``csyn``/``cosim`` do
   * - ``SYSTEMC_HOME``
     - SystemC headers and ``libsystemc`` for the standalone ``csim`` compile
   * - ``ALLO_SYNC_RESET``
     - synchronous instead of asynchronous reset (smaller flops on the ASIC path)

Where things live
~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 44 56

   * - path
     - what
   * - ``mlir/lib/Translation/EmitSystemC.cpp``
     - the emitter; ``EmitSystemC.md`` beside it is its internal documentation
   * - ``allo/backend/hls.py``
     - the ``systemc`` platform: modes, the csim runner, the ``Wire``/``Channel`` guard
   * - ``allo/backend/catapult.py``
     - the TCL generator both Catapult flows share
   * - ``examples/eva/``
     - the EVA design and its emitted project
   * - ``tests/systemc/``
     - the emitter demonstrations, in Allo, with ``test_emit.py`` collecting the
       claim each one makes about the emitted text; plus what acts on them: RTL
       cosim testbenches, the Catapult ``csyn`` driver, and ``rtlsim/`` — this
       fork's SystemC-vs-RTL cross-check harness (:ref:`limitation-22`)
   * - ``dev/records/systemc/``
     - what was measured: ``VERDICTS.md``, ``reports/``, and the archived emitter output
   * - ``tests/dataflow/test_systemc_backend.py``
     - the suite; emit cases run anywhere, csim/cosim cases skip without ``MGC_HOME``
   * - ``dev/systemc/``
     - the emitter author's working notes, kept whole and not rewritten

Limits and known failures
-------------------------

- **Neither Catapult nor Xcelium is installed on this fork's development host**
  (``dev/toolchains.rst``). Everything past emission is therefore unverified here: the
  emit-only tests pass, and every ``csim``/``csyn``/``cosim`` case skips. The evidence that
  they pass elsewhere is in ``dev/records/systemc/reports/`` and ``tests/dataflow/COSIM_REGRESSION.md``.
- **A ``Wire`` is not a cheap ``Channel``.** With no storage and no alignment it reads garbage
  unless the producer and consumer are cycle-locked; ``tests/systemc/rtlsim/`` reproduces
  that on two simulators, and :ref:`limitation-22` records it.
- **Bit-slicing a packed value wider than 64 bits** works in csim and can fail at ``csyn``:
  Catapult rejects subclassing its builtin ``ac_int`` (CIN-15), and the emitted ``ap_int`` shim
  is such a subclass.
- **``csyn`` needs a build subdirectory.** Running Catapult in the directory holding
  ``kernel.cpp`` degrades ``Connections::In``/``Out`` ports to raw ``sc_signal``\ s (CIN-124,
  SCHD-30). ``tests/systemc/csyn_subdir.py`` works around it.
- **``cosim`` does not work against a ``synth_top`` submodule.** SCVerify wraps the design top,
  and the stimulus path into the region's memories disappears. Measure and verify separately.
- **The timed dataflow simulator is not here.** ``SystemC-emitter`` also carries a per-PE
  simulated-clock simulator with a ``get_cycles()`` read-out; it was not merged, because it
  rewrites the same code this fork rewrote for OpenMP team sizing. ``dev/systemc/SIMULATOR.md``
  describes it.

Known fixed
-----------

Each of these passed every emit-only test and failed the first time g++ or the
testbench ran. ``tests/dataflow/test_systemc_csim_regress.py`` compiles and runs
one design per bug (it skips without ``MGC_HOME`` and ``SYSTEMC_HOME``).

- **bf16 ports did not compile** (found by the MiniTPU ``bf16_add`` unit,
  2026-10-02): ``no matching function for call to sc_trace(sc_trace_file*&,
  const ac::bfloat16&, ...)``. Connections and ``sc_signal`` call ``sc_trace``
  unqualified from inside their own namespaces, so only argument-dependent lookup
  finds an overload, and it looks in the float type's namespace. The emitted
  overload was global; ``ac::bfloat16`` lives in ``ac``. It is now emitted in
  ``namespace ac``. ``ac_ieee_float`` (f16, f32) is a global template, so its
  global overloads were found; ``double`` uses SystemC's own.
- **The testbench deadlocked with two or more boundary streams** (same unit):
  one ``src()`` thread pushed all of input 0 before any of input 1, and one
  ``snk()`` drained all of output 0 before output 1, over ``Combinational``
  channels that hold no data. A kernel computing ``a[i] + b[i]`` blocked on its
  first pop of ``b`` while ``src`` blocked on its second push of ``a``: no
  output, and the csim spun forever. The testbench now runs one ``SC_THREAD`` per
  boundary stream (``src_<port>``/``snk_<port>``), so it accepts any order the
  kernel can consume or produce in, and the last sink calls ``sc_stop()``. The
  run is bounded too: after ``ALLO_TB_MAX_CYCLES`` cycles (default ``2000 x``
  the largest array ``+ 200000``; override with ``-D``) an undrained output
  prints ``TB DEADLOCK`` and exits 1, so a hang in csim now means the design
  deadlocks, and says so.
- **Float testbench data went through decimal text** (same unit). Inputs were
  written by Python as ``str(x)`` and read with ``>> float``; outputs were
  printed at ``setprecision(9)``. Three losses: libstdc++'s ``>> float`` sets
  ``failbit`` on ``"nan"`` and ``"inf"``, after which every later read of the
  file silently fails; a NaN's sign and payload cannot survive text; and
  ``ac::bfloat16(float)`` truncates, so the shortest decimal of a bf16 value
  reads back as the bf16 below it (``0x3f81`` prints as ``1.00781``, which
  becomes ``0x3f80``). The data files now carry each float's IEEE bit pattern as
  an unsigned integer: ``allo/backend/systemc.py`` writes and reads bits, and the
  testbench converts with ``_ffrombits<T>`` (``set_data``) and ``_fbits``. This
  holds for f16, bf16, f32 and f64, for stream and memory ports; integers are
  unchanged. Memory read-out merges replicas by OR-ing bits instead of summing
  floats, and ``_fbits`` no longer sign-extends a negative 16-bit float.
- **csim flipped the sign of zeros on a signal** (found while testing the
  previous fix). ``sc_signal::write()`` and ``update()`` drop a write whose value
  ``==`` the current one, and IEEE says ``+0 == -0``: a ``-0`` written after a
  ``+0``, or the reverse, never reached the reader of a Connections channel or a
  memory pin. An RTL wire carries the sign bit, so csim and RTL disagreed. The
  header now specializes both members for each float payload (bf16, f16, f32,
  f64) and writer policy to compare bits. It applies to the OSCI kernel only
  (2.3.2 and later); not under ``__SYNTHESIS__``, and not under Xcelium's own
  SystemC (``NCSC``), so **the testbench side of an RTL cosim still has this
  defect** -- a cosim mismatch on a signed zero is the testbench's, not the RTL's.
- **Any region with a ``uint16`` port did not compile** (the ``bf16_add``
  ``bits`` variant, S1 in ``dev/records/minitpu/u1_bf16_add_bits_2026-10-02.rst``).
  A link's sign is not in its MLIR type (signless ``i16``); the emitter recovered
  it from an ``unsigned`` attribute on the defining op or on a user. A kernel's
  write port has only stores, which carry none, and a region's ports have no
  tagged user at all, so both emitted as signed ``ac_int`` and could not bind to
  the unsigned read port. A function argument's sign is now read from the
  function's ``itypes``, as the HLS emitters do.
- **A nested function returning ``UInt`` did not compile** (S2, same record).
  The callee's signature took its result's sign from its ``otypes``
  (``f(..., ac_int<5,false>*)``) but the call site declared the result buffer
  from the signless call result (``ac_int<5,true>``), and the pointer did not
  convert. The call site now reads the callee's ``otypes`` too. The fix is in
  the shared ``VhlsModuleEmitter::emitCall``, so Vitis and Catapult C++ get it.
- **bf16 ports failed Catapult's ``go analyze``** (C1, the MiniTPU Catapult
  track, ``dev/records/minitpu/u1_bf16_add_catapult_2026-10-02/``): ``CRD-135
  class "ac::bfloat16" has no member "Marshall"``. Connections' ``marshaller.h``
  defines ``Wrapped<ac::bfloat16>`` (and ``Wrapped<ac_ieee_float<...>>``) only if
  ``ac_std_float.h`` was included before ``mc_connections.h``; the emitter
  included it after. The g++ csim compiles the non-synthesis Connections path and
  never saw it; ``g++ -fsyntax-only -D__SYNTHESIS__`` does, in a second, and is
  what ``test_float_ports_pass_the_synthesis_front_end`` runs. The ac headers now
  come first. With that order ``ac_sc.h`` also supplies ``sc_trace`` for every
  ac float, so the emitter's own overloads (the first entry above) are gone: they
  were ambiguous with the library's. Checked by hand: the ``bf16_add`` native unit
  passes ``go analyze`` and ``go compile`` unpatched (Catapult 2024.2,
  2026-10-02).
- **A kernel whose links are all ``Wire``\ s failed synthesis** (C2, same
  record): ``CIN-123 Loop 'while' in thread 'run' must have a wait``. A
  steady-state kernel's synthesized ``while (1)`` relied on its Connections
  handshake for the cycle boundary and emitted its ``wait()`` for csim only; a
  ``Wire`` is a plain ``sc_signal`` and has no handshake. A loop body with no
  stream or channel op now gets its ``wait()`` under synthesis too (valid_only
  channel helpers already wait). Checked by hand: the ``bf16_add`` Wire variant
  passes ``go analyze`` and ``go compile`` unpatched.
- **Float casts did not compile, or truncated** (S4, the MiniTPU multiplier
  units, ``dev/records/minitpu/u1_mul_2026-10-02.rst``). The shared emitter wrote
  every cast as ``D v = x;``. ac floats have only explicit constructors, so g++
  rejected 24 of the 42 float/int pairs among bf16, f16, f32, f64, i8, i32 and
  ui16 (bf16 -> f32 first). Of the pairs that compiled, int -> bf16 truncated
  (``ac::bfloat16``'s constructors hard-code ``AC_TRN_ZERO``), where arith rounds
  to nearest even. The SystemC emitter now writes each such cast through header
  helpers (``_fstd``, ``_fto``, ``_ito``; float -> int through
  ``convert_to_ac_int``, toward zero), converting via ``ac_std_float`` with one
  round-to-nearest-even. The other 40 pairs then match numpy; bf16 <-> f16 fails,
  earlier, in the MLIR pipeline. Vitis and Catapult output is unchanged (a
  hook); the Catapult HLS flow keeps the old behaviour.
- **A ``UInt(24)`` port could not be read back** (S5, same record): ``KeyError:
  'ui24'``. The data-file reader knew only 8-, 16-, 32- and 64-bit integers. An
  ``i<N>``/``ui<N>`` of any width up to 64 now reads into the smallest numpy
  container (``ui24`` -> ``uint32``), as the simulator's argument path accepts it.
- **An array written and read back did not compile** (S6, the MiniTPU
  multiplier units): ``c[i] = a[i] + 1; c[i] = c[i] * 256`` failed g++ with
  ``'v1' was not declared``. ``c`` was classified read+write *and* sequentially
  streamable, so it got neither a stream port (pure in/out only) nor a memory
  port (non-streamable only). A stream moves each element once, one way, so an
  array with both a load and a store site, or with two of either, is no longer
  streamable and takes the memory-port path (an in-place ``c[i] = c[i] + a[i]``
  too). A read+write array that still has no port form is refused with an
  error instead of emitted.
