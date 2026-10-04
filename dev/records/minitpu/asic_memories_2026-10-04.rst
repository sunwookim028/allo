..  Copyright Allo authors. All Rights Reserved.
    SPDX-License-Identifier: Apache-2.0

####################################################################
ASIC memories: an SRAM-macro path, and ASIC-flavoured swap-ins (D-12)
####################################################################

.. note::

   **Dated record, 2026-10-04.** zhang-21, branch ``asic-memories`` (worktree
   ``scratch/wt-asicmem`` from ``origin/u1-pilot-sync2`` at ``8fc71db2``, its
   own ``mlir/build``), the owner away (mandate ``1160a495``). The owner's
   decision of the day: *"add an SRAM-macro path now -- for instance the
   codebase should have OpenRAM; also revisit FPGA-SRAM-primitive-flavoured
   design choices like the multi-copy LUTRAM VREG implementation, and
   implement then integrate standard ASIC-flavoured swap-ins."* MiniTPU at
   ``b3ba0a4d``, read only. Catapult 2024.2/1130128, DC W-2024.09, Library
   Compiler W-2024.09-SP5-3, Verilator 5.052. Scratch (not kept):
   ``scratch/asicmem_*``. Record files: ``asic_memories_2026-10-04/``.

Every memory D-12 has lowered so far is flops: the ``server`` and ``replica``
lowerings (``u2_d12_prototype_2026-10-04.rst``) and the one-kernel forms
before them. MiniTPU's VREG is three LUTRAM copies and its VMEM an URAM
(``xpm_memory_tdpram``); both are FPGA primitives, and an ASIC number for
either has so far been a flop array. This record adds the macro path and
names the swap-ins, in three stages, each committed on its own.

1. Inventory and plan
=====================

1.1 What exists for FreePDK45 / NanGate 45 nm on this host
-----------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 48 30

   * - Option
     - Finding
     - Verdict
   * - **Vendor / PDK memory ``.lib``** on the host
     - ``find`` over ``/work/shared/common``, ``/opt`` and the FreePDK45 ADK
       (``ADK_PKGS/freepdk-45nm``; the mflowgen ``view-standard`` used by every
       DC run, ``stdcells.db`` md5 ``f5560259…``) for ``*ram*.lib``,
       ``*sram*.lib``, ``*ram*.db``: **none**. The ADK is standard cells only
       (``stdcells.{lib,db,lef,gds,v,cdl}``); the NCSU FreePDK45 PDK itself
       (``ncsu_basekit``) is **not installed** anywhere readable. No memory
       compiler (no Artisan/Synopsys MC) either.
     - nothing to reuse
   * - **Catapult's own RAM models** (``$MGC_HOME/pkgs/siflibs``)
     - ``ccs_sample_mem.lib`` (encrypted ``%MGC_HLSFormat%``) holds
       ``ccs_ram_sync_{1R1W,dualport,singleport,singleport_wmask}``: behavioural
       ``.v``, **no area** (``u2_d12_prototype`` s.3 used it; DC then
       synthesizes the ``translate_off`` body as nothing). All synchronous,
       read latency 1. The D-12 ``mid`` VMEM on ``ccs_ram_sync_dualport``
       reached only **II=2** (SCHD-30 at II=1).
     - a stand-in with no ASIC number; keep as the comparison row
   * - **OpenRAM** (`VLSIDA/OpenRAM <https://github.com/VLSIDA/OpenRAM>`_)
     - Not installed on the host. Cloned to
       ``/work/shared/users/phd/sk3463/tools/OpenRAM`` at **``b2b069ce``**
       (``v1.2.48-41-gb2b069ce``, 2026-08-16). BSD-3-Clause compiler; its
       **outputs are the user's** (the licence covers the tool; the FreePDK45
       technology files it bundles are the OpenRAM authors' own cells under
       the same licence, the transistor models ``technology/freepdk45/models``
       Apache-2.0 PTM). ``freepdk45`` is one of its reference technologies
       and ships **``cell_1rw`` and ``cell_2rw``** bitcells (6T and 8T
       dual-port), so a true 2RW macro -- MiniTPU's VMEM port shape -- is
       available without the NCSU PDK: the PDK is needed only for Calibre
       DRC/LVS (``DRCLVS_HOME``), which is off by default
       (``check_lvsdrc = False``). Generates ``.v`` (behavioural: inputs
       registered at ``posedge``, read and write at ``negedge``, so read
       latency 1 and a write visible to the next cycle's read), ``.lib``
       (per corner; ``area`` = layout width x height in um^2), ``.lef``,
       ``.gds``, ``.sp``, ``.html`` datasheet. Needs Python >= 3.8 with
       numpy/scipy/scikit-learn; ngspice only for simulated characterization
       (``analytical_delay = True`` by default, an Elmore model). **No root
       needed**: a conda env at ``tools/envs/openram`` (Python 3.11.16,
       numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1, ngspice 41, ciel 3.0.0;
       the exact package list is ``asic_memories_2026-10-04/openram_env/``).
       ``use_nix = False`` is required (the default now wants Nix for tool
       setup; ``globals.py:223`` errors otherwise). Generation time: s.1.3.
     - **the macro source**
   * - **Catapult user memory: Library Builder / Memory Generator**
     - ``flow package require MemGen; flow run /MemGen/MemoryGenerator_BuildLib
       {…}`` from a TCL spec (``catapult_lb_useref.pdf`` ch. 1.2; the shipped
       example ``$MGC_HOME/shared/examples/libraries/mem_pipe/my_custom_ram.tcl``)
       imports a Verilog model and writes ``<module>.lib`` (Catapult's format)
       plus SystemC/SCVerify transactors. **Tried on the shipped example:
       runs under the plain Catapult product licence** (LIC-14; no Library
       Builder feature was checked out) in 20 s. The library is then
       ``solution library add <module> -file <path>`` (or via
       ``options set /ComponentLibs/SearchPath``) and
       ``directive set <array>:rsc -MAP_TO_MODULE <lib>.<module>``. Limits
       stated in the manual: one READLATENCY/WRITELATENCY for all ports;
       asynchronous read ports not supported; ``AREA`` is an integer in the
       base library's units.
     - **the integration route**
   * - **Hand-written ``ccs_*``-style wrapper**
     - Not possible: ``ccs_sample_mem.lib`` is encrypted, so there is no
       text library to copy; a wrapper ``.v`` would still need MemGen to
       become a Catapult library.
     - not needed
   * - **``.lib`` -> ``.db`` for DC**
     - ``lc_shell`` is not in the DC module's PATH; ``/opt/synopsys/lc/
       {W-2024.09,V-2023.12-SP4}/bin/lc_shell`` die with a ``libkrb5``
       symbol error on this RHEL 8.10; **``/opt/synopsys/lc/W-2024.09-SP5-3/
       bin/lc_shell`` runs** (``LC_OK``; also ``R-2020.09-SP5``). ``dc_shell``
       alone cannot (``read_lib`` -> LCSH-3).
     - ``lc_shell`` W-2024.09-SP5-3

1.2 Plan and pins
------------------

* **Macro source: OpenRAM ``b2b069ce``**, FreePDK45 as bundled with it (no
  NCSU PDK; DRC/LVS not run -- stated on every number). Pinned in
  ``dev/toolchains.rst`` in this commit, with the conda env's explicit
  package list beside this record.
* **Catapult integration: Memory Generator** from a TCL spec written by a
  small generator (``asic_memories_2026-10-04/scripts/memgen_spec.py``) from
  the macro's ``.v`` and ``.lib``: ports as OpenRAM names them
  (``clk<i>, csb<i>, web<i>, addr<i>, din<i>, dout<i>``), ``READLATENCY 1``,
  ``WRITELATENCY 1``, ``RDWRRESOLUTION UNKNOWN`` (OpenRAM's model: a
  same-port read in a write cycle does not happen -- ``web`` selects one;
  cross-port same-word is the D-12 collision obligation), ``AREA`` the
  ``.lib`` area in um^2. The D-12 server's storage array is mapped onto it
  by ``MAP_TO_MODULE``.
* **Stage 2 memory: MiniTPU's VMEM ``mid`` (4,096 x 64 b)** as one 2RW macro
  if OpenRAM can build it in the time this record has; the largest 2RW
  OpenRAM macro that builds otherwise, stated. ``narrow`` (8 x 64 b) is not
  an SRAM's size and stays flops.
* **Verification**: the Catapult RTL with the macro's behavioural ``.v``
  through the harness against ``vpu_word_array.sv`` per cycle
  (``cmp_wa_heldaddr_probe.py``), expecting the macro's own latency to show
  as the per-port read latency; MiniTPU's 3/2/1 (compute read / DMA read /
  write visible) are the reference.
* **DC**: the macro as a black box -- ``.lib`` -> ``.db`` by ``lc_shell``,
  linked beside ``stdcells.db``; the U2 flow (``dc_u1.tcl``, 3.33 ns)
  otherwise unchanged. Reported: macro area (from the ``.lib``) + logic area,
  beside MiniTPU's flop-mapped number and the flop-mapped Allo number.
* **Stage 3 (``allo/``)**: the lowerings as D-12 names them, in
  ``memory.json``: ``registers`` (today's ``server``, the ASIC form),
  ``sram`` (this macro path, with the macro's ports and latency declared and
  checked against each ``Port``; refused when the macro cannot honour a
  port), ``replica`` marked FPGA-only (LUTRAM-style; refused unless the
  target says FPGA). An ``impl=`` on ``Memory``, tests, the regfile on
  ``registers`` and the VMEM on ``sram`` in the D-12 unit files; TinyTPU
  emission identity (vhls ``6bc774bc…``, catapult ``ade1ab5d…``) checked.

1.3 OpenRAM generation time (measured here)
--------------------------------------------

All runs: ``freepdk45``, 2RW (``num_rw_ports = 2``), 64-bit words, nominal
corner only, analytical delay, DRC/LVS off, one process (the compiler is
single-threaded outside characterization). Configs in ``openram/``.

.. list-table::
   :header-rows: 1
   :widths: 22 16 62

   * - Macro
     - Wall time
     - Note
   * - 32 x 64 b, routers on (defaults)
     - **> 25 min, killed**
     - stuck in ``route_supplies`` (``router/supply_router.py``, a
       pure-Python maze router over the gds); ``py-spy`` stack in the log.
   * - 32 x 64 b, ``route_supplies = False``
     - 273 s
     - 263 s of it the signal escape router (``perimeter_pins``).
   * - 32 x 64 b, both routers off
     - **56 s**
     - ``** Routing: 45.6 s`` (the internal channel routes), the rest 10 s.
       Output: ``.v .lib .lef .gds .sp .html``; LEF ``SIZE 263.275 BY
       107.475`` (28,295.5 um^2, the Liberty ``area``); the macro core
       without power ring or escape pins. **This is the setting used.**
   * - 512 x 64 b, both routers off
     - **1,151 s**
     - 839 s in ``bitcell_array.create_instances`` -> ``connect_pin`` ->
       ``__eq__`` (a linear pin search per connection), 300 s routing.
       103,782.1 um^2 (410.3 x 252.9 um).
   * - 1,024 x 64 b, both routers off
     - **3,190 s**
     - 2,698 s submodules, 474 s routing. 182,918.1 um^2 (411.7 x 444.3 um;
       2.79 um^2/bit). ``openram/sram_2rw_64x1024_*``; no instance uses it
       yet.
   * - 4,096 x 64 b (``mid``), both routers off
     - **not finished**
     - still in the bitcell array's pin connection after 76 min when this
       record closed (the scaling above says hours); killed by its 4 h
       timeout. One ``mid`` macro is a job for a faster pin connection or
       OpenRAM's ``num_banks``; eight ``w512`` banks are the alternative.

The ``.v`` is a behavioural model (inputs registered at ``posedge``, read
and write at ``negedge``, ``#(DELAY)`` on the read data, ``$display``
traffic); ``memgen_spec.py`` derives a Verilator copy from it (s.2).

2. The macro path, end to end
=============================

2.1 What was built
------------------

* **Macros** (``openram/``, OpenRAM as pinned, routers off): ``small`` 32 x
  64 b 2RW in 56 s, **28,295.5 um^2** (263.3 x 107.5 um); ``w512`` 512 x
  64 b 2RW in **1,151 s** (839 s of it the bitcell array's pin connection),
  **103,782.1 um^2** (410.3 x 252.9 um; 3.17 um^2/bit against the 32-word
  macro's 13.8: the 2RW periphery amortising). The one-macro ``mid`` (4,096 x 64 b) did not build in this
  record's time (s.1.3); ``mid`` on eight ``w512`` banks is the next step
  (Catapult's ``-BLOCK_SIZE`` decomposition, not tried).
* **Catapult library**: ``memgen_spec.py`` (now also
  ``allo.backend.catapult.memgen_spec``/``build_memory_library``) writes the
  Memory Generator spec from the macro's ``.v`` and ``.lib`` -- two
  ReadWrite ports, ``csb`` an active-low PORT_ENABLE, ``web`` an active-low
  WRITE_ENABLE, READLATENCY/WRITELATENCY 1, RDWRRESOLUTION UNKNOWN, AREA
  the Liberty area, READDELAY = clock/2 + the Liberty's dout delay (OpenRAM's
  output moves after the falling edge) -- and ``catapult -shell -f`` builds
  ``<module>.lib`` in 6-8 s (``PATHTYPE copy``; ``absolute`` is not a value).
* **The D-12 ``mid``/``narrow`` server on the macro**: ``sram_csyn.py``
  (the D-12 record's driver plus ``--macro-lib/--macro``, or the ``sram``
  lowering of stage 3) adds the library after ``go compile`` and maps the
  server's array with ``directive set /wa_d12g/vmem_mem_0/run/mem:rsc
  -MAP_TO_MODULE <m>.<m>``. Catapult confirms ``MEM-4: ... mapped to
  'sram_2rw_64x32_freepdk45.sram_2rw_64x32_freepdk45' (size: 32 x 64)`` and
  ``concat_rtl.v`` instantiates the macro (``sram_2rw_64x32_freepdk45
  #(.DATA_WIDTH(64), .ADDR_WIDTH(5)) mem_rsc_comp``) with the model copied
  in front.

2.2 II=1 is refused on a read-or-write port, II=2 schedules
-----------------------------------------------------------

The two ``rw`` ports each make one access per cycle: ``if e: mem[a] = d
else: _p[0] = mem[a]`` (the RAM-mappable form ``u2_word_array`` found).
Catapult refuses this at **II=1** on the macro exactly as it did on
``ccs_ram_sync_dualport`` (``u2_d12_prototype`` s.3): SCHD-30, "chained
feedback data dependency at time 11cy+2.664" from the port's MEMORYREAD
(output ``_c_p(0).sva#1``, the pipe head held across the write branch) and
its MEMORYWRITE (output ``mem:rsc.@``, the memory state). Six one-minute
probes on ``small`` (``scratch/asicmem_cat/small_ii1*``), all SCHD-30:

.. list-table::
   :header-rows: 1
   :widths: 60 40

   * - Variant
     - Result
   * - as emitted (READDELAY 1.939 ns)
     - SCHD-30
   * - ``ignore_memory_precedences -from *read_mem -to *write_mem`` (the
       direction the D-12 record did not try)
     - SCHD-30
   * - both directions ignored
     - SCHD-30
   * - READDELAY 0.3 ns (a posedge-style output), both ignored
     - SCHD-30
   * - ports declared at latency 1/1 (no pipe-as-data at all)
     - SCHD-30, output ``_c_p.sva#2``
   * - the pipe head assigned in both branches (write-first; no held value)
     - SCHD-30, now ``pmx.sva#1``: the port's read-or-write mux itself

*(Superseded by s.6.3: the edge none of these probes released is the order
between the two ports' WRITES; released together with the cross-port
read/write edges -- what the collision obligation already states -- the
2RW macro schedules at II=1 and matches MiniTPU per cycle.)* So the obstacle
is Catapult's model of a ReadWrite port whose read and write
alternate under a per-cycle condition: the memory-state and read-data
feedbacks are chained within one cycle whatever the delays and precedences.
A 1R1W macro (read and write on different physical ports) would avoid the
mux, but is not MiniTPU's VMEM, whose two ports are both read/write. **At
II=2 the design schedules on the macro** (``vmem_mem_0`` 3 c-steps,
``port_c_0``/``port_d_0`` ii 2, latency 2; ``latency.json`` and
``memory.json`` in ``catapult/``), the same as on ``ccs_ram_sync_dualport``.

2.3 Per cycle against ``vpu_word_array.sv``
--------------------------------------------

``cmp_wa_sram.py`` = the D-12 record's held-address probe script plus the
macro's Verilator model (``--extra-src``, the ``_sim.v`` copy: no
``#(T_HOLD) dout = 'bx`` -- Verilator ignores the delay and the X would race
the consumer's posedge sample -- and no ``#(DELAY)``, which Verilator 5
refuses without a timing mode; the copy of the model that Catapult put in
``concat_rtl.v`` is cut out, ``--cut-module``) and ``--stretch 2``: each
command is held for the port kernels' II (2 cycles) and the output sampled
every 2 rows, MiniTPU still taking one command per cycle.

**Result, ``small``, II=2 on the macro: defined 10,513/10,513** -- compute
5,241/5,241, DMA 5,272/5,272 (14,705 masked, the U2 census), 12,609 RTL
cycles, 22 s (``logs/cmp_small_sram_ii2.txt``, the stage-3 ``sram`` build);
**``w512``: defined 9,667/9,667** -- compute 4,850/4,850, DMA 4,817/4,817
(``logs/cmp_w512_sram_ii2.txt``). Every defined slot of both ports
matches MiniTPU at a constant per-port output offset. **The latency column
is not MiniTPU's**: at II=2 one command occupies two cycles, so the
measured edge counts (compute data visible 5-6 edges after the command's
first edge = 3 iterations, the declared 3-deep pipe over a latency-1 macro;
DMA 1-2 edges) are iteration counts times two, not the 3/2/1 cycles of
MiniTPU at II=1, and ``cmp_wa_sram`` prints LATENCY-MISMATCH for all three
kernels against the declared 3. The step probes (read/visibility) are not
meaningful under a stretch and are recorded as printed, not read. **The
macro's own latency (1, write visible next cycle) is honoured**: the
visibility probes same-port read 1 iteration, as MiniTPU's 1 cycle.

2.4 DC with the macro as a black box
------------------------------------

``dc/dc_sram.tcl`` is the U2 flow (``dc_u1.tcl``, 3.33 ns, unchanged) plus
``MACRO_DB`` (the Liberty through ``lc_shell`` W-2024.09-SP5-3, ``read_lib;
write_lib -format db``, 4 s; linked beside ``stdcells.db``) and ``DEFINES``
(MiniTPU's ``vpu_pkg`` instance defines). The macro module is cut from
``concat_rtl.v`` (it is linked, not synthesized); DC reports it as
``Macro/Black Box area`` with the Liberty's number and
``report_cell -filter is_black_box`` names it (``dc/*/`` reports).
MiniTPU's number uses ``dc/mtpu_word_array_flops.sv``: ``vpu_word_array.sv``
with the ``ifdef SYNTHESIS`` (Xilinx ``xpm_memory_tdpram``) branch removed
and the model's two ``always_ff`` merged (DC refuses two processes driving
one array, ELAB-366); the ``ifdef`` would otherwise hand DC the XPM.

.. list-table::
   :header-rows: 1
   :widths: 34 16 16 16 18

   * - 3.33 ns, um^2
     - total
     - comb
     - seq (cells)
     - macro
   * - **``small`` (32 x 64 b, 2,048 bit)**
     -
     -
     -
     -
   * - MiniTPU flop-mapped (``vpu_word_array``, sim model)
     - **17,792.5**
     - 6,948.7
     - 10,843.8 (2,402)
     - --
   * - Allo ``registers`` (D-12 server, unreset, II=1)
     - **28,493.7**
     - 7,108.3
     - 21,385.3 (4,335)
     - --
   * - Allo ``sram`` (OpenRAM 2RW macro, II=2)
     - **30,474.0**
     - 409.6
     - 1,768.9 (333)
     - **28,295.5**
   * - **``w512`` (512 x 64 b, 32,768 bit)**
     -
     -
     -
     -
   * - MiniTPU flop-mapped
     - **262,797.6**
     - 111,122.8
     - 151,674.8 (33,602)
     - --
   * - Allo ``registers`` (Catapult area score 205,477, 560 s; DC 1,604 s)
     - **438,411.6**
     - 112,722.3
     - 325,689.3 (66,263)
     - --
   * - Allo ``sram`` (OpenRAM 2RW macro, II=2)
     - **105,968.1**
     - 417.1
     - 1,768.9 (333)
     - ****103,782.1****

Read it with care. **At 512 words the macro path is 2.5x smaller than
MiniTPU's flop array** (105,968 against 262,798 um^2; the macro itself
103,782) and the crossover lies between 32 and 512 words. At 32 words the
macro is no saving: OpenRAM's 2RW
periphery (two decoders, two sets of sense amplifiers and write drivers,
the control and replica columns) is 28 k um^2 before the first bitcell, and
the flop array it replaces is 11-21 k; the logic beside the macro is small
(2.2 k: two port kernels and the server's address/data pins and read pipes).
The flop-mapped Allo number is 1.6x MiniTPU's at this size (4,335 against
2,402 sequential cells for 2,048 bits plus 320 pipe flops): the unreset
``sc_signal`` server form carries more flops than the storage -- a finding
for the ``registers`` lowering, not investigated here. All macro numbers
are the core without power ring or escape routing (routers off, s.1.3) and
without DRC/LVS, as stated on every row.

3. The swap-ins as stated lowerings (``allo/compose.py``)
=========================================================

D-12 names the implementations a backend may lower a ported memory through:
"replica, registers, SRAM". They are now the lowerings ``compose`` knows,
each reported in ``memory.json``:

.. list-table::
   :header-rows: 1
   :widths: 14 50 36

   * - Lowering
     - What it builds
     - Where it is allowed
   * - ``registers``
     - one flop array in the generated ``<mem>_mem`` kernel; latency-0 reads
       are combinational muxes (D-13), ``L >= 1`` a pipe written as data;
       unreset when declared (D-14). The D-12 prototype's ``server``, which
       stays as an alias (``LOWERING_ALIASES``). **The default** for a
       multi-owner memory. The ASIC form.
     - every target
   * - ``sram``
     - the same server with its storage a plain array that the backend maps
       onto the macro the memory declares, ``Memory(impl=Sram(...))``
       (``Sram.from_openram(<.v>, <.lib>)`` reads module, geometry, ports and
       timing from an OpenRAM model); every ``rw`` port in the one-access
       form. On SystemC the build adds the macro's Catapult library and the
       ``MAP_TO_MODULE`` directive to ``run.tcl`` (``configs["memories"]``,
       ``allo.backend.catapult.memory_directives``), building the library
       with the Memory Generator first when ``Sram.catapult_lib`` is unset
       (``build_memory_library``); ``memory.json`` carries the macro (module,
       rows x width, ports, read latency, visible, Liberty area, files) and
       each declared port's macro port.
     - ``systemc``; the simulator runs it untimed; Vitis refused
       (``NotImplementedError``: no macro path)
   * - ``replica``
     - one copy per read port, the one writer broadcasting -- the LUTRAM form
       (MiniTPU's VREG). **FPGA-only**: refused unless the target says FPGA
       (``vhls``) or the caller does (``technology="fpga"`` on
       ``plan/region/build``); ``systemc`` is an ASIC target (D-1), the
       simulator says nothing -- both refuse.
     - FPGA targets only
   * - ``local``, ``shared``
     - unchanged (one owner; D-11's refusal)
     - --

**What ``sram`` checks against the declaration, refusing by port name**
(``Architecture._check_sram``, ``sram_ports``; ``tests/dataflow/
test_compose_sram.py``): each declared port gets a physical port of its own
kind, else an ``rw`` one, and a port with none left is named with what the
macro offers; ``count > 1`` is refused (a macro port is one access per
cycle); ``latency=0`` is refused ("an asynchronous read; the macro's read is
synchronous"); ``latency`` below the macro's; ``visible`` other than the
macro's; more rows or wider words than the macro; ``reset=True`` ("an SRAM
macro is not reset; declare reset=False or lower it to registers"); and
``impl="sram"`` without a macro, or a ``lowering={"m": "sram"}`` on a memory
that declares none. The choice order is ``lowering[m]`` (per build) over the
memory's ``impl`` over the default, so the D-12 harness variants
(``d12_server``) keep working on a memory that declares an ``Sram``.

**Applied**: ``vpu_regfile_d12.VREG`` declares ``impl="registers"``; its
``replica`` variants pass ``technology="fpga"`` (the harness names the LUTRAM
form on purpose). ``vpu_word_array_d12.SRAMS`` holds the macros this record
built (``small``, ``w512``; from the record's ``openram/`` files, the
compiled Catapult library from ``$MINITPU_SRAM_<INST>_CATAPULT_LIB`` or
built on the fly), ``architecture(..., impl=None)`` takes the instance's
macro when there is one, and the harness variants ``d12_sram`` /
``d12_sram_wire`` refuse an instance without a macro, naming it. ``make
TARGET`` untouched. The end-to-end run of the ``sram`` lowering through
``compose`` alone (library built by ``build_memory_library``, ``run.tcl``
emitted by compose, Catapult, DC) is what s.2.3-2.4 measured on ``small``
and ``w512`` (``catapult/small_sram_ii2``, ``catapult/w512_sram_ii2``).

**Regression**: ``pytest tests/dataflow/test_systemc*.py tests/test_memory.py
tests/dataflow/test_compose_memory_ports.py tests/dataflow/test_compose_sram.py``
REGRESS_RESULT; TinyTPU emission unchanged (``vhls`` sha256 ``6bc774bc…``
166,563 B, ``catapult`` ``ade1ab5d…`` 170,812 B, ``hash_tinytpu.py``);
``pylint`` on ``compose.py``/``catapult.py`` adds no message.

4. Open (as of stage 3; see s.5-6 for what the follow-up closed)
=================================================================

* *(Closed in s.6: II=1 on the 2RW macro once the cross-port order is
  released.)* **II=1 on a read-or-write RAM port** is refused by Catapult in every form
  tried (s.2.2). A macro with separate read and write ports (1R1W, which
  OpenRAM's FreePDK45 also offers as ``1rw_1r``) avoids the per-cycle mux and
  is the thing to try when a design can live with it; MiniTPU's VMEM cannot.
  RTLGen/AMC (M2) are the other place to ask, as D-12 s.6 said.
* *(Closed in s.5: eight ``w512`` banks, II=1.)* **``mid`` (4,096 x 64 b) as one macro** did not build in this record's time
  (OpenRAM's pin connection is O(cells x pins); 512 words took 19 min,
  1,024 words 53 min, 4,096 was still connecting pins after 76 min); eight
  ``w512`` or four ``w1024`` banks under Catapult's ``-BLOCK_SIZE``
  decomposition is the next step, or OpenRAM with ``num_banks``.
* **The ``registers`` lowering's flop count** is 1.8x MiniTPU's for the
  same bits (66,263 against 33,602 sequential cells at 512 words, 4,335
  against 2,402 at 32; s.2.4): the unreset ``sc_signal`` array form carries more than
  the storage. Worth a look before any area comparison leans on it.
* **Power ring, escape routing, DRC/LVS** are off on every macro here
  (routers do not finish; no NCSU PDK). The Liberty area is the core.
* Per-cycle latency at II=2 is reported but not comparable with MiniTPU's
  3/2/1 (s.2.3); ``cmp_wa_sram.py --stretch`` is the only harness path for
  an II=2 unit.

5. ``mid`` on eight ``w512`` banks, at II=1
===========================================

.. note::

   **Follow-up, 2026-10-04 afternoon.** zhang-21, branch ``asic-memories-2``
   (worktree ``scratch/wt-asicmem2`` from ``origin/asic-memories`` at
   ``5aa08a5d``, its own ``mlir/build``). Scratch (not kept):
   ``scratch/asicmem2_*``. Tools as s.1 (OpenRAM ``b2b069ce`` unchanged,
   ``git status`` clean). s.6 is the second item of the same session; this
   section uses its result (the cross-port release that gives II=1).

5.1 The bank macro, pinned
--------------------------

``sram_2rw_64x512_freepdk45`` (s.2.1): OpenRAM ``b2b069ce``,
``openram/cfg_2rw_64x512.py`` (routers and DRC/LVS off), generated **once**
(``scratch/asicmem_openram/w512``, 1,151 s) and reused for every bank. LEF
``SIZE 410.295 BY 252.945`` = **103,782.07 um^2** a bank. sha256: ``.v``
``bc3ceb63fd91…``, ``.lib`` ``f207b9246b42…``, ``.lef`` ``d5cc4f848f79…``
(the three files are in ``openram/``; the LEF is new here), ``.gds``
``96c2b751fd07…`` (14.4 MB, not committed: regenerate from the config and
compare), the ``lc_shell`` ``.db`` md5 ``8b619e1feb7f…``. Full hashes in
``dev/toolchains.rst``. ``mid`` as eight of them is **830,256.6 um^2** of
macro; four ``w1024`` banks (s.1.3, 182,918.1 each) would be 731,672.4
(-12 %) and were not taken through Catapult.

5.2 Banks are declared
----------------------

``Sram.banked(n)`` (``allo/compose.py``) declares ``n`` copies of the macro,
each a contiguous block of ``rows`` words (the high address bits pick the
bank). Never inferred, as D-13/D-14 say of their properties: a memory with
more rows than one macro is refused until it declares its banks (``4096 rows
do not fit the macro ... (512 rows x 1 bank); declare the banks,
Sram.banked(8) -- banking is never inferred``), and a bank count with an
empty bank is refused too. The SystemC build adds ``directive set <rsc>
-BLOCK_SIZE <macro rows>`` after the ``MAP_TO_MODULE`` (``memory_directives``);
``memory.json`` carries ``banks``, ``banking`` and ``total_area_um2``.
Catapult's block split was taken over an explicit bank unit in ``compose``:
every bank keeps both macro ports, so each declared port is still one
physical port per bank and the port calendar does not change; Catapult builds
the decode and the read-data mux itself. ``-INTERLEAVE`` would put adjacent
words in different banks, which nothing in a two-port-per-bank VMEM needs.
``vpu_word_array_d12.SRAMS["mid"]`` is ``SRAMS["w512"]().banked(8).mapped(c=0,
d=1)`` (the placement of s.6.4, stated).

5.3 Catapult
------------

``sram_csyn.py sram <prj> --unit word_array --inst mid --ii {1,2}``, the
library built by ``build_memory_library`` (``RDWRRESOLUTION UNKNOWN``, as
s.2.1), nothing hand-patched. Catapult reports ``MEM-4`` eight times,
``mem:rsc(0..7)(0)`` each ``mapped to 'sram_2rw_64x512_freepdk45...' (size:
512 x 64)``, and ``concat_rtl.v`` instantiates eight macros.

.. list-table::
   :header-rows: 1
   :widths: 22 30 48

   * - ``mid``, 8 x ``w512``
     - schedule
     - note
   * - **II=1** (``catapult/mid_sram_ii1``)
     - ``port_c_0``/``port_d_0`` ii 1, 1 c-step; ``vmem_mem_0`` ii 1, 2 c-steps;
       44 s; area score 833,499.3
     - with the six cross-port ``ignore_memory_precedences`` the collision
       obligation implies (s.6.3), emitted by ``compose``
   * - II=2 (``catapult/mid_sram_ii2``)
     - ports ii 2; ``vmem_mem_0`` 3 c-steps; 40 s; area score 833,041.5
     - as ``w512`` in s.2.2; the first run, before s.6, scheduled the same with
       no release at all
   * - II=1 without the release
     - SCHD-30, chained feedback ``while:if:write_mem(mem:rsc(0)(0).@)`` ->
       ``while:if#1:write_mem(...)``
     - ``probes_s6.txt``, ``mid_sram_rbw_ii1`` (an RBW library, s.6.3, so the
       read/write chain of s.2.2 is gone and this one shows): the two ports'
       writes, the edge s.2.2's probes never released

5.4 Per cycle against ``vpu_word_array.sv``
--------------------------------------------

``cmp_wa_sram.py`` as s.2.3 (macro model ``_sim.v``, ``--cut-module``),
``inst=mid`` (the 4,096 x 64 b MiniTPU instance, ``MINITPU_NUM_LANES=1``).

* **II=1: defined 8,244/8,244** with one idle cycle after reset (``--lead
  1``): compute 4,129/4,129, DMA 4,115/4,115, 16,976 masked, both ports at
  output-row offset **+0**, i.e. MiniTPU's own rows; the step probes give
  read **3** and **2** cycles and same-port write->read visibility **1**, all
  MiniTPU's (``logs/cmp_mid_sram_ii1_lead1.txt``). The two cross-port
  held-address visibility probes give 0 against MiniTPU's 1: that probe
  writes a word on one port while the other port, disabled, still reads it
  (S6, an unconditional read) -- a same-cycle cross-port access of one word,
  the collision obligation's cycle; OpenRAM's model resolves it
  write-through. Masked in verdicts, as D-12 says.
* **Without the idle lead: 8,243/8,244** (``cmp_mid_sram_ii1.txt``). The one
  miss is a write on the first cycle after reset release
  (``wr-offsets-compute->compute`` cycle 0, read back at cycle 6, which
  returns Verilator's initial word): the pipelined server takes its first
  command one cycle after reset. Same at 32 and 512 words
  (``cmp_{small,w512}_sram_ii1*.txt``: 10,512/10,513 and 9,666/9,667, all
  defined with the lead). Inside MiniTPU this cycle carries no VMEM access on
  the compute side -- ``vpu_vmem_simd`` registers the request and resets
  ``compute_valid_q`` -- and the DMA side needs a descriptor from the
  sequencer first (not traced cycle by cycle here). A recorded deviation of
  the unit, not of the core.
* **II=2: defined 8,244/8,244** with ``--stretch 2``, the same offsets and
  probe values as ``w512`` in s.2.3 (``logs/cmp_mid_sram_ii2.txt``).
* The trace reaches every bank: accesses per bank 1,817 / 3,843 / 4,031 /
  3,980 / 1,766 / 504 / 595 / 1,715, and of the cycles where both ports are
  enabled 4,952 are in different banks and 1,869 in the same bank.

5.5 DC: eight macros as black boxes
-----------------------------------

As s.2.4 (``dc/dc_sram.tcl``, 3.33 ns, the bank macro's ``.db`` linked).

.. list-table::
   :header-rows: 1
   :widths: 36 15 13 17 19

   * - 3.33 ns, um^2
     - total
     - comb
     - seq (cells)
     - macro
   * - **``mid`` (4,096 x 64 b, 262,144 bit)**
     -
     -
     -
     -
   * - MiniTPU flop-mapped (``mtpu_word_array_flops.sv``)
     - see s.5.6
     -
     -
     -
   * - Allo ``sram``, 8 x ``w512``, **II=1** (``dc/a_mid_sram_ii1_3p33``)
     - **833,541.9**
     - 1,448.6
     - 1,836.7 (354)
     - **830,256.6**
   * - Allo ``sram``, 8 x ``w512``, II=2 (``dc/a_mid_sram_ii2_3p33``)
     - 833,333.1
     - 1,589.6
     - 1,486.9 (288)
     - 830,256.6
   * - *one ``w512`` macro, II=1, for the bank cost* (``dc/a_w512_sram_ii1_3p33``)
     - 105,911.4
     - 377.7
     - 1,751.6 (331)
     - 103,782.1

Slack 1.00 ns (II=1) and 0.95 ns (II=2) at 3.33 ns; the macro's own
Liberty timing is the analytical model (s.1.3). **The bank decode and mux
cost 1,156 um^2** at II=1 (comb +1,070.9, seq +85.1 against the one-macro
build): 0.14 % of the macros. All macro numbers are the core without power
ring, escape routing or DRC/LVS (s.1.3).

5.6 Still open in this section
------------------------------

* **MiniTPU's flop-mapped ``mid``**: no earlier record has it (``u2_word_array``
  and ``u2_d12_prototype`` measured ``narrow``; s.2.4 ``small`` and ``w512``).
  DC on ``mtpu_word_array_flops.sv`` with ``MINITPU_NUM_LANES=1`` was started
  here (``scratch/asicmem2_dc``, 4 h timeout) and was still in its first
  mapping pass when this section was committed; its number is recorded below
  when it ends. Linear scale from ``w512`` (262,797.6 um^2 x 8 = 2,102,381) is
  an estimate only.
* **The one-macro ``mid`` OpenRAM job** (s.1.3, started 11:07, 4 h timeout):
  still running at 13:30 (CPU 2 h). Its result is recorded below when it ends.

6. II=1 on MiniTPU's two read/write ports
=========================================

The question this section was given: does a macro with a read-only port
(1RW+1R, or 1R1W) reach II=1 where the 2RW macro did not (s.2.2), and if
MiniTPU's ports do not fit such a macro, which access pattern forces ``rw``?
Both are answered below; the answer to the question behind them -- II=1 for
MiniTPU's VMEM on an SRAM macro -- turned out not to need a different macro.

6.1 What MiniTPU's two ports do per cycle
-----------------------------------------

From the RTL at ``b3ba0a4d`` (``vpu_vmem_simd.sv``, ``vpu_dma_group.sv``) and
the U2 conflict table (``u2_phase0_2026-10-02.rst`` rows 9-11):

* **compute port**: ``en = compute_valid_q``, ``we = compute_req_q.op`` --
  a ``vld`` reads, a ``vst`` writes. **DMA port**: ``word_en = commit ||
  read_fetch``, ``word_we = commit`` -- a DMA-out word fetch reads, a DMA-in
  word commit writes. **Each port reads in some cycles and writes in others,
  one access a cycle**: both are ``rw`` in D-12's kinds. A same-port read and
  write in one cycle never happens (row 11: one MEM op per bundle; the sim
  model's read in a write cycle is masked, the XPM is ``no_change``).
* **Across the ports, in one cycle**, all four combinations are legal on
  different words: two reads (row 10, also on one word), a read and a write
  either way, and **two writes** (row 9 forbids only one word). The harness's
  ``mid`` trace has, of the cycles with both ports enabled, 2,437 read/read,
  1,652 + 1,607 write/read, and **1,125 write/write**.
* So a cycle may need **two writes, or two reads**. A two-port macro serves
  that only with both ports ``rw``; without ``rw`` ports it needs 2R+2W.
  **The pattern that forces ``rw`` is a ``vst`` and a DMA-in commit in the
  same cycle** (two writes) -- with, in other cycles, a ``vld`` beside a
  DMA-out fetch (two reads). A 1RW+1R macro has one writer: the DMA port's
  commits (or the compute port's stores) have no port in a write/write cycle.

6.2 Macros built (OpenRAM ``b2b069ce``, FreePDK45, routers and DRC/LVS off)
----------------------------------------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 30 14 18 38

   * - Macro (``openram/``)
     - Wall time
     - Liberty area, um^2
     - Note
   * - ``sram_1rw1r_64x32`` (1RW+1R)
     - 48 s
     - 19,990.8
     - vs 28,295.5 for the 2RW at 32 words
   * - ``sram_1r1w_64x32`` (1W+1R, ``num_rw_ports=0``)
     - 37 s
     - 18,562.5
     - OpenRAM lists the write port as port 0
   * - ``sram_1rw1r_64x512``
     - 1,069 s
     - 98,967.6
     - vs 103,782.1 for the 2RW bank macro (-4.6 %)
   * - ``sram_1r1w_64x512``
     - 826 s
     - 97,488.3
     - built; no Catapult run uses it
   * - 2R+2W at 32 words (``cfg_2r2w_64x32.py``)
     - 7 s, **refused**
     - --
     - ``pbitcell.py:1173``: "Two ports for bitcell_2port only" -- OpenRAM at
       the pin builds at most two ports, so no 4-port macro

6.3 Catapult at II=1: the probes, and the release that works
------------------------------------------------------------

All at 3.33 ns on 32 words unless named, ``sram_csyn.py`` with ``--kinds``
(the port-kind probes of ``vpu_word_array_d12``: a MiniTPU port narrowed to
read-only or write-only -- **not** MiniTPU's VMEM), ``--openram`` and, for
two rows, ``--rdwr``. Every probe's macro, resolution, hand-patch lines and
result are in ``catapult/probes_s6.txt``.

.. list-table::
   :header-rows: 1
   :widths: 14 11 13 34 28

   * - macro
     - kinds c,d
     - RDWR
     - released (``ignore_memory_precedences``)
     - II=1
   * - 2RW
     - rw,rw
     - UNKNOWN
     - nothing / cross read<->write (s.2.2)
     - SCHD-30
   * - 2RW
     - rw,rw
     - RBW
     - nothing
     - SCHD-30 on ``write_mem`` c -> ``write_mem`` d
   * - 2RW
     - rw,rw
     - RBW
     - write<->write only
     - SCHD-4 (3 port uses for 2 ports)
   * - 2RW
     - rw,rw
     - RBW
     - all memory precedences (same-port too)
     - scheduled; 10,512/10,513 (s.6.4)
   * - **2RW**
     - **rw,rw**
     - **RBW / UNKNOWN**
     - **the six cross-port pairs** (``while:if``/``else`` = c, ``if#1``/``else#1`` = d)
     - **scheduled**, ports 1 c-step, server 2
   * - 1RW+1R
     - rw,r
     - UNKNOWN / RBW
     - nothing
     - SCHD-30 / SCHD-4
   * - 1RW+1R
     - rw,r / r,w
     - UNKNOWN
     - the cross-port pairs (compose)
     - SCHD-4: 2 ``rwport`` needed, 1 available
   * - 1RW+1R
     - r,w
     - RBW
     - nothing; also with the Read port listed first in the MemGen spec; also at 512 words
     - SCHD-4
   * - 1RW+1R
     - rw,r (II=2)
     - RBW / UNKNOWN
     - --
     - scheduled at II=2 **with the R port tied off** (``.csb1(1'b1)``)
   * - 1R1W
     - r,w
     - UNKNOWN
     - nothing
     - SCHD-30, ``read_mem`` c -> ``write_mem`` d
   * - 1R1W
     - r,w
     - RBW, or UNKNOWN + the cross-port pairs
     - --
     - scheduled; 8,624/8,624 (s.6.4)

What this shows:

* **The obstacle was never the ``rw`` port.** It is the order Catapult keeps
  between the two ports' accesses of one memory: a write on c and a write on d
  in one iteration may hit one word, so Catapult chains them (and each read
  behind the other port's write), and the chain plus the macro's read delay
  is longer than one cycle. s.2.2's probes released the read/write edges but
  not the **write/write** one. D-12's collision obligation is exactly the
  statement that the two ports never touch one word in one cycle (MiniTPU
  issue #21); with it, no order between them is observable and Catapult may
  drop it. Same-port order is kept: releasing it too (the "all" row)
  scheduled as well, but nothing licenses it.
* **The release is now emitted by ``compose``**, not hand-patched:
  ``Architecture.cross_port_independence(m)`` gives the ``(from, to)``
  operation patterns of every cross-port pair with a write in it, only when
  ``collision`` is ``obligation`` or ``undefined``; ``memory_independence``
  writes them after ``go architect`` in ``run.tcl``; ``memory.json`` states
  ``cross_port_independent``. The patterns follow the server ``compose``
  generates (``_server_ops``: each ``rw`` port one ``if``/``else`` in order,
  an ``r`` port a plain read, each ``w`` port an ``if`` after them), so they
  name each port's operations, in every bank.
* **RBW is not needed** (the UNKNOWN rows schedule identically with the
  release), and it would be a false statement about this macro: OpenRAM's
  model resolves a same-cycle cross-port read of a written word write-through
  (the held-address probes, s.5.4). The library stays ``UNKNOWN``;
  ``Sram.rdwr`` records it, and a memory with ``collision="refuse"`` and a
  reader beside a different writer is refused on such a macro ("visible=1
  says it sees the old word; the macro leaves that cycle UNKNOWN").
* **Catapult does not use the Read port of a ReadWrite+Read macro**: every
  access is bound to the ``rwport``, the R port is tied off. A 1RW+1R macro
  through Catapult is a 1RW memory, which would drop a declared port; the
  SystemC build now refuses a macro that mixes ``rw`` with ``r``/``w`` ports.
  1R1W (no ``rw``) works.

6.4 Per cycle at II=1
---------------------

* **MiniTPU's VMEM on the 2RW macro, from ``compose`` alone (UNKNOWN
  library, the cross-port release)**: ``small`` 10,513/10,513, ``w512``
  9,667/9,667, ``mid`` (8 banks, s.5.4) 8,244/8,244, each with one idle
  cycle after reset and each **at MiniTPU's own rows** (offset +0 on both
  ports): read latency 3 and 2, same-port visibility 1, as the RTL. Without
  the idle cycle: 10,512/10,513, 9,666/9,667, 8,243/8,244 -- the first-cycle
  command (s.5.4). ``logs/cmp_{small,w512,mid}_sram_ii1*.txt``;
  ``catapult/{small,w512,mid}_sram_ii1``.
* **The 1R1W probe** (compute read-only on the R port, DMA write-only on the W
  port; ``cmp_wa_sram.py --restrict r,w`` narrows the trace for both RTLs):
  compute defined 8,624/8,624 at offset +0, read probe 3 = MiniTPU's
  (``logs/cmp_small_1r1w_rw_ii1.txt``, ``catapult/small_1r1w_rw_ii1``).
* The D-10 manifest: the port kernels report latency 1, ii 1 (LATENCY-MATCH
  against the measured kernel latency 1); ``vmem_mem_0`` reports latency 2
  against the measured 1, the kind of server-manifest mismatch s.2.3 already
  printed; not investigated here.

6.5 The answer, and the mapping
-------------------------------

* **MiniTPU's VMEM reaches II=1 on a 2RW SRAM macro**, one or eight banks,
  cycle-exact with ``vpu_word_array.sv`` at latencies 3/2/1, once the
  composition's collision obligation is handed to Catapult as cross-port
  independence. II=2 is no longer the ASIC answer; s.2.2-2.4's II=2 rows
  stand as measured. D-12's reverses-if ("a two-port VMEM cannot reach II=1")
  does not fire on the macro path either.
* **A 1RW+1R (or 1R1W) macro does not fit MiniTPU's VMEM**: both ports are
  ``rw``, and the cycle that forces it is a ``vst`` beside a DMA-in commit
  (two writes), with a ``vld`` beside a DMA-out fetch in others (two reads)
  -- s.6.1. ``compose`` refuses the placement naming the port ("port d (rw)
  has no port of the macro ... left to honour it ... a declared rw port needs
  an rw macro port"); a pinned placement on the wrong kind is refused the same
  way (``Sram.mapped(c=1)``: "port c (rw) cannot sit on port 1 ... a 'r'
  port"). Even where the kinds fit, Catapult would not drive the 1RW+1R
  macro's R port (s.6.3).
* **``memory.json``** of the shipped ``mid`` (``catapult/mid_sram_ii1``):
  ``macro`` ``sram_2rw_64x512_freepdk45`` x 8 (``banking`` "8 x 512 rows,
  contiguous blocks"), ``port_map`` ``{"c": "0 (rw)", "d": "1 (rw)"}``,
  ``cross_port_independent`` stated, ``obligation`` MiniTPU issue #21. Of the
  1R1W probe: ``{"c": "1 (r)", "d": "0 (w)"}``.
* Tests (``tests/dataflow/test_compose_sram.py``): ``test_sram_banked_mid``,
  ``test_sram_port_kinds_placement`` (every kind refusal and the mixed-macro
  refusal), ``test_sram_cross_port_independence`` (the six pairs, nothing
  same-port, nothing under ``refuse``, the ``run.tcl`` order). Regression
  as s.3 plus these: **132 passed** (``pytest tests/dataflow/test_systemc*.py tests/test_memory.py tests/dataflow/test_compose_memory_ports.py tests/dataflow/test_compose_sram.py``, 315 s); TinyTPU emission unchanged (``vhls``
  ``6bc774bc…`` 166,563 B, ``catapult`` ``ade1ab5d…`` 170,812 B);
  ``pylint`` on ``compose.py``/``catapult.py`` adds no message type (counts
  of the existing ``too-many-*`` move by one or two).
