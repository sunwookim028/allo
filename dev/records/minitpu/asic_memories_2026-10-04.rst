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
   * - 4,096 x 64 b (``mid``), both routers off
     - see s.2
     - ``bitcell_array.create_instances`` -> ``connect_pin`` -> ``__eq__``
       (a linear pin search per connection over 262,144 cells) is where the
       first half hour went; the 512 x 64 and 1,024 x 64 fallbacks were
       started beside it.

The ``.v`` is a behavioural model (inputs registered at ``posedge``, read
and write at ``negedge``, ``#(DELAY)`` on the read data, ``$display``
traffic); ``memgen_spec.py`` derives a Verilator copy from it (s.2).
