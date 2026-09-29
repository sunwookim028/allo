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

.. _tinytpu-isa-spec:

####################################
TinyTPU-isa: The ISA
####################################

The instruction encoding TinyTPU-isa executes: bit layout, opcodes, the
derived properties each unit relies on, the instruction-memory header, the
memory map, the parameter set and the numerics. Every fact here is generated
from ``isa_spec.json`` and checked against the design, so this is the page to
skim for a field width or an opcode number. The machine that executes it is
:doc:`tinytpu_isa`; the measurements are on :doc:`tinytpu_isa_results`.

**Every fact on this page is generated from**
``examples/tinytpu/isa_spec.json``, which is the source of
truth for the instruction encoding, the opcodes, the instruction-memory header,
the memory map, the two program contracts, the build parameters and the
numerics. ``gen_isa.py --write`` regenerates the tables below and
``isa_encoding.py``; ``gen_isa.py --check`` fails if either is stale or if the
design or the reference model has drifted from the spec. The prose outside the
generated region is hand-written rationale.

.. BEGIN GENERATED: examples/tinytpu/gen_isa.py

.. Generated from examples/tinytpu/isa_spec.json.
   Edit the spec and run ``python gen_isa.py --write``; ``--check``
   fails if this region is stale.

Bit layout
~~~~~~~~~~

Most significant on the left. ``enc()`` builds word 0 and ``enc_agu()`` word 1; both are generated from the field table below, so neither picture can go stale.

.. code-block:: text

   word 0:
   +------------+------------+------------+------------+------------+------------+------------+
   |            |     nr     |     f3     |     f2     |     f1     |     f0     |     op     |
   |   63:62    |   61:54    |   53:42    |   41:30    |   29:18    |    17:6    |    5:0     |
   +------------+------------+------------+------------+------------+------------+------------+

   word 1: up to AGU_TERMS = 3 address terms of 19 bits:
   +------------+------------+------------+------------+
   |            |   term 2   |   term 1   |   term 0   |
   |   63:57    |   56:38    |   37:19    |    18:0    |
   +------------+------------+------------+------------+

   one term:    stride [18:7]   level [6:4]   target [3:0]
   resolves to: field[target] += iv[level] * stride, for each term with target != 0, in term order

Bits 62, 63 of word 0 and bits 63:57 of word 1 are unused.

.. list-table:: Who reads this spec, and how each one is held to it
   :header-rows: 1

   * - file
     - held
     - note
   * - ``examples/tinytpu/isa_encoding.py``
     - generated
     - The spec as Python. Regenerated and diffed byte for byte by gen_isa.py --check.
   * - ``docs/source/designs/tinytpu_isa.rst``
     - generated
     - The ISA tables, between the GENERATED markers. The prose around them is hand-written.
   * - ``examples/tinytpu/microarch_isa.py``
     - checked
     - The shipped instantiation: the build parameters, read from the environment, and the names the harness imports. Held to this file by parameter range and by parameter agreement under several configurations.
   * - ``examples/tinytpu/ip/isa.py``
     - checked
     - Opcodes, DMA flags, AGU targets and budgets, and the instruction layout named once (OP_LO .. NR_HI, AGU_*_BITS), which ``enc``/``enc_agu`` build from. Held by value, layout name by layout name, and by behaviour.
   * - ``examples/tinytpu/ip/assembler.py``
     - checked
     - ``expand``, the program validator and the imem header, decoding through ip/isa.py. Held by behaviour on every program the check generates, and AR_RAW_DIST by value.
   * - ``examples/tinytpu/ip/units``
     - checked
     - The hardware. Its bit slices of a 64-bit word are literal because a symbolic bound widens the extract to i32 (docs/source/designs/tinytpu_library.rst), so every slice is checked against the field table here, as is the mvout saturation.
   * - ``examples/tinytpu/isa_dsl.py``
     - checked
     - The program generator, through the encoder it shares with the design.
   * - ``examples/tinytpu/isa_ref.py``
     - built on the generated module
     - The reference model. Names operands by the ``name`` given below and never sees a bit position, an opcode number or a numeric width the design chose.
   * - ``allo/encoding.py``
     - checked
     - ``TINYTPU_ISA``: the compiler-side descriptor of what one instruction word can carry. It imports nothing from examples/, so its address_terms and loop_depth budgets are checked against ``agu.terms`` and ``loop_stack.depth`` rather than generated.

Instruction word
~~~~~~~~~~~~~~~~

An instruction is **2 64-bit words** (``IWORDS``). Word 0 carries the opcode and five fields; word 1 carries up to 3 address-generation terms.

.. list-table:: Instruction word 0
   :header-rows: 1

   * - field
     - bits
     - width
     - usable range
     - role
   * - ``op``
     - ``[0:6]``
     - 6
     - 0 .. 31
     - opcode
   * - ``f0``
     - ``[6:18]``
     - 12
     - 0 .. 2047
     - operand field, AGU target 1
   * - ``f1``
     - ``[18:30]``
     - 12
     - 0 .. 2047
     - operand field, AGU target 2
   * - ``f2``
     - ``[30:42]``
     - 12
     - 0 .. 2047
     - operand field, AGU target 3
   * - ``f3``
     - ``[42:54]``
     - 12
     - 0 .. 2047
     - operand field, AGU target 4
   * - ``nr``
     - ``[54:62]``
     - 8
     - 0 .. 127
     - row count, or loop trip count

Every field carries one more bit than its value range needs: an N-bit field safely holds ``0 .. 2^(N-1) - 1``, the spare-bit rule below.

.. list-table:: Instruction word 1: one AGU term (19 bits, 3 of them)
   :header-rows: 1

   * - subfield
     - bits
     - width
     - usable range
     - meaning
   * - ``target``
     - ``[0:4]``
     - 4
     - 0 .. 7
     - 4 bits for 0..4 under the spare-bit rule.
   * - ``level``
     - ``[4:7]``
     - 3
     - 0 .. 3
     - 3 bits for 0..LOOP_DEPTH-1.
   * - ``stride``
     - ``[7:19]``
     - 12
     - 0 .. 2047
     - 12 bits for 0..2047.

A term resolves to ``field[target] += iv[level] * stride, for each term with target != 0, in term order``. Targets: 0 = unused, 1 = ``f0``, 2 = ``f1``, 3 = ``f2``, 4 = ``f3``.

Opcodes
~~~~~~~

.. list-table:: Opcodes
   :header-rows: 1

   * - constant
     - value
     - name
     - operand fields
     - ``nr``
     - units
   * - ``OP_NOP``
     - 0
     - nop
     -
     - none
     - --
   * - ``OP_DMA_LD``
     - 1
     - dma_ld
     - | ``f0`` = mode: source matrix | destination memory
       | ``f1`` = dram_row0: first DRAM row of the source matrix
       | ``f2`` = col_block: packed-word column block within the row, 0 .. WPR-1
       | ``f3`` = dst_row0: first destination row, in spad or vr by the mode bit
     - nr DRAM rows
     - ``dma_ld``, ``spm``, ``vru``
   * - ``OP_DMA_ST``
     - 2
     - dma_st
     - Retired: results leave via mvout. ``check_program`` refuses it, so the number stays reserved rather than reusable.
     - none
     - --
   * - ``OP_VLD``
     - 3
     - vld
     - | ``f0`` = vr0: first destination vreg row
       | ``f1`` = spad0: first source scratchpad row
     - nr packed words
     - ``spm``, ``vru``
   * - ``OP_MM``
     - 4
     - mm
     - | ``f0`` = vr_a: first activation row in vr
       | ``f1`` = ar0: first accumulator row
       | ``f2`` = acc: 0 = overwrite ar, 1 = accumulate into ar
       | ``f3`` = spad_w: first of T weight rows in spad
     - nr activation rows (wavefront rows through the array)
     - ``spm``, ``array``, ``vru``, ``accu``
   * - ``OP_VADD``
     - 5
     - vadd
     - | ``f0`` = ar_d: first destination accumulator row
       | ``f1`` = ar_s1: first row of the left source
       | ``f2`` = ar_s2: first row of the right source
     - nr accumulator rows
     - ``accu``
   * - ``OP_VRELU``
     - 6
     - vrelu
     - | ``f0`` = ar_d: first destination accumulator row
       | ``f1`` = ar_s: first source accumulator row
     - nr accumulator rows
     - ``accu``
   * - ``OP_MVOUT``
     - 7
     - mvout
     - | ``f0`` = ar0: first accumulator row to retire
       | ``f1`` = dram_row0: first DRAM row of C
       | ``f2`` = col_block: packed-word column block of C, 0 .. WPR-1
     - nr accumulator rows
     - ``accu``, ``dma_st``
   * - ``OP_LOOP``
     - 8
     - loop
     -
     - nr is the trip count, minimum 1
     - ``sequencer``
   * - ``OP_ENDLOOP``
     - 9
     - endloop
     -
     - none
     - ``sequencer``
   * - ``OP_VADDRELU``
     - 10
     - vaddrelu
     - | ``f0`` = ar_d: first destination accumulator row
       | ``f1`` = ar_s1: first row of the left source
       | ``f2`` = ar_s2: first row of the right source
     - nr accumulator rows
     - ``accu``

The ``units`` column is **derived** from the actions below, not written beside each opcode: an opcode reaches whichever units its actions name. It used to be typed, and it was wrong twice -- ``mm`` did not name the array, and ``dma_ld`` named its destination in prose.

Instructions as compositions of Actions
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Every instruction is an ordered list of per-unit **effects**. Each effect names a unit, one of that unit's ports, and the element it touches. ``allo.actions`` holds the model and its legality rule; this table is what the spec declares, and the per-unit work counts in the header, the dispatch rewrites and the reads and writes below are queries over it rather than further declarations.

.. list-table:: Actions
   :header-rows: 1

   * - opcode
     - unit
     - port
     - kind
     - state
     - base
     - items
     - only if
   * - dma_ld
     - dma_ld
     - dram.read
     - read
     - ``A``
     - dram_row0
     - nr
     - (mode & 1) == 0
   * - dma_ld
     - dma_ld
     - dram.read
     - read
     - ``B``
     - dram_row0
     - nr
     - (mode & 1) == 1
   * - dma_ld
     - dma_ld
     - mux
     - compute
     - ``select``
     - --
     - nr
     - --
   * - dma_ld
     - dma_ld
     - dma2sp
     - emit
     - --
     - --
     - nr
     - (mode & 2) == 0
   * - dma_ld
     - dma_ld
     - dma2vr
     - emit
     - --
     - --
     - nr
     - (mode & 2) == 2
   * - dma_ld
     - spm
     - dma2sp
     - receive
     - --
     - --
     - nr
     - (mode & 2) == 0
   * - dma_ld
     - spm
     - spad.write
     - write
     - ``spad``
     - dst_row0
     - nr
     - (mode & 2) == 0
   * - dma_ld
     - vru
     - dma2vr
     - receive
     - --
     - --
     - nr
     - (mode & 2) == 2
   * - dma_ld
     - vru
     - vr.write
     - write
     - ``vr``
     - dst_row0
     - nr
     - (mode & 2) == 2
   * - vld
     - spm
     - spad.read
     - read
     - ``spad``
     - spad0
     - nr
     - --
   * - vld
     - spm
     - sp2vr
     - emit
     - --
     - --
     - nr
     - --
   * - vld
     - vru
     - sp2vr
     - receive
     - --
     - --
     - nr
     - --
   * - vld
     - vru
     - vr.write
     - write
     - ``vr``
     - vr0
     - nr
     - --
   * - mm
     - spm
     - spad.read
     - read
     - ``spad``
     - spad_w
     - T
     - --
   * - mm
     - spm
     - wcol
     - emit
     - --
     - --
     - T + 1
     - --
   * - mm
     - array
     - wcol
     - receive
     - --
     - --
     - T + 1
     - --
   * - mm
     - array
     - instructions
     - emit
     - --
     - --
     - 1
     - --
   * - mm
     - vru
     - vr.read
     - read
     - ``vr``
     - vr_a
     - nr
     - --
   * - mm
     - vru
     - acol
     - emit
     - --
     - --
     - nr
     - --
   * - mm
     - array
     - acol
     - receive
     - --
     - --
     - nr
     - --
   * - mm
     - array
     - mac
     - compute
     - ``matmul``
     - --
     - nr
     - --
   * - mm
     - array
     - cw
     - emit
     - --
     - --
     - nr
     - --
   * - mm
     - accu
     - cw
     - receive
     - --
     - --
     - nr
     - --
   * - mm
     - accu
     - ar.read
     - read
     - ``ar``
     - ar0
     - nr
     - acc == 1
   * - mm
     - accu
     - alu
     - compute
     - ``acc_add``
     - --
     - nr
     - --
   * - mm
     - accu
     - ar.write
     - write
     - ``ar``
     - ar0
     - nr
     - --
   * - vadd
     - accu
     - ar.read
     - read
     - ``ar``
     - ar_s1
     - nr
     - --
   * - vadd
     - accu
     - ar.read
     - read
     - ``ar``
     - ar_s2
     - nr
     - --
   * - vadd
     - accu
     - alu
     - compute
     - ``add``
     - --
     - nr
     - --
   * - vadd
     - accu
     - ar.write
     - write
     - ``ar``
     - ar_d
     - nr
     - --
   * - vrelu
     - accu
     - ar.read
     - read
     - ``ar``
     - ar_s
     - nr
     - --
   * - vrelu
     - accu
     - alu
     - compute
     - ``max0``
     - --
     - nr
     - --
   * - vrelu
     - accu
     - ar.write
     - write
     - ``ar``
     - ar_d
     - nr
     - --
   * - mvout
     - accu
     - ar.read
     - read
     - ``ar``
     - ar0
     - nr
     - --
   * - mvout
     - accu
     - alu
     - compute
     - ``to_operand``
     - --
     - nr
     - --
   * - mvout
     - accu
     - ac2sp
     - emit
     - --
     - --
     - nr
     - --
   * - mvout
     - dma_st
     - ac2sp
     - receive
     - --
     - --
     - nr
     - --
   * - mvout
     - dma_st
     - dram.write
     - write
     - ``C``
     - dram_row0
     - nr
     - --
   * - loop
     - sequencer
     - fetch
     - compute
     - ``push_loop``
     - --
     - 1
     - --
   * - endloop
     - sequencer
     - fetch
     - compute
     - ``pop_loop``
     - --
     - 1
     - --
   * - vaddrelu
     - accu
     - ar.read
     - read
     - ``ar``
     - ar_s1
     - nr
     - --
   * - vaddrelu
     - accu
     - ar.read
     - read
     - ``ar``
     - ar_s2
     - nr
     - --
   * - vaddrelu
     - accu
     - alu
     - compute
     - ``add``
     - --
     - nr
     - --
   * - vaddrelu
     - accu
     - alu
     - compute
     - ``max0``
     - --
     - nr
     - --
   * - vaddrelu
     - accu
     - ar.write
     - write
     - ``ar``
     - ar_d
     - nr
     - --

.. list-table:: Units: ports, step rate, elasticity
   :header-rows: 1

   * - unit
     - ports (items per cycle)
     - II
     - elastic
   * - sequencer
     - ``fetch``, ``dispatch`` x5
     - 1
     - yes
   * - dma_ld
     - ``dram.read``, ``mux``, ``dma2sp``, ``dma2vr``
     - 1
     - yes
   * - spm
     - ``spad.read``, ``spad.write``, ``dma2sp``, ``sp2vr``, ``wcol``
     - 1
     - yes
   * - vru
     - ``vr.read``, ``vr.write``, ``acol``, ``sp2vr``, ``dma2vr``
     - 1
     - yes
   * - array
     - ``wcol``, ``acol``, ``instructions``, ``mac``, ``cw``
     - 1
     - yes
   * - accu
     - ``cw``, ``ar.read``, ``ar.write``, ``alu`` x2, ``ac2sp``
     - 1
     - yes
   * - dma_st
     - ``ac2sp``, ``dram.write``
     - 1
     - yes

A unit's cost for an instruction is its busiest port's item count, so ``spm`` charging an ``mm`` ``T + 1`` iterations and ``accu`` charging a ``vadd`` two steps a row are consequences of these ports and not entries in a table. ``elastic`` says whether contention inside a unit costs a step or is illegal.

Derived properties
~~~~~~~~~~~~~~~~~~

Facts that **follow** from the tables above rather than being written in them. ``gen_isa.py --check`` recomputes each one from the spec and then confirms the design behaves that way, so they are checked rather than asserted -- every one of them was documented wrongly here until 2026-09-21, which is the argument for computing them.

``every_operand_field_is_an_agu_target``
   Every operand field is a legal AGU target, including one with a restricted value set such as mm's ``acc``. Nothing in the encoding or in the sequencer distinguishes them: the sequencer adds iv[level] * stride to whichever field the term names and writes the sum back, and it does not know which fields have restricted values.

   Derived from: ``agu.targets``, ``agu.semantics``.

   **Corrects:** Three documents in this project described ``acc`` as a static field that cannot be predicated on an induction variable. That is false. The real obstacle is the next property.

``agu_terms_are_additively_monotone``
   An AGU term contributes base + iv * stride with no predication, no saturation and no wrap, so the values a field takes over a loop of trip t are the arithmetic progression base, base + stride, ..., base + (t-1) * stride, and nothing bends it back inside a bound. A field with a restricted value set is therefore drivable from a loop for exactly as long as that progression stays inside the set -- ``isa_encoding.agu_legal_trip`` computes how long from ``legal_values`` alone.

   Derived from: ``agu.semantics``, ``opcodes[mm].operands[acc].legal_values``.

   At base 0 stride 1 the frontier is 2: an accumulate field driven by the reduce loop assembles for exactly two k-tiles and is rejected at the third.

``a_term_on_acc_costs_one_of_the_budget``
   A term on ``acc`` costs one of the AGU_TERMS terms an instruction carries. The accumulating mm of the shipped tiled GEMM already spends all of them -- activations walk the reduce loop, weights walk both the column loop and the reduce loop -- so driving ``acc`` from the reduce loop needs one more than the instruction word has, and the budget refuses the instruction before the monotonicity frontier above is ever reached. Widening the address generator and relieving ``acc`` are therefore COMPLEMENTS, not alternatives: at AGU_TERMS = 3 the second constraint is unreachable for any program with a column loop.

   Derived from: ``agu.terms``, ``opcodes[mm].operands``.

   ``terms_used_by_shipped_acc_mm`` is read out of the shipped program's own AGU word, not restated here.

``the_maxdim_ceilings_are_the_encoding's``
   What bounds MAXDIM is the ENCODING, not the datapath, and both bounds follow from numbers already in this file. They answer different questions -- one asks what the operand layout can ADDRESS, the other what a CUBIC GEMM's header count can promise -- so neither is a correction of the other, and the binding one depends on the program. ``isa_encoding.maxdim_ceiling`` computes each over multiples of T, and the design refuses a cubic GEMM at exactly the value the cubic-header ceiling gives.

   Derived from: ``maxdim_ceilings``, ``encoding_rule.usable_max``, ``imem.count_slice_width``, ``imem.entries[accu_iterations]``.

   The numbers are computed, not typed: at T=4 the addressing ceiling is 88 and the cubic-header ceiling is 76, so the cubic one binds; at T=8 they are 128 and 120, so the cubic one binds there too. An earlier revision of this file said 90 for the first, copied from the design's comment, which solved the inequality over the reals instead of over multiples of T.

``program_limits_predict_what_assembles``
   The limits in ``program_limits`` decide exactly which GEMM shapes this build can run, and ``isa_encoding.gemm_limits`` computes which limit refuses a shape and by how much. For the shipped layout, a shape assembles if and only if every limit but the ``split_operand_load`` alternative fits.

   Derived from: ``program_limits.limits``, ``derived_constants[MAXROWS]``, ``imem.count_usable_max``.

   The ``split_operand_load`` row is the one that makes the answer actionable rather than final: at T=8 MAXDIM=128 the shipped layout is over by 1 on ``max(M, K)`` for 32x128x128 and 64x128x128, but splitting the B load removes K from the row count and both assemble and compute A@B exactly -- measured on the design, not argued. 128x128x128 is over by 1 even then, and its accu header count is over by 2049, so it needs both a wider nr and a wider header slice.

Per-unit rewrites
~~~~~~~~~~~~~~~~~

The sequencer hands two units a rewritten copy of the word, so each unit's flat row loop reads its own work count out of ``nr``:

.. list-table::
   :header-rows: 1

   * - opcode
     - unit
     - rewritten to
     - why
   * - ``mm``
     - ``spm``
     - ``nr`` = T + 1
     - One header word plus T weight rows down wcol; f1 carries the array's row count into the header word.
   * - ``vadd``
     - ``accu``
     - ``nr`` = 2 * nr
     - accu takes two iterations per vadd row: first source on the even one, second source and the write on the odd one. 2 * MAXROWS fits the 8-bit field.
   * - ``vaddrelu``
     - ``accu``
     - ``nr`` = 2 * nr
     -

Which units need a rewrite, and to what, is **derived**: it is every unit the sequencer dispatches to whose own work count differs from the instruction's row count. Nothing states it, so adding an instruction adds no entry here.

Instruction memory header
~~~~~~~~~~~~~~~~~~~~~~~~~

``imem[0:NHDR]`` (``NHDR = 8``) is a header of per-unit **work** counts; instructions follow, two words each. Each count is read back through a 16-bit slice, so it stops at 32767.

.. list-table:: Header words
   :header-rows: 1

   * - word
     - bits
     - name
     - count
     - consumer
   * - ``imem[0]``
     - ``[0:16]``
     - ``n_instr``
     - static instruction count
     - ``sequencer``
   * - ``imem[1]``
     - ``[0:16]``
     - ``dma_ld_rows``
     - the steps ``dma_ld`` spends
     - ``dma_ld``
   * - ``imem[2]``
     - ``[0:16]``
     - ``spm_rows``
     - the steps ``spm`` spends
     - ``spm``
   * - ``imem[3]``
     - ``[0:16]``
     - ``vru_words``
     - the steps ``vru`` spends
     - ``vru``
   * - ``imem[4]``
     - ``[0:16]``
     - ``mm_count``
     - items on ``array``'s ``instructions`` port
     - the array, via spm's header word
   * - ``imem[4]``
     - ``[16:32]``
     - ``mm_rows``
     - items on ``array``'s ``mac`` port
     - the array, via spm's header word
   * - ``imem[5]``
     - ``[0:16]``
     - ``accu_iterations``
     - the steps ``accu`` spends
     - ``accu``
   * - ``imem[6]``
     - ``[0:16]``
     - ``dma_st_rows``
     - the steps ``dma_st`` spends
     - ``dma_st``
   * - ``imem[7]``
     - ``[0:16]``
     - ``a_span``
     - the row span of ``A`` every ``dma_ld`` reads
     - ``dma_ld``
   * - ``imem[7]``
     - ``[16:32]``
     - ``b_span``
     - the row span of ``B`` every ``dma_ld`` reads
     - ``dma_ld``

Every count but the static one is **the work a unit does**, summed over the issues the program makes, and it is computed from the actions above rather than written here. The per-opcode costs that follow are therefore derived too:

.. list-table:: Per-unit cost, derived from the ports
   :header-rows: 1

   * - opcode
     - unit
     - steps
   * - dma_ld
     - dma_ld
     - ``1 x nr``
   * - dma_ld
     - spm
     - ``1 x nr``
   * - vld
     - spm
     - ``1 x nr``
   * - vld
     - vru
     - ``1 x nr``
   * - mm
     - spm
     - ``T + 1`` per issue
   * - mm
     - array
     - 13 at nr=8
   * - mm
     - vru
     - ``1 x nr``
   * - mm
     - accu
     - ``1 x nr``
   * - vadd
     - accu
     - ``2 x nr``
   * - vrelu
     - accu
     - ``1 x nr``
   * - mvout
     - accu
     - ``1 x nr``
   * - mvout
     - dma_st
     - ``1 x nr``
   * - loop
     - sequencer
     - 1 at nr=8
   * - endloop
     - sequencer
     - 1 at nr=8
   * - vaddrelu
     - accu
     - ``2 x nr``

Memory map
~~~~~~~~~~

.. list-table:: Memories
   :header-rows: 1

   * - memory
     - owner
     - depth
     - row width
     - written by
     - read by
     - cleared at start
   * - ``spad``
     - ``spm``
     - ``SPAD_ROWS``
     - ``T * 8`` bits
     - ``dma_ld`` [(mode & 2) == 0]
     - ``vld`` (the row to copy), ``mm`` (weights)
     - **no**
   * - ``vr``
     - ``vru``
     - ``NVR``
     - ``T * 8`` bits
     - ``dma_ld`` [(mode & 2) == 2], ``vld``
     - ``mm`` (activations)
     - **no**
   * - ``ar``
     - ``accu``
     - ``NAR``
     - ``T * 32`` bits
     - ``mm``, ``vadd``, ``vrelu``, ``vaddrelu``
     - ``mm`` (the accumulate base) [acc == 1], ``vadd`` (a source), ``vrelu`` (a source), ``mvout`` (the value to retire), ``vaddrelu`` (a source)
     - **no**
   * - ``imem``
     - ``sequencer``
     - ``IMEM_SIZE``
     - ``64`` bits
     - the host
     - ``sequencer``
     - **no**
   * - ``A``
     - ``dma_ld``
     - ``MAXDIM * MAXDIM``
     - ``8`` bits
     - the host
     - ``dma_ld`` (the source rows) [(mode & 1) == 0]
     - **no**
   * - ``B``
     - ``dma_ld``
     - ``MAXDIM * MAXDIM``
     - ``8`` bits
     - the host
     - ``dma_ld`` (the source rows) [(mode & 1) == 1]
     - **no**
   * - ``C``
     - ``dma_st``
     - ``MAXDIM * MAXDIM``
     - ``8`` bits
     - ``mvout``
     - the host
     - **no**

The ``written by`` and ``read by`` columns are **derived** from the actions. They used to be two lists beside each memory that nothing computed from and nothing checked.

No on-chip memory is cleared by the hardware, so every read of one is the program's obligation; see the contracts below.

Contracts
~~~~~~~~~

**Write before read.**

* Every ar row read by an accumulating mm, by either source of a vadd, by a vrelu source or by an mvout must have been written earlier in the SAME program, by an overwriting mm, a vadd or a vrelu.
* Every vr row an mm reads as activations, and every spad row it reads as weights, must hold data a dma_ld put there -- directly, or into spad and then through a vld.
* vld is a pure copy and may copy an unwritten spad row; the copy is then unwritten too, and consuming it in an mm is an error.
* nr >= 1 on every data op: a unit fetches an instruction whenever its row counter runs out, so a zero-row instruction is fetched as if it had one row. It desynchronises the unit; it is not a no-op.
* Every resolved field must be within 0 .. 2047, the range the encoding rule admits.

Enforced by microarch_isa.check_program, which microarch_isa.assemble calls, so a violating program cannot be assembled.

**The accumulator read-after-write distance.** A read of an ar row must come at least AR_RAW_DIST accu iterations after the write it depends on. ``AR_RAW_DIST = 4`` accu iterations; cost per opcode: ``mm`` 1 per row, ``vrelu`` 1 per row, ``mvout`` 1 per row, ``vadd`` 2 per row: first source on the even iteration, second source and the write on the odd one. Enforced by microarch_isa.check_program. Exercised at its edge by isa_dsl.ar_distance_program(AR_RAW_DIST), run by TPU_TB=stress cosim on every build.

Parameters
~~~~~~~~~~

.. list-table:: Build parameters
   :header-rows: 1

   * - constant
     - environment variable
     - default
     - legal range
     - role
   * - ``T``
     - ``TPU_T``
     - 4
     - 4 .. unbounded
     - SIMD width, and the array dimension: the array is T*T processing elements
   * - ``MAXDIM``
     - ``TPU_MAXDIM``
     - 64
     - 4 .. unbounded
     - largest M, K, N supported by one build
   * - ``SPAD_ROWS``
     - ``TPU_SPAD``
     - ``max(TEST_WINDOW, OPERAND_ROWS)``
     - 1 .. 2048
     - scratchpad depth in packed words
   * - ``NVR``
     - ``TPU_NVR``
     - ``max(TEST_WINDOW, OPERAND_ROWS)``
     - 1 .. 2048
     - operand vector registers
   * - ``NAR``
     - ``TPU_NAR``
     - ``max(128, TEST_WINDOW, 2 * MAXDIM + 8)``
     - 1 .. 2048
     - accumulator vector registers
   * - ``IMEM_SIZE``
     - ``TPU_IMEM``
     - ``NHDR + IWORDS * 24``
     - 8 .. unbounded
     - instruction memory depth in 64-bit words
   * - ``QD``
     - ``TPU_QD``
     - 16
     - 2 .. unbounded
     - stream depth on every point-to-point channel
   * - ``DMA_WORDS``
     - ``TPU_DMA_WORDS``
     - ``min(WPR, max(1, BUS_BYTES // (VW // 8))) if os.environ.get("TPU_DMA_WIDEN") == "1" else 1``
     - 1 .. unbounded
     - packed words the operand burst moves per loop iteration

.. list-table:: Derived constants
   :header-rows: 1

   * - constant
     - value
     - role
   * - ``VW``
     - ``T * 8``
     - packed operand word: T operand lanes
   * - ``AW``
     - ``T * 32``
     - packed accumulator word: T accumulator lanes
   * - ``WPR``
     - ``MAXDIM // T``
     - packed words per DRAM row; the legal range of col_block
   * - ``OPERAND_ROWS``
     - ``(MAXDIM // T) * MAXDIM``
     - the highest operand row the shipped GEMM's layout names; what the operand memories are sized from
   * - ``TEST_WINDOW``
     - ``64``
     - the fixed row window the stress harness's fuzz programs address, independently of MAXDIM; a floor under the operand memories
   * - ``BUS_BYTES``
     - ``64``
     - the m_axi beat the build aligns to (align_value 64)
   * - ``MAXROWS``
     - ``127``
     - the largest nr the 8-bit field admits under the encoding rule

Cross-parameter constraints, asserted by ``isa_encoding.check_parameters()``:

* ``T >= 4`` -- a packed operand word must hold the two 16-bit counts spm sends down wcol
* ``MAXDIM % T == 0`` -- a DRAM row must be a whole number of packed words
* ``OPERAND_ROWS <= 2048`` -- an address field carries 11 usable bits, and the layout numbers its rows 0 .. OPERAND_ROWS-1, so the highest address must be <= 2047
* ``AR_RAW_DIST <= T`` -- a T-row GEMM must satisfy the accumulator distance contract
* ``IMEM_SIZE % 8 == 0`` -- the program prefetch moves 8 words per iteration
* ``IMEM_SIZE >= NHDR + IWORDS`` -- imem must hold the header and at least one instruction
* ``SPAD_ROWS <= 2048 and NVR <= 2048 and NAR <= 2048`` -- a 12-bit address field carries 0..2047 under the encoding rule, and a memory of depth D is addressed 0..D-1, so D <= 2048 -- the same count-versus-address distinction as the addressing ceiling
* ``SPAD_ROWS >= OPERAND_ROWS and NVR >= OPERAND_ROWS`` -- an operand memory smaller than the layout addresses assembles and gives wrong answers
* ``NAR >= 2 * MAXDIM + 2`` -- AR_C is MAXDIM rows and AR_P another MAXDIM from MAXDIM+1
* ``DMA_WORDS >= 1`` -- the operand burst moves at least one packed word per iteration

What bounds ``MAXDIM``
~~~~~~~~~~~~~~~~~~~~~~

Two ceilings, both in the **encoding** rather than the datapath. They answer different questions, so neither is a correction of the other, and which one binds depends on the program and on ``T``. The values below are **computed** by ``isa_encoding.maxdim_ceiling``, over multiples of ``T``: solving either inequality over the reals gives a number no build can use.

.. list-table:: ``MAXDIM`` ceilings
   :header-rows: 1

   * - ceiling
     - at T=4
     - at T=8
     - predicate
     - the question it answers
     - how the design refuses past it
   * - ``addressing``
     - 88
     - 128
     - ``(MAXDIM // T) * MAXDIM <= 2048``
     - How large a MAXDIM can the shipped GEMM's operand layout ADDRESS? A property of the layout and the address field, independent of the shape being run.
     - check_program: "AGU-resolved f3=... is outside the 0..2047 range", or the import-time assert on OPERAND_ROWS
   * - ``cubic_header``
     - 76
     - 120
     - ``MAXDIM ** 3 // T ** 2 + MAXDIM ** 2 // T <= 32767``
     - How large a MAXDIM can a CUBIC GEMM's header count promise accu? A property of the WORKLOAD as well as the encoding -- a non-cubic shape gives a different count, so this ceiling moves with the program.
     - assemble: "header count ... does not fit 15 bits"

The binding ceiling for a build is the smaller of the two, and which one binds depends on the program. The design ships at MAXDIM=64, inside both.

Numerics
~~~~~~~~

Active configuration: **int8**. A configuration states what happens to *values*, not only how wide they are, so that a format whose arithmetic is inexact can be added without restructuring anything above.

.. list-table:: ``int8``
   :header-rows: 1

   * - stage
     - operation
     - range, or exactness and what replaces it
     - where / note
   * - operand
     - integer, signed, 8 bits
     - -128 .. 127
     - A, B, C at the region boundary; spad and vr lanes
   * - accumulator
     - integer, signed, 32 bits
     - -2147483648 .. 2147483647
     - ar lanes, and the partial sum travelling south through the array
   * - multiply
     - operand x operand -> 16 bits
     - exact
     - int8 x int8 is bounded by 128*128 = 16384, so a 16-bit product is exact and the multiplier stays narrow.
   * - accumulate
     - integer addition
     - rounding: none; overflow: wraparound, two's complement, at 32 bits
     - order: strictly sequential in ascending contraction index, south down the PE column, then the mm's acc term
   * - output
     - accumulator -> operand, saturate
     - to -128 .. 127, rounding none
     - mvout, in accu, before the word leaves for dma_st
   * - ``vadd``
     - integer addition
     - overflow: wraparound, two's complement, at 32 bits
     - in the accumulator format
   * - ``vrelu``
     - max(x, 0)
     - exact
     - in the accumulator format

.. END GENERATED



Rationale
~~~~~~~~~

**Control flow.** ``LOOP``/``ENDLOOP`` drive a ``LOOP_DEPTH = 4`` hardware loop
stack, as MiniTPU's (:doc:`minitpu`). The sequencer's loop is do-while, so a
trip count of 0 would run **once**; the generator refuses it (below).

**Address generation.** Each AGU term is ``(target, level, stride)`` and resolves
to ``field[target] += iv[level] * stride``, so an address can be relative to any
enclosing loop's induction variable. Terms name their target rather than being
fixed one-per-field, because a single field often needs two: the ``mm`` inside
the k loop names its weights at an offset by both the n tile and the k tile,
``B_SP + nb*MAXDIM + kb*T``, and a one-term-per-field encoding cannot say it.
Without this a loop body would reissue identical addresses every iteration and
simply redo the same work.

.. _tinytpu-isa-spare-bit:

The spare-bit rule
~~~~~~~~~~~~~~~~~~

Every field carries one more bit than its value range needs, because
**an N-bit ISA field safely carries 0 .. 2^(N-1) - 1**. So ``nr`` is 8 bits
for ``MAXROWS = 127``, the address fields are 12 bits for a 2047 maximum, AGU
targets are 4 bits for 0..4, levels 3 bits for 0..3, strides 12 bits for
0..2047, and ``enc()`` asserts ``0 <= v < 2^(w-1)`` for every field.

The rule was earned from an emitter bug that extracted a bit-slice into a
**signed** ``ap_int<N>``, so a field whose top bit was set read back negative
and the loop it bounded ran zero times. The general lesson it left is
**keep cosim in the loop rather than running it once at the end**: the Allo
dataflow simulator did not model the narrowing, so the fault was a genuine
simulator/RTL divergence and was invisible to every functional check that had
been passing. The bug itself is in
:doc:`tinytpu_isa_results`.

.. note::

   The emitter has since been fixed: the fork's ``3de74846`` ("hls: emit bit
   slices as unsigned", 2026-09-18) and upstream's ``094ab413`` (PR #612,
   "Preserve unsigned bit-slice types during HLS codegen") fix the same bug;
   ``main`` took upstream's implementation in merge ``dc6b8fa6`` (2026-09-19).
   The design's spare-bit encoding predates the fix and is unchanged. See
   :ref:`limitation-12`.

The ``nr`` width is also load-bearing for synthesis, not only for correctness;
see :ref:`tinytpu-isa-csynth-bound`.

Header and program memory
~~~~~~~~~~~~~~~~~~~~~~~~~

The header words are tabulated above. ``imem[0]`` bounds the sequencer's fetch;
every other count is dynamic, from ``expand()``, which runs the program's
control flow at assembly time and resolves the AGU exactly as the sequencer
does. With a hardware loop the static and dynamic counts differ, and **a unit
promised more work than it receives does not produce a wrong answer, it
hangs** -- the one place where the assembler and the microarchitecture are
coupled.

``expand()`` yields one ``(opcode, row count, f0, f1, f2, f3)`` tuple per
dynamic issue, and the resolved address fields travel alongside the row count
for the same reason: ``dma_ld`` now bursts, so it needs the DRAM row span
before its first instruction arrives (``imem[7]``, above), and ``f1`` is an
AGU target in the general case, not a constant readable off the static
encoding. Mirroring the sequencer's control flow here is also what makes
``bench_isa.py``'s loop-vs-flat equivalence check strict: the two program
forms must agree on every resolved address, not only on the opcode and
row-count stream. ``assemble()`` runs every program through ``check_program()``
first; ``check=False`` exists only so a test can put a known-bad program on
the machine and watch it fail -- nothing that ships passes it.

``IMEM_SIZE`` is ``NHDR + IWORDS * 24`` (the longest program shipped, plus
headroom). The program is pulled on-chip by one burst of ``IMEM_SIZE`` words
before anything runs, so imem size is startup time whether the program uses it
or not. Measured when the prefetch moved one word per cycle: moving to the
2-word instruction format cost exactly +68 cycles at all five shapes, exactly
the 68 extra words it added -- the loop logic itself cost nothing. Since
``e24e433b`` the prefetch moves **8 words per cycle** (gmem0 is 512 bits wide;
``ib`` is cyclically partitioned by 8), so the 56 words cost 7 cycles, not 56
-- worth 52 cycles at every shape (``v_imem8``). With control flow the program
is O(nesting), not O(tiles): the looped GEMM is at most 14 instructions (with
``relu``) at every shape, where the unrolled one reaches 32.

.. _tinytpu-isa-programs:

Programs, and the loop-nest generator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Three program forms are kept, and ``bench_isa.py`` checks them against each
other before building anything:

* ``isa_dsl.gemm_program`` -- the shipped program, from the loop-nest generator:
  A ``dma_ld``'d straight into the vregs, B into the scratchpad, and every
  ``mm`` naming its T weight rows there.
* ``microarch_isa.gemm_program_handwritten`` -- the hand-emitted reference. The
  first k-tile is peeled out of the k loop deliberately: it carries ``f2=0``
  (overwrite the accumulator) where the looped tiles carry ``f2=1``
  (accumulate), and peeling is how you say that without a predicate on the
  induction variable.
* ``microarch_isa.gemm_program_flat`` -- the fully unrolled differential
  reference, absolute addresses and no control flow.
* ``vadd_program`` computes ``relu(2 * (A @ B))`` on the first output tile, so
  ``vadd``/``vrelu`` stay exercised now that the GEMM inner loop no longer needs
  them.
* ``isa_dsl.vector_program`` and ``isa_dsl.ar_distance_program`` are test
  programs: the first varies every field GEMM holds constant (both ``dma_ld``
  destinations, ``vld``, distinct accumulator regions), the second sits exactly
  on the accumulator distance contract (:ref:`tinytpu-isa-dependence`).

``isa_dsl.py`` is **a code generator, not a compiler**. It chooses nothing:

.. list-table::
   :widths: 25 75

   * - the programmer writes
     - the tiling, the loop order, the layout, which k-tile is peeled, and which
       loop an address walks
   * - the generator derives
     - every ``agu_level``, every AGU term, every loop open/close pair, and the
       encoded instruction word
   * - the generator refuses
     - a nest deeper than the hardware loop stack, an induction variable used
       outside its own loop, and more address terms than one instruction can
       carry
   * - nobody decides
     - what the best tiling is -- there is no search here

It ports MiniTPU's mechanism (``board_package/dsl.py:126-141``, its ``loop()``):
**nesting depth IS the AGU level**, and the only handle on a level is the
induction variable the ``with`` yields:

.. code-block:: python

   with k.loop(Nt, "n") as nb:                       # level 0, derived
       k.mm(A_VR, AR_C, Ref(B_SP).at(nb, MAXDIM), rows=M, acc=False)  # peeled
       with k.loop(Kt - 1, "k") as kb:               # level 1, derived
           k.mm(Ref(A_VR + MAXDIM).at(kb, MAXDIM), AR_C,
                Ref(B_SP + T).at(nb, MAXDIM).at(kb, T), rows=M, acc=True)

Swap two ``with`` statements and every address term follows, because the
induction variables carry their levels with them. It is generalised to this
AGU rather than copied: MiniTPU's AGU is one term on one field as a
power-of-two shift, this one is three ``(target, level, stride)`` terms with
arbitrary 11-bit strides, any number of which may land on the same field --
``B_SP + nb*MAXDIM + kb*T`` is two terms on ``f3`` and their encoding cannot say
it. So ``Ref`` is a base plus an ordered list of ``(iv, stride)`` terms and any
field may be one.

**It emits the identical instruction stream, and that is the whole test.**
``isa_dsl.assert_matches_handwritten`` compares both 64-bit words of every
instruction against ``gemm_program_handwritten`` at all five shapes and both
relu settings; ``bench_isa.py`` runs it before it builds anything. Bit-identical
means cosim cannot move, so it was not re-run.

**What it bought.** Three errors that were previously expressible are not: a
level that disagrees with the nest, an induction variable used after its loop
closed (the sequencer would resolve it against a stale ``iv_now[level]``), and a
``loop`` whose ``endloop`` was forgotten. Two more are now caught at the point
of writing rather than at ``enc_agu``: a nest deeper than ``LOOP_DEPTH``, named
("the nest is n > k > m > j > i"), and a trip count of 0, which the do-while
sequencer would run **once**.

**What it cost.** It is ~120 lines of machinery to remove four typed integers
from a 40-line program, and at this size the hand-written form was not actually
hard to keep right -- the trade only pays once there is more than one program.
It also trades one hand-maintained correspondence for another: the instruction
methods (``vld(vr, spad, rows)``) restate the field layout that the opcode table
comments give. The new correspondence is stated once per opcode instead of once
per instruction and the bit-identity assertion checks it, which is why it is the
better of the two, but it is not free. And the genuinely subtle part of the
program -- the peeled first k-tile, and the ``B_SP + T`` base that encodes "kb
starts at 1" -- is exactly as subtle as it was.

Programs are written against this encoder (by hand, or through ``isa_dsl.py``)
because the compiler backend that lowers a TOSA matmul into these instructions
lives only on the ``chia-codesign`` branch; see :doc:`/extensions/act`.
