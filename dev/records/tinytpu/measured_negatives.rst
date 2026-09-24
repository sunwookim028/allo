Measured negatives: changes that were tried and moved nothing
=============================================================

:Date: 2026-09-24 (register opened; each entry carries its own date)
:Scope: TinyTPU-isa, the Vitis HLS flow, and the RTL export

A negative result is the cheapest thing in this project to lose and the most
expensive to re-derive. Every entry below was **measured**, not reasoned, and
each one exists because somebody was about to make the change it refuses.

These lived as long comment blocks in the source files. They are here instead,
and each site carries a one-line pointer back, because a paragraph three lines
above the code did not stop the change being re-proposed: a language model
editing ``ip/units/dma_load.py`` re-derived entry 1 with the answer in its own
context window. Proximity is not findability.

Do not delete an entry. If a later measurement overturns one, add the new
measurement under it and say so; that is a second result, not an erratum.

.. contents::
   :local:
   :depth: 1


1. Merging the two operand bursts into one loop
-----------------------------------------------

:Site: ``examples/tinytpu/ip/units/dma_load.py`` (the ``a_onchip`` /
   ``b_onchip`` staging loops)
:Verdict: **moves nothing**

``dma_ld`` runs one variable-length burst per matrix, covering exactly the
DRAM rows the program names. Merging the two into a single loop bounded by
``max(a_rows, b_rows)`` halves the burst time on paper. It does not change the
cycle count, at any of the published five: the bursts are already hidden behind
the sequencer's instruction prefetch, so halving them shortens a phase that is
not on the critical path.


2. "Strided access costs 4x", the justification for the on-chip mirror
-----------------------------------------------------------------------

:Site: ``examples/tinytpu/ip/units/dma_load.py``
:Verdict: **the original justification is STALE**; the mirror survives on a
   different one
:Measured: 2026-09-24, on ``dma_ld_0_1_Pipeline_VITIS_LOOP_645_1`` of a T=4
   MAXDIM=16 QD=16 build

The on-chip mirror was introduced because a per-row ``m_axi`` read was strided
and Vitis turned each row into a four-beat AXI transaction. That measurement
predates ``align_value(64)`` and ``-m_axi_max_widen_bitwidth 512`` being in the
build. With both of them:

* ``gmem1`` and ``gmem2`` are 32 bits wide -- exactly one packed word at T=4 --
  so a row read is **one beat**;
* Vitis emits **no** ``[HLS 214-115]`` note for the operand ports at all (the
  only one left is ``gmem0``, the 512-bit instruction port);
* a flat per-row ``m_axi`` loop closes at ``Final II = 1, Depth = 17``.

The same result is reached independently in
``docs/source/backends/vitis.rst`` ("two burst loops from II=4 to II=1").

**So the mirror must be defended on what it still buys** -- amortising the
re-read of a DRAM row that the program names more than once -- and never again
on burst shape. ``big_shapes_settlement.rst`` in this directory has the cycle
cost of removing it outright: **+6 / 0 / -2 / -6 / +10** on the published five.


3. ``wrap_io=True``: the trade that was not a trade
----------------------------------------------------

:Site: ``examples/tinytpu/cosim.py`` (``s.build(..., wrap_io=...)``)
:Verdict: ``wrap_io=False`` is **strictly better**; the earlier contrary
   measurement was an artefact

``wrap_io=True`` makes Allo hoist every ``m_axi`` argument into a local buffer
before the region starts, with ``wrap_data_movement``'s extent taken from the
STATIC type. At MAXDIM=16 that is ``imem 56 + A 256 + B 256 + C 256 = 824``
words copied whether the program touches them or not -- **907 of the 1586
cycles at 16x16x16, and 90 % of them at 4x4x4**.

``wrap_io=False`` lets each unit address ``m_axi`` itself and burst exactly
what its program names: **fixed cost 557, marginal 20.1 cycles/instruction**,
and faster at all five shapes.

An earlier measurement of ``wrap_io=False`` -- **fixed 481, marginal 39.8** --
is what once made this look like a genuine trade. It was taken with the
strided operand access pattern (entry 2) and with instruction fetch going to
``m_axi`` at a data-dependent address, which Vitis can only burst two words at
a time. Both are properties of the access pattern, not of the configuration,
and the design now avoids both. ``TPU_WRAP=1`` still builds the hoisted variant
for comparison.


4. A structural rule for "which modules are the memories"
----------------------------------------------------------

:Site: ``examples/tinytpu/export_rtl.py`` (``MEM_ARRAY``)
:Verdict: **rejected**; no threshold separates the two cases

The export drops the scratchpad and the accumulator so DC reports logic area
with the memory treatment taken out, and it names them with a per-toolchain
regex. A structural rule was tried instead -- *any module that declares an
array of ``reg``*. It also selects Gemmini's depth-2 queue RAMs, which are
flops in any implementation, and **no depth threshold separates those from our
own depth-4 ``lp_trip``**. The criterion stayed semantic (drop the scratchpad
and the accumulator, keep everything else) and is applied on both sides of the
comparison.


5. Re-laying out the RTL export
--------------------------------

:Site: ``examples/tinytpu/export_rtl.py`` (``RTL_SUBDIR``)
:Verdict: **buys nothing**; do not move it while runs are queued

Every variant directory is flat: ``sv2v_manifest.f``, ``MANIFEST.json``,
``README.md`` and the ``.v`` files side by side, manifest entries bare
filenames resolving against the manifest's own directory. All three variants
have always been this shape. The only inconsistency was against a separate,
now-deleted ``asic/shipped_t4`` export that put files under ``rtl/`` while
still listing them bare -- two conventions that do not compose. Flat is what is
deployed and what the synthesis sessions run against.


6. Correcting the Action cost model LOST a mutation catch
-----------------------------------------------------------

:Site: ``examples/tinytpu/mutate_actions.py`` (the ``accu_alu_narrowed`` row)
:Verdict: **the lost catch was an artefact**; the loss is the honest result

``accu_alu_narrowed`` used to be caught at the ``spec_check`` level, while
``Machine.work`` charged a row's dependency span as its occupancy: a
single-issue ALU pushed ``vaddrelu``'s rectify into a third cycle and the work
count went to 3 a row against the sequencer's 2.

That catch was wrong. A pipelined unit retires a row every ``max(resource)``
cycles, and ``accu`` reads ``ar`` twice a row through one port, so the
accumulator's read port binds at 2 a row whether the ALU chains one lane
operation or two -- the rectify of row *r* and the add of row *r+1* use
alternate cycles and fit a single-issue ALU exactly. Correcting the cost model
(``docs/source/developer/actions.rst``) therefore removed a catch, and the row
now records that **the ALU's width is the one hardware fact ``vaddrelu`` rests
on that no work count can see**. It is measurable against synthesis and nowhere
else.


7. The per-level-rounding reduction tree
-----------------------------------------

:Site: ``examples/tinytpu/ip/units/reduction_tree.py``
:Verdict: costed and rejected by its own author, for a reason that does not
   apply here

MiniTPU's tree rounds at every one of its six levels. Its owner costed the
form this unit ships -- quantise once, sum exactly at one declared width,
narrow at the consumer once -- at roughly **-6,400 LUT and L ~ 8 instead of
16**, and kept theirs only because their reference vectors were already matched
against the rounding form. Ours are not frozen, so the exact form is the
default here and the per-level-rounding form is the variant.

The full account, with the separate lane-reduce network that synthesised at
**19,198 LUT against the tree's own 20,216**, is on
``docs/source/designs/ip_gaps.rst`` ("The tree we did not build").
