Settling ``big-shapes``: the disputed row was right, and the branch is still not mergeable
==========================================================================================

:Date: 2026-09-24
:Tree: ``big-shapes`` @ ``b0271e18``; ``main`` @ ``92f0618f``
:Toolchain: Vitis HLS 2023.2, xcu280 at 3.33 ns, T=4, default testbench

``big-shapes`` sat unmerged for two days over two questions: an unexplained
mixed-sign edit to ``reproduce.sh``'s ``EXPECTED``, and a
``TPU_TILED=32x512x128`` cosim that burned 29 hours for 28 seconds of CPU and
had to be killed. Both are now settled by measurement.

The rows, each with the configuration it was taken at
-----------------------------------------------------

Every row below is the default (performance) testbench, T=4, ``COSIM OK``
(every shape bit-exact against ``isa_ref``).

.. list-table::
   :header-rows: 1

   * - design
     - MAXDIM
     - QD
     - 4x4x4
     - 8x8x8
     - 12x12x12
     - 16x16x8
     - 16x16x16
   * - ``main`` (on-chip operand mirror), published
     - 16
     - 16
     - 175
     - 265
     - 421
     - 482
     - 674
   * - ``big-shapes`` (per-row ``m_axi``), **measured here**
     - 16
     - 16
     - **181**
     - **265**
     - **419**
     - **476**
     - **684**
   * - ``big-shapes``, **measured here**
     - 64
     - 8
     - 178
     - 262
     - 416
     - 478
     - 696
   * - ``big-shapes``' own ``EXPECTED``, as the branch left it
     - 64
     - 8
     - 178
     - 262
     - 416
     - 478
     - 696

The mixed-sign row was never wrong
-----------------------------------

The branch's ``EXPECTED`` reproduces **exactly**, to the cycle, at the
configuration it was taken at. It was a real measurement that left no log in
the tree, which is why it read as unverified.

The mixed sign -- ``+6 / 0 / -2 / -6 / +10`` against the then-published
``172 / 262 / 418 / 484 / 686`` -- is not a configuration confound and not
noise. It is **the cycle signature of the DMA change itself**, and it
reproduces: the same ``+6 / 0 / -2 / -6 / +10`` appears when the two designs
are compared like-for-like two configurations later, at MAXDIM=16 and QD=16
(181/265/419/476/684 against main's 175/265/421/482/674). Removing the mirror
costs a few cycles of un-hidden AXI latency where the program names each row
once, and saves where the mirror over-fetched whole DRAM rows.

The branch's central claim -- that MAXDIM no longer has to be pinned, because
the DRAM row stride became runtime data -- also holds, and sharply. Subtracting
main's independently measured QD 8->16 deltas (``+4/+4/+4/-1/-11``) from the
MAXDIM=64 row predicts ``182/266/420/477/685``; the measured MAXDIM=16 row is
``181/265/419/476/684``. **MAXDIM=64 costs exactly +1 cycle at every one of the
five shapes**, which is the known memory-sizing effect. On the mirror design
the same comparison is 47 cycles at 4x4x4 (218 against 171).

``QD=16`` clears the deadlock class at a large tiled shape
-----------------------------------------------------------

The blocked ``TPU_TILED=32x512x128`` measurement was re-taken at ``QD=16``
under ``ACT_COSIM_TIMEOUT``. Note that the *first* attempt did not hang at
all: it was refused at assembly, ``a 32x512 operand is 16384 bytes, past the
DRAM_WORDS=4096 an operand port addresses``. The shape needs ``TPU_DRAM=65536``
as well, which is a second reason the original run was not measuring what it
thought it was.

With ``TPU_TILED=32x512x128 TPU_QD=16 TPU_DRAM=65536``, on the branch's
design, the measurement the branch was blocked on **completes**:

.. code-block:: text

   32x512x128  cycles=262426   TB 32x512x128 mismatches = 0 / 4096
   COSIM OK (testbench=default)

and it completes in about two minutes of RTL simulation, against a run of the
same shape that burned **29 hours for 28 seconds of CPU** and had to be killed.
The log carries the completing signature item 24 defines -- the *second*
progress line and a ``$finish``, which a hanging run never prints:

.. code-block:: text

   // RTL Simulation : 0 / 1 [n/a] @ "109000"
   // RTL Simulation : 1 / 1 [n/a] @ "876618000"
   $finish called at time : 876638110 ps

So this is **both** the branch's missing number and a second, independent
confirmation that ``QD=16`` clears the item-24 deadlock class -- the first was
the ten-program ``act/rtl_hang.py`` family, all small and fully unrolled; this
one is a 262,426-cycle tiled GEMM with a real loop nest, three orders of
magnitude larger. 262,426 cycles against the 262,144 packed items the shape
implies is **1.001x**, so the critical unit is not merely finishing, it is
finishing at its promised rate.

Why the branch still cannot be rebased
---------------------------------------

Not staleness: ``main`` re-architected the design underneath it.
``microarch_isa.py`` is now a 102-line re-export shim over an ``ip/`` package
(``b057712c``), and ``isa_spec.json`` plus ``gen_isa.py --check`` became a
conformance gate that re-derives every literal bit slice in ``ip/units/`` from
the spec (``7c5df6a6``). A rebase conflicts the whole 1850-line file against a
file that no longer exists; a merge keeps both the monolith and the package.

Re-implementing the branch's work on ``main`` is roughly **30-34 files and
1,400-1,900 lines**. The riskiest part is not the ``dma_ld`` rewrite, which is
contained, but the spec gate: the branch widens ``nr`` from 8 to 10 bits, which
moves two computed spec properties that assert *which* ceiling binds, and
``main`` moved the other way on the key premise -- ``WPR`` is a build-time
derived constant (``MAXDIM // T``), declared in ``isa_spec.json`` and used as
the legal range of ``col_block``, where the branch needs it to be runtime data.

What survives regardless
-------------------------

* **The strided per-row read is II=1 on this build, not II=4** (branch commit
  ``8001d696``). Confirmed here independently: one ``[HLS 214-115]`` note in
  the whole csynth log, on gmem0 (the 512-bit instruction port) and none for
  the operand ports, and ``dma_ld_0_1_Pipeline_VITIS_LOOP_645_1`` at
  ``Final II = 1, Depth = 17``. ``docs/source/backends/vitis.rst`` already says
  the same thing from a different measurement. The comment in
  ``ip/units/dma_load.py`` asserting II=4 is corrected in the same commit as
  this record -- it is the stated justification for a mirror that is still
  shipped.
* **``cosim.py`` had no timeout.** ``ACT_COSIM_TIMEOUT`` now bounds each
  ``cosim_design`` and kills the child as a process group.
* The measured cost of removing the mirror, above, which is the number any
  future re-port has to beat.

Reproducing
-----------

.. code-block:: bash

   export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build OMP_NUM_THREADS=8
   cd examples/accelerator/tinytpu_vitis
   TPU_MAXDIM=16 TPU_QD=16 python cosim.py     # the published configuration
   TPU_TILED=32x512x128 TPU_QD=16 TPU_DRAM=65536 \
       ACT_COSIM_TIMEOUT=3600 python cosim.py  # the tiled shape, bounded

The branch is preserved as the tag ``archive/big-shapes``.
