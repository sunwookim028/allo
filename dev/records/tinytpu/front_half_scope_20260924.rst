The front half's boundary, swept: what maps, where it stops, and what "verified" is evidence of
================================================================================================

:Date: 2026-09-24
:Tree: worktree off ``origin/main`` @ ``991d68ba``; bindings borrowed from
       ``/home/sk3463/allo-coord`` (this worktree has no ``mlir/build``; the
       "borrowed build" case ``dev/toolchains.rst`` describes)
:Build: ``T=4 MAXDIM=64 QD=16 DMA_WORDS=1`` unless a row says otherwise
:Tool: ``examples/tinytpu/workloads/scope.py``; the committed map is
       ``examples/tinytpu/workloads/scope_map.json``

``gate.py`` says five models pass. It does not say what the front half would
do with a sixth, and nobody had walked the boundary. This is that walk. The
cheap, re-runnable part of it is now ``scope.py`` and lives with the design;
what is here is the part too expensive to run every time, plus the reasoning
the map cannot carry.

Nothing below was patched. Two of the findings are bugs and they are recorded,
not fixed, because a map taken across a repair describes neither side of it.

What the mapper does with a shape that does not fit the instruction
-------------------------------------------------------------------

**It tiles, within ``MAXDIM``, and it refuses outright above it.** This was
the open question, and the answer is not the sibling implementation's.

``allo/act/mapspace.py`` factors each extent into outer loops times the
target's intrinsic, and ``examples/tinytpu/act_target.py``'s intrinsic is one
``mm`` over a ``T x T`` weight block on ``rows`` activation rows. So an 8x8x8
GEMM on a 4x4 instruction is *not* rejected: it is enumerated as 680 candidate
nests, priced, and the cheapest is emitted. A 64x64x64 GEMM enumerates 17 192
nests and maps.

Above ``MAXDIM`` there is no tiling at all. ``TinyTpu.lower`` raises
``Refused("shape", "rank K=68 exceeds MAXDIM=64")`` before anything else, and
``extract.py`` refuses the layer one step earlier for the same reason. The
operands are one ``int8[MAXDIM*MAXDIM]`` DRAM image and nothing stages a
larger tensor through it.

So the tiling is **register- and scratchpad-level only**, and the granularity
is not free:

* **K and N must be multiples of T.** ``mapspace.residual`` returns ``None``
  unless ``extent % intrinsic == 0``, and the intrinsic is ``T`` on both.
* **M may be any integer** in ``[AR_RAW_DIST, MAXDIM]`` --- 5, 6, 7, 63 all
  map --- because ``TinyTpu.intrinsics`` yields every divisor of M as the row
  count.

The interior: 275 shapes, every one bit-exact
----------------------------------------------

``M`` in ``{4,5,6,7,8,12,16,32,48,63,64}`` crossed with ``K`` and ``N`` in
``{4,8,16,32,64}``, every cell mapped through the ACT search and then run
through ``act.correctness.check`` --- ``isa_ref`` against ``spec.gold`` over
four operand distributions with the write window enforced.

**275 mapped, 275 bit-exact, 0 refused, 0 silently wrong.** Elapsed 6 m 15 s.

That run is not in ``scope_map.json`` and should not be: the nest enumeration
is exponential in the extents, and the tail is brutal.

.. list-table:: measured search cost, one shape each
   :header-rows: 1

   * - shape
     - nests considered
     - seconds
   * - ``4x16x16``
     - 296
     - 0.08
   * - ``64x16x16``
     - 2 804
     - 0.9
   * - ``16x32x32``
     - 3 686
     - 0.7
   * - ``64x64x64``
     - 17 192
     - 19.8
   * - ``63x64x64``
     - 8 828
     - 28.5
   * - ``48x64x64``
     - --
     - 52.5

``scope.py`` therefore sweeps each axis with the other two held at 16, plus
corners, and finishes in about 50 s: the mapper is memoised on the shape, so
the op table, the shape sweep and the entry table share one search each. A scope map that costs a minute gets run;
one that costs seven does not.

Every boundary, and how each one fails
---------------------------------------

.. list-table::
   :header-rows: 1

   * - boundary
     - verdict
     - how it fails
   * - ``M < AR_RAW_DIST`` (4)
     - refused
     - **misleading cause.** ``Census.rows()`` sorts by count, so a 1x16x16
       reports ``acc-split x9`` first; ``ar-distance`` is third with 2. The
       binding constraint is ``ar-distance``.
   * - ``M > MAXDIM``
     - refused
     - clean: ``shape``, "rank M=65 exceeds MAXDIM=64"
   * - ``K`` or ``N`` not a multiple of ``T``
     - refused
     - **names no constraint at all.** ``mapspace.nests`` yields nothing, the
       census is empty, ``run.py`` prints ``every nest refused`` and
       ``gate.py`` says "every nest was refused, but the layer is declared
       mappable". Neither says ``T``. See below.
   * - ``K`` or ``N`` > ``MAXDIM``
     - refused
     - clean: ``shape``
   * - accu header count > 32767
     - refused
     - **misleading cause.** Only reachable above ``MAXDIM=76`` at ``T=4``
       (``isa_encoding.maxdim_ceiling("cubic_header", 4)``). Measured on a
       ``TPU_MAXDIM=88`` build: 76x76x76 maps and is bit-exact (12 s);
       80x80x80 and 88x88x88 refuse, but the census reports ``acc-split``
       (45 420 and 7 338) ahead of the binding ``resources`` (15 and 11),
       which is ``assemble()``'s assertion on the 15-bit count slice.
   * - anything in the box
     - maps
     - 275/275 bit-exact, above

None of the boundaries measured here hangs and none returns a wrong answer:
every one refuses, within the elapsed times above. Two of them refuse for a
reason the user is not told, and those are the rows below.

Bug 1: a Linear whose features are not a multiple of T is refused with no reason
--------------------------------------------------------------------------------

Measured end to end from PyTorch::

    Linear(6,8): 1 layer(s), 0 refusal(s) from the EXTRACTOR
      mapper: considered 0 nest(s); census (empty)
      run.py would print: 'every nest refused'

    Linear(8,6): 1 layer(s), 0 refusal(s) from the EXTRACTOR
      mapper: considered 0 nest(s); census (empty)

``extract.py`` checks only ``max(m,k,n) > MAXDIM``, so it hands the layer on;
``mapspace.residual`` then rejects every intrinsic silently, because a
refusal is only recorded when ``lower`` raises, and ``lower`` is never
reached. The user is told a nest failed when no nest existed.

This is a *message* bug, not a correctness one --- the shape genuinely is out
of scope --- but it is the boundary a real model hits first, and it is the
boundary that explains itself worst. Note that ``act/corpus/`` already knows
the shape of the answer: ``13_gemm_reduce_8x6x8_zeropad.json`` carries
``operand_pad: "zero"`` for a K of 6 and ``12_gemm_cols_8x8x6.json`` carries
``write_window: "column_block"`` for an N of 6, while ``extract.to_spec``
always emits ``arbitrary`` and ``exact``. The mapper does not read
``operand_pad`` at all, so closing this needs ``mapspace`` to round the
extent up as well as the spec to say it is allowed to.

Bug 2, the serious one: the end-to-end check does not run the model
--------------------------------------------------------------------

``run.py``'s two halves --- ``quantized_reference`` and ``run_on_machine`` ---
both walk ``extraction.layers`` in order and feed layer *i*'s int8 output to
layer *i+1*. Neither reads the fx graph's dataflow. For a chain of Linears
that is the model. For any other graph it is not, **and both sides are wrong
the same way, so they agree.**

Measured, on a module whose every node the extractor maps and nothing in
which it refuses::

    class Parallel(nn.Module):
        def forward(s, x): return s.fc1(x), s.fc2(x)

    layers 2, refusals 0, both mapped
    gate's check: 0 bytes differ over 128  -> PASSES (claims VERIFIED)
    against the REAL forward(): 62 of 128 bytes differ

So the gate would certify this module "VERIFIED against PyTorch" while the
machine's output disagrees with ``model(x)`` on half its bytes. All five
committed models are chains --- ``scope.py`` recomputes that rather than
assuming it, and reports ``topology_sound=True`` for each --- so no
*published* number is affected. What is affected is the claim's reach: the
suite's PyTorch evidence is valid **only for chain-topology graphs**, and
nothing in the suite enforces that scope. The probe is a permanent row,
``assumptions/chained_verification``, in ``scope_map.json``.

What each entry's bit-exactness is actually evidence of
--------------------------------------------------------

The suite runs three comparisons and they are not the same strength. Measured
per entry, at ``T=4 MAXDIM=64 QD=16 DMA_WORDS=1``:

.. list-table::
   :header-rows: 1

   * - entry
     - tier
     - vs ``spec.gold`` (**ours vs ours**)
     - vs PyTorch (bytes)
     - RTL
   * - ``mlp_tiny``
     - confirmed
     - pass
     - 128
     - 1 150, transcribed
   * - ``mlp_deep``
     - confirmed
     - pass
     - 208
     - 2 117, transcribed
   * - ``mlp_small``
     - confirmed
     - pass
     - 384
     - 2 781, transcribed
   * - ``mlp_wide``
     - correct
     - pass
     - 8 192
     - none (limitations item 24)
   * - ``mlp_bias``
     - probe
     - pass (its one mapped layer)
     - **none at all**
     - none

* **vs ``spec.gold``** is ``isa_ref`` --- a numpy model of the ISA written
  here --- against ``spec.gold`` --- a numpy einsum written here. It proves
  the mapper agrees with our own semantics. PyTorch is on neither side. All
  five entries clear it, including the probe's single mapped layer.
* **vs PyTorch** is the only comparison with anything outside this repository
  on one side, and what is on that side is ``torch.nn.Linear``'s forward, not
  ``model(x)`` --- the chain order is ours on both sides (bug 2). Four entries
  have it, 8 912 bytes in total, of which ``mlp_wide`` is 8 192.
  ``mlp_bias`` has none: ``gate.py`` skips the end-to-end check for a probe.
* **RTL** is a *record*, declared in ``claims.json``, that xsim once printed
  these cycles at this configuration. ``claims.json`` says so itself under
  ``source.transcribed``. ``gate.py`` checks it is self-consistent and
  configuration-matched; nothing re-measures it.

The honest one-line version: *the strongest evidence the suite carries is
8 912 bytes of agreement with PyTorch's Linear kernel, over four
chain-topology MLPs, at one configuration.*

The model set is T=4-specific
------------------------------

``scope.py --emit`` was run at ``TPU_T=8`` as well. At ``T=8``,
``mlp_deep`` **no longer maps**: its ``fc3`` is ``Linear(16, 12)`` and 12 is
not a multiple of 8, so the layer hits bug 1 and ``vs_spec`` is ``False``.
19 of the 44 swept cells are ``refused-without-a-cause`` at ``T=8`` against 12
at ``T=4``, for the same reason. Both configurations are in
``scope_map.json``, keyed by ``T MAXDIM QD DMA_WORDS``, and ``--check``
refuses to compare a configuration it has never recorded.

Ops: what a graph may contain
------------------------------

Taken from what two reference graphs actually emit --- a pre-norm transformer
block and a small CNN, both in ``scope.py`` so the list cannot be asserted
without being run --- plus the suite's five models.

**Maps:** ``nn.Linear(bias=False)``, and a ReLU that is a Linear's sole
consumer, which becomes the ``vrelu`` epilogue.

**Refused, each naming the node and the reason:** ``nn.LayerNorm``,
``nn.Conv2d``, ``nn.BatchNorm2d``, ``nn.MaxPool2d``,
``nn.Linear(bias=True)``, ``Tensor.transpose``, ``Tensor.flatten``,
``torch.matmul`` (both of the attention matmuls), ``add`` (both residuals),
``truediv``, ``softmax``, ``sigmoid``, and a ReLU that does not sit on a
Linear. Every one of these is a clean refusal with a readable message; the
transformer block yields 6 mapped layers and 9 refusals, the CNN 1 and 5.

There is no silently-wrong op and no op that crashes the extractor. The only
bad message in the op dimension is bug 1, which arrives as a *shape*.

If this is to be a loop objective
----------------------------------

It is fit as a **regression gate**, not as a search objective. ``--check``
regenerates every cell and diffs; there is no hand-maintained expected list to
drift, because the committed map is this program's own output. What would make
it unfit: raising the swept shapes toward the ceiling (the 52 s cell), or
adding models whose search cost is ``mlp_wide``'s.

The objective a front-half CHIA loop should score is *"every entry still maps,
is still bit-exact against PyTorch, its topology is still sound, and no cell
moved"* --- four booleans and a diff. "How many models map" is the wrong
objective against a fixed set of five: it is maximised at five and stays
there, so a loop optimising it has nothing to climb after the first
iteration.

If a *search* objective is wanted later it has to come from a set that can
grow, and the map is the right place to grow it: the shape section is already
a scored surface (26 of 44 cells mapped at ``T=4``, 19 at ``T=8``) and does
not depend on anyone writing new models. But scoring it would immediately
reward the wrong thing --- widening ``mapspace`` to admit a ``K`` that is not
a multiple of ``T`` raises the cell count without the operand padding that
makes the answer right --- unless every cell stays verified against
``spec.gold``\ 's einsum, which is why the sweep verifies rather than counts.
A loop that scored ``mapped`` without ``bit-exact`` would find bug 1's
shortest path first.

Reproducing
-----------

.. code-block:: bash

    source $(conda info --base)/etc/profile.d/conda.sh && conda activate allo
    export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build OMP_NUM_THREADS=8
    python examples/tinytpu/workloads/scope.py --check          # ~50 s
    TPU_T=8 python examples/tinytpu/workloads/scope.py --check   # ~35 s
