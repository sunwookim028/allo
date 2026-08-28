/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef ALLO_TRANSFORMS_PASSES_TD
#define ALLO_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

def MaterializeTopologyPass : Pass<"materialize-topology", "ModuleOp"> {
  let summary = "Materialize static stream topology.";
  let description = [{
    Specialize mapped process interfaces by scalarizing statically indexed
    ranked stream lanes and trimming unused process arguments.
  }];
  let options = [
    Option<"allowPublicAbiChange", "allow-public-abi-change", "bool", "false",
           "Allow scalarizing ranked stream ports on public kernels.">
  ];
}

def GridMappingPass : Pass<"grid-mapping", "ModuleOp"> {
  let summary = "Map processes to a grid.";
  let description = [{
    Map processes to a grid.
  }];
}

def DataflowSpawnPass : Pass<"dataflow-spawn", "ModuleOp"> {
  let summary = "Launch tagged dataflow PEs as fibers on the marl runtime.";
  let description = [{
    Rewrite the (already LLVM-lowered) sequential calls to dataflow PEs, the
    callees tagged `allo.dataflow.pe` by lower-dataflow, into concurrent fiber
    spawns onto the marl runtime: `allo_df_open` / `allo_df_spawn` per PE (with a
    generated `void(ptr)` thunk that reloads the PE's operands from an alloca'd
    context) / `allo_df_join` / `allo_df_close`. Run after the module is in the
    LLVM dialect (operands are pure LLVM values, casts reconciled).
  }];
  let dependentDialects = ["mlir::LLVM::LLVMDialect"];
}

def FoldConstantCallsPass : Pass<"fold-constant-calls", "ModuleOp"> {
  let summary = "Evaluate lazy-consteval kernel invokes with constant arguments.";
  let description = [{
    Lowers the "lazy consteval" feature. A `@consteval(lazy=True)` function
    enters the IR as an `allo.kernel` tagged `allo.lazy` (fast filter) and is
    called via `allo.invoke`. Once grid-mapping has specialized the call sites so
    every invoke argument is constant, this pass evaluates each invoke at compile
    time, replaces it with the constant result, and deletes the unused lazy
    kernels so they never reach codegen. Evaluation clones the kernel body
    into a throwaway func.func, substitutes the constant arguments, and folds it
    to a constant via a fixpoint of canonicalize + CSE + affine store-to-load
    forwarding (so local scratch arrays fold) + full unrolling of constant-trip
    affine.for and scf.for loops. Because `@consteval(lazy=True)` is an explicit
    request to fold the function away, an invoke that cannot be reduced to a
    constant (non-constant arguments or a body that does not evaluate) is a hard
    error. Run between grid-mapping and materialize-topology.
  }];
  let dependentDialects = [
    "mlir::arith::ArithDialect", "mlir::affine::AffineDialect"
  ];
}

def ReuseCleanupPass : Pass<"reuse-cleanup", "ModuleOp"> {
  let summary = "Simplify the IR produced by reuse_at.";
  let description = [{
    Clean up the redundant IR left behind by `transform.allo.reuse_at`: run
    CSE + canonicalization, merge adjacent `affine.if` wrappers that share a
    condition (the per-access conditional loads), and sink pure tails into the
    branches. This is a no-op on modules that contain no reuse_at output, which
    is detected via the `allo.reuse.maintenance` markers; those internal markers
    are stripped once cleanup completes.
  }];
  let dependentDialects = ["affine::AffineDialect"];
}

def GenerateApintWrapperPass
    : Pass<"generate-apint-wrapper", "ModuleOp"> {
  let summary = "Wrap an APInt-boundary top kernel with a std-width interface";
  let description = [{
    For a top kernel whose arguments/results use non-standard-width integers
    (e.g. ap_int<5>), rename it `<name>__impl` (private) and synthesize a public
    wrapper with the original name whose boundary uses standard widths (smallest
    of 8/16/32/64). The wrapper truncates inputs into ap_int temporaries, invokes
    the real kernel, and sign/zero extends results and in-out buffers back to the
    standard widths, so numpy/ctypes and the LLVM memref ABI can call it.
    Signedness is read from the `allo.signed` marker.
  }];
  let options = [
    Option<"topName", "top", "std::string", /*default=*/"\"\"",
           "Name of the top kernel to wrap.">
  ];
  let dependentDialects = [
    "mlir::allo::AlloDialect", "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect", "mlir::memref::MemRefDialect"
  ];
}

def FoldIfStatementsPass
    : Pass<"fold-if-statements", "func::FuncOp"> {
  let summary = "Fold away affine.if / scf.if the scheduler cannot pipeline across";
  let description = [{
    A collection of rewrites that remove or simplify `if` control flow so the SDC
    scheduler sees as little of it as possible inside a loop body. Greedily
    applied; the home for all `if`-oriented transforms:

    * **Hyperblock predication.** Flatten an `affine.if`/`scf.if` into a single
      predicated (hyperblock) datapath. The branch condition is an `i1`
      (materialized from the `affine.if` integer set, or taken directly from
      `scf.if`); both branches are speculated, with side-effect-free ops and
      loads hoisted unconditionally, value results becoming `arith.select`, each
      guarded store a predicated read-modify-write (`store select(cond, v,
      load)`), and each guarded stream `get`/`put` gated on the condition via
      its i1 predicate operand (the token is consumed/produced only where the
      branch is taken, keeping the access in the pipelined region). An `if`
      whose body holds an un-maskable side effect (a nested region such as a
      loop or an inner if, a call, or any other effecting op) is not
      predicated.
    * **Guard-to-bound (affine index-set splitting).** When an `affine.if` guards
      the *entire* body of its enclosing `affine.for` with a single inequality
      that is affine in the loop's induction variable, fold the guard into the
      loop's bounds (`for j in [0,N): if j > i` -> `for j in [i+1,N)`) and drop
      the `if`. This eliminates the control construct entirely, skipping the
      dead iterations rather than masking them, and only applies to `affine.if`
      (its predicate is a static function of the IVs; a data-dependent `scf.if`
      cannot be folded this way).

    An `if` left standing after these rewrites schedules as one opaque unit (the
    scheduler keeps it as a raw guard around its decomposed body).
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect", "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect"
  ];
}

def RaiseCountedWhilePass
    : Pass<"raise-counted-while", "func::FuncOp"> {
  let summary = "Raise a counted scf.while to scf.for";
  let description = [{
    Raise an `scf.while` whose exit is a pure ordered test of a monotonically
    increasing induction variable against a loop-invariant bound to an
    `scf.for`, so the counted-loop scheduler handles it unchanged. Only
    provably-counted, increasing loops are raised (`<`/`<=`, constant positive
    step, `index` IV, unused IV result, identity condition forwarding, a
    test-only `before` region); every other `scf.while` is left untouched for
    the uncounted (flushable-pipeline) path.
  }];
  let dependentDialects = [
    "mlir::scf::SCFDialect", "mlir::arith::ArithDialect"
  ];
}

def LoopCanonicalizationPass
    : Pass<"loop-canonicalization", "func::FuncOp"> {
  let summary = "Shape every counted loop nest into one pipelineable region";
  let description = [{
    One sweep that erases the loops which never run, then four rewrites run as
    one recursive descent per nest: unroll the loops under a pipelined one, sink
    an imperfect nest's prologue and epilogue into its inner loop, normalize a
    perfect band to lower bound 0 and step 1, and coalesce that band into a
    single loop.

    ## Erasing zero-trip loops

    A counted loop that provably runs zero times is erased with its whole body,
    its results replaced by its `iter_args` inits, before anything else reads a
    nest's shape. Two shapes are recognized: constant bounds that do not cross
    (`range(1, 1)`, `range(0)`), and an `affine.for` whose lower and upper
    bounds are the same expression over the same operands (`range(n, n)`,
    whose trip-count map folds to the constant 0); an `scf.for` adds the same
    lower/upper SSA value. Nothing below is written for a trip count of 0 (the
    perfectizing guard would fire on iteration -1, a coalesced band would carry
    a zero factor), and the hardware for such a loop is a controller that only
    ever takes its empty escape.

    ## Unrolling under a pipeline (`unroll-under-pipeline`, default on)

    Pipelining a loop implies its body is a single straight-line datapath
    clocked at one II, so every loop nested inside a pipelined loop
    (`allo.pipeline.ii` != -1) is fully unrolled into it, matching Vitis, where
    `#pragma HLS pipeline` on an outer loop auto-unrolls the inner loops. A
    nested loop that is not a constant-trip counted loop (a dynamic bound, or an
    `scf.while`) cannot be unrolled, so the pipeline cannot be formed and the
    nest is left unchanged with a remark. An unrolled interior is straight-line,
    so nothing below applies and the descent stops there.

    ## Perfectizing (`perfectize`, default off)

    An outer counted loop whose body is `{ prologue..., inner loop,
    epilogue... }` is made perfect by sinking the surrounding ops into the inner
    loop: the epilogue under a last-iteration guard (remapping the inner loop's
    results to its yielded values); a prologue store under a first-iteration
    guard (runs once, e.g. an accumulator init); a pure prologue op or a load of
    a memref not written in the nest unguarded (recompute). The guard is an
    `affine.if` (constant IV) for an affine.for inner or an `scf.if` (runtime
    `cmpi`) for an scf.for inner; `fold-if-statements` predicates it. Bails,
    leaving the nest untouched, on sibling inner loops, an `scf.while`, an
    escaping or aliasing value, or other side effects. Off by default: it is a
    QoR alternative, and the scheduler runs an imperfect nest as sequential
    sub-regions without it.

    ## Normalizing

    Moving a loop's lower bound and step out of the loop and into the subscripts
    that read its induction variable leaves `for i' in [0, trip)` and
    `i = lb + step*i'`, which is what coalescing requires and nothing else
    establishes. `s.unroll(k)` leaves exactly the form that needs it:
    `loopUnrollByFactor` turns a unit-step loop into one stepping by the factor,
    so the standard way of opening memory bandwidth was also the way of refusing
    to flatten the nest it opened.

    Two loops are left alone, both decided structurally before the rewrite. A
    **runtime lower bound**: `for j in range(i, N)` would become `j = i + j'`,
    so an address naming `j` would name two counter registers and the adder
    joining them where it named one, and it buys nothing, a runtime lower bound
    meaning a runtime trip count, which coalescing separately refuses. An
    induction variable with a **non-affine use**: only an affine consumer
    absorbs the leftover `affine.apply` (canonicalization composes it into the
    map), and anywhere else it stays an op the scheduler pays for.

    ## Coalescing

    The maximal normalized, constant-trip-count prefix of each perfect affine
    band is coalesced into a single loop (`affine::coalesceLoops`), so the SDC
    scheduler pipelines the whole nest at one II instead of scheduling only the
    innermost body and paying the pipeline fill/drain at every inner-loop
    re-entry. A loop with a symbolic bound is left alone: the coalesced
    induction-variable arithmetic (`floordiv`/`mod` by the bound) would be
    semi-affine and defeat the dependence analysis. A loop carrying `iter_args`
    is also left alone: coalescing merges the induction spaces but drops
    loop-carried values, which would silently miscompile an inner accumulator.
    Levels are then given back until no array's composed address needs a
    divider: recovering an induction variable divides by a trip count, which is
    wiring only when that count is a power of two, and the address is the one
    path nothing else prices. Banking needs no such rule, since a bank digit
    that folded to a constant holds no dimension and so survives the
    delinearization unchanged. Only simple nests are handled; complex ones
    should go through the `transform.allo.flatten` op, which coalesces
    unconditionally.
  }];
  let options = [
    Option<"unrollUnderPipeline", "unroll-under-pipeline", "bool", "true",
           "Fully unroll the loops nested inside a pipelined loop.">,
    Option<"perfectize", "perfectize", "bool", "false",
           "Sink an imperfect nest's prologue and epilogue into its inner "
           "loop.">
  ];
  let dependentDialects = [
    "mlir::affine::AffineDialect", "mlir::arith::ArithDialect",
    "mlir::scf::SCFDialect"
  ];
}

def DropTrivialFuncPass : Pass<"drop-trivial-func", "ModuleOp"> {
  let summary = "Erase a function that computes nothing, along with its calls";
  let description = [{
    A function whose entire body is `return` of its own arguments computes
    nothing: an empty one and an identity one, which returns an argument unchanged.
    Each is erased with every `func.call` naming it, a call's results taking the
    operands the callee forwarded.

    Only PRIVATE functions are erased: a public one is the module's interface,
    so its symbol outlives whatever its body says. A symbol reached any way
    other than a `func.call` keeps its function whole, there being no call site
    to rewrite. Dropping a call can leave its own caller trivial, so the pass
    runs to a fixpoint.
  }];
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def AssignBanksPass : Pass<"assign-banks", "func::FuncOp"> {
  let summary = "Decide which bank of a partitioned array each access reaches";
  let description = [{
    Runs before the scheduler, once the `allo.part` attributes are final
    (`reconcile-array-directives`) and the access maps are in their final affine
    form (after `loop-canonicalization` / `scalarize-memory`). It records the
    bank an access reaches as an `allo.bank` attribute on the access, which
    `convert-schedule-to-dcp` then moves onto the `dcp.load`/`dcp.store` op's
    own `bank` attribute.

    The bank an access uses is a DERIVED fact, and deriving it separately at
    each consumer means three IR layers, three map forms and a
    delinearize/linearize round trip in between whose losslessness they only
    happen to agree on: the scheduler's port model before the solve,
    `dcp-resolve-banking` after it, and the emitter's routing and crossbar after
    that. The dangerous direction is silent: an access billed one bank's port
    that reaches the crossbar and takes a port on every bank breaks the II the
    solve proved. Deciding once and recording makes them agree by construction.

    The decision itself is exactly what the partition attribute implies
    (`staticBankOf`): a bank index that folds to a constant, else none, meaning
    the access reaches every bank. Being a recorded fact rather than a predicate
    is what makes it a place a real bank-assignment policy could live.

    The fold is asked of the map over the values its subscripts TAKE, not over
    the operands it names: an induction variable of constant lower bound `lb`
    and step `s` runs over `lb + s*d`. The step is the fact the map alone cannot
    carry, and `s.unroll` is what leaves it behind, since `loopUnrollByFactor`
    turns a unit-step loop into one stepping by the factor. Without the
    substitution the canonical way to open memory bandwidth, unroll by `k` over
    a cyclic partition of factor `k`, resolves NO bank and doubles the II. That
    form is for reasoning only and never escapes this pass: the counter still
    holds `i` itself, so pricing or emitting the substituted expression would
    charge arithmetic that is not built.

    A partition that resolves no access at all is reported, per array. It is
    legal and it is sometimes what the author wants (a crossbar), but it is also
    what a mis-shaped partition looks like, and every other layer downstream is
    silent about it.

    ## A SKEWED array records a slot, not a bank

    A skew banks on the SUM of the subscripts, so no access to it has a
    compile-time bank: `(i + j) mod F` is a runtime value whichever way it is
    read. What it has instead is a SLOT. The sum splits into a runtime part and
    a constant, so the bank is `(class + slot) mod F`, and two accesses of the
    same class reach the same bank exactly when their slots agree, at every
    value of the class. A slot is therefore billable exactly as a static bank
    is: F accesses with F distinct slots take one port per bank rather than a
    port on every bank, which is the whole reason a skew is worth having. It is
    not ROUTABLE, since only `MemUnit::skewed` consumers that select among banks
    at run time may read it, which is why `dcp-resolve-banking` declines such an
    array.

    Two conditions, both checked here rather than downstream, because the port
    model reads the answer before the solve:

    * The classes must agree among accesses that can CONTEND, which is those in
      one block: two accesses in different blocks are in different scheduling
      regions and never take a port in the same cycle. An array filled in one
      nest and read in another therefore has two classes and no conflict.
    * Every access must get a slot, or none does. The emitter shares one port
      per bank across a lane on the strength of its accesses staying off each
      other's banks; one access left to crossbar would take a port on every one
      of them.

    An argument array gets no slot at all. Its banks are boundary interfaces the
    manifest published, one set per access, so there is no shared port for a
    lane to economize. That is the same boundary, and the same reason, as
    `dcp-resolve-banking`'s internal-only split.
  }];
}

def ResolveBankingPass : Pass<"dcp-resolve-banking"> {
  let summary = "Split a statically-banked partitioned array into per-bank memrefs";
  let description = [{
    A DCP-normalization pass, the only one, run between
    `convert-schedule-to-dcp` and emit. An `allo.part` partition asks for N
    physical banks; this pass materializes them for an **internal** array (a
    `memref.alloc`/`alloca`) every access of which `assign-banks` assigned a
    bank. It creates one per-bank alloc (bank-shaped, no partition attribute)
    and routes each access to its bank with the in-bank coordinate
    (`bankSplitOf`), so the physical banks match the ones the scheduler bound
    its per-bank ResII against. The emitter then sees N plain (`numBanks == 1`)
    memories with no banking logic. A partitioned array with an access that was
    assigned no bank (or an external/argument array) is left untouched, so it
    reaches the emitter still `numBanks > 1`, the crossbar fallback.

    This is a REALIZATION step, not a decision: it splits where the recorded
    decision permits, and asks nothing about the address itself.

    Two asymmetries the name hides. **Internal only** is not a limitation to be
    lifted: an argument array's banks are boundary interfaces the manifest
    publishes so the host can shard a numpy array across them
    (`Interface.cpp`), so its layout is a contract rather than a choice, and
    the per-bank interfaces ARE its realization. For an argument the emitter
    path is not a fallback, it is the only path.

    And the granularity differs: the bank decision is PER ACCESS (that is what
    the port model bills and what the emitter routes on), while this pass is all
    or nothing per array, since one unassigned user needs a crossbar over the
    very banks a split removes. That costs nothing in the schedule. An unsplit
    array's assigned accesses still reach one bank each and its unassigned ones
    still reach all of them, so the split changes which IR carries the banks,
    never how many ports an access takes.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect", "mlir::memref::MemRefDialect"
  ];
}

def ElideDeadInitPass
    : Pass<"elide-dead-init", "ModuleOp"> {
  let summary = "Drop a local array's initializer when every element that is "
                "read is written first";
  let description = [{
    `buf: i32[N] = 0` declared inside a loop lowers to a `linalg.fill` that
    re-executes on every enclosing iteration. When nothing ever reads an element
    the fill left behind, that fill is dead, and it is not cheap: its stores are
    scheduled, they contend for the array's write ports, and they lengthen the
    enclosing region. Measured on a 32-entry buffer under a 16-iteration loop,
    latency 1616 -> 1088; on an 8-entry one that `auto-complete-partition` turns
    into registers, 448 -> 304.

    The condition is EVERY READ IS WRITTEN FIRST, not whole-array coverage. A
    nest writing `0..k` followed by one reading `0..k` qualifies while never
    covering the declared shape; a `+=` accumulator does not, its first read
    preceding every write; nor does a sliding window, whose taps ARE the
    filter's startup state.

    A read qualifies through either of two tests. The first walks the array's
    live range from the initializer onward, through nested regions, threading
    polyhedral element sets: what every execution has definitely written by the
    current op against what has been read before any write covered it. The
    initializer is dead when the second comes back empty. The second test pairs
    the read with each write through affine dependence analysis, which settles a
    read that trails its own producer by an iteration, as in a scan or a
    wavefront. A `func.call` is summarized per parameter, bottom up over the
    call graph, by the same walk over the callee body.

    Element sets are taken at the depth of the loops around the DECLARATION so
    that those stay symbols. Guards and loop domains go into the iteration space
    before it is projected away, so a store below an `affine.if` claims only
    what it really writes and a store below a loop that may not run claims
    nothing; a data-dependent `scf.if` leaves no condition to fold, so only what
    both arms write is certain. A read may be over-approximated, a write may
    not, which is why inner IVs and mod/div terms are existentially quantified
    rather than projected out.

    Qualification: a local `memref.alloc` / `alloca` whose every use is an
    affine load or store, a whole-array fill, or a call this pass can summarize,
    so no traffic is hidden; an initializer that is the FIRST thing to touch the
    array, since a read before it observes the previous execution's residue; and
    an initializer that is safe to delete whole, meaning a `linalg.fill` or a
    nest whose only effect is writing this array. Recursion and any unmodelled
    use read as opaque. `ALLO_LOG_LEVEL=debug` reports what survived and why.

    Position: after `convert-allo-to-func`, so calls are resolvable, and before
    `convert-linalg-to-affine-loops`, where the initializer is still one op whose
    coverage is definitional rather than a nest to be matched.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect", "mlir::linalg::LinalgDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def RaiseToAffinePass
    : Pass<"raise-to-affine", "func::FuncOp"> {
  let summary = "Raise scf.for loops and memref accesses to their affine form";
  let description = [{
    Rewrite an `scf.for` to an `affine.for`, and a `memref.load` / `memref.store`
    to its affine counterpart, wherever the affine form computes what the
    original computed: bounds and subscripts must be affine functions of the
    enclosing induction variables and loop-invariant values, and the step a
    positive constant.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect", "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect", "mlir::scf::SCFDialect"
  ];
}

def ScalarizeMemoryPass
    : Pass<"scalarize-memory", "func::FuncOp"> {
  let summary = "Scalarize a small local array to registers or remove it entirely";
  let description = [{
    Perform two steps of small-array handling. First wraps `affine-scalrep` pass to
    perform store-to-load forwarding on a local array, eliminate or fold away
    redundant memory accesses. It runs under Allo's aliasing contract
    (`alloAliasAnalysis`), which also lets it drop the overwritten stores a fully
    unrolled accumulation over a kernel argument leaves behind.

    Second, for the untouched survivors, automatically partition an array with
    `allo.part` so it lowers to registers rather than a memory whose ports and
    access latency bound the II. An array whose every subscript is a constant
    fully dissolves into wired registers with no mux, so it is scalarized at any
    size. A variable subscript instead pays a read mux and a write demux, worth
    it only while the array is small enough (at most `maxElements` elements) to
    bound the mux and one block issues more than two accesses, which a
    dual-ported row cannot serve in a cycle. An array touched once or twice per
    iteration with a variable subscript keeps the storage the memory tables
    select. `maxElements` 0 disables auto-partition (forwarding still runs).
  }];
  let dependentDialects = [
    "mlir::allo::AlloDialect", "mlir::affine::AffineDialect",
    "mlir::memref::MemRefDialect"
  ];
  let options = [
    Option<"maxElements", "max-elements", "int64_t", "16",
           "Cap on a variable-subscript array's element count; a constant-"
           "subscript array is scalarized at any size, 0 disables both">
  ];
}

def HoistInvariantReadsPass
    : Pass<"hoist-invariant-reads", "func::FuncOp"> {
  let summary = "Hoist loop-invariant memory reads in front of their loop";
  let description = [{
    Move a read whose address does not change across a loop's iterations out in
    front of the loop, so the body meets a held register instead of spending a
    memory port on the same word every iteration. An unrolled or pipelined body
    re-reading a small coefficient array is the shape this targets: the
    repeated reads bound the initiation interval by port count.

    A read moves only when every operand is defined outside the loop, the loop
    provably runs at least once, and the array sees no write and no opaque use
    inside the loop. Reads of a complete-partitioned array stay put: they are
    already wires, and hoisting would only add registers carrying the values
    across the region boundary. Identical hoisted reads collapse to one access.

    Out of a leaf body, the one the modulo scheduler paces, reads move only
    when their removal lowers the body's port-pressure floor, the widest
    array's accesses per bank per iteration; a preload that leaves that maximum
    in place buys no II and only spends the held registers. A body with loops
    inside is paced by its children, so an invariant read there always moves.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect", "mlir::memref::MemRefDialect"
  ];
}

def RaiseMemoryReductionsPass
    : Pass<"raise-memory-reductions", "func::FuncOp"> {
  let summary = "Promote a loop-carried memory reduction to an SSA iter_arg";
  let description = [{
    Rewrite a reduction that a loop carries through memory into one it carries
    through an SSA iter_arg. The shape targeted is a load, an associative
    combine, and a store to one location whose address does not change across
    the loop:

        affine.for %k = 0 to N {
          %a = affine.load M[inv]
          %s = arith.addf %a, %leaf
          affine.store %s, M[inv]
        }

    becomes a preload before the loop, an iter_arg threaded through the combine,
    and a single write-back after it. The rewrite is bit-exact: it reorders no
    operations, it only moves the accumulator off the memory port and onto a
    register. Its purpose is to expose the reduction to `tree-height-reduction`
    and `rotate-reductions`, which recognize an iter_arg accumulator but not a
    memory one, so a rolled `acc[i] += x` loop can reach II=1 the same way an
    unrolled one already does.

    The combine is an associative add or mul, or a subtractive accumulate
    `M -= leaf` (the accumulator being the minuend). Since raising reorders
    nothing, moving the subtraction onto a register is bit-exact too, dropping
    its recurrence from the memory round trip to the operator latency.

    A location is promoted only when it is provably self-contained across the
    loop: the address is loop-invariant (`isInvariantAccess`), the load and store
    are unconditional in the body with the load reaching the store, the stored
    value and loaded value each have a single use, the array has no opaque or
    hidden user, and the polyhedral dependence test proves no other access in the
    loop touches the location. Several independent reductions in one loop are
    raised together into one loop carrying several iter_args.
  }];
  let dependentDialects = [
    "mlir::affine::AffineDialect", "mlir::arith::ArithDialect",
    "mlir::memref::MemRefDialect"
  ];
}

def RotateReductionsPass
    : Pass<"rotate-reductions", "func::FuncOp"> {
  let summary = "Rotate reductions across N accumulators to reach II=1";
  let description = [{
    Replace a loop-carried reduction's single accumulator with N accumulators in
    a shift register, so a given partial sum is updated once every N iterations.
    This turns the distance-1 recurrence (which bounds the modulo II at the
    operator latency L) into a distance-N recurrence over one pipelined operator,
    i.e. `RecII = ceil(L / N)`, which is 1 once `N >= L`. The N partial sums are
    combined by a balanced tree after the loop.

    A loop's iter_args are classified independently, so several independent
    reductions in one loop (a complex accumulate's real and imaginary parts, say)
    each rotate on their own; plain carried values are left alone. `add`/`mul`
    reductions rotate for both float (reassociation is inexact) and integer
    (exact) accumulators. Only leaf loops are rotated: the shift register is
    realized on a childless modulo loop, so a container reduction stays put.
    Composes after tree-height-reduction, which first folds an unrolled
    reduction chain into a single carried operator.

    `accumulators` sets the count N: 0 is off, a positive value forces that N on
    every rotatable reduction, and -1 is auto. Auto reads each reduction
    operator's latency L at `period-ns` from the same operator library the
    scheduler uses, and rotates on N = L, the least count that brings
    `RecII = ceil(L / N)` to 1 without surplus accumulators. A count below 2
    (a forced 1, or an operator already single-cycle at the period) leaves the
    loop alone, as does a trip not known to reach N: a runtime-short trip would
    drain the accumulators wrong.
  }];
  let options = [
    Option<"accumulators", "accumulators", "int", "0",
           "Rotated accumulator count: 0 off, a positive value forces N, -1 "
           "auto (N = the reduction operator's latency at period-ns).">,
    Option<"periodNs", "period-ns", "double", "0.0",
           "Clock period auto mode reads the reduction operator's latency at; "
           "unused unless accumulators is -1.">
  ];
  let dependentDialects = [
    "mlir::affine::AffineDialect", "mlir::arith::ArithDialect"
  ];
}

def TreeHeightReductionPass
    : Pass<"tree-height-reduction", "func::FuncOp"> {
  let summary = "Balance associative datapath and reduction trees, folding a "
                "carried accumulator at the root";
  let description = [{
    Balance every maximal single-operator associative tree into minimum height,
    so the SDC scheduler sees a logarithmic combinational cone instead of a
    linear chain. One pass covers both within-iteration datapath and loop-carried
    reductions; a value that recurs across iterations (an iter_arg, or a memory
    tap reading an earlier iteration's store) is folded in last, at the root, so
    its recurrence spans a single operator and the modulo II does not grow with
    an unroll factor.

    A tree is balanced in one of two modes. The direct mode handles three
    families: the additive family treats `sub` as a sign flip, so a mixed add/sub
    tree (a stencil tap `a - b + c - d`, or a subtractive accumulate) collapses
    to a signed-leaf multiset rebuilt as (sum of the positives) - (sum of the
    negatives); the multiplicative family balances a `mul` tree; and each of
    `and`/`or`/`xor` balances its own. The widened mode handles the integer
    bit-growth idiom `trunc(core(ext, ext))`, rebuilding the same idiom so operand
    widths stay put. A different-operator operand is a leaf, balanced separately.

    The balanced subtree is built as a latency-weighted Huffman tree: a leaf's
    arrival delay (from the operator library at `period-ns`) places later-arriving
    operands nearer the root. Equal or zero weights give a plain balanced tree, so
    the result is never worse than an unweighted one.

    Integer trees are exactly reassociable; an operator carrying an overflow flag
    ends the tree as a leaf (reassociation moves where the wrap is observed), and
    index arithmetic stays out. Floating-point reassociation is inexact, so a
    float tree is balanced, and a float division by a constant is strength-reduced
    to a reciprocal multiply, only under `enable-fp`. A carried tree is rewritten
    from three leaves (the recurrence fold is the win); a non-carried one needs
    four leaves and a balanced depth below the linear chain.
  }];
  let options = [
    Option<"enableFp", "enable-fp", "bool", "false",
           "Rebalance floating-point trees and reciprocalize constant "
           "float divides (both inexact).">,
    Option<"periodNs", "period-ns", "double", "0.0",
           "Clock period the Huffman leaf weights are read at; 0 leaves the tree "
           "unweighted (a plain balanced tree).">
  ];
  let dependentDialects = [
    "mlir::affine::AffineDialect", "mlir::arith::ArithDialect"
  ];
}

def NarrowDemandedBitsPass
    : Pass<"narrow-demanded-bits", "func::FuncOp"> {
  let summary = "Narrow integer arithmetic to the bits that carry information";
  let description = [{
    Bit-growth typing gives an expression its exact natural width, computed from
    the leaves up, and applies the destination type as a single `arith.trunci`
    at the end. Every operator in between is built at a width nothing reads:
    `a: i48 = b * c` over two `i32` values builds a 64-bit multiplier and then
    discards 16 bits.

    This pass propagates that truncation back up the expression. For the
    two's-complement ring operators (`addi`, `subi`, `muli`) and the bitwise
    operators (`andi`, `ori`, `xori`) the low `w` bits of the result are a
    function of the low `w` bits of the operands alone, so

        trunc_w(a `op` b) == trunc_w(a) `op` trunc_w(b)

    exactly, for signed and unsigned alike. A constant-amount shift sinks too: a
    left shift by `c < w` always, a right shift when the bits the narrow shift
    cannot see are known clear (`shrui`) or replicated sign (`shrsi`). Sinking
    the truncation lands it on the leaves, where it meets the extends bit growth
    introduced and folds away, which exposes the next operator up to the same
    rewrite. Division, remainder, comparison, min and max read the high bits, so
    the demand stops at them and their operands keep the natural width. A wide
    operator with more than one use keeps its width, since the wide value stays
    live.

    The pass also reads the bits forward. `x & K` is dropped when every bit `K`
    clears is one the value provably never sets, which a forward walk over the
    casts, shifts and bitwise operators establishes (`llvm::KnownBits` carries
    the algebra). Writing a bit field splices with such a mask on every field
    after the first. A load-bearing mask stays: one overwriting a mid-word
    field stays an AND, and a low mask, being a zero-extended truncation,
    becomes resize casts, which are wiring.

    Values are narrowed forward as well. An interval walk over constants,
    constant loop bounds and the monotone arith transfers bounds each value,
    and a ring or bitwise operator whose bound needs fewer bits than its
    carrier is rebuilt at that width behind resize casts. In particular an
    `index` operator, otherwise built at the one width the datapath gives
    `index`, enters the integer domain at its value's width: a product of two
    loop counters under a `% 2^k` mask ends as a k-bit fabric multiply instead
    of an `index`-wide multiplier IP.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect"];
}

def FloatToIntPass : Pass<"float-to-int", "func::FuncOp"> {
  let summary = "Demote a provably integer-valued float cone to integer arithmetic";
  let description = [{
    A float computation whose values are all integers, fed in through
    `sitofp`/`uitofp` and read back out through `fptosi`/`fptoui`/`cmpf`, builds
    float arithmetic IP where integer operators would do. This pass walks up from
    those float-to-int roots through the truncation-free float operators
    (`addf`, `subf`, `mulf`, `negf`, `maximumf`/`minimumf`/`maxnumf`/`minnumf`,
    the float resizes `extf`/`truncf`, and `select` through its two arms) to the
    integer-to-float leaves, tracking the integer range each value can take with
    `llvm::ConstantRange`, and rewrites the whole cone to
    `addi`/`subi`/`muli`/`maxsi`/`minsi`/`cmpi`/`select` when the range proves the
    rewrite exact. Min and max collapse to one signed op because an integer-valued
    float is never NaN; the same fact makes a `uno`/`ord` NaN test a constant
    `false`/`true`, which is how a NaN-quieting min/max expansion demotes; a resize
    is the identity on the integer value.

    Float arithmetic on integers is exact only while the value stays inside the
    type's exactly representable integer band (`2^24` for f32, `2^53` for f64).
    Each op is held to its OWN float type's band, so a mixed-precision cone is
    sound: an f64 subexpression may range past f32's band as long as it is not
    truncated to f32 there. The accumulated range over-approximates every value,
    so when each op needs no more signed bits than its mantissa holds, no
    operation ever rounded and the integer replacement is bit-identical. A cone
    whose range escapes its band, or that reads a fractional constant, a
    non-finite constant, a loop-carried value, or `divf`/`remf`, is left as float.
    Signedness follows the cast opcode: `sitofp`/`fptosi` sign extend,
    `uitofp`/`fptoui` zero extend, and `cmpf` maps to a signed `cmpi`.

    A loop-carried sum reduction demotes too: an `affine.for` accumulator whose
    running sum over a constant trip count stays inside the band becomes an
    integer recurrence, with the loop rebuilt to carry the integer carrier.

    Being bit-exact it needs no fast-math flag. It runs both before the integer
    passes and again after if-conversion and reduction raising, so a demoted cone
    is balanced, narrowed and IP-bound as ordinary integer arithmetic, and an
    if-converted diamond, an expanded min/max, or a raised reduction is caught
    once it is a `select` or an iter_arg.
  }];
  let dependentDialects =
      ["mlir::arith::ArithDialect", "mlir::affine::AffineDialect"];
}

def LegalizeArithPass : Pass<"legalize-arith", "ModuleOp"> {
  let summary = "Device-IP-aware expansion of composite arith ops (RTL path)";
  let description = [{
    The RTL-path replacement for MLIR's device-blind `arith-expand`. A composite
    arith op that the device provides a dedicated operator IP for (a
    `max`/`min`/`maxnum`/`minnum`/`ceildiv`/`floordiv` `dcp.operator` matching the
    op's kind and signature) is KEPT, so the scheduler binds it to that IP and
    the emitter instantiates it as one unit. Otherwise the op is EXPANDED into
    primitive arith via the upstream arith-expand patterns (a float max becomes
    `cmpf`+`select`, a ceildiv becomes div/mul/cmp/select), so a device that
    provides only the primitives (e.g. a float compare IP) still schedules.

    Integer `maxsi`/`maxui`/`minsi`/`minui` are left untouched: they lower to a
    native combinational op and need no IP. The keep/expand decision reads the
    injected `dcp.operator` / `dcp.device` IR, so this pass must run after operator
    injection and before scheduling.

    A read of a constant table (an initialized, never-written `memref.global`)
    at literal indices folds to the element it names, which is what lets a
    multiply see its literal factor. A table that variable reads still address
    keeps its storage for those reads.

    Under `expand-const-arith`, a `divsi`/`divui`/`remsi`/`remui` by a constant
    that is not a power of two is reduced to a reciprocal multiply and shift, a
    signed one carrying the sign around the magnitude's multiply. An index op
    expands regardless: it carries no width for an IP signature to match, so it
    would otherwise be priced as a full combinational divider, one operator no
    register can split. A typed op has a divider IP to fall back on, so it
    expands only where the product multiply fits the clock period stated in
    `period-ns`, and keeps the IP everywhere else. Under the same flag, a
    `muli` by a literal whose non-adjacent form takes at most three add/subs
    recodes into that shift-add network, exact modulo the result width for
    either sign; the shifts are by literals and so are wiring. A wider factor
    keeps its multiplier, which a DSP slice serves better than a deep adder
    tree. The pipeline sets the flag only on its second run, since before
    subscripts are raised to affine maps arithmetic under an access can still
    become a carried register digit or part of an affine map.
  }];
  let options = [Option<"expandConstArith", "expand-const-arith", "bool", "false",
                        "Reduce division by a non-power-of-two constant to a "
                        "reciprocal multiply and shift, and a multiply by a "
                        "small constant to its shift-add network">,
                 Option<"periodNs", "period-ns", "double", "0.0",
                        "Clock period a typed reciprocal multiply must fit; "
                        "0 keeps every typed division on its IP">,
                 Option<"areaFuseGate", "area-fuse-gate", "bool", "false",
                        "Fuse a multiply-add only where the fused core prices "
                        "no higher than the multiply plus the combinational "
                        "add it replaces; the area objective sets this">];
  let dependentDialects = ["mlir::arith::ArithDialect"];
}

def ExpandRegionBoundsPass
    : Pass<"expand-region-bounds", "mlir::func::FuncOp"> {
  let summary = "Reify affine region boundary expressions as operations";
  let description = [{
    A counted loop's runtime bounds and a guard's predicate are an `AffineMap`
    and an `IntegerSet`, which the schedule cannot hold: a constraint system has
    a vertex only for an operation. This evaluates each of them into arith
    beside its anchor, so the boundary is scheduled and cut against the clock
    like any other computation.

    The loop and the guard themselves are not rewritten, which keeps the
    polyhedral dependence test exact. Arith is not a valid affine symbol, so the
    reified values reach their anchor through an `allo.volatile` marker placed
    immediately before it, which is also the use that keeps them alive. The
    marker carries the runtime lower bound then the upper bound (a guard: its
    predicate), omitting whatever the anchor's own maps make constant;
    `entryConeOf` decodes it.

    The arithmetic is typed at `datapathWidth` rather than left on `index`,
    which carries no width for an operator row to be priced at or for an IP
    signature to match. That is the width the emitted bound always had.

    Run after every pass that reshapes loops (perfectization, if-folding,
    raising) and before scheduling: arith beside an inner loop breaks a perfect
    band deliberately, that band's bound arithmetic being work of its own.
  }];
  let dependentDialects = ["mlir::arith::ArithDialect",
                           "mlir::allo::AlloDialect"];
}

def ReconcileArrayDirectivesPass
    : Pass<"reconcile-array-directives", "ModuleOp"> {
  let summary = "Give every carrier of one array the single `allo.part` and "
                "`allo.bind.storage` its kernels agree on";
  let description = [{
    A partition is a property of the array, but the array is named once per
    function it is visible in: a `memref.alloc`, a `memref.global`, a parameter
    of each kernel it is passed to. Each place carries its own `allo.part`, and
    every consumer reads whichever one is nearest, so left alone a callee is
    scheduled against an unbanked parameter while its caller materializes banks.

    This pass makes them one statement. Carriers are unified along call-argument
    edges into per-array classes (an async spawn is a call like any other), every
    partition stated on a class is JOINED, and the result is written back to all
    of them. The relation is symmetric because a sub-kernel masters ports on the
    caller's array rather than receiving a copy, so a directive travels in either
    direction and reaches the array's other kernels.

    The join is `joinPartitions`, which lives beside `bankLayoutOf` so the order
    and the decode it orders cannot drift apart: axes on different dimensions
    compose, factors on one dimension must divide, a complete partition absorbs,
    and a block against a cyclic axis on one dimension is reported as the
    conflict `Skew` exists to answer. One sweep suffices, since which carriers
    denote one array follows from the call graph and never from the attributes.

    `allo.bind.storage` rides the same classes, and its two axes reconcile
    differently. `impl` names the structure the array is held in, of which there
    is one: two carriers naming different rows are refused whatever their
    latencies. `type` asks for a port topology, and those form a chain (`t2p`
    serves everything `s2p` does, `s2p` everything `1p` does), so the carriers
    take the one that covers the others and a side that asked for less is not in
    conflict.

    Runs before `sdc-scheduling`, so a callee is scheduled knowing its banks:
    banking a composed array pays off only if the *callee* issues N accesses per
    cycle, which needs its per-bank ResII and per-bank port groups. (Routing one
    unbanked callee port to N banks through a caller-side crossbar would add
    hardware and buy nothing.) After this, leaf banking handles a partitioned
    parameter unchanged, and a container merely materializes the banks its
    children already agree on.
  }];
  let options = [
    Option<"top", "top", "std::string", "\"top\"",
           "Name of the top kernel of the design">
  ];
  let dependentDialects = ["mlir::func::FuncDialect"];
}

def OutlineLooseProcessesPass
    : Pass<"outline-loose-processes", "ModuleOp"> {
  let summary = "Make a dataflow container's own datapath one more process";
  let description = [{
    A concurrent (dataflow) container composes child instances, channels and
    shared memory; it has no datapath of its own, and the structural top has no
    way to grow one, since "the top computes nothing" is the invariant its whole
    composition rests on. Compute written directly in such a container
    (`aux[0] = x[0] * 2` beside the process network, a scalar computed for a
    process to consume, a post-pass over what the network produced) therefore
    has nowhere to go.

    This pass normalizes the IR into the shape the backend already handles
    rather than teaching the backend a second shape: each maximal run of loose
    datapath ops is outlined into a private function and replaced by a call to
    it, so the container satisfies the no-datapath invariant by construction and
    the outlined work is scheduled, emitted and composed exactly like any other
    child. The same move `loop-canonicalization` makes for an imperfect nest.

    The outlined call is deliberately NOT marked `allo.async`: a spawn is
    broadcast the container's `start`, whereas a plain child fires at its
    scheduled offset, which is what orders it against the work it reads from or
    writes for. Runs are split by the calls between them, so program order is
    preserved, and with it the scheduler's ordering and the emitter's start
    gates.

    Best effort by design: a run whose captured values or results cannot cross a
    module boundary is left alone, and the emitter's own check reports it. That
    keeps this pass free of its own error paths and the diagnostic in the one
    place that knows what a structural top can wire.
  }];
  let dependentDialects = ["mlir::func::FuncDialect"];
}

#endif // ALLO_TRANSFORMS_PASSES_TD
