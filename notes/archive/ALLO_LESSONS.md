# Working with Allo — lessons from the L2 FlashAttention session

Companion to `ALLO_SHORTCOMINGS.md` (catalog of bugs / gaps).
This file is about *how to work* given those gaps, especially for an
LLM coding agent whose context is finite.

## Core insight

**Allo is fixable, but every Allo fix and every TPU feature both want to
own the agent's full context window.** Mixing them — "I'll add VPU
regions to L2 *and* fix the simulator stream lowering in the same
session" — burns context on cross-talk:

- TPU work wants the agent thinking in terms of opcodes, address
  layouts, stride flags, dataflow rendezvous.
- Allo work wants the agent thinking in terms of MLIR dialects, ASTContext
  state, lowering pipelines, `func.func` translation.

The two share almost no vocabulary. When the agent is forced to swap
between them mid-session, it does what just happened in this session:
spends hours rewriting tpu.py to dodge a library bug, instead of
recognizing the library bug and patching it directly.

## Rule of thumb: separate the two layers, always

| Layer | Repo | Mode |
|---|---|---|
| Allo (compiler) | `~/projects/allo` | Library work — touches `allo/ir/builder.py`, `allo/backend/simulator.py`, etc. |
| TPU (user code) | `~/projects/allo-tpu` | Application work — only touches `levels/L*`, `kernels/`, `programs/`. |

**Never edit both repos in the same session.**

If TPU work surfaces an Allo bug:
1. Stop the TPU session. Capture the bug as a minimal reproducer
   (≤ 50 lines, no TPU vocabulary in it).
2. Open a separate Allo session. Fix the compiler. Land the patch on a
   feature branch. Confirm with the minimal reproducer + the existing
   `tests/dataflow/test_*` regression set.
3. Resume the TPU session against the patched Allo. Re-bind to the
   original goal.

The TPU session's job was never to fix Allo. Trying to "do both" while
juggling context is what made this session 4× longer than it needed to
be.

## Trust nothing without a clean rebuild

- `.cache/llvm_sim/` hashes only the level's `tpu.py` and top-level
  `tpu_config.py`. Edits to `levels/_common/mxu_fp32.py`, `kernels/*`,
  or the Allo library do not invalidate it.
- Stale cache reports "PASS" for code that doesn't even compile.
- Before claiming any commit is validated:

      rm -rf .cache/llvm_sim/
      LEVEL=L2 make sim

  If the cache wasn't cleared, the commit message lies. (Several
  recent commits in this repo say "validated" — they were validated
  against a cached object from an older tpu.py.)

## Always start a TPU session with a green baseline

Before adding a feature:

1. Clear the sim cache.
2. Run the existing validate (`levels/LN/validate.py`) end-to-end on a
   real rebuild.
3. Only then start changing things.

If step 2 fails, that is the *only* thing the session works on. Do not
add features on top of a red baseline; you cannot tell whether new
breakage is yours or pre-existing.

## Minimal reproducer before any "Allo feels broken" theory

When the agent thinks Allo is misbehaving, the cheapest sanity check is
a 30–50 line file at `/tmp/repro_X.py`:

- Imports just `allo`, `allo.dataflow as df`, types.
- Defines the smallest region/kernel that exercises the suspected
  feature combination.
- `df.build(top, target='simulator')` and run.

In this session, `/tmp/test_stateful_subregion.py` (Stateful +
sub-region call) ran in 5 seconds and immediately ruled out *most* of
the search space. It should have been written first, not after hours of
tpu.py rewrites.

If the reproducer fails: hand it to the Allo session as the bug report.

If the reproducer passes: the bug is interaction-specific, narrow it
further (add streams? add nesting?). Each failed reproducer is
disposable — keep one feature delta per file.

## Stop on the first compiler-internal error

These are library bugs, not user-code bugs:

- `cannot be converted to LLVM IR: missing LLVMTranslationDialectInterface`
- `Assertion 'value' failed`
- `Failure while creating the ExecutionEngine`
- `AttributeError: 'ASTContext' object has no attribute 'global_op_cache'`
- `LLVM ERROR: Option 'fast' already exists!`

When any of these appear, the right move is *not* to rewrite tpu.py
hoping to dodge them. The right move is:
- File the bug (minimal reproducer + observed error + expected behavior)
- Switch to an Allo session
- Patch the compiler

Trying to dodge a compiler-internal bug from user code wastes hours and
usually produces ugly user code that doesn't actually work.

## Plans need a kill switch

The TPU plan that drove this session ("dynamic-beaming-newt") had a
linear shape: do step 1, then step 2, then validate. No fallback.

Plans should encode:
- "If approach A fails after N attempts, fall back to approach B."
- "If we hit a compiler-internal error, stop, file it, switch
  sessions."
- A concrete checkpoint to retreat to when things go sideways
  (e.g. last known-good tag of both repos).

## Inspecting MLIR safely

- The MLIR Context can't be re-initialized in the same Python process
  (`Option 'fast' already exists!`). Don't try to `customize()` twice.
- One process per MLIR dump:

      conda run -n allo python -c "...customize and print str(s.module)..."

- Pipe through `head` / `sed -n` to grab specific line ranges; never
  try to dump the full module to terminal (10k+ lines).
- Generated MLIR has no source attribution back to tpu.py — match by
  function name (`func.func @decoder_0`, `@mxu_fp32__0`, etc.) not by
  line number.

## What the agent's session looked like, in retrospect

Approximate breakdown of where time went:

- ~10 % actual TPU feature implementation (VPU, hardware loop,
  transpose, validate_fa.py)
- ~30 % rewriting tpu.py to dodge Allo limitations that were
  patchable upstream
- ~40 % bouncing between Allo branches (`main`,
  `feature/region-scope-stateful`) without a clear bisect
- ~20 % chasing stale-cache "validation passed" reports

The 10 % is the only useful work. Everything else is the cost of mixing
the two layers in one context.

## TL;DR for the next session

1. Pick a single layer (TPU or Allo). Do not edit the other repo.
2. Start with `rm -rf .cache/llvm_sim/ && LEVEL=L2 make sim` and a
   passing baseline before any feature work.
3. First compiler-internal error → minimal reproducer → close session
   → file Allo bug → fix in dedicated Allo session → resume.
4. Don't trust any commit message that says "validated" without
   evidence the cache was cleared.
