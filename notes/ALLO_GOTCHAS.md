# Allo gotchas — verified in this project

Findings from the SystemC/Catapult backend and NoC rebuild work. Everything here was hit
and confirmed on real designs; each entry says how it showed up, because most of these
give **wrong answers rather than errors**.

Inherited notes from the `sup` fork (`ALLO_SHORTCOMINGS.md`, `ALLO_LESSONS.md`,
`PITFALLS_DATAFLOW_REGION.md`) are in `archive/` — they describe a different project
(`allo-tpu`) and were not re-verified here. At least one of their claims is wrong now:
they say bitwise `&` is unsupported, but `&` works and is used throughout
`final_noc/designs/whvcrouter/whvcrouter.py`.

---

## 1. Silent wrong answers

The dangerous class — these compile, run, and produce plausible but incorrect results.

### `~x` is broken

Bitwise NOT silently produces the wrong value. Use `(0 - x)` instead. It corrupted a
one-hot grant vector in the arbiter with no error of any kind, and cost real debugging
time because every other operator in the expression behaved.

### `UInt(N)` locals read back signed

A value stored in an `UInt(N)` local and read back is interpreted as signed, so anything
≥ 2^(N-1) flips sign. **Every width needs one spare bit.** Below half range you are safe;
at or above it, you are not — and nothing warns you.

### Unused results are deleted

`MemRefDCE.cpp:28` drops any result-producing op whose result has no uses. This includes
the `try_put` ok flag and `memref.atomic_rmw`. **Always consume the result** — assigning
it to a variable you never read is not enough.

### A `Wire` has no synchronisation

A `Wire` boundary gives **zero storage and zero alignment**. It silently reads garbage
unless the kernels on both ends are cycle-locked. In the arbiter design, `Stream`,
`try_*` and `Channel` were all flawless while `Wire` returned a consistent *wrong* skew —
even with a `Channel` handshake sequencing the kernels.

---

## 2. Checks that pass because they checked nothing

The most common failure mode in this codebase, and the hardest to notice: a verification
step reports SUCCESS without having compared anything. It is worse than a wrong answer,
because "the test passes" is exactly the evidence you would go looking for.

**Five instances found in a single day (2026-08-13/14).** They are listed together because
the pattern matters more than any one of them.

| where | mechanism | what it printed |
|---|---|---|
| `run_experiment.sh` | verdict loop iterated over an EMPTY `expected*.npy` glob | `COSIM BIT-EXACT` |
| sweep plugin | `zip(golden, rtl)` truncates to the shorter list, so a missing RTL array was never compared | `COSIM_MATCH` |
| dataflow tests | output buffer still held the PREVIOUS run's correct results | the numpy assertion passed |
| `test_stream_of_blocks` | a re-zero wiped an INPUT (`B`), so the design computed `A + 0` | mismatch — this one at least failed |
| `test_mlp` | weight cache was all zeros, and the weights feed BOTH the design and the golden | `assert_allclose(0, 0)` passed |

### The tell

**A pass with no evidence of what was compared.** Every real verdict names a quantity:

    output0 BIT-EXACT (64 values)          <- real: 64 things were compared
    cosim MATCH ... (2 output array(s))    <- real: 2 arrays
    COSIM BIT-EXACT                        <- VACUOUS: no count, nothing compared

If a verdict does not say HOW MUCH it checked, assume it checked nothing until proven
otherwise.

### The rules that prevent it

- **Never let a comparison loop run over an empty collection.** Fail loudly instead;
  `run_experiment.sh` now exits 3 with `NO EXPECTED VECTORS -- this is NOT a pass`.
- **Never `zip()` two lists whose lengths you have not checked.** Compare the arities first.
- **Clear the output buffer before every run.** Allo's calling convention is in-place
  mutation (a `@df.region` is void; arrays are memrefs the callee writes), so consecutive
  runs share one buffer and a run that writes nothing inherits the previous result.
- **Only re-zero what the SAME function zero-initialised.** A file-wide scan for
  `x = np.zeros(...)` will happily clobber an input that shares a name with another test's
  output.
- **A self-writing cache must be checked for degenerate content**, not just existence.
  `test_mlp` guarded with `if os.path.exists(...)` and loaded 41,088 zeros.

### Why Allo makes this easy to hit

Two language properties conspire. Outputs are **written in place** rather than returned, so
evidence from an earlier run survives into a later check. And constants baked into a design
are often ALSO used to compute the golden (`W0: Ty[M0,M1] = np_W0` on one side,
`np.dot(X, np_W0)` on the other), so a degenerate constant makes both sides agree.

## 3. Build-time traps

### `S.put(i)` with a raw loop index fails to build

A raw `range()` index is an `index`, not an `i32`, and the stream op rejects it. Assign
to a typed local first.

### Scalar `@df.region()` args are rejected

A bare `int32` in `args=[...]` is rejected (PR #577). Use `int32[1]`, which maps to
`m_axi`. The auto-capture → `s_axilite` redesign is still pending upstream.

### `Wire` and `Channel` are SystemC-only

`hls.py:293` raises `NotImplementedError` for any target other than `systemc`. A design
using either link type cannot be built for `vhls`/`vitis` at all — worth knowing before
writing a comparison harness that assumes it can.

---

## 4. Environment

### SystemC csim needs `SYSTEMC_HOME` **and** `ALLO_CXX_EXTRA`

Calling a `mode="csim"` module (`mod(A, B)`) links against a SystemC library, and fails two
ways in sequence if the environment is incomplete:

```
RuntimeError: Set SYSTEMC_HOME for systemc csim.          # hls.py:974
./sim: /lib64/libstdc++.so.6: version `GLIBCXX_3.4.26' not found
```

The second is the nastier one: the g++ build **succeeds**, then the binary dies at run time,
and `hls.py` reports only `RuntimeError: Simulation failed.` Catapult's bundled
`libsystemc-2.3.3.so` needs a newer libstdc++ than the system one, so point the linker at
conda's:

```bash
export SYSTEMC_HOME=$MGC_HOME/shared      # has include/systemc.h + lib/libsystemc.so
export ALLO_CXX_EXTRA="-L$CONDA_PREFIX/lib -Wl,-rpath,$CONDA_PREFIX/lib"
```

The generated project's own `csim.sh` sidesteps both — it uses Catapult's `g++` and its
bundled SystemC, and needs only `MGC_HOME`. If `mod()` is being awkward, run `bash csim.sh`
in the project dir instead and read `output0.data`.

### Never override `LLVM_BUILD_DIR`

The conda `allo` env already points it at the RHEL8-compatible build
(`/work/shared/common/llvm-project-main/build-rhel8`). Overriding it with `build/` causes
a GLIBC_2.33 crash at simulator init.

### Catapult's `python3` shadows conda's

The login profile puts `$MGC_HOME/bin` on `PATH`, so vhls/vitis csim builds spawn a
`python3` that has no `nanobind` — producing mass false test failures. Put conda first:

```bash
export PATH=/home/zsm9/miniconda3/envs/allo/bin:$MGC_HOME/bin:$PATH
```

### OMP segfault at interpreter exit

With OMP threads, Python GC teardown can race OMP thread-local storage and segfault. Set
`OMP_NUM_THREADS=N` explicitly and run each region in its own process. A Python/OpenMP
interaction, not an Allo bug.

### One MLIR Context per process

`LLVM ERROR: Option 'fast' already exists!` means a second `customize()` was attempted in
the same process. Debugging by "build twice and diff" does not work — use one process per
MLIR dump and match by function name, since there is no source attribution back to your
Python.

---

## 5. Measurement traps

These cost the most time, because a wrong measurement looks exactly like a right one.

### Genus `create_clock -period` takes **ns**, not ps

Passing `2000` intending picoseconds sets a 2000 ns constraint, Genus does zero timing
optimisation, and the reported "Fmax" is just the critical path of an area-optimised
netlist. Check with `grep GENUS_SLACK`: a slack near +2e6 ps means the number is
meaningless. Fixing this *reversed* the RaveNoC Fmax verdict.

### Synthesize `concat_rtl.v`, not `rtl.v`

`rtl.v` leaves the Connections modular-IO wrappers as unresolved black boxes — 12 of
them in one design — and black boxes are excluded from **both** area and the timing
graph. Verify with `grep -c CDFG-428 genus.log` (must be 0). Check **CDFG-331** too — a
"logic abstract" is an empty module, which is how Catapult emits the `AlloMem`
`ccs_ram_sync`. Grepping only for 428 misses every Allo array.

### A Genus effort is an ATTRIBUTE, and a bare one exits 0 having done nothing

`syn_generic_effort high` is not a command. Genus reports `invalid command name`,
**abandons the rest of the script**, and prints `Normal exit` with status 0. Reports written
before that line survive on disk, so a runner that greps them finds stale numbers from a
previous run and reports them as current. It must be `set_db syn_generic_effort high`;
`genus_hi.tcl` has always had it, `genus_preserve.tcl` had lost it in an edit and therefore
never once completed a synthesis.

### `set_db <inst> .preserve` cannot protect anything pre-`syn_map`

The obvious way to stop Genus sweeping unobservable logic does not work on an unmapped
netlist:

```
Error : Cannot preserve unmapped leaf instance.                [TUI-210]
Error : Cannot preserve partially mapped hierarchical instance. [TUI-214]
```

`preserve` applies to already-mapped instances, so it is useless against the sweep that
happens *during* `syn_generic`. Worse, `get_db [current_design] .insts` returns **leaf
gates**, not hierarchical instances (`get_db hinsts` gives those) — so a `catch`-wrapped
loop over it fails on every single element and still prints "PRESERVED 291 instances".

The lever that works is to disable the optimisations themselves, as root attributes:
`hdl_preserve_unused_registers true` (before `read_hdl`), `delete_unloaded_seqs false`,
`delete_unloaded_insts false`, `optimize_constant_0_flops false`,
`optimize_constant_1_flops false`, plus `boundary_opto false` and `auto_ungroup none` to
keep the hierarchy reportable.

### Never assume a reference design's cycles/step

MatchLib's crossbar was assumed to be 1 cycle/step and is actually 3 — which flipped that
comparison from "parity" to Allo ahead. Hand-written RTL has no `cycle.rpt`, so measure
it: drive a saturating stream, timestamp arrivals, take the minimum inter-arrival gap.

### `Init` in `cycle.rpt` is the achieved II

Blank means the loop was **not** pipelined. Any II sweep that produces a blank `Init`
silently measured nothing — this happened once when the pipelining env var only existed
in one design's driver.

### Verify the golden before blaming the backend

A `test_systolic_conv` "mismatch" turned out to be an error in the hand-written numpy
golden (correlation vs convolution — the design flips the filter). When a backend bug
surfaces, check the golden against the design's *own* reference first.

### Distinguish a harness bug from a design bug

If `cosim MATCH: RTL is bit-exact with the software golden` is present and the checker
still fails, the design is not the suspect — the stimulus, the expectation or the capture
is. Also run the unpipelined control: a byte-identical failure at II=0 exonerates
pipelining. Both checks pointed at the harness before any design change was attempted.
