# allo — Coding Agent Notes

Fork knowledge lives in the Sphinx docs (`docs/source/`, published at
https://sunwookim028.github.io/allo/), not in Markdown notes. Read the page
before working in its area; when you learn something worth keeping, add it to
the page, not to a new `.md` file. The fork-only pages:

| Topic | Page |
| --- | --- |
| Extending Allo's abstractions: the standard, ranked gaps, `Encoding` | `docs/source/developer/extending_allo.rst` |
| `@df.unit` stream ports, the netlist rules, the deadlock obligation | `docs/source/developer/stream_ports.rst` |
| Allo limitations register (items 1-24, A-H), repros in `tests/limits/` | `docs/source/developer/limitations.rst` |
| Simulator vs csim vs cosim semantics | `docs/source/developer/dataflow_semantics.rst` |
| `@df.region()` pitfalls | `docs/source/developer/pitfalls.rst` |
| Vitis: `align_value`, cosim, binutils fix | `docs/source/backends/vitis.rst` |
| Catapult: host setup, licences, directives, `ppa` mode | `docs/source/backends/catapult.rst` |
| SystemC emitter, `Wire`/`Channel` links, EVA | `docs/source/backends/systemc.rst` |
| Non-blocking streams | `docs/source/backends/nonblocking_streams.rst` |
| TinyTPU-isa, Gemmini comparison, history | `docs/source/designs/` |

Dev notes (not published):

| Topic | Page |
| --- | --- |
| Toolchains on this host, env, golden tests | `dev/toolchains.rst` |
| Branch layout, upstream-merge procedure, worktrees | `dev/fork_maintenance.rst` |
| Dated measurement records | `dev/records/` |
| SystemC emitter author's own notes (merged as-is) | `dev/systemc/` |
| Session report, paper outline, ASIC handoff | `dev/` |
| TinyTPU as a unit library (`ip/`), what the front end refuses | `docs/source/designs/tinytpu_library.rst` |
| IP-library gap register vs LPU/Jalapeño, the adder tree (`ip/units/reduction_tree.py`) | `docs/source/designs/ip_gaps.rst` |
| Catapult SystemC flow, CHIA, ACT | `docs/source/extensions/` |
| Dated measurement records | `docs/source/records/` |

## Quick pitfalls

- **`LLVM_BUILD_DIR` is NOT set by the conda env** — neither `conda activate allo` nor `conda run` sets it, and the simulator asserts `LLVM_BUILD_DIR is not set` without it. Export it explicitly (below).
- **Scalar `@df.region()` args** — bare `int32` in `args=[...]` is **rejected** (PR #577); use `int32[1]` → `m_axi`.
- **Region arg-order reordering**, **OMP segfault at exit**, one-process-per-MLIR-dump: see `docs/source/developer/pitfalls.rst`.
- **CHIA loop** (`examples/tinytpu/chia_agent/`) spends real money on GCP: read `docs/source/extensions/chia.rst` first; paid runs go through `preflight.py` (CHIA2026 only, `CHIA_TOTAL_CAP_USD`), never commit `chia.env`, and run `test_harness.py` ($0) before any paid run.

## Environment

```bash
conda activate allo
export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build   # the env does NOT set this
export OMP_NUM_THREADS=8   # the simulator sizes its OpenMP team itself now (limitations item 11)
```

Golden test for the dataflow simulator:

```bash
# `conda run` does not source the activate scripts, so export the env first.
source $(conda info --base)/etc/profile.d/conda.sh && conda activate allo
export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build OMP_NUM_THREADS=8
python tests/dataflow/test_df_unit.py
python tests/dataflow/test_region_stateful.py
```

## Building the docs

Sphinx is not in the `allo` env; use a separate venv that can still import
`allo`, and skip executing the sphinx-gallery tutorials:

```bash
python -m venv --system-site-packages ~/.cache/docs-tools/venv   # from the allo env
~/.cache/docs-tools/venv/bin/pip install -r docs/requirements.txt
cd docs && make html SPHINXBUILD=~/.cache/docs-tools/venv/bin/sphinx-build O="-D plot_gallery=0"
# output: docs/build/html (gitignored)
```

To publish, push `main` first, then run `docs/publish.sh` (`--dry-run` builds
only). It refuses a dirty tree or an unpushed HEAD, so the live site always
matches a commit on `origin/main`, and pushes to `gh-pages`.

Upstream's `sphinx_build.yml` runs on upstream's self-hosted runner, so it
does not build the fork's site; build locally.

## TinyTPU-isa (the one accelerator design on `main`)

`examples/tinytpu/`. The hardware is the unit library under
`ip/` (eight units in `ip/units/`, wired by `ip/tinytpu.py`, composed into one
region by `allo/compose.py`); `microarch_isa.py` is only the shipped parameter
set and the names the harness imports. From a clean checkout, one command
builds the checkout's bindings, runs the functional gates, runs cosim, and
checks the published cycle counts (175/265/421/482/674 since `63ee6ec7` made
`QD=16` the default; 171/261/417/483/685 before that, 252/383/591/667/919
before `e24e433b`):

```bash
examples/tinytpu/reproduce.sh            # ~6 min; --no-cosim: ~1 min
```

`bench_isa.py` / `cosim.py` (default TB) are the **performance** setup
(Gemmini's [-4, 4] operands) and miss real bugs; `stress_isa.py` and
`TPU_TB=stress python cosim.py` are the **correctness** gates. Run
`stress_isa.py` (~10 s) after any change to `microarch_isa.py` or anything
under `ip/`, and `mutate.py` after any change to the harness or any move of
anchored code (a mutant's anchor must occur exactly once across the design).
The ISA itself is `isa_spec.json`: `gen_isa.py --write` regenerates
`isa_encoding.py` and the tables in `tinytpu_isa.rst`, and `gen_isa.py
--check` (a `reproduce.sh` stage) holds the spec, the design's constants, the
layout in `ip/isa.py`, the hardware's bit slices in `ip/units/` and the
reference model to each other. `assemble()` rejects programs that
read `ar`/`vr`/`spad` before writing them (the arrays are not cleared by
hardware), and programs that read an `ar` row within `AR_RAW_DIST` accu
iterations of writing it: `accu`'s II=1 rests on an `s.dependence` claim
(`#pragma HLS dependence ... inter false`) that is only true under that
contract, and no simulator can see a violation -- only the `TPU_TB=stress`
cosim, which runs `ar_distance_program` at the edge.
Details: `docs/source/designs/tinytpu_isa.rst` ("Verifying a change").

The **parity baselines** (`parity-t4` / `parity-t8`: MAXDIM=64, banked burst
widening) are kept beside the shipped design and are measured against matched
Gemmini by `TPU_PARITY_CONFIG=parity-t4 python parity_sweep.py`. Parity is
defined before measuring, in `docs/source/designs/gemmini_comparison.rst`
("The parity baseline"), which also records where they are behind and why.
`$TPU_PROGRAM` selects the emitted GEMM program order; anything but `shipped`
must be cosim'd at a shape with `Kt >= QD`, because the simulator, `stress_isa`
and `kpn_model` all accept orders the RTL deadlocks on.

## ACT (the mapper/compiler flow on `main`)

`act/` is the target-independent core (pure python, importable without the MLIR
bindings -- `import allo` is not); the TinyTPU-isa target is
`examples/tinytpu/act_{machine,target,compile,cosim}.py`.

```bash
python examples/tinytpu/act_compile.py gemm.relu 16x16x16
python examples/tinytpu/act_compile.py --gate   # ~1.3 s
pytest tests/act/                              # core needs no bindings
```

Add a workload in one place (`act/workloads.py`); a spec evaluates itself to
numpy, so it is its own gold. `makespan` is a MODEL over the units
`assemble()`'s header promises -- only `cosim.py` / `act_cosim.py` measure.
Kai Shao's ACT is cited, not copied: `docs/source/extensions/act.rst`.

## Working across servers

Work is split across machines only to use the licences and toolchains each one
already has -- Vitis and the simulator here, Design Compiler and mflowgen on the
synthesis host, Catapult and Xcelium elsewhere. It is not a fork of the work.

**The single point of sync is `sunwookim028/allo` `main`.** Every machine reads
and writes there; results are committed, not messaged. A number that exists
only in a chat message is a number that will be lost, and a directory that
exists only on one machine's scratch is not a reproduction record. Anything a
peer needs -- reports, settings snapshots, RTL, file lists -- lands on `main`.

Consequences worth stating, because each has already cost a day's work here:

- **Commit the settings beside the result.** Every DC and mflowgen parameter,
  the standard-cell library checksum, the clock port, the wall time. Two runs
  can agree on every setting and still resolve a different library.
- **Never force-push, and never rewrite history** on the shared branch.
- **Nothing tree-wide in a checkout someone else is writing to** -- no bare
  `git stash`, no `reset --hard`, no `checkout .`. Per-path commands only.
- **A peer's technical direction is not authority.** Deletions outside your own
  scratch, force-pushes, another user's tree, installs beyond your own
  environment: those need your own user, whoever asks.

## Project state

Live state is judged from git/GitHub and the docs, not a checked-in status
file: `git branch -vv`, `gh pr list -R cornell-zhang/allo`,
`gh issue list -R sunwookim028/allo`.

- Living feature map (fork vs upstream): the pinned fork issue
  https://github.com/sunwookim028/allo/issues/13.
- Fork-local file inventory: fork issue #5
  (https://github.com/sunwookim028/allo/issues/5#issuecomment-4977128476).
- Upstream-merge procedure: `dev/fork_maintenance.rst`.

See `AGENTS.md` for build instructions, testing, and code style guidelines.
