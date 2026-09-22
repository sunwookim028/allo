# allo — Coding Agent Notes

Fork knowledge lives in the Sphinx docs (`docs/source/`, published at
https://sunwookim028.github.io/allo/), not in Markdown notes. Read the page
before working in its area; when you learn something worth keeping, add it to
the page, not to a new `.md` file. The fork-only pages:

| Topic | Page |
| --- | --- |
| Extending Allo's abstractions: the standard, ranked gaps, `Encoding` | `docs/source/developer/extending_allo.rst` |
| Allo limitations register (items 1-23, A-G), repros in `tests/limits/` | `docs/source/developer/limitations.rst` |
| Simulator vs csim vs cosim semantics | `docs/source/developer/dataflow_semantics.rst` |
| `@df.region()` pitfalls | `docs/source/developer/pitfalls.rst` |
| Toolchains on this host, env, golden tests | `docs/source/developer/toolchains.rst` |
| Branch layout, upstream-merge procedure, worktrees | `docs/source/developer/fork_maintenance.rst` |
| Vitis: `align_value`, cosim, binutils fix | `docs/source/backends/vitis.rst` |
| Catapult: host setup, licences, directives, `ppa` mode | `docs/source/backends/catapult.rst` |
| Non-blocking streams | `docs/source/backends/nonblocking_streams.rst` |
| TinyTPU-isa, Gemmini comparison, history | `docs/source/designs/` |
| Catapult SystemC flow, CHIA, ACT | `docs/source/extensions/` |
| Dated measurement records | `docs/source/records/` |

## Quick pitfalls

- **`LLVM_BUILD_DIR` is NOT set by the conda env** — neither `conda activate allo` nor `conda run` sets it, and the simulator asserts `LLVM_BUILD_DIR is not set` without it. Export it explicitly (below).
- **Scalar `@df.region()` args** — bare `int32` in `args=[...]` is **rejected** (PR #577); use `int32[1]` → `m_axi`.
- **Region arg-order reordering**, **OMP segfault at exit**, one-process-per-MLIR-dump: see `docs/source/developer/pitfalls.rst`.
- **CHIA loop** (`examples/accelerator/tinytpu_vitis/chia_agent/`) spends real money on GCP: read `docs/source/extensions/chia.rst` first; paid runs go through `preflight.py` (CHIA2026 only, `CHIA_TOTAL_CAP_USD`), never commit `chia.env`, and run `test_harness.py` ($0) before any paid run.

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

`examples/accelerator/tinytpu_vitis/`. From a clean checkout, one command
builds the checkout's bindings, runs the functional gates, runs cosim, and
checks the published cycle counts (172/262/418/484/686; 252/383/591/667/919
before `e24e433b`):

```bash
examples/accelerator/tinytpu_vitis/reproduce.sh            # ~6 min; --no-cosim: ~1 min
```

`bench_isa.py` / `cosim.py` (default TB) are the **performance** setup
(Gemmini's [-4, 4] operands) and miss real bugs; `stress_isa.py` and
`TPU_TB=stress python cosim.py` are the **correctness** gates. Run
`stress_isa.py` (~10 s) after any change to `microarch_isa.py`, and
`mutate.py` after any change to the harness. `assemble()` rejects programs that
read `ar`/`vr`/`spad` before writing them (the arrays are not cleared by
hardware), and programs that read an `ar` row within `AR_RAW_DIST` accu
iterations of writing it: `accu`'s II=1 rests on an `s.dependence` claim
(`#pragma HLS dependence ... inter false`) that is only true under that
contract, and no simulator can see a violation -- only the `TPU_TB=stress`
cosim, which runs `ar_distance_program` at the edge.
Details: `docs/source/designs/tinytpu_isa.rst` ("Verifying a change").

## Project state

Live state is judged from git/GitHub and the docs, not a checked-in status
file: `git branch -vv`, `gh pr list -R cornell-zhang/allo`,
`gh issue list -R sunwookim028/allo`.

- Living feature map (fork vs upstream): the pinned fork issue
  https://github.com/sunwookim028/allo/issues/13.
- Fork-local file inventory: fork issue #5
  (https://github.com/sunwookim028/allo/issues/5#issuecomment-4977128476).
- Upstream-merge procedure: `docs/source/developer/fork_maintenance.rst`.

See `AGENTS.md` for build instructions, testing, and code style guidelines.
