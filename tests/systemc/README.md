# SystemC backend — demonstrations and validation harness

Everything that exercises the SystemC / Catapult backend: the small Allo
programs each of which demonstrates one emitter behaviour, the hand-written RTL
testbenches, the Catapult synthesis scripts, and the SystemC-vs-RTL cross-check.
The logs these produced are in [`dev/records/systemc/`](../../dev/records/systemc/).

**Why these are not in `examples/`.** They were, as `examples/systemc/`, until
2026-09-24. `examples/` holds designs and is named after them; these are named
after a backend because that is what they are about. Five of the seven assert on
the *emitted SystemC text* rather than on a result — `assert "AlloMemPins<" in
code` is a claim about the emitter, not about a design — and `tiled_systolic.py`
says outright that it is `tests/dataflow/test_tiled_systolic.py` run through
`target="systemc"`, the same design on a different backend. `demos/` is "the
smallest possible program for one language feature each". See
`dev/repo_layout.md`.

## Quick start

```bash
source $(conda info --base)/etc/profile.d/conda.sh && conda activate allo
export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build OMP_NUM_THREADS=8

pytest tests/systemc/test_emit.py       # every claim below, ~5 s, no Catapult
python tests/systemc/pc_channel.py mlir      # the frontend MLIR
python tests/systemc/pc_channel.py systemc   # the generated SystemC
python tests/systemc/pc_channel.py csim      # build + run csim  (needs MGC_HOME)
```

`test_emit.py` is the only pytest-collected file here; it emits each program
below and checks the assertion that program's own `__main__` block makes. Before
it existed nothing ran those assertions, and `tiled_systolic.py` had been
asserting `AlloMem<`/`AlloMemW<` counts for a boundary form the emitter had
stopped producing. The other pytest-visible SystemC tests are
`tests/dataflow/test_systemc_backend.py` and
`tests/dataflow/test_stateful_systemc.py`.

Everything else here drives external EDA tools. **Neither Catapult nor Xcelium
is installed on this fork's development host** (`dev/toolchains.rst`), so it
cannot be exercised here; it is kept runnable for a host that has them.

## The programs

| File | What it demonstrates | Exercises |
|---|---|---|
| `pc_channel.py` | producer → consumer over a handshake channel | `Channel[valid_ready]`, the minimal path |
| `stream_boundary.py` | region boundary arrays vs. stream-only kernels | `Stream`, boundary I/O, which module synthesizes |
| `systolic_chain.py` | `mapping=[P]` unrolled into P kernel modules | multi-kernel stream chains, no emitter special case |
| `tiled_systolic.py` | tiled GEMM over three memory-port boundaries | `AlloMemPins`, both directions, `ap_int<65>` accumulation |
| `mem_port_reverse.py` | reversed array reads | random-access memory ports (loads) |
| `mem_port_scatter.py` | scattered writes | memory-port stores |
| `dot_product_four_links.py` | one dot product, four ways | `Wire` vs `Stream` vs `Channel` vs fused — the modularity-tax comparison |

Two of these exist to establish a finding rather than to be copied:
`dot_product_four_links.py` (the comparison is the point; no one of the four is
*the* design — cited by `docs/source/developer/limitations.rst`) and
`demos/nb_producer_consumer.py` (non-blocking stream ops are non-deterministic
under the current simulator; its output is the finding).

`demos/` holds the smallest possible program for one language feature each: link
types (`link_types_demo.py`), a stream (`stream_producer_consumer.py`), a wire
(`wire_producer_consumer.py`), non-blocking stream ops (`nb_stream_rtl.py`) and
the non-determinism they show in the untimed simulator
(`nb_producer_consumer.py`).

Running a program standalone writes its generated SystemC next to itself. Those
`.cpp` files are **output, not source** — gitignored here, with the archived
copies under [`dev/records/systemc/generated/`](../../dev/records/systemc/generated/).

## The harness

| File | What it does | Needs |
|---|---|---|
| `csyn_subdir.py` | runs Catapult `csyn` on a program from this directory, from a build **subdirectory** — the workaround for CIN-124 / SCHD-30 port degradation | `MGC_HOME` |
| `synth_compute.tcl`, `synth_source.tcl` | Catapult synthesis of the two `stream_boundary` submodule tops | Catapult |
| `cosim_tb_producer_consumer.v` | hand-written RTL cosim testbench for the producer/consumer design (B = A + 1) | Xcelium or VCS |
| `cosim_tb_mem_port_reverse.v` | the same for the memory-port design; **currently deadlocks**, and the header explains why that is a real csim-vs-RTL divergence rather than a testbench bug | Xcelium or VCS |
| `rtlsim/` | the SystemC-vs-RTL cross-check behind limitation 22 — see its own README | Vivado xsim or Xcelium |

```bash
# csyn a program by module name; the module is looked up in this directory
python tests/systemc/csyn_subdir.py pc_channel pc_channel
ALLO_DESIGN_TOP=producer_0 python tests/systemc/csyn_subdir.py pc_channel pc_channel

# Catapult synthesis of a stream-interface submodule top
catapult -shell -f tests/systemc/synth_compute.tcl
```

`ALLO_DESIGN_TOP` matters for **any area number you report**: a `@df.region`
contains the testbench kernels and their memories, so synthesizing the region
top includes the harness in the reported area. The flow also has a first-class
`configs["synth_top"]` for the same purpose. Note cosim does *not* work against
a submodule top — SCVerify wraps the design top and the `input<k>.data →
AlloMem` stimulus path disappears. Run measurement and verification separately.

The two `.tcl` scripts read an archived emitter output,
`dev/records/systemc/generated/stream_boundary.cpp`. They find it by searching
**upward** from the script for the repository root, so neither the cwd nor the depth
of this directory matters — see `dev/roadmap.md` on why nothing here resolves a root
by counting levels. `csyn_subdir.py` and `rtlsim/guard_experiment/emit.py` do the same.
Regenerate the archived `.cpp` with `python tests/systemc/stream_boundary.py`.

The two `cosim_tb_*.v` testbenches are driven by hand; each file's header gives the
exact `df.build(..., mode="csyn")` call that produces the RTL it expects and the
`xrun` invocation that runs it.

## Results

[`dev/records/systemc/VERDICTS.md`](../../dev/records/systemc/VERDICTS.md) — per-example
outcome for every `tests/dataflow` design run against the SystemC backend (simulator vs
`mode="csim"`, identical seeded inputs), with the failures categorized by root cause.

## See also

- [`docs/source/backends/systemc.rst`](../../docs/source/backends/systemc.rst) — the published backend page
- [`../../dev/systemc/SYSTEMC_BACKEND.md`](../../dev/systemc/SYSTEMC_BACKEND.md) — the backend guide
- [`../../dev/systemc/DATAFLOW_LINKS.md`](../../dev/systemc/DATAFLOW_LINKS.md) — link types
- [`../../dev/systemc/ALLO_GOTCHAS.md`](../../dev/systemc/ALLO_GOTCHAS.md) — read before writing Allo code
