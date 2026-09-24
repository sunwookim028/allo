# SystemC / Catapult examples

Runnable Allo dataflow designs that go through `target="systemc"`. The
testbenches, run scripts and recorded results that used to sit beside them live
elsewhere now — see [Where the rest went](#where-the-rest-went).

**Two of these are experiments rather than examples**, and the difference
matters because one of them is the evidence behind a published limitation. Both
are expressed as Allo designs, which is why they are here rather than under
`tests/`, but neither is a pattern to copy:

- **`demos/nb_producer_consumer.py`** empirically demonstrates that non-blocking
  stream operations are non-deterministic under the current simulator — it
  exists to establish a finding, and its output is the finding.
- **`dot_product_four_links.py`** asks whether a `Wire` removes the modularity tax, by
  expressing one dot product four ways (fused, `Wire`, `Stream`, `Channel`).
  The comparison is the point; no single one of the four is *the* design.

Everything else here is an example in the ordinary sense: a design you can read
and copy.

## Start here

**[`pc_channel.py`](pc_channel.py)** — the minimal example: one producer, one consumer, one
`Channel[int32, valid_ready]`. Small enough to read in a minute, and it exercises the whole
path.

```bash
source $(conda info --base)/etc/profile.d/conda.sh && conda activate allo
export LLVM_BUILD_DIR=/home/sk3463/llvm-allo-6b09f739/build OMP_NUM_THREADS=8

python pc_channel.py mlir      # the frontend MLIR
python pc_channel.py systemc   # the generated SystemC
python pc_channel.py csim      # build + run csim, check B == A   (needs MGC_HOME)
```

## The designs

| File | Design | Exercises |
|---|---|---|
| `pc_channel.py` | producer → consumer over a handshake channel | `Channel[valid_ready]`, the minimal path |
| `stream_boundary.py` | region boundary arrays as streams | `Stream`, boundary I/O |
| `systolic_chain.py` | chained PEs | multi-kernel stream chains |
| `tiled_systolic.py` | tiled GEMM | stream-output, self-synchronizing termination |
| `mem_port_reverse.py` | reversed array access | random-access memory ports (`AlloMem`) |
| `mem_port_scatter.py` | scattered writes | memory-port stores, replica write-merge |
| `dot_product_four_links.py` | one dot product, four ways | `Wire` vs `Stream` vs `Channel` vs fused — the modularity-tax comparison |

`demos/` holds the smallest possible programs for one language feature each:
link types (`link_types_demo.py`), a stream (`stream_producer_consumer.py`), a wire
(`wire_producer_consumer.py`), non-blocking stream ops (`nb_stream_rtl.py`) and the
non-determinism they show in the untimed simulator (`nb_producer_consumer.py`).

Running a design writes its generated SystemC next to itself. Those `.cpp` files are
**output, not source** — the archived copies are under `dev/records/systemc/generated/`.

## Where the rest went

This directory used to also hold the validation harness and the measurement logs.
Both act on designs rather than being designs, so they moved:

| What | Where | Why |
|---|---|---|
| `csyn_subdir.py`, `synth_*.tcl`, `cosim_tb_*.v` | [`tests/systemc/`](../../tests/systemc/) | validation: testbenches and run scripts |
| the SystemC-vs-RTL cross-check (was `examples/systemc_rtlsim/`) | [`tests/systemc/rtlsim/`](../../tests/systemc/rtlsim/) | same |
| `VERDICTS.md`, `reports/` | [`dev/records/systemc/`](../../dev/records/systemc/) | dated measurement records |
| the generated `.cpp` kept for reference | [`dev/records/systemc/generated/`](../../dev/records/systemc/generated/) | emitter output, not a source |

## Synthesis and cosim

`csynth` needs a build subdirectory — running Catapult in the directory that holds
`kernel.cpp` degrades `Connections::In`/`Out` ports to raw `sc_signal`s (CIN-124 / SCHD-30).
[`tests/systemc/csyn_subdir.py`](../../tests/systemc/csyn_subdir.py) works around it, and
takes the design module by name from this directory:

```bash
python tests/systemc/csyn_subdir.py pc_channel pc_channel          # synthesize the whole region
ALLO_DESIGN_TOP=producer_0 python tests/systemc/csyn_subdir.py pc_channel pc_channel   # one kernel only
```

`ALLO_DESIGN_TOP` matters for **any area number you report**: a `@df.region` contains the
testbench kernels and their `AlloMem` memories, so synthesizing the region top includes the
harness in the reported area. The flow also has a first-class `configs["synth_top"]` for the
same purpose.

Note cosim does *not* work against a submodule top — SCVerify wraps the design top and the
`input<k>.data → AlloMem` stimulus path disappears. Run measurement and verification
separately.

## Results

[`dev/records/systemc/VERDICTS.md`](../../dev/records/systemc/VERDICTS.md) — per-example
outcome for every `tests/dataflow` design run against the SystemC backend (simulator vs
`mode="csim"`, identical seeded inputs), with the failures categorized by root cause.

## See also

- [`docs/source/backends/systemc.rst`](../../docs/source/backends/systemc.rst) — the published backend page
- [`../../dev/systemc/SYSTEMC_BACKEND.md`](../../dev/systemc/SYSTEMC_BACKEND.md) — the backend guide
- [`../../dev/systemc/DATAFLOW_LINKS.md`](../../dev/systemc/DATAFLOW_LINKS.md) — link types
- [`../../dev/systemc/ALLO_GOTCHAS.md`](../../dev/systemc/ALLO_GOTCHAS.md) — read before writing Allo code
