# SystemC / Catapult examples

Runnable Allo dataflow designs that go through `target="systemc"`. Each one is a small,
self-contained design plus the commands to simulate, synthesize and cosim it.

## Start here

**[`pc_channel.py`](pc_channel.py)** — the minimal example: one producer, one consumer, one
`Channel[int32, valid_ready]`. Small enough to read in a minute, and it exercises the whole
path.

```bash
conda activate allo
export OMP_NUM_THREADS=8
export PYTHONPATH=/home/zsm9/allo_sup

python pc_channel.py mlir      # the frontend MLIR
python pc_channel.py systemc   # the generated SystemC
python pc_channel.py csim      # build + run csim, check B == A
```

## The examples

| File | Design | Exercises |
|---|---|---|
| `pc_channel.py` | producer → consumer over a handshake channel | `Channel[valid_ready]`, the minimal path |
| `stream_boundary.py` | region boundary arrays as streams | `Stream`, boundary I/O |
| `systolic_chain.py` | chained PEs | multi-kernel stream chains |
| `tiled_systolic.py` | tiled GEMM | stream-output, self-synchronizing termination |
| `mem_port_reverse.py` | reversed array access | random-access memory ports (`AlloMem`) |
| `mem_port_scatter.py` | scattered writes | memory-port stores, replica write-merge |

`.cpp` files next to them are **generated** SystemC kept for reference, not sources.

## Synthesis and cosim

`csynth` needs a build subdirectory — running Catapult in the directory that holds
`kernel.cpp` degrades `Connections::In`/`Out` ports to raw `sc_signal`s (CIN-124 / SCHD-30).
[`csyn_subdir.py`](csyn_subdir.py) works around it:

```bash
python csyn_subdir.py pc_channel pc_channel          # synthesize the whole region
ALLO_DESIGN_TOP=producer_0 python csyn_subdir.py pc_channel pc_channel   # one kernel only
```

`ALLO_DESIGN_TOP` matters for **any area number you report**: a `@df.region` contains the
testbench kernels and their `AlloMem` memories, so synthesizing the region top includes the
harness in the reported area. The flow also has a first-class `configs["synth_top"]` for the
same purpose.

Note cosim does *not* work against a submodule top — SCVerify wraps the design top and the
`input<k>.data → AlloMem` stimulus path disappears. Run measurement and verification
separately.

## Results

[`VERDICTS.md`](VERDICTS.md) — per-example outcome for every `tests/dataflow` design run
against the SystemC backend (simulator vs `mode="csim"`, identical seeded inputs), with the
failures categorized by root cause.

## See also

- [`../../docs/SYSTEMC_BACKEND.md`](../../docs/SYSTEMC_BACKEND.md) — the backend guide
- [`../../docs/DATAFLOW_LINKS.md`](../../docs/DATAFLOW_LINKS.md) — link types
- [`../../notes/ALLO_GOTCHAS.md`](../../notes/ALLO_GOTCHAS.md) — read before writing Allo code
