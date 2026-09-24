# devtools/

Introspection and sweep scripts for working *on* the compiler, as opposed to designs built
*with* it. Nothing here is imported by `allo`; these are standalone tools.

All of them assume:

```bash
conda activate allo
export OMP_NUM_THREADS=8
export PYTHONPATH=/home/zsm9/allo_sup    # else the import grabs the installed allo
```

## Pipeline introspection

Print what the compiler produces at each stage, so you can see exactly where a design stops
looking the way you expect. Most default to `examples/stream_producer_consumer.py`.

| Script | Prints |
|---|---|
| `print_ast.py` | compact Python AST of one function (pure stdlib, no allo needed) |
| `print_typed_ast.py` | the AST after Allo's type inference |
| `print_mlir.py` | the frontend MLIR module (or one function of it) |
| `print_sim_rewrite.py` | the simulator's rewritten form |
| `print_stop3_input.py` / `print_stop3_output.py` | IR either side of the stop-3 pass |
| `print_stop4a.py` | the stop-4a stage |
| `dump_ir_passes.py` / `dump_ir_stages.py` | IR after each pass / each stage |
| `dump_stream_backends.py` | one design through **every** backend emitter, each in its own subprocess so a crash in one (Intel HLS segfaults on this design) can't kill the rest |

```bash
python print_mlir.py                 # whole module
python print_mlir.py producer_0      # just that function
python dump_stream_backends.py       # all backends
python dump_stream_backends.py systemc
```

## Build and test helpers

| Script | Does |
|---|---|
| `rebuild.sh` | rebuild the MLIR emitter (`ninja` in `mlir/build`) |
| `sysc_env.sh` | set up the SystemC/Catapult environment |
| `cosim_sweep.sh` | run cosim across the dataflow suite |
| `run_empty_test.sh` / `test_empty_smoke.py` | minimal smoke test |
| `scpatch_cosim2.py` | patch generated cosim projects |

## Note

These scripts hardcode `/home/zsm9/allo_sup`. That is deliberate — it defeats the
two-checkouts trap where `import allo` silently picks up the installed `/home/zsm9/allo`
instead of this working tree. If you move the checkout, update the paths.
