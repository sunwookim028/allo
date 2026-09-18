# Does the free-running-loop guard fix `pe_wire`? No.

A negative result for `notes/ALLO_SHORTCOMINGS.md` #22. The emitter rewrites a
kernel's outermost loop to `while (1)` under `__SYNTHESIS__` when its induction
variable is unused. The hypothesis was that this rewrite is what makes `pe_wire`
wrong in RTL. `guard.patch` stops the rewrite for any loop that reads a `Wire`.

## Result (Catapult 2024.2, Xcelium 24.03, 2026-09-18)

| netlist | `pe_wire`, 18 pacings | cycles at unit pacing |
| --- | --- | --- |
| `rtl_base` (emitter as is) | FAIL 8/8 at all 18 | 20 |
| `rtl_guard` (with `guard.patch`) | FAIL 8/8 at all 18 | 20 (identical at every pacing) |

- **Emitted code:** exactly as intended. `acc_0` loses its free-running loop,
  `mul_0` keeps its own, and `pe_stream`/`pe_channel` come out byte-identical
  with and without the patch.
- **Controls:** `rtl_guard/pe_stream` and `rtl_guard/pe_channel` pass all 36
  pacings, and `BREAK_DATA` turns both red.
- **Why the guard doesn't help:** `acc_0` still advances on its consumer's
  handshake and never on its producer. The Wire boundary has no
  synchronisation, with or without `while (1)`.

## Replaying from the netlists (no Allo build needed)

    cd ..   # examples/systemc_rtlsim
    RTLDIR=$PWD/guard_experiment/rtl_guard ./run_mulacc_xrun.sh pe_wire
    RTLDIR=$PWD/guard_experiment/rtl_base  ./run_mulacc_xrun.sh pe_wire
    RTLDIR=$PWD/guard_experiment/rtl_guard ./run_mulacc_xrun.sh pe_stream -d CONNECTIONS_FIFO
    RTLDIR=$PWD/guard_experiment/rtl_guard ./run_mulacc_xrun.sh pe_channel

The lockstep positive control (`-d LOCKSTEP`) is for the August netlists only,
because it taps an internal signal (`v8_and_cse`) that these netlists don't have.

## Regenerating the netlists

1. Build `choonsik1/allo:SystemC-emitter` (`72c70dcb` or later, LLVM 6b09f739).
2. Run `ALLO_ROOT=<that worktree> python run_sc.py emit.py out_base`.
   `run_sc.py` makes Python import that worktree's `allo` even when a conda env
   has an editable install of another one.
3. Apply `guard.patch`, rebuild, and emit to `out_guard`.
4. In each project directory, run
   `MGLS_LICENSE_FILE=1717@en-license-05.coecis.cornell.edu catapult -shell -file run.tcl`.
   The netlist lands in `Catapult/<design>.v1/rtl.v`.
