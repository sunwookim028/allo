# Fast-feedback measurements, 2026-09-25

The raw outputs behind "the fast gate is 41 s and mapspace_report 42 s" (commit
`3e8db6fe`), from `~/work/measure` on the Vitis host, harvested when that
scratch directory was retired on 2026-09-29.

- `before-loop/`: the loop and gates before the feedback ladder.
- `after-full/`: after it, including the `spad_zero` candidate's acceptance
  and the control it was measured against.

Not kept: the per-worker `spec/` snapshots (copies of `isa_spec.json`), the
generated sandbox, and the `.log` files.
