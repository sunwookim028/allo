@AGENTS.md

# CHIA co-design — agent notes

`AGENTS.md` above is the ACT fork's general guidance. This file carries what is
specific to the co-design work, including the places the two disagree.

## Start here

- `CODESIGN.md` — the frontmatter: key ideas, the claim register (C1–C8
  deterministic, S1–S3 stochastic), setup from zero, budgets, repository map.
- `notes/CHIA_CHECKPOINT.md` — version pins, disk footprint, outstanding work.
- `examples/accelerator/tinytpu/` — the design under optimization, plus the
  agent that searches it (`chia_agent/`).

## Verify before you claim

```bash
./scripts/claims.sh --fast    # no synthesis, no API: the pure-software claims
./scripts/claims.sh           # adds Vitis HLS: QoR claims and replayed results
./scripts/claims.sh --full    # adds the backend scaffolds and one billed agent call
```

Each step prints its own wall time, so the numbers in the docs stay honest. The
runner refuses to start when the conda environment imports `allo` from a
different checkout: an editable install resolves through a meta-path finder that
outranks `PYTHONPATH`, so a green result could otherwise come from someone
else's tree.

A number produced by an agent is not a result until
`examples/accelerator/tinytpu/verify_variant.py` replays its recorded diffs into
a throwaway worktree, re-synthesizes, and re-derives it.

## Environment

Copy `scripts/chia.env.example` to `chia.env` at the repo root and source it —
every script here reads it, and `TINYTPU_ENV` names the conda environment
holding the built `allo`.

## Where this branch's setup differs from AGENTS.md

- **OR-Tools**: `CMAKE_PREFIX_PATH=$PWD/externals/circt/ext` (9.5, fetched by
  `scripts/build-circt.sh`), not `~/.local/share/or-tools-9.15`.
- **Vitis**: sourced directly from `/opt/xilinx/Vitis_HLS/2023.2/settings64.sh`.
  C-synthesis needs no licence and no container; `docker/run-vitis.sh` is for
  the `hw_emu`/`hw` builds this flow does not run.
- **Compiler**: build with gcc. clang's `enable_if` on `llvm::StringLiteral`
  breaks the `std::optional<StringLiteral>` conversion the HLS emitters need.
- `export CMAKE_POLICY_VERSION_MINIMUM=3.5` — cmake 4 rejects OR-Tools 9.5's
  fetched dependencies.

## Constrain what an agent may write

`examples/accelerator/tinytpu/chia_agent/spec_policy.py` rejects spec edits the
evaluator would execute on import — an agent once wrote `open(__file__, "w")` at
module level into `examples/accelerator/tinytpu/isa.py`, which the scorer then
ran. Any new edit path goes behind that gate.

## Lineage

This branch descends from Kai's ACT fork, not from `main`. `main`'s
`allo/dataflow.py`, `allo/ir/`, and `allo/passes.py` do not exist here, and a
trial merge produces 26 conflicts. The two lineages are maintained separately;
the fork-wide branch layout is in `notes/MAINTENANCE_CHECKLIST.md` on `main`.
