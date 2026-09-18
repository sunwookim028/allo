# Attribution

**Most of this branch is Kai Shao's work, imported unmodified.** Of the 539
commits `chia-codesign` carries that `main` does not, **503 are Kai's and 36 are
Sunwoo Kim's**. This file records exactly which parts came from where, so that
nothing here is mistaken for ours.

Upstream repository for all three imports: **https://github.com/kkkaishao/allo**
(remote `kai`), author **Kai Shao `<kaishao0582@gmail.com>`**, who also commits as
`kaishao <skkfighting@gmail.com>`, `kaishao <kaishao0582@gmail.com>` and
`kkkaishao <skkfighting@gmail.com>`.

## What was imported

| Component | Paths | Lines | From | Merged here |
| --- | --- | --- | --- | --- |
| **CIRCT RTL code generation** | `allo/backend/rtl/`, `mlir/lib/allo/Microarch/`, `mlir/lib/allo/Scheduling/`, `tests/rtl/`, `externals/circt` @ `af5369d`, `scripts/build-circt.sh` | ~54,750 | branch **`allo-rtlgen`**, commit **`9e2d5716`** ("Tighten comments across the post-upstream changes", 2026-08-29) | `882f7dd6` (tag `tinytpu-rtlgen-base`) |
| **ACT / DSA compiler flow** | `allo/exp/dsa/`, `mlir/.../AlloISAOps.cpp`, `AlloISATypes.cpp`, `Conversion/LowerInstructions.cpp`, `tests/dsa/` | ~17,350 | branch **`act`**, commit **`3c1ad38d`** (2026-08-19) | `29cb1d99` |
| **Core re-architecture** | `allo/compiler/`, `allo/lang/`, `allo/operators/`, `allo/schedule/` | ~13,130 | branch **`allov2`**, commit **`b22ed847`** (2026-07-10) | — |

**The RTL generator is verbatim.** `git merge-base kai/allo-rtlgen chia-codesign`
is `9e2d5716`, a commit on Kai's branch, and

    git diff 9e2d5716..chia-codesign -- allo/backend/rtl mlir/lib/allo/Microarch \
                                        mlir/lib/allo/Scheduling tests/rtl

is **empty**. Not one line of it is ours. The ACT flow differs by 8 insertions
and 60 deletions across 3 files.

## What is ours

| Component | Paths | Author |
| --- | --- | --- |
| CHIA agent search | `examples/accelerator/tinytpu/chia_agent/` (1,461 lines), `chia_runs/`, `CODESIGN.md`, `notes/CHIA_CHECKPOINT.md`, `notes/VALIDATION_2026-09-07.md`, `scripts/chia.env.example`, `scripts/claims.sh` | Sunwoo Kim |
| The TinyTPU design this branch searches over | `examples/accelerator/tinytpu/` — `microarch*.py`, `isa.py`, `feedback.py`, `oracle.py`, `ppa.py`, `synth.py`, `verify.py`, `bench/`, `probes/` | Sunwoo Kim (25 commits), Kai Shao (6) |

`examples/accelerator/tinytpu/rtlgen/` is **jointly derived**: it is CIRCT output
produced by Kai's generator running on Sunwoo's `microarch.py`, so it is cited to
both rather than assigned to either.

## Do not copy this work into a fork-local branch

`allo-rtlgen` and `act` **already exist as branches on Kai's repository**, and
this branch's snapshot of `allo-rtlgen` is **23 commits behind** its tip
(or-tools work, cross-region unit sharing and lit tests we do not have).
Re-publishing them under a name of ours would fork Kai's work rather than cite
it, and would immediately be the stale copy. Reference `kai/allo-rtlgen` and
`kai/act` read-only instead.

## Open: consent is not recorded anywhere

Both merges (`882f7dd6`, `29cb1d99`) are plain merge commits with no co-author
trailers, and neither `CODESIGN.md` nor `notes/CHIA_CHECKPOINT.md` states the
terms under which this code was taken. **Nothing in this tree records that Kai
agreed to the import, or on what basis.** That is a question for Kai and should
be settled before any of this is published, submitted, or landed on `main` —
particularly efforts (2) and (3), which are substantially his.
