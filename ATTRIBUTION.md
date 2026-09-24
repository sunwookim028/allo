<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# Attribution

This file records code in this repository that came from someone else, so that
nothing here is mistaken for ours. It is the `main`-branch counterpart of the
file at tag `chia-codesign-final` (`629c2767`), which records that branch's
larger import of the same author's work.

Upstream repository: **https://github.com/kkkaishao/allo**, author **Kai Shao
`<kaishao0582@gmail.com>`**, who also commits as `kaishao
<skkfighting@gmail.com>`, `kaishao <kaishao0582@gmail.com>` and `kkkaishao
<skkfighting@gmail.com>`.

## The TOSA recognizer (2026-09-24)

Source: branch **`act`**, commit **`3c1ad38`**. Destination: `allo/act/`.

TOSA is not an implementation choice of Kai's that we could decline. ACT
*compiles a TOSA program* -- `docs/source/extensions/act.rst` says so of our own
design, and `@I.compute` takes a TOSA DAG -- so the recognizer is the part of
that design we had not built. Our own epilogue vocabulary was `{"relu"}`
(`allo/act/workload.py`) and `("relu", "saturate")`
(`examples/tinytpu/act/spec.py`), which is not standard vocabulary by any
reading.

| # | item | Kai's location | destination |
| --- | --- | --- | --- |
| 1 | prim registry (29 ops, 8 categories) | `allo/exp/dsa/primitive.py:26-95` | `allo/act/primitive.py` |
| 2 | source recognizer (`source_tag`, `_DATA_OPERANDS`, `_source_ins`, `const_elements`) | `search.py:157-227` | `allo/act/recognize.py` |
| 3 | fail-safe policy + quantization neutrality | `search.py:200-227` | `allo/act/recognize.py` |
| 4 | `relu` as bounded `tosa.clamp` + `_FLOAT_MAX` | `search.py:169-175, 229-246` | `allo/act/recognize.py` |
| 5 | `reshape`/`const` transparency (`_canon`, `_LAYOUT_AND_CONST`) | `search.py:109-121` | `allo/act/recognize.py` |
| 6 | `normalize_source` (torch-mlir's 2-D/3-D matmul bracketing) | `search.py:338-389` | `allo/act/recognize.py` |
| 7 | error taxonomy | `errors.py` (whole file) | `allo/act/errors.py` |
| 8 | bespoke-op table | `search.py:95-107` | `allo/act/recognize.py` |

Tests: `tests/act/test_recognizer.py` is ported from his
`tests/dsa/test_recognizer.py`. The cases that ran through his
`ISA.compile_program` are restated against `source_tag` directly, because this
fork took the recognizer and not the matcher it fed; the quantization group (S4)
carries over close to verbatim. `tests/dsa/test_prims.py` was **not** ported:
seventeen of its tests are oracle round-trips through his `ISA`/`@isa.oracle`
machinery, and the registry facts they imply are restated in
`test_recognizer.py` instead.

Every file above keeps its Apache header and carries a provenance comment naming
the author, the repository and the commit.

### What was deliberately left behind

- `instruction_pattern` (`search.py:131-155`) sits in the same block as items
  2-5 but reaches into ACT's `Instruction` and `trace_instruction`. Not
  portable; the seam is there.
- `mapping.py` (1,292 lines), `epoch.py` (480), `mapspace.py` (304) and
  `check.py` (248) -- 2,324 lines that landed in one "checkpoint" commit with no
  test or example reference anywhere in that tree.
- `core.py` and `oracle.py` are the `allov2` dependency itself, which this fork
  does not carry.
- `examples/accelerator/cornell_tpu/microarch.py:155-159` is a literal
  `SyntaxError` on the tip commit.

### What is ours in this import

`allo/act/frontend.py` (TOSA program -> `allo.act.workload.Workload`),
`tests/act/test_tosa_frontend.py`, the integer branch of `_is_relu_clamp`, and
the `Quantization` / `Match` records that carry an op's shift and zero-points
instead of only rejecting them. `allo.act.errors.QuantizationError` is
fork-local. Reasons are in `allo/act/recognize.py` and on
`docs/source/extensions/act.rst`.

## Basis for use

Decided by the project owner on 2026-09-19 and unchanged: this work is used **in
good faith** on the basis that

* it is **public** -- https://github.com/kkkaishao/allo, branches `allov2`,
  `act` and `allo-rtlgen`;
* it is **acknowledged explicitly**, here and in the published documentation
  (https://sunwookim028.github.io/allo/), with the exact upstream branch and
  commit for every imported component; and
* this project adds **substantial work of its own** on top of it.

This records the owner's decision, not a statement from Kai.

## Do not re-publish his branches under a name of ours

`act` and `allo-rtlgen` already exist on Kai's repository. Cherry-pick with
attribution, as above, or reference them read-only; copying a branch wholesale
would fork his work rather than cite it, and would immediately be the stale copy.
