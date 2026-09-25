# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Score one TinyTPU-isa candidate. FROZEN: the agent can neither edit nor import it.

A candidate is exactly the files `chia_agent/design.py` names editable --
`microarch_isa.py`, `isa_dsl.py`, the ISA, the architecture, the assembler, the
programs and the eight units under `ip/units/` -- sitting in a spec directory
that mirrors the package. Everything else the score depends on is taken from
git at `FROZEN_REF`, never from the working tree, so no edit anywhere on disk
can move the objective:

    frozen (from git @ FROZEN_REF)          editable (from --spec-dir)
    ------------------------------          --------------------------
    tinytpu/cosim.py                        tinytpu/microarch_isa.py
      the testbench generator,              tinytpu/isa_dsl.py
      the numpy golden reference,           tinytpu/ip/isa.py
      every Vitis TCL setting               tinytpu/ip/tinytpu.py
    tinytpu/shapes.py                       tinytpu/ip/assembler.py
    tinytpu/bench_isa.py                    tinytpu/ip/programs.py
    tinytpu/stress_isa.py                   tinytpu/ip/units/*.py  (eight)
    tinytpu/gen_isa.py                      tinytpu/isa_spec.json
      holds the spec and BOTH its             the ISA as data
      generated artefacts to each            tinytpu/isa_encoding.py
      other, whatever they say                 generated from the spec
    tinytpu/kpn_model.py                    tinytpu/isa_ref.py
    tinytpu/ip/params.py                      the ISA as numpy
      (the parameter set's invariants)
    tinytpu/ip/__init__.py, ip/units/__init__.py, ip/reduce.py,
    tinytpu/ip/units/reduction_tree.py   (the package, and the reduce IP it
                                          imports -- a different design)
    chia_agent/gate_runner.py     (runs each check, vouches for its verdict)
    chia_agent/mapspace.py        (THE MAPPER: its enumerator, its selection
                                   rule, and its objective)
    chia_agent/codesign_gate.py   (the exhaustive mapspace enumeration)
    chia_agent/codesign_cosim.py  (binds the mapper's pick into cosim.py)
    examples/__init__.py

With `--codesign` two more stages run, and they are the co-design loop:

3. **mapspace** -- `codesign_gate.py` enumerates the WHOLE mapspace for each
   scored shape against this candidate's hardware, records how many nests
   became encodable and which constraint refused the rest, proves every
   survivor correct against `isa_ref.run`, and requires the canonical nest to
   re-emit the candidate's own `gemm_program` word for word. The inner loop is
   exhaustive, not agentic: the agent proposes hardware, and enumeration -- not
   the agent -- answers what that hardware can run. A non-exhaustive inner
   search would make this a comparison of search quality instead of hardware.
4. **score** -- `codesign_cosim.py` runs main's frozen `cosim.py` with
   `gemm_program` bound, in frozen code, to the nest the mapper chose. The
   reported objective is a PAIR: cosim cycles per shape and the csynth resource
   estimate, never collapsed into one number. No modelled cycle count is ever
   reported; the only proxy in the loop ranks nests inside one candidate, in
   `mapspace.py`, and never leaves it.

WHY THE ISA IS EDITABLE, AND WHAT KEEPS THE ORACLE HONEST. `isa_spec.json`
used to be frozen, with `isa_encoding.py` (generated from it) and `isa_ref.py`
(built on that) frozen beside it, because `isa_ref` is what the stress gate
compares random programs against: a candidate that can rewrite its own
reference model can weaken the rule and score strictly better. That froze the
instruction set, which is half of a co-design space. Three things replace the
freeze, and none of them is a file the candidate can reach:

* **`run.py --verify`, the PyTorch oracle, is now a GATE and runs first.** It
  is the only check in this system with something OUTSIDE this repository on
  one side -- `torch.nn.Linear`'s own forward on the model's real weights, with
  the machine's epilogue applied in torch -- and nothing a candidate writes can
  move it. A byte that differs is a refusal (`gate:pytorch`). Its corpus is
  thin and the gate says so in its own output: at MAXDIM=16 it is two MLPs of
  16x16 layers over 2,688 bytes, eight input/weight draws each, and no shape
  the GEMM sweep does not already cover. A gate is only as strong as its
  corpus, and this one buys an oracle, not breadth.
* **`gen_isa.py --conform` is a gate on the candidate's OWN spec.** It is
  frozen, and it holds the spec, `isa_encoding.py`, the design's bit slices,
  the assembler, the reference model and the emitted HLS to each other --
  internal consistency, which stays valuable when the spec moves, and which no
  amount of editing the spec can satisfy dishonestly: the reference model may
  not import an opcode number or a field position from the design
  (`check_reference`), so it still names operands the way the SPEC names them.
* **The GEMM goldens were never `isa_ref`.** `bench_isa.py`, `stress_isa.py`
  and `cosim.py`'s testbench compute their own numpy goldens and every one of
  the three is frozen. `isa_ref` is the reference for RANDOM PROGRAMS only, so
  weakening it cannot make a wrong GEMM pass.

`allo/actions.py` is NOT the answer to this and was not used for it: it is
another model of the same ISA inside this repository, so it has the same
failure mode as `isa_ref` -- it would be an editable referee.

The two tiers:

1. **gate** -- `bench_isa.py` (the published [-4, 4] setup) and main's
   `stress_isa.py` (492 runs at 476a70d8: full-range/corner/boundary operands, all 64
   shapes, prefilled C compared in full, vector and random programs, many
   invocations of one build) must both pass. Functional, on Allo's simulator,
   ~12 s. Each runs under `gate_runner.py`, and the verdict is its
   `CHIA-GATE <check> OK <nonce>` line -- a fresh nonce per run, handed over on
   stdin before the candidate is imported -- never the check's own printed
   `ALL EXACT` / `STRESS OK`, which the candidate's code could print itself.
   Then the parametricity gate: `param_check.py` rebuilds the candidate at
   TPU_MAXDIM=8, TPU_MAXDIM=12 and TPU_T=8/TPU_MAXDIM=32 (so T is varied too)
   and requires it to honour the parameters and be exact at
   every GEMM shape of that configuration (and on random programs), so a win
   that only exists at the scored T=4 / MAXDIM=16 is rejected (`gate:param`).
   The policy also refuses a literal T/MAXDIM and a net loss of more than 15
   comment/docstring lines against the frozen ref.
2. **score** -- `cosim.py` (Vitis HLS 2023.2 csynth + xsim C/RTL cosim), one
   testbench per shape, each bit-exact against numpy. The score is the SUM of
   cosim cycles over the requested shapes. It is an RTL measurement.

The memory model is not the candidate's to choose: every `TPU_*` environment
variable is scrubbed before `cosim.py` runs, except the scored configuration
and the project path, so `-m_axi_latency` stays at its default 0 and
`-random_stall` stays off, and the generated `kernel.cpp` / TCL are checked for
interface-latency overrides afterwards.

THE SCORED CONFIGURATION IS A CANDIDATE'S TO PROPOSE, within a stated envelope.
It used to be `SCORED = {T: 4, MAXDIM: 16}`, pinned in the environment and
re-checked afterwards, so the loop searched the implementation of eight units at
ONE architecture -- microarchitecture search, not co-design. A candidate now
declares `CHIA_CONFIG = {"T": ..., "MAXDIM": ...}` at module level in its own
`microarch_isa.py`; the declaration is read as a LITERAL (`resolve_config`,
before the tree is composed and before any candidate code runs), and every
stage then runs pinned to it, so the whole evaluation is one configuration and
`check_invariants` still requires the build to honour it. A candidate that
declares nothing is scored at `DEFAULT_CONFIG` -- T=4, MAXDIM=16, the published
row -- so no existing candidate, control or published number moves.

Four things refuse a proposal, each before anything expensive:

* **the area proxy's fitted envelope.** `area_proxy.py` is fitted to seven
  committed Design Compiler runs and has exactly ONE calibration point below
  MAXDIM=64. Outside the envelope those runs span it would be extrapolating,
  so it REFUSES TO PRICE the configuration rather than returning a number
  nobody can defend (`area_proxy.envelope_refusals`, stage `config`). The DC
  runs that would widen it are named in that function.
* **both computed MAXDIM ceilings.** `(MAXDIM // T) * MAXDIM <= 2048` for the
  operand layout's addressing and `MAXDIM**3 // T**2 + MAXDIM**2 // T <= 32767`
  for a cubic GEMM's header count: 88 and 76 at T=4, 128 and 120 at T=8.
  Derived from the FROZEN `isa_encoding.MAXDIM_CEILINGS`, never typed in here,
  and the refusal names which ceiling and its value.
* **`QD` must be 16.** Depth 8 deadlocks legal programs -- three of the tiled
  family never complete in cosim (limitations item 24) -- so a proposal of 8 is
  refused rather than allowed to hang for the length of the timeout.
* **`ip/params.py`'s invariants**, `T >= 4` and `MAXDIM % T == 0`.

Independent checks on the result, none of which trust the candidate's process:
T and MAXDIM are the proposed ones (the Gemmini comparison's matched-array
premise holds for its own row, T=4 MAXDIM=16, and a candidate that moves off it
is no longer comparable with that row -- `docs/source/designs/
gemmini_comparison.rst`); each
requested shape reported exactly once; each shape's own cosim log carries
`mismatches = 0 / M*N` and a PASS; the cycle count agrees with that log's
simulated time; the csynth estimated clock meets the 3.33 ns target. The work
directory is wiped first, so a stale report can never be read as a result.

Every process that imports the candidate runs in a bubblewrap sandbox when
`bwrap` is available: the whole filesystem read-only, the evaluation tree
read-only too, only the work directory and a private `/tmp` writable, in its own
PID namespace so that a kill takes every descendant with it. That is the answer
to "the candidate's import-time code rewrites the gate": a test mutant narrowed
the datapath AND rewrote the tree's stress gate through `numpy.savetxt`, which
the policy did not know, and passed the gate. After every stage the tree is
also compared byte-for-byte with what was composed (no file changed, none
added) and the checkout's tracked files with how they were, so a host without
`bwrap` still catches the tamper after the fact.

Prints one JSON line (last line of stdout) and exits 0 iff the candidate passed.

    python evaluate.py --spec-dir DIR --work DIR [--shapes 4x4x4,16x16x16] [--gate-only]
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path, PurePosixPath

AGENT_DIR = Path(__file__).resolve().parent
REPO = AGENT_DIR.parents[2]
PKG = "examples/tinytpu"

sys.path.insert(0, str(AGENT_DIR))
from design import EDITABLE, FROZEN_DESIGN  # noqa: E402

#: Frozen files are read from this commit (resolved to a hash per run), i.e.
#: from what is COMMITTED, never from the working tree. Only a person, in a
#: commit, can change them.
FROZEN_REF = os.environ.get("CHIA_FROZEN_REF", "HEAD")
#: The main commit the frozen ref is based on. The design's own evaluator
#: (DESIGN_EVALUATOR below) must be byte-identical there, so a branch -- or a
#: run -- cannot drift from how main measures and verifies the design.
#:
#: DERIVED, never typed in. It used to be a hand-written hash, and it went
#: stale four times in five days (476a70d8 -> acb080bd -> 39ba9aaa ->
#: f59a65f6, and then the whole design moved from
#: examples/accelerator/tinytpu_vitis/ to examples/tinytpu/ and f59a65f6 no
#: longer had the paths at all). Every time, compose() refused EVERY candidate
#: at stage `setup` and the message blamed the design rather than the pin. A
#: constant nobody notices going stale is the defect, not its value.
#:
#: The merge-base of the frozen ref with main is what the constant's own
#: comment always said it was -- "main's commit this branch is based on" -- and
#: it cannot go stale. On main it is the frozen ref itself, so the check is
#: vacuous there and says so; on a branch it bites, which is where it was ever
#: doing work. `CHIA_MAIN_BASE` names a different commit deliberately (a
#: held-out ref pins to its own; `chia_abstraction/heldout.py` sets it).
MAIN_REF = os.environ.get("CHIA_MAIN_REF", "origin/main")


def main_base(ref: str) -> str:
    """`CHIA_MAIN_BASE` if set, else the merge-base of `ref` and main."""
    pinned = os.environ.get("CHIA_MAIN_BASE")
    if pinned:
        return resolve_ref(pinned)
    for main in (MAIN_REF, "main"):
        out = subprocess.run(["git", "merge-base", ref, main], cwd=REPO,
                             capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    raise Reject("setup", f"no merge-base of {ref} with {MAIN_REF} or main; set "
                          f"CHIA_MAIN_BASE to the main commit this ref is based on")


#: What MEASURES or JUDGES the design, and therefore may not move: the
#: testbench and its golden reference (`cosim.py`), the two functional gates and
#: the numpy goldens they compute themselves (`bench_isa.py`, `stress_isa.py`),
#: the scored shapes, the deadlock diagnosis, and `gen_isa.py` -- which holds
#: the ISA spec and its generated artefacts to each other and to the design.
#:
#: `isa_spec.json`, `isa_encoding.py` and `isa_ref.py` USED TO BE HERE. They
#: are the instruction set, and freezing them froze half of the co-design
#: space; they are in `design.EDITABLE` now. What replaces the freeze is in the
#: module docstring: an oracle outside this repository (`run.py --verify`),
#: `gen_isa.py --conform` on the candidate's own spec, and the fact that every
#: GEMM golden was always computed by a frozen file and never by `isa_ref`.
#: `gen_isa.py` STAYS here, and staying is what makes the rest safe: it is the
#: only thing that can say "your reference model and your spec disagree", and
#: `check_reference` is what stops a reference model from simply importing the
#: design's own opcode numbers and agreeing with it by construction.
DESIGN_EVALUATOR = [f"{PKG}/{f}" for f in (
    "cosim.py", "bench_isa.py", "stress_isa.py", "kpn_model.py",
    "shapes.py", "gen_isa.py")]
GATE_RUNNER = f"{PKG}/chia_agent/gate_runner.py"
PARAM_CHECK = f"{PKG}/chia_agent/param_check.py"
#: The area proxy, and the workload suite that supplies the model term of the
#: objective. Frozen for the same reason the mapper is: a candidate may move
#: the design, never the thing that prices it or the workload it is priced on.
AREA_PROXY = f"{PKG}/chia_agent/area_proxy.py"
WORKLOAD_SUITE = [f"{PKG}/workloads/{f}" for f in
                  ("__init__.py", "models.py", "extract.py", "run.py",
                   "burst.py", "gate.py", "claims.json")]
#: The co-design loop's frozen half: the mapper (its enumerator, its selection
#: rule and its objective), the mapspace gate, and the cosim driver that binds
#: the mapper's chosen program into main's frozen `cosim.py`. The agent edits
#: the hardware and its encoding; it may not move the thing that scores it.
CODESIGN = [f"{PKG}/chia_agent/{f}" for f in
            ("mapspace.py", "codesign_gate.py", "codesign_cosim.py")]
#: The design's own frozen machinery -- `design.FROZEN_DESIGN`, the one
#: definition -- taken from git like every other frozen file.
#: Entry points the evaluation actually executes. Everything they import,
#: transitively and first-party, is measurement apparatus and is frozen with
#: them -- see `import_closure`.
ENTRY_POINTS = [*DESIGN_EVALUATOR, AREA_PROXY, *WORKLOAD_SUITE, GATE_RUNNER,
                PARAM_CHECK, *CODESIGN]
#: First-party MODULE PREFIXES: a module under one of these is measurement
#: apparatus, so it must be in the tree. Dotted prefixes and not top-level
#: directories, because `act` -- the mapper -- moved from `act/` to
#: `allo/act/` while this was being written, and because the rest of `allo`
#: is the compiler: it comes from the environment the gate runs in, and
#: `CHECKOUT_WATCH` guards it against tampering instead of composing it.
#: Everything else (numpy, torch) is the environment's too.
FIRST_PARTY = ("examples", "act", "allo.act")
FROZEN_LITERAL = [
    "examples/__init__.py",
    *DESIGN_EVALUATOR,
    AREA_PROXY,
    *WORKLOAD_SUITE,
    *[f"{PKG}/{rel}" for rel in FROZEN_DESIGN],
    GATE_RUNNER,
    PARAM_CHECK,
    *CODESIGN,
]
#: The parametricity gate: configurations the candidate is rebuilt at and must
#: be exact at (param_check.py), besides the one it is scored at. 12 is
#: deliberately not a power of two, and the third case VARIES T.
#:
#: This list used to say "MAXDIM only: main's design supports T=4 alone (it
#: fails check_program at TPU_T=8)". That was FALSE. Measured on main,
#: `TPU_T=8 TPU_MAXDIM=32 param_check.py` prints `PARAM OK: 408/408 runs
#: exact` and `TPU_T=8 TPU_MAXDIM=32 bench_isa.py 32 32 32` prints
#: `ALL EXACT`. What is limited is the HARNESS, and the limit is MAXDIM/T >= 3,
#: not T == 4: three test-program generators address column block 2, which
#: exists only at that ratio, so at T=8 with MAXDIM=16 nine of 24 random seeds
#: cannot be generated and param_check refuses for want of programs rather
#: than for a wrong answer. MAXDIM=32 gives ratio 4 and every seed generates.
PARAM_CONFIGS = [{"TPU_MAXDIM": "8"}, {"TPU_MAXDIM": "12"},
                 {"TPU_T": "8", "TPU_MAXDIM": "32"}]


def param_configs(cfg):
    """`PARAM_CONFIGS`, moved onto the configuration the candidate proposed.

    DERIVED, because the list above is only right for T=4. The harness limit
    the comment records is MAXDIM/T >= 3 for three of the fuzz generators, and
    it bites differently at each T: at T=4 MAXDIM=8 (ratio 2) three of 24 seeds
    cannot be generated, which `MIN_FUZZ` tolerates, while at T=8 MAXDIM=16
    (also ratio 2) NINE cannot and `param_check` refuses for want of programs.
    So a ratio-2 case is only used at T=4, where it is measured to pass; above
    it the two cases are ratios 3 and 4. A configuration equal to the scored
    one is dropped, because rebuilding at the point already scored checks
    nothing.

    The cross-T case is kept at every T, because varying T is the whole reason
    it exists: at T=4 it is the shipped T=8/MAXDIM=32 (ratio 4), and at any
    other T it is T=4/MAXDIM=16 (ratio 4), which is also the published row.
    """
    t, maxdim = int(cfg["TPU_T"]), int(cfg["TPU_MAXDIM"])
    ratios = (2, 3) if t == 4 else (3, 4, 5)
    out = [{"TPU_MAXDIM": str(r * t)} for r in ratios if r * t != maxdim][:2]
    cross = ({"TPU_T": "8", "TPU_MAXDIM": "32"} if t == 4
             else {"TPU_T": "4", "TPU_MAXDIM": "16"})
    return out + [cross]


#: The scored configuration a candidate that proposes NOTHING is measured at:
#: the published row. SET explicitly in every evaluation rather than assumed to
#: be the design's default -- the default moved to MAXDIM=64 while
#: check_invariants still demanded 16, so every candidate died at stage
#: `invariant`, and a pin on someone else's default is not a pin.
#: `reproduce.sh` exports the same point for the same reason.
DEFAULT_CONFIG = {"TPU_T": "4", "TPU_MAXDIM": "16", "TPU_QD": "16"}
#: Kept under its old name because `accept.py`, `control.py` and
#: `test_harness.py` all quote it as the configuration of the published row,
#: which it still is. It is no longer the configuration of every candidate.
SCORED = DEFAULT_CONFIG
#: The module-level name a candidate declares its configuration with, in its own
#: `microarch_isa.py`. A literal dict, read by `ast.literal_eval` and never
#: executed: reading a proposal must not run candidate code, because the whole
#: point of reading it early is to refuse an unpriceable configuration before
#: anything imports the design.
CONFIG_DECL = "CHIA_CONFIG"
#: Parameters a candidate may propose. `DMA_WORDS` is deliberately NOT here:
#: it is the burst-widening axis, the loop's own accepted win moved it, and it
#: is reached by editing the design rather than by declaring a number.
PROPOSABLE = ("T", "MAXDIM", "QD")
#: `QD` may be proposed and there is exactly one legal value. Depth 8 deadlocks
#: legal programs -- three of the ten tiled programs never complete in cosim,
#: limitations item 24 -- so a proposal of 8 is REFUSED here rather than left to
#: hang for `COSIM_TIMEOUT` and be reported as a timeout. It is in `PROPOSABLE`
#: so that the refusal names the reason instead of the parameter being silently
#: unavailable.
QD_REQUIRED = 16
#: The model term of the objective: PyTorch MLPs, layer by layer, through the
#: same one csynth as the GEMM shapes.
#:
#: Why a model at all. The same optimisation is worth 4.3-7.0 % on GEMM shapes
#: and 25-34 % on multi-layer models (docs/source/designs/workload_suite.rst):
#: a model does not make the problem bigger, it makes it LONGER, so every layer
#: repays a fixed cost that one large GEMM amortises away. A loop scored only
#: on GEMM shapes ranks changes on the workload class least sensitive to the
#: cost they remove.
#:
#: Why these two. Cost. Every model costs one cosim PER LAYER, so these two are
#: six, and `mlp_small` -- `confirmed` on RTL at this configuration, and the
#: model with the LARGEST measured divergence from the GEMM table (34.5 %
#: against 4.3-7.0 %) -- would be two more. It is the first thing to add if the
#: term earns its cost. `mlp_wide` cannot be added: it is `correct` and not
#: `confirmed` precisely because no RTL run of it has ever completed
#: (limitations item 24). `mlp_bias` is in the suite to NOT map.
#:
#: Their published cycles -- 1,150 and 2,117 -- are MAXDIM=64 measurements and
#: are NOT the control here. The control is measured per run, like the GEMM
#: one: `accept.py` takes the measurement, it does not look it up.
SCORED_MODELS = ("mlp_tiny", "mlp_deep")
#: The configuration the MODEL term is measured at -- MAXDIM=64, not `SCORED`'s
#: 16. MEASURED, not assumed: at T=4 MAXDIM=16 QD=16 the burst widening is
#: worth 0 cycles on every layer of both models (861 -> 861, 1,530 -> 1,530,
#: layer by layer, to the cycle), against 287 and 583 on the same two models at
#: MAXDIM=64. `dev/records/tinytpu/model-term-maxdim-20260924.rst`.
#:
#: At MAXDIM=16 a DRAM row is 4 packed words instead of 16, so every operand
#: burst is four times shorter and the per-layer prologue covers all of it. The
#: relationship the model term exists for does not just weaken there, it
#: INVERTS: at MAXDIM=16 the GEMM shapes see the widening (run 1 measured
#: 0/0/-42/-59/-59) and the models do not. A model term at MAXDIM=16 would be
#: the LESS burst-sensitive of the two terms, which is the opposite of the
#: reason for adding it.
#:
#: The GEMM control stays at `SCORED` because that is the published row. Two
#: configurations means TWO csynths per candidate, and that is the price of the
#: term being worth anything.
SCORED_MODEL_ENV = {"TPU_T": "4", "TPU_MAXDIM": "64", "TPU_QD": "16"}


def model_env(cfg):
    """`SCORED_MODEL_ENV`, moved onto the candidate's own T.

    MAXDIM stays 64 for the measured reason in the constant's comment -- at 16
    the widening is worth 0 cycles on every layer of both models, so a model
    term there would be the LESS burst-sensitive of the two terms -- and 64 is
    the top of the area proxy's fitted envelope, so it is also the largest
    MAXDIM any configuration is priced at. T follows the candidate, because a
    model term measured on a different array than the one being scored is a
    measurement of a machine nobody proposed.
    """
    return {"TPU_T": cfg["TPU_T"], "TPU_MAXDIM": "64",
            "TPU_QD": str(QD_REQUIRED)}
#: What in the checkout itself the evaluation depends on: the `allo` package
#: (on PYTHONPATH), and this directory's evaluator, policy and design.
CHECKOUT_WATCH = ["allo", "examples/__init__.py", PKG]
#: The five benchmark shapes come from `{PKG}/shapes.py`, the one definition,
#: loaded BY PATH: this harness runs in a conda env that has no `allo`, so it
#: cannot import any design module, and `shapes.py` imports nothing so that it
#: can. `accept.py` and `test_harness.py` take `ALL_SHAPES` from here.
def _load_shapes():
    path = REPO / PKG / "shapes.py"
    spec = importlib.util.spec_from_file_location("tinytpu_shapes", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


SHAPES = _load_shapes().SHAPES
ALL_SHAPES = [f"{M}x{K}x{N}" for (M, K, N) in SHAPES]
SEARCH_SHAPES = ["4x4x4", "16x16x16"]
TARGET_NS = 3.33
#: The unmodified design gates in ~5 s per script and cosims in ~125 s. A
#: candidate whose dataflow deadlocks blocks forever in the simulator, so the
#: gate fails it in minutes rather than the quarter-hour the smoke run lost.
GATE_TIMEOUT = 240
COSIM_TIMEOUT = 1800
#: Per-layer bound inside the model measurement. A layer that does not complete
#: is limitations item 24 (`Kt >= QD` deadlocks legal programs), and the run
#: refuses rather than reporting the model short by a layer.
MODEL_TIMEOUT = 600
#: xsim's transaction window runs a few cycles past HLS's latency count.
SIMTIME_SLACK = 12

#: Candidate processes run under this. Absent -> integrity checks only.
BWRAP = shutil.which("bwrap")


# -- the proposed configuration ----------------------------------------------
def frozen_ceilings(ref) -> dict:
    """`{name: (predicate, question, rule)}` -- `isa_encoding.MAXDIM_CEILINGS`
    read out of the FROZEN source at `ref`.

    Derived, never typed in here. The two ceilings are properties of the
    encoding and this project has already written one of them down wrong in
    three documents ("MAXDIM <= 90" against an answer of 88), which is what
    computing them is for. Read from git and not from the tree because
    `isa_encoding.py` is the CANDIDATE's now: a design may widen its own
    address field, but the number a proposal is held to is the committed
    encoding's, because that is the encoding everything else in the evaluation
    was measured on. A candidate that really has widened the field has raised a
    ceiling nobody has priced, and the conservative refusal names it.

    Only the one assignment is executed, in an empty namespace: the module
    imports numpy and `allo.actions`, and this harness has neither.
    """
    src = git_show(ref, f"{PKG}/isa_encoding.py").decode("utf-8")
    for node in ast.parse(src).body:
        if isinstance(node, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == "MAXDIM_CEILINGS"
                for t in node.targets):
            ns: dict = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]),  # noqa: S102
                         "isa_encoding.py", "exec"), ns)
            return ns["MAXDIM_CEILINGS"]
    raise Reject("config", "the frozen isa_encoding.py declares no "
                           "MAXDIM_CEILINGS; the ceilings cannot be computed "
                           "and a proposal cannot be checked against them")


def maxdim_ceiling(predicate, t, limit=1 << 14) -> int | None:
    """The largest MAXDIM a ceiling admits at `t`, over MULTIPLES OF T.

    `isa_encoding.maxdim_ceiling`'s own loop, and over multiples of T for its
    reason: `MAXDIM % T == 0` is an assertion of this ISA, and solving either
    inequality over the reals gives a number no build can use.
    """
    best = None
    for m in range(t, limit + 1, t):
        if predicate(m, t):
            best = m
    return best


def ceilings_at(ref, t) -> dict:
    """`{ceiling name: largest MAXDIM}` at `t`: 88 / 76 at T=4, 128 / 120 at 8."""
    return {name: maxdim_ceiling(spec[0], t)
            for name, spec in frozen_ceilings(ref).items()}


def usable_shapes(cfg: dict, shapes) -> tuple[list, dict]:
    """`(the scored shapes this configuration can run, why the rest cannot)`.

    `cosim.py` asserts every shape is a multiple of T and within MAXDIM -- it
    is the testbench's own precondition, and it is frozen. The five published
    shapes were chosen at T=4, and two of them (`4x4x4`, `12x12x12`) are not
    multiples of 8, so a T=8 candidate cannot be measured on them AT ALL. That
    is a property of the shape list, not of the candidate: refusing the
    candidate for it would make every T other than 4 unreachable and leave the
    loop searching one architecture, which is what this whole change is
    undoing.

    So the GEMM term becomes the shapes that CAN run, named alongside the ones
    that cannot, and the verdict carries both. A configuration that can run
    none of them is refused -- there is no control left to compare against.
    """
    t, maxdim = int(cfg["TPU_T"]), int(cfg["TPU_MAXDIM"])
    usable, skipped = [], {}
    for tag in shapes:
        dims = [int(x) for x in tag.split("x")]
        if any(d % t for d in dims):
            skipped[tag] = (f"not a multiple of T={t}; cosim.py's testbench "
                            f"asserts every dimension is")
        elif max(dims) > maxdim:
            skipped[tag] = f"does not fit MAXDIM={maxdim}"
        else:
            usable.append(tag)
    return usable, skipped


def resolve_config(spec_dir: Path) -> tuple[dict, bool]:
    """`(the configuration this candidate is scored at, did it propose one)`.

    The proposal is a module-level `CHIA_CONFIG = {...}` literal in the
    candidate's `microarch_isa.py`, read with `ast.literal_eval`. Nothing is
    imported and nothing is executed: this runs before `compose`, so that an
    unpriceable configuration is refused before the tree exists, let alone
    before Vitis starts.
    """
    src = (spec_dir / "microarch_isa.py").read_text(encoding="utf-8")
    try:
        tree = ast.parse(src, filename="microarch_isa.py")
    except SyntaxError as exc:
        raise Reject("setup", f"microarch_isa.py does not parse: {exc}")
    found = [node.value for node in tree.body
             if isinstance(node, ast.Assign)
             and any(isinstance(t, ast.Name) and t.id == CONFIG_DECL
                     for t in node.targets)]
    if not found:
        return dict(DEFAULT_CONFIG), False
    if len(found) > 1:
        raise Reject("config", f"{CONFIG_DECL} is declared {len(found)} times; "
                               f"a configuration is one declaration")
    try:
        proposed = ast.literal_eval(found[0])
    except ValueError:
        raise Reject("config", f"{CONFIG_DECL} must be a literal dict of ints, "
                               f"e.g. {CONFIG_DECL} = {{\"T\": 8, "
                               f"\"MAXDIM\": 32}}; it is computed, and this "
                               f"is read without executing the candidate")
    if not isinstance(proposed, dict):
        raise Reject("config", f"{CONFIG_DECL} must be a dict, not "
                               f"{type(proposed).__name__}")
    bad = sorted(k for k in proposed if k not in PROPOSABLE)
    if bad:
        raise Reject("config", f"{CONFIG_DECL} names {bad}; a candidate may "
                               f"propose {list(PROPOSABLE)}. DMA_WORDS is "
                               f"reached by editing the design, not by "
                               f"declaring a number; everything else about the "
                               f"memory model is not the candidate's to choose")
    if any(not isinstance(v, int) or isinstance(v, bool) for v in proposed.values()):
        raise Reject("config", f"{CONFIG_DECL} values must be ints: {proposed}")
    cfg = dict(DEFAULT_CONFIG)
    cfg.update({f"TPU_{k}": str(v) for k, v in proposed.items()})
    return cfg, True


def config_refusals(cfg: dict, ref: str) -> list[str]:
    """Every reason this configuration cannot be scored, all of them at once.

    All of them rather than the first, because a proposal past both a ceiling
    and the fitted envelope is refused for two different reasons and an agent
    told only the first would fix it and be refused again.
    """
    t, maxdim, qd = (int(cfg["TPU_T"]), int(cfg["TPU_MAXDIM"]),
                     int(cfg["TPU_QD"]))
    out = []
    # 1. `ip/params.py`'s invariants. Restated rather than imported: this
    #    harness cannot import the design (no `allo` in its env), and
    #    `test_harness` phase `s` checks the restatement against params.py.
    if t < 4:
        out.append(f"T={t}: the parameter set requires T >= 4 (ip/params.py)")
    if maxdim % t:
        out.append(f"MAXDIM={maxdim} is not a multiple of T={t} "
                   f"(ip/params.py: MAXDIM % T == 0)")
    # 2. The channel depth. Not a matter of degree: 8 hangs.
    if qd != QD_REQUIRED:
        out.append(f"QD={qd}: the only depth legal programs complete at is "
                   f"{QD_REQUIRED}. At 8, three of the ten tiled programs never "
                   f"finish in cosim (limitations item 24, `Kt >= QD` "
                   f"deadlocks), so this is refused rather than left to hang "
                   f"for the {COSIM_TIMEOUT}s timeout")
    # 3. The two encoding ceilings, computed from the frozen encoding.
    for name, ceiling in ceilings_at(ref, t).items():
        if ceiling is not None and maxdim > ceiling:
            out.append(f"MAXDIM={maxdim} is past the {name} ceiling, which is "
                       f"{ceiling} at T={t} "
                       f"(isa_encoding.MAXDIM_CEILINGS[{name!r}])")
    # 4. The area proxy's fitted envelope. Executed from git, like the policy.
    # Executed from git, like the policy, and given a `__file__` because the
    # module derives its report directory from one. Nothing here reads a
    # report: `envelope_refusals` is arithmetic over `COMMITTED`.
    proxy = {"__name__": "area_proxy",
             "__file__": str(REPO / AREA_PROXY)}
    exec(compile(git_show(ref, AREA_PROXY), "area_proxy.py", "exec"), proxy)  # noqa: S102
    out += proxy["envelope_refusals"]({"T": t, "MAXDIM": maxdim, "QD": qd})
    return out

ALLO_PYTHON = os.environ.get(
    "TINYTPU_ALLO_PYTHON", "/home/sk3463/miniconda3/envs/allo/bin/python")
LLVM_BUILD_DIR = os.environ.get(
    "LLVM_BUILD_DIR", "/home/sk3463/llvm-allo-6b09f739/build")


class Reject(Exception):
    def __init__(self, stage, detail):
        super().__init__(detail)
        self.stage, self.detail = stage, detail


def git_show(ref, path):
    out = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=REPO,
                         capture_output=True, check=False)
    if out.returncode:
        raise Reject("setup", f"cannot read frozen {path} @ {ref}: "
                              f"{out.stderr.decode()[:300]}")
    return out.stdout


def _imported_modules(src: bytes, rel: str) -> set[str]:
    """First-party dotted module names `rel` imports, relative ones resolved."""
    pkg = (rel[: -len("/__init__.py")] if rel.endswith("/__init__.py")
           else str(PurePosixPath(rel).parent)).replace("/", ".")
    out: set[str] = set()
    for node in ast.walk(ast.parse(src, rel)):
        if isinstance(node, ast.Import):
            out.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = pkg.split(".")
                base = base[: len(base) - node.level + 1]
                mod = ".".join(base + ([node.module] if node.module else []))
            else:
                mod = node.module or ""
            out.add(mod)
            # `from pkg import submodule` names a module, not an attribute.
            out.update(f"{mod}.{a.name}" if mod else a.name for a in node.names)
    return {m for m in out if m and any(
        m == pre or m.startswith(pre + ".") for pre in FIRST_PARTY)}


def _walk_imports(entries, exists, read):
    """(files reached, first-party modules that did not resolve)."""
    seen: set[str] = set()
    unresolved: set[str] = set()
    todo = list(entries)
    while todo:
        rel = todo.pop()
        if rel in seen or not exists(rel):
            continue
        seen.add(rel)
        if not rel.endswith(".py"):
            continue
        for mod in _imported_modules(read(rel), rel):
            stem = mod.replace(".", "/")
            for cand in (f"{stem}.py", f"{stem}/__init__.py"):
                if exists(cand):
                    if cand not in seen:
                        todo.append(cand)
                    break
            else:
                # A dotted name can also be `from module import name`, where
                # `name` is an attribute and not a module. Only the prefix
                # failing to resolve is a real miss.
                if not any(exists(f"{'/'.join(mod.split('.')[:n])}{suffix}")
                           for n in range(1, len(mod.split(".")) + 1)
                           for suffix in (".py", "/__init__.py")):
                    unresolved.add(mod)
    return sorted(seen), sorted(unresolved)


def import_closure(ref, entries=None, exists=None, read=None) -> list[str]:
    """Every first-party file the evaluation imports, transitively.

    Derived, never listed. A hand-written frozen set has broken the loop twice
    the same way -- `EDITABLE` naming two files after the design became a
    package, then `WORKLOAD_SUITE` missing the import closure of its own
    runner, which failed every candidate at stage `model`. A list cannot stay
    right for longer than the layout holds still; this is a property of the
    code instead. A file is frozen if the evaluation imports it. Read from the
    REF, not from disk, so a dirty working tree cannot change what is frozen.
    """
    files, _ = _walk_imports(
        entries if entries is not None else ENTRY_POINTS,
        exists or (lambda rel: _in_ref(ref, rel)),
        read or (lambda rel: git_show(ref, rel)))
    return files


def _in_ref(ref, rel) -> bool:
    return not subprocess.run(["git", "cat-file", "-e", f"{ref}:{rel}"],
                              cwd=REPO, capture_output=True).returncode


def resolve_ref(ref):
    out = subprocess.run(["git", "rev-parse", "--verify", f"{ref}^{{commit}}"],
                         cwd=REPO, capture_output=True, text=True)
    if out.returncode:
        raise Reject("setup", f"cannot resolve {ref}")
    return out.stdout.strip()


def unresolved_in_tree(tree: Path) -> list[str]:
    """First-party modules the tree imports and does not contain.

    `allo.*` is expected not to be there: it resolves from the checkout and is
    frozen by `CHECKOUT_WATCH` instead of by composition (`_from_checkout`).
    """
    _, unresolved = _walk_imports(
        [rel for rel in ENTRY_POINTS if (tree / rel).is_file()],
        lambda r: (tree / r).is_file(), lambda r: (tree / r).read_bytes())
    return [m for m in unresolved
            if not (m == "allo" or m.startswith("allo."))]


def frozen_paths(ref) -> list[str]:
    """Every file composed into the tree from git: the literal seed plus the
    entry points' import closure, minus what the candidate supplies.

    A function and not a constant: it costs a git read per file, and
    `evaluate` is imported in every sandboxed gate process.
    """
    editable = {f"{PKG}/{rel}" for rel in EDITABLE}
    return [rel for rel in sorted(set(FROZEN_LITERAL) | set(import_closure(ref)))
            if rel not in editable and not _from_checkout(rel)]


#: Measurement apparatus that lives inside the `allo` package -- the ACT
#: mapper, since `act/` moved to `allo/act/`. It is NOT composed into the
#: tree, because it cannot win import resolution there: `allo` is a regular
#: package with an `__init__.py` in the checkout, so `allo.act` always comes
#: from the checkout and a tree copy would sit unread. Synthesising an
#: `allo/__init__.py` into the tree would make the evaluation diverge from a
#: clean reproduction, and the tree's `allo` has no compiler in it anyway.
#: It is frozen by the OTHER mechanism instead: `CHECKOUT_WATCH` hashes it
#: before and after every gate, and `loop.FROZEN_PATHS` refuses to start a
#: search while it is dirty. `closure_cover` is what checks that every module
#: the evaluation imports is under one mechanism or the other.
def _from_checkout(rel: str) -> bool:
    return rel.startswith("allo/")


def closure_cover(ref) -> dict:
    """How each file of the closure is frozen: composed, or watched."""
    composed, watched, uncovered = [], [], []
    for rel in import_closure(ref):
        if f"{PKG}/{rel.removeprefix(PKG + '/')}" in {f"{PKG}/{e}" for e in EDITABLE}:
            continue                       # the candidate supplies it
        if _from_checkout(rel):
            (watched if any(rel == w or rel.startswith(w + "/")
                            for w in CHECKOUT_WATCH) else uncovered).append(rel)
        else:
            composed.append(rel)
    return {"composed": composed, "watched": watched, "uncovered": uncovered}


def compose(spec_dir: Path, tree: Path, ref: str):
    """Evaluation tree = frozen files from git + the candidate's two files.

    Returns {relative path: sha256} for every file in the tree."""
    base = main_base(ref)
    for rel in DESIGN_EVALUATOR:
        if git_show(ref, rel) != git_show(base, rel):
            raise Reject("setup", f"{rel} @ {ref[:8]} differs from main @ {base[:8]}")
    # The policy is executed from git too, not imported from the working tree.
    policy = {"__name__": "spec_policy"}
    exec(compile(git_show(ref, f"{PKG}/chia_agent/spec_policy.py"),
                 "spec_policy.py", "exec"), policy)
    policy_violations = policy["policy_violations"]
    # Frozen = the literal seed plus everything the entry points import,
    # transitively, minus what the candidate supplies. Derived from the ref so
    # that a file moving in the repository cannot silently drop out of the
    # tree -- which is how the workload runner shipped without `act_target`.
    for rel in frozen_paths(ref):
        dst = tree / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(git_show(ref, rel))
    doc_losses = {}
    for rel in EDITABLE:
        src = spec_dir / rel
        if not src.is_file():
            raise Reject("setup", f"spec dir has no {rel}")
        text = src.read_text(encoding="utf-8")
        base_text = git_show(ref, f"{PKG}/{rel}").decode("utf-8")
        # Only files the candidate CHANGED are held to the policy. A file it
        # never opened, byte-identical to the frozen ref, is main's own code
        # that a person committed; rejecting a candidate for a pre-existing
        # violation in it names a file absent from its diff, which the agent
        # can neither see, act on nor diagnose. accept.py was fixed the same
        # way in 29ddc55c, and this is the evaluator's half of it: the guard
        # keeps full strength on everything an agent writes, and the policy
        # itself -- the dunder rule included -- is untouched.
        if text != base_text:
            problems = policy_violations(rel, text) + policy["doc_violations"](
                rel, base_text, text)
            if problems:
                raise Reject("policy", "; ".join(problems))
            doc_losses[rel] = policy["doc_loss"](base_text, text, rel)
        dst = tree / PKG / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(text, encoding="utf-8")
    # Per file is not enough once the design is fourteen files: the budget the
    # guard was written with is a budget for the whole candidate.
    total = policy["doc_violations_total"](doc_losses)
    if total:
        raise Reject("policy", "; ".join(total))
    # Now that the candidate's own files are in it, the tree must be closed
    # under its imports: every first-party module the evaluation reaches has
    # to BE here. A tree that reaches outside itself has an input nobody
    # audited, which is the whole reason the files come from git. This also
    # catches a candidate that adds an import of something not frozen.
    dangling = unresolved_in_tree(tree)
    if dangling:
        raise Reject("setup", "the evaluation tree is not closed under its "
                              f"imports; cannot resolve {dangling}")
    # Anything else in the spec dir is ignored, not merged: only the files
    # `design.EDITABLE` names are the candidate.
    manifest = tree_manifest(tree)
    for f in tree.rglob("*"):
        if f.is_file():
            f.chmod(0o444)
    return manifest


def tree_manifest(tree: Path) -> dict:
    return {str(f.relative_to(tree)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(tree.rglob("*")) if f.is_file()}


def checkout_state() -> str:
    """Content of this checkout's tracked files (the evaluator included), as a
    hash of their diff against HEAD -- a file already dirty still counts."""
    diff = subprocess.run(["git", "diff", "HEAD", "--binary", "--", *CHECKOUT_WATCH],
                          cwd=REPO, capture_output=True).stdout
    return hashlib.sha256(diff).hexdigest()


def verify(tree: Path, manifest: dict, checkout: str, after: str):
    """Nothing the candidate ran may have changed the tree or the checkout."""
    now = tree_manifest(tree)
    if now != manifest:
        changed = sorted(k for k in set(now) | set(manifest)
                         if now.get(k) != manifest.get(k))
        raise Reject("tamper", f"after {after}, the evaluation tree changed: {changed}")
    if checkout_state() != checkout:
        raise Reject("tamper", f"after {after}, the checkout's tracked files changed")


def env_for(tree: Path, cfg: dict | None = None):
    """The environment every stage runs in: TPU_* scrubbed, then the ONE
    configuration this candidate is scored at pinned into it.

    `cfg` is `resolve_config`'s answer. It defaults to the published row so
    that a caller with no candidate in hand (`test_harness`, `control.py`)
    still gets the configuration the published numbers were taken at.
    """
    env = {k: v for k, v in os.environ.items() if not k.startswith("TPU_")}
    env.update(cfg or DEFAULT_CONFIG)
    env.update({
        # tree first (the candidate + frozen files), then this checkout for the
        # `allo` package and its in-tree mlir bindings.
        "PYTHONPATH": f"{tree}:{REPO}",
        "LLVM_BUILD_DIR": LLVM_BUILD_DIR,
        "OMP_NUM_THREADS": "8",
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    return env


def sandboxed(cmd, work: Path, tree: Path):
    """`cmd` under bubblewrap: read-only everything, but `work` and a private
    /tmp; `tree` (inside `work`) read-only again; own PID namespace."""
    if not BWRAP:
        return cmd
    # The checkout is re-bound READ-ONLY after the private /tmp, and the order
    # is the point: a checkout under /tmp -- a worktree in a scratch directory,
    # which is how this is developed -- is otherwise hidden by that tmpfs, and
    # every candidate then fails at stage `import` with "No module named
    # allo.compose", which names neither the sandbox nor the cause. Where the
    # checkout is not under /tmp this re-binds what `--ro-bind / /` already
    # gave, read-only both times, so nothing is opened up.
    return [BWRAP, "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc",
            "--tmpfs", "/tmp", "--tmpfs", "/dev/shm",
            "--ro-bind", str(REPO), str(REPO),
            "--bind", str(work), str(work), "--ro-bind", str(tree), str(tree),
            "--unshare-pid", "--die-with-parent", "--", *cmd]


def run(cmd, cwd, env, timeout, work=None, tree=None, stdin=None):
    t = time.time()
    if work is not None:
        cmd = sandboxed(cmd, work, tree)
    p = subprocess.Popen(cmd, cwd=cwd, env=env, text=True, stdout=subprocess.PIPE,
                         stderr=subprocess.STDOUT, start_new_session=True,
                         stdin=subprocess.PIPE if stdin is not None else subprocess.DEVNULL)
    try:
        out, _ = p.communicate(input=stdin, timeout=timeout)
        rc = p.returncode
    except subprocess.TimeoutExpired:
        # The whole group: a deadlocked simulator (or vitis_hls under cosim)
        # must not outlive the verdict.
        os.killpg(p.pid, signal.SIGKILL)
        out, _ = p.communicate()
        out, rc = (out or "") + f"\nTIMEOUT after {timeout}s (deadlock?)", 124
    return rc, out, time.time() - t


def area_estimate(tree, env, work):
    """The candidate's own structural bit census, priced by the frozen model.

    The census is taken INSIDE the tree, from the candidate's composed
    architecture, so a unit that declares a bigger array or a channel wired
    deeper is seen. The coefficients are in the frozen `area_proxy`, so the
    candidate cannot move the price it is charged.

    It is an ESTIMATE and says so in every field it returns. csynth's resource
    table is still reported beside it, because the two disagree and the
    disagreement is the finding, not an error to resolve.
    """
    code = ("import json, sys; sys.path.insert(0, %r); import area_proxy as p; "
            "print(json.dumps(p.estimate(p.live_census())))"
            % str(tree / PKG / "chia_agent"))
    rc, out, _ = run([ALLO_PYTHON, "-c", code], tree, env, 300, work, tree)
    if rc:
        raise Reject("area-proxy", out[-3000:])
    return json.loads(out.strip().splitlines()[-1])


def check_invariants(tree, env, work, cfg: dict | None = None):
    """The built module reports the configuration it was asked for.

    Unchanged in what it enforces; what moved is that the configuration is the
    candidate's proposal rather than a constant. A design that DECLARES T=8 and
    builds T=4 anyway is refused here, which is what makes the declaration
    worth reading: the proposal is a claim, and this is the check on it.
    """
    cfg = cfg or DEFAULT_CONFIG
    code = ("import json; from examples.tinytpu import "
            "microarch_isa as u; print(json.dumps({'T': u.T, 'MAXDIM': u.MAXDIM,"
            " 'QD': u.QD, 'DMA_WORDS': u.DMA_WORDS,"
            " 'IMEM_SIZE': u.IMEM_SIZE}))")
    rc, out, _ = run([ALLO_PYTHON, "-c", code], tree, env, 300, work, tree)
    if rc:
        raise Reject("import", out[-3000:])
    inv = json.loads(out.strip().splitlines()[-1])
    want = {k.removeprefix("TPU_"): int(v) for k, v in cfg.items()}
    got = {k: inv[k] for k in want}
    if got != want:
        raise Reject("invariant",
                     f"the design built {got} under {cfg}; it does not honour "
                     f"the configuration it is scored at")
    return inv


def gate_runner_cmd(root: Path, check, args=()):
    """The command that runs a frozen check under `root`'s gate_runner.py."""
    return [ALLO_PYTHON, str(root / GATE_RUNNER), check, *args]


def vouch(check, root: Path, spawn, args=()):
    """The nonce-vouched gate call. ONE definition; `accept.py` imports this
    one rather than keeping a second copy, because it is the primitive that
    decides whether a candidate's gate really passed.

    `spawn(cmd, stdin)` runs `cmd` and returns `(rc, out, seconds)`; how it is
    sandboxed and logged is the caller's business. Everything
    security-relevant is here:

    * the nonce is minted per call and reaches the runner on **stdin** only --
      never the environment, the command line, or a file the candidate reads;
    * `vouched` is True only if the output carries
      `CHIA-GATE <check> OK <nonce>` as a WHOLE line, with this run's nonce.
      Nothing the candidate prints can produce it: the runner reads the nonce
      before the candidate is imported and prints the line only when the check
      RETURNED success;
    * the nonce is scrubbed from the output that is returned, so it never
      reaches a verdict, a log, or anything the agent can read.

    Returns `(vouched, rc, out, seconds)`."""
    nonce = secrets.token_hex(16)
    rc, out, sec = spawn(gate_runner_cmd(root, check, args), nonce + "\n")
    ok = rc == 0 and f"CHIA-GATE {check} OK {nonce}" in out.splitlines()
    return ok, rc, out.replace(nonce, "<nonce>"), sec


def vouched(check, tree, env, work, cwd, timeout, args=()):
    """Run a frozen check under gate_runner.py; (vouched, rc, out, seconds)."""
    return vouch(check, tree, lambda cmd, stdin: run(
        cmd, cwd, env, timeout, work, tree, stdin=stdin), args)


#: The PyTorch oracle's bound. It is 4 s at MAXDIM=16 and 57 s at MAXDIM=64
#: (the mapper's enumeration for mlp_wide's 64x64x64 layers is nearly all of
#: it), measured on this host; five minutes is slack, not a budget.
VERIFY_TIMEOUT = 300


def pytorch_gate(tree, env, work, verify_now):
    """The oracle, FIRST and cheapest: torch's own forward, byte for byte.

    This runs before `bench_isa` deliberately. It is the only check with
    something outside this repository on one side, so with the ISA editable it
    is the check that cannot be satisfied by rewriting a reference model -- and
    a candidate that fails it should be refused in seconds rather than after a
    csynth.

    The corpus is reported and not pinned: which of the suite's four MLPs fit
    depends on the configuration (`mlp_small` needs MAXDIM >= 32, `mlp_wide`
    64), and `run.py` requires the two that fit every build and refuses if
    either is missing. A gate is only as strong as its corpus, and this one is
    two models over 2,688 bytes at MAXDIM=16.
    """
    ok, rc, out, sec = vouched("workloads", tree, env, work, tree,
                               VERIFY_TIMEOUT, args=("--verify",))
    verify_now("pytorch")
    m = re.search(r"^  VERIFY OK: (\d+)/(\d+) model\(s\) bit-exact against "
                  r"torch over (\d+) bytes", out, re.M)
    if not ok or not m:
        raise Reject("gate:pytorch", out[-4000:])
    return {"models": f"{m.group(1)}/{m.group(2)}", "bytes": int(m.group(3)),
            "reference": "torch.nn.Linear's own forward on the model's real "
                         "weights, with the machine's epilogue applied in "
                         "torch; nothing a candidate writes is on either side",
            "corpus": lines_with(out, "BIT-EXACT against torch"),
            "vouched": True, "seconds": round(sec, 1)}


def isa_gate(tree, env, work, verify_now):
    """`gen_isa.py --conform` on the CANDIDATE's own spec.

    Internal consistency, which is exactly what stays valuable when the spec
    moves: `isa_encoding.py` byte-identical to what this spec generates, every
    software constant and bit slice in the design at the spec's positions, the
    assembler and the program generator through the same encoder, the reference
    model structurally forbidden from importing an opcode number or a field
    position from the design, and -- this is `--conform`'s own addition -- the
    bit ranges THE EMITTED HLS actually reads.

    `--no-doc` because the generated ISA tables live in
    `docs/source/designs/tinytpu_isa_spec.rst`, which is not in the evaluation
    tree and is not a candidate's to regenerate. The artefact that decides
    whether the spec and the code agree is `isa_encoding.py`, and it is checked
    byte for byte.

    Every arm of `--conform` fails the candidate, including the one that holds
    the spec's unit ports against the composed region. That arm had a false
    positive worth recording, because the first fix for it was to turn it off
    IN THE GATE and leave it on in `reproduce.sh` -- which would have been a
    weakened referee, the exact hole freezing `isa_spec.json` was protecting.
    The real cause was in `allo/compose.py`: `Unit.arrays()` ignored any array
    declared with an initialiser, so re-adding the `spad` zero-fill that
    `b4be2b10` removed (one line, bit-exact, merely slower --
    `test_harness`'s case (b)) hid the array and the arm then refused the
    candidate for a change in what the projection could SEE. Fixing that made
    the arm strictly stronger: four sequencer loop-stack arrays it had been
    blind to became visible, and `gen_isa.UNMODELLED` now states each with its
    reason.
    """
    ok, rc, out, sec = vouched("gen_isa", tree, env, work, tree, GATE_TIMEOUT,
                               args=("--conform", "--no-doc"))
    verify_now("gen_isa")
    if not ok or not re.search(r"^  ISA OK:", out, re.M) or "ISA FAIL" in out:
        raise Reject("gate:isa", out[-4000:])
    return {"conform": lines_with(out, "ISA OK")[0], "vouched": True,
            "seconds": round(sec, 1)}


def gate(tree, env, work, verify_now, cfg=None):
    result = {"pytorch": pytorch_gate(tree, env, work, verify_now),
              "isa": isa_gate(tree, env, work, verify_now)}
    ok, rc, out, sec = vouched("bench_isa", tree, env, work, tree, GATE_TIMEOUT)
    verify_now("bench_isa")
    if not ok or not re.search(r"^  ALL EXACT$", out, re.M) or "FAILURES" in out:
        raise Reject("gate:bench_isa", out[-4000:])
    ok2, rc2, out2, sec2 = vouched("stress_isa", tree, env, work, tree, GATE_TIMEOUT)
    verify_now("stress_isa")
    # The runner vouches that stress_isa.main() returned 0. The count is
    # recorded, and must be internally consistent (n/n); it is not pinned,
    # because stress_isa skips the unrolled reference programs that do not fit
    # the candidate's imem, as it does on main.
    m = re.search(r"^  STRESS OK: (\d+)/(\d+) runs exact", out2, re.M)
    if not ok2 or not m or m.group(1) != m.group(2):
        raise Reject("gate:stress", out2[-4000:])
    # 3. Parametricity: the same candidate, rebuilt at other MAXDIMs, must
    # build, honour the parameter, and be exact (param_check.py).
    param, sec3 = {}, 0.0
    for pcfg in param_configs(cfg or DEFAULT_CONFIG):
        ok3, rc3, out3, s3 = vouched("param_check", tree, dict(env, **pcfg), work,
                                     tree, GATE_TIMEOUT)
        sec3 += s3
        tag = ",".join(f"{k}={v}" for k, v in pcfg.items())
        verify_now(f"param_check {tag}")
        m3 = re.search(r"^  PARAM OK: (\d+)/(\d+) runs exact", out3, re.M)
        if not ok3 or not m3 or m3.group(1) != m3.group(2):
            raise Reject("gate:param", f"at {tag}:\n" + out3[-4000:])
        param[tag] = lines_with(out3, "PARAM OK")[0]
    result.update(
        bench_isa="ALL EXACT", stress=lines_with(out2, "STRESS OK")[0],
        stress_runs=int(m.group(1)), param=param, vouched=True,
        seconds=round(sec + sec2 + sec3 + result["pytorch"]["seconds"]
                      + result["isa"]["seconds"], 1))
    return result


def lines_with(text, needle):
    return [l.strip() for l in text.splitlines() if needle in l]


# -- the co-design stages ----------------------------------------------------
_MAP_COUNT = re.compile(r"^MAPSPACE (\S+): (\d+)/(\d+) encodable$", re.M)
_MAP_REFUSED = re.compile(r"^MAPSPACE (\S+): refused\s+(\d+)\s+(.+?)\s*$", re.M)
_MAP_CHOSEN = re.compile(
    r"^MAPSPACE (\S+): CHOSEN (.+?) \((\d+) fetches, (\d+) words, "
    r"IMEM_SIZE=(\d+)\)$", re.M)


def parse_mapspace(out):
    """The refusal histogram, per shape, out of `codesign_gate.py`'s lines.

    This is the co-design signal, and it is parsed from a vouched gate's own
    output: how many nests this hardware can encode, and which constraint
    refused the rest. It is a REPORT, never the objective -- the objective is
    the pair (cosim cycles, csynth resources) below.
    """
    shapes = {}
    for tag, enc, total in _MAP_COUNT.findall(out):
        shapes[tag] = {"encodable": int(enc), "total": int(total), "refused": {}}
    for tag, n, cause in _MAP_REFUSED.findall(out):
        shapes.setdefault(tag, {"refused": {}})["refused"][cause] = int(n)
    for tag, name, dyn, words, imem in _MAP_CHOSEN.findall(out):
        shapes.setdefault(tag, {"refused": {}}).update(
            chosen=name, fetches=int(dyn), words=int(words), imem_size=int(imem))
    return shapes


def mapspace_gate(tree, env, work, shapes, verify_now):
    """Enumerate the mapspace exhaustively and prove every survivor correct."""
    ok, rc, out, sec = vouched("codesign", tree, env, work, tree, GATE_TIMEOUT,
                               args=(",".join(shapes),))
    verify_now("codesign")
    if not ok or not re.search(r"^  MAPSPACE OK$", out, re.M) \
            or "MAPSPACE FAILED" in out or "SEAM OK" not in out:
        raise Reject("gate:mapspace", out[-4000:])
    found = parse_mapspace(out)
    missing = [s for s in shapes if s not in found]
    if missing:
        raise Reject("gate:mapspace", f"no mapspace report for {missing}\n"
                                      f"{out[-3000:]}")
    for s in shapes:
        if not found[s].get("encodable") or not found[s].get("chosen"):
            raise Reject("gate:mapspace",
                         f"{s}: {found[s].get('encodable')} encodable nests")
    return {"shapes": found, "seconds": round(sec, 1),
            "rule": "min(instruction fetches incl. LOOP/ENDLOOP, static words, "
                    "nest) over the nests this "
                    "hardware can encode; exhaustive over the mapspace",
            "vouched": True}


def codesign_score(tree, env, work: Path, shapes, verify_now):
    """cosim of the mapper's chosen program: the measured half of the objective.

    Identical to `score()` except that the check is `codesign_cosim`, which
    binds `cosim.gemm_program` to the mapper's pick in frozen code. Every
    independent check on the number is the same: one row per shape,
    `mismatches = 0` in the summary AND in that shape's own log, a PASS, the
    cycle count agreeing with the log's simulated time, the csynth clock met,
    and no interface-latency override anywhere.
    """
    return _cosim("codesign_cosim", tree, env, work, shapes, verify_now)


def parse_synth(prj: Path):
    xml = prj / "out.prj/solution1/syn/report/csynth.xml"
    if not xml.exists():
        raise Reject("csynth", "no csynth.xml -- synthesis failed")
    root = ET.parse(xml).getroot()
    target = float(root.findtext(".//TargetClockPeriod"))
    est = float(root.findtext(".//EstimatedClockPeriod"))
    res = root.find(".//AreaEstimates/Resources")
    area = {k.lower(): int(res.findtext(k)) for k in
            ("BRAM_18K", "DSP", "FF", "LUT", "URAM")} if res is not None else {}
    if abs(target - TARGET_NS) > 1e-6:
        raise Reject("csynth", f"target clock {target} ns != frozen {TARGET_NS}")
    if est > TARGET_NS:
        raise Reject("timing", f"estimated clock {est} ns misses {TARGET_NS} ns; "
                               f"cycles at a clock the design cannot meet are "
                               f"not comparable")
    return {"target_ns": target, "estimated_ns": est, "area": area}


def check_memory_model(prj: Path):
    kernel = (prj / "kernel.cpp").read_text(errors="replace")
    pragmas = re.findall(r"#pragma HLS interface m_axi[^\n]*", kernel)
    if len(pragmas) != 4:
        raise Reject("memory-model", f"expected 4 m_axi ports, found {len(pragmas)}")
    bad = [p for p in pragmas if re.search(r"\blatency\s*=", p)]
    if bad or "config_interface" in kernel:
        raise Reject("memory-model", f"interface latency override in kernel.cpp: {bad}")
    for log in [prj / "csynth.log", *prj.glob("cosim_*.log"), prj / "run.tcl"]:
        if log.exists():
            text = log.read_text(errors="replace")
            if "m_axi_latency" in text or "random_stall" in text:
                raise Reject("memory-model", f"{log.name} changes the memory model")


def score(tree, env, work: Path, shapes, verify_now):
    return _cosim("cosim", tree, env, work, shapes, verify_now)


def model_score(tree, env, work: Path, models, verify_now, menv=None):
    """The model term: each model's layers, in order, on one csynth of the
    candidate's own RTL.

    Measured the same way as the GEMM term and held to the same evidence: a
    vouched runner, the tree re-verified afterwards, and a layer that does not
    complete is a refusal rather than a missing row. It costs one csynth plus
    one cosim per LAYER -- six for the two scored models against two for the
    GEMM shapes -- which is what the model term is worth paying, because it is
    the workload class the measured 25-34 %/4.3-7.0 % split says the GEMM
    shapes cannot rank.
    """
    if not models:
        return {}, 0.0, {}
    menv = menv or SCORED_MODEL_ENV
    out_json = work / "models.json"
    # The model term's OWN configuration (SCORED_MODEL_ENV), not the GEMM
    # term's: measured, and the reason is in that constant's comment.
    ok, rc, out, sec = vouched(
        "workloads", tree,
        dict(env, **menv, TPU_PRJ=str(work / "workload.prj")),
        work, work, COSIM_TIMEOUT,
        args=("--cosim", *models, "--project", str(work / "workload.prj"),
              "--json", str(out_json), "--timeout", str(MODEL_TIMEOUT)))
    verify_now("workloads")
    if not ok or not out_json.exists():
        raise Reject("model", out[-4000:])
    report = json.loads(out_json.read_text())
    measured, skipped = report.get("cycles", {}), report.get("skipped", {})
    # Two different things, and collapsing them was refusing every T=8
    # candidate. A model the MAPPER cannot express at this configuration is out
    # of scope -- at T=8 `mlp_deep`'s 16x16x12 layer is refused because 12 is
    # not a multiple of T, which `workloads/scope_map.json` records and the
    # candidate had no part in. A model that MAPPED and then did not produce a
    # cycle count is limitations item 24 and is a refusal.
    missing = [m for m in models if m not in skipped
               and (not isinstance(measured.get(m), int) or measured[m] <= 0)]
    if missing:
        raise Reject("model", f"no measured cycles for {missing}; a layer that "
                              f"does not complete is limitations item 24, not "
                              f"a zero\n{out[-3000:]}")
    got = {m: measured[m] for m in models if m not in skipped}
    if not got:
        raise Reject("model", f"no model of {list(models)} maps at this "
                              f"configuration, so the objective has no model "
                              f"term at all: {skipped}")
    return got, round(sec, 1), {m: skipped[m] for m in models if m in skipped}


def _cosim(check, tree, env, work: Path, shapes, verify_now):
    prj = work / "isa_sweep.prj"
    # cosim.py (main since e620576d) puts its project next to itself by default,
    # which is the read-only tree here; TPU_PRJ is a path, not a memory-model
    # knob, and is the only TPU_* variable set.
    env = dict(env, TPU_SHAPES=",".join(shapes), TPU_PRJ=str(prj))
    ok, rc, out, sec = vouched(check, tree, env, work, work, COSIM_TIMEOUT)
    verify_now(check)
    if not ok:
        raise Reject("cosim", out[-4000:])
    synth = parse_synth(prj)
    check_memory_model(prj)
    cycles = {}
    for s in shapes:
        M, K, N = (int(x) for x in s.split("x"))
        pat = re.compile(rf"^\s*{M}x\s*{K}x\s*{N}\s+cycles=(\S+)\s+(.*)$")
        rows = [pat.match(l) for l in out.splitlines()]
        rows = [r for r in rows if r]
        if len(rows) != 1:
            raise Reject("cosim", f"shape {s} reported {len(rows)} times\n{out[-3000:]}")
        n, tb = rows[0].group(1), rows[0].group(2)
        want = f"TB {M}x{K}x{N} mismatches = 0 / {M * N}"
        if n == "None" or tb.strip() != want:
            raise Reject("cosim", f"{s}: cycles={n} tb='{tb}' (want '{want}')\n"
                                  f"{out[-3000:]}")
        log = prj / f"cosim_{M}x{K}x{N}.log"
        text = log.read_text(errors="replace") if log.exists() else ""
        if want not in text or "C/RTL co-simulation finished: PASS" not in text:
            raise Reject("cosim", f"{s}: its own cosim log lacks '{want}' + PASS")
        t = [int(x) for x in re.findall(r'RTL Simulation : \d+ / 1 \[n/a\] @ "(\d+)"', text)]
        if len(t) == 2:
            sim_cycles = (t[1] - t[0]) / (TARGET_NS * 1000)
            if abs(sim_cycles - int(n)) > SIMTIME_SLACK:
                raise Reject("cosim", f"{s}: report says {n} cycles, simulated "
                                      f"time says {sim_cycles:.0f}")
        cycles[s] = int(n)
    # `chosen`: which nest the mapper's cosim driver actually put under the RTL,
    # read back out of the scoring process's own output. Empty for plain `cosim`.
    chosen = {tag: name for tag, name in re.findall(
        r"^MAPSPACE (\S+): CHOSEN (.+?) of \d+/\d+ encodable$", out, re.M)}
    return cycles, synth, round(sec, 1), chosen


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--spec-dir", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--shapes", default=",".join(SEARCH_SHAPES))
    ap.add_argument("--models", default=",".join(SCORED_MODELS),
                    help="the model term's workloads; empty measures none")
    ap.add_argument("--gate-only", action="store_true")
    ap.add_argument("--codesign", action="store_true",
                    help="run the co-design stages: the exhaustive mapspace "
                         "enumeration with its refusal histogram, and cosim of "
                         "the program the frozen mapper chose for this hardware")
    a = ap.parse_args()
    shapes = [s for s in a.shapes.split(",") if s]
    started = time.time()
    result = {"ok": False, "frozen_ref": FROZEN_REF, "shapes": shapes,
              "codesign": a.codesign,
              "measurement": "cosim: Vitis HLS 2023.2 csynth + xsim C/RTL cosim "
                             "(RTL), m_axi_latency default 0, one bit-exact TB "
                             "per shape"}
    if a.codesign:
        result["measurement"] += ("; the program under the RTL is the best nest "
                                  "the frozen mapper can encode on this "
                                  "hardware, chosen by exhaustive enumeration")
    try:
        bad = [s for s in shapes if s not in ALL_SHAPES]
        if bad:
            raise Reject("setup", f"shapes {bad} are not in the frozen SHAPES")
        work = a.work.resolve()
        spec = a.spec_dir.resolve()
        if (work == spec or work in spec.parents or spec in work.parents
                or work == REPO or work in REPO.parents
                or (work.is_relative_to(REPO)
                    and not work.is_relative_to(REPO / ".chia_scratch"))):
            # It is rmtree'd below: never the spec, the checkout, or above them.
            raise Reject("setup", f"work dir {work} overlaps the spec dir {spec} "
                                  f"or the checkout; it would be wiped")
        # Wiped every time: a stale cosim report must never be read as a result.
        if work.exists():
            shutil.rmtree(work)
        tree = work / "tree"
        tree.mkdir(parents=True)
        ref = resolve_ref(FROZEN_REF)
        result["frozen_ref"] = ref
        # The configuration FIRST, from the spec dir alone: nothing is composed
        # and no candidate code runs, so a configuration that cannot be priced
        # or cannot be built is refused in milliseconds.
        cfg, proposed = resolve_config(spec)
        result["config"] = dict(cfg)
        result["config_proposed"] = proposed
        result["ceilings"] = ceilings_at(ref, int(cfg["TPU_T"]))
        refusals = config_refusals(cfg, ref)
        if refusals:
            raise Reject("config", "; ".join(refusals))
        # The GEMM term is the requested shapes this configuration can RUN.
        shapes, unrunnable = usable_shapes(cfg, shapes)
        result["shapes"], result["shapes_skipped"] = shapes, unrunnable
        if not shapes:
            raise Reject("config", f"none of the requested shapes can run at "
                                   f"{cfg}: {unrunnable}. There is no GEMM "
                                   f"control left, so there is nothing to "
                                   f"compare a cycle count against")
        checkout = checkout_state()
        manifest = compose(spec, tree, ref)
        verify_now = lambda after: verify(tree, manifest, checkout, after)
        result["sandbox"] = bool(BWRAP)
        env = env_for(tree, cfg)
        result["invariants"] = check_invariants(tree, env, work, cfg)
        verify_now("the import check")
        result["gate"] = gate(tree, env, work, verify_now, cfg)
        if a.codesign:
            result["mapspace"] = mapspace_gate(tree, env, work, shapes, verify_now)
        if not a.gate_only:
            runner = codesign_score if a.codesign else score
            cyc, synth, sec, chosen = runner(tree, env, work, shapes, verify_now)
            result.update(cycles=cyc, total_cycles=sum(cyc.values()),
                          synth=synth, cosim_seconds=sec)
            models = [m for m in a.models.split(",") if m]
            menv = model_env(cfg)
            mcyc, msec, mskip = model_score(tree, env, work, models,
                                             verify_now, menv)
            result.update(model_cycles=mcyc, total_model_cycles=sum(mcyc.values()),
                          model_seconds=msec, model_env=dict(menv),
                          model_skipped=mskip)
            # The area ESTIMATE, from the candidate's own bit census. It costs
            # milliseconds, so it is taken on every scored candidate; a DC run
            # is ~70 minutes on another host and cannot be.
            result["area"] = area_estimate(tree, env, work)
            if chosen:
                result["scored_nest"] = chosen
                # The nest the mapspace gate chose and the nest that was
                # actually cosimmed are two independent runs of the same frozen
                # rule. If they disagree the candidate is not deterministic, and
                # the cycle count does not belong to the reported mapping.
                planned = {s: v.get("chosen")
                           for s, v in result.get("mapspace", {})
                           .get("shapes", {}).items()}
                bad = {s: (planned.get(s), chosen.get(s)) for s in shapes
                       if planned.get(s) is not None
                       and planned[s] != chosen.get(s)}
                if bad:
                    raise Reject("nondeterministic",
                                 f"the mapper chose a different nest in the gate "
                                 f"and in the scorer: {bad}")
            # The objective is a TRIPLE and is never collapsed:
            #
            #   gemm     cosim cycles per GEMM shape   -- the CONTROL
            #   model    cosim cycles per model        -- the workload term
            #   area     the standard-cell ESTIMATE    -- silicon, not FPGA
            #
            # Two of the three are new, and each replaces an axis this project
            # measured to be misleading. `gemm` stays, unchanged and unweighted:
            # a change that helps models and hurts GEMM shapes is a real trade,
            # and the search should be made to STATE it rather than hide it,
            # which needs the control kept. csynth's resource table is still
            # reported -- it is the FPGA answer, and where it disagrees with
            # `area` that disagreement is a result.
            result["objective"] = {
                "gemm": cyc,
                "model": result.get("model_cycles", {}),
                "model_skipped": result.get("model_skipped") or None,
                "model_env": result.get("model_env"),
                "area": result.get("area"),
                "resources": dict(synth.get("area", {}),
                                  estimated_ns=synth.get("estimated_ns")),
                "note": "three terms, reported per shape and per model; never "
                        "summed into one score. `area` is an ESTIMATE "
                        "(chia_agent/area_proxy.py), `resources` is csynth's "
                        "FPGA table, and the two disagree by design",
            }
        result["ok"] = True
    except Reject as r:
        result.update(stage=r.stage, detail=r.detail[-4000:])
    result["seconds"] = round(time.time() - started, 1)
    detail = result.pop("detail", None)
    if detail:
        print(detail)
    print(json.dumps(result, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
