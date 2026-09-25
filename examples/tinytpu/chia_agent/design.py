# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which files are the design, and which are the machinery around it.

One definition, imported by everything that needs it: the MCP edit tool
(`allo_tool.py`), the evaluator (`evaluate.py`), the loop's spec seeding
(`loop.py`), the acceptance gate (`accept.py`), the control record
(`control.py`) and the $0 preflight (`test_harness.py`). It used to be a
two-name tuple repeated in six places; the design is a package now, and a list
that can drift in six places is a list that will.

The split is the point of the decomposition
(`docs/source/designs/tinytpu_library.rst`), not an accident of it:

* **EDITABLE** is the design under search -- the INSTRUCTION SET (the spec, the
  encoding generated from it, and the reference model built on that), the eight
  units, the wiring that composes them into an architecture, the assembler that
  must agree with them, the programs, and the instantiation that chooses the
  parameters and declares the configuration it wants to be scored at. A CHIA agent editing one 60-line unit is a
  better-scoped experiment than one editing a 1,582-line file. Both wins of
  run 1 (the front-end rewrite and the operand burst) landed inside what is
  now `ip/units/dma_load.py` and `ip/units/sequencer.py`; a version of this
  list that stopped at `microarch_isa.py` would have taken the loop's
  demonstrated capability away, because `microarch_isa.py` is now a hundred
  lines of instantiation with no hardware in it.
* **FROZEN_DESIGN** is the machinery a candidate may not reach: `params.py`
  (which owns the parameter set's invariants, `T >= 4` and `MAXDIM % T == 0`),
  the package `__init__` files, and the reduce IP (`reduce.py`,
  `units/reduction_tree.py`) -- a different design that `ip/__init__.py`
  imports and that nothing in the GEMM datapath touches. The composer itself
  is `allo/compose.py`, outside the package and covered by `CHECKOUT_WATCH`.
  The evaluator takes every frozen file from git, like all the others.

Paths are relative to `examples/tinytpu/`, and they are paths rather than bare
names: the spec directory mirrors the package.
"""

from __future__ import annotations

#: The design under search. Order is stable so that a diff, a manifest and a
#: prompt list them the same way every time.
EDITABLE = (
    "microarch_isa.py",
    "isa_dsl.py",
    # The ISA. `isa_spec.json` is the source of truth, `isa_encoding.py` is
    # GENERATED from it by the frozen `gen_isa.py` (regenerate with the
    # `regenerate_isa` tool, never by hand -- `gen_isa.py --conform` compares
    # them byte for byte), and `isa_ref.py` is the reference model built on the
    # generated module. All three were frozen until the loop was allowed to
    # co-design the instruction set; what replaced the freeze is the PyTorch
    # oracle and `gen_isa.py --conform`, and `chia_agent/evaluate.py`'s
    # docstring is where that argument is written down.
    "isa_spec.json",
    "isa_encoding.py",
    "isa_ref.py",
    "ip/isa.py",
    "ip/tinytpu.py",
    "ip/assembler.py",
    "ip/programs.py",
    "ip/units/sequencer.py",
    "ip/units/dma_load.py",
    "ip/units/scratchpad.py",
    "ip/units/vector_regs.py",
    "ip/units/weight_loader.py",
    "ip/units/pe.py",
    "ip/units/accumulator.py",
    "ip/units/dma_store.py",
)

#: Design files a candidate may NOT change; the evaluator reads them from git.
#: `ip/__init__.py` imports `reduce`, and `ip/units/__init__.py` imports
#: `reduction_tree`, so both must be in the tree for the package to import at
#: all -- they are here because they are needed, not because they are part of
#: the machine under search.
FROZEN_DESIGN = (
    "ip/__init__.py",
    "ip/params.py",
    "ip/reduce.py",
    "ip/units/__init__.py",
    "ip/units/reduction_tree.py",
)

#: The unit bodies alone -- what "edit one unit" means, for a prompt or for a
#: search that wants to scope a candidate to a single process.
UNITS = tuple(rel for rel in EDITABLE if rel.startswith("ip/units/"))

#: What a spec file may import from `examples.*`: the editable modules and the
#: frozen ones they are composed by. `spec_policy.ALLOWED_EXAMPLES` is a
#: literal that must agree with this (test_harness phase `s` checks it); it
#: cannot import this file, because `evaluate.compose()` execs the policy out
#: of git and a candidate must not be able to widen it.
MODULE_PREFIX = "examples.tinytpu"


def module_name(rel: str) -> str:
    """`ip/units/pe.py` -> `examples.tinytpu.ip.units.pe`."""
    return f"{MODULE_PREFIX}." + rel.removesuffix(".py").replace("/", ".")


#: Editable files that are DATA rather than modules: nothing imports them, and
#: `module_name` would turn `isa_spec.json` into a dotted name that is not one.
DATA = tuple(rel for rel in EDITABLE if not rel.endswith(".py"))

IMPORTABLE_MODULES = frozenset(
    module_name(rel) for rel in EDITABLE + FROZEN_DESIGN
    if rel.endswith(".py") and not rel.endswith("__init__.py"))
