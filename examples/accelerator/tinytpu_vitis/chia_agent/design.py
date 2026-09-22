# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Which files are the design, and which are the machinery around it.

One definition, imported by everything that needs it: the MCP edit tool
(`allo_tool.py`), the evaluator (`evaluate.py`), the loop's spec seeding
(`loop.py`), the acceptance gate (`accept.py`) and the $0 preflight
(`test_harness.py`). It used to be a two-name tuple repeated in five places;
the design is a package now, and a list that can drift in five places is a
list that will.

The split is the point of the decomposition
(`docs/source/designs/tinytpu_library.rst`), not an accident of it:

* **EDITABLE** is the design under search -- the instruction encoding, the
  eight units, the wiring that composes them into an architecture, the
  assembler that must agree with them, the programs, and the instantiation
  that chooses the parameters. A CHIA agent editing one 60-line unit is a
  better-scoped experiment than one editing a 1,582-line file.
* **FROZEN_DESIGN** is the machinery: `compose.py` (which emits the region's
  source, and therefore uses `exec` and `open` -- constructs the spec policy
  refuses, correctly, in anything a candidate may write), `params.py` (which
  owns the parameter set's invariants, `T >= 4` and `MAXDIM % T == 0`), and
  the package `__init__` files. A candidate cannot reach them, and the
  evaluator takes them from git like every other frozen file.

Paths are relative to `examples/accelerator/tinytpu_vitis/`, and they are
paths rather than bare names: the spec directory mirrors the package.
"""

from __future__ import annotations

#: The design under search. Order is stable so that a diff, a manifest and a
#: prompt list them the same way every time.
EDITABLE = (
    "microarch_isa.py",
    "isa_dsl.py",
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
FROZEN_DESIGN = (
    "ip/__init__.py",
    "ip/compose.py",
    "ip/params.py",
    "ip/units/__init__.py",
)

#: The unit bodies alone -- what "edit one unit" means, for a prompt or for a
#: search that wants to scope a candidate to a single process.
UNITS = tuple(rel for rel in EDITABLE if rel.startswith("ip/units/"))

#: What a spec file may import from `examples.*`: the editable modules and the
#: frozen ones they are composed by. `spec_policy.ALLOWED_EXAMPLES` is built
#: from this, so adding a unit does not mean remembering to widen the policy.
MODULE_PREFIX = "examples.accelerator.tinytpu_vitis"


def module_name(rel: str) -> str:
    """`ip/units/pe.py` -> `examples.accelerator.tinytpu_vitis.ip.units.pe`."""
    return f"{MODULE_PREFIX}." + rel.removesuffix(".py").replace("/", ".")


IMPORTABLE_MODULES = frozenset(
    module_name(rel) for rel in EDITABLE + FROZEN_DESIGN
    if not rel.endswith("__init__.py"))
