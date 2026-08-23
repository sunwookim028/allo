# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a benchmark is, and how the runner finds them.

A benchmark is one workload plus the SCHEDULES it can be given. The schedule is
the subject: the same algorithm at several points of hardware pressure is what
lets a compiler change be argued from a spread rather than from one arbitrary
configuration. The points:

    none    no scheduling primitives at all. The baseline, and the thing phase 2
            checks the others against: a schedule may change the hardware, never
            the function.
    v1, v2  scheduled points, numbered rather than named. The names carry no
            ranking on purpose: a variant only has to be a DIFFERENT legal
            hardware, and several are slower than the baseline (a recurrence-
            bound kernel gains nothing from unrolling and pays for the area).
            Calling one "optimized" would assert something the numbers do not.
            `v2` is optional: a kernel whose second point differs from `v1` in
            no way the hardware can see carries one, because a duplicate
            measurement is worse than an absent one.

Everything here is a schedule of ONE body. A streaming or `async`/`await`
dataflow form is a different body, not a variant, so it belongs in its own
entry: keeping a variant set to one body is what makes phase 2's check mean
something, namely that every variant computes the same function.

Three facts about this toolchain shape the contract, all measured rather than
assumed:

  - `export()` lowers the Kernel's module IN PLACE, so one kernel object cannot
    carry two schedules. `build()` is therefore a factory, called once per
    variant, and it must define its kernels locally rather than at module level.
  - `s.loop(name)` only sees the top kernel's own body. A loop inside a callee
    is reached by scheduling that callee separately and `compose`-ing it, which
    is why `build()` returns every kernel a schedule might need to name, not
    just the top one.
  - Greedy modulo placement costs about the CUBE of a region's access count
    (measured: 48 accesses 0.1 s, 192 accesses 2.8 s, 432 accesses 40.5 s). The
    RTL backend fully unrolls everything under a pipelined loop, so pipelining
    an outer loop of a large nest builds one enormous region and does not finish
    in any usable time. An `aggressive` point must therefore keep the innermost
    region bounded; it is a real constraint on this bed, not a style preference.
"""

from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

_SUITES = ("polybench", "machsuite", "rosetta", "pp4fpgas")


@dataclass(frozen=True)
class Benchmark:
    """One workload. `suite/name` identifies it; everything else is callables so
    that nothing is built until a runner asks for it."""

    suite: str
    name: str
    #: () -> {"top": Kernel, <callee name>: Kernel, ...}. A fresh set every call.
    build: Callable[[], dict]
    #: variant name -> (parts) -> Schedule, ready to export.
    schedules: dict[str, Callable[[dict], object]]
    #: (rng) -> the kernel's positional arguments, output buffers included.
    inputs: Callable[[np.random.Generator], tuple]
    #: (*args) -> the expected contents of whichever arguments the kernel writes,
    #: as a tuple in the same order `outputs` names them.
    reference: Callable[..., tuple]
    #: indices into the argument tuple that the kernel writes.
    outputs: tuple[int, ...]
    #: comparison tolerance for a float kernel; None means compare exactly.
    tolerance: tuple[float, float] | None = None
    #: one line on what the workload is and where it came from.
    doc: str = ""
    #: set when a variant is known to be unsupported, so a runner reports it as
    #: skipped rather than as a failure. variant name -> reason.
    skip: dict[str, str] = field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.suite}/{self.name}"


def discover(suites: tuple[str, ...] = _SUITES) -> list[Benchmark]:
    """Every `BENCHMARK` under the named suite packages, in a stable order."""
    found = []
    root = Path(__file__).parent
    for suite in suites:
        pkg_path = root / suite
        if not pkg_path.is_dir():
            continue
        for mod in sorted(
            pkgutil.walk_packages([str(pkg_path)], f"benchmark.{suite}.")
        ):
            module = importlib.import_module(mod.name)
            bench = getattr(module, "BENCHMARK", None)
            if bench is not None:
                found.append(bench)
    return found


def find(key: str) -> Benchmark:
    """The one benchmark named `suite/name`; how a runner's child process gets
    back to the workload its parent chose."""
    for b in discover():
        if b.key == key:
            return b
    raise SystemExit(f"no benchmark {key!r}")
