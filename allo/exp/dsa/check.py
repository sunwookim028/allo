# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The checker: the v2 constraint system (§3.4), executable.

One constraint object, two entries (v2 §5.2): the compiler *chooses* (argmin
over configurations, ASAP over dependences) and this module *checks* — the same
domains, predicates, regions and σ, evaluated on a program rather than searched
over. Everything here consumes the **serialized** program through its epoch
reading (``epoch.py``), never the planner's in-memory state, which is what makes
it equally a self-check of the pipeline and a verifier for an externally
assembled program (a hand-written stream, or an imported mapping —
``mapping.py``, which assembles one and hands its σ to ``check(sigma=...)``).

What each of the seven constraints becomes, executably — and honestly, since
several have fragments no wire-level checker can discharge:

1. **Coverage** → *definedness*: every element an epoch reads, and every element
   the host reads back as an output, was written first (by an epoch, or preloaded
   as an input / constant). Full coverage — σ total on the *source's* iteration
   domain — needs a D no emitted program carries; an imported mapping does carry
   one, and ``mapping.Mapping.check`` discharges it there. Definedness is the
   wire-level shadow that remains when D is gone.
2. **No overbooking** → per-PE issue intervals pairwise disjoint under σ.
3. **Dependence** → every RAW/WAR/WAW edge (exact, ``depends(exact=True)``)
   finishes before its consumer starts under σ. Communication latency is inside
   this check by construction: movement is explicit mover epochs, and the edge
   chain through one carries its ``(pe, t)``.
4. **Reachability** → every exact RAW pair — the (write event → read event)
   pairs that *are* R's domain — admitted by the machine's own declaration
   (``ISA.network(reaches=...)``, over σ's spatial names), or by a caller-supplied
   predicate that overrides it. With neither, R is the total relation: precisely
   the operation-ISA case where R drops out of the model.
5. **Capacity** → *bounds*: every region inside its buffer. The liveness half
   reduces to constraint 3 for a fixed emit stream — preserving all three
   dependence classes preserves every read's last writer — so it is not a
   separate check; what no address-level checker can see is whether the
   *allocator meant* two overlapping locations (value identity dies at emission).
6. **Bandwidth** — the unit-issue form is constraint 2 (``ii`` is issue
   bandwidth); port/link contention needs machine declarations that do not exist
   yet. Not checked.
7. **Reduction** — a machine that combines partial sums across instances
   declares the instruction that does it (``ISA.network(reduces=...)``), and that
   instruction is then an ordinary epoch with an ordinary compute region: the
   combination is *in the stream*, so constraints 1 and 3 cover it and there is
   nothing left here to check separately. Undeclared, an imported mapping may not
   fan a reduction rank at all (``mapping.Mapping.check`` refuses it), so the
   obligation never arises. What is still unchecked is a machine that reduces
   **inside** ``K`` without emitting anything — there the combination leaves no
   trace on the wire.

Plus the machine's own validity predicate ``V``, per epoch: the emitted schedule
fields and a mover's routed residence params must be a configuration the
instruction ``admits`` — ACT's ``e_θ``, evaluated on the wire instead of during
selection.

The substrate is exact finite enumeration (``epoch.region_elements``): a
compiled program is fully concrete, so sets are finite and the checks are
decision procedures, not approximations. Presburger machinery (ISL) becomes the
substrate the day something *symbolic* needs checking — a parametric mapping
family (targets 4–5) — and enters through that one seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import prod

from .core import access_names, dense_strides, is_mover, mover_domains
from .epoch import (
    Region,
    _extent,
    depends,
    epochs as read_epochs,
    region_elements,
    schedule,
)
from .search import CompiledProgram, _placement_dims


@dataclass(frozen=True)
class Violation:
    """One discharged obligation that failed: which constraint, which epoch(s)
    (indices into the epoch sequence; empty for program-level), and why."""

    constraint: str  # "validity" | "definedness" | "bounds" | "overbooking"
    #                | "dependence" | "reachability", and from `mapping.py`
    #                (which reports in the same currency): "coverage" |
    #                "intrinsic" | "residence" | "reduction" | "binding"
    epochs: tuple
    message: str

    def __str__(self):
        where = ",".join(map(str, self.epochs)) or "program"
        return f"[{self.constraint}] epoch {where}: {self.message}"


def _block(isa, buf, offset, shape) -> frozenset:
    """The element set of a host-placed block (an input, constant or output):
    ``shape`` placed at ``offset`` in ``buf``, dense in the buffer's own pitch —
    the same placement ``CompiledProgram._region`` stages through."""
    dims = _placement_dims(shape, buf)
    weight = dense_strides(buf.memref_shape)
    n = len(dims)
    return region_elements(
        isa, Region(buf.name, tuple(offset), tuple(zip(dims, weight[:n])))
    )


def check(program: CompiledProgram, *, sigma=None, reachable=None) -> list[Violation]:
    """Check ``program`` against the constraint system; ``[]`` means it passes.

    ``sigma`` supplies an external ``Schedule`` to verify instead of the derived
    ASAP one — the verify half of the choose/check split, and the entry an
    imported mapping will use. ``reachable(producer, consumer, dep) -> bool`` is R
    over RAW event pairs (both epochs carry their σ); omitted, the machine's own
    declaration is used (``ISA.network(reaches=...)``), and with neither, R is
    total — the operation-ISA case where it drops out of the model."""
    isa = program.isa
    eps = read_epochs(isa, program.emits)
    out: list[Violation] = []
    if reachable is None and isa.reaches is not None:
        # The machine's own R, lifted from σ's spatial names to the RAW pairs it
        # quantifies over. A caller-supplied predicate still wins: verifying an
        # externally assembled program may need a relation the ISA does not state.
        def reachable(src, dst, _dep):
            return isa.reaches(src.sigma.pe, dst.sigma.pe)

    # --- V: the machine's own validity predicate, per instruction -------------
    # Per *member*, not per epoch: `admits` is ACT's e_θ and θ is an instruction, so
    # a multi-instruction epoch owes the predicate once for each thing it issues.
    for j, e in enumerate(eps):
        for who, cfg in e.members:
            spec = isa._ops[who].spec
            free = mover_domains(spec) if is_mover(spec) else {}
            names = access_names(spec)
            shapes = {names.index(n): v for n, v in cfg.shapes.items()}
            chosen = dict(cfg.schedule)
            chosen |= {n: cfg.residence[n] for n in free if n in cfg.residence}
            if not spec.admits(shapes, chosen, free):
                out.append(
                    Violation(
                        "validity",
                        (j,),
                        f"{who}: configuration {chosen} is not one the instruction "
                        f"admits at shapes {cfg.shapes}",
                    )
                )

    # --- 1 (fragment): definedness — reads find written data ------------------
    io = program.io_buffer
    defined: dict[str, set] = {name: set() for name in isa.buffers}
    for offset, shape in program.inputs:
        defined[io.name] |= _block(isa, io, offset, shape)
    for offset, data in program.constants:
        defined[io.name] |= _block(isa, io, offset, data.shape)
    for j, e in enumerate(eps):
        for r in e.regions:
            elems = region_elements(isa, r)
            if not r.writes:
                missing = elems - defined[r.buffer]
                if missing:
                    out.append(
                        Violation(
                            "definedness",
                            (j,),
                            f"{e.name} reads {len(missing)} element(s) of "
                            f"'{r.buffer}' nothing has written",
                        )
                    )
            else:
                defined[r.buffer] |= elems
    for offset, shape, label in program.outputs:
        missing = _block(isa, io, offset, shape) - defined[io.name]
        if missing:
            out.append(
                Violation(
                    "definedness",
                    (),
                    f"output {label}: {len(missing)} element(s) never written",
                )
            )

    # --- 5 (fragment): bounds — every region inside its buffer ----------------
    for j, e in enumerate(eps):
        for r in e.regions:
            lo, hi = _extent(isa, r)
            total = prod(isa.buffers[r.buffer].memref_shape)
            if lo < 0 or hi >= total:
                out.append(
                    Violation(
                        "bounds",
                        (j,),
                        f"{e.name} touches [{lo}, {hi}] of '{r.buffer}', which "
                        f"holds {total} element(s)",
                    )
                )

    # --- 2, 3, 4: the σ obligations -------------------------------------------
    sched = sigma if sigma is not None else schedule(isa, eps)
    assert len(sched.epochs) == len(eps) and all(
        s.name == e.name for s, e in zip(sched.epochs, eps)
    ), "sigma does not schedule this program"

    spans: dict = {}
    for j, e in enumerate(sched.epochs):
        spans.setdefault(e.sigma.pe, []).append((e.sigma.start, e.sigma.issue, j))
    for pe, intervals in spans.items():
        intervals.sort()
        for (s0, i0, a), (s1, _i1, b) in zip(intervals, intervals[1:]):
            if s0 + i0 > s1:
                out.append(
                    Violation(
                        "overbooking",
                        (a, b),
                        f"'{pe}' issues {sched.epochs[b].name} at {s1} while "
                        f"{sched.epochs[a].name} occupies it until {s0 + i0}",
                    )
                )

    deps = depends(isa, eps, exact=True)
    for d in deps:
        src, dst = sched.epochs[d.src], sched.epochs[d.dst]
        if dst.sigma.start < src.sigma.finish:
            out.append(
                Violation(
                    "dependence",
                    (d.src, d.dst),
                    f"{dst.name} starts at {dst.sigma.start} but {d.kind} on "
                    f"'{d.buffer}' requires {src.name} to finish first "
                    f"(at {src.sigma.finish})",
                )
            )
    if reachable is not None:
        for d in deps:
            if d.kind != "raw":
                continue
            src, dst = sched.epochs[d.src], sched.epochs[d.dst]
            if not reachable(src, dst, d):
                out.append(
                    Violation(
                        "reachability",
                        (d.src, d.dst),
                        f"R does not connect {src.name}'s write of '{d.buffer}' "
                        f"to {dst.name}'s read",
                    )
                )
    return out
