# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The workload-spec schema and its meaning. Prose: docs/source/extensions/act_specs.rst."""

import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                "..", "..", "..")))
from examples.tinytpu.microarch_isa import (  # noqa: E402
    MAXDIM, T, WPR,
)
from examples.tinytpu.stress_isa import operands  # noqa: E402

CORPUS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")

DTYPE_RANGE = {"int8": (-128, 127), "int32": (-(1 << 31), (1 << 31) - 1)}
BUFFERS = ("A", "B", "C")
EPILOGUE_OPS = ("relu", "saturate")
PAD_POLICIES = ("arbitrary", "zero")
WRITE_WINDOWS = ("exact", "column_block")
REQUIRED = ("name", "stresses", "einsum", "dims", "inputs", "output",
            "accumulator", "epilogue", "operand_pad")


class SpecError(ValueError):
    """A spec that does not say what it must, named field by field."""


def constant(kind, rows, cols):
    if kind == "identity":
        return np.eye(rows, cols, dtype=np.int8)
    if kind == "ones_column":
        out = np.zeros((rows, cols), np.int8)
        out[:, 0] = 1
        return out
    raise SpecError(f"unknown constant {kind!r}; known: identity, ones_column")


def _extent(sp, dims, subscript):
    return tuple(dims[c] for c in subscript)


def shape2d(extent):
    """Every tensor lies row-major in a DRAM buffer: all but the last
    subscript index rows, the last indexes columns, and a vector is a column."""
    if len(extent) == 1:
        return extent[0], 1
    rows = 1
    for n in extent[:-1]:
        rows *= n
    return rows, extent[-1]


def needs_maxdim(sp):
    """The smallest `MAXDIM` whose buffers hold every region this spec names."""
    reach = 0
    for t in sp["inputs"] + [sp["output"]] + sp.get("constants", []):
        rows, cols = (shape2d(_extent(sp["name"], sp["dims"], t["subscript"]))
                      if "subscript" in t else (t["rows"], t["cols"]))
        row, col = t["origin"]
        reach = max(reach, row + rows, col + cols)
    return reach


def fits_build(sp):
    """None if this build can hold the spec, else the reason it cannot.

    A spec outlives one build: the corpus keeps the steady-state shapes
    `bench_isa.LATENCY`/`STEADY` name even where `MAXDIM` is too small to run
    them, and the judge skips those rather than calling them malformed."""
    need = needs_maxdim(sp)
    if need > MAXDIM:
        return f"needs MAXDIM >= {need}; this build has TPU_MAXDIM={MAXDIM}"
    return None


def _check_region(sp, where, buf, origin, extent):
    if buf not in BUFFERS:
        raise SpecError(f"{sp}: {where} names buffer {buf!r}, not one of {BUFFERS}")
    row, col = origin
    if row < 0 or col < 0:
        raise SpecError(f"{sp}: {where} starts at {origin}")
    if col % T:
        raise SpecError(
            f"{sp}: {where} starts at column {col}, which is not a multiple of "
            f"T={T}. Every DRAM access the ISA has is a packed word of T lanes "
            f"at a column-block boundary, so a tensor cannot start between them.")


def validate(sp):
    """Raise `SpecError` unless `sp` is a complete, self-consistent spec."""
    name = sp.get("name", "<unnamed>")
    for key in REQUIRED:
        if key not in sp:
            raise SpecError(f"{name}: missing required field {key!r}")
    if not isinstance(sp["stresses"], str) or len(sp["stresses"]) < 16:
        raise SpecError(
            f"{name}: 'stresses' must say in one line what this case tests "
            f"that the others do not; a corpus of unexplained shapes teaches "
            f"a compiler nothing")
    dims = sp["dims"]
    for c, n in dims.items():
        if len(c) != 1 or not c.isalpha() or not c.islower():
            raise SpecError(f"{name}: dim {c!r} must be one lowercase letter")
        if not isinstance(n, int) or n < 1:
            raise SpecError(f"{name}: dim {c!r} has extent {n!r}")
    subs = [t["subscript"] for t in sp["inputs"]]
    out_sub = sp["output"]["subscript"]
    derived = ",".join(subs) + "->" + out_sub
    if sp["einsum"] != derived:
        raise SpecError(
            f"{name}: einsum is {sp['einsum']!r} but the subscripts of the "
            f"inputs and the output spell {derived!r}. The einsum is the "
            f"human-readable statement of the same thing; they must agree.")
    for c in "".join(subs) + out_sub:
        if c not in dims:
            raise SpecError(f"{name}: subscript {c!r} has no extent in 'dims'")
    for t in sp["inputs"]:
        if t["dtype"] not in DTYPE_RANGE:
            raise SpecError(f"{name}: input {t['name']} dtype {t['dtype']!r}")
        _check_region(name, f"input {t['name']}", t["buffer"], t["origin"],
                      _extent(name, dims, t["subscript"]))
    for c in sp.get("constants", []):
        constant(c["kind"], c["rows"], c["cols"])
        _check_region(name, f"constant {c['name']}", c["buffer"], c["origin"],
                      (c["rows"], c["cols"]))
        if c["buffer"] == sp["output"]["buffer"]:
            raise SpecError(
                f"{name}: constant {c['name']} sits in the output buffer, "
                f"which the program overwrites")
    out = sp["output"]
    if out["dtype"] not in DTYPE_RANGE:
        raise SpecError(f"{name}: output dtype {out['dtype']!r}")
    if out.get("write_window", "exact") not in WRITE_WINDOWS:
        raise SpecError(f"{name}: write_window must be one of {WRITE_WINDOWS}")
    _check_region(name, "output", out["buffer"], out["origin"],
                  _extent(name, dims, out_sub))
    if sp["accumulator"] not in DTYPE_RANGE:
        raise SpecError(f"{name}: accumulator {sp['accumulator']!r}")
    for op in sp["epilogue"]:
        if op not in EPILOGUE_OPS:
            raise SpecError(f"{name}: epilogue op {op!r} not in {EPILOGUE_OPS}")
    if sp["epilogue"] and sp["epilogue"][-1] != "saturate":
        raise SpecError(
            f"{name}: the epilogue's last op must be 'saturate' whenever there "
            f"is one, because the narrowing to {out['dtype']} happens on the "
            f"way out and every earlier op sees the accumulator's width")
    if sp["operand_pad"] not in PAD_POLICIES:
        raise SpecError(f"{name}: operand_pad must be one of {PAD_POLICIES}")
    return sp


def load(path):
    with open(path) as f:
        return validate(json.load(f))


def corpus():
    """Every spec in `corpus/`, in filename order."""
    return [load(p) for p in sorted(glob.glob(os.path.join(CORPUS, "*.json")))]


def by_name(name):
    sp = [s for s in corpus() if s["name"] == name]
    if not sp:
        raise SpecError(f"no spec named {name!r} in {CORPUS}")
    return sp[0]


def region(sp, tensor):
    """`(buffer, row slice, column slice)` of one tensor's declared bytes."""
    if "subscript" in tensor:
        rows, cols = shape2d(_extent(sp["name"], sp["dims"], tensor["subscript"]))
    else:
        rows, cols = tensor["rows"], tensor["cols"]
    row, col = tensor["origin"]
    return tensor["buffer"], slice(row, row + rows), slice(col, col + cols)


def write_window(sp):
    """`(row slice, column slice)` of `C` the program is allowed to write."""
    _, rs, cs = region(sp, sp["output"])
    if sp["output"].get("write_window", "exact") == "column_block":
        cs = slice(cs.start, (cs.stop + T - 1) // T * T)
    return rs, cs


def buffers(sp, dist="full", seed=0):
    """The DRAM images the judge hands the design: `(A, B, C)`, flat int8.

    Declared regions carry the workload's operands, every other byte carries
    what `operand_pad` promises, and `C` always arrives prefilled with random
    bytes so that whatever the program leaves outside its write window is a
    detected clobber rather than a lucky zero."""
    rng = np.random.default_rng(0x5EED + seed)
    src = dict(zip("AB", operands(dist, seed)))
    img = {}
    for b in "AB":
        img[b] = (np.zeros((MAXDIM, MAXDIM), np.int8) if sp["operand_pad"] == "zero"
                  else rng.integers(-128, 128, (MAXDIM, MAXDIM)).astype(np.int8))
    if sp["operand_pad"] == "arbitrary":
        for b in "AB":
            img[b][img[b] == 0] = 1
    for t in sp["inputs"]:
        buf, rs, cs = region(sp, t)
        img[buf][rs, cs] = src[buf][rs, cs]
    for c in sp.get("constants", []):
        buf, rs, cs = region(sp, c)
        img[buf][rs, cs] = constant(c["kind"], c["rows"], c["cols"])
    C = rng.integers(-128, 128, (MAXDIM, MAXDIM)).astype(np.int8)
    return (img["A"].reshape(-1), img["B"].reshape(-1), C.reshape(-1))


def gold(sp, A, B, C):
    """What the spec MEANS, as numpy: the whole of `C` after the workload.

    This is the semantics of the *spec*, not of the machine -- the machine's
    semantics is `isa_ref.run`, and the judge holds the two against each other
    rather than trusting either alone."""
    img = {"A": np.asarray(A, np.int8).reshape(MAXDIM, MAXDIM),
           "B": np.asarray(B, np.int8).reshape(MAXDIM, MAXDIM)}
    axis = {c: i for i, c in enumerate(sorted(sp["dims"]))}
    args = []
    for t in sp["inputs"]:
        buf, rs, cs = region(sp, t)
        extent = _extent(sp["name"], sp["dims"], t["subscript"])
        args += [img[buf][rs, cs].astype(np.int64).reshape(extent),
                 [axis[c] for c in t["subscript"]]]
    acc = np.einsum(*args, [axis[c] for c in sp["output"]["subscript"]],
                    dtype=np.int64)
    for op in sp["epilogue"]:
        if op == "relu":
            acc = np.maximum(acc, 0)
        elif op == "saturate":
            lo, hi = DTYPE_RANGE[sp["output"]["dtype"]]
            acc = np.clip(acc, lo, hi)
    out = np.array(C, np.int8).reshape(MAXDIM, MAXDIM)
    _, rs, cs = region(sp, sp["output"])
    out[rs, cs] = acc.reshape(rs.stop - rs.start, cs.stop - cs.start)
    return out.reshape(-1)


def known_gap(sp):
    """Why no program on the current build can satisfy this spec, or None."""
    return sp.get("known_gap")


def compare(sp, got, want):
    """None if `got` satisfies the spec given the gold `want`, else one line."""
    got = np.asarray(got, np.int8).reshape(MAXDIM, MAXDIM)
    want = np.asarray(want, np.int8).reshape(MAXDIM, MAXDIM)
    _, rs, cs = region(sp, sp["output"])
    wrs, wcs = write_window(sp)
    inside = int((got[rs, cs] != want[rs, cs]).sum())
    free = np.zeros((MAXDIM, MAXDIM), bool)
    free[wrs, wcs] = True
    free[rs, cs] = False
    clobber = int(((got != want) & ~free)[:].sum()) - inside
    if not inside and not clobber:
        return None
    return (f"{inside} of {(rs.stop - rs.start) * (cs.stop - cs.start)} result "
            f"bytes wrong, {clobber} bytes clobbered outside the write window")


def summary(sp):
    dims = ", ".join(f"{c}={n}" for c, n in sp["dims"].items())
    return (f"{sp['name']:28s} {sp['einsum']:14s} {dims:26s} "
            f"pad={sp['operand_pad']:9s} "
            f"win={sp['output'].get('write_window', 'exact')}")


if __name__ == "__main__":
    cs = corpus()
    runnable = [sp for sp in cs if fits_build(sp) is None]
    print(f"{len(cs)} specs in {CORPUS}; {len(runnable)} fit this build "
          f"(MAXDIM={MAXDIM} T={T} WPR={WPR})")
    for sp in cs:
        print("  " + summary(sp))
        print(f"      {sp['stresses']}")
        if fits_build(sp):
            print(f"      SKIPPED on this build: {fits_build(sp)}")
