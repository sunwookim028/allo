# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""``mutate.py``'s mutants on the instance, where they apply.

    python -m examples.minitpu.template.instances.tinytpu.run_mutate [--jobs N] [names...]

``examples/tinytpu/mutate.py`` is reused for its MUTANTS table and its
levels; what changes is the DESIGN the anchors are located in and the
tree the mutant is run from. The design here is the instance: the frozen
``ip/`` units it reuses, this package's own files (``pe.py``,
``geometry.py``, ``accumulator.py``, ``instance.py``), the template's
``engines.py``, and ``glue/microarch_isa.py`` in place of the frozen glue.
``mutate.py`` writes its trees under ``examples/tinytpu/.mutants``; this
driver writes under ``$TPU_MUTANTS`` (default: a scratch directory) and
never under the frozen tree (README D-4).

A frozen mutant whose anchor is in the frozen PE (``ip/units/pe.py``, not
composed here) is reported NOT APPLICABLE, not caught: the PE is
re-expressed and its text differs. ``INSTANCE_MUTANTS`` are their
counterparts on the instance's PE and on the ENGINE, which is where a MAC
mutant lives once the MAC is a plug-in (D-15). ``ar_claim_false`` is RTL-only
and is not run (no cosim level here); it is reported as not run.
"""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", "..", "..", ".."))
TINYTPU = os.path.join(REPO, "examples", "tinytpu")
TEMPLATE = os.path.join(REPO, "examples", "minitpu", "template")
WORK = os.environ.get("TPU_MUTANTS") or os.path.join(tempfile.gettempdir(), "ttinst_mutants")

sys.path.insert(0, REPO)
sys.path.insert(0, TINYTPU)
M = importlib.import_module("mutate")   # examples/tinytpu/mutate.py, as a module

#: The files a mutant tree carries, keyed by the path the mount serves them
#: at: ``("tinytpu", rel)`` under ``examples.tinytpu``, ``("instance", rel)``
#: under this package, ``("template", rel)`` under the template. The frozen
#: ``ip/`` goes whole (its ``units/__init__`` imports every unit, the frozen
#: PE included) so the overlay is a regular package.
TREE = (
    [("tinytpu", os.path.relpath(os.path.join(d, f), TINYTPU))
     for d, _, fs in os.walk(os.path.join(TINYTPU, "ip")) for f in sorted(fs)
     if f.endswith(".py")]
    + [("instance", f) for f in ("pe.py", "geometry.py", "accumulator.py", "instance.py")]
    + [("instance", "glue/microarch_isa.py")]
    + [("template", "engines.py")]
)
#: Where an anchor is located: the design the instance COMPOSES, so not the
#: frozen PE.
DESIGN = [k for k in TREE if k != ("tinytpu", os.path.join("ip", "units", "pe.py"))]
ROOTS = {"tinytpu": TINYTPU, "instance": HERE, "template": TEMPLATE}

#: Counterparts of the frozen PE mutants on the instance's PE and engine.
INSTANCE_MUTANTS = [
    ("inst_pe_act_lane_swapped", "column-0 PE taps activation lane j, not i",
     "lane_shifted: UInt(T * MAC_IN_BITS) = activation_word >> (MAC_IN_BITS * i)",
     "(MAC_IN_BITS * i)", "(MAC_IN_BITS * j)"),
    ("inst_pe_shadow_not_swapped", "weight double buffer: the PE keeps its first nonzero weight",
     "            weight = pe_word\n",
     "            weight = pe_word\n", "            if mm_rows == 0:\n                weight = pe_word\n"),
    ("inst_pe_psum_not_forwarded", "a PE adds the product to 0 instead of the partial sum from the north",
     "psum: MAC_ACC = MAC_ADD(product, psum_north)",
     "MAC_ADD(product, psum_north)", "MAC_ADD(product, 0)"),
    ("engine_mul_int8_truncated", "the engine's product truncated to int8 (a MAC mutant on the ENGINE)",
     "def mul_int8(a: int8, w: int8) -> int32:",
     "    p: int32 = a16 * w16\n", "    p8: int8 = a16 * w16\n    p: int32 = p8\n"),
    ("engine_add_int32_sub", "the engine's accumulate subtracts",
     "def add_int32(p: int32, q: int32) -> int32:",
     "s: int32 = p + q", "s: int32 = p - q"),
]

LEVELS = {k: v for k, v in M.LEVELS.items() if k != "cosim"}

SHIM = """
import runpy, sys
sys.path.insert(0, {repo!r})
import examples.tinytpu as pkg
pkg.__path__ = [{tinytpu!r}] + list(pkg.__path__)
import examples.minitpu.template as tpl
tpl.__path__ = [{template!r}] + list(tpl.__path__)
import examples.minitpu.template.instances.tinytpu as inst
inst.__path__ = [{instance!r}] + list(inst.__path__)
import examples.minitpu.template.instances.tinytpu.glue as glue
glue.__path__ = [{instance!r} + '/glue'] + list(glue.__path__)
sys.argv = [{script!r}] + {args!r}
runpy.run_path({script!r}, run_name="__main__")
"""


def read(key):
    root, rel = key
    with open(os.path.join(ROOTS[root], rel), encoding="utf-8") as f:
        return f.read()


def locate(name, anchor):
    counts = {k: read(k).count(anchor) for k in DESIGN}
    hits = [k for k, n in counts.items() if n]
    if len(hits) != 1 or counts[hits[0]] != 1:
        return None
    return hits[0]


def mutant_tree(name):
    """-> ({key: text}, status): the design with one file mutated, or the
    reason the mutant does not apply to the instance."""
    tree = {k: read(k) for k in TREE}
    entry = next(m for m in list(M.MUTANTS) + INSTANCE_MUTANTS if m[0] == name)
    _, _, anchor, old, new = entry
    if anchor is None:
        return tree, "control"
    key = locate(name, anchor)
    if key is None:
        frozen = [f for f in M.DESIGN
                  if open(os.path.join(TINYTPU, f), encoding="utf-8").read().count(anchor)]
        return None, f"NOT APPLICABLE (anchor in {', '.join(frozen) or 'no design file'})"
    src = tree[key]
    i = src.find(old, src.index(anchor))
    assert i >= 0, f"{name}: {old!r} not found after its anchor in {key}"
    tree[key] = src[:i] + new + src[i + len(old):]
    return tree, f"in {key[1]}"


def run_level(name, level):
    d = os.path.join(WORK, name)
    script, args, marker, timeout = LEVELS[level]
    env = dict(os.environ)
    env["TPU_INSTANCE"] = "template"
    env.setdefault("TPU_MAXDIM", "16")
    glue_mod = os.path.join(d, "instance", "glue")
    shim = SHIM.format(repo=REPO, tinytpu=os.path.join(d, "tinytpu"),
                       template=os.path.join(d, "template"),
                       instance=os.path.join(d, "instance"),
                       script=os.path.join(TINYTPU, script), args=args)
    # The mount serves `examples.tinytpu.microarch_isa` from the mutant
    # tree's copy of the GLUE, not from the frozen file.
    os.makedirs(os.path.join(d, "tinytpu"), exist_ok=True)
    shutil.copyfile(os.path.join(glue_mod, "microarch_isa.py"),
                    os.path.join(d, "tinytpu", "microarch_isa.py"))
    log = os.path.join(d, f"{level}.log")
    with open(log, "w", encoding="utf-8") as f:
        try:
            rc = subprocess.run([sys.executable, "-c", shim], cwd=REPO, env=env,
                                stdout=f, stderr=subprocess.STDOUT,
                                timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            return "CAUGHT (hang)"
    out = open(log, errors="replace", encoding="utf-8").read()
    return "pass" if rc == 0 and marker in out else "CAUGHT"


def evaluate(name):
    tree, status = mutant_tree(name)
    if tree is None:
        return {"status": status}
    d = os.path.join(WORK, name)
    shutil.rmtree(d, ignore_errors=True)
    for (root, rel), text in tree.items():
        path = os.path.join(d, root, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    result = {"status": status}
    for lv in LEVELS:
        result[lv] = run_level(name, lv)
        if result[lv] != "pass":
            break
    return result


def main(argv):
    jobs = 4
    if "--jobs" in argv:
        jobs = int(argv[argv.index("--jobs") + 1])
        argv = [a for i, a in enumerate(argv) if a != "--jobs" and argv[i - 1] != "--jobs"]
    names = [a for a in argv if not a.startswith("--")] or \
        [m[0] for m in M.MUTANTS] + [m[0] for m in INSTANCE_MUTANTS]
    os.makedirs(WORK, exist_ok=True)
    print(f"  instance mutants: {len(names)} named, {jobs} at a time, trees under {WORK}")
    with ThreadPoolExecutor(jobs) as pool:
        results = dict(zip(names, pool.map(evaluate, names)))
    holes, not_applicable, not_run, caught = [], [], [], []
    for name, r in results.items():
        levels = {k: v for k, v in r.items() if k != "status"}
        line = f"  {name:28s} {r['status']:40s} " + "  ".join(f"{k}={v}" for k, v in levels.items())
        print(line, flush=True)
        if name in M.RTL_ONLY:
            not_run.append(name)
        elif r["status"].startswith("NOT APPLICABLE"):
            not_applicable.append(name)
        elif name == "none":
            if any(v != "pass" for v in levels.values()):
                print("  MUTATE FAILED: the unmodified instance did not pass through the mount")
                return 1
        elif all(v == "pass" for v in levels.values()):
            holes.append(name)
        else:
            caught.append(name)
    print(f"  MUTATE {'OK' if not holes else 'FAILED'}: {len(caught)} caught, "
          f"{len(holes)} survived {holes}, {len(not_applicable)} not applicable "
          f"{not_applicable}, {len(not_run)} RTL-only not run {not_run}")
    return 1 if holes else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
