# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal repros for the U1 ``vpu_alu`` composition findings.

    python dev/records/minitpu/u1_alu/repros.py            # every repro, one process each
    python dev/records/minitpu/u1_alu/repros.py C3 C4      # just these

Run from the worktree root, after ``source examples/minitpu/harness/env-zhang21.sh``.
Each repro prints one ``<id> ...`` line saying what happened; the expected
outcome is in its docstring. Allo exits the process on some front-end errors,
so each repro runs in a subprocess of its own.
"""

import os
import subprocess
import sys
import tempfile

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[".."] * 4))
sys.path.insert(0, ROOT)


def C1():
    """An engine passed in as a value is called by the name it is bound to,
    but emitted under its def name: ``func.call @eng`` has no callee."""
    import numpy as np
    import allo.dataflow as df
    from allo.ir.types import uint16

    def twice(x: uint16) -> uint16:
        return x + x

    def make(eng):
        @df.region()
        def top(A: uint16[4], C: uint16[4]):
            @df.kernel(mapping=[1], args=[A, C])
            def k(a: uint16[4], c: uint16[4]):
                for i in range(4):
                    c[i] = eng(a[i])

        return top

    try:
        df.build(make(twice), target="simulator")
        print("C1 built (fixed?)")
    except Exception as e:  # noqa: BLE001
        print("C1", type(e).__name__, str(e).splitlines()[1][:120])


def C2():
    """Two engines with one def name (``add`` in two modules) cannot share a
    region: one symbol, ``redefinition of symbol named 'add'``."""
    import numpy as np
    import allo.dataflow as df
    from allo.ir.types import uint16

    src = {"m1": "def add(x: uint16) -> uint16:\n    return x + 1\n",
           "m2": "def add(x: uint16) -> uint16:\n    return x + 100\n"}
    d = tempfile.mkdtemp()
    for name, body in src.items():
        with open(os.path.join(d, name + ".py"), "w") as f:
            f.write("from allo.ir.types import uint16\n" + body)
    sys.path.insert(0, d)
    import m1
    import m2

    def make(add, add2):
        @df.region()
        def top(A: uint16[4], C: uint16[4], D: uint16[4]):
            @df.kernel(mapping=[1], args=[A, C])
            def k1(a: uint16[4], c: uint16[4]):
                for i in range(4):
                    c[i] = add(a[i])

            @df.kernel(mapping=[1], args=[A, D])
            def k2(a: uint16[4], d: uint16[4]):
                for i in range(4):
                    d[i] = add2(a[i])

        return top

    try:
        df.build(make(m1.add, m2.add), target="simulator")
        print("C2 built (fixed?)")
    except Exception as e:  # noqa: BLE001
        print("C2", type(e).__name__, str(e).splitlines()[1][:120])


def C3():
    """SILENT. A function reused from another module reads the *caller's*
    global of the same name: ``get_global_vars`` flattens every reachable
    module's globals into one dict, first name wins. Expect allo [0 5 10 15]
    where Python gives [0 3 6 9]."""
    import numpy as np
    import allo.dataflow as df
    from allo.ir.types import int32

    d = tempfile.mkdtemp()
    with open(os.path.join(d, "engine_lib.py"), "w") as f:
        f.write("from allo.ir.types import int32\nK = 3\n"
                "def scale(x: int32) -> int32:\n    return x * K\n")
    sys.path.insert(0, d)
    from engine_lib import scale

    g = {"df": df, "int32": int32, "scale": scale, "K": 5}  # the caller's K
    src = '''
@df.region()
def top(A: int32[4], C: int32[4]):
    @df.kernel(mapping=[1], args=[A, C])
    def k(a: int32[4], c: int32[4]):
        for i in range(4):
            c[i] = scale(a[i])
'''
    _exec(src, g, "c3")
    m = df.build(g["top"], target="simulator")
    c = np.zeros(4, np.int32)
    m(np.arange(4, dtype=np.int32), c)
    print("C3 allo", c, "python", np.array([scale(x) for x in range(4)]))


def C4():
    """SILENT. A module-level numpy array shadows a kernel parameter of the
    same name in a sliced read (``infer.py`` ``visit_assignment_val`` looks in
    ``global_vars`` first). Expect allo [7 7 7 7] where the input says [1 1 1 1];
    with a loop index instead of a constant slice it is a ``NameError``."""
    import numpy as np
    import allo.dataflow as df
    from allo.ir.types import int32

    g = {"df": df, "int32": int32, "a": np.full(4, 7, np.int32)}
    src = '''
@df.region()
def top(A: int32[4], C: int32[4]):
    @df.kernel(mapping=[1], args=[A, C])
    def k(a: int32[4], c: int32[4]):
        x: int32[1] = a[1:2]
        for i in range(4):
            c[i] = x[0]
'''
    _exec(src, g, "c4")
    m = df.build(g["top"], target="simulator")
    c = np.zeros(4, np.int32)
    m(np.arange(4, dtype=np.int32), c)
    print("C4 allo", c, "want [1 1 1 1]")


def C5():
    """A call argument is not converted to the parameter's type (an
    assignment is): ``f(b ^ 0x8000)`` passes an i32 to a uint16 parameter."""
    import allo.dataflow as df
    from allo.ir.types import uint16

    def f(x: uint16) -> uint16:
        return x

    @df.region()
    def top(A: uint16[4], C: uint16[4]):
        @df.kernel(mapping=[1], args=[A, C])
        def k(a: uint16[4], c: uint16[4]):
            for i in range(4):
                c[i] = f(a[i] ^ 0x8000)

    try:
        df.build(top, target="simulator")
        print("C5 built (fixed?)")
    except Exception as e:  # noqa: BLE001
        print("C5", type(e).__name__, str(e).splitlines()[1][:120])


def C6():
    """B2 in the ALU's SUB select: ``b[15] ^ (op == 1)`` is an i1 ^ i4 xori,
    because the compare is typed as its operands, not as uint1."""
    import allo.dataflow as df
    from allo.ir.types import UInt, uint16

    @df.region()
    def top(OP: uint16[4], B: uint16[4], C: uint16[4]):
        @df.kernel(mapping=[1], args=[OP, B, C])
        def k(opv: uint16[4], bv: uint16[4], cv: uint16[4]):
            for i in range(4):
                op: UInt(4) = opv[i][0:4]
                b: uint16 = bv[i]
                r: uint16 = b
                r[15] = b[15] ^ (op == 1)
                cv[i] = r

    try:
        df.build(top, target="simulator")
        print("C6 built (fixed?)")
    except Exception as e:  # noqa: BLE001
        print("C6", type(e).__name__, str(e).splitlines()[1][:120])


def C7():
    """Simulator only. A bfloat16 value that merges at a phi (two or more
    conditional assignments) is legalized by x86 LLVM as f32 and truncated
    back by ``__truncsfbf2``, which drops a NaN's payload; one conditional
    assignment (no phi) keeps it. Expect 7fc0 7fc0 ffc0 3f80 for the inputs
    7f81 7fc1 ffff 3f80 passed through untouched (op = 0)."""
    import ml_dtypes
    import numpy as np
    import allo.dataflow as df
    from allo.ir.types import bfloat16, uint16

    @df.region()
    def top(OP: uint16[4], A: bfloat16[4], B: bfloat16[4], C: bfloat16[4]):
        @df.kernel(mapping=[1], args=[OP, A, B, C])
        def k(opv: uint16[4], av: bfloat16[4], bv: bfloat16[4], cv: bfloat16[4]):
            for i in range(4):
                o: uint16 = opv[i]
                x: bfloat16 = av[i]
                y: bfloat16 = bv[i]
                r: bfloat16 = x
                if o == 1:
                    r = x + y
                elif o == 2:
                    r = x - y
                cv[i] = r

    m = df.build(top, target="simulator")
    xs = np.array([0x7F81, 0x7FC1, 0xFFFF, 0x3F80], np.uint16).view(ml_dtypes.bfloat16)
    c = np.zeros(4, ml_dtypes.bfloat16)
    m(np.zeros(4, np.uint16), xs, np.ones(4, ml_dtypes.bfloat16), c)
    print("C7 simulator", " ".join(f"{v:04x}" for v in c.view(np.uint16)), "(input 7f81 7fc1 ffff 3f80)")


def C8():
    """SystemC only. ``max``/``min`` on floats are ``arith.maximumf`` /
    ``minimumf`` (IEEE 754-2019: NaN wins, -0 < +0), which the simulator
    honours; the SystemC emitter prints ``max(a, b)`` bound to ``std::max``,
    which returns ``a`` when the compare is unordered or equal. Expect
    simulator 7fc0 0000, systemc 0000 8000 for max(0, NaN), max(-0, +0)."""
    import ml_dtypes
    import numpy as np
    import allo.dataflow as df
    from allo.ir.types import bfloat16

    @df.region()
    def top(A: bfloat16[2], B: bfloat16[2], C: bfloat16[2]):
        @df.kernel(mapping=[1], args=[A, B, C])
        def k(av: bfloat16[2], bv: bfloat16[2], cv: bfloat16[2]):
            for i in range(2):
                cv[i] = max(av[i], bv[i])

    a = np.array([0x0000, 0x8000], np.uint16).view(ml_dtypes.bfloat16)
    b = np.array([0x7FC0, 0x0000], np.uint16).view(ml_dtypes.bfloat16)
    out = []
    for target, kw in (("simulator", {}),
                       ("systemc", {"mode": "csim", "project": tempfile.mkdtemp()})):
        m = df.build(top, target=target, **kw)
        c = np.zeros(2, ml_dtypes.bfloat16)
        m(a, b, c)
        out.append(f"{target} " + " ".join(f"{v:04x}" for v in c.view(np.uint16)))
    print("C8", "; ".join(out))


def C9():
    """A ``@df.unit``'s names are a snapshot taken at the decorator: setting
    the module global it was sized by afterwards reaches the region (which
    reads it at build) but not the unit. With an array port the memref
    shapes disagree at build; with stream ports only, the unit loops the old
    count and the region deadlocks (not run here)."""
    import allo.dataflow as df

    d = tempfile.mkdtemp()
    with open(os.path.join(d, "sized_unit.py"), "w") as f:
        f.write('''import allo.dataflow as df
from allo.ir.types import int32, Stream
N = 4
@df.unit()
def produce(dst: Stream[int32, 2], mem: int32[N]):
    for i in range(N):
        dst.put(mem[i])
@df.unit()
def consume(src: Stream[int32, 2], mem: int32[N]):
    for i in range(N):
        mem[i] = src.get()
''')
    sys.path.insert(0, d)
    import sized_unit as U
    from allo.ir.types import Stream, int32

    U.N = 8
    n = U.N
    g = {"df": df, "int32": int32, "Stream": Stream, "U": U, "n": n,
         "produce": U.produce, "consume": U.consume}
    src = '''
@df.region()
def top(A: int32[n], B: int32[n]):
    s: Stream[int32, 2]
    produce(dst=s, mem=A)
    consume(src=s, mem=B)
'''
    _exec(src, g, "c9")
    try:
        df.build(g["top"], target="simulator")
        print("C9 built (fixed?)")
    except Exception as e:  # noqa: BLE001
        line = [l for l in str(e).splitlines() if "mismatch" in l]
        print("C9", type(e).__name__, (line or [str(e)])[0][:160])


def _exec(src, g, tag):
    """Define a region from source text, registered so Allo can read it back."""
    import linecache

    fn = f"<repro_{tag}>"
    linecache.cache[fn] = (len(src), None, src.splitlines(True), fn)
    exec(compile(src, fn, "exec"), g)  # pylint: disable=exec-used


REPROS = {k: v for k, v in globals().items() if k[:1] == "C" and k[1:].isdigit()}

if __name__ == "__main__":
    names = sys.argv[1:]
    if len(names) == 1 and names[0] in REPROS and os.environ.get("_REPRO_CHILD"):
        REPROS[names[0]]()
        sys.exit(0)
    for name in names or REPROS:
        env = dict(os.environ, _REPRO_CHILD="1")
        r = subprocess.run([sys.executable, __file__, name], env=env, cwd=ROOT,
                           capture_output=True, text=True, timeout=600)
        lines = [l for l in r.stdout.splitlines() if l.startswith(name + " ")]
        print(lines[-1] if lines else f"{name} no result (exit {r.returncode}): "
              + (r.stderr.strip().splitlines() or ["?"])[-1][:160], flush=True)
