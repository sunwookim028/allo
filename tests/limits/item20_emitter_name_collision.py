# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item 20: the HLS emitter can give a local the same C identifier as one of
the function's parameters (HLS 207-3746 "subscripted value is not an array").

Minimal trigger, from the fixing commit aece11c9: a loop variable whose
explicit name (`v1`) is also the default name the generator hands the second
parameter. Plus the reported dataflow shape: a kernel whose third parameter
is a stream and whose body is a meta_for making many temporaries. Codegen
only; checks every emitted function for a body declaration that reuses a
parameter name."""
import re
import _worktree
from _worktree import verdict

ITEM = 20
import allo
import allo.dataflow as df
from allo.ir.types import Stream, int8, int32


def loopvar_named_v1(A: int32[8], B: int32[8]):
    for v1 in allo.grid(8):
        B[v1] = A[v1] + 1


@df.region()
def mover_region(ctl: int32[8], A: int8[256], B: int32[256]):
    s: Stream[int32, 4]

    @df.kernel(mapping=[1], args=[ctl, A])
    def mover(c: int32[8], a: int8[256]):
        off: int32 = c[0]
        for r in range(c[1]):
            acc: int32 = 0
            with allo.meta_for(4) as e:
                v: int8 = a[(off + r) * 16 + e]
                acc = acc + v
            s.put(acc)

    @df.kernel(mapping=[1], args=[ctl, B])
    def sink(c2: int32[8], b: int32[256]):
        for r in range(c2[1]):
            b[r] = s.get()


def clashes(code):
    out = []
    for m in re.finditer(r"\nvoid (\w+)\(([^)]*)\)\s*\{(.*?)\n\}", code, re.S):
        params = set(re.findall(r"\b(v\d+|\w+)\s*(?:\[[^\]]*\])*\s*(?:,|$)", m.group(2).strip()))
        params |= set(re.findall(r"&\s*(\w+)", m.group(2)))
        decls = set(re.findall(r"^\s+(?:[\w:]+(?:<[^;]*?>)?\s+)+(\w+)(?:\[[^\]]*\])*\s*(?:;|=)", m.group(3), re.M))
        decls |= set(re.findall(r"for\s*\(\s*\w+\s+(\w+)\s*=", m.group(3)))
        both = sorted(params & decls)
        if both:
            out.append((m.group(1), both))
    return out


def main():
    res = {}
    code1 = allo.customize(loopvar_named_v1).build(target="vhls").hls_code
    res["loop var named v1"] = clashes(code1)
    code2 = df.customize(mover_region).build(target="vhls").hls_code
    res["mover (3rd param a stream)"] = clashes(code2)
    print(res)
    verdict(ITEM, any(res.values()), str(res))


if __name__ == "__main__":
    _worktree.run_guarded(ITEM, main)
