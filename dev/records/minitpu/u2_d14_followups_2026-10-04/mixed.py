"""Mixed-reset test design (D-14 follow-up a): one Wire-only kernel `rf` with
unreset storage `mem` (comb read port), reset storage `tag` (thread-written),
and a reset counter `cnt`."""
import allo.dataflow as df
from allo.ir.types import Stateful, UInt, Wire, comb, int32, uint1

A5 = UInt(5)
D16 = UInt(16)


def mixed(n):
    @df.region()
    def top(RA: A5[n], WA: A5[n], WD: D16[n], WE: uint1[n], QA: D16[n], QT: D16[n], QC: D16[n]):
        w_ra: Wire[A5]
        w_wa: Wire[A5]
        w_wd: Wire[D16]
        w_we: Wire[uint1]
        w_qa: Wire[D16, comb]
        w_qt: Wire[D16]
        w_qc: Wire[D16]

        @df.kernel(mapping=[1], args=[RA, WA, WD, WE])
        def src(ra: A5[n], wa: A5[n], wd: D16[n], we: uint1[n]):
            for t in range(n):
                w_ra.put(ra[t])
                w_wa.put(wa[t])
                w_wd.put(wd[t])
                w_we.put(we[t])

        @df.kernel(mapping=[1], args=[])
        def rf():
            mem: D16[32] @ Stateful(reset=False)
            tag: D16[4] @ Stateful = 0
            cnt: D16[1] @ Stateful = 0
            for _ in range(n):
                a5: A5 = w_ra.get()
                a: int32 = a5
                x5: A5 = w_wa.get()
                x: int32 = x5
                d: D16 = w_wd.get()
                e: uint1 = w_we.get()
                w_qa.put(mem[a])
                w_qt.put(tag[a & 3])
                c: D16 = cnt[0]
                w_qc.put(c)
                cnt[0] = c + 1
                if e:
                    mem[x] = d
                    tag[x & 3] = d

        @df.kernel(mapping=[1], args=[QA, QT, QC])
        def sink(qa: D16[n], qt: D16[n], qc: D16[n]):
            for t in range(n):
                qa[t] = w_qa.get()
                qt[t] = w_qt.get()
                qc[t] = w_qc.get()

    return top
