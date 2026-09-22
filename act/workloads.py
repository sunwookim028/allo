# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The registry. One entry per workload -- adding one is adding one entry."""

from act.workload import Tensor, Workload

MKN = ("M", "K", "N")
A = Tensor("A", ("M", "K"))
B = Tensor("B", ("K", "N"))
B2 = Tensor("B2", ("K", "N"))
C = Tensor("C", ("M", "N"))


def gemm(name, contractions, epilogue=(), operands=(A, B)):
    return Workload(name=name, ranks=MKN, reduce=("K",), operands=operands,
                    result=C, contractions=contractions, epilogue=epilogue)


WORKLOADS = {w.name: w for w in (
    gemm("gemm", (("A", "B"),)),
    gemm("gemm.relu", (("A", "B"),), ("relu",)),
    gemm("gemm.sum", (("A", "B"), ("A", "B2")), operands=(A, B, B2)),
    gemm("gemm.sum.relu", (("A", "B"), ("A", "B2")), ("relu",),
         operands=(A, B, B2)),
)}


def get(name):
    if name not in WORKLOADS:
        raise KeyError(
            f"no workload {name!r}; registered: {sorted(WORKLOADS)}. Add one "
            f"to act/workloads.py")
    return WORKLOADS[name]
