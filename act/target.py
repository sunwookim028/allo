# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What a machine must answer for the search to run over it."""

TARGETS = {}


class Target:
    name = "abstract"

    def intrinsics(self, workload, extents):
        raise NotImplementedError

    def lower(self, workload, extents, nest):
        raise NotImplementedError

    def steps(self, program):
        raise NotImplementedError

    def emits(self, program):
        raise NotImplementedError

    def verify(self, workload, extents, program, seed=0):
        raise NotImplementedError

    def report(self, program):
        return {}


def register(target):
    TARGETS[target.name] = target
    return target


def get(name):
    if name not in TARGETS:
        raise KeyError(
            f"no target {name!r}; registered: {sorted(TARGETS)}. Register one "
            f"with act.target.register()")
    return TARGETS[name]
