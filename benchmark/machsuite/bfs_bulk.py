# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Breadth-first search by level sweeps over the whole frontier each round."""

import numpy as np

from allo.lang import i32, kernel

from ..spec import Benchmark

N_NODES = 32
N_NODES_2 = N_NODES * 2
N_EDGES = 128
N_LEVELS = 6
MAX_LEVEL = 999999


def build():
    @kernel
    def bfs_bulk(
        nodes: i32[N_NODES_2],
        edges: i32[N_EDGES],
        starting_node: i32,
        level: i32[N_NODES],
        level_counts: i32[N_LEVELS],
    ):
        for init in range(N_NODES):
            level[init] = MAX_LEVEL
        for init2 in range(N_LEVELS):
            level_counts[init2] = 0
        level[starting_node] = 0
        level_counts[0] = 1

        for horizon in range(N_LEVELS):
            cnt: i32 = 0
            horizon_i32: i32 = horizon
            for n in range(N_NODES):
                if level[n] == horizon_i32:
                    tmp_begin: i32 = nodes[2 * n]
                    tmp_end: i32 = nodes[2 * n + 1]
                    for e in range(tmp_begin, tmp_end):
                        tmp_dst: i32 = edges[e]
                        tmp_level: i32 = level[tmp_dst]
                        if tmp_level == MAX_LEVEL:
                            level[tmp_dst] = horizon_i32 + 1
                            cnt += 1
            if cnt != 0:
                level_counts[horizon + 1] = cnt

    return {"top": bfs_bulk}


def _none(parts):
    return parts["top"].schedule()


def _graph(rng):
    nodes = np.zeros(N_NODES_2, np.int32)
    edges = np.zeros(N_EDGES, np.int32)
    fanout = min(N_NODES - 1, N_EDGES)
    nodes.fill(fanout)
    nodes[0] = 0
    nodes[1] = fanout
    edges[:fanout] = np.arange(1, fanout + 1, dtype=np.int32)
    for node in range(1, N_NODES):
        nodes[2 * node] = fanout
        nodes[2 * node + 1] = fanout
    return nodes, edges


def inputs(rng):
    nodes, edges = _graph(rng)
    return (
        nodes,
        edges,
        np.int32(0),
        np.zeros(N_NODES, np.int32),
        np.zeros(N_LEVELS, np.int32),
    )


def reference(nodes, edges, starting_node, level, level_counts):
    lv = np.full(N_NODES, MAX_LEVEL, np.int32)
    counts = np.zeros(N_LEVELS, np.int32)
    lv[starting_node] = 0
    counts[0] = 1
    for horizon in range(N_LEVELS):
        cnt = 0
        for n in range(N_NODES):
            if lv[n] == horizon:
                for e in range(nodes[2 * n], nodes[2 * n + 1]):
                    dst = edges[e]
                    if lv[dst] == MAX_LEVEL:
                        lv[dst] = horizon + 1
                        cnt += 1
        if cnt != 0:
            counts[horizon + 1] = cnt
    return lv, counts


BENCHMARK = Benchmark(
    suite="machsuite",
    name="bfs_bulk",
    build=build,
    schedules={"none": _none},
    inputs=inputs,
    reference=reference,
    outputs=(3, 4),
)
