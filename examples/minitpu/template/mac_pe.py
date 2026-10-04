# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""One PE source, two MAC engines (Q1 / H10).

``mac_pe`` is the MAC datapath of a weight-stationary PE: per work item it
takes an operand, a weight and the partial sum from the north and hands
``MAC_ADD(MAC_MUL(a, w), psum)`` south. The engine is bound at composition:
``MAC_IN``.. ``MAC_ADD`` are PARAMETERS of the unit (an ``Engine.namespace()``),
exactly as ``RED_IN``/``RED_ACC`` are parameters of ``reduce_tree``. Nothing
in the body names bf16 or int8.

Three rigs: the PE at the bf16->acc24 engine, at the int8->int32 engine, and
BOTH in one region (two instances of one unit, each at its own engine, via
``instantiate.instance``). The weight-bank bookkeeping of ``mxu_pe.sv``
(two pending banks, commit, forwards) is track B's P2, not this file's.
"""

from __future__ import annotations

import numpy as np

import allo.dataflow as df
from allo.compose import Architecture, Channel, Memory, unit

from examples.minitpu.template.engines import BF16_ACC24, INT8_INT32, Engine
from examples.minitpu.template.instantiate import bound, instance


@unit(
    reads=("lhs", "wq", "psum_in"),
    writes=("psum_out",),
    parameters=("N_WORK", "MAC_IN", "MAC_ACC", "MAC_MUL", "MAC_ADD"),
)
def mac_pe():
    for work in range(N_WORK):
        a: MAC_IN = lhs.get()
        w: MAC_IN = wq.get()
        north: MAC_ACC = psum_in.get()
        product: MAC_ACC = MAC_MUL(a, w)
        south: MAC_ACC = MAC_ADD(product, north)
        psum_out.put(south)


@unit(
    memories=("A", "W", "P"),
    writes=("lhs", "wq", "psum_in"),
    parameters=("N_WORK", "MAC_IN", "MAC_ACC"),
)
def pe_feed(a_mem: UInt(32)[N_WORK], w_mem: UInt(32)[N_WORK], p_mem: UInt(32)[N_WORK]):
    for work in range(N_WORK):
        a_word: UInt(32) = a_mem[work]
        w_word: UInt(32) = w_mem[work]
        p_word: UInt(32) = p_mem[work]
        # Typed assignment, not a slice: a slice whose bound is a NAME has
        # no inferable width (tinytpu_library.rst, "symbolic slice"), so a
        # type parameter cannot appear in a slice bound today.
        a: MAC_IN = a_word
        w: MAC_IN = w_word
        p: MAC_ACC = p_word
        lhs.put(a)
        wq.put(w)
        psum_in.put(p)


@unit(
    memories=("OUT",),
    reads=("psum_out",),
    parameters=("N_WORK", "MAC_ACC"),
)
def pe_sink(out_mem: UInt(32)[N_WORK]):
    for work in range(N_WORK):
        south: MAC_ACC = psum_out.get()
        word: UInt(32) = south
        out_mem[work] = word


def _channels(suffix=""):
    s = f"_{suffix}" if suffix else ""
    return (Channel(f"lhs{s}", f"MAC_IN{'__' + suffix if suffix else ''}", "4"),
            Channel(f"wq{s}", f"MAC_IN{'__' + suffix if suffix else ''}", "4"),
            Channel(f"psum_in{s}", f"MAC_ACC{'__' + suffix if suffix else ''}", "4"),
            Channel(f"psum_out{s}", f"MAC_ACC{'__' + suffix if suffix else ''}", "4"))


def pe_rig(engine: Engine, n: int, name=None) -> Architecture:
    """The PE at one engine, in its own region."""
    params = {"N_WORK": n, "QD": 4} | engine.namespace()
    return Architecture(
        name=name or f"pe_{engine.name}", parameters=params,
        memories=(Memory("A", "UInt(32)[N_WORK]"), Memory("W", "UInt(32)[N_WORK]"),
                  Memory("P", "UInt(32)[N_WORK]"), Memory("OUT", "UInt(32)[N_WORK]")),
        channels=_channels(),
        units=(pe_feed, instance(mac_pe, "mac_pe", directives=engine.directives), pe_sink))


def pe_rig_two(eng_a: Engine, eng_b: Engine, n: int, name="pe_two") -> Architecture:
    """Two PEs, two engines, ONE region: each instance binds its own engine
    names and its own channels; the feed and sink are instantiated twice too."""
    params = {"N_WORK": n, "QD": 4}
    units, channels, memories = [], [], []
    for suffix, eng in (("a", eng_a), ("b", eng_b)):
        ns, bind = bound(eng.namespace(), suffix)
        params.update(ns)
        chan = {c: f"{c}_{suffix}" for c in ("lhs", "wq", "psum_in", "psum_out")}
        mems = {m: f"{m}_{suffix}" for m in ("A", "W", "P", "OUT")}
        memories += [Memory(mems[m], "UInt(32)[N_WORK]") for m in ("A", "W", "P", "OUT")]
        channels += _channels(suffix)
        units += [
            instance(pe_feed, f"pe_feed_{suffix}", bind | chan,
                     memories=tuple(mems[m] for m in ("A", "W", "P"))),
            instance(mac_pe, f"mac_pe_{suffix}", bind | chan, directives=eng.directives),
            instance(pe_sink, f"pe_sink_{suffix}", bind | chan, memories=(mems["OUT"],)),
        ]
    return Architecture(name=name, parameters=params, memories=tuple(memories),
                        channels=tuple(channels), units=tuple(units))


# --- stimulus and reference ---------------------------------------------------

def stimulus(engine: Engine, n: int, seed=0):
    rng = np.random.default_rng(seed)
    if engine is BF16_ACC24:
        from examples.minitpu.units.mxu_pe import acc24, bf16
        import random
        r = random.Random(seed)
        a = np.array([bf16(r) for _ in range(n)], dtype=np.uint32)
        w = np.array([bf16(r) for _ in range(n)], dtype=np.uint32)
        p = np.array([acc24(r) for _ in range(n)], dtype=np.uint32)
    else:
        a = rng.integers(-128, 128, n).astype(np.int8).astype(np.uint8).astype(np.uint32)
        w = rng.integers(-128, 128, n).astype(np.int8).astype(np.uint8).astype(np.uint32)
        p = rng.integers(-2**20, 2**20, n).astype(np.int32).astype(np.uint32)
    return a, w, p


def reference(engine: Engine, a, w, p):
    """``MAC_ADD(MAC_MUL(a, w), p)`` in the engine's own numpy arithmetic,
    returned as the ``ACC_BITS``-bit pattern in a uint32."""
    if engine is BF16_ACC24:
        return engine.ref_add(engine.ref_mul(a, w), p).astype(np.uint32)
    sa = a.astype(np.uint8).astype(np.int8).astype(np.int64)
    sw = w.astype(np.uint8).astype(np.int8).astype(np.int64)
    sp = p.astype(np.uint32).astype(np.int32).astype(np.int64)
    return engine.ref_add(engine.ref_mul(sa, sw), sp).astype(np.int32).astype(np.uint32)


def run_rig(arch: Architecture, engine: Engine, n: int, seed=0):
    mod = df.build(arch.region(), target="simulator")
    a, w, p = stimulus(engine, n, seed)
    out = np.zeros(n, dtype=np.uint32)
    mod(a, w, p, out)
    want = reference(engine, a, w, p)
    return out, want


def run_rig_two(arch: Architecture, eng_a: Engine, eng_b: Engine, n: int, seed=0):
    mod = df.build(arch.region(), target="simulator")
    aa, wa, pa = stimulus(eng_a, n, seed)
    ab, wb, pb = stimulus(eng_b, n, seed + 1)
    oa = np.zeros(n, dtype=np.uint32)
    ob = np.zeros(n, dtype=np.uint32)
    mod(aa, wa, pa, oa, ab, wb, pb, ob)
    return (oa, reference(eng_a, aa, wa, pa)), (ob, reference(eng_b, ab, wb, pb))
