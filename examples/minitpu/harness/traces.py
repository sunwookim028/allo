# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Command traces for storage units (U2): building, random, directed.

A trace is ``{input port: list of ints}``, one entry per cycle, driven by
``rtl.run_trace`` and read by the unit's trace reference (``ref.py``). Each
unit file builds its own from these pieces and tags every trace *legal* (it
obeys MiniTPU's rules for the unit) or *illegal*; illegal traces are kept, so
the census shows what the RTL does there.
"""

import random


class Trace:
    """Build a command trace cycle by cycle; unset ports hold ``defaults``."""

    def __init__(self, defaults):
        self.defaults = dict(defaults)
        self.rows = []

    def cycle(self, **vals):
        bad = set(vals) - set(self.defaults)
        assert not bad, f"unknown ports {bad}"
        self.rows.append({**self.defaults, **vals})
        return self

    def idle(self, k=1, **vals):
        for _ in range(k):
            self.cycle(**vals)
        return self

    def __len__(self):
        return len(self.rows)

    def cmd(self):
        return {p: [r[p] for r in self.rows] for p in self.defaults}


def concat(*cmds):
    """Join command traces end to end."""
    out = {p: [] for p in cmds[0]}
    for c in cmds:
        for p in out:
            out[p] += list(c[p])
    return out


def word(rng, width):
    """A random ``width``-bit word, biased to all-zeros/all-ones/one-hot now and then."""
    r = rng.random()
    if r < 0.05:
        return 0
    if r < 0.10:
        return (1 << width) - 1
    if r < 0.15:
        return 1 << rng.randrange(width)
    return rng.getrandbits(width)


def hot_addr(rng, space, hot, p_hot=0.75):
    """An address drawn from a small hot set most of the time, so read/write
    distances 0..4 on every port pair occur often."""
    if rng.random() < p_hot:
        return rng.choice(hot)
    return rng.randrange(space)


def rng_for(*key):
    """A ``random.Random`` seeded by ``key`` (deterministic across runs:
    ``str`` seeds are hashed with SHA-512, not ``hash()``)."""
    return random.Random(repr(("u2", *key)))
