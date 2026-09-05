# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""What the co-design agent is allowed to put in a spec file.

The two writable files are imported by the evaluator, so anything at their
import path executes inside the scoring process. These tests pin that boundary,
including a regression for the self-rewriting module a live agent run actually
produced.
"""

import sys
from pathlib import Path

import pytest

AGENT_DIR = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "accelerator"
    / "tinytpu"
    / "chia_agent"
)
sys.path.insert(0, str(AGENT_DIR))

from spec_policy import policy_violations  # noqa: E402

TINYTPU_DIR = AGENT_DIR.parent


@pytest.mark.parametrize("name", ["isa.py", "microarch.py"])
def test_the_real_spec_satisfies_its_own_policy(name):
    """The policy must accept everything the shipped spec already does --
    module-level schedule calls, the bind loop, an ``if __name__`` guard."""
    source = (TINYTPU_DIR / name).read_text(encoding="utf-8")
    assert policy_violations(name, source) == []


def test_self_rewriting_module_is_refused():
    """Regression: a live agent wrote this shape into isa.py to revert its own
    failed candidate. It runs at import, inside the process that scores it."""
    source = """
import os
with open(__file__, 'r') as f:
    lines = f.readlines()
with open(__file__, 'w') as f:
    f.write(''.join(lines))
"""
    problems = policy_violations("isa.py", source)
    assert any("imports 'os'" in p for p in problems)
    assert any("uses 'open'" in p for p in problems)
    assert any("module-level with block" in p for p in problems)


@pytest.mark.parametrize(
    "source, expected",
    [
        ("import subprocess\n", "imports 'subprocess'"),
        ("from pathlib import Path\nP = Path('x').write_text('y')\n", "'.write_text'"),
        ("X = eval('1+1')\n", "uses 'eval'"),
        ("X = __import__('os')\n", "uses '__import__'"),
        ("import shutil\n", "imports 'shutil'"),
    ],
)
def test_escape_routes_are_refused(source, expected):
    problems = policy_violations("isa.py", source)
    assert any(expected in p for p in problems), problems


def test_legitimate_hardware_description_is_accepted():
    """The policy must not get in the way of ordinary co-design edits."""
    source = """
from allo.lang.core import f32, i32
from .isa import tpu

OP_VADD = 0

@tpu.unit
def vpu(a: i32, b: i32) -> i32:
    return a + b

vpu_s = vpu.schedule()
vpu_s.pipeline("lane")
tpu.latency(vpu, ii=1, depth=5)
for _op in (1, 2, 3):
    tpu.bind(_op, vpu, trips=lambda: 8)
"""
    assert policy_violations("microarch.py", source) == []
