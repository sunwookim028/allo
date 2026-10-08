# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""The walk-through's commands still do what examples/tinytpu/README.md says
they do: the walk-through compiles and matches PyTorch, a model the ISA cannot
express is refused, and the fused-instruction patch still applies to this
tree. No Vitis, no design build."""

import os
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT)
pytest.importorskip("torch", reason="the tutorial's front end is torch.fx")
pytest.importorskip("allo._mlir", reason="the specs are validated against the build")

from examples.tinytpu.workloads import demo  # noqa: E402

PATCH = os.path.join(ROOT, "examples", "tinytpu", "mvoutrelu.patch")


def test_the_walk_through_compiles_and_matches_pytorch(capsys):
    assert demo.main(["mlp_small", "--no-sim"]) == 0
    out = capsys.readouterr().out
    assert "0 of 384 output bytes differ" in out
    assert "vrelu" in out and "mvoutrelu" not in out, (
        "the unpatched tree retires a ReLU layer with vrelu then mvout")


def test_a_model_the_isa_cannot_express_is_refused(capsys):
    assert demo.main(["mlp_bias", "--no-sim"]) == 1
    assert "REFUSED" in capsys.readouterr().out


def test_the_fused_instruction_patch_still_applies():
    """The patch applies to this tree. A failure here means a file it edits
    has moved on: regenerate it against the tree (apply by hand, run
    `gen_isa.py --write` and `lift_units.py`, then `git diff > mvoutrelu.patch`)."""
    r = subprocess.run(["git", "apply", "--check", PATCH], cwd=ROOT,
                       capture_output=True, text=True, check=False)
    assert r.returncode == 0, r.stderr
