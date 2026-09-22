# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a generated program reaches the judge: an importable callable, or words.

`module:function` is called with the spec and must return the
`(instruction word, AGU word)` list `assemble` takes. A `.json` file instead
holds those words directly, which is what an agent that emits text rather than
Python should produce."""

import importlib
import json

BASELINE = "examples.accelerator.tinytpu_vitis.act.baseline:program"


class SubmissionError(ValueError):
    """A submission the judge cannot even read, named at the point it fails."""


def _from_json(path):
    with open(path) as f:
        words = json.load(f)
    if not isinstance(words, list) or not words:
        raise SubmissionError(
            f"{path}: expected a non-empty JSON list of [instruction_word, "
            f"agu_word] pairs")
    prog = []
    for i, pair in enumerate(words):
        if not isinstance(pair, list) or len(pair) != 2:
            raise SubmissionError(
                f"{path}: entry {i} is {pair!r}; every entry is a pair "
                f"[instruction_word, agu_word] of unsigned 64-bit integers")
        prog.append((int(pair[0]), int(pair[1])))
    return prog


def load(ref):
    """A callable taking a spec and returning a program."""
    if ref.endswith(".json"):
        prog = _from_json(ref)
        return lambda sp: prog
    if ":" not in ref:
        raise SubmissionError(
            f"{ref!r} is neither a .json file of instruction words nor a "
            f"'module:function' reference")
    mod, fn = ref.split(":", 1)
    try:
        return getattr(importlib.import_module(mod), fn)
    except (ImportError, AttributeError) as e:
        raise SubmissionError(f"cannot load {ref!r}: {e}") from e


def dump(prog, path):
    with open(path, "w") as f:
        json.dump([[int(w0), int(w1)] for w0, w1 in prog], f)
        f.write("\n")
    return path
