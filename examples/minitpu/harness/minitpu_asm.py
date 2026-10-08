# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Import MiniTPU's own assembler (``board_package/asm.py``) without writing
into the read-only clone (U4).

``asm.py`` is imported from the pinned clone as data: no ``__pycache__`` is
written beside it, and its on-disk schedule cache is off (it would otherwise
write under ``~/.cache``). ``load()`` returns the module.
"""

import importlib
import os
import sys

from examples.minitpu.harness import rtl

_MOD = None


def load():
    global _MOD
    if _MOD is None:
        os.environ["MINITPU_SCHEDULE_CACHE"] = "off"
        prev = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        path = os.path.join(rtl.minitpu_home(), "board_package")
        sys.path.insert(0, path)
        try:
            _MOD = importlib.import_module("asm")
        finally:
            sys.path.remove(path)
            sys.dont_write_bytecode = prev
    return _MOD
