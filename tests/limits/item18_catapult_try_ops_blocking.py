# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Item 18: Catapult lowers try_get/try_put to BLOCKING read/write with the
success flag hard-coded `true`, silently.

Emits the same arbitration kernel (try channel 0, else try channel 1) for
Vivado HLS and Catapult (codegen only; Catapult is not installed here) and
checks what each backend makes of it. Also reports `full()`, which the item
does not mention."""
import re
import _worktree
from _worktree import verdict

ITEM = 18
import allo
import allo.dataflow as df
from allo.ir.types import Stream, int32


@df.region()
def top():
    S: Stream[int32, 2][2]
    O: Stream[int32, 2][1]

    @df.kernel(mapping=[1])
    def arb():
        d0, ok0 = S[0].try_get()
        if ok0:
            O[0].put(d0)
        else:  # dead code if ok0 is a compile-time true
            d1, ok1 = S[1].try_get()
            if ok1:
                O[0].put(d1)
        if not O[0].full():
            pass


def code_for(target):
    return allo.customize(top).build(target=target).hls_code


def main():
    res = {}
    vh = code_for("vhls")
    res["vhls read_nb"] = ".read_nb(" in vh
    try:
        cat = code_for("catapult")
        res["catapult nb_read"] = "nb_read(" in cat
        res["catapult success hard-coded true"] = bool(re.search(r"bool \w+ = true;", cat))
        res["catapult full() hard-coded false"] = bool(re.search(r"= false;\s*/\* ac_channel: no \.full\(\)", cat))
        res["catapult warns"] = ("#warning" in cat or "#error" in cat)
    except (Exception, SystemExit) as e:  # noqa: BLE001
        res["catapult"] = f"{type(e).__name__}: {str(e)[:100]}"
    print(res)
    verdict(ITEM, res.get("catapult success hard-coded true", False) and not res.get("catapult warns"), str(res))


if __name__ == "__main__":
    _worktree.run_guarded(ITEM, main)
