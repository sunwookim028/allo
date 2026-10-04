"""compose.Unit.check: a nested helper ``def`` inside a unit body is reported as an undeclared free name."""
from __future__ import annotations
from allo.compose import unit

@unit(instances=("1",), reads=("a",), writes=("b",), parameters=("N",))
def body():
    def helper(v: UInt(8)) -> UInt(8):
        return v + 1
    for t in range(N):
        x: UInt(8) = a.get()
        b.put(helper(x))

try:
    body.check()
    print("accepted")
except AssertionError as e:
    print("refused:", e)
