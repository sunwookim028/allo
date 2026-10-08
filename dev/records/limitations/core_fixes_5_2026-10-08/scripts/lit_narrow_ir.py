import re
import allo
from allo.ir.types import int32


def lit_narrow(a: int32[4], o: int32[4]):
    for i in range(4):
        x: int32 = a[i]
        o[i] = ((x | (1 << 4)) & 0xFFFFFFFF) + 300 - (x >> 31)


for l in str(allo.customize(lit_narrow).module).splitlines():
    if "arith." in l:
        print(re.sub(r"%[\w]+", "%v", l.strip()))
