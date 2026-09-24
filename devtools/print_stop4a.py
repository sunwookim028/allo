# Stop 4a: after move_stream_to_interface, BEFORE _build_top.
# Kernels now take the stream as an argument (%arg1: !allo.stream), local
# constructs erased, direction tagged (stypes "_o"/"_i") -- but `top` is not
# yet wired (still has its stray construct, no calls).
#
# The three Stop-4 snapshots:
#   print_stop3_output.py  -> BEFORE Stop 4  (3 local constructs)
#   print_stop4a.py        -> AFTER 4a       (this script)
#   print_mlir.py          -> AFTER 4b       (final: top wired)
#
# Usage:  conda run -n allo python print_stop4a.py
import sys

sys.path.insert(0, "/home/zsm9/allo_sup")
sys.path.insert(0, "/home/zsm9/allo_sup/examples")

from allo.dataflow import _customize, move_stream_to_interface
from stream_producer_consumer import top

s = _customize(top)                       # Stop 3
info = move_stream_to_interface(s)        # Stop 4a ONLY (no _build_top)

print("# stream_info (the 4a -> 4b bridge):", info)
print("# " + "-" * 60)
print(s.module)
