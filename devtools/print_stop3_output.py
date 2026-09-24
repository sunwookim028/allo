# Stop 3 OUTPUT: the raw builder result, BEFORE interface lifting (Stop 4).
# Uses _customize(top) only (parse -> infer -> build), skipping the dataflow
# finishers (move_stream_to_interface + _build_top).
#
# Compare with print_mlir.py, which runs the full df.customize (Stop 3 + Stop 4).
#
# Usage:  python print_stop3_output.py
import sys

sys.path.insert(0, "/home/zsm9/allo_sup")
sys.path.insert(0, "/home/zsm9/allo_sup/examples")

from allo.dataflow import _customize          # the generic customize (Stop 3 only)
from stream_producer_consumer import top

s = _customize(top)
print("# Stop 3 output — raw builder, BEFORE interface lifting")
print("# (3 separate `pipe` constructs; streams are locals, not args; top has no calls)")
print("# " + "-" * 60)
print(s.module)
