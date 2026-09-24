# Print the Stop-2 output: the Schedule and its frontend MLIR module.
# Usage:  OMP_NUM_THREADS=4 python print_mlir.py [function_name]
#   no arg      -> whole module
#   producer_0  -> just that one function
# Always imports allo from /home/zsm9/allo_sup (avoids the installed-package trap).
import os
import sys

sys.path.insert(0, "/home/zsm9/allo_sup")
sys.path.insert(0, "/home/zsm9/allo_sup/examples")  # the design lives in examples/

import allo.dataflow as df
from stream_producer_consumer import top

s = df.customize(top)                       # Schedule (Stop 2 output)
want = sys.argv[1] if len(sys.argv) > 1 else None

print("# Schedule object :", type(s).__name__)
print("# top_func_name   :", s.top_func_name)
print("# functions       :", [op.name.value for op in s.module.body.operations
                              if hasattr(op, "name")])
print("# " + "-" * 60)

if want:
    for op in s.module.body.operations:
        if hasattr(op, "name") and op.name.value == want:
            print(op)
else:
    print(s.module)
