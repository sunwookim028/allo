# Inspect the ACTUAL input to Stop 3 (infer + build):
#   (1) the AST tree returned by parse_ast(top)
#   (2) the global_vars dict returned by get_global_vars(top)
# These are exactly the two things customize() feeds into TypeInferer/ASTTransformer.
#
# Usage:  python print_stop3_input.py
import ast
import sys

sys.path.insert(0, "/home/zsm9/allo_sup")
sys.path.insert(0, "/home/zsm9/allo_sup/examples")

from allo.ir.utils import parse_ast, get_global_vars      # the very calls customize() makes
from stream_producer_consumer import top

tree = parse_ast(top)                 # input (1): the AST  (Stop 1's output)
gvars = get_global_vars(top)          # input (2): resolved names/constants


# ---- compact AST printer (same style as print_ast.py) ----
def label(n):
    for f in ("name", "id", "attr", "arg"):
        v = getattr(n, f, None)
        if isinstance(v, str):
            return f"{type(n).__name__}({v!r})"
    if isinstance(n, ast.Constant):
        return f"Constant({n.value!r})"
    return type(n).__name__


def show(n, depth=0):
    print("  " * depth + label(n))
    for child in ast.iter_child_nodes(n):
        show(child, depth + 1)


print("=" * 64)
print("INPUT (1): the AST  (what parse_ast returned)")
print("=" * 64)
show(tree)

print()
print("=" * 64)
print("INPUT (2): global_vars  (only the names your kernel references)")
print("=" * 64)
# collect every Name(id=...) used anywhere in the kernel's AST
used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
for k in sorted(used):
    if k in gvars:
        print(f"  {k!r:10} -> {gvars[k]!r}")
    else:
        print(f"  {k!r:10} -> (not a global: a local/loop var or builtin)")
