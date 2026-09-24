# The intermediate between AST and MLIR: the TYPE-ANNOTATED AST.
# Runs Pass 1 (TypeInferer) ONLY -- no MLIR emitted -- then prints each node
# with the dtype/shape that inference attached to it.
#
# Usage:  conda run -n allo python print_typed_ast.py
import ast
import sys

sys.path.insert(0, "/home/zsm9/allo_sup")
sys.path.insert(0, "/home/zsm9/allo_sup/examples")

from allo._mlir.ir import Context
from allo.ir.utils import parse_ast, get_global_vars
from allo.ir.visitor import ASTContext
from allo.ir.infer import TypeInferer
from stream_producer_consumer import top

tree = parse_ast(top)
gvars = get_global_vars(top)

# Replicate exactly what customize() does for Pass 1
ctx = ASTContext(
    tree=tree,
    global_vars=gvars.copy(),
    mlir_ctx=Context(),
    inst=[],
    unroll=True,
    enable_tensor=False,
    typing_rule_set="default",
    verbose=False,
)
tree = TypeInferer()(ctx, tree)      # <-- Pass 1: fills in node.dtype / node.shape


def label(n):
    for f in ("name", "id", "attr", "arg"):
        v = getattr(n, f, None)
        if isinstance(v, str):
            base = f"{type(n).__name__}({v!r})"
            break
    else:
        base = f"Constant({n.value!r})" if isinstance(n, ast.Constant) else type(n).__name__
    # append the inferred type, if any
    if hasattr(n, "dtype"):
        base += f"   ::  dtype={n.dtype!r}  shape={getattr(n, 'shape', None)}"
    return base


def show(n, depth=0):
    print("  " * depth + label(n))
    for child in ast.iter_child_nodes(n):
        show(child, depth + 1)


print("TYPE-ANNOTATED AST  (Pass 1 output — between AST and MLIR)")
print("=" * 64)
show(tree)
