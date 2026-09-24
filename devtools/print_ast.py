# Print a compact AST of one function in a Python file.
# Usage:  python print_ast.py <file.py> [function_name]
#   e.g.  python print_ast.py ../examples/stream_producer_consumer.py top
# Pure stdlib — no allo / conda env needed.
import ast
import os
import sys

# Default to the example this tool was written against; it lives in examples/,
# while this script lives in devtools/.
_DEFAULT = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "examples", "stream_producer_consumer.py")

path = sys.argv[1] if len(sys.argv) > 1 else _DEFAULT
want = sys.argv[2] if len(sys.argv) > 2 else None

tree = ast.parse(open(path).read())
if want:
    node = next(n for n in ast.walk(tree)
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == want)
else:
    node = tree


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


show(node)
