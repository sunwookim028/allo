"""Greedy line-level reduction of an AMC crash: drop each line (then each
if-block) while the build still dies with the same message."""
import os, sys, subprocess, ast
TMP = os.environ["TMPDIR"]
src, sig = sys.argv[1], sys.argv[2]
L = open(src).read().replace("N = 40214", "N = 64").split("\n")
RUN = '''import importlib.util, sys
spec = importlib.util.spec_from_file_location("m", sys.argv[1]); K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
import allo; allo.customize(K.saguk).build(target="amc"); print("BUILT")'''
open(f"{TMP}/ddrun.py", "w").write(RUN)
def crashes(lines):
    txt = "\n".join(l for l in lines if l is not None)
    try: ast.parse(txt)
    except SyntaxError: return False
    p = f"{TMP}/dd_cand.py"; open(p, "w").write(txt)
    r = subprocess.run(["python", f"{TMP}/ddrun.py", p], capture_output=True, text=True, timeout=300)
    return sig in (r.stdout + r.stderr)
assert crashes(L)
changed = True
while changed:
    changed = False
    for i in range(len(L) - 1, 9, -1):
        if L[i] is None or not L[i].strip(): continue
        ind = len(L[i]) - len(L[i].lstrip())
        j = i + 1
        while j < len(L) and (L[j] is None or not L[j].strip() or len(L[j]) - len(L[j].lstrip()) > ind): j += 1
        cand = L[:i] + [None] * (j - i) + L[j:]
        if crashes(cand):
            L = cand; changed = True
open("ddmin_out.py", "w").write("\n".join(l for l in L if l is not None))
print("REDUCED to", sum(1 for l in L if l is not None and l.strip()), "lines")
