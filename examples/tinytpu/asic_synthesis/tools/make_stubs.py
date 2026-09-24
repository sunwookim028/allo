"""Emit empty black-box stubs for modules excluded from a logic-only file list.

Takes each excluded module's port list verbatim from its own source file and
writes a module with the same header and no body, so DC links and reports it as
a black box of zero area. Mechanical: the header is copied, never retyped.
"""
import re, sys, pathlib

src_dir, out_dir, names = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3:]
out_dir.mkdir(parents=True, exist_ok=True)
for name in names:
    src = next((p for ext in ('.v', '.sv') for p in [src_dir / (name + ext)] if p.exists()), None)
    if src is None:
        sys.exit(f"no source for {name}")
    text = src.read_text()
    m = re.search(r"(?m)^\s*module\s+" + re.escape(name) + r"\b(.*?);", text, re.S)
    if not m:
        sys.exit(f"no module header for {name} in {src}")
    header = m.group(0)
    header = re.sub(r"//[^\n]*", "", header)          # drop firtool source comments
    (out_dir / (name + "_stub.v")).write_text(header + "\nendmodule\n")
    print(f"{name}: {len(header.splitlines())} header lines")
