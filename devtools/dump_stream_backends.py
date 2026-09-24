# Dump the stream producer/consumer example through every backend.
# Each backend emitter runs in its OWN subprocess so a crash in one
# (e.g. Intel HLS segfaults on this module) can't kill the others.
#
# Usage:  OMP_NUM_THREADS=4 python dump_stream_backends.py [backend]
#   backend in: frontend vhls catapult systemc tapa ihls xls llvm ll
#   (no arg = run all, each in a fresh subprocess)
import io
import os
import subprocess
import sys

# Import the allo checkout we are EDITING (/home/zsm9/allo_sup), not whatever
# editable package happens to be installed (/home/zsm9/allo). Without this,
# running from examples/ picks up the installed package, which is an older
# checkout missing newer bindings like emit_systemc. Prepend repo root (for
# `import allo`) and examples/ (for `import stream_producer_consumer` -- this
# script lives in devtools/, the design it dumps lives in examples/).
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_HERE)
sys.path.insert(0, os.path.join(_REPO_ROOT, "examples"))
sys.path.insert(0, _REPO_ROOT)

BACKENDS = ["frontend", "vhls", "catapult", "systemc", "tapa", "ihls", "xls", "llvm", "ll"]


def run_one(which):
    import allo.dataflow as df
    from allo._mlir.dialects import allo as allo_d
    from stream_producer_consumer import top

    if which == "frontend":
        print(df.customize(top).module)
        return
    if which in ("llvm", "ll"):
        mod = df.build(top, target="simulator")
        txt = str(mod.module)
        if which == "llvm":
            print(txt)
            return
        path = "/tmp/claude-1838657/stream_llvm.mlir"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        open(path, "w").write(txt)
        tr = os.path.join(os.getenv("LLVM_BUILD_DIR"), "bin", "mlir-translate")
        out = subprocess.run([tr, "--mlir-to-llvmir", path], capture_output=True, text=True)
        print(out.stdout if out.returncode == 0 else out.stderr)
        return
    # HLS-family text emitters. Resolve lazily via getattr so one missing
    # binding (e.g. an older allo build without emit_systemc) reports itself
    # instead of breaking every other backend at dict-construction time.
    fn_names = {
        "vhls": "emit_vhls",
        "catapult": "emit_catapult",
        "systemc": "emit_systemc",
        "tapa": "emit_thls",
        "ihls": "emit_ihls",
        "xls": "emit_xhls",
    }
    fn = getattr(allo_d, fn_names[which], None)
    if fn is None:
        print(f"<{which}: allo build has no {fn_names[which]} "
              f"(imported from {os.path.dirname(allo_d.__file__)})>")
        return
    s = df.customize(top)
    buf = io.StringIO()
    ok = fn(s.module, buf)
    buf.seek(0)
    print(buf.read() if ok else "<emit returned False>")


LABELS = {
    "frontend": "FRONTEND MLIR (!allo.stream + allo.stream_*)",
    "vhls": "VITIS / VIVADO HLS  (hls::stream, .write()/.read())",
    "catapult": "CATAPULT HLS  (ac_channel)",
    "systemc": "SYSTEMC  (sc_fifo / Connections)",
    "tapa": "TAPA HLS  (tapa::stream)",
    "ihls": "INTEL HLS  (ihls::stream)",
    "xls": "XLS  (__xls_channel)",
    "llvm": "LLVM CPU SIMULATOR - lowered LLVM-dialect MLIR (streams -> memref FIFO + OMP)",
    "ll": "LLVM CPU SIMULATOR - textual LLVM IR (.ll)",
}

if __name__ == "__main__":
    if len(sys.argv) > 1:               # child: emit one backend
        run_one(sys.argv[1])
    else:                               # parent: fan out one subprocess each
        for b in BACKENDS:
            print("\n" + "=" * 70 + f"\n{LABELS[b]}\n" + "=" * 70)
            r = subprocess.run([sys.executable, __file__, b], capture_output=True, text=True)
            sys.stdout.write(r.stdout)
            if r.returncode != 0:
                print(f"<{b} backend crashed: exit {r.returncode}>")
                tail = "\n".join(r.stderr.strip().splitlines()[-3:])
                if tail:
                    print(tail)
