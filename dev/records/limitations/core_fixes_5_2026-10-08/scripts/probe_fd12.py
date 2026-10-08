"""Probe fd12a's Catapult RTL: per cycle, the IRAM server's write pins, read address and q."""
import os, sys, re
sys.path.insert(0, os.getcwd())
R = "dev/records/minitpu/u4_track_d_2026-10-08/scripts"
sys.path.insert(0, R)
import u4d_check as U
OUT = os.environ["PROBE_OUT"]
SIG = {"wa": "v282", "we": "v284", "ra": "v285", "q": "v286", "rst": "v267", "wd": "v283"}
orig_driver = U.CatapultRtl.driver
def driver(self, nin, nout):
    code = orig_driver(self, nin, nout)
    t = self.top
    def s(n, wide=False):
        x = f"dut.rootp->{t}__DOT__{SIG[n]}"
        return f"(unsigned long long)({x}[0])" if wide else f"(unsigned long long)({x})"
    pr = (f'    std::fprintf(pf, "%llu %llu %llu %llu %llu %llx %llx\\n", (unsigned long long)cyc, '
          f'(unsigned long long)k[0], {s("rst")}, {s("we")}, {s("wa")}, {s("ra")}, {s("wd", True)}, {s("q", True)});\n')
    pr = (f'    std::fprintf(pf, "%llu %llu %llu %llu %llu %llu %llx %llx\\n", (unsigned long long)cyc, '
          f'(unsigned long long)k[0], {s("rst")}, {s("we")}, {s("wa")}, {s("ra")}, {s("wd", True)}, {s("q", True)});\n')
    code = code.replace('#include "verilated.h"', f'#include "verilated.h"\n#include "V{t}___024root.h"')
    code = code.replace("  uint64_t cyc = 0, last = 0;", f'  uint64_t cyc = 0, last = 0;\n  FILE* pf = std::fopen("{OUT}", "w");')
    # sample after the clk=0 eval (combinational values of this cycle)
    marker = "    dut.clk = 1;\n    dut.eval();"
    pk = '    std::fprintf(pf, "K %llu", (unsigned long long)cyc); for (int i = 0; i < 8; ++i) std::fprintf(pf, " %lld", (long long)k[i] - (long long)k[0]); std::fprintf(pf, "\\n");\n'
    code = code.replace(marker, pr + pk + marker, 1)
    return code
U.CatapultRtl.driver = driver
orig_build = U.CatapultRtl.build
import subprocess
_run = subprocess.run
def run(cmd, *a, **k):
    if cmd and cmd[0] == "verilator":
        cmd = cmd[:1] + ["--public-flat-rw"] + cmd[1:]
    return _run(cmd, *a, **k)
U.subprocess.run = run
os.environ["MINITPU_HARNESS_CACHE"] = os.environ.get("PROBE_CACHE", "/work/shared/users/phd/sk3463/scratch/fix5/hcache")
sys.argv = ["u4d_check.py", "fetch", "form:fetch_d12_packed", sys.argv[1], "--inst", "iram", "--shift", "4"]
U.main()
