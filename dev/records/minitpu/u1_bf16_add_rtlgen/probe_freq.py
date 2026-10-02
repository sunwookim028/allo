import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["BF_N"] = "16"
from bits_kernel import bf16_add_bits
from bits_scalar_kernel import bf16_add_scalar
for name, k in (("array16", bf16_add_bits), ("scalar", bf16_add_scalar)):
    for mhz in (300, 100, 50, 10):
        try:
            rtl = k.schedule().export("rtl", freq_mhz=mhz)
            q = rtl.estimation
            v = rtl.verilog
            hdr = [l.strip() for l in v.splitlines() if l.strip().startswith(("module", "input", "output"))][:12]
            print(f"PROBE {name} {mhz}MHz: latency={q.latency} II={q.interval} fmax={q.fmax:.0f} lut={q.area.lut} ff={q.area.ff}")
            if mhz == 300: print("   ports:", hdr); open(f"{name}_{mhz}.sv","w").write(v)
        except Exception as e:
            print(f"PROBE {name} {mhz}MHz: ERROR {type(e).__name__}: {str(e)[:600]}")
