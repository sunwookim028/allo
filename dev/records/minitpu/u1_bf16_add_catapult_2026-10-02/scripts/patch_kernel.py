"""Hand-patches applied to the emitted kernel.cpp (workarounds, not fixes).
P1: include ac_std_float.h before mc_connections.h, so Connections' marshaller.h
    sees __AC_STD_FLOAT_H and defines Wrapped<ac::bfloat16> (CRD-135 otherwise).
P2 (only if the front end hits it): sc_trace overloads into namespace ac."""
import sys, re
p = sys.argv[1]; s = open(p).read()
inc = "#include <ac_std_float.h>   // IEEE floats: ac_ieee_float<binaryNN>\n"
assert inc in s
s = s.replace(inc, "")
s = s.replace("#include <mc_connections.h>", "#include <ac_std_float.h>   // [hand-patch P1] moved before mc_connections.h\n#include <mc_connections.h>", 1)
if "--trace" in sys.argv:
    a = s.index("inline void sc_trace(sc_core::sc_trace_file *tf, const half &h")
    b = s.index("// Make ac_ieee_float<Format> a valid")
    blk = s[a:b].replace("  sc_trace(tf,", "  sc_core::sc_trace(tf,")
    s = s[:a] + "namespace ac { // [hand-patch P2]\n" + blk + "} // namespace ac\n" + s[b:]
open(p, "w").write(s)
# P3 (--wire-wait): the steady-state while(1) has its wait() only under !__SYNTHESIS__;
#    a Wire-only body has no Pop/Push to supply one, so Catapult refuses (CIN-123).
if "--wire-wait" in sys.argv:
    s = open(p).read()
    a = s.index("SC_MODULE(add_0)"); b = s.index("\n};", a)
    old = "#ifndef __SYNTHESIS__\n      wait();\n#endif\n"
    assert old in s[a:b]
    s = s[:a] + s[a:b].replace(old, "      wait();  // [hand-patch P3] also under __SYNTHESIS__\n", 1) + s[b:]
    open(p, "w").write(s)
