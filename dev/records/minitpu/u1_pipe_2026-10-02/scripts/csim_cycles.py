"""Measure, in SystemC csim, the cycle at which each output leaves vs its inputs.

    python csim_cycles.py <unit> <variant> [--n N] [--pipeline LOOP ...] [--prj DIR]

Builds ``units/<unit>.VARIANTS[<variant>]`` for ``target="systemc", mode="csim"``,
then stamps the emitted testbench (a workaround in a generated file, not a fix):
each ``src_<port>`` thread writes the cycle after every ``Push`` returns, the
``snk_<port>`` thread the cycle after every ``Pop`` returns. Element k's csim
latency is ``out[k] - max_i in_i[k]``; the rate is the spacing of outputs.
Values are compared with the RTL too, so a timing hack cannot hide a value bug.
"""
import argparse, importlib, os, re, sys, time
import numpy as np

sys.path.insert(0, os.getcwd())
import allo.dataflow as df  # noqa: E402
from examples.minitpu.harness import rtl  # noqa: E402

STAMP = '(long long)(sc_time_stamp() / sc_time(1, SC_NS))'


def stamp_tb(path):
    s = open(path).read()
    tb = s.index("SC_MODULE(tb)")
    head, t = s[:tb], s[tb:]
    # sources: one stamp file per input port, written after each Push returns
    t, k1 = re.subn(r'\{ std::ifstream _f\("(input\d+)\.data"\);(.*?)(ch_\w+\.Push\([^;]*\);)',
                    lambda m: '{ std::ofstream _st("stamp_%s.txt"); std::ifstream _f("%s.data");%s%s _st << %s << "\\n";'
                    % (m.group(1), m.group(1), m.group(2), m.group(3), STAMP), t, flags=re.S)
    # sinks: braces round the loop body, stamp after the Pop's statement
    t, k2 = re.subn(r'\{ std::ofstream _f\("(output\d+)\.data"\); for \(int f = 0; f < (\d+); \+\+f\) (_f << [^\n]*?\.Pop\(\)[^\n]*?;) \}',
                    lambda m: '{ std::ofstream _st("stamp_%s.txt"); std::ofstream _f("%s.data"); for (int f = 0; f < %s; ++f) { %s _st << %s << "\\n"; } }'
                    % (m.group(1), m.group(1), m.group(2), m.group(3), STAMP), t)
    assert k1 >= 1 and k2 >= 1, (k1, k2)
    open(path, "w").write(head + t)
    return k1, k2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("unit"); ap.add_argument("variant")
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--pipeline", action="append", default=[])
    ap.add_argument("--prj", default="/tmp/csim_cycles")
    a = ap.parse_args()
    u = importlib.import_module(f"examples.minitpu.units.{a.unit}")
    stim = u.stimulus()[: a.n]
    n = len(stim)
    make, runner = u.VARIANTS[a.variant]
    s = df.customize(make(n))
    for loop in a.pipeline:
        s.pipeline(loop)
    prj = os.path.join(a.prj, f"{a.unit}_{a.variant}_n{n}" + ("_pipe" if a.pipeline else ""))
    mod = s.build(target="systemc", mode="csim", project=prj)
    print("stamped src/snk:", stamp_tb(os.path.join(prj, "kernel.cpp")))
    t = time.time()
    got = runner(mod, stim)
    want, _ = rtl.run(u.RTL, stim.astype(np.uint64))
    want = want[:, 0].astype(got.dtype)
    k = int((got != want).sum())
    ins = [np.loadtxt(os.path.join(prj, f), dtype=np.int64) for f in sorted(os.listdir(prj))
           if f.startswith("stamp_input")]
    out = np.loadtxt(os.path.join(prj, "stamp_output0.txt"), dtype=np.int64)
    last_in = np.max(np.stack(ins), axis=0)
    lat = out - last_in
    vals, cnt = np.unique(lat, return_counts=True)
    gaps = np.diff(out)
    print(f"CSIM {a.unit} {a.variant} pipeline={a.pipeline or '-'} n={n}: values {n-k}/{n} vs rtl; "
          f"latency(out-in) {dict(zip(vals.tolist(), cnt.tolist()))}; "
          f"first in {last_in[0]} first out {out[0]} last out {out[-1]}; "
          f"rate {(out[-1]-out[0])/(n-1):.3f} cyc/elem; input spacing {np.diff(last_in).mean():.3f} "
          f"({time.time()-t:.1f}s)")


if __name__ == "__main__":
    main()
