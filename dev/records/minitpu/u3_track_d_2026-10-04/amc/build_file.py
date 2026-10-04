import sys, importlib.util, allo
spec = importlib.util.spec_from_file_location("kk", sys.argv[1]); K = importlib.util.module_from_spec(spec); spec.loader.exec_module(K)
try:
    allo.customize(getattr(K, sys.argv[2])).build(target="amc"); print("BUILT", sys.argv[1].split("/")[-1], flush=True)
except Exception as e:
    print("FAIL", type(e).__name__, str(e)[:160], flush=True)
