import sys, os
sys.path.insert(0, os.path.join(os.environ["WT"], "tests/dataflow"))
import test_systemc_unreset as t
import allo.dataflow as df
which = sys.argv[1]
make = {"stream": t._regfile_stream, "plain": lambda n: t._plain_regfile(n, reset=False),
        "stream_reset": None}[which]
prj = f"/work/shared/users/phd/sk3463/scratch/fix4/cat_c2/{which}"
mod = df.build(make(16), target="systemc", mode="csyn", project=prj,
               configs={"synth_top": "rf_0", "clock_period": 3.33})
mod()
