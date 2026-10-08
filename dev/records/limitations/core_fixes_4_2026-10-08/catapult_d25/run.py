# Catapult csyn of tests/dataflow/test_stream_flush._region(8): the whole region
# (the flushable stream's AlloFifoClr lives in the top), 3.33 ns.
import os, sys
sys.path.insert(0, os.path.join(os.environ["WT"], "tests/dataflow"))
import test_stream_flush as t
import allo.dataflow as df
prj = sys.argv[1]
mod = df.build(t._region(8), target="systemc", mode="csyn", project=prj, configs={"clock_period": 3.33})
mod()
