"""Small 1D array of independent signed 32-bit adders."""

from pathlib import Path

import allo
import allo.dataflow as df
import numpy as np
from allo.ir.types import Stream, int32


# Start small for PD experiments. Set N = 32 for Vincent's vector length.
# N is both the vector length and the number of parallel adder kernels.
N = 4
FIFO_DEPTH = 2


@df.region()
def top(A: int32[N], B: int32[N], C: int32[N]):
    a_s: Stream[int32, FIFO_DEPTH][N]
    b_s: Stream[int32, FIFO_DEPTH][N]
    out_s: Stream[int32, FIFO_DEPTH][N]

    @df.kernel(mapping=[1], args=[A, B])
    def load(local_A: int32[N], local_B: int32[N]):
        with allo.meta_for(N) as i:
            a_s[i].put(local_A[i])
            b_s[i].put(local_B[i])

    # One copy of Vincent's a + b datapath per vector element.
    @df.kernel(mapping=[N])
    def adder():
        i = df.get_pid()
        a: int32 = a_s[i].get()
        b: int32 = b_s[i].get()
        out_s[i].put(a + b)

    @df.kernel(mapping=[1], args=[C])
    def store(local_C: int32[N]):
        with allo.meta_for(N) as i:
            local_C[i] = out_s[i].get()


def build(project, target, mode, configs):
    """Entry point expected by the allo-asic-compilation flow node."""
    module = df.build(
        top,
        target=target,
        mode=mode,
        project=Path(project).resolve(),
        wrap_io=False,
        configs=configs,
    )
    module()


def workload():
    """A deterministic input pair, including positive and negative values."""
    a = np.arange(N, dtype=np.int32) - N // 2
    b = 3 - 2 * np.arange(N, dtype=np.int32)
    return {
        "inputs": {"A": a, "B": b},
        "expected_outputs": {"C": np.add(a, b, dtype=np.int32)},
    }


def testbench_workload():
    """Supply the existing flow's generated RTL testbench with one call."""
    transaction = workload()
    return {
        "schema_version": 1,
        "top_function": "top",
        "call_signature": ["A", "B", "C"],
        "calls": [
            {
                "name": "vadd_1d",
                "reset_before": True,
                "arguments": {
                    "A": transaction["inputs"]["A"].copy(),
                    "B": transaction["inputs"]["B"].copy(),
                    "C": np.full(N, -7, dtype=np.int32),
                },
                "expected": {"C": transaction["expected_outputs"]["C"].copy()},
                "comparison": {"C": {"mode": "bit_exact"}},
            }
        ],
        "default_timeout_cycles": 10_000,
    }


def run_workload():
    """Check the actual dataflow design against NumPy without HLS tools."""
    simulator = df.build(top, target="simulator")
    transaction = workload()
    rng = np.random.default_rng(0)
    cases = [
        (transaction["inputs"]["A"], transaction["inputs"]["B"]),
        (np.zeros(N, dtype=np.int32), np.zeros(N, dtype=np.int32)),
        (
            rng.integers(-1000, 1000, N, dtype=np.int32),
            rng.integers(-1000, 1000, N, dtype=np.int32),
        ),
    ]
    for a, b in cases:
        c = np.full(N, -7, dtype=np.int32)
        simulator(a.copy(), b.copy(), c)
        np.testing.assert_array_equal(c, np.add(a, b, dtype=np.int32))
    print(f"PASS: {N} parallel int32 adders; {len(cases)} simulation cases")


if __name__ == "__main__":
    run_workload()
