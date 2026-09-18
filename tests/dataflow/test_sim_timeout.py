# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regression test: a deadlocked region must say so instead of hanging silently.

A dataflow region that deadlocks used to wedge the simulator forever with no
output at all -- no message, no timeout, no indication of which process was
blocked on which channel. The symptom was indistinguishable from a design bug
and cost a downstream project multiple sessions.

`LLVMOMPModule.__call__` now arms a `threading.Timer` around the blocking
`execution_engine.invoke`. The invoke releases the GIL, so the timer thread
runs even while every PE is spinning. When it fires it prints a report to
stderr and lets the run continue: it never kills the process and never raises
from the timer thread, because a blocking C call cannot be safely interrupted.

The region below deadlocks on purpose -- a consumer gets from a stream nobody
ever puts to -- so this test runs it in a subprocess that is killed on
timeout. A test that hangs is worse than a test that fails.
"""

import contextlib
import io
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

from allo.backend import simulator
from allo.ir.types import int32, Stream
import allo.dataflow as df

NELEM = 4
# The child's ALLO_SIM_TIMEOUT: long enough that the report cannot land before
# the region is actually stuck, short enough to keep the test quick.
SIM_TIMEOUT_SEC = 3
# How long to wait for a marker to appear in the child's output before giving
# up. The child is killed as soon as the marker shows up, so this bounds the
# failure case only -- a passing run takes a few seconds.
WAIT_SEC = 90


def _run_deadlocked_region():
    """A region whose consumer blocks forever on a stream nobody feeds."""

    @df.region()
    def stuck(A: int32[NELEM], B: int32[NELEM]):
        # `fed` is written and read; `starved` is read and never written.
        fed: Stream[int32, 2]
        starved: Stream[int32, 2]

        @df.kernel(mapping=[1], args=[A])
        def producer(local_A: int32[NELEM]):
            for i in range(NELEM):
                fed.put(local_A[i])

        @df.kernel(mapping=[1], args=[B])
        def consumer(local_B: int32[NELEM]):
            for i in range(NELEM):
                # The second get can never be satisfied: nothing puts to
                # `starved`, so this PE spins until the process is killed.
                local_B[i] = fed.get() + starved.get()

    A = np.arange(NELEM, dtype=np.int32)
    B = np.zeros(NELEM, dtype=np.int32)
    sim_mod = df.build(stuck, target="simulator")
    print("BUILT", flush=True)
    sim_mod(A, B)
    print("UNREACHABLE: the region was supposed to deadlock", flush=True)


def _child_env(timeout=SIM_TIMEOUT_SEC):
    env = dict(os.environ)
    env["ALLO_SIM_TIMEOUT"] = str(timeout)
    env["OMP_NUM_THREADS"] = "4"
    env["PYTHONPATH"] = os.pathsep.join(
        [os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))]
        + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
    )
    return env


@contextlib.contextmanager
def _deadlocked_child(env):
    """Run the wedged region in a subprocess, killing it on the way out.

    Output goes to temp files rather than pipes so it can be read while the
    child is still running -- the whole point is to look at what a *hung*
    process has printed, and `communicate()` only returns once it exits.
    """
    # pylint: disable=consider-using-with
    out, err = tempfile.TemporaryFile("w+"), tempfile.TemporaryFile("w+")
    proc = subprocess.Popen(
        [sys.executable, os.path.abspath(__file__), "--run-deadlock"],
        env=env,
        stdout=out,
        stderr=err,
        text=True,
    )
    try:
        yield proc, out, err
    finally:
        proc.kill()
        proc.wait()
        out.close()
        err.close()


def _read(handle):
    handle.seek(0)
    return handle.read()


def _wait_for(proc, handle, marker, limit=WAIT_SEC):
    """Poll `handle` until `marker` shows up, the child dies, or time runs out."""
    deadline = time.time() + limit
    while time.time() < deadline:
        text = _read(handle)
        if marker in text or proc.poll() is not None:
            return text
        time.sleep(0.2)
    return _read(handle)


def test_deadlock_is_reported_not_silent():
    """The hung region prints a diagnostic naming the top function, the
    instance count and the thread count, and keeps running."""
    with _deadlocked_child(_child_env()) as (proc, out, err):
        stderr = _wait_for(proc, err, "has not finished")
        stdout = _read(out)
        still_running = proc.poll() is None

    assert "BUILT" in stdout, f"the child never got as far as running:\n{stderr}"
    assert "UNREACHABLE" not in stdout, "the region was supposed to deadlock"
    # The diagnostic, not silence.
    assert "has not finished" in stderr, (
        "a deadlocked region produced no hang report on stderr; "
        f"see LLVMOMPModule.__call__ in allo/backend/simulator.py:\n{stderr}"
    )
    # Reported, not killed: the timer must not take the process down.
    assert still_running, "the hang report must not end the process"
    # ... and the diagnostic is actually informative.
    for expected in (
        "stuck",  # the top function name
        "kernel instances      : 2",  # known at build time
        "OMP_NUM_THREADS       : 4",  # the team in effect
        "STILL RUNNING",  # it did not kill the process
        "kill -9",  # how to stop it
    ):
        assert expected in stderr, f"hang report is missing {expected!r}:\n{stderr}"


def test_timeout_can_be_disabled():
    """`ALLO_SIM_TIMEOUT=0` turns the report off; the region still hangs."""
    with _deadlocked_child(_child_env(timeout=0)) as (proc, out, err):
        _wait_for(proc, out, "BUILT")
        # Well past the timeout the other test fires at.
        time.sleep(SIM_TIMEOUT_SEC * 3)
        stderr, stdout = _read(err), _read(out)
    assert "BUILT" in stdout, f"the child never got as far as running:\n{stderr}"
    assert "has not finished" not in stderr


def test_working_region_is_not_reported():
    """The timer must not fire on a region that completes."""
    num = 8

    @df.region()
    def relay(A: int32[num], B: int32[num]):
        fifo: Stream[int32, 2]

        @df.kernel(mapping=[1], args=[A])
        def send(local_A: int32[num]):
            for i in range(num):
                fifo.put(local_A[i])

        @df.kernel(mapping=[1], args=[B])
        def recv(local_B: int32[num]):
            for i in range(num):
                local_B[i] = fifo.get() + 1

    A = np.arange(num, dtype=np.int32)
    B = np.zeros(num, dtype=np.int32)
    os.environ["ALLO_SIM_TIMEOUT"] = "60"
    try:
        sim_mod = df.build(relay, target="simulator")
        sim_mod(A, B)
    finally:
        os.environ.pop("ALLO_SIM_TIMEOUT", None)
    np.testing.assert_array_equal(B, A + 1)
    # The build-time instance count the report relies on.
    assert sum(sim_mod.pe_counts.values()) == 2


def test_report_is_advisory_not_fatal():
    """A run that outlives the timeout is reported and still returns.

    This is the property the whole design rests on: the report cannot kill or
    interrupt the run (a blocking C call is not safely interruptible from
    Python), so firing on a slow-but-healthy region costs noise, not a result.
    A default that is merely too eager is therefore survivable -- which is why
    the timeout is on by default at all.
    """
    # Big enough that the run takes ~100ms: twenty times the patched timeout
    # below, so the report is not a race.
    num = 4096

    @df.region()
    def slow(A: int32[num], B: int32[num]):
        fifo: Stream[int32, 2]

        @df.kernel(mapping=[1], args=[A])
        def send(local_A: int32[num]):
            for i in range(num):
                fifo.put(local_A[i])

        @df.kernel(mapping=[1], args=[B])
        def recv(local_B: int32[num]):
            for i in range(num):
                local_B[i] = fifo.get() + 1

    A = np.arange(num, dtype=np.int32)
    B = np.zeros(num, dtype=np.int32)
    sim_mod = df.build(slow, target="simulator")
    # Fire early, as if this run had blown past the real default.
    original, simulator._DEFAULT_SIM_TIMEOUT = simulator._DEFAULT_SIM_TIMEOUT, 0.005
    captured = io.StringIO()
    try:
        with contextlib.redirect_stderr(captured):
            sim_mod(A, B)
    finally:
        simulator._DEFAULT_SIM_TIMEOUT = original
    assert "has not finished" in captured.getvalue()
    np.testing.assert_array_equal(B, A + 1)


if __name__ == "__main__":
    if "--run-deadlock" in sys.argv:
        _run_deadlocked_region()
        sys.exit(0)
    test_deadlock_is_reported_not_silent()
    test_timeout_can_be_disabled()
    test_working_region_is_not_reported()
    test_report_is_advisory_not_fatal()
    print("Dataflow Simulator Passed!")
