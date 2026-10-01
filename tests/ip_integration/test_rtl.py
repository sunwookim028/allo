# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""RTLModule contracts, project emission, and real Verilator/Allo integration."""

import json
import os
from pathlib import Path
import shutil

import numpy as np
import pytest
import allo
import allo.dataflow as df
from allo import RTLModule, Port, MemPort, HLSBlackBox
from allo.ir.types import int32, uint32, Stream

RTL = Path(__file__).parent / "rtl"
EXAMPLE = Path(__file__).resolve().parents[2] / "examples/ip_integration"
N = 8


def accumulator(name=None):
    return RTLModule(
        "accumulator",
        EXAMPLE / "rtl_accumulator.v",
        name=name,
        ports=[
            Port("A", "a_dout", "a_empty_n", "a_read", size=1, protocol="ap_fifo"),
            Port(
                "C",
                "c_din",
                "c_write",
                "c_full_n",
                dir="out",
                size=1,
                protocol="ap_fifo",
            ),
        ],
        done="ap_done",
        persistent=True,
        hls=HLSBlackBox(str(EXAMPLE / "rtl_accumulator.cpp"), latency=3),
    )


def region(ip):
    @df.region()
    def top(A: int32[N], C: int32[N]):
        a: Stream[int32, 2]
        c: Stream[int32, 2]

        @df.kernel(mapping=[1], args=[A])
        def feed(x: int32[N]):
            for i in range(N):
                a.put(x[i])

        @df.kernel(mapping=[1])
        def compute():
            for i in range(N):
                ip(a, c)

        @df.kernel(mapping=[1], args=[C])
        def drain(x: int32[N]):
            for i in range(N):
                x[i] = c.get()

    return top


@pytest.fixture
def verilator():
    tool = os.environ.get("VERILATOR") or shutil.which("verilator")
    if not tool:
        pytest.skip("Verilator is not installed; set VERILATOR")
    return tool


def test_lazy_project_and_relocation(tmp_path, monkeypatch):
    ip = accumulator()
    monkeypatch.setattr(
        ip, "_tool", lambda: pytest.fail("Project emission must not invoke Verilator")
    )
    project = tmp_path / "project"
    df.build(region(ip), target="vitis_hls", mode="csyn", project=str(project))
    assert not Path(ip.impl).exists()
    manifest = json.loads((project / "rtl_accumulator/blackbox.json").read_text())
    assert manifest["c_parameters"][0]["rtl_ports"]["FIFO_read_enable"] == "a_read"
    assert manifest["rtl_common_signal"]["module_clock_enable"] == "ap_ce"
    assert manifest["rtl_performance"] == {"latency": "3", "II": "0"}
    assert (
        "add_files -blackbox {rtl_accumulator/blackbox.json}"
        in (project / "run.tcl").read_text()
    )
    code = (project / "kernel.cpp").read_text()
    tcl = (project / "run.tcl").read_text()
    assert tcl.count("config_compile -pipeline_loops 0") == 1
    assert tcl.index("config_compile -pipeline_loops 0") < tcl.index("csynth_design")
    assert '#include "rtl_accumulator/accumulator.h"' in code
    assert "transactor" not in code and "accumulator(" in code
    relocated = tmp_path / "relocated"
    shutil.copytree(project, relocated)
    for path in manifest["rtl_files"] + [manifest["c_files"][0]["c_file"]]:
        assert (relocated / path).is_file()
        assert not Path(path).is_absolute()


def test_alias_package(tmp_path):
    ip = accumulator("second")
    _, manifest = ip.export_hls(tmp_path)
    data = json.loads((tmp_path / manifest).read_text())
    assert data["rtl_top_module_name"] == data["c_function_name"] == "second"
    assert "module second" in (tmp_path / "rtl_second/second.v").read_text()
    assert (
        (tmp_path / "rtl_second/second.v")
        .read_text()
        .startswith("`timescale 1ns / 1ps\n")
    )
    assert "void second" in (tmp_path / "rtl_second/model.cpp").read_text()


def test_invalid_contracts(tmp_path):
    with pytest.raises(ValueError, match="positive"):
        RTLModule(
            "memory",
            RTL / "memory.v",
            ports=[MemPort("X", 0, "int32_t", "addr", "ce", q="q")],
            done="ap_done",
        )
    with pytest.raises(ValueError, match="payload"):
        RTLModule(
            "x",
            EXAMPLE / "rtl_accumulator.v",
            ports=[Port("x", "d", "v", "r", ctype="int64_t")],
            done="ap_done",
            persistent=True,
        )
    ip = accumulator()
    with pytest.raises(NotImplementedError, match="vitis_hls"):
        ip.validate_hls("catapult", "csyn")
    with pytest.raises(NotImplementedError, match="does not support"):
        df.build(region(ip), target="xls", project=str(tmp_path / "xls"))
    ip.ports[0].protocol = "ready_valid"
    with pytest.raises(NotImplementedError, match="ap_fifo"):
        ip.export_hls(tmp_path)


def test_signedness_mismatch(capsys):
    ip = accumulator()

    @df.region()
    def top():
        a: Stream[uint32, 2]
        c: Stream[int32, 2]

        @df.kernel(mapping=[1])
        def compute():
            ip(a, c)

    with pytest.raises(SystemExit):
        df.customize(top)
    assert "requires int32_t" in capsys.readouterr().out


def test_rtl_pin_validation(verilator):
    ip = accumulator()
    ip.validate_rtl()
    ip.ports[0].data = "missing"
    with pytest.raises(ValueError, match="RTL pin missing"):
        ip.validate_rtl()


def test_stream_backpressure_and_repeated_calls(verilator):
    ip = accumulator()
    mod = df.build(region(ip), target="simulator")
    x = np.arange(-3, N - 3, dtype=np.int32)
    y = np.zeros(N, dtype=np.int32)
    mod(x, y)
    np.testing.assert_array_equal(y, np.cumsum(x, dtype=np.int32))
    # A second independently built model must start from reset.
    other = df.build(region(accumulator("other")), target="simulator")
    y2 = np.zeros_like(y)
    other(x, y2)
    np.testing.assert_array_equal(y2, y)


def test_array_binding(verilator):
    ip = RTLModule(
        "vadd_rtl",
        RTL / "vadd_rtl.v",
        n=8,
        inputs=[
            Port("A", "a_tdata", "a_tvalid", "a_tready"),
            Port("B", "b_tdata", "b_tvalid", "b_tready"),
        ],
        outputs=[Port("C", "c_tdata", "c_tvalid", "c_tready")],
    )

    def top(A: int32[8], B: int32[8]) -> int32[8]:
        C: int32[8] = 0
        ip(A, B, C)
        return C

    mod = allo.customize(top).build(target="llvm")
    x = np.arange(-4, 4, dtype=np.int32)
    np.testing.assert_array_equal(mod(x, x), x + x)


def test_memory_binding(verilator):
    ip = RTLModule(
        "memory",
        RTL / "memory.v",
        ports=[MemPort("X", 4, "int32_t", "addr", "ce", q="q", we="we", d="d")],
        done="ap_done",
    )

    def top(X: int32[4]):
        for _ in range(2):
            ip(X)

    mod = allo.customize(top).build(target="llvm")
    x = np.array([9, 2, 3, 4], dtype=np.int32)
    mod(x)
    np.testing.assert_array_equal(x, [11, 2, 3, 4])


def test_wrapper_control_and_reset(verilator, tmp_path):
    """Exercise CE, output stalls, done retention, continuation, and reset in RTL."""
    ip = accumulator()
    ip._prepare_simulation()
    bench = tmp_path / "control.cpp"
    bench.write_text(
        r"""
#include "Vaccumulator.h"
#include <cassert>
int main() {
  VerilatedContext context;
  context.threads(1);
  Vaccumulator m(&context);
  auto tick = [&]() { m.ap_clk=0; m.eval(); m.ap_clk=1; m.eval(); m.ap_clk=0; m.eval(); };
  m.ap_ce=1; m.ap_rst=1; m.ap_start=0; m.ap_continue=0;
  m.a_empty_n=0; m.c_full_n=0;
  tick(); m.ap_rst=0; tick(); assert(m.ap_idle);
  m.ap_ce=0; m.ap_start=1; tick(); assert(m.ap_idle && !m.ap_ready);
  m.ap_ce=1; tick(); m.ap_start=0;
  m.a_dout=7; m.a_empty_n=1; tick(); m.a_empty_n=0;
  for(int i=0; i<8; ++i) { tick(); assert(!m.ap_done && !m.c_write && m.c_din==7); }
  m.c_full_n=1; tick(); assert(m.ap_done);
  for(int i=0; i<4; ++i) { tick(); assert(m.ap_done); }
  m.ap_continue=1; tick(); m.ap_continue=0; assert(m.ap_idle);
  m.ap_start=1; tick(); m.ap_start=0;
  m.a_dout=3; m.a_empty_n=1; tick(); m.a_empty_n=0;
  assert(m.c_din==10); tick(); assert(m.ap_done);
  m.ap_rst=1; tick(); m.ap_rst=0; tick(); assert(m.ap_idle && m.c_din==0);
}
"""
    )
    executable = tmp_path / "control"
    import subprocess

    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-pthread",
            *[f"-I{p}" for p in ip.include_paths],
            str(bench),
            *ip._link_inputs,
            "-o",
            str(executable),
        ],
        check=True,
        timeout=120,
    )
    subprocess.run([str(executable)], check=True, timeout=20)


def test_multiple_static_calls_rejected(capsys):
    ip = accumulator()

    @df.region()
    def top():
        a: Stream[int32, 2]
        c: Stream[int32, 2]

        @df.kernel(mapping=[1])
        def compute():
            ip(a, c)
            ip(a, c)

    with pytest.raises(SystemExit):
        df.customize(top)
    assert "one static call site" in capsys.readouterr().out


def test_array_synthesis_rejected(tmp_path):
    ip = RTLModule(
        "vadd_rtl",
        RTL / "vadd_rtl.v",
        n=8,
        inputs=[
            Port("A", "a_tdata", "a_tvalid", "a_tready"),
            Port("B", "b_tdata", "b_tvalid", "b_tready"),
        ],
        outputs=[Port("C", "c_tdata", "c_tvalid", "c_tready")],
    )
    with pytest.raises(ValueError, match="HLSBlackBox"):
        ip.export_hls(tmp_path)


def test_bad_c_model_rejected(tmp_path):
    model = tmp_path / "bad.cpp"
    model.write_text("void accumulator(int A, int C) {}")
    ip = accumulator()
    ip.c_model = model
    with pytest.raises(ValueError, match="signature"):
        ip.export_hls(tmp_path / "project")


def test_two_instances_in_one_graph(verilator, tmp_path):
    first, second = accumulator("first"), accumulator("second")

    @df.region()
    def top(A: int32[N], C: int32[2 * N]):
        a: Stream[int32, 2]
        b: Stream[int32, 2]
        c: Stream[int32, 2]
        d: Stream[int32, 2]

        @df.kernel(mapping=[1], args=[A])
        def feed(x: int32[N]):
            for i in range(N):
                a.put(x[i])
                b.put(x[i])

        @df.kernel(mapping=[1])
        def compute_first():
            for i in range(N):
                first(a, c)

        @df.kernel(mapping=[1])
        def compute_second():
            for i in range(N):
                second(b, d)

        @df.kernel(mapping=[1], args=[C])
        def drain(x: int32[2 * N]):
            for i in range(N):
                x[i] = c.get()
                x[N + i] = d.get()

    mod = df.build(top, target="simulator")
    x = np.arange(N, dtype=np.int32)
    y = np.zeros(2 * N, dtype=np.int32)
    mod(x, y)
    np.testing.assert_array_equal(y, np.tile(np.cumsum(x, dtype=np.int32), 2))
    df.build(top, target="vitis_hls", mode="csyn", project=str(tmp_path))
    manifests = [
        json.loads((tmp_path / f"rtl_{name}/blackbox.json").read_text())
        for name in ("first", "second")
    ]
    # Both aliases reference the same staged definition of accumulator.
    assert manifests[0]["rtl_files"][0] == manifests[1]["rtl_files"][0]
    assert manifests[0]["rtl_top_module_name"] != manifests[1]["rtl_top_module_name"]
