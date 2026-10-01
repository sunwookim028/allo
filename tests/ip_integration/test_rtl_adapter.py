# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generated adapter validation, project emission, and adversarial RTL tests."""

import json
import os
from pathlib import Path
import shutil
import subprocess

import numpy as np
import pytest
import allo.dataflow as df
from allo import HLSBlackBox, Port, RTLModule, ReadyValidAdapter
from allo.ir.types import int32, Stream

EXAMPLE = Path(__file__).resolve().parents[2] / "examples/ip_integration"
N = 8


def make_ip(**overrides):
    config = dict(
        top="ready_valid_accumulator",
        rtl=EXAMPLE / "ready_valid_accumulator.v",
        ports=[
            Port("A", "a_data", "a_valid", "a_ready", size=4),
            Port("B", "b_data", "b_valid", "b_ready", size=4),
            Port("C", "c_data", "c_valid", "c_ready", dir="out", size=4),
        ],
        clock="clk",
        reset="rst_n",
        reset_active_high=False,
        start=None,
        persistent=True,
        hls=HLSBlackBox(
            str(EXAMPLE / "ready_valid_accumulator.cpp"),
            latency=8,
            adapter=ReadyValidAdapter("ce", start_mode="none"),
        ),
    )
    config.update(overrides)
    return RTLModule(**config)


def region(ip):
    @df.region()
    def top(A: int32[N], B: int32[N], C: int32[N]):
        a: Stream[int32, 2]
        b: Stream[int32, 2]
        c: Stream[int32, 2]

        @df.kernel(mapping=[1], args=[A, B])
        def feed(x: int32[N], y: int32[N]):
            for i in range(N):
                a.put(x[i])
                b.put(y[i])

        @df.kernel(mapping=[1])
        def compute():
            for i in range(N // 4):
                ip(a, b, c)

        @df.kernel(mapping=[1], args=[C])
        def drain(x: int32[N]):
            for i in range(N):
                x[i] = c.get()

    return top


@pytest.fixture
def verilator():
    binary = os.getenv("VERILATOR") or shutil.which("verilator")
    if not binary:
        pytest.skip("Set VERILATOR to run generated RTL tests")
    return binary


def test_deterministic_package(tmp_path, monkeypatch):
    def no_tools(*args, **kwargs):
        pytest.fail("Generating a wrapper/project must not launch a tool")

    monkeypatch.setattr(subprocess, "run", no_tools)
    first, second = make_ip(), make_ip()
    assert first.generate_wrapper() == second.generate_wrapper()
    packages = []
    for index, ip in enumerate((first, second)):
        project = tmp_path / str(index)
        df.build(region(ip), target="vitis_hls", mode="csyn", project=str(project))
        header, manifest_path = ip.export_hls(project)
        manifest = json.loads((project / manifest_path).read_text())
        assert manifest["rtl_top_module_name"] == "ready_valid_accumulator_allo"
        assert manifest["c_parameters"][0]["rtl_ports"]["FIFO_read_enable"] == "p0_read"
        assert manifest["rtl_common_signal"]["module_reset"] == "ap_rst"
        assert (project / header).is_file()
        interfaces = [
            line
            for line in (project / "kernel.cpp").read_text().splitlines()
            if "#pragma HLS interface m_axi" in line
        ]
        assert len(interfaces) == 3
        assert all("depth=8" in line for line in interfaces)
        tcl = (project / "run.tcl").read_text()
        assert tcl.count("config_compile -pipeline_loops 0") == 1
        assert tcl.index("config_compile -pipeline_loops 0") < tcl.index(
            "csynth_design"
        )
        wrapper = [
            p
            for p in manifest["rtl_files"]
            if p.endswith("ready_valid_accumulator_allo.v")
        ]
        assert len(wrapper) == 1
        for source in manifest["rtl_files"]:
            assert "`timescale 1ns / 1ps" in (project / source).read_text()
        assert (project / wrapper[0]).read_text() == ip.generate_wrapper()
        assert (
            "void ready_valid_accumulator_allo"
            in (project / manifest["c_files"][0]["c_file"]).read_text()
        )
        packages.append(manifest)
    assert packages[0] == packages[1]
    other = make_ip(name="other_instance")
    project = tmp_path / "other"
    _, path = other.export_hls(project)
    other_manifest = json.loads((project / path).read_text())
    assert other_manifest["rtl_files"][0] == packages[0]["rtl_files"][0]


@pytest.mark.parametrize("name", [None, "renamed_adapter"])
def test_blackbox_body_visible_during_synthesis(tmp_path, name):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the preprocessing regression")
    ip = make_ip(name=name)
    _, manifest_path = ip.export_hls(tmp_path)
    manifest = json.loads((tmp_path / manifest_path).read_text())
    model = tmp_path / manifest["c_files"][0]["c_file"]
    # Preprocessing alone needs no vendor implementation of hls::stream.
    (tmp_path / "hls_stream.h").write_text("")
    for defines in ([], ["-D__SYNTHESIS__"]):
        result = subprocess.run(
            [compiler, "-E", "-P", *defines, "-I", str(tmp_path), str(model)],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert f"void {ip.top}(" in result.stdout
        assert "static uint32_t total = 0;" in result.stdout
        assert "C.write(static_cast<int32_t>(total));" in result.stdout


@pytest.mark.parametrize("name", [None, "renamed_adapter"])
def test_blackbox_cpp_linkage(tmp_path, name):
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the linkage regression")
    ip = make_ip(name=name)
    header, manifest_path = ip.export_hls(tmp_path)
    manifest = json.loads((tmp_path / manifest_path).read_text())
    model = tmp_path / manifest["c_files"][0]["c_file"]
    # A small stand-in keeps this C++ linkage check independent of Vitis.
    (tmp_path / "hls_stream.h").write_text(
        "#pragma once\nnamespace hls { template<class T> class stream {\n"
        "public: T read() { return T(); } void write(T) {} }; }\n"
    )
    caller = tmp_path / "caller.cpp"
    caller.write_text(
        "#include <stdint.h>\n#include <hls_stream.h>\n"
        f"void {ip.top}(hls::stream<int32_t>&, hls::stream<int32_t>&, "
        "hls::stream<int32_t>&);\n"
        'extern "C" {\n'
        f'#include "{header}"\n'
        "void caller() { hls::stream<int32_t> a, b, c;\n"
        f"{ip.top}(a, b, c); }}\n}}\nint main() {{ caller(); }}\n"
    )
    subprocess.run(
        [
            compiler,
            "-I",
            str(tmp_path),
            str(caller),
            str(model),
            "-o",
            str(tmp_path / "linked"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )


@pytest.mark.parametrize(
    "change, message",
    [
        ({"persistent": False}, "persistent=True"),
        ({"start": "go"}, "start=None"),
        ({"name": "ready_valid_accumulator"}, "must differ"),
        ({"defines": {"FOO": 1}}, "self-contained"),
    ],
)
def test_invalid_module_contract(change, message):
    with pytest.raises((ValueError, NotImplementedError), match=message):
        make_ip(**change)


@pytest.mark.parametrize(
    "adapter, message",
    [
        (ReadyValidAdapter("ce", "level"), "start_mode"),
        (ReadyValidAdapter("ce", "none", "done"), "requires the core done"),
        (ReadyValidAdapter("ce", "none", "guess"), "completion"),
        (ReadyValidAdapter("", "none"), "identifier"),
    ],
)
def test_invalid_adapter_contract(adapter, message):
    with pytest.raises(ValueError, match=message):
        make_ip(
            hls=HLSBlackBox(
                str(EXAMPLE / "ready_valid_accumulator.cpp"), 8, adapter=adapter
            )
        )


def test_missing_count_and_wrong_protocol():
    ip = make_ip()
    ip.ports[0].size = None
    with pytest.raises(ValueError, match="positive"):
        ip.generate_wrapper()
    ip.ports[0].size = 4
    ip.ports[0].protocol = "ap_fifo"
    with pytest.raises(NotImplementedError, match="ready_valid"):
        ip.generate_wrapper()


def test_allo_runs_generated_wrapper(verilator):
    ip = make_ip()
    mod = df.build(region(ip), target="simulator")
    x = np.array([-3, 4, -1, 8, 0, 2, -7, 1], dtype=np.int32)
    y = np.arange(N, dtype=np.int32)
    out = np.zeros(N, dtype=np.int32)
    mod(x, y, out)
    np.testing.assert_array_equal(out, np.cumsum(x + y, dtype=np.int32))
    assert ip._adapted_module is not None
    assert "Vready_valid_accumulator_allo.h" in Path(ip.generated_source).read_text()


def compile_bench(ip, tmp_path, code):
    ip.validate_rtl()
    wrapped = ip._get_adapted_module()
    wrapped._prepare_simulation()
    bench = tmp_path / "bench.cpp"
    bench.write_text(code)
    binary = tmp_path / "bench"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-pthread",
            *[f"-I{p}" for p in wrapped.include_paths],
            str(bench),
            *wrapped._link_inputs,
            "-o",
            str(binary),
        ],
        check=True,
        timeout=120,
    )
    subprocess.run([str(binary)], check=True, timeout=30)


def test_randomized_stalls_counts_reset(verilator, tmp_path):
    compile_bench(
        make_ip(parameters={"W": 32}),
        tmp_path,
        r"""
#include "Vready_valid_accumulator_allo.h"
#include <cassert>
#include <cstdint>
#include <random>
int main() {
  VerilatedContext context; context.threads(1);
  Vready_valid_accumulator_allo m(&context);
  auto tick=[&]() { m.ap_clk=0; m.eval(); m.ap_clk=1; m.eval(); m.ap_clk=0; m.eval(); };
  std::mt19937 rng(314159);
  auto reset=[&]() {
    m.ap_ce=0; m.ap_rst=1; m.ap_start=0; m.ap_continue=0;
    m.p0_empty_n=0; m.p1_empty_n=0; m.p2_full_n=0;
    tick(); m.ap_rst=0; m.ap_ce=1; tick(); assert(m.ap_idle);
  };
  reset();
  uint32_t total=0;
  for(int transaction=0; transaction<100; ++transaction) {
    m.ap_start=1; m.ap_ce=1; m.eval(); assert(m.ap_ready); tick(); m.ap_start=0;
    int a=0, b=0, c=0, cycles=0;
    const bool abort_transaction=transaction%9==8;
    while(!m.ap_done) {
      assert(++cycles<1000);
      const bool random=transaction!=0;
      m.ap_ce=!random || rng()%5!=0;
      m.p0_empty_n=!random || rng()%3!=0;
      m.p1_empty_n=!random || rng()%3!=0;
      m.p2_full_n=!random || rng()%2!=0;
      m.p0_data=static_cast<uint32_t>(a-3);
      m.p1_data=static_cast<uint32_t>(b+7);
      m.eval();
      // Samples are taken before the active edge, even if source data changes
      // immediately after a FIFO read on that edge.
      bool ar=m.p0_read, br=m.p1_read, cw=m.p2_write;
      uint32_t value=m.p2_data;
      if(!m.ap_ce) assert(!ar && !br && !cw && !m.ap_ready);
      tick();
      if(ar) { ++a; assert(a<=4); }
      if(br) { ++b; assert(b<=4); }
      if(cw) {
        assert(c<4); total+=static_cast<uint32_t>(2*c+4);
        assert(value==total); ++c;
      }
      if(abort_transaction && a>=2 && b>=2) { reset(); total=0; break; }
    }
    if(abort_transaction) continue;
    assert(a==4 && b==4 && c==4);
    if(transaction==0) assert(cycles==8); // Declared unstalled wrapper latency.
    // Completion persists; start alone cannot retrigger or consume extra tokens.
    m.ap_start=1; m.p0_empty_n=1; m.p1_empty_n=1; m.p2_full_n=1;
    for(int i=0; i<7; ++i) {
      m.ap_ce=i%2; tick(); assert(m.ap_done && !m.ap_ready);
      assert(!m.p0_read && !m.p1_read && !m.p2_write);
    }
    m.ap_start=0; m.ap_continue=1; m.ap_ce=0; tick(); assert(m.ap_done);
    m.ap_ce=1; tick(); m.ap_continue=0; assert(m.ap_idle && !m.ap_done);
  }
}
""",
    )


def test_pulse_start_and_delayed_done(verilator, tmp_path):
    sources = Path(__file__).parent / "rtl"
    ip = RTLModule(
        "pulse_core",
        sources / "pulse_core.v",
        clock="clk",
        reset="rst",
        start="go",
        done="done",
        persistent=True,
        ports=[
            Port("A", "a_data", "a_valid", "a_ready", size=1),
            Port("C", "c_data", "c_valid", "c_ready", size=1, dir="out"),
        ],
        hls=HLSBlackBox(
            str(sources / "pulse_core.cpp"),
            latency=9,
            adapter=ReadyValidAdapter("ce", completion="done"),
        ),
    )
    compile_bench(
        ip,
        tmp_path,
        r"""
#include "Vpulse_core_allo.h"
#include <cassert>
int main() {
  VerilatedContext context; context.threads(1);
  Vpulse_core_allo m(&context);
  auto tick=[&]() { m.ap_clk=0; m.eval(); m.ap_clk=1; m.eval(); m.ap_clk=0; m.eval(); };
  m.ap_ce=1; m.ap_rst=1; m.ap_start=0; m.ap_continue=0;
  m.p0_empty_n=0; m.p1_full_n=0; tick(); m.ap_rst=0; tick();
  for(int tx=0; tx<5; ++tx) {
    m.ap_start=1; tick(); m.ap_start=0;
    // Stall LAUNCH: no start may be consumed until the enabled edge.
    m.ap_ce=0; tick(); tick(); m.ap_ce=1;
    m.p0_empty_n=1; m.p0_data=tx;
    int reads=0, writes=0, cycles=0, output_cycle=-1;
    while(!m.ap_done) {
      assert(++cycles<100);
      m.p1_full_n=cycles>8;
      m.eval(); bool rd=m.p0_read, wr=m.p1_write; auto value=m.p1_data;
      tick(); if(rd) ++reads;
      if(wr) { ++writes; assert(value==static_cast<unsigned>(tx+1)); output_cycle=cycles; }
      assert(reads<=1 && writes<=1);
    }
    assert(reads==1 && writes==1 && cycles-output_cycle>=4);
    for(int i=0; i<5; ++i) { tick(); assert(m.ap_done); }
    m.ap_continue=1; tick(); m.ap_continue=0; assert(m.ap_idle);
  }
}
""",
    )
