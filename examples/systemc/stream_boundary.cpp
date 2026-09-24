
//===------------------------------------------------------------*- C++ -*-===//
// Automatically generated file for SystemC (Catapult HLS / MatchLib Connections).
//===----------------------------------------------------------------------===//
#include <systemc.h>
#include <mc_connections.h>   // MatchLib Connections (LI valid/ready channels)
#include <mc_scverify.h>      // SCVerify testbench macros (CCS_MAIN / CCS_DESIGN)
#include <ac_int.h>
#include <ac_fixed.h>
#include <stdint.h>
#include <iostream>
#include <fstream>
#include <algorithm>
// The reused body emits bare max()/min() for the Allo max/min intrinsics (Vitis
// resolves them via hls::); bind them to std:: so the same body compiles here.
using std::max;
using std::min;
// The reused Vivado-emitter body prints Vitis ap_(u)int types; alias them to
// Catapult's ac_int so the same body compiles. (TODO: emit ac_int/ac_fixed
// natively via a type-name override, like getCatapultTypeName in the Catapult
// emitter, and drop this shim.)
// For W<=64, ap_(u)int is a plain ac_int alias. For W>64, ac_int has NO implicit
// conversion to a native int, so the reused body's narrowing `int32_t x = wide;`
// (e.g. a GEMM accumulator widened past 64 bits) fails to compile. Add one via a
// thin subclass ONLY in that range — ac_int's own operators remain exact/derived-
// to-base matches, so arithmetic still resolves to them (no builtin ambiguity).
template <int W, bool Big = (W > 64)> struct ap_sel {
  using s = ac_int<W, true>;
  using u = ac_int<W, false>;
};
template <int W> struct ap_sel<W, true> {
  struct s : ac_int<W, true> {
    using ac_int<W, true>::ac_int;
    operator long long() const { return this->to_int64(); }
  };
  struct u : ac_int<W, false> {
    using ac_int<W, false>::ac_int;
    operator unsigned long long() const { return this->to_uint64(); }
  };
};
template <int W> using ap_int = typename ap_sel<W>::s;
template <int W> using ap_uint = typename ap_sel<W>::u;

// Random-access memory port for a non-sequential boundary array (SystemC/
// Connections flow — internal memory, the only kind SystemC supports; useref
// 14.9.1). Request packed into one ac_int: bit0 = opcode (0=LOAD,1=STORE),
// [ADDRW] addr, [DATAW] wdata. Response = the read value T. One txn/cycle.
// The kernel is the CLIENT (Out<req>/In<rsp>); this module holds the storage.
template <typename T, int SIZE, int ADDRW, int DATAW>
SC_MODULE(AlloMem) {
  sc_in_clk clk;
  sc_in<bool> rst;
  Connections::In< ac_int<1 + ADDRW + DATAW, false> > req;
  Connections::Out<T> rsp;
  T mem[SIZE];
  SC_HAS_PROCESS(AlloMem);
  AlloMem(sc_module_name n) : sc_module(n), req("req"), rsp("rsp") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  void run() {
    req.Reset();
    rsp.Reset();
    wait();
    while (1) {
      ac_int<1 + ADDRW + DATAW, false> r = req.Pop();
      ac_int<ADDRW, false> a = r.template slc<ADDRW>(1);
      if (r[0])
        mem[a] = (T)(int64_t)r.template slc<DATAW>(1 + ADDRW).to_int64();
      else
        rsp.Push(mem[a]);
      wait();
    }
  }
};

// Write-only random-access memory port for a non-sequential OUTPUT array. Same
// packed request as AlloMem but STORE-only, so there is NO response port (a
// store is fire-and-forget; nothing to bind an rsp Out to). The testbench reads
// mem[] out after the run.  ⚠ the int64 cast makes float wdata lossy (defer).
template <typename T, int SIZE, int ADDRW, int DATAW>
SC_MODULE(AlloMemW) {
  sc_in_clk clk;
  sc_in<bool> rst;
  Connections::In< ac_int<1 + ADDRW + DATAW, false> > req;
  T mem[SIZE];
  SC_HAS_PROCESS(AlloMemW);
  AlloMemW(sc_module_name n) : sc_module(n), req("req") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  void run() {
    req.Reset();
    for (int z = 0; z < SIZE; z++) // 0-init so unwritten elements sum-merge as 0
      mem[z] = 0;
    wait();
    while (1) {
      ac_int<1 + ADDRW + DATAW, false> r = req.Pop();
      ac_int<ADDRW, false> a = r.template slc<ADDRW>(1);
      mem[a] = (T)(int64_t)r.template slc<DATAW>(1 + ADDRW).to_int64();
      wait();
    }
  }
};

SC_MODULE(source_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  Connections::In< int32_t > v0;
  Connections::Out< int32_t > v1;
  SC_HAS_PROCESS(source_0);
  source_0(sc_module_name n) : sc_module(n), v0("v0"), v1("v1") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  void run() {
    v0.Reset();
    v1.Reset();
    wait();
    while (1) {
      l_S_i_0_i: for (int i = 0; i < 8; i++) {	// L3
        int32_t v3 = v0.Pop();	// L4
        v1.Push(v3);	// L5
      }
    }
  }
};

SC_MODULE(compute_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  Connections::In< int32_t > v4;
  Connections::Out< int32_t > v5;
  SC_HAS_PROCESS(compute_0);
  compute_0(sc_module_name n) : sc_module(n), v4("v4"), v5("v5") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  void run() {
    v4.Reset();
    v5.Reset();
    wait();
    while (1) {
      l_S_i_0_i1: for (int i1 = 0; i1 < 8; i1++) {	// L11
        int32_t v7 = v4.Pop();	// L12
        ap_int<33> v8 = v7;	// L13
        ap_int<33> v9 = v8 + 1;	// L14
        v5.Push(v9);	// L15
      }
    }
  }
};

SC_MODULE(sink_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  Connections::Out< int32_t > v10;
  Connections::In< int32_t > v11;
  SC_HAS_PROCESS(sink_0);
  sink_0(sc_module_name n) : sc_module(n), v10("v10"), v11("v11") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  void run() {
    v10.Reset();
    v11.Reset();
    wait();
    while (1) {
      l_S_i_0_i2: for (int i2 = 0; i2 < 8; i2++) {	// L20
        int32_t v13 = v11.Pop();	// L21
        v10.Push(v13);	// L22
      }
    }
  }
};

SC_MODULE(top) {
  sc_in_clk clk;
  sc_in<bool> rst;
  Connections::In< int32_t > v14;
  Connections::Out< int32_t > v15;
  Connections::Combinational< int32_t > v16;
  Connections::Combinational< int32_t > v17;
  source_0 u0;
  compute_0 u1;
  sink_0 u2;
  SC_CTOR(top) : v14("v14"), v15("v15"), v16("v16"), v17("v17"), u0("u0"), u1("u1"), u2("u2") {
    u0.clk(clk);
    u0.rst(rst);
    u0.v0(v14);
    u0.v1(v16);
    u1.clk(clk);
    u1.rst(rst);
    u1.v4(v16);
    u1.v5(v17);
    u2.clk(clk);
    u2.rst(rst);
    u2.v10(v15);
    u2.v11(v17);
  }
};

SC_MODULE(tb) {
  sc_clock clk;
  sc_signal<bool> rst;
  top dut;
  Connections::Combinational< int32_t > ch_v14;
  Connections::Combinational< int32_t > ch_v15;
  SC_HAS_PROCESS(tb);
  tb(sc_module_name n) : sc_module(n), clk("clk", 1, SC_NS), dut("dut"), ch_v14("ch_v14"), ch_v15("ch_v15") {
    dut.clk(clk); dut.rst(rst);
    dut.v14(ch_v14);
    dut.v15(ch_v15);
    SC_THREAD(src); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(snk); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
  }
  void src() {
    ch_v14.ResetWrite();
    wait();
    { std::ifstream _f("input0.data"); int32_t _v; for (int f = 0; f < 8; ++f) { _f >> _v; ch_v14.Push(_v); } }
  }
  void snk() {
    ch_v15.ResetRead();
    wait();
    { std::ofstream _f("output0.data"); for (int f = 0; f < 8; ++f) _f << ch_v15.Pop() << "\n"; }
    sc_stop();
  }
};

int sc_main(int, char *[]) {
  tb t("t");
  t.rst = 0; sc_start(1, SC_NS);
  t.rst = 1;
  sc_start();
  return 0;
}
