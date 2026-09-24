
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

SC_MODULE(feed_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  Connections::In< int32_t > v0;
  Connections::Out< int32_t > v1;
  SC_HAS_PROCESS(feed_0);
  feed_0(sc_module_name n) : sc_module(n), v0("v0"), v1("v1") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  void run() {
    v0.Reset();
    v1.Reset();
    wait();
    while (1) {
      l_S_k_0_k: for (int k = 0; k < 8; k++) {	// L3
        int32_t v3 = v0.Pop();	// L4
        v1.Push(v3);	// L5
      }
    }
  }
};

SC_MODULE(pe_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  Connections::In< int32_t > v4;
  Connections::Out< int32_t > v5;
  SC_HAS_PROCESS(pe_0);
  pe_0(sc_module_name n) : sc_module(n), v4("v4"), v5("v5") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  void run() {
    v4.Reset();
    v5.Reset();
    wait();
    while (1) {
      l_S_k_0_k1: for (int k1 = 0; k1 < 8; k1++) {	// L11
        int32_t v7 = v4.Pop();	// L12
        int32_t v;	// L13
        v = v7;	// L14
        int32_t v9 = v;	// L15
        ap_int<33> v10 = v9;	// L16
        ap_int<33> v11 = v10 + 1;	// L17
        int32_t v12 = v11;	// L18
        int32_t w;	// L19
        w = v12;	// L20
        int32_t v14 = w;	// L21
        v5.Push(v14);	// L22
      }
    }
  }
};

SC_MODULE(pe_1) {
  sc_in_clk clk;
  sc_in<bool> rst;
  Connections::In< int32_t > v15;
  Connections::Out< int32_t > v16;
  SC_HAS_PROCESS(pe_1);
  pe_1(sc_module_name n) : sc_module(n), v15("v15"), v16("v16") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  void run() {
    v15.Reset();
    v16.Reset();
    wait();
    while (1) {
      l_S_k_0_k2: for (int k2 = 0; k2 < 8; k2++) {	// L28
        int32_t v18 = v15.Pop();	// L29
        int32_t v1;	// L30
        v1 = v18;	// L31
        int32_t v20 = v1;	// L32
        ap_int<33> v21 = v20;	// L33
        ap_int<33> v22 = v21 + 1;	// L34
        int32_t v23 = v22;	// L35
        int32_t w1;	// L36
        w1 = v23;	// L37
        int32_t v25 = w1;	// L38
        v16.Push(v25);	// L39
      }
    }
  }
};

SC_MODULE(pe_2) {
  sc_in_clk clk;
  sc_in<bool> rst;
  Connections::In< int32_t > v26;
  Connections::Out< int32_t > v27;
  SC_HAS_PROCESS(pe_2);
  pe_2(sc_module_name n) : sc_module(n), v26("v26"), v27("v27") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  void run() {
    v26.Reset();
    v27.Reset();
    wait();
    while (1) {
      l_S_k_0_k3: for (int k3 = 0; k3 < 8; k3++) {	// L45
        int32_t v29 = v26.Pop();	// L46
        int32_t v2;	// L47
        v2 = v29;	// L48
        int32_t v31 = v2;	// L49
        ap_int<33> v32 = v31;	// L50
        ap_int<33> v33 = v32 + 1;	// L51
        int32_t v34 = v33;	// L52
        int32_t w2;	// L53
        w2 = v34;	// L54
        int32_t v36 = w2;	// L55
        v27.Push(v36);	// L56
      }
    }
  }
};

SC_MODULE(pe_3) {
  sc_in_clk clk;
  sc_in<bool> rst;
  Connections::In< int32_t > v37;
  Connections::Out< int32_t > v38;
  SC_HAS_PROCESS(pe_3);
  pe_3(sc_module_name n) : sc_module(n), v37("v37"), v38("v38") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  void run() {
    v37.Reset();
    v38.Reset();
    wait();
    while (1) {
      l_S_k_0_k4: for (int k4 = 0; k4 < 8; k4++) {	// L62
        int32_t v40 = v37.Pop();	// L63
        int32_t v3;	// L64
        v3 = v40;	// L65
        int32_t v42 = v3;	// L66
        ap_int<33> v43 = v42;	// L67
        ap_int<33> v44 = v43 + 1;	// L68
        int32_t v45 = v44;	// L69
        int32_t w3;	// L70
        w3 = v45;	// L71
        int32_t v47 = w3;	// L72
        v38.Push(v47);	// L73
      }
    }
  }
};

SC_MODULE(drain_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  Connections::Out< int32_t > v48;
  Connections::In< int32_t > v49;
  SC_HAS_PROCESS(drain_0);
  drain_0(sc_module_name n) : sc_module(n), v48("v48"), v49("v49") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  void run() {
    v48.Reset();
    v49.Reset();
    wait();
    while (1) {
      l_S_k_0_k5: for (int k5 = 0; k5 < 8; k5++) {	// L78
        int32_t v51 = v49.Pop();	// L79
        v48.Push(v51);	// L80
      }
    }
  }
};

SC_MODULE(top) {
  sc_in_clk clk;
  sc_in<bool> rst;
  Connections::In< int32_t > v52;
  Connections::Out< int32_t > v53;
  Connections::Combinational< int32_t > v54;
  Connections::Combinational< int32_t > v55;
  Connections::Combinational< int32_t > v56;
  Connections::Combinational< int32_t > v57;
  Connections::Combinational< int32_t > v58;
  feed_0 u0;
  pe_0 u1;
  pe_1 u2;
  pe_2 u3;
  pe_3 u4;
  drain_0 u5;
  SC_CTOR(top) : v52("v52"), v53("v53"), v54("v54"), v55("v55"), v56("v56"), v57("v57"), v58("v58"), u0("u0"), u1("u1"), u2("u2"), u3("u3"), u4("u4"), u5("u5") {
    u0.clk(clk);
    u0.rst(rst);
    u0.v0(v52);
    u0.v1(v54);
    u1.clk(clk);
    u1.rst(rst);
    u1.v4(v54);
    u1.v5(v55);
    u2.clk(clk);
    u2.rst(rst);
    u2.v15(v55);
    u2.v16(v56);
    u3.clk(clk);
    u3.rst(rst);
    u3.v26(v56);
    u3.v27(v57);
    u4.clk(clk);
    u4.rst(rst);
    u4.v37(v57);
    u4.v38(v58);
    u5.clk(clk);
    u5.rst(rst);
    u5.v48(v53);
    u5.v49(v58);
  }
};

SC_MODULE(tb) {
  sc_clock clk;
  sc_signal<bool> rst;
  top dut;
  Connections::Combinational< int32_t > ch_v52;
  Connections::Combinational< int32_t > ch_v53;
  SC_HAS_PROCESS(tb);
  tb(sc_module_name n) : sc_module(n), clk("clk", 1, SC_NS), dut("dut"), ch_v52("ch_v52"), ch_v53("ch_v53") {
    dut.clk(clk); dut.rst(rst);
    dut.v52(ch_v52);
    dut.v53(ch_v53);
    SC_THREAD(src); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(snk); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
  }
  void src() {
    ch_v52.ResetWrite();
    wait();
    { std::ifstream _f("input0.data"); int32_t _v; for (int f = 0; f < 8; ++f) { _f >> _v; ch_v52.Push(_v); } }
  }
  void snk() {
    ch_v53.ResetRead();
    wait();
    { std::ofstream _f("output0.data"); for (int f = 0; f < 8; ++f) _f << ch_v53.Pop() << "\n"; }
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
