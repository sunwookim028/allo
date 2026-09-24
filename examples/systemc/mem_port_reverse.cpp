
//===------------------------------------------------------------*- C++ -*-===//
// Automatically generated file for SystemC (Catapult HLS / MatchLib Connections).
//===----------------------------------------------------------------------===//
#include <systemc.h>
#include <mc_connections.h>   // MatchLib Connections (LI valid/ready channels)
#include <connections/connections_fifo.h>  // vendor FWFT Connections::Fifo (buffered streams)
#include <ac_int.h>
#include <ac_fixed.h>
#include <ac_channel.h>     // local self-FIFO streams (one-kernel put+get+status)
#include <ac_std_float.h>   // IEEE floats: ac_ieee_float<binaryNN>
#include <cstring>          // std::memcpy for bit-reinterpret (bitcast)
#include <stdint.h>
// f16: Catapult has no native `half`; alias it to ac_ieee_float<binary16>.
// (f32 -> ac_ieee_float<binary32> is emitted directly by getCatapultTypeName.)
typedef ac_ieee_float<binary16> half;
#include <iostream>
#include <fstream>
#include <iomanip>          // std::setprecision for lossless float tb output
#include <algorithm>
// --- float support helpers (half / ac_ieee_float<binary32> / double) ---
// Floats have no implicit int/stream conversions (and half is non-trivial), so
// memory ports (which transport a raw bit pattern in an ac_int req word) and the
// testbench (text I/O + waveform trace) need these shims.
// float -> raw bits. Catapult's front end rejects memcpy/void* casts under
// __SYNTHESIS__ (CIN-71: "Invalid pointer cast from ... to void *"), which broke
// every float MEMORY-PORT design at csynth. Floats use their IEEE type's own bit
// accessor (data_ac_int(), works in csim AND synthesis); the generic template is
// a csim-only fallback for any non-float mem-port payload (memcpy guarded out of
// synthesis, where it would only be reached by an untested double mem-port).
inline unsigned long long _fbits(const half &v) {
  return (unsigned long long)v.data_ac_int().to_uint();
}
inline unsigned long long _fbits(const ac_ieee_float<binary32> &v) {
  return (unsigned long long)v.data_ac_int().to_uint();
}
template <class T> inline unsigned long long _fbits(const T &v) {
#ifdef __SYNTHESIS__
  return (unsigned long long)v;
#else
  unsigned long long b = 0; std::memcpy(&b, &v, sizeof(T)); return b;
#endif
}
// Reconstruct a memory element from the DATAW raw bits: value-cast for integers,
// set_data() bit-load for floats (a value-cast would corrupt the float; memcpy is
// rejected under synthesis as above).
// tb: data files hold float text -> read a float and convert.
inline std::istream &operator>>(std::istream &is, half &h) { float f; is >> f; h = half(f); return is; }
inline std::istream &operator>>(std::istream &is, ac_ieee_float<binary32> &h) {
  float f; is >> f; h = ac_ieee_float<binary32>(f); return is;
}
// tb/Connections waveform trace of a float (trace its raw bit pattern). Needed
// because Connections/sc_signal ports templated on these types call sc_trace.
inline void sc_trace(sc_core::sc_trace_file *tf, const half &h, const std::string &n) {
  sc_trace(tf, (unsigned short)_fbits(h), n);
}
inline void sc_trace(sc_core::sc_trace_file *tf, const ac_ieee_float<binary32> &h,
                     const std::string &n) {
  sc_trace(tf, (unsigned)_fbits(h), n);
}
// Make ac_ieee_float<Format> a valid Connections channel/Combinational payload.
// Connections' marshaller.h ships Wrapped<> specializations for ac_std_float,
// ac::bfloat16 and ac_float, but NOT ac_ieee_float -- so a Stream/Channel of
// f32 (ac_ieee_float<binary32>) fails to synthesize (marshaller.h needs
// T::width + T::Marshall()). This specialization marshals the raw IEEE bits
// (data_ac_int()/set_data(), a lossless bit copy through the standard ac_int
// AddField path), exactly as the ac_std_float specialization does. Wrapped lives
// at GLOBAL scope (marshaller.h opens no namespace), so this must be global too.
// Guarded: the SCVerify wrapper TU (sysc_sim.cpp) needs this same specialization,
// and the cosim flow re-injects an identical guarded copy into sysc_sim.h; the
// guard keeps the kernel.cpp TU (which sees both this AND, via mc_scverify ->
// sysc_sim.h, the injected copy) from double-defining it.
#ifndef ALLO_IEEE_FLOAT_MARSHALL_DEF
#define ALLO_IEEE_FLOAT_MARSHALL_DEF
template <ac_ieee_float_format Format>
class Wrapped<ac_ieee_float<Format> > {
public:
  ac_ieee_float<Format> val;
  Wrapped() : val(0.0f) {}
  Wrapped(const ac_ieee_float<Format> &v) : val(v) {}
  static const unsigned int width = ac_ieee_float<Format>::width;
  static const bool is_signed = 1;
  template <unsigned int Size>
  void Marshall(Marshaller<Size> &m) {
    ac_int<ac_ieee_float<Format>::width, true> bits = val.data_ac_int();
    m & bits;                // packs bits on marshal-out, fills bits on marshal-in
    val.set_data(bits);      // write unpacked bits back into the float
  }
};
#endif // ALLO_IEEE_FLOAT_MARSHALL_DEF
// mc_scverify.h MUST come after the float shims above: under CCS_DUT_RTL it pulls
// in the SCVerify RTL wrapper (sysc_sim.h), which instantiates the ports'
// Wrapped<ac_ieee_float<...>>::Marshall and sc_trace(ac_ieee_float) at include
// time. A C++ specialization must be visible BEFORE the first implicit
// instantiation, so both must be declared above this include -- otherwise the
// wrapper binds the primary Wrapped<T> template ("ac_ieee_float has no member
// Marshall") and float-channel cosim fails to compile.
#include <mc_scverify.h>      // SCVerify testbench macros (CCS_MAIN / CCS_DESIGN)
// Single-shot completion counter (csim/testbench ONLY). Each kernel bumps this
// once, right after its body finishes its single pass; the sc_main testbench for
// memory-mapped-output designs advances the clock until all kernels are done
// before reading the memories out. Declared unconditionally so the tb compiles
// under __SYNTHESIS__ too, but the kernel's increment is guarded out of synthesis
// (see emitKernelModule) so the synthesized logic stays side-effect free.
static long __allo_done = 0;
// The reused body emits bare max()/min() for the Allo max/min intrinsics (Vitis
// resolves them via hls::); bind them to std:: so the same body compiles here.
using std::max;
using std::min;
// The reused Vivado-emitter body prints Vitis ap_(u)int types; alias them to
// Catapult's ac_int so the same body compiles.
// TODO: REVISIT — this ap_int/ap_rng shim exists ONLY to keep the reused Vivado body
// compiling; ap_int is just ac_int + two Vitis affordances (x(hi,lo) via ap_rng, >64-bit
// operator long long() narrowing). Emitting ac_int natively everywhere (a type-name
// override like getCatapultTypeName, plus the bit-op overrides already doing slc/set_slc)
// would let this shim + ap_rng be deleted. See EmitSystemC.md.
// ap_(u)int is a thin ac_int subclass adding the two Vitis affordances the reused
// body relies on and ac_int lacks:
//  (1) the x(hi,lo) BIT-RANGE operator (packed streams unpack via v(31,16) etc) —
//      a proxy that extracts on read and inserts on write, for const/runtime hi,lo;
//  (2) for W>64 only, an implicit narrowing conversion (`int32_t x = wide;`, e.g. a
//      GEMM accumulator past 64 bits) which ac_int omits above 64 bits.
// Adding operator() doesn't disturb arithmetic (it's the call operator, not a
// conversion); the W>64 narrowing stays gated so ac_int's own operators keep
// winning overload resolution (no builtin-conversion ambiguity).
//
// CSYNTH NOTE: the subclass-of-ac_int form below is CSIM-ONLY. Catapult's front-
// end treats ac_int as a builtin, so a struct deriving from it trips
// "struct assignment from non-struct type" (CIN-15) on every `v = <ac_int expr>;`
// — breaking synthesis branch-wide. Under __SYNTHESIS__ we therefore fall back to
// a PLAIN ac_int alias, which loses the two csim affordances (the x(hi,lo) bit-
// range and the >64-bit implicit narrowing). Consequence: every non-bit-slicing
// design synthesizes; a design that actually bit-slices (packed streams) fails at
// its `(hi,lo)` site instead — a clear, local error, and those need native ac_int
// .slc emission anyway. csim keeps the full-featured shim so behavior is unchanged.
// TODO: REVISIT — packed/wide bit-slicing fails at csynth here (CIN-15 forces this bare
// alias); emit those slices as native ac_int .slc<>()/.set_slc() (cf. emitGetSlice/
// emitSetSlice) so packed streams synthesize. Empirically checked 2026-08 (see EmitSystemC.md).
#ifdef __SYNTHESIS__
template <int W> using ap_int = ac_int<W, true>;
template <int W> using ap_uint = ac_int<W, false>;
#else
template <class AC> struct ap_rng {
  AC &r;
  int hi, lo;
  ap_rng(AC &x, int h, int l) : r(x), hi(h), lo(l) {}
  operator long long() const { // read bits [hi:lo]
    AC m = (AC(1) << (hi - lo + 1)) - 1;
    return ((r >> lo) & m).to_int64();
  }
  template <class V> ap_rng &operator=(V v) { // write bits [hi:lo] = v
    AC m = (AC(1) << (hi - lo + 1)) - 1;
    r = (r & ~(m << lo)) | ((AC(v) & m) << lo);
    return *this;
  }
};
// `using ac_int<W,S>::ac_int;` inherits ac_int's ctors, but C++ never inherits the
// copy-shaped ctor (parameter = the base type), so building the wrapper from an ac_int
// expression -- what every arith/logic op yields -- needs this explicit converting ctor.
// Without it construction fails "non-scalar conversion" at EVERY width (verified 16/22/34/
// 64/68); standard widths 8/16/32/64 dodge it only because they emit as native (u)intN_t,
// not this shim. (>64 additionally has no native-int narrowing -- see getSCTypeName.)
template <int W, bool Big = (W > 64)> struct ap_sel {
  struct s : ac_int<W, true> {
    using ac_int<W, true>::ac_int;
    s() = default;
    s(const ac_int<W, true> &v) : ac_int<W, true>(v) {}
    ap_rng<ac_int<W, true>> operator()(int hi, int lo) { return {*this, hi, lo}; }
  };
  struct u : ac_int<W, false> {
    using ac_int<W, false>::ac_int;
    u() = default;
    u(const ac_int<W, false> &v) : ac_int<W, false>(v) {}
    ap_rng<ac_int<W, false>> operator()(int hi, int lo) { return {*this, hi, lo}; }
  };
};
template <int W> struct ap_sel<W, true> {
  struct s : ac_int<W, true> {
    using ac_int<W, true>::ac_int;
    s() = default;
    s(const ac_int<W, true> &v) : ac_int<W, true>(v) {}
    ap_rng<ac_int<W, true>> operator()(int hi, int lo) { return {*this, hi, lo}; }
    operator long long() const { return this->to_int64(); }
  };
  struct u : ac_int<W, false> {
    using ac_int<W, false>::ac_int;
    u() = default;
    u(const ac_int<W, false> &v) : ac_int<W, false>(v) {}
    ap_rng<ac_int<W, false>> operator()(int hi, int lo) { return {*this, hi, lo}; }
    operator unsigned long long() const { return this->to_uint64(); }
  };
};
template <int W> using ap_int = typename ap_sel<W>::s;
template <int W> using ap_uint = typename ap_sel<W>::u;
#endif

// A DEFINED zero for any memory element type. `T()` is not usable: ac_ieee_float's
// default constructor is `{}`, which leaves the payload uninitialised, and a plain
// literal 0 does not convert (sc_out<ac_ieee_float>::write takes const T& and there is
// no implicit int->T conversion). Only needed for the reset action; the float cases get
// an explicit construction from 0.0f.
template <typename T> inline T _mem_zero() { return (T)0; }
template <> inline half _mem_zero<half>() { return half(0.0f); }
template <>
inline ac_ieee_float<binary32> _mem_zero<ac_ieee_float<binary32> >() {
  return ac_ieee_float<binary32>(0.0f);
}

// Pin-interface memory: the counterpart of the RAM pins a kernel presents. Used in BOTH
// places, which is the point -- a kernel cannot tell whether its memory sits inside the
// design (a replicated, multi-client array) or in the testbench (a single-client array
// exposed at the boundary). Only the binding differs.
//
// Synchronous read: the address is captured on the clock edge and `q` is presented on the
// NEXT one, which is why the kernel-side _rd() accessor waits twice.
template <typename T, int SIZE, int ADDRW>
SC_MODULE(AlloMemPins) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_in< ac_int<ADDRW, false> > radr;
  sc_in<bool> re;
  sc_out<T> q;
  sc_out<bool> rrdy;
  sc_in< ac_int<ADDRW, false> > wadr;
  sc_in<T> d;
  sc_in<bool> we;
  sc_out<bool> wrdy;
  T mem[SIZE];
  SC_HAS_PROCESS(AlloMemPins);
  AlloMemPins(sc_module_name n) : sc_module(n) {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  void run() {
    // NOTE: mem[] is deliberately NOT cleared here -- the testbench preloads it before
    // reset is released, exactly as it did with AlloMem, and clearing would wipe that.
    // This memory is ALWAYS READY: single-cycle, unarbitrated, no bank conflicts. The
    // ready lines exist for the integrator's memory, which may not be -- here they are
    // simply held high, so the accessors' stall loops fall through in one edge.
    rrdy.write(true);
    wrdy.write(true);
    // `q` is a driven sc_out, so Catapult requires it written in the reset action too
    // (CIN-233). Only reachable when this memory is INSIDE the design -- a replicated,
    // multi-client array -- since a testbench-side instance is never synthesized.
    //
    // NOT `T()`: ac_ieee_float's default constructor is `{}`, so that would leave the
    // payload UNINITIALISED and the reset value of a float data pin indeterminate.
    q.write(_mem_zero<T>());
    wait();
    while (1) {
      // Both accesses are GUARDED BY THEIR ENABLE and bounds-checked. Not defensive
      // padding: a write-only array leaves `radr` an undriven sc_signal, and reading
      // mem[] at that uninitialized address indexes out of bounds and segfaults --
      // which is exactly how this first showed up (mem_port_scatter).
      unsigned wa = wadr.read().to_uint();
      if (we.read() && wa < (unsigned)SIZE)
        mem[wa] = d.read();
      unsigned ra = radr.read().to_uint();
      if (re.read() && ra < (unsigned)SIZE)
        q.write(mem[ra]);
      wait();
    }
  }
};

// Depth-N buffered stream channel (Stream[T, N>=1]) — a TWO-THREAD ring-buffer
// FIFO that Catapult can synthesize AND schedule.
//
// Single-thread ring FIFO. This was historically TWO threads (enq/deq) because a
// single SC_THREAD doing BOTH in.PopNB() and out.PushNB() couples the two
// handshakes' sc_signal writes, and Catapult's DEFAULT -IO_MODE fixed can't place
// them at the fixed cycle offsets its iomode requires -> the loop won't close at
// II=1 (SCHD-30). The systemc flow now sets -IO_MODE super (catapult.py), which
// lets the scheduler place BOTH handshakes within the loop window -- the same fix
// that makes multi-handshake router kernels schedule -- so the bidirectional FIFO
// schedules in ONE thread. That collapses everything the split needed: the
// cross-thread head/tail sc_signals, the shared sc_signal register file (the
// HIER-41 dodge for a plain array shared across threads), and the sacrificed N+1
// slot -> plain single-owner locals + a count. PushNB runs before PopNB so an
// empty FIFO does not forward an arriving element the same cycle (preserves the
// >=1-cycle buffer latency of the two-thread version).
template <typename T, int N>
SC_MODULE(AlloFifo) {
  sc_in_clk clk;
  sc_in<bool> rst;
  Connections::In<T> in;
  Connections::Out<T> out;
  // Occupancy sidebands: regular Connections In/Out ports cannot report Empty/Full
  // in HLS (In::Empty() reads a sim-only latched-data flag our channel never sets),
  // so a Stream.empty()/full() query reads these plain status wires instead. `count`
  // mirrors the thread-local `cnt`; empty_o/full_o are driven COMBINATIONALLY from it
  // (registered-pointers + combinational-empty, matching RTL FIFO semantics -- a
  // registered flag would be a cycle stale and shift arbitration traces).
  sc_signal<int> count;
  sc_out<bool> empty_o;
  sc_out<bool> full_o;
  SC_HAS_PROCESS(AlloFifo);
  AlloFifo(sc_module_name nm) : sc_module(nm), in("in"), out("out") {
    SC_THREAD(run); sensitive << clk.pos(); async_reset_signal_is(rst, false);
    SC_METHOD(set_flags); sensitive << count;
  }
  void set_flags() {
    empty_o.write(count.read() == 0);
    full_o.write(count.read() == N);
  }
  void run() {
    in.Reset();
    out.Reset();
    T buf[N];                         // single owner -> plain local, no HIER-41
    int head = 0, tail = 0, cnt = 0;  // full N slots (cnt distinguishes full/empty)
    count.write(0);
    wait();
    while (1) {
      if (cnt > 0 && out.PushNB(buf[head])) { head = (head + 1) % N; cnt = cnt - 1; }
      if (cnt < N) {
        T v;
        if (in.PopNB(v)) { buf[tail] = v; tail = (tail + 1) % N; cnt = cnt + 1; }
      }
      count.write(cnt);               // mirror cnt out for the combinational flags
      wait();
    }
  }
};

// AlloFifoC -- the ACTIVE buffered-stream FIFO. Subclasses the vendor's
// first-word-fall-through Connections::Fifo (enq/deq In/Out ports) and adds
// coherent occupancy flags from the port signals (deq.vld == !empty,
// enq.rdy == !full, the vendor Fifo_with_idle pattern). Because Connections::Fifo
// is FWFT (the head word is presented combinationally and a slot is only freed
// AFTER the consumer's Pop handshake completes), empty_o agrees with what the
// consumer can read THIS cycle -- fixing the last-item drop that AlloFifo's eager
// PushNB + early decrement caused for empty()/full()-polling consumers. It is
// also smaller/faster (~49% area, higher Fmax) since it uses nbits<N> pointers
// and a 1-bit full instead of a 32-bit count. AlloFifo above is kept as an
// unused reference/backup.
template <typename T, int N>
struct AlloFifoC : public Connections::Fifo<T, N> {
  SC_HAS_PROCESS(AlloFifoC);
  typedef Connections::Fifo<T, N> Base;
  using Base::enq;   // In<T>  -- the top binds this via .enq(<stream>_in)
  using Base::deq;   // Out<T> -- the top binds this via .deq(<stream>_out)
  using Base::sensitive;
  sc_out<bool> empty_o;
  sc_out<bool> full_o;
  AlloFifoC(sc_module_name nm) : Base(nm), empty_o("empty_o"), full_o("full_o") {
    SC_METHOD(gen_status);
    sensitive << enq._RDYNAME_ << deq._VLDNAME_;
  }
  void gen_status() {
    empty_o.write(!deq._VLDNAME_.read()); // deq.vld == !empty (FWFT-coherent)
    full_o.write(!enq._RDYNAME_.read());  // enq.rdy == !full
  }
};

SC_MODULE(rev_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_out<bool> done;
  sc_out< ac_int<3, false> > v0_radr;
  sc_out<bool> v0_re;
  sc_in< ac_int<32, true> > v0_q;
  sc_in<bool> v0_rrdy;
  Connections::Out< ac_int<32, true> > v1;
  SC_HAS_PROCESS(rev_0);
  rev_0(sc_module_name n) : sc_module(n), done("done"), v1("v1") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v0_rd(ac_int<3, false> addr) {
    v0_radr.write(addr); v0_re.write(true);
    wait();                    // edge N: address captured
    v0_re.write(false);
    wait();                    // data valid on this edge
    return v0_q.read();
  }
  void run() {
    v1.Reset();
    v0_radr.write(0);
    v0_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    l_S_i_0_i: for (int i = 0; i < 8; i++) {	// L4
      int32_t v3;
      v3 = v0_rd((ac_int<3, false>)(((((i * -1) + 7)))));	// L5
      ac_int<33, true> v4 = v3;	// L6
      ac_int<33, true> v5 = v4 + 1;	// L7
      int32_t v6 = v5;	// L8
      v1.Push(v6);	// L9
    }
    done.write(true);  // RTL-observable completion (see the done port)
#ifndef __SYNTHESIS__
    __allo_done++; // csim: this kernel finished its single pass
#endif
    while (1) { wait(); }
  }
};

SC_MODULE(top) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_out<bool> done;
  Connections::Out< ac_int<32, true> > v8;
  rev_0 u0;
  sc_signal<bool> u0_done;
  sc_out< ac_int<3, false> > v7_radr;
  sc_out<bool> v7_re;
  sc_in< ac_int<32, true> > v7_q;
  sc_in<bool> v7_rrdy;
  SC_CTOR(top) : v8("v8"), u0("u0") {
    u0.clk(clk);
    u0.rst(rst);
    u0.done(u0_done);
    u0.v0_radr(v7_radr);
    u0.v0_re(v7_re);
    u0.v0_q(v7_q);
    u0.v0_rrdy(v7_rrdy);
    u0.v1(v8);
    SC_METHOD(_agg_done); sensitive << u0_done;
  }
  void _agg_done() { done.write(u0_done.read()); }
};

SC_MODULE(tb) {
  sc_clock clk;
  sc_signal<bool> rst;
  top dut;
  sc_signal<bool> done_sig;  // DUT completion (polled by sc_main)
  Connections::Combinational< ac_int<32, true> > ch_v8;
  sc_signal< ac_int<3, false> > mp0_0_radr, mp0_0_wadr;
  sc_signal<bool> mp0_0_re, mp0_0_we, mp0_0_rrdy, mp0_0_wrdy;
  sc_signal< ac_int<32, true> > mp0_0_q, mp0_0_d;
  AlloMemPins< ac_int<32, true>, 8, 3 > mp0_0_mem;
  SC_HAS_PROCESS(tb);
  tb(sc_module_name n) : sc_module(n), clk("clk", 1, SC_NS), dut("dut"), ch_v8("ch_v8"), mp0_0_mem("mp0_0_mem") {
    dut.clk(clk); dut.rst(rst); dut.done(done_sig);
    dut.v8(ch_v8);
    mp0_0_mem.clk(clk); mp0_0_mem.rst(rst);
    dut.v7_radr(mp0_0_radr); mp0_0_mem.radr(mp0_0_radr);
    dut.v7_re(mp0_0_re); mp0_0_mem.re(mp0_0_re);
    dut.v7_q(mp0_0_q); mp0_0_mem.q(mp0_0_q);
    dut.v7_rrdy(mp0_0_rrdy); mp0_0_mem.rrdy(mp0_0_rrdy);
    mp0_0_mem.wadr(mp0_0_wadr);
    mp0_0_mem.d(mp0_0_d);
    mp0_0_mem.we(mp0_0_we);
    mp0_0_mem.wrdy(mp0_0_wrdy);
    SC_THREAD(src); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(snk); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
  }
  void src() {
    wait();
  }
  void snk() {
    ch_v8.ResetRead();
    wait();
    { std::ofstream _f("output0.data"); for (int f = 0; f < 8; ++f) _f << (long long)(ch_v8.Pop()) << "\n"; }
    sc_stop();
  }
};

int sc_main(int, char *[]) {
  static tb t("t");
  #ifdef CONNECTIONS_ACCURATE_SIM
  Connections::set_sim_clk(&t.clk);
  #endif
  { std::ifstream _f("input0.data"); long long _v; for (int f = 0; f < 8; ++f) { _f >> _v; t.mp0_0_mem.mem[f] = (ac_int<32, true>)_v; } }
  t.rst = 0; sc_start(1, SC_NS);
  t.rst = 1;
  sc_start();
  return 0;
}
