
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

SC_MODULE(node_0_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_out<bool> done;
  sc_out< ac_int<1, false> > v0_radr;
  sc_out<bool> v0_re;
  sc_in< ac_int<32, true> > v0_q;
  sc_in<bool> v0_rrdy;
  Connections::Out< ac_int<26, false> > v1;
  Connections::Out< ac_int<26, false> > v2;
  Connections::Out< ac_int<26, false> > v3;
  Connections::Out< ac_int<26, false> > v4;
  Connections::Out< ac_int<17, false> > v5;
  Connections::Out< ac_int<17, false> > v6;
  Connections::Out< ac_int<17, false> > v7;
  Connections::Out< ac_int<17, false> > v8;
  Connections::Out< ac_int<32, true> > v9;
  Connections::Out< ac_int<32, true> > v10;
  Connections::Out< ac_int<32, true> > v11;
  Connections::Out< ac_int<32, true> > v12;
  Connections::Out< ac_int<32, true> > v13;
  Connections::Out< ac_int<32, true> > v14;
  Connections::Out< ac_int<32, true> > v15;
  Connections::Out< ac_int<32, true> > v16;
  Connections::In< ac_int<26, false> > v17;
  Connections::In< ac_int<26, false> > v18;
  Connections::In< ac_int<26, false> > v19;
  Connections::In< ac_int<26, false> > v20;
  Connections::In< ac_int<32, true> > v21;
  Connections::In< ac_int<32, true> > v22;
  Connections::In< ac_int<32, true> > v23;
  Connections::In< ac_int<32, true> > v24;
  Connections::In< ac_int<32, true> > v25;
  Connections::In< ac_int<32, true> > v26;
  Connections::In< ac_int<32, true> > v27;
  Connections::In< ac_int<32, true> > v28;
  Connections::In< ac_int<17, false> > v29;
  Connections::In< ac_int<17, false> > v30;
  Connections::In< ac_int<17, false> > v31;
  Connections::In< ac_int<17, false> > v32;
  SC_HAS_PROCESS(node_0_0);
  node_0_0(sc_module_name n) : sc_module(n), done("done"), v1("v1"), v2("v2"), v3("v3"), v4("v4"), v5("v5"), v6("v6"), v7("v7"), v8("v8"), v9("v9"), v10("v10"), v11("v11"), v12("v12"), v13("v13"), v14("v14"), v15("v15"), v16("v16"), v17("v17"), v18("v18"), v19("v19"), v20("v20"), v21("v21"), v22("v22"), v23("v23"), v24("v24"), v25("v25"), v26("v26"), v27("v27"), v28("v28"), v29("v29"), v30("v30"), v31("v31"), v32("v32") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v0_rd(ac_int<1, false> addr) {
    v0_radr.write(addr); v0_re.write(true);
    wait();                    // edge N: address captured
    v0_re.write(false);
    wait();                    // data valid on this edge
    return v0_q.read();
  }
  void run() {
    v1.Reset();
    v2.Reset();
    v3.Reset();
    v4.Reset();
    v5.Reset();
    v6.Reset();
    v7.Reset();
    v8.Reset();
    v9.Reset();
    v10.Reset();
    v11.Reset();
    v12.Reset();
    v13.Reset();
    v14.Reset();
    v15.Reset();
    v16.Reset();
    v17.Reset();
    v18.Reset();
    v19.Reset();
    v20.Reset();
    v21.Reset();
    v22.Reset();
    v23.Reset();
    v24.Reset();
    v25.Reset();
    v26.Reset();
    v27.Reset();
    v28.Reset();
    v29.Reset();
    v30.Reset();
    v31.Reset();
    v32.Reset();
    v0_radr.write(0);
    v0_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    #pragma hls_resource irf_rsc variables="irf" map_to_module="[Register]"
    int32_t irf[8];	// L41
    for (int v34 = 0; v34 < 8; v34++) {	// L42
      irf[v34] = 0;	// L42
    }
    #pragma hls_resource drf_rsc variables="drf" map_to_module="[Register]"
    half drf[8];	// L43
    for (int v36 = 0; v36 < 8; v36++) {	// L44
      drf[v36] = half(0.000000f);	// L44
    }
    #pragma hls_resource drf_full_rsc variables="drf_full" map_to_module="[Register]"
    int32_t drf_full[8];	// L45
    for (int v38 = 0; v38 < 8; v38++) {	// L46
      drf_full[v38] = 0;	// L46
    }
    int32_t dsmask;	// L47
    dsmask = 0;	// L48
    int32_t crv_vld;	// L49
    crv_vld = 0;	// L50
    half crv_data;	// L51
    crv_data = half(0.000000f);	// L52
    int32_t crv_addr;	// L53
    crv_addr = 0;	// L54
    int32_t crv_mode;	// L55
    crv_mode = 0;	// L56
    int32_t crv_raw;	// L57
    crv_raw = 0;	// L58
    int32_t csd_vld;	// L59
    csd_vld = 0;	// L60
    ac_int<26, false> csd_pkt;	// L61
    csd_pkt = 0;	// L62
    int32_t csd_dir;	// L63
    csd_dir = 0;	// L64
    int32_t row_id;	// L65
    row_id = 0;	// L66
    int32_t col_id;	// L67
    col_id = 0;	// L68
    ac_int<26, false> oe_r;	// L69
    oe_r = 0;	// L70
    ac_int<26, false> ow_r;	// L71
    ow_r = 0;	// L72
    ac_int<26, false> on_r;	// L73
    on_r = 0;	// L74
    ac_int<26, false> os_r;	// L75
    os_r = 0;	// L76
    ac_int<17, false> txn_r;	// L77
    txn_r = 0;	// L78
    ac_int<17, false> txs_r;	// L79
    txs_r = 0;	// L80
    ac_int<17, false> txw_r;	// L81
    txw_r = 0;	// L82
    ac_int<17, false> txe_r;	// L83
    txe_r = 0;	// L84
    #pragma hls_resource hold_v_rsc variables="hold_v" map_to_module="[Register]"
    half hold_v[4][2];	// L85
    for (int v59 = 0; v59 < 4; v59++) {	// L86
      for (int v60 = 0; v60 < 2; v60++) {	// L86
        hold_v[v59][v60] = half(0.000000f);	// L86
      }
    }
    #pragma hls_resource hold_cnt_rsc variables="hold_cnt" map_to_module="[Register]"
    uint8_t hold_cnt[4];	// L87
    for (int v62 = 0; v62 < 4; v62++) {	// L88
      hold_cnt[v62] = 0;	// L88
    }
    #pragma hls_resource rbuf_rsc variables="rbuf" map_to_module="[Register]"
    ac_int<26, false> rbuf[4][2];	// L89
    for (int v64 = 0; v64 < 4; v64++) {	// L90
      for (int v65 = 0; v65 < 2; v65++) {	// L90
        rbuf[v64][v65] = 0;	// L90
      }
    }
    #pragma hls_resource rbcnt_rsc variables="rbcnt" map_to_module="[Register]"
    uint8_t rbcnt[4];	// L91
    for (int v67 = 0; v67 < 4; v67++) {	// L92
      rbcnt[v67] = 0;	// L92
    }
    #pragma hls_resource rcred_rsc variables="rcred" map_to_module="[Register]"
    uint8_t rcred[4];	// L93
    for (int v69 = 0; v69 < 4; v69++) {	// L94
      rcred[v69] = 0;	// L94
    }
    uint8_t cre_r;	// L95
    cre_r = 2;	// L96
    uint8_t crw_r;	// L97
    crw_r = 2;	// L98
    uint8_t crs_r;	// L99
    crs_r = 2;	// L100
    uint8_t crn_r;	// L101
    crn_r = 2;	// L102
    #pragma hls_resource scred_rsc variables="scred" map_to_module="[Register]"
    int32_t scred[4];	// L103
    for (int v75 = 0; v75 < 4; v75++) {	// L104
      scred[v75] = 0;	// L104
    }
    #pragma hls_resource txp_v_rsc variables="txp_v" map_to_module="[Register]"
    int32_t txp_v[4];	// L105
    for (int v77 = 0; v77 < 4; v77++) {	// L106
      txp_v[v77] = 0;	// L106
    }
    #pragma hls_resource txp_d_rsc variables="txp_d" map_to_module="[Register]"
    half txp_d[4];	// L107
    for (int v79 = 0; v79 < 4; v79++) {	// L108
      txp_d[v79] = half(0.000000f);	// L108
    }
    #pragma hls_resource txp_r_rsc variables="txp_r" map_to_module="[Register]"
    int32_t txp_r[4];	// L109
    for (int v81 = 0; v81 < 4; v81++) {	// L110
      txp_r[v81] = 0;	// L110
    }
    #pragma hls_resource sc_r_rsc variables="sc_r" map_to_module="[Register]"
    int32_t sc_r[4];	// L111
    for (int v83 = 0; v83 < 4; v83++) {	// L112
      sc_r[v83] = 2;	// L112
    }
    int32_t cfg_isz;	// L113
    cfg_isz = 0;	// L114
    int32_t cfg_itsz;	// L115
    cfg_itsz = 0;	// L116
    uint8_t fetch_en;	// L117
    fetch_en = 0;	// L118
    uint8_t instr_cnt;	// L119
    instr_cnt = 0;	// L120
    uint8_t iter_cnt;	// L121
    iter_cnt = 0;	// L122
    uint8_t condition_reg;	// L123
    condition_reg = 0;	// L124
    #pragma hls_resource sb_v_rsc variables="sb_v" map_to_module="[Register]"
    uint8_t sb_v[5];	// L125
    for (int v91 = 0; v91 < 5; v91++) {	// L126
      sb_v[v91] = 0;	// L126
    }
    #pragma hls_resource sb_dst_rsc variables="sb_dst" map_to_module="[Register]"
    uint8_t sb_dst[5];	// L127
    for (int v93 = 0; v93 < 5; v93++) {	// L128
      sb_dst[v93] = 0;	// L128
    }
    #pragma hls_resource sb_cmp_rsc variables="sb_cmp" map_to_module="[Register]"
    uint8_t sb_cmp[5];	// L129
    for (int v95 = 0; v95 < 5; v95++) {	// L130
      sb_cmp[v95] = 0;	// L130
    }
    #pragma hls_resource sb_rtr_rsc variables="sb_rtr" map_to_module="[Register]"
    uint8_t sb_rtr[5];	// L131
    for (int v97 = 0; v97 < 5; v97++) {	// L132
      sb_rtr[v97] = 0;	// L132
    }
    #pragma hls_resource sb_inj_rsc variables="sb_inj" map_to_module="[Register]"
    uint8_t sb_inj[5];	// L133
    for (int v99 = 0; v99 < 5; v99++) {	// L134
      sb_inj[v99] = 0;	// L134
    }
    #pragma hls_resource sb_dir_rsc variables="sb_dir" map_to_module="[Register]"
    uint8_t sb_dir[5];	// L135
    for (int v101 = 0; v101 < 5; v101++) {	// L136
      sb_dir[v101] = 0;	// L136
    }
    #pragma hls_resource sb_id_rsc variables="sb_id" map_to_module="[Register]"
    uint8_t sb_id[5];	// L137
    for (int v103 = 0; v103 < 5; v103++) {	// L138
      sb_id[v103] = 0;	// L138
    }
    #pragma hls_resource sb_rvld_rsc variables="sb_rvld" map_to_module="[Register]"
    uint8_t sb_rvld[5];	// L139
    for (int v105 = 0; v105 < 5; v105++) {	// L140
      sb_rvld[v105] = 0;	// L140
    }
    #pragma hls_resource sb_ix_rsc variables="sb_ix" map_to_module="[Register]"
    uint8_t sb_ix[5];	// L141
    for (int v107 = 0; v107 < 5; v107++) {	// L142
      sb_ix[v107] = 0;	// L142
    }
    #pragma hls_resource sb_long_rsc variables="sb_long" map_to_module="[Register]"
    uint8_t sb_long[5];	// L143
    for (int v109 = 0; v109 < 5; v109++) {	// L144
      sb_long[v109] = 0;	// L144
    }
    #pragma hls_resource resq_rsc variables="resq" map_to_module="[Register]"
    half resq[8];	// L145
    for (int v111 = 0; v111 < 8; v111++) {	// L146
      resq[v111] = half(0.000000f);	// L146
    }
    #pragma hls_resource cmpq_rsc variables="cmpq" map_to_module="[Register]"
    uint8_t cmpq[8];	// L147
    for (int v113 = 0; v113 < 8; v113++) {	// L148
      cmpq[v113] = 0;	// L148
    }
    uint8_t resq_wr;	// L149
    resq_wr = 0;	// L150
    ac_int<26, false> zpkt;	// L151
    zpkt = 0;	// L152
    ac_int<17, false> zsys;	// L153
    zsys = 0;	// L154
    int32_t zcr;	// L155
    zcr = 0;	// L156
    int32_t v118;
    v118 = v0_rd((ac_int<1, false>)(((0) + (0))));	// L157
    ac_int<33, true> v119 = v118;	// L158
    ac_int<33, true> v120 = v119 - 1;	// L159
    int v121 = v120;	// L160
    for (int v122 = 0; v122 < v121; v122 += 1) {	// L161
      ac_int<26, true> v123 = zpkt;	// L162
      v1.Push(v123);	// L163
      ac_int<26, true> v124 = zpkt;	// L164
      v2.Push(v124);	// L165
      ac_int<26, true> v125 = zpkt;	// L166
      v3.Push(v125);	// L167
      ac_int<26, true> v126 = zpkt;	// L168
      v4.Push(v126);	// L169
      ac_int<17, true> v127 = zsys;	// L170
      v5.Push(v127);	// L171
      ac_int<17, true> v128 = zsys;	// L172
      v6.Push(v128);	// L173
      ac_int<17, true> v129 = zsys;	// L174
      v7.Push(v129);	// L175
      ac_int<17, true> v130 = zsys;	// L176
      v8.Push(v130);	// L177
      int32_t v131 = zcr;	// L178
      v9.Push(v131);	// L179
      int32_t v132 = zcr;	// L180
      v10.Push(v132);	// L181
      int32_t v133 = zcr;	// L182
      v11.Push(v133);	// L183
      int32_t v134 = zcr;	// L184
      v12.Push(v134);	// L185
      int32_t v135 = zcr;	// L186
      v13.Push(v135);	// L187
      int32_t v136 = zcr;	// L188
      v14.Push(v136);	// L189
      int32_t v137 = zcr;	// L190
      v15.Push(v137);	// L191
      int32_t v138 = zcr;	// L192
      v16.Push(v138);	// L193
    }
    ac_int<26, true> v139 = oe_r;	// L195
    v1.Push(v139);	// L196
    ac_int<26, true> v140 = ow_r;	// L197
    v2.Push(v140);	// L198
    ac_int<26, true> v141 = os_r;	// L199
    v3.Push(v141);	// L200
    ac_int<26, true> v142 = on_r;	// L201
    v4.Push(v142);	// L202
    ac_int<17, true> v143 = txe_r;	// L203
    v5.Push(v143);	// L204
    ac_int<17, true> v144 = txw_r;	// L205
    v6.Push(v144);	// L206
    ac_int<17, true> v145 = txs_r;	// L207
    v7.Push(v145);	// L208
    ac_int<17, true> v146 = txn_r;	// L209
    v8.Push(v146);	// L210
    int8_t v147 = cre_r;	// L211
    v9.Push(v147);	// L212
    int8_t v148 = crw_r;	// L213
    v10.Push(v148);	// L214
    int8_t v149 = crs_r;	// L215
    v11.Push(v149);	// L216
    int8_t v150 = crn_r;	// L217
    v12.Push(v150);	// L218
    int32_t v151 = sc_r[0];	// L219
    v15.Push(v151);	// L220
    int32_t v152 = sc_r[1];	// L221
    v16.Push(v152);	// L222
    int32_t v153 = sc_r[2];	// L223
    v13.Push(v153);	// L224
    int32_t v154 = sc_r[3];	// L225
    v14.Push(v154);	// L226
#ifdef __SYNTHESIS__
    done.write(true);  // steady-state: no completion, so assert on entry (the post-body write is unreachable here)
    #pragma hls_pipeline_init_interval 1
    while (1) {  // steady-state loop (was `for t`): 1 iteration = 1 step
#else
    #pragma hls_pipeline_init_interval 1
    l_steady: for (int t = 0; t < 55; t += 1) {
#endif
      ac_int<26, false> v156 = v17.Pop();	// L228
      ac_int<26, false> p_w;	// L229
      p_w = v156;	// L230
      ac_int<26, false> v158 = v18.Pop();	// L231
      ac_int<26, false> p_e;	// L232
      p_e = v158;	// L233
      ac_int<26, false> v160 = v19.Pop();	// L234
      ac_int<26, false> p_n;	// L235
      p_n = v160;	// L236
      ac_int<26, false> v162 = v20.Pop();	// L237
      ac_int<26, false> p_s;	// L238
      p_s = v162;	// L239
      int32_t v164 = v21.Pop();	// L240
      uint8_t v165 = rcred[0];	// L241
      ac_int<33, true> v166 = v165;	// L242
      ac_int<33, true> v167 = v164;	// L243
      ac_int<33, true> v168 = v166 + v167;	// L244
      uint8_t v169 = v168;	// L245
      rcred[0] = v169;	// L246
      int32_t v170 = v22.Pop();	// L247
      uint8_t v171 = rcred[1];	// L248
      ac_int<33, true> v172 = v171;	// L249
      ac_int<33, true> v173 = v170;	// L250
      ac_int<33, true> v174 = v172 + v173;	// L251
      uint8_t v175 = v174;	// L252
      rcred[1] = v175;	// L253
      int32_t v176 = v23.Pop();	// L254
      uint8_t v177 = rcred[2];	// L255
      ac_int<33, true> v178 = v177;	// L256
      ac_int<33, true> v179 = v176;	// L257
      ac_int<33, true> v180 = v178 + v179;	// L258
      uint8_t v181 = v180;	// L259
      rcred[2] = v181;	// L260
      int32_t v182 = v24.Pop();	// L261
      uint8_t v183 = rcred[3];	// L262
      ac_int<33, true> v184 = v183;	// L263
      ac_int<33, true> v185 = v182;	// L264
      ac_int<33, true> v186 = v184 + v185;	// L265
      uint8_t v187 = v186;	// L266
      rcred[3] = v187;	// L267
      int32_t v188 = v25.Pop();	// L268
      int32_t v189 = scred[0];	// L269
      ac_int<33, true> v190 = v189;	// L270
      ac_int<33, true> v191 = v188;	// L271
      ac_int<33, true> v192 = v190 + v191;	// L272
      int32_t v193 = v192;	// L273
      scred[0] = v193;	// L274
      int32_t v194 = v26.Pop();	// L275
      int32_t v195 = scred[1];	// L276
      ac_int<33, true> v196 = v195;	// L277
      ac_int<33, true> v197 = v194;	// L278
      ac_int<33, true> v198 = v196 + v197;	// L279
      int32_t v199 = v198;	// L280
      scred[1] = v199;	// L281
      int32_t v200 = v27.Pop();	// L282
      int32_t v201 = scred[2];	// L283
      ac_int<33, true> v202 = v201;	// L284
      ac_int<33, true> v203 = v200;	// L285
      ac_int<33, true> v204 = v202 + v203;	// L286
      int32_t v205 = v204;	// L287
      scred[2] = v205;	// L288
      int32_t v206 = v28.Pop();	// L289
      int32_t v207 = scred[3];	// L290
      ac_int<33, true> v208 = v207;	// L291
      ac_int<33, true> v209 = v206;	// L292
      ac_int<33, true> v210 = v208 + v209;	// L293
      int32_t v211 = v210;	// L294
      scred[3] = v211;	// L295
      ac_int<26, false> fin[4];	// L296
      for (int v213 = 0; v213 < 4; v213++) {	// L297
        fin[v213] = 0;	// L297
      }
      ac_int<26, true> v214 = p_w;	// L298
      fin[0] = v214;	// L299
      ac_int<26, true> v215 = p_e;	// L300
      fin[1] = v215;	// L301
      ac_int<26, true> v216 = p_n;	// L302
      fin[2] = v216;	// L303
      ac_int<26, true> v217 = p_s;	// L304
      fin[3] = v217;	// L305
      l_S_d_1_d: for (int d = 0; d < 4; d++) {	// L306
        ac_int<26, false> v219 = fin[d];	// L307
        bool v220;
        ac_int<26, true> _bs_v220 = v219;
        v220 = _bs_v220[25];	// L308
        int32_t v221 = v220;	// L309
        bool v222 = v221 == 1;	// L310
        uint8_t v223 = rbcnt[d];	// L311
        int32_t v224 = v223;	// L312
        bool v225 = v224 < 2;	// L313
        bool v226 = v222 & v225;	// L314
        if (v226) {	// L315
          ac_int<26, false> v227 = fin[d];	// L316
          uint8_t v228 = rbcnt[d];	// L317
          int v229 = v228;	// L318
          rbuf[d][v229] = v227;	// L319
          uint8_t v230 = rbcnt[d];	// L320
          ac_int<33, true> v231 = v230;	// L321
          ac_int<33, true> v232 = v231 + 1;	// L322
          uint8_t v233 = v232;	// L323
          rbcnt[d] = v233;	// L324
        }
      }
      ac_int<26, false> hd[4];	// L327
      for (int v235 = 0; v235 < 4; v235++) {	// L328
        hd[v235] = 0;	// L328
      }
      int32_t hvld[4];	// L329
      for (int v237 = 0; v237 < 4; v237++) {	// L330
        hvld[v237] = 0;	// L330
      }
      int32_t hit[4];	// L331
      for (int v239 = 0; v239 < 4; v239++) {	// L332
        hit[v239] = 0;	// L332
      }
      int32_t axis[4];	// L333
      for (int v241 = 0; v241 < 4; v241++) {	// L334
        axis[v241] = 0;	// L334
      }
      int32_t v242 = col_id;	// L335
      axis[0] = v242;	// L336
      int32_t v243 = col_id;	// L337
      axis[1] = v243;	// L338
      int32_t v244 = row_id;	// L339
      axis[2] = v244;	// L340
      int32_t v245 = row_id;	// L341
      axis[3] = v245;	// L342
      l_S_d_2_d1: for (int d1 = 0; d1 < 4; d1++) {	// L343
        uint8_t v247 = rbcnt[d1];	// L344
        int32_t v248 = v247;	// L345
        bool v249 = v248 > 0;	// L346
        if (v249) {	// L347
          ac_int<26, false> v250 = rbuf[d1][0];	// L348
          hd[d1] = v250;	// L349
          hvld[d1] = 1;	// L350
          ac_int<26, false> v251 = hd[d1];	// L351
          ac_int<4, true> v252;
          ac_int<26, true> _bs_v252 = v251;
          v252 = _bs_v252.slc<4>(21);	// L352
          int32_t v253 = axis[d1];	// L353
          int32_t v254 = v252;	// L354
          bool v255 = v254 == v253;	// L355
          if (v255) {	// L356
            hit[d1] = 1;	// L357
          }
        }
      }
      ac_int<26, false> o_crv;	// L361
      o_crv = 0;	// L362
      int32_t crv_in;	// L363
      crv_in = -1;	// L364
      int32_t v258 = hit[3];	// L365
      bool v259 = v258 == 1;	// L366
      if (v259) {	// L367
        ac_int<26, false> v260 = hd[3];	// L368
        o_crv = v260;	// L369
        crv_in = 3;	// L370
      } else {
        int32_t v261 = hit[2];	// L372
        bool v262 = v261 == 1;	// L373
        if (v262) {	// L374
          ac_int<26, false> v263 = hd[2];	// L375
          o_crv = v263;	// L376
          crv_in = 2;	// L377
        } else {
          int32_t v264 = hit[1];	// L379
          bool v265 = v264 == 1;	// L380
          if (v265) {	// L381
            ac_int<26, false> v266 = hd[1];	// L382
            o_crv = v266;	// L383
            crv_in = 1;	// L384
          } else {
            int32_t v267 = hit[0];	// L386
            bool v268 = v267 == 1;	// L387
            if (v268) {	// L388
              ac_int<26, false> v269 = hd[0];	// L389
              o_crv = v269;	// L390
              crv_in = 0;	// L391
            }
          }
        }
      }
      ac_int<26, false> o_out[4];	// L396
      for (int v271 = 0; v271 < 4; v271++) {	// L397
        o_out[v271] = 0;	// L397
      }
      int32_t pop[4];	// L398
      for (int v273 = 0; v273 < 4; v273++) {	// L399
        pop[v273] = 0;	// L399
      }
      int32_t inj_done;	// L400
      inj_done = 0;	// L401
      int32_t idir;	// L402
      idir = -1;	// L403
      ac_int<26, true> v276 = csd_pkt;	// L404
      bool v277;
      ac_int<26, true> _bs_v277 = v276;
      v277 = _bs_v277[25];	// L405
      int32_t v278 = v277;	// L406
      bool v279 = v278 == 1;	// L407
      if (v279) {	// L408
        int32_t v280 = csd_dir;	// L409
        ac_int<33, true> v281 = v280;	// L410
        ac_int<33, true> v282 = 3 - v281;	// L411
        int32_t v283 = v282;	// L412
        idir = v283;	// L413
      }
      l_S_o_3_o: for (int o = 0; o < 4; o++) {	// L415
        uint8_t v285 = rcred[o];	// L416
        int32_t v286 = v285;	// L417
        bool v287 = v286 > 0;	// L418
        if (v287) {	// L419
          int32_t v288 = idir;	// L420
          ac_int<33, true> v289 = v288;	// L421
          ac_int<33, true> v290 = o;	// L422
          bool v291 = v289 == v290;	// L423
          if (v291) {	// L424
            ac_int<26, true> v292 = csd_pkt;	// L425
            o_out[o] = v292;	// L426
            uint8_t v293 = rcred[o];	// L427
            ac_int<33, true> v294 = v293;	// L428
            ac_int<33, true> v295 = v294 - 1;	// L429
            uint8_t v296 = v295;	// L430
            rcred[o] = v296;	// L431
            inj_done = 1;	// L432
          } else {
            int32_t v297 = hvld[o];	// L434
            bool v298 = v297 == 1;	// L435
            int32_t v299 = hit[o];	// L436
            bool v300 = v299 == 0;	// L437
            bool v301 = v298 & v300;	// L438
            if (v301) {	// L439
              ac_int<26, false> v302 = hd[o];	// L440
              o_out[o] = v302;	// L441
              uint8_t v303 = rcred[o];	// L442
              ac_int<33, true> v304 = v303;	// L443
              ac_int<33, true> v305 = v304 - 1;	// L444
              uint8_t v306 = v305;	// L445
              rcred[o] = v306;	// L446
              pop[o] = 1;	// L447
            }
          }
        }
      }
      int32_t v307 = crv_in;	// L452
      bool v308 = v307 >= 0;	// L453
      if (v308) {	// L454
        int32_t v309 = crv_in;	// L455
        int v310 = v309;	// L456
        pop[v310] = 1;	// L457
      }
      int32_t ret[4];	// L459
      for (int v312 = 0; v312 < 4; v312++) {	// L460
        ret[v312] = 0;	// L460
      }
      l_S_d_4_d2: for (int d2 = 0; d2 < 4; d2++) {	// L461
        int32_t v314 = pop[d2];	// L462
        bool v315 = v314 == 1;	// L463
        if (v315) {	// L464
          l_S_sft_4_sft: for (int sft = 0; sft < 1; sft++) {	// L465
            ac_int<26, false> v317 = rbuf[d2][(sft + 1)];	// L466
            rbuf[d2][sft] = v317;	// L467
          }
          uint8_t v318 = rbcnt[d2];	// L469
          ac_int<33, true> v319 = v318;	// L470
          ac_int<33, true> v320 = v319 - 1;	// L471
          uint8_t v321 = v320;	// L472
          rbcnt[d2] = v321;	// L473
          ret[d2] = 1;	// L474
        }
      }
      int32_t v322 = ret[0];	// L477
      uint8_t v323 = v322;	// L478
      cre_r = v323;	// L479
      int32_t v324 = ret[1];	// L480
      uint8_t v325 = v324;	// L481
      crw_r = v325;	// L482
      int32_t v326 = ret[2];	// L483
      uint8_t v327 = v326;	// L484
      crs_r = v327;	// L485
      int32_t v328 = ret[3];	// L486
      uint8_t v329 = v328;	// L487
      crn_r = v329;	// L488
      ac_int<26, false> v330 = o_out[0];	// L489
      oe_r = v330;	// L490
      ac_int<26, false> v331 = o_out[1];	// L491
      ow_r = v331;	// L492
      ac_int<26, false> v332 = o_out[2];	// L493
      os_r = v332;	// L494
      ac_int<26, false> v333 = o_out[3];	// L495
      on_r = v333;	// L496
      int32_t v334 = inj_done;	// L497
      bool v335 = v334 == 1;	// L498
      if (v335) {	// L499
        csd_pkt = 0;	// L500
      }
      ac_int<26, true> v336 = o_crv;	// L502
      bool v337;
      ac_int<26, true> _bs_v337 = v336;
      v337 = _bs_v337[25];	// L503
      int32_t v338 = v337;	// L504
      crv_vld = v338;	// L505
      ac_int<26, true> v339 = o_crv;	// L506
      int16_t v340;
      ac_int<26, true> _bs_v340 = v339;
      v340 = _bs_v340.slc<16>(0);	// L507
      half v341; v341.set_data(ac_int<16, true>(v340));	// L508
      crv_data = v341;	// L509
      ac_int<26, true> v342 = o_crv;	// L510
      ac_int<4, true> v343;
      ac_int<26, true> _bs_v343 = v342;
      v343 = _bs_v343.slc<4>(16);	// L511
      int32_t v344 = v343;	// L512
      crv_addr = v344;	// L513
      ac_int<26, true> v345 = o_crv;	// L514
      bool v346;
      ac_int<26, true> _bs_v346 = v345;
      v346 = _bs_v346[20];	// L515
      int32_t v347 = v346;	// L516
      crv_mode = v347;	// L517
      ac_int<26, true> v348 = o_crv;	// L518
      int16_t v349;
      ac_int<26, true> _bs_v349 = v348;
      v349 = _bs_v349.slc<16>(0);	// L519
      int32_t v350 = v349;	// L520
      crv_raw = v350;	// L521
      ac_int<17, false> v351 = v29.Pop();	// L522
      ac_int<17, false> rx_w;	// L523
      rx_w = v351;	// L524
      ac_int<17, false> v353 = v30.Pop();	// L525
      ac_int<17, false> rx_e;	// L526
      rx_e = v353;	// L527
      ac_int<17, false> v355 = v31.Pop();	// L528
      ac_int<17, false> rx_n;	// L529
      rx_n = v355;	// L530
      ac_int<17, false> v357 = v32.Pop();	// L531
      ac_int<17, false> rx_s;	// L532
      rx_s = v357;	// L533
      half rxv[4];	// L534
      for (int v360 = 0; v360 < 4; v360++) {	// L535
        rxv[v360] = half(0.000000f);	// L535
      }
      int32_t rxvld[4];	// L536
      for (int v362 = 0; v362 < 4; v362++) {	// L537
        rxvld[v362] = 0;	// L537
      }
      ac_int<17, true> v363 = rx_n;	// L538
      int16_t v364;
      ac_int<17, true> _bs_v364 = v363;
      v364 = _bs_v364.slc<16>(1);	// L539
      half v365; v365.set_data(ac_int<16, true>(v364));	// L540
      rxv[0] = v365;	// L541
      ac_int<17, true> v366 = rx_n;	// L542
      bool v367;
      ac_int<17, true> _bs_v367 = v366;
      v367 = _bs_v367[0];	// L543
      int32_t v368 = v367;	// L544
      rxvld[0] = v368;	// L545
      ac_int<17, true> v369 = rx_s;	// L546
      int16_t v370;
      ac_int<17, true> _bs_v370 = v369;
      v370 = _bs_v370.slc<16>(1);	// L547
      half v371; v371.set_data(ac_int<16, true>(v370));	// L548
      rxv[1] = v371;	// L549
      ac_int<17, true> v372 = rx_s;	// L550
      bool v373;
      ac_int<17, true> _bs_v373 = v372;
      v373 = _bs_v373[0];	// L551
      int32_t v374 = v373;	// L552
      rxvld[1] = v374;	// L553
      ac_int<17, true> v375 = rx_w;	// L554
      int16_t v376;
      ac_int<17, true> _bs_v376 = v375;
      v376 = _bs_v376.slc<16>(1);	// L555
      half v377; v377.set_data(ac_int<16, true>(v376));	// L556
      rxv[2] = v377;	// L557
      ac_int<17, true> v378 = rx_w;	// L558
      bool v379;
      ac_int<17, true> _bs_v379 = v378;
      v379 = _bs_v379[0];	// L559
      int32_t v380 = v379;	// L560
      rxvld[2] = v380;	// L561
      ac_int<17, true> v381 = rx_e;	// L562
      int16_t v382;
      ac_int<17, true> _bs_v382 = v381;
      v382 = _bs_v382.slc<16>(1);	// L563
      half v383; v383.set_data(ac_int<16, true>(v382));	// L564
      rxv[3] = v383;	// L565
      ac_int<17, true> v384 = rx_e;	// L566
      bool v385;
      ac_int<17, true> _bs_v385 = v384;
      v385 = _bs_v385[0];	// L567
      int32_t v386 = v385;	// L568
      rxvld[3] = v386;	// L569
      l_S_d_6_d3: for (int d3 = 0; d3 < 4; d3++) {	// L570
        int32_t v388 = rxvld[d3];	// L571
        bool v389 = v388 == 1;	// L572
        uint8_t v390 = hold_cnt[d3];	// L573
        int32_t v391 = v390;	// L574
        bool v392 = v391 < 2;	// L575
        bool v393 = v389 & v392;	// L576
        if (v393) {	// L577
          half v394 = rxv[d3];	// L578
          uint8_t v395 = hold_cnt[d3];	// L579
          int v396 = v395;	// L580
          hold_v[d3][v396] = v394;	// L581
          uint8_t v397 = hold_cnt[d3];	// L582
          ac_int<33, true> v398 = v397;	// L583
          ac_int<33, true> v399 = v398 + 1;	// L584
          uint8_t v400 = v399;	// L585
          hold_cnt[d3] = v400;	// L586
        }
      }
      int32_t retire_ok;	// L589
      retire_ok = 1;	// L590
      uint8_t v402 = sb_v[0];	// L591
      int32_t v403 = v402;	// L592
      bool v404 = v403 == 1;	// L593
      uint8_t v405 = sb_rtr[0];	// L594
      int32_t v406 = v405;	// L595
      bool v407 = v406 == 0;	// L596
      uint8_t v408 = sb_dst[0];	// L597
      int32_t v409 = v408;	// L598
      bool v410 = v409 >= 12;	// L599
      bool v411 = v404 & v407;	// L600
      bool v412 = v411 & v410;	// L601
      if (v412) {	// L602
        uint8_t v413 = sb_rvld[0];	// L603
        int32_t v414 = v413;	// L604
        bool v415 = v414 == 1;	// L605
        uint8_t v416 = sb_dst[0];	// L606
        int32_t v417 = v416;	// L607
        int32_t v418 = v417 & 3;	// L608
        int v419 = v418;	// L609
        int32_t v420 = txp_v[v419];	// L610
        bool v421 = v420 == 1;	// L611
        bool v422 = v415 & v421;	// L612
        if (v422) {	// L613
          retire_ok = 0;	// L614
        }
      }
      uint8_t v423 = sb_v[0];	// L617
      int32_t v424 = v423;	// L618
      bool v425 = v424 == 1;	// L619
      int32_t v426 = retire_ok;	// L620
      bool v427 = v426 == 1;	// L621
      bool v428 = v425 & v427;	// L622
      if (v428) {	// L623
        uint8_t v429 = sb_ix[0];	// L624
        int v430 = v429;	// L625
        half v431 = resq[v430];	// L626
        half wb;	// L627
        wb = v431;	// L628
        uint8_t v433 = sb_cmp[0];	// L629
        int32_t v434 = v433;	// L630
        bool v435 = v434 == 1;	// L631
        if (v435) {	// L632
          uint8_t v436 = sb_ix[0];	// L633
          int v437 = v436;	// L634
          uint8_t v438 = cmpq[v437];	// L635
          condition_reg = v438;	// L636
        }
        uint8_t v439 = sb_rtr[0];	// L638
        int32_t v440 = v439;	// L639
        bool v441 = v440 == 1;	// L640
        if (v441) {	// L641
          uint8_t v442 = sb_inj[0];	// L642
          int32_t v443 = v442;	// L643
          bool v444 = v443 == 1;	// L644
          ac_int<26, true> v445 = csd_pkt;	// L645
          bool v446;
          ac_int<26, true> _bs_v446 = v445;
          v446 = _bs_v446[25];	// L646
          int32_t v447 = v446;	// L647
          bool v448 = v447 == 0;	// L648
          bool v449 = v444 & v448;	// L649
          if (v449) {	// L650
            half v450 = wb;	// L651
            uint16_t v451 = (uint16_t)_fbits(v450);	// L652
            ac_int<26, true> v452 = csd_pkt;	// L653
            ac_int<26, true> v453;
            ac_int<26, true> _bs_v453 = v452;
            _bs_v453.set_slc(0, ac_int<16, false>(v451));
            v453 = _bs_v453;	// L654
            csd_pkt = v453;	// L655
            uint8_t v454 = sb_dst[0];	// L656
            ac_int<4, false> v455 = v454;	// L657
            ac_int<26, true> v456 = csd_pkt;	// L658
            ac_int<26, true> v457;
            ac_int<26, true> _bs_v457 = v456;
            _bs_v457.set_slc(16, ac_int<4, false>(v455));
            v457 = _bs_v457;	// L659
            csd_pkt = v457;	// L660
            uint8_t v458 = sb_id[0];	// L661
            ac_int<4, false> v459 = v458;	// L662
            ac_int<26, true> v460 = csd_pkt;	// L663
            ac_int<26, true> v461;
            ac_int<26, true> _bs_v461 = v460;
            _bs_v461.set_slc(21, ac_int<4, false>(v459));
            v461 = _bs_v461;	// L664
            csd_pkt = v461;	// L665
            uint8_t v462 = sb_rvld[0];	// L666
            bool v463 = v462;	// L667
            ac_int<26, true> v464 = csd_pkt;	// L668
            ac_int<26, true> v465;
            ac_int<26, true> _bs_v465 = v464;
            _bs_v465[25] = v463;
            v465 = _bs_v465;	// L669
            csd_pkt = v465;	// L670
            uint8_t v466 = sb_dir[0];	// L671
            int32_t v467 = v466;	// L672
            csd_dir = v467;	// L673
          }
        } else {
          uint8_t v468 = sb_dst[0];	// L676
          int32_t v469 = v468;	// L677
          bool v470 = v469 >= 12;	// L678
          if (v470) {	// L679
            uint8_t v471 = sb_rvld[0];	// L680
            int32_t v472 = v471;	// L681
            bool v473 = v472 == 1;	// L682
            if (v473) {	// L683
              uint8_t v474 = sb_dst[0];	// L684
              int32_t v475 = v474;	// L685
              int32_t v476 = v475 & 3;	// L686
              int v477 = v476;	// L687
              txp_v[v477] = 1;	// L688
              half v478 = wb;	// L689
              uint8_t v479 = sb_dst[0];	// L690
              int32_t v480 = v479;	// L691
              int32_t v481 = v480 & 3;	// L692
              int v482 = v481;	// L693
              txp_d[v482] = v478;	// L694
              uint8_t v483 = sb_dst[0];	// L695
              int32_t v484 = v483;	// L696
              int32_t v485 = v484 & 3;	// L697
              int v486 = v485;	// L698
              txp_r[v486] = 1;	// L699
            }
          } else {
            uint8_t v487 = sb_rvld[0];	// L702
            int32_t v488 = v487;	// L703
            bool v489 = v488 == 1;	// L704
            if (v489) {	// L705
              uint8_t v490 = sb_dst[0];	// L706
              int32_t v491 = v490;	// L707
              bool v492 = v491 < 8;	// L708
              int32_t v493 = dsmask;	// L709
              int32_t v494 = v493 >> v491;	// L710
              int32_t v495 = v494 & 1;	// L711
              bool v496 = v495 == 1;	// L712
              bool v497 = v492 & v496;	// L713
              if (v497) {	// L714
                uint8_t v498 = sb_dst[0];	// L715
                int v499 = v498;	// L716
                int32_t v500 = drf_full[v499];	// L717
                bool v501 = v500 == 0;	// L718
                if (v501) {	// L719
                  half v502 = wb;	// L720
                  uint8_t v503 = sb_dst[0];	// L721
                  int v504 = v503;	// L722
                  drf[v504] = v502;	// L723
                  uint8_t v505 = sb_dst[0];	// L724
                  int v506 = v505;	// L725
                  drf_full[v506] = 1;	// L726
                }
              } else {
                half v507 = wb;	// L729
                uint8_t v508 = sb_dst[0];	// L730
                int32_t v509 = v508;	// L731
                int32_t v510 = v509 & 7;	// L732
                int v511 = v510;	// L733
                drf[v511] = v507;	// L734
              }
            }
          }
        }
      }
      int32_t pc;	// L740
      pc = -1;	// L741
      int8_t v513 = fetch_en;	// L742
      int32_t v514 = v513;	// L743
      bool v515 = v514 == 1;	// L744
      if (v515) {	// L745
        int8_t v516 = instr_cnt;	// L746
        int32_t v517 = v516;	// L747
        pc = v517;	// L748
      }
      int32_t instr;	// L750
      instr = 0;	// L751
      int32_t v519 = pc;	// L752
      bool v520 = v519 >= 0;	// L753
      if (v520) {	// L754
        int32_t v521 = pc;	// L755
        int v522 = v521;	// L756
        int32_t v523 = irf[v522];	// L757
        instr = v523;	// L758
      }
      int32_t v524 = instr;	// L760
      int32_t v525 = v524 & 15;	// L761
      int32_t op;	// L762
      op = v525;	// L763
      int32_t v527 = instr;	// L764
      int32_t v528 = v527 >> 4;	// L765
      int32_t v529 = v528 & 15;	// L766
      int32_t dst;	// L767
      dst = v529;	// L768
      int32_t v531 = instr;	// L769
      int32_t v532 = v531 >> 8;	// L770
      int32_t v533 = v532 & 15;	// L771
      int32_t s1;	// L772
      s1 = v533;	// L773
      int32_t v535 = instr;	// L774
      int32_t v536 = v535 >> 12;	// L775
      int32_t v537 = v536 & 15;	// L776
      int32_t s2;	// L777
      s2 = v537;	// L778
      half a;	// L779
      a = half(0.000000f);	// L780
      half b;	// L781
      b = half(0.000000f);	// L782
      int32_t v541 = s1;	// L783
      bool v542 = v541 >= 12;	// L784
      if (v542) {	// L785
        int32_t v543 = s1;	// L786
        int32_t v544 = v543 & 3;	// L787
        int v545 = v544;	// L788
        half v546 = hold_v[v545][0];	// L789
        a = v546;	// L790
      } else {
        int32_t v547 = s1;	// L792
        int v548 = v547;	// L793
        half v549 = drf[v548];	// L794
        a = v549;	// L795
      }
      int32_t v550 = s2;	// L797
      bool v551 = v550 >= 12;	// L798
      if (v551) {	// L799
        int32_t v552 = s2;	// L800
        int32_t v553 = v552 & 3;	// L801
        int v554 = v553;	// L802
        half v555 = hold_v[v554][0];	// L803
        b = v555;	// L804
      } else {
        int32_t v556 = s2;	// L806
        int v557 = v556;	// L807
        half v558 = drf[v557];	// L808
        b = v558;	// L809
      }
      int32_t a_vld;	// L811
      a_vld = 1;	// L812
      int32_t b_vld;	// L813
      b_vld = 1;	// L814
      int32_t v561 = s1;	// L815
      bool v562 = v561 >= 12;	// L816
      if (v562) {	// L817
        a_vld = 0;	// L818
        int32_t v563 = s1;	// L819
        int32_t v564 = v563 & 3;	// L820
        int v565 = v564;	// L821
        uint8_t v566 = hold_cnt[v565];	// L822
        int32_t v567 = v566;	// L823
        bool v568 = v567 > 0;	// L824
        if (v568) {	// L825
          a_vld = 1;	// L826
        }
      }
      int32_t v569 = s2;	// L829
      bool v570 = v569 >= 12;	// L830
      if (v570) {	// L831
        b_vld = 0;	// L832
        int32_t v571 = s2;	// L833
        int32_t v572 = v571 & 3;	// L834
        int v573 = v572;	// L835
        uint8_t v574 = hold_cnt[v573];	// L836
        int32_t v575 = v574;	// L837
        bool v576 = v575 > 0;	// L838
        if (v576) {	// L839
          b_vld = 1;	// L840
        }
      }
      int32_t v577 = s1;	// L843
      bool v578 = v577 < 8;	// L844
      int32_t v579 = dsmask;	// L845
      int32_t v580 = v579 >> v577;	// L846
      int32_t v581 = v580 & 1;	// L847
      bool v582 = v581 == 1;	// L848
      bool v583 = v578 & v582;	// L849
      if (v583) {	// L850
        int32_t v584 = s1;	// L851
        int v585 = v584;	// L852
        int32_t v586 = drf_full[v585];	// L853
        bool v587 = v586 == 0;	// L854
        if (v587) {	// L855
          a_vld = 0;	// L856
        }
      }
      int32_t v588 = s2;	// L859
      bool v589 = v588 < 8;	// L860
      int32_t v590 = dsmask;	// L861
      int32_t v591 = v590 >> v588;	// L862
      int32_t v592 = v591 & 1;	// L863
      bool v593 = v592 == 1;	// L864
      bool v594 = v589 & v593;	// L865
      if (v594) {	// L866
        int32_t v595 = s2;	// L867
        int v596 = v595;	// L868
        int32_t v597 = drf_full[v596];	// L869
        bool v598 = v597 == 0;	// L870
        if (v598) {	// L871
          b_vld = 0;	// L872
        }
      }
      int32_t binop;	// L875
      binop = 0;	// L876
      int32_t v600 = op;	// L877
      bool v601 = v600 == 0;	// L878
      bool v602 = v600 == 1;	// L879
      bool v603 = v600 == 2;	// L880
      bool v604 = v600 == 8;	// L881
      bool v605 = v600 == 9;	// L882
      bool v606 = v601 | v602;	// L883
      bool v607 = v606 | v603;	// L884
      bool v608 = v607 | v604;	// L885
      bool v609 = v608 | v605;	// L886
      if (v609) {	// L887
        binop = 1;	// L888
      }
      int32_t raw;	// L890
      raw = 0;	// L891
      int32_t cmp_busy;	// L892
      cmp_busy = 0;	// L893
      int32_t fwd_a;	// L894
      fwd_a = 0;	// L895
      int32_t fwd_a_ix;	// L896
      fwd_a_ix = 0;	// L897
      int32_t raw_a;	// L898
      raw_a = 0;	// L899
      int32_t fwd_b;	// L900
      fwd_b = 0;	// L901
      int32_t fwd_b_ix;	// L902
      fwd_b_ix = 0;	// L903
      int32_t raw_b;	// L904
      raw_b = 0;	// L905
      l_S_k_7_k: for (int k = 0; k < 4; k++) {	// L906
        ac_int<34, true> v619 = k;	// L907
        ac_int<34, true> v620 = v619 + 1;	// L908
        int32_t v621 = v620;	// L909
        int32_t kk;	// L910
        kk = v621;	// L911
        int32_t v623 = kk;	// L912
        ac_int<34, true> v624 = v623;	// L913
        ac_int<34, true> v625 = 4 - v624;	// L914
        int32_t v626 = v625;	// L915
        int32_t inflight;	// L916
        inflight = v626;	// L917
        int32_t need;	// L918
        need = 0;	// L919
        int32_t v629 = kk;	// L920
        int v630 = v629;	// L921
        uint8_t v631 = sb_long[v630];	// L922
        int32_t v632 = v631;	// L923
        bool v633 = v632 == 1;	// L924
        if (v633) {	// L925
          need = 1;	// L926
        }
        int32_t rdy;	// L928
        rdy = 0;	// L929
        int32_t v635 = inflight;	// L930
        int32_t v636 = need;	// L931
        bool v637 = v635 >= v636;	// L932
        if (v637) {	// L933
          rdy = 1;	// L934
        }
        int32_t v638 = kk;	// L936
        int v639 = v638;	// L937
        uint8_t v640 = sb_v[v639];	// L938
        int32_t v641 = v640;	// L939
        bool v642 = v641 == 1;	// L940
        uint8_t v643 = sb_rtr[v639];	// L941
        int32_t v644 = v643;	// L942
        bool v645 = v644 == 0;	// L943
        uint8_t v646 = sb_dst[v639];	// L944
        int32_t v647 = v646;	// L945
        bool v648 = v647 < 12;	// L946
        bool v649 = v642 & v645;	// L947
        bool v650 = v649 & v648;	// L948
        if (v650) {	// L949
          int32_t v651 = s1;	// L950
          bool v652 = v651 < 12;	// L951
          int32_t v653 = kk;	// L952
          int v654 = v653;	// L953
          uint8_t v655 = sb_dst[v654];	// L954
          int32_t v656 = v655;	// L955
          int32_t v657 = v656 & 7;	// L956
          int32_t v658 = v651 & 7;	// L957
          bool v659 = v657 == v658;	// L958
          bool v660 = v652 & v659;	// L959
          if (v660) {	// L960
            int32_t v661 = rdy;	// L961
            bool v662 = v661 == 1;	// L962
            if (v662) {	// L963
              fwd_a = 1;	// L964
              int32_t v663 = kk;	// L965
              int v664 = v663;	// L966
              uint8_t v665 = sb_ix[v664];	// L967
              int32_t v666 = v665;	// L968
              fwd_a_ix = v666;	// L969
              raw_a = 0;	// L970
            } else {
              fwd_a = 0;	// L972
              raw_a = 1;	// L973
            }
          }
          int32_t v667 = binop;	// L976
          bool v668 = v667 == 1;	// L977
          int32_t v669 = s2;	// L978
          bool v670 = v669 < 12;	// L979
          int32_t v671 = kk;	// L980
          int v672 = v671;	// L981
          uint8_t v673 = sb_dst[v672];	// L982
          int32_t v674 = v673;	// L983
          int32_t v675 = v674 & 7;	// L984
          int32_t v676 = v669 & 7;	// L985
          bool v677 = v675 == v676;	// L986
          bool v678 = v668 & v670;	// L987
          bool v679 = v678 & v677;	// L988
          if (v679) {	// L989
            int32_t v680 = rdy;	// L990
            bool v681 = v680 == 1;	// L991
            if (v681) {	// L992
              fwd_b = 1;	// L993
              int32_t v682 = kk;	// L994
              int v683 = v682;	// L995
              uint8_t v684 = sb_ix[v683];	// L996
              int32_t v685 = v684;	// L997
              fwd_b_ix = v685;	// L998
              raw_b = 0;	// L999
            } else {
              fwd_b = 0;	// L1001
              raw_b = 1;	// L1002
            }
          }
        }
        int32_t v686 = kk;	// L1006
        int v687 = v686;	// L1007
        uint8_t v688 = sb_v[v687];	// L1008
        int32_t v689 = v688;	// L1009
        bool v690 = v689 == 1;	// L1010
        uint8_t v691 = sb_cmp[v687];	// L1011
        int32_t v692 = v691;	// L1012
        bool v693 = v692 == 1;	// L1013
        bool v694 = v690 & v693;	// L1014
        if (v694) {	// L1015
          cmp_busy = 1;	// L1016
        }
      }
      int32_t v695 = raw_a;	// L1019
      raw = v695;	// L1020
      int32_t v696 = binop;	// L1021
      bool v697 = v696 == 1;	// L1022
      int32_t v698 = raw_b;	// L1023
      bool v699 = v698 == 1;	// L1024
      bool v700 = v697 & v699;	// L1025
      if (v700) {	// L1026
        raw = 1;	// L1027
      }
      int32_t v701 = fwd_a;	// L1029
      bool v702 = v701 == 1;	// L1030
      if (v702) {	// L1031
        int32_t v703 = fwd_a_ix;	// L1032
        int v704 = v703;	// L1033
        half v705 = resq[v704];	// L1034
        a = v705;	// L1035
        a_vld = 1;	// L1036
      }
      int32_t v706 = fwd_b;	// L1038
      bool v707 = v706 == 1;	// L1039
      if (v707) {	// L1040
        int32_t v708 = fwd_b_ix;	// L1041
        int v709 = v708;	// L1042
        half v710 = resq[v709];	// L1043
        b = v710;	// L1044
        b_vld = 1;	// L1045
      }
      int32_t is_cond;	// L1047
      is_cond = 0;	// L1048
      int32_t v712 = op;	// L1049
      bool v713 = v712 >= 12;	// L1050
      ac_int<33, true> v714 = v712;	// L1051
      bool v715 = v714 <= 15;	// L1052
      bool v716 = v713 & v715;	// L1053
      if (v716) {	// L1054
        is_cond = 1;	// L1055
      }
      int32_t grant;	// L1057
      grant = 0;	// L1058
      int32_t v718 = pc;	// L1059
      bool v719 = v718 >= 0;	// L1060
      if (v719) {	// L1061
        grant = 1;	// L1062
      }
      int32_t v720 = pc;	// L1064
      bool v721 = v720 >= 0;	// L1065
      int32_t v722 = a_vld;	// L1066
      bool v723 = v722 == 0;	// L1067
      int32_t v724 = binop;	// L1068
      bool v725 = v724 == 1;	// L1069
      int32_t v726 = b_vld;	// L1070
      bool v727 = v726 == 0;	// L1071
      bool v728 = v725 & v727;	// L1072
      bool v729 = v723 | v728;	// L1073
      bool v730 = v721 & v729;	// L1074
      if (v730) {	// L1075
        grant = 0;	// L1076
      }
      int32_t v731 = pc;	// L1078
      bool v732 = v731 >= 0;	// L1079
      int32_t v733 = raw;	// L1080
      bool v734 = v733 == 1;	// L1081
      int32_t v735 = is_cond;	// L1082
      bool v736 = v735 == 1;	// L1083
      int32_t v737 = cmp_busy;	// L1084
      bool v738 = v737 == 1;	// L1085
      bool v739 = v736 & v738;	// L1086
      bool v740 = v734 | v739;	// L1087
      bool v741 = v732 & v740;	// L1088
      if (v741) {	// L1089
        grant = 0;	// L1090
      }
      int32_t v742 = retire_ok;	// L1092
      bool v743 = v742 == 0;	// L1093
      if (v743) {	// L1094
        grant = 0;	// L1095
      }
      int32_t v744 = grant;	// L1097
      bool v745 = v744 == 1;	// L1098
      if (v745) {	// L1099
        int8_t v746 = instr_cnt;	// L1100
        int32_t v747 = cfg_isz;	// L1101
        int32_t v748 = v746;	// L1102
        bool v749 = v748 == v747;	// L1103
        if (v749) {	// L1104
          instr_cnt = 0;	// L1105
          int8_t v750 = iter_cnt;	// L1106
          int32_t v751 = cfg_itsz;	// L1107
          ac_int<33, true> v752 = v751;	// L1108
          ac_int<33, true> v753 = v752 - 1;	// L1109
          ac_int<33, true> v754 = v750;	// L1110
          bool v755 = v754 == v753;	// L1111
          if (v755) {	// L1112
            fetch_en = 0;	// L1113
          } else {
            int8_t v756 = iter_cnt;	// L1115
            ac_int<33, true> v757 = v756;	// L1116
            ac_int<33, true> v758 = v757 + 1;	// L1117
            uint8_t v759 = v758;	// L1118
            iter_cnt = v759;	// L1119
          }
        } else {
          int8_t v760 = instr_cnt;	// L1122
          ac_int<33, true> v761 = v760;	// L1123
          ac_int<33, true> v762 = v761 + 1;	// L1124
          uint8_t v763 = v762;	// L1125
          instr_cnt = v763;	// L1126
        }
      }
      int32_t c1;	// L1129
      c1 = -1;	// L1130
      int32_t c2;	// L1131
      c2 = -1;	// L1132
      int32_t v766 = grant;	// L1133
      bool v767 = v766 == 1;	// L1134
      int32_t v768 = s1;	// L1135
      bool v769 = v768 >= 12;	// L1136
      bool v770 = v767 & v769;	// L1137
      if (v770) {	// L1138
        int32_t v771 = s1;	// L1139
        int32_t v772 = v771 & 3;	// L1140
        c1 = v772;	// L1141
      }
      int32_t v773 = grant;	// L1143
      bool v774 = v773 == 1;	// L1144
      int32_t v775 = s2;	// L1145
      bool v776 = v775 >= 12;	// L1146
      bool v777 = v774 & v776;	// L1147
      if (v777) {	// L1148
        int32_t v778 = s2;	// L1149
        int32_t v779 = v778 & 3;	// L1150
        c2 = v779;	// L1151
      }
      int32_t v780 = c1;	// L1153
      bool v781 = v780 >= 0;	// L1154
      if (v781) {	// L1155
        int32_t v782 = c1;	// L1156
        int v783 = v782;	// L1157
        half v784 = hold_v[v783][1];	// L1158
        hold_v[v783][0] = v784;	// L1159
        int32_t v785 = c1;	// L1160
        int v786 = v785;	// L1161
        uint8_t v787 = hold_cnt[v786];	// L1162
        ac_int<33, true> v788 = v787;	// L1163
        ac_int<33, true> v789 = v788 - 1;	// L1164
        uint8_t v790 = v789;	// L1165
        hold_cnt[v786] = v790;	// L1166
      }
      int32_t v791 = c2;	// L1168
      bool v792 = v791 >= 0;	// L1169
      int32_t v793 = c1;	// L1170
      bool v794 = v791 != v793;	// L1171
      bool v795 = v792 & v794;	// L1172
      if (v795) {	// L1173
        int32_t v796 = c2;	// L1174
        int v797 = v796;	// L1175
        half v798 = hold_v[v797][1];	// L1176
        hold_v[v797][0] = v798;	// L1177
        int32_t v799 = c2;	// L1178
        int v800 = v799;	// L1179
        uint8_t v801 = hold_cnt[v800];	// L1180
        ac_int<33, true> v802 = v801;	// L1181
        ac_int<33, true> v803 = v802 - 1;	// L1182
        uint8_t v804 = v803;	// L1183
        hold_cnt[v800] = v804;	// L1184
      }
      l_S_d_8_d4: for (int d4 = 0; d4 < 4; d4++) {	// L1186
        sc_r[d4] = 0;	// L1187
      }
      int32_t v806 = c1;	// L1189
      bool v807 = v806 >= 0;	// L1190
      if (v807) {	// L1191
        int32_t v808 = c1;	// L1192
        int v809 = v808;	// L1193
        sc_r[v809] = 1;	// L1194
      }
      int32_t v810 = c2;	// L1196
      bool v811 = v810 >= 0;	// L1197
      int32_t v812 = c1;	// L1198
      bool v813 = v810 != v812;	// L1199
      bool v814 = v811 & v813;	// L1200
      if (v814) {	// L1201
        int32_t v815 = c2;	// L1202
        int v816 = v815;	// L1203
        sc_r[v816] = 1;	// L1204
      }
      int32_t v817 = grant;	// L1206
      bool v818 = v817 == 1;	// L1207
      int32_t v819 = s1;	// L1208
      bool v820 = v819 < 8;	// L1209
      int32_t v821 = dsmask;	// L1210
      int32_t v822 = v821 >> v819;	// L1211
      int32_t v823 = v822 & 1;	// L1212
      bool v824 = v823 == 1;	// L1213
      bool v825 = v818 & v820;	// L1214
      bool v826 = v825 & v824;	// L1215
      if (v826) {	// L1216
        int32_t v827 = s1;	// L1217
        int v828 = v827;	// L1218
        drf_full[v828] = 0;	// L1219
      }
      int32_t v829 = grant;	// L1221
      bool v830 = v829 == 1;	// L1222
      int32_t v831 = s2;	// L1223
      bool v832 = v831 < 8;	// L1224
      int32_t v833 = dsmask;	// L1225
      int32_t v834 = v833 >> v831;	// L1226
      int32_t v835 = v834 & 1;	// L1227
      bool v836 = v835 == 1;	// L1228
      bool v837 = v830 & v832;	// L1229
      bool v838 = v837 & v836;	// L1230
      if (v838) {	// L1231
        int32_t v839 = s2;	// L1232
        int v840 = v839;	// L1233
        drf_full[v840] = 0;	// L1234
      }
      half res;	// L1236
      res = half(0.000000f);	// L1237
      int32_t v842 = op;	// L1238
      bool v843 = v842 == 0;	// L1239
      if (v843) {	// L1240
        half v844 = a;	// L1241
        half v845 = b;	// L1242
        half v846 = v844 + v845;	// L1243
        res = v846;	// L1244
      } else {
        int32_t v847 = op;	// L1246
        bool v848 = v847 == 1;	// L1247
        if (v848) {	// L1248
          half v849 = a;	// L1249
          half v850 = b;	// L1250
          half v851 = v849 - v850;	// L1251
          res = v851;	// L1252
        } else {
          int32_t v852 = op;	// L1254
          bool v853 = v852 == 2;	// L1255
          if (v853) {	// L1256
            half v854 = a;	// L1257
            half v855 = b;	// L1258
            half v856 = v854 * v855;	// L1259
            res = v856;	// L1260
          } else {
            int32_t v857 = op;	// L1262
            bool v858 = v857 == 8;	// L1263
            if (v858) {	// L1264
              half v859 = a;	// L1265
              half v860 = b;	// L1266
              bool v861 = v859 >= v860;	// L1267
              if (v861) {	// L1268
                res = half(1.000000f);	// L1269
              } else {
                res = half(-1.000000f);	// L1271
              }
            } else {
              int32_t v862 = op;	// L1274
              bool v863 = v862 == 9;	// L1275
              if (v863) {	// L1276
                half v864 = a;	// L1277
                half v865 = b;	// L1278
                bool v866 = v864 < v865;	// L1279
                if (v866) {	// L1280
                  res = half(1.000000f);	// L1281
                } else {
                  res = half(-1.000000f);	// L1283
                }
              } else {
                half v867 = a;	// L1286
                res = v867;	// L1287
              }
            }
          }
        }
      }
      int32_t v868 = a_vld;	// L1293
      int32_t res_vld;	// L1294
      res_vld = v868;	// L1295
      int32_t v870 = op;	// L1296
      bool v871 = v870 == 0;	// L1297
      bool v872 = v870 == 1;	// L1298
      bool v873 = v870 == 2;	// L1299
      bool v874 = v870 == 8;	// L1300
      bool v875 = v870 == 9;	// L1301
      bool v876 = v871 | v872;	// L1302
      bool v877 = v876 | v873;	// L1303
      bool v878 = v877 | v874;	// L1304
      bool v879 = v878 | v875;	// L1305
      if (v879) {	// L1306
        int32_t v880 = a_vld;	// L1307
        int32_t v881 = b_vld;	// L1308
        int64_t v882 = v880;	// L1309
        int64_t v883 = v881;	// L1310
        int64_t v884 = v882 * v883;	// L1311
        int32_t v885 = v884;	// L1312
        res_vld = v885;	// L1313
      }
      int32_t v886 = grant;	// L1315
      bool v887 = v886 == 0;	// L1316
      if (v887) {	// L1317
        res_vld = 0;	// L1318
      }
      int32_t is_rtr;	// L1320
      is_rtr = 0;	// L1321
      int32_t v889 = op;	// L1322
      bool v890 = v889 >= 4;	// L1323
      ac_int<33, true> v891 = v889;	// L1324
      bool v892 = v891 <= 7;	// L1325
      bool v893 = v890 & v892;	// L1326
      if (v893) {	// L1327
        is_rtr = 1;	// L1328
      }
      int32_t v894 = retire_ok;	// L1330
      bool v895 = v894 == 1;	// L1331
      if (v895) {	// L1332
        l_S_k_9_k1: for (int k1 = 0; k1 < 4; k1++) {	// L1333
          uint8_t v897 = sb_v[(k1 + 1)];	// L1334
          sb_v[k1] = v897;	// L1335
          uint8_t v898 = sb_dst[(k1 + 1)];	// L1336
          sb_dst[k1] = v898;	// L1337
          uint8_t v899 = sb_cmp[(k1 + 1)];	// L1338
          sb_cmp[k1] = v899;	// L1339
          uint8_t v900 = sb_rtr[(k1 + 1)];	// L1340
          sb_rtr[k1] = v900;	// L1341
          uint8_t v901 = sb_inj[(k1 + 1)];	// L1342
          sb_inj[k1] = v901;	// L1343
          uint8_t v902 = sb_dir[(k1 + 1)];	// L1344
          sb_dir[k1] = v902;	// L1345
          uint8_t v903 = sb_id[(k1 + 1)];	// L1346
          sb_id[k1] = v903;	// L1347
          uint8_t v904 = sb_rvld[(k1 + 1)];	// L1348
          sb_rvld[k1] = v904;	// L1349
          uint8_t v905 = sb_ix[(k1 + 1)];	// L1350
          sb_ix[k1] = v905;	// L1351
          uint8_t v906 = sb_long[(k1 + 1)];	// L1352
          sb_long[k1] = v906;	// L1353
        }
        sb_v[4] = 0;	// L1355
      }
      int32_t v907 = grant;	// L1357
      bool v908 = v907 == 1;	// L1358
      if (v908) {	// L1359
        half v909 = res;	// L1360
        int8_t v910 = resq_wr;	// L1361
        int v911 = v910;	// L1362
        resq[v911] = v909;	// L1363
        int32_t cq;	// L1364
        cq = 0;	// L1365
        int32_t v913 = op;	// L1366
        bool v914 = v913 == 8;	// L1367
        if (v914) {	// L1368
          half v915 = a;	// L1369
          half v916 = b;	// L1370
          bool v917 = v915 >= v916;	// L1371
          if (v917) {	// L1372
            cq = 1;	// L1373
          }
        }
        int32_t v918 = op;	// L1376
        bool v919 = v918 == 9;	// L1377
        if (v919) {	// L1378
          half v920 = a;	// L1379
          half v921 = b;	// L1380
          bool v922 = v920 < v921;	// L1381
          if (v922) {	// L1382
            cq = 1;	// L1383
          }
        }
        int32_t v923 = cq;	// L1386
        uint8_t v924 = v923;	// L1387
        int8_t v925 = resq_wr;	// L1388
        int v926 = v925;	// L1389
        cmpq[v926] = v924;	// L1390
        sb_v[4] = 1;	// L1391
        int32_t v927 = dst;	// L1392
        uint8_t v928 = v927;	// L1393
        sb_dst[4] = v928;	// L1394
        int8_t v929 = resq_wr;	// L1395
        sb_ix[4] = v929;	// L1396
        int32_t v930 = binop;	// L1397
        uint8_t v931 = v930;	// L1398
        sb_long[4] = v931;	// L1399
        sb_cmp[4] = 0;	// L1400
        int32_t v932 = op;	// L1401
        bool v933 = v932 == 8;	// L1402
        bool v934 = v932 == 9;	// L1403
        bool v935 = v933 | v934;	// L1404
        if (v935) {	// L1405
          sb_cmp[4] = 1;	// L1406
        }
        int32_t v936 = is_rtr;	// L1408
        int32_t rtrf;	// L1409
        rtrf = v936;	// L1410
        int32_t v938 = is_cond;	// L1411
        bool v939 = v938 == 1;	// L1412
        if (v939) {	// L1413
          rtrf = 1;	// L1414
        }
        int32_t v940 = rtrf;	// L1416
        uint8_t v941 = v940;	// L1417
        sb_rtr[4] = v941;	// L1418
        int32_t v942 = is_rtr;	// L1419
        int32_t inj;	// L1420
        inj = v942;	// L1421
        int32_t v944 = is_cond;	// L1422
        bool v945 = v944 == 1;	// L1423
        int8_t v946 = condition_reg;	// L1424
        int32_t v947 = v946;	// L1425
        bool v948 = v947 == 1;	// L1426
        bool v949 = v945 & v948;	// L1427
        if (v949) {	// L1428
          inj = 1;	// L1429
        }
        int32_t v950 = inj;	// L1431
        uint8_t v951 = v950;	// L1432
        sb_inj[4] = v951;	// L1433
        int32_t v952 = op;	// L1434
        int32_t v953 = v952 & 3;	// L1435
        uint8_t v954 = v953;	// L1436
        sb_dir[4] = v954;	// L1437
        int32_t v955 = s2;	// L1438
        uint8_t v956 = v955;	// L1439
        sb_id[4] = v956;	// L1440
        int32_t v957 = res_vld;	// L1441
        uint8_t v958 = v957;	// L1442
        sb_rvld[4] = v958;	// L1443
        int8_t v959 = resq_wr;	// L1444
        ac_int<33, true> v960 = v959;	// L1445
        ac_int<33, true> v961 = v960 + 1;	// L1446
        ac_int<33, true> v962 = v961 & 7;	// L1447
        uint8_t v963 = v962;	// L1448
        resq_wr = v963;	// L1449
      }
      txn_r = 0;	// L1451
      txs_r = 0;	// L1452
      txw_r = 0;	// L1453
      txe_r = 0;	// L1454
      int32_t v964 = txp_v[0];	// L1455
      bool v965 = v964 == 1;	// L1456
      int32_t v966 = scred[0];	// L1457
      bool v967 = v966 > 0;	// L1458
      bool v968 = v965 & v967;	// L1459
      if (v968) {	// L1460
        ac_int<17, false> twn;	// L1461
        twn = 0;	// L1462
        ac_int<17, true> v970 = twn;	// L1463
        ac_int<17, true> v971;
        ac_int<17, true> _bs_v971 = v970;
        _bs_v971[0] = 1;
        v971 = _bs_v971;	// L1464
        twn = v971;	// L1465
        half v972 = txp_d[0];	// L1466
        uint16_t v973 = (uint16_t)_fbits(v972);	// L1467
        ac_int<17, true> v974 = twn;	// L1468
        ac_int<17, true> v975;
        ac_int<17, true> _bs_v975 = v974;
        _bs_v975.set_slc(1, ac_int<16, false>(v973));
        v975 = _bs_v975;	// L1469
        twn = v975;	// L1470
        ac_int<17, true> v976 = twn;	// L1471
        txn_r = v976;	// L1472
        txp_v[0] = 0;	// L1473
        int32_t v977 = scred[0];	// L1474
        ac_int<33, true> v978 = v977;	// L1475
        ac_int<33, true> v979 = v978 - 1;	// L1476
        int32_t v980 = v979;	// L1477
        scred[0] = v980;	// L1478
      }
      int32_t v981 = txp_v[1];	// L1480
      bool v982 = v981 == 1;	// L1481
      int32_t v983 = scred[1];	// L1482
      bool v984 = v983 > 0;	// L1483
      bool v985 = v982 & v984;	// L1484
      if (v985) {	// L1485
        ac_int<17, false> tws;	// L1486
        tws = 0;	// L1487
        ac_int<17, true> v987 = tws;	// L1488
        ac_int<17, true> v988;
        ac_int<17, true> _bs_v988 = v987;
        _bs_v988[0] = 1;
        v988 = _bs_v988;	// L1489
        tws = v988;	// L1490
        half v989 = txp_d[1];	// L1491
        uint16_t v990 = (uint16_t)_fbits(v989);	// L1492
        ac_int<17, true> v991 = tws;	// L1493
        ac_int<17, true> v992;
        ac_int<17, true> _bs_v992 = v991;
        _bs_v992.set_slc(1, ac_int<16, false>(v990));
        v992 = _bs_v992;	// L1494
        tws = v992;	// L1495
        ac_int<17, true> v993 = tws;	// L1496
        txs_r = v993;	// L1497
        txp_v[1] = 0;	// L1498
        int32_t v994 = scred[1];	// L1499
        ac_int<33, true> v995 = v994;	// L1500
        ac_int<33, true> v996 = v995 - 1;	// L1501
        int32_t v997 = v996;	// L1502
        scred[1] = v997;	// L1503
      }
      int32_t v998 = txp_v[2];	// L1505
      bool v999 = v998 == 1;	// L1506
      int32_t v1000 = scred[2];	// L1507
      bool v1001 = v1000 > 0;	// L1508
      bool v1002 = v999 & v1001;	// L1509
      if (v1002) {	// L1510
        ac_int<17, false> tww;	// L1511
        tww = 0;	// L1512
        ac_int<17, true> v1004 = tww;	// L1513
        ac_int<17, true> v1005;
        ac_int<17, true> _bs_v1005 = v1004;
        _bs_v1005[0] = 1;
        v1005 = _bs_v1005;	// L1514
        tww = v1005;	// L1515
        half v1006 = txp_d[2];	// L1516
        uint16_t v1007 = (uint16_t)_fbits(v1006);	// L1517
        ac_int<17, true> v1008 = tww;	// L1518
        ac_int<17, true> v1009;
        ac_int<17, true> _bs_v1009 = v1008;
        _bs_v1009.set_slc(1, ac_int<16, false>(v1007));
        v1009 = _bs_v1009;	// L1519
        tww = v1009;	// L1520
        ac_int<17, true> v1010 = tww;	// L1521
        txw_r = v1010;	// L1522
        txp_v[2] = 0;	// L1523
        int32_t v1011 = scred[2];	// L1524
        ac_int<33, true> v1012 = v1011;	// L1525
        ac_int<33, true> v1013 = v1012 - 1;	// L1526
        int32_t v1014 = v1013;	// L1527
        scred[2] = v1014;	// L1528
      }
      int32_t v1015 = txp_v[3];	// L1530
      bool v1016 = v1015 == 1;	// L1531
      int32_t v1017 = scred[3];	// L1532
      bool v1018 = v1017 > 0;	// L1533
      bool v1019 = v1016 & v1018;	// L1534
      if (v1019) {	// L1535
        ac_int<17, false> twe;	// L1536
        twe = 0;	// L1537
        ac_int<17, true> v1021 = twe;	// L1538
        ac_int<17, true> v1022;
        ac_int<17, true> _bs_v1022 = v1021;
        _bs_v1022[0] = 1;
        v1022 = _bs_v1022;	// L1539
        twe = v1022;	// L1540
        half v1023 = txp_d[3];	// L1541
        uint16_t v1024 = (uint16_t)_fbits(v1023);	// L1542
        ac_int<17, true> v1025 = twe;	// L1543
        ac_int<17, true> v1026;
        ac_int<17, true> _bs_v1026 = v1025;
        _bs_v1026.set_slc(1, ac_int<16, false>(v1024));
        v1026 = _bs_v1026;	// L1544
        twe = v1026;	// L1545
        ac_int<17, true> v1027 = twe;	// L1546
        txe_r = v1027;	// L1547
        txp_v[3] = 0;	// L1548
        int32_t v1028 = scred[3];	// L1549
        ac_int<33, true> v1029 = v1028;	// L1550
        ac_int<33, true> v1030 = v1029 - 1;	// L1551
        int32_t v1031 = v1030;	// L1552
        scred[3] = v1031;	// L1553
      }
      int32_t v1032 = crv_vld;	// L1555
      bool v1033 = v1032 == 1;	// L1556
      if (v1033) {	// L1557
        int32_t v1034 = crv_mode;	// L1558
        bool v1035 = v1034 == 1;	// L1559
        if (v1035) {	// L1560
          int32_t v1036 = crv_addr;	// L1561
          int32_t v1037 = v1036 >> 3;	// L1562
          int32_t v1038 = v1037 & 1;	// L1563
          bool v1039 = v1038 == 1;	// L1564
          if (v1039) {	// L1565
            int32_t v1040 = crv_raw;	// L1566
            int32_t v1041 = crv_addr;	// L1567
            int32_t v1042 = v1041 & 7;	// L1568
            int v1043 = v1042;	// L1569
            irf[v1043] = v1040;	// L1570
          } else {
            int32_t v1044 = crv_addr;	// L1572
            bool v1045 = v1044 == 0;	// L1573
            if (v1045) {	// L1574
              int32_t v1046 = crv_raw;	// L1575
              int32_t v1047 = v1046 & 255;	// L1576
              dsmask = v1047;	// L1577
              int32_t v1048 = crv_raw;	// L1578
              int32_t v1049 = v1048 >> 8;	// L1579
              int32_t v1050 = v1049 & 7;	// L1580
              cfg_isz = v1050;	// L1581
              int32_t v1051 = crv_raw;	// L1582
              int32_t v1052 = v1051 >> 15;	// L1583
              int32_t v1053 = v1052 & 1;	// L1584
              bool v1054 = v1053 == 1;	// L1585
              if (v1054) {	// L1586
                fetch_en = 1;	// L1587
                instr_cnt = 0;	// L1588
                iter_cnt = 0;	// L1589
              }
            } else {
              int32_t v1055 = crv_addr;	// L1592
              bool v1056 = v1055 == 1;	// L1593
              if (v1056) {	// L1594
                int32_t v1057 = crv_raw;	// L1595
                int32_t v1058 = v1057 & 255;	// L1596
                cfg_itsz = v1058;	// L1597
              }
            }
          }
        } else {
          int32_t v1059 = crv_addr;	// L1602
          int32_t v1060 = v1059 >> 2;	// L1603
          int32_t v1061 = v1060 & 3;	// L1604
          bool v1062 = v1061 == 3;	// L1605
          if (v1062) {	// L1606
            int32_t v1063 = crv_addr;	// L1607
            int32_t v1064 = v1063 & 3;	// L1608
            int v1065 = v1064;	// L1609
            txp_v[v1065] = 1;	// L1610
            half v1066 = crv_data;	// L1611
            int32_t v1067 = crv_addr;	// L1612
            int32_t v1068 = v1067 & 3;	// L1613
            int v1069 = v1068;	// L1614
            txp_d[v1069] = v1066;	// L1615
            int32_t v1070 = crv_addr;	// L1616
            int32_t v1071 = v1070 & 3;	// L1617
            int v1072 = v1071;	// L1618
            txp_r[v1072] = 1;	// L1619
          } else {
            int32_t v1073 = crv_addr;	// L1621
            bool v1074 = v1073 < 8;	// L1622
            int32_t v1075 = dsmask;	// L1623
            int32_t v1076 = v1075 >> v1073;	// L1624
            int32_t v1077 = v1076 & 1;	// L1625
            bool v1078 = v1077 == 1;	// L1626
            bool v1079 = v1074 & v1078;	// L1627
            if (v1079) {	// L1628
              int32_t v1080 = crv_addr;	// L1629
              int v1081 = v1080;	// L1630
              int32_t v1082 = drf_full[v1081];	// L1631
              bool v1083 = v1082 == 0;	// L1632
              if (v1083) {	// L1633
                half v1084 = crv_data;	// L1634
                int32_t v1085 = crv_addr;	// L1635
                int v1086 = v1085;	// L1636
                drf[v1086] = v1084;	// L1637
                int32_t v1087 = crv_addr;	// L1638
                int v1088 = v1087;	// L1639
                drf_full[v1088] = 1;	// L1640
              }
            } else {
              half v1089 = crv_data;	// L1643
              int32_t v1090 = crv_addr;	// L1644
              int v1091 = v1090;	// L1645
              drf[v1091] = v1089;	// L1646
            }
          }
        }
      }
      ac_int<26, true> v1092 = oe_r;	// L1651
      v1.Push(v1092);	// L1652
      ac_int<26, true> v1093 = ow_r;	// L1653
      v2.Push(v1093);	// L1654
      ac_int<26, true> v1094 = os_r;	// L1655
      v3.Push(v1094);	// L1656
      ac_int<26, true> v1095 = on_r;	// L1657
      v4.Push(v1095);	// L1658
      ac_int<17, true> v1096 = txe_r;	// L1659
      v5.Push(v1096);	// L1660
      ac_int<17, true> v1097 = txw_r;	// L1661
      v6.Push(v1097);	// L1662
      ac_int<17, true> v1098 = txs_r;	// L1663
      v7.Push(v1098);	// L1664
      ac_int<17, true> v1099 = txn_r;	// L1665
      v8.Push(v1099);	// L1666
      int8_t v1100 = cre_r;	// L1667
      v9.Push(v1100);	// L1668
      int8_t v1101 = crw_r;	// L1669
      v10.Push(v1101);	// L1670
      int8_t v1102 = crs_r;	// L1671
      v11.Push(v1102);	// L1672
      int8_t v1103 = crn_r;	// L1673
      v12.Push(v1103);	// L1674
      int32_t v1104 = sc_r[0];	// L1675
      v15.Push(v1104);	// L1676
      int32_t v1105 = sc_r[1];	// L1677
      v16.Push(v1105);	// L1678
      int32_t v1106 = sc_r[2];	// L1679
      v13.Push(v1106);	// L1680
      int32_t v1107 = sc_r[3];	// L1681
      v14.Push(v1107);	// L1682
#ifndef __SYNTHESIS__
      wait();
#endif
    }
    done.write(true);  // RTL-observable completion (see the done port)
#ifndef __SYNTHESIS__
    __allo_done++; // csim: this kernel finished its single pass
#endif
    while (1) { wait(); }
  }
};

SC_MODULE(drv_w_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_out<bool> done;
  sc_out< ac_int<6, false> > v1108_radr;
  sc_out<bool> v1108_re;
  sc_in< half > v1108_q;
  sc_in<bool> v1108_rrdy;
  sc_out< ac_int<6, false> > v1109_radr;
  sc_out<bool> v1109_re;
  sc_in< ac_int<32, true> > v1109_q;
  sc_in<bool> v1109_rrdy;
  sc_out< ac_int<1, false> > v1110_radr;
  sc_out<bool> v1110_re;
  sc_in< ac_int<32, true> > v1110_q;
  sc_in<bool> v1110_rrdy;
  Connections::Out< ac_int<17, false> > v1111;
  Connections::In< ac_int<32, true> > v1112;
  SC_HAS_PROCESS(drv_w_0);
  drv_w_0(sc_module_name n) : sc_module(n), done("done"), v1111("v1111"), v1112("v1112") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <in>
  half v1108_rd(ac_int<6, false> addr) {
    v1108_radr.write(addr); v1108_re.write(true);
    wait();                    // edge N: address captured
    v1108_re.write(false);
    wait();                    // data valid on this edge
    return v1108_q.read();
  }
  #pragma design modulario <in>
  ac_int<32, true> v1109_rd(ac_int<6, false> addr) {
    v1109_radr.write(addr); v1109_re.write(true);
    wait();                    // edge N: address captured
    v1109_re.write(false);
    wait();                    // data valid on this edge
    return v1109_q.read();
  }
  #pragma design modulario <in>
  ac_int<32, true> v1110_rd(ac_int<1, false> addr) {
    v1110_radr.write(addr); v1110_re.write(true);
    wait();                    // edge N: address captured
    v1110_re.write(false);
    wait();                    // data valid on this edge
    return v1110_q.read();
  }
  void run() {
    v1111.Reset();
    v1112.Reset();
    v1108_radr.write(0);
    v1108_re.write(0);
    v1109_radr.write(0);
    v1109_re.write(0);
    v1110_radr.write(0);
    v1110_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t dcred[1];	// L1695
    for (int v1114 = 0; v1114 < 1; v1114++) {	// L1696
      dcred[v1114] = 0;	// L1696
    }
    int32_t sp[1];	// L1697
    for (int v1116 = 0; v1116 < 1; v1116++) {	// L1698
      sp[v1116] = 0;	// L1698
    }
    ac_int<17, false> zw;	// L1699
    zw = 0;	// L1700
    int32_t v1118;
    v1118 = v1110_rd((ac_int<1, false>)(((0) + (0))));	// L1701
    ac_int<33, true> v1119 = v1118;	// L1702
    ac_int<33, true> v1120 = v1119 - 1;	// L1703
    int v1121 = v1120;	// L1704
    for (int v1122 = 0; v1122 < v1121; v1122 += 1) {	// L1705
      ac_int<17, true> v1123 = zw;	// L1706
      v1111.Push(v1123);	// L1707
    }
    l_S_t_1_t1: for (int t1 = 0; t1 < 55; t1++) {	// L1709
      int32_t v1125 = v1112.Pop();	// L1710
      int32_t v1126 = dcred[0];	// L1711
      ac_int<33, true> v1127 = v1126;	// L1712
      ac_int<33, true> v1128 = v1125;	// L1713
      ac_int<33, true> v1129 = v1127 + v1128;	// L1714
      int32_t v1130 = v1129;	// L1715
      dcred[0] = v1130;	// L1716
      ac_int<17, false> w;	// L1717
      w = 0;	// L1718
      int32_t v1132 = sp[0];	// L1719
      bool v1133 = v1132 < 55;	// L1720
      ac_int<33, true> v1134 = t1;	// L1721
      ac_int<33, true> v1135 = v1132;	// L1722
      bool v1136 = v1134 >= v1135;	// L1723
      bool v1137 = v1133 & v1136;	// L1724
      if (v1137) {	// L1725
        int32_t v1138 = sp[0];	// L1726
        int v1139 = v1138;	// L1727
        int32_t v1140;
        v1140 = v1109_rd((ac_int<6, false>)(((0) * 55 + (v1139))));	// L1728
        bool v1141 = v1140 == 0;	// L1729
        if (v1141) {	// L1730
          int32_t v1142 = sp[0];	// L1731
          ac_int<33, true> v1143 = v1142;	// L1732
          ac_int<33, true> v1144 = v1143 + 1;	// L1733
          int32_t v1145 = v1144;	// L1734
          sp[0] = v1145;	// L1735
        } else {
          int32_t v1146 = dcred[0];	// L1737
          bool v1147 = v1146 > 0;	// L1738
          if (v1147) {	// L1739
            ac_int<17, true> v1148 = w;	// L1740
            ac_int<17, true> v1149;
            ac_int<17, true> _bs_v1149 = v1148;
            _bs_v1149[0] = 1;
            v1149 = _bs_v1149;	// L1741
            w = v1149;	// L1742
            int32_t v1150 = sp[0];	// L1743
            int v1151 = v1150;	// L1744
            half v1152;
            v1152 = v1108_rd((ac_int<6, false>)(((0) * 55 + (v1151))));	// L1745
            uint16_t v1153 = (uint16_t)_fbits(v1152);	// L1746
            ac_int<17, true> v1154 = w;	// L1747
            ac_int<17, true> v1155;
            ac_int<17, true> _bs_v1155 = v1154;
            _bs_v1155.set_slc(1, ac_int<16, false>(v1153));
            v1155 = _bs_v1155;	// L1748
            w = v1155;	// L1749
            int32_t v1156 = dcred[0];	// L1750
            ac_int<33, true> v1157 = v1156;	// L1751
            ac_int<33, true> v1158 = v1157 - 1;	// L1752
            int32_t v1159 = v1158;	// L1753
            dcred[0] = v1159;	// L1754
            int32_t v1160 = sp[0];	// L1755
            ac_int<33, true> v1161 = v1160;	// L1756
            ac_int<33, true> v1162 = v1161 + 1;	// L1757
            int32_t v1163 = v1162;	// L1758
            sp[0] = v1163;	// L1759
          }
        }
      }
      ac_int<17, true> v1164 = w;	// L1763
      v1111.Push(v1164);	// L1764
    }
    done.write(true);  // RTL-observable completion (see the done port)
#ifndef __SYNTHESIS__
    __allo_done++; // csim: this kernel finished its single pass
#endif
    while (1) { wait(); }
  }
};

SC_MODULE(drv_e_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_out<bool> done;
  sc_out< ac_int<6, false> > v1165_radr;
  sc_out<bool> v1165_re;
  sc_in< half > v1165_q;
  sc_in<bool> v1165_rrdy;
  sc_out< ac_int<6, false> > v1166_radr;
  sc_out<bool> v1166_re;
  sc_in< ac_int<32, true> > v1166_q;
  sc_in<bool> v1166_rrdy;
  sc_out< ac_int<1, false> > v1167_radr;
  sc_out<bool> v1167_re;
  sc_in< ac_int<32, true> > v1167_q;
  sc_in<bool> v1167_rrdy;
  Connections::Out< ac_int<17, false> > v1168;
  Connections::In< ac_int<32, true> > v1169;
  SC_HAS_PROCESS(drv_e_0);
  drv_e_0(sc_module_name n) : sc_module(n), done("done"), v1168("v1168"), v1169("v1169") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <in>
  half v1165_rd(ac_int<6, false> addr) {
    v1165_radr.write(addr); v1165_re.write(true);
    wait();                    // edge N: address captured
    v1165_re.write(false);
    wait();                    // data valid on this edge
    return v1165_q.read();
  }
  #pragma design modulario <in>
  ac_int<32, true> v1166_rd(ac_int<6, false> addr) {
    v1166_radr.write(addr); v1166_re.write(true);
    wait();                    // edge N: address captured
    v1166_re.write(false);
    wait();                    // data valid on this edge
    return v1166_q.read();
  }
  #pragma design modulario <in>
  ac_int<32, true> v1167_rd(ac_int<1, false> addr) {
    v1167_radr.write(addr); v1167_re.write(true);
    wait();                    // edge N: address captured
    v1167_re.write(false);
    wait();                    // data valid on this edge
    return v1167_q.read();
  }
  void run() {
    v1168.Reset();
    v1169.Reset();
    v1165_radr.write(0);
    v1165_re.write(0);
    v1166_radr.write(0);
    v1166_re.write(0);
    v1167_radr.write(0);
    v1167_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t dcred1[1];	// L1777
    for (int v1171 = 0; v1171 < 1; v1171++) {	// L1778
      dcred1[v1171] = 0;	// L1778
    }
    int32_t sp1[1];	// L1779
    for (int v1173 = 0; v1173 < 1; v1173++) {	// L1780
      sp1[v1173] = 0;	// L1780
    }
    ac_int<17, false> zw1;	// L1781
    zw1 = 0;	// L1782
    int32_t v1175;
    v1175 = v1167_rd((ac_int<1, false>)(((0) + (0))));	// L1783
    ac_int<33, true> v1176 = v1175;	// L1784
    ac_int<33, true> v1177 = v1176 - 1;	// L1785
    int v1178 = v1177;	// L1786
    for (int v1179 = 0; v1179 < v1178; v1179 += 1) {	// L1787
      ac_int<17, true> v1180 = zw1;	// L1788
      v1168.Push(v1180);	// L1789
    }
    l_S_t_1_t2: for (int t2 = 0; t2 < 55; t2++) {	// L1791
      int32_t v1182 = v1169.Pop();	// L1792
      int32_t v1183 = dcred1[0];	// L1793
      ac_int<33, true> v1184 = v1183;	// L1794
      ac_int<33, true> v1185 = v1182;	// L1795
      ac_int<33, true> v1186 = v1184 + v1185;	// L1796
      int32_t v1187 = v1186;	// L1797
      dcred1[0] = v1187;	// L1798
      ac_int<17, false> w1;	// L1799
      w1 = 0;	// L1800
      int32_t v1189 = sp1[0];	// L1801
      bool v1190 = v1189 < 55;	// L1802
      ac_int<33, true> v1191 = t2;	// L1803
      ac_int<33, true> v1192 = v1189;	// L1804
      bool v1193 = v1191 >= v1192;	// L1805
      bool v1194 = v1190 & v1193;	// L1806
      if (v1194) {	// L1807
        int32_t v1195 = sp1[0];	// L1808
        int v1196 = v1195;	// L1809
        int32_t v1197;
        v1197 = v1166_rd((ac_int<6, false>)(((0) * 55 + (v1196))));	// L1810
        bool v1198 = v1197 == 0;	// L1811
        if (v1198) {	// L1812
          int32_t v1199 = sp1[0];	// L1813
          ac_int<33, true> v1200 = v1199;	// L1814
          ac_int<33, true> v1201 = v1200 + 1;	// L1815
          int32_t v1202 = v1201;	// L1816
          sp1[0] = v1202;	// L1817
        } else {
          int32_t v1203 = dcred1[0];	// L1819
          bool v1204 = v1203 > 0;	// L1820
          if (v1204) {	// L1821
            ac_int<17, true> v1205 = w1;	// L1822
            ac_int<17, true> v1206;
            ac_int<17, true> _bs_v1206 = v1205;
            _bs_v1206[0] = 1;
            v1206 = _bs_v1206;	// L1823
            w1 = v1206;	// L1824
            int32_t v1207 = sp1[0];	// L1825
            int v1208 = v1207;	// L1826
            half v1209;
            v1209 = v1165_rd((ac_int<6, false>)(((0) * 55 + (v1208))));	// L1827
            uint16_t v1210 = (uint16_t)_fbits(v1209);	// L1828
            ac_int<17, true> v1211 = w1;	// L1829
            ac_int<17, true> v1212;
            ac_int<17, true> _bs_v1212 = v1211;
            _bs_v1212.set_slc(1, ac_int<16, false>(v1210));
            v1212 = _bs_v1212;	// L1830
            w1 = v1212;	// L1831
            int32_t v1213 = dcred1[0];	// L1832
            ac_int<33, true> v1214 = v1213;	// L1833
            ac_int<33, true> v1215 = v1214 - 1;	// L1834
            int32_t v1216 = v1215;	// L1835
            dcred1[0] = v1216;	// L1836
            int32_t v1217 = sp1[0];	// L1837
            ac_int<33, true> v1218 = v1217;	// L1838
            ac_int<33, true> v1219 = v1218 + 1;	// L1839
            int32_t v1220 = v1219;	// L1840
            sp1[0] = v1220;	// L1841
          }
        }
      }
      ac_int<17, true> v1221 = w1;	// L1845
      v1168.Push(v1221);	// L1846
    }
    done.write(true);  // RTL-observable completion (see the done port)
#ifndef __SYNTHESIS__
    __allo_done++; // csim: this kernel finished its single pass
#endif
    while (1) { wait(); }
  }
};

SC_MODULE(drv_n_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_out<bool> done;
  sc_out< ac_int<6, false> > v1222_radr;
  sc_out<bool> v1222_re;
  sc_in< half > v1222_q;
  sc_in<bool> v1222_rrdy;
  sc_out< ac_int<6, false> > v1223_radr;
  sc_out<bool> v1223_re;
  sc_in< ac_int<32, true> > v1223_q;
  sc_in<bool> v1223_rrdy;
  sc_out< ac_int<1, false> > v1224_radr;
  sc_out<bool> v1224_re;
  sc_in< ac_int<32, true> > v1224_q;
  sc_in<bool> v1224_rrdy;
  Connections::Out< ac_int<17, false> > v1225;
  Connections::In< ac_int<32, true> > v1226;
  SC_HAS_PROCESS(drv_n_0);
  drv_n_0(sc_module_name n) : sc_module(n), done("done"), v1225("v1225"), v1226("v1226") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <in>
  half v1222_rd(ac_int<6, false> addr) {
    v1222_radr.write(addr); v1222_re.write(true);
    wait();                    // edge N: address captured
    v1222_re.write(false);
    wait();                    // data valid on this edge
    return v1222_q.read();
  }
  #pragma design modulario <in>
  ac_int<32, true> v1223_rd(ac_int<6, false> addr) {
    v1223_radr.write(addr); v1223_re.write(true);
    wait();                    // edge N: address captured
    v1223_re.write(false);
    wait();                    // data valid on this edge
    return v1223_q.read();
  }
  #pragma design modulario <in>
  ac_int<32, true> v1224_rd(ac_int<1, false> addr) {
    v1224_radr.write(addr); v1224_re.write(true);
    wait();                    // edge N: address captured
    v1224_re.write(false);
    wait();                    // data valid on this edge
    return v1224_q.read();
  }
  void run() {
    v1225.Reset();
    v1226.Reset();
    v1222_radr.write(0);
    v1222_re.write(0);
    v1223_radr.write(0);
    v1223_re.write(0);
    v1224_radr.write(0);
    v1224_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t dcred2[1];	// L1859
    for (int v1228 = 0; v1228 < 1; v1228++) {	// L1860
      dcred2[v1228] = 0;	// L1860
    }
    int32_t sp2[1];	// L1861
    for (int v1230 = 0; v1230 < 1; v1230++) {	// L1862
      sp2[v1230] = 0;	// L1862
    }
    ac_int<17, false> zw2;	// L1863
    zw2 = 0;	// L1864
    int32_t v1232;
    v1232 = v1224_rd((ac_int<1, false>)(((0) + (0))));	// L1865
    ac_int<33, true> v1233 = v1232;	// L1866
    ac_int<33, true> v1234 = v1233 - 1;	// L1867
    int v1235 = v1234;	// L1868
    for (int v1236 = 0; v1236 < v1235; v1236 += 1) {	// L1869
      ac_int<17, true> v1237 = zw2;	// L1870
      v1225.Push(v1237);	// L1871
    }
    l_S_t_1_t3: for (int t3 = 0; t3 < 55; t3++) {	// L1873
      int32_t v1239 = v1226.Pop();	// L1874
      int32_t v1240 = dcred2[0];	// L1875
      ac_int<33, true> v1241 = v1240;	// L1876
      ac_int<33, true> v1242 = v1239;	// L1877
      ac_int<33, true> v1243 = v1241 + v1242;	// L1878
      int32_t v1244 = v1243;	// L1879
      dcred2[0] = v1244;	// L1880
      ac_int<17, false> w2;	// L1881
      w2 = 0;	// L1882
      int32_t v1246 = sp2[0];	// L1883
      bool v1247 = v1246 < 55;	// L1884
      ac_int<33, true> v1248 = t3;	// L1885
      ac_int<33, true> v1249 = v1246;	// L1886
      bool v1250 = v1248 >= v1249;	// L1887
      bool v1251 = v1247 & v1250;	// L1888
      if (v1251) {	// L1889
        int32_t v1252 = sp2[0];	// L1890
        int v1253 = v1252;	// L1891
        int32_t v1254;
        v1254 = v1223_rd((ac_int<6, false>)(((0) * 55 + (v1253))));	// L1892
        bool v1255 = v1254 == 0;	// L1893
        if (v1255) {	// L1894
          int32_t v1256 = sp2[0];	// L1895
          ac_int<33, true> v1257 = v1256;	// L1896
          ac_int<33, true> v1258 = v1257 + 1;	// L1897
          int32_t v1259 = v1258;	// L1898
          sp2[0] = v1259;	// L1899
        } else {
          int32_t v1260 = dcred2[0];	// L1901
          bool v1261 = v1260 > 0;	// L1902
          if (v1261) {	// L1903
            ac_int<17, true> v1262 = w2;	// L1904
            ac_int<17, true> v1263;
            ac_int<17, true> _bs_v1263 = v1262;
            _bs_v1263[0] = 1;
            v1263 = _bs_v1263;	// L1905
            w2 = v1263;	// L1906
            int32_t v1264 = sp2[0];	// L1907
            int v1265 = v1264;	// L1908
            half v1266;
            v1266 = v1222_rd((ac_int<6, false>)(((0) * 55 + (v1265))));	// L1909
            uint16_t v1267 = (uint16_t)_fbits(v1266);	// L1910
            ac_int<17, true> v1268 = w2;	// L1911
            ac_int<17, true> v1269;
            ac_int<17, true> _bs_v1269 = v1268;
            _bs_v1269.set_slc(1, ac_int<16, false>(v1267));
            v1269 = _bs_v1269;	// L1912
            w2 = v1269;	// L1913
            int32_t v1270 = dcred2[0];	// L1914
            ac_int<33, true> v1271 = v1270;	// L1915
            ac_int<33, true> v1272 = v1271 - 1;	// L1916
            int32_t v1273 = v1272;	// L1917
            dcred2[0] = v1273;	// L1918
            int32_t v1274 = sp2[0];	// L1919
            ac_int<33, true> v1275 = v1274;	// L1920
            ac_int<33, true> v1276 = v1275 + 1;	// L1921
            int32_t v1277 = v1276;	// L1922
            sp2[0] = v1277;	// L1923
          }
        }
      }
      ac_int<17, true> v1278 = w2;	// L1927
      v1225.Push(v1278);	// L1928
    }
    done.write(true);  // RTL-observable completion (see the done port)
#ifndef __SYNTHESIS__
    __allo_done++; // csim: this kernel finished its single pass
#endif
    while (1) { wait(); }
  }
};

SC_MODULE(drv_s_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_out<bool> done;
  sc_out< ac_int<6, false> > v1279_radr;
  sc_out<bool> v1279_re;
  sc_in< half > v1279_q;
  sc_in<bool> v1279_rrdy;
  sc_out< ac_int<6, false> > v1280_radr;
  sc_out<bool> v1280_re;
  sc_in< ac_int<32, true> > v1280_q;
  sc_in<bool> v1280_rrdy;
  sc_out< ac_int<1, false> > v1281_radr;
  sc_out<bool> v1281_re;
  sc_in< ac_int<32, true> > v1281_q;
  sc_in<bool> v1281_rrdy;
  Connections::Out< ac_int<17, false> > v1282;
  Connections::In< ac_int<32, true> > v1283;
  SC_HAS_PROCESS(drv_s_0);
  drv_s_0(sc_module_name n) : sc_module(n), done("done"), v1282("v1282"), v1283("v1283") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <in>
  half v1279_rd(ac_int<6, false> addr) {
    v1279_radr.write(addr); v1279_re.write(true);
    wait();                    // edge N: address captured
    v1279_re.write(false);
    wait();                    // data valid on this edge
    return v1279_q.read();
  }
  #pragma design modulario <in>
  ac_int<32, true> v1280_rd(ac_int<6, false> addr) {
    v1280_radr.write(addr); v1280_re.write(true);
    wait();                    // edge N: address captured
    v1280_re.write(false);
    wait();                    // data valid on this edge
    return v1280_q.read();
  }
  #pragma design modulario <in>
  ac_int<32, true> v1281_rd(ac_int<1, false> addr) {
    v1281_radr.write(addr); v1281_re.write(true);
    wait();                    // edge N: address captured
    v1281_re.write(false);
    wait();                    // data valid on this edge
    return v1281_q.read();
  }
  void run() {
    v1282.Reset();
    v1283.Reset();
    v1279_radr.write(0);
    v1279_re.write(0);
    v1280_radr.write(0);
    v1280_re.write(0);
    v1281_radr.write(0);
    v1281_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t dcred3[1];	// L1941
    for (int v1285 = 0; v1285 < 1; v1285++) {	// L1942
      dcred3[v1285] = 0;	// L1942
    }
    int32_t sp3[1];	// L1943
    for (int v1287 = 0; v1287 < 1; v1287++) {	// L1944
      sp3[v1287] = 0;	// L1944
    }
    ac_int<17, false> zw3;	// L1945
    zw3 = 0;	// L1946
    int32_t v1289;
    v1289 = v1281_rd((ac_int<1, false>)(((0) + (0))));	// L1947
    ac_int<33, true> v1290 = v1289;	// L1948
    ac_int<33, true> v1291 = v1290 - 1;	// L1949
    int v1292 = v1291;	// L1950
    for (int v1293 = 0; v1293 < v1292; v1293 += 1) {	// L1951
      ac_int<17, true> v1294 = zw3;	// L1952
      v1282.Push(v1294);	// L1953
    }
    l_S_t_1_t4: for (int t4 = 0; t4 < 55; t4++) {	// L1955
      int32_t v1296 = v1283.Pop();	// L1956
      int32_t v1297 = dcred3[0];	// L1957
      ac_int<33, true> v1298 = v1297;	// L1958
      ac_int<33, true> v1299 = v1296;	// L1959
      ac_int<33, true> v1300 = v1298 + v1299;	// L1960
      int32_t v1301 = v1300;	// L1961
      dcred3[0] = v1301;	// L1962
      ac_int<17, false> w3;	// L1963
      w3 = 0;	// L1964
      int32_t v1303 = sp3[0];	// L1965
      bool v1304 = v1303 < 55;	// L1966
      ac_int<33, true> v1305 = t4;	// L1967
      ac_int<33, true> v1306 = v1303;	// L1968
      bool v1307 = v1305 >= v1306;	// L1969
      bool v1308 = v1304 & v1307;	// L1970
      if (v1308) {	// L1971
        int32_t v1309 = sp3[0];	// L1972
        int v1310 = v1309;	// L1973
        int32_t v1311;
        v1311 = v1280_rd((ac_int<6, false>)(((0) * 55 + (v1310))));	// L1974
        bool v1312 = v1311 == 0;	// L1975
        if (v1312) {	// L1976
          int32_t v1313 = sp3[0];	// L1977
          ac_int<33, true> v1314 = v1313;	// L1978
          ac_int<33, true> v1315 = v1314 + 1;	// L1979
          int32_t v1316 = v1315;	// L1980
          sp3[0] = v1316;	// L1981
        } else {
          int32_t v1317 = dcred3[0];	// L1983
          bool v1318 = v1317 > 0;	// L1984
          if (v1318) {	// L1985
            ac_int<17, true> v1319 = w3;	// L1986
            ac_int<17, true> v1320;
            ac_int<17, true> _bs_v1320 = v1319;
            _bs_v1320[0] = 1;
            v1320 = _bs_v1320;	// L1987
            w3 = v1320;	// L1988
            int32_t v1321 = sp3[0];	// L1989
            int v1322 = v1321;	// L1990
            half v1323;
            v1323 = v1279_rd((ac_int<6, false>)(((0) * 55 + (v1322))));	// L1991
            uint16_t v1324 = (uint16_t)_fbits(v1323);	// L1992
            ac_int<17, true> v1325 = w3;	// L1993
            ac_int<17, true> v1326;
            ac_int<17, true> _bs_v1326 = v1325;
            _bs_v1326.set_slc(1, ac_int<16, false>(v1324));
            v1326 = _bs_v1326;	// L1994
            w3 = v1326;	// L1995
            int32_t v1327 = dcred3[0];	// L1996
            ac_int<33, true> v1328 = v1327;	// L1997
            ac_int<33, true> v1329 = v1328 - 1;	// L1998
            int32_t v1330 = v1329;	// L1999
            dcred3[0] = v1330;	// L2000
            int32_t v1331 = sp3[0];	// L2001
            ac_int<33, true> v1332 = v1331;	// L2002
            ac_int<33, true> v1333 = v1332 + 1;	// L2003
            int32_t v1334 = v1333;	// L2004
            sp3[0] = v1334;	// L2005
          }
        }
      }
      ac_int<17, true> v1335 = w3;	// L2009
      v1282.Push(v1335);	// L2010
    }
    done.write(true);  // RTL-observable completion (see the done port)
#ifndef __SYNTHESIS__
    __allo_done++; // csim: this kernel finished its single pass
#endif
    while (1) { wait(); }
  }
};

SC_MODULE(col_w_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_out<bool> done;
  sc_out< ac_int<6, false> > v1336_wadr;
  sc_out< half > v1336_d;
  sc_out<bool> v1336_we;
  sc_in<bool> v1336_wrdy;
  sc_out< ac_int<1, false> > v1337_radr;
  sc_out<bool> v1337_re;
  sc_in< ac_int<32, true> > v1337_q;
  sc_in<bool> v1337_rrdy;
  Connections::Out< ac_int<32, true> > v1338;
  Connections::In< ac_int<17, false> > v1339;
  SC_HAS_PROCESS(col_w_0);
  col_w_0(sc_module_name n) : sc_module(n), done("done"), v1338("v1338"), v1339("v1339") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <out>
  void v1336_wr(ac_int<6, false> addr, half val) {
    v1336_wadr.write(addr); v1336_d.write(val);
    v1336_we.write(true);
    wait();
    v1336_we.write(false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v1337_rd(ac_int<1, false> addr) {
    v1337_radr.write(addr); v1337_re.write(true);
    wait();                    // edge N: address captured
    v1337_re.write(false);
    wait();                    // data valid on this edge
    return v1337_q.read();
  }
  void run() {
    v1338.Reset();
    v1339.Reset();
    v1336_wadr.write(0);
    v1336_d.write(half(0.0f));
    v1336_we.write(0);
    v1337_radr.write(0);
    v1337_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t k2[1];	// L2023
    for (int v1341 = 0; v1341 < 1; v1341++) {	// L2024
      k2[v1341] = 0;	// L2024
    }
    int32_t cret[1];	// L2025
    for (int v1343 = 0; v1343 < 1; v1343++) {	// L2026
      cret[v1343] = 0;	// L2026
    }
    int32_t zc;	// L2027
    zc = 0;	// L2028
    int32_t v1345;
    v1345 = v1337_rd((ac_int<1, false>)(((0) + (0))));	// L2029
    ac_int<33, true> v1346 = v1345;	// L2030
    ac_int<33, true> v1347 = v1346 - 1;	// L2031
    int v1348 = v1347;	// L2032
    for (int v1349 = 0; v1349 < v1348; v1349 += 1) {	// L2033
      int32_t v1350 = zc;	// L2034
      v1338.Push(v1350);	// L2035
    }
    cret[0] = 2;	// L2037
    int32_t v1351 = cret[0];	// L2038
    v1338.Push(v1351);	// L2039
    l_S_t_1_t5: for (int t5 = 0; t5 < 55; t5++) {	// L2040
      ac_int<17, false> v1353 = v1339.Pop();	// L2041
      ac_int<17, false> w4;	// L2042
      w4 = v1353;	// L2043
      cret[0] = 0;	// L2044
      ac_int<17, true> v1355 = w4;	// L2045
      bool v1356;
      ac_int<17, true> _bs_v1356 = v1355;
      v1356 = _bs_v1356[0];	// L2046
      int32_t v1357 = v1356;	// L2047
      bool v1358 = v1357 == 1;	// L2048
      if (v1358) {	// L2049
        cret[0] = 1;	// L2050
        int32_t v1359 = k2[0];	// L2051
        bool v1360 = v1359 < 55;	// L2052
        if (v1360) {	// L2053
          ac_int<17, true> v1361 = w4;	// L2054
          int16_t v1362;
          ac_int<17, true> _bs_v1362 = v1361;
          v1362 = _bs_v1362.slc<16>(1);	// L2055
          half v1363; v1363.set_data(ac_int<16, true>(v1362));	// L2056
          int32_t v1364 = k2[0];	// L2057
          int v1365 = v1364;	// L2058
          v1336_wr((ac_int<6, false>)(((0) * 55 + (v1365))), v1363);	// L2059
          int32_t v1366 = k2[0];	// L2060
          ac_int<33, true> v1367 = v1366;	// L2061
          ac_int<33, true> v1368 = v1367 + 1;	// L2062
          int32_t v1369 = v1368;	// L2063
          k2[0] = v1369;	// L2064
        }
      }
      int32_t v1370 = cret[0];	// L2067
      v1338.Push(v1370);	// L2068
    }
    done.write(true);  // RTL-observable completion (see the done port)
#ifndef __SYNTHESIS__
    __allo_done++; // csim: this kernel finished its single pass
#endif
    while (1) { wait(); }
  }
};

SC_MODULE(col_e_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_out<bool> done;
  sc_out< ac_int<6, false> > v1371_wadr;
  sc_out< half > v1371_d;
  sc_out<bool> v1371_we;
  sc_in<bool> v1371_wrdy;
  sc_out< ac_int<1, false> > v1372_radr;
  sc_out<bool> v1372_re;
  sc_in< ac_int<32, true> > v1372_q;
  sc_in<bool> v1372_rrdy;
  Connections::Out< ac_int<32, true> > v1373;
  Connections::In< ac_int<17, false> > v1374;
  SC_HAS_PROCESS(col_e_0);
  col_e_0(sc_module_name n) : sc_module(n), done("done"), v1373("v1373"), v1374("v1374") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <out>
  void v1371_wr(ac_int<6, false> addr, half val) {
    v1371_wadr.write(addr); v1371_d.write(val);
    v1371_we.write(true);
    wait();
    v1371_we.write(false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v1372_rd(ac_int<1, false> addr) {
    v1372_radr.write(addr); v1372_re.write(true);
    wait();                    // edge N: address captured
    v1372_re.write(false);
    wait();                    // data valid on this edge
    return v1372_q.read();
  }
  void run() {
    v1373.Reset();
    v1374.Reset();
    v1371_wadr.write(0);
    v1371_d.write(half(0.0f));
    v1371_we.write(0);
    v1372_radr.write(0);
    v1372_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t k3[1];	// L2081
    for (int v1376 = 0; v1376 < 1; v1376++) {	// L2082
      k3[v1376] = 0;	// L2082
    }
    int32_t cret1[1];	// L2083
    for (int v1378 = 0; v1378 < 1; v1378++) {	// L2084
      cret1[v1378] = 0;	// L2084
    }
    int32_t zc1;	// L2085
    zc1 = 0;	// L2086
    int32_t v1380;
    v1380 = v1372_rd((ac_int<1, false>)(((0) + (0))));	// L2087
    ac_int<33, true> v1381 = v1380;	// L2088
    ac_int<33, true> v1382 = v1381 - 1;	// L2089
    int v1383 = v1382;	// L2090
    for (int v1384 = 0; v1384 < v1383; v1384 += 1) {	// L2091
      int32_t v1385 = zc1;	// L2092
      v1373.Push(v1385);	// L2093
    }
    cret1[0] = 2;	// L2095
    int32_t v1386 = cret1[0];	// L2096
    v1373.Push(v1386);	// L2097
    l_S_t_1_t6: for (int t6 = 0; t6 < 55; t6++) {	// L2098
      ac_int<17, false> v1388 = v1374.Pop();	// L2099
      ac_int<17, false> w5;	// L2100
      w5 = v1388;	// L2101
      cret1[0] = 0;	// L2102
      ac_int<17, true> v1390 = w5;	// L2103
      bool v1391;
      ac_int<17, true> _bs_v1391 = v1390;
      v1391 = _bs_v1391[0];	// L2104
      int32_t v1392 = v1391;	// L2105
      bool v1393 = v1392 == 1;	// L2106
      if (v1393) {	// L2107
        cret1[0] = 1;	// L2108
        int32_t v1394 = k3[0];	// L2109
        bool v1395 = v1394 < 55;	// L2110
        if (v1395) {	// L2111
          ac_int<17, true> v1396 = w5;	// L2112
          int16_t v1397;
          ac_int<17, true> _bs_v1397 = v1396;
          v1397 = _bs_v1397.slc<16>(1);	// L2113
          half v1398; v1398.set_data(ac_int<16, true>(v1397));	// L2114
          int32_t v1399 = k3[0];	// L2115
          int v1400 = v1399;	// L2116
          v1371_wr((ac_int<6, false>)(((0) * 55 + (v1400))), v1398);	// L2117
          int32_t v1401 = k3[0];	// L2118
          ac_int<33, true> v1402 = v1401;	// L2119
          ac_int<33, true> v1403 = v1402 + 1;	// L2120
          int32_t v1404 = v1403;	// L2121
          k3[0] = v1404;	// L2122
        }
      }
      int32_t v1405 = cret1[0];	// L2125
      v1373.Push(v1405);	// L2126
    }
    done.write(true);  // RTL-observable completion (see the done port)
#ifndef __SYNTHESIS__
    __allo_done++; // csim: this kernel finished its single pass
#endif
    while (1) { wait(); }
  }
};

SC_MODULE(col_n_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_out<bool> done;
  sc_out< ac_int<6, false> > v1406_wadr;
  sc_out< half > v1406_d;
  sc_out<bool> v1406_we;
  sc_in<bool> v1406_wrdy;
  sc_out< ac_int<1, false> > v1407_radr;
  sc_out<bool> v1407_re;
  sc_in< ac_int<32, true> > v1407_q;
  sc_in<bool> v1407_rrdy;
  Connections::Out< ac_int<32, true> > v1408;
  Connections::In< ac_int<17, false> > v1409;
  SC_HAS_PROCESS(col_n_0);
  col_n_0(sc_module_name n) : sc_module(n), done("done"), v1408("v1408"), v1409("v1409") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <out>
  void v1406_wr(ac_int<6, false> addr, half val) {
    v1406_wadr.write(addr); v1406_d.write(val);
    v1406_we.write(true);
    wait();
    v1406_we.write(false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v1407_rd(ac_int<1, false> addr) {
    v1407_radr.write(addr); v1407_re.write(true);
    wait();                    // edge N: address captured
    v1407_re.write(false);
    wait();                    // data valid on this edge
    return v1407_q.read();
  }
  void run() {
    v1408.Reset();
    v1409.Reset();
    v1406_wadr.write(0);
    v1406_d.write(half(0.0f));
    v1406_we.write(0);
    v1407_radr.write(0);
    v1407_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t k4[1];	// L2139
    for (int v1411 = 0; v1411 < 1; v1411++) {	// L2140
      k4[v1411] = 0;	// L2140
    }
    int32_t cret2[1];	// L2141
    for (int v1413 = 0; v1413 < 1; v1413++) {	// L2142
      cret2[v1413] = 0;	// L2142
    }
    int32_t zc2;	// L2143
    zc2 = 0;	// L2144
    int32_t v1415;
    v1415 = v1407_rd((ac_int<1, false>)(((0) + (0))));	// L2145
    ac_int<33, true> v1416 = v1415;	// L2146
    ac_int<33, true> v1417 = v1416 - 1;	// L2147
    int v1418 = v1417;	// L2148
    for (int v1419 = 0; v1419 < v1418; v1419 += 1) {	// L2149
      int32_t v1420 = zc2;	// L2150
      v1408.Push(v1420);	// L2151
    }
    cret2[0] = 2;	// L2153
    int32_t v1421 = cret2[0];	// L2154
    v1408.Push(v1421);	// L2155
    l_S_t_1_t7: for (int t7 = 0; t7 < 55; t7++) {	// L2156
      ac_int<17, false> v1423 = v1409.Pop();	// L2157
      ac_int<17, false> w6;	// L2158
      w6 = v1423;	// L2159
      cret2[0] = 0;	// L2160
      ac_int<17, true> v1425 = w6;	// L2161
      bool v1426;
      ac_int<17, true> _bs_v1426 = v1425;
      v1426 = _bs_v1426[0];	// L2162
      int32_t v1427 = v1426;	// L2163
      bool v1428 = v1427 == 1;	// L2164
      if (v1428) {	// L2165
        cret2[0] = 1;	// L2166
        int32_t v1429 = k4[0];	// L2167
        bool v1430 = v1429 < 55;	// L2168
        if (v1430) {	// L2169
          ac_int<17, true> v1431 = w6;	// L2170
          int16_t v1432;
          ac_int<17, true> _bs_v1432 = v1431;
          v1432 = _bs_v1432.slc<16>(1);	// L2171
          half v1433; v1433.set_data(ac_int<16, true>(v1432));	// L2172
          int32_t v1434 = k4[0];	// L2173
          int v1435 = v1434;	// L2174
          v1406_wr((ac_int<6, false>)(((0) * 55 + (v1435))), v1433);	// L2175
          int32_t v1436 = k4[0];	// L2176
          ac_int<33, true> v1437 = v1436;	// L2177
          ac_int<33, true> v1438 = v1437 + 1;	// L2178
          int32_t v1439 = v1438;	// L2179
          k4[0] = v1439;	// L2180
        }
      }
      int32_t v1440 = cret2[0];	// L2183
      v1408.Push(v1440);	// L2184
    }
    done.write(true);  // RTL-observable completion (see the done port)
#ifndef __SYNTHESIS__
    __allo_done++; // csim: this kernel finished its single pass
#endif
    while (1) { wait(); }
  }
};

SC_MODULE(col_s_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_out<bool> done;
  sc_out< ac_int<6, false> > v1441_wadr;
  sc_out< half > v1441_d;
  sc_out<bool> v1441_we;
  sc_in<bool> v1441_wrdy;
  sc_out< ac_int<1, false> > v1442_radr;
  sc_out<bool> v1442_re;
  sc_in< ac_int<32, true> > v1442_q;
  sc_in<bool> v1442_rrdy;
  Connections::Out< ac_int<32, true> > v1443;
  Connections::In< ac_int<17, false> > v1444;
  SC_HAS_PROCESS(col_s_0);
  col_s_0(sc_module_name n) : sc_module(n), done("done"), v1443("v1443"), v1444("v1444") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <out>
  void v1441_wr(ac_int<6, false> addr, half val) {
    v1441_wadr.write(addr); v1441_d.write(val);
    v1441_we.write(true);
    wait();
    v1441_we.write(false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v1442_rd(ac_int<1, false> addr) {
    v1442_radr.write(addr); v1442_re.write(true);
    wait();                    // edge N: address captured
    v1442_re.write(false);
    wait();                    // data valid on this edge
    return v1442_q.read();
  }
  void run() {
    v1443.Reset();
    v1444.Reset();
    v1441_wadr.write(0);
    v1441_d.write(half(0.0f));
    v1441_we.write(0);
    v1442_radr.write(0);
    v1442_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t k5[1];	// L2197
    for (int v1446 = 0; v1446 < 1; v1446++) {	// L2198
      k5[v1446] = 0;	// L2198
    }
    int32_t cret3[1];	// L2199
    for (int v1448 = 0; v1448 < 1; v1448++) {	// L2200
      cret3[v1448] = 0;	// L2200
    }
    int32_t zc3;	// L2201
    zc3 = 0;	// L2202
    int32_t v1450;
    v1450 = v1442_rd((ac_int<1, false>)(((0) + (0))));	// L2203
    ac_int<33, true> v1451 = v1450;	// L2204
    ac_int<33, true> v1452 = v1451 - 1;	// L2205
    int v1453 = v1452;	// L2206
    for (int v1454 = 0; v1454 < v1453; v1454 += 1) {	// L2207
      int32_t v1455 = zc3;	// L2208
      v1443.Push(v1455);	// L2209
    }
    cret3[0] = 2;	// L2211
    int32_t v1456 = cret3[0];	// L2212
    v1443.Push(v1456);	// L2213
    l_S_t_1_t8: for (int t8 = 0; t8 < 55; t8++) {	// L2214
      ac_int<17, false> v1458 = v1444.Pop();	// L2215
      ac_int<17, false> w7;	// L2216
      w7 = v1458;	// L2217
      cret3[0] = 0;	// L2218
      ac_int<17, true> v1460 = w7;	// L2219
      bool v1461;
      ac_int<17, true> _bs_v1461 = v1460;
      v1461 = _bs_v1461[0];	// L2220
      int32_t v1462 = v1461;	// L2221
      bool v1463 = v1462 == 1;	// L2222
      if (v1463) {	// L2223
        cret3[0] = 1;	// L2224
        int32_t v1464 = k5[0];	// L2225
        bool v1465 = v1464 < 55;	// L2226
        if (v1465) {	// L2227
          ac_int<17, true> v1466 = w7;	// L2228
          int16_t v1467;
          ac_int<17, true> _bs_v1467 = v1466;
          v1467 = _bs_v1467.slc<16>(1);	// L2229
          half v1468; v1468.set_data(ac_int<16, true>(v1467));	// L2230
          int32_t v1469 = k5[0];	// L2231
          int v1470 = v1469;	// L2232
          v1441_wr((ac_int<6, false>)(((0) * 55 + (v1470))), v1468);	// L2233
          int32_t v1471 = k5[0];	// L2234
          ac_int<33, true> v1472 = v1471;	// L2235
          ac_int<33, true> v1473 = v1472 + 1;	// L2236
          int32_t v1474 = v1473;	// L2237
          k5[0] = v1474;	// L2238
        }
      }
      int32_t v1475 = cret3[0];	// L2241
      v1443.Push(v1475);	// L2242
    }
    done.write(true);  // RTL-observable completion (see the done port)
#ifndef __SYNTHESIS__
    __allo_done++; // csim: this kernel finished its single pass
#endif
    while (1) { wait(); }
  }
};

SC_MODULE(rdrv_w_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_out<bool> done;
  sc_out< ac_int<6, false> > v1476_radr;
  sc_out<bool> v1476_re;
  sc_in< ac_int<32, true> > v1476_q;
  sc_in<bool> v1476_rrdy;
  sc_out< ac_int<1, false> > v1477_radr;
  sc_out<bool> v1477_re;
  sc_in< ac_int<32, true> > v1477_q;
  sc_in<bool> v1477_rrdy;
  Connections::Out< ac_int<26, false> > v1478;
  Connections::In< ac_int<32, true> > v1479;
  SC_HAS_PROCESS(rdrv_w_0);
  rdrv_w_0(sc_module_name n) : sc_module(n), done("done"), v1478("v1478"), v1479("v1479") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v1476_rd(ac_int<6, false> addr) {
    v1476_radr.write(addr); v1476_re.write(true);
    wait();                    // edge N: address captured
    v1476_re.write(false);
    wait();                    // data valid on this edge
    return v1476_q.read();
  }
  #pragma design modulario <in>
  ac_int<32, true> v1477_rd(ac_int<1, false> addr) {
    v1477_radr.write(addr); v1477_re.write(true);
    wait();                    // edge N: address captured
    v1477_re.write(false);
    wait();                    // data valid on this edge
    return v1477_q.read();
  }
  void run() {
    v1478.Reset();
    v1479.Reset();
    v1476_radr.write(0);
    v1476_re.write(0);
    v1477_radr.write(0);
    v1477_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t dcred4[1];	// L2254
    for (int v1481 = 0; v1481 < 1; v1481++) {	// L2255
      dcred4[v1481] = 0;	// L2255
    }
    int32_t sp4[1];	// L2256
    for (int v1483 = 0; v1483 < 1; v1483++) {	// L2257
      sp4[v1483] = 0;	// L2257
    }
    ac_int<26, false> zp;	// L2258
    zp = 0;	// L2259
    int32_t v1485;
    v1485 = v1477_rd((ac_int<1, false>)(((0) + (0))));	// L2260
    ac_int<33, true> v1486 = v1485;	// L2261
    ac_int<33, true> v1487 = v1486 - 1;	// L2262
    int v1488 = v1487;	// L2263
    for (int v1489 = 0; v1489 < v1488; v1489 += 1) {	// L2264
      ac_int<26, true> v1490 = zp;	// L2265
      v1478.Push(v1490);	// L2266
    }
    l_S_t_1_t9: for (int t9 = 0; t9 < 55; t9++) {	// L2268
      int32_t v1492 = v1479.Pop();	// L2269
      int32_t v1493 = dcred4[0];	// L2270
      ac_int<33, true> v1494 = v1493;	// L2271
      ac_int<33, true> v1495 = v1492;	// L2272
      ac_int<33, true> v1496 = v1494 + v1495;	// L2273
      int32_t v1497 = v1496;	// L2274
      dcred4[0] = v1497;	// L2275
      ac_int<26, false> pw;	// L2276
      pw = 0;	// L2277
      int32_t v1499 = sp4[0];	// L2278
      bool v1500 = v1499 < 55;	// L2279
      if (v1500) {	// L2280
        ac_int<26, false> cand;	// L2281
        cand = 0;	// L2282
        int32_t v1502 = sp4[0];	// L2283
        int v1503 = v1502;	// L2284
        int32_t v1504;
        v1504 = v1476_rd((ac_int<6, false>)(((0) * 55 + (v1503))));	// L2285
        ac_int<26, false> v1505 = v1504;	// L2286
        ac_int<26, true> v1506 = cand;	// L2287
        ac_int<26, true> v1507;
        ac_int<26, true> _bs_v1507 = v1506;
        _bs_v1507.set_slc(0, ac_int<26, false>(v1505));
        v1507 = _bs_v1507;	// L2288
        cand = v1507;	// L2289
        ac_int<26, true> v1508 = cand;	// L2290
        bool v1509;
        ac_int<26, true> _bs_v1509 = v1508;
        v1509 = _bs_v1509[25];	// L2291
        int32_t v1510 = v1509;	// L2292
        bool v1511 = v1510 == 0;	// L2293
        if (v1511) {	// L2294
          int32_t v1512 = sp4[0];	// L2295
          ac_int<33, true> v1513 = v1512;	// L2296
          ac_int<33, true> v1514 = v1513 + 1;	// L2297
          int32_t v1515 = v1514;	// L2298
          sp4[0] = v1515;	// L2299
        } else {
          int32_t v1516 = dcred4[0];	// L2301
          bool v1517 = v1516 > 0;	// L2302
          if (v1517) {	// L2303
            ac_int<26, true> v1518 = cand;	// L2304
            pw = v1518;	// L2305
            int32_t v1519 = dcred4[0];	// L2306
            ac_int<33, true> v1520 = v1519;	// L2307
            ac_int<33, true> v1521 = v1520 - 1;	// L2308
            int32_t v1522 = v1521;	// L2309
            dcred4[0] = v1522;	// L2310
            int32_t v1523 = sp4[0];	// L2311
            ac_int<33, true> v1524 = v1523;	// L2312
            ac_int<33, true> v1525 = v1524 + 1;	// L2313
            int32_t v1526 = v1525;	// L2314
            sp4[0] = v1526;	// L2315
          }
        }
      }
      ac_int<26, true> v1527 = pw;	// L2319
      v1478.Push(v1527);	// L2320
    }
    done.write(true);  // RTL-observable completion (see the done port)
#ifndef __SYNTHESIS__
    __allo_done++; // csim: this kernel finished its single pass
#endif
    while (1) { wait(); }
  }
};

SC_MODULE(rdrv_e_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_out<bool> done;
  sc_out< ac_int<6, false> > v1528_radr;
  sc_out<bool> v1528_re;
  sc_in< ac_int<32, true> > v1528_q;
  sc_in<bool> v1528_rrdy;
  sc_out< ac_int<1, false> > v1529_radr;
  sc_out<bool> v1529_re;
  sc_in< ac_int<32, true> > v1529_q;
  sc_in<bool> v1529_rrdy;
  Connections::Out< ac_int<26, false> > v1530;
  Connections::In< ac_int<32, true> > v1531;
  SC_HAS_PROCESS(rdrv_e_0);
  rdrv_e_0(sc_module_name n) : sc_module(n), done("done"), v1530("v1530"), v1531("v1531") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v1528_rd(ac_int<6, false> addr) {
    v1528_radr.write(addr); v1528_re.write(true);
    wait();                    // edge N: address captured
    v1528_re.write(false);
    wait();                    // data valid on this edge
    return v1528_q.read();
  }
  #pragma design modulario <in>
  ac_int<32, true> v1529_rd(ac_int<1, false> addr) {
    v1529_radr.write(addr); v1529_re.write(true);
    wait();                    // edge N: address captured
    v1529_re.write(false);
    wait();                    // data valid on this edge
    return v1529_q.read();
  }
  void run() {
    v1530.Reset();
    v1531.Reset();
    v1528_radr.write(0);
    v1528_re.write(0);
    v1529_radr.write(0);
    v1529_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t dcred5[1];	// L2332
    for (int v1533 = 0; v1533 < 1; v1533++) {	// L2333
      dcred5[v1533] = 0;	// L2333
    }
    int32_t sp5[1];	// L2334
    for (int v1535 = 0; v1535 < 1; v1535++) {	// L2335
      sp5[v1535] = 0;	// L2335
    }
    ac_int<26, false> zp1;	// L2336
    zp1 = 0;	// L2337
    int32_t v1537;
    v1537 = v1529_rd((ac_int<1, false>)(((0) + (0))));	// L2338
    ac_int<33, true> v1538 = v1537;	// L2339
    ac_int<33, true> v1539 = v1538 - 1;	// L2340
    int v1540 = v1539;	// L2341
    for (int v1541 = 0; v1541 < v1540; v1541 += 1) {	// L2342
      ac_int<26, true> v1542 = zp1;	// L2343
      v1530.Push(v1542);	// L2344
    }
    l_S_t_1_t10: for (int t10 = 0; t10 < 55; t10++) {	// L2346
      int32_t v1544 = v1531.Pop();	// L2347
      int32_t v1545 = dcred5[0];	// L2348
      ac_int<33, true> v1546 = v1545;	// L2349
      ac_int<33, true> v1547 = v1544;	// L2350
      ac_int<33, true> v1548 = v1546 + v1547;	// L2351
      int32_t v1549 = v1548;	// L2352
      dcred5[0] = v1549;	// L2353
      ac_int<26, false> pw1;	// L2354
      pw1 = 0;	// L2355
      int32_t v1551 = sp5[0];	// L2356
      bool v1552 = v1551 < 55;	// L2357
      if (v1552) {	// L2358
        ac_int<26, false> cand1;	// L2359
        cand1 = 0;	// L2360
        int32_t v1554 = sp5[0];	// L2361
        int v1555 = v1554;	// L2362
        int32_t v1556;
        v1556 = v1528_rd((ac_int<6, false>)(((0) * 55 + (v1555))));	// L2363
        ac_int<26, false> v1557 = v1556;	// L2364
        ac_int<26, true> v1558 = cand1;	// L2365
        ac_int<26, true> v1559;
        ac_int<26, true> _bs_v1559 = v1558;
        _bs_v1559.set_slc(0, ac_int<26, false>(v1557));
        v1559 = _bs_v1559;	// L2366
        cand1 = v1559;	// L2367
        ac_int<26, true> v1560 = cand1;	// L2368
        bool v1561;
        ac_int<26, true> _bs_v1561 = v1560;
        v1561 = _bs_v1561[25];	// L2369
        int32_t v1562 = v1561;	// L2370
        bool v1563 = v1562 == 0;	// L2371
        if (v1563) {	// L2372
          int32_t v1564 = sp5[0];	// L2373
          ac_int<33, true> v1565 = v1564;	// L2374
          ac_int<33, true> v1566 = v1565 + 1;	// L2375
          int32_t v1567 = v1566;	// L2376
          sp5[0] = v1567;	// L2377
        } else {
          int32_t v1568 = dcred5[0];	// L2379
          bool v1569 = v1568 > 0;	// L2380
          if (v1569) {	// L2381
            ac_int<26, true> v1570 = cand1;	// L2382
            pw1 = v1570;	// L2383
            int32_t v1571 = dcred5[0];	// L2384
            ac_int<33, true> v1572 = v1571;	// L2385
            ac_int<33, true> v1573 = v1572 - 1;	// L2386
            int32_t v1574 = v1573;	// L2387
            dcred5[0] = v1574;	// L2388
            int32_t v1575 = sp5[0];	// L2389
            ac_int<33, true> v1576 = v1575;	// L2390
            ac_int<33, true> v1577 = v1576 + 1;	// L2391
            int32_t v1578 = v1577;	// L2392
            sp5[0] = v1578;	// L2393
          }
        }
      }
      ac_int<26, true> v1579 = pw1;	// L2397
      v1530.Push(v1579);	// L2398
    }
    done.write(true);  // RTL-observable completion (see the done port)
#ifndef __SYNTHESIS__
    __allo_done++; // csim: this kernel finished its single pass
#endif
    while (1) { wait(); }
  }
};

SC_MODULE(rdrv_n_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_out<bool> done;
  sc_out< ac_int<6, false> > v1580_radr;
  sc_out<bool> v1580_re;
  sc_in< ac_int<32, true> > v1580_q;
  sc_in<bool> v1580_rrdy;
  sc_out< ac_int<1, false> > v1581_radr;
  sc_out<bool> v1581_re;
  sc_in< ac_int<32, true> > v1581_q;
  sc_in<bool> v1581_rrdy;
  Connections::Out< ac_int<26, false> > v1582;
  Connections::In< ac_int<32, true> > v1583;
  SC_HAS_PROCESS(rdrv_n_0);
  rdrv_n_0(sc_module_name n) : sc_module(n), done("done"), v1582("v1582"), v1583("v1583") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v1580_rd(ac_int<6, false> addr) {
    v1580_radr.write(addr); v1580_re.write(true);
    wait();                    // edge N: address captured
    v1580_re.write(false);
    wait();                    // data valid on this edge
    return v1580_q.read();
  }
  #pragma design modulario <in>
  ac_int<32, true> v1581_rd(ac_int<1, false> addr) {
    v1581_radr.write(addr); v1581_re.write(true);
    wait();                    // edge N: address captured
    v1581_re.write(false);
    wait();                    // data valid on this edge
    return v1581_q.read();
  }
  void run() {
    v1582.Reset();
    v1583.Reset();
    v1580_radr.write(0);
    v1580_re.write(0);
    v1581_radr.write(0);
    v1581_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t dcred6[1];	// L2410
    for (int v1585 = 0; v1585 < 1; v1585++) {	// L2411
      dcred6[v1585] = 0;	// L2411
    }
    int32_t sp6[1];	// L2412
    for (int v1587 = 0; v1587 < 1; v1587++) {	// L2413
      sp6[v1587] = 0;	// L2413
    }
    ac_int<26, false> zp2;	// L2414
    zp2 = 0;	// L2415
    int32_t v1589;
    v1589 = v1581_rd((ac_int<1, false>)(((0) + (0))));	// L2416
    ac_int<33, true> v1590 = v1589;	// L2417
    ac_int<33, true> v1591 = v1590 - 1;	// L2418
    int v1592 = v1591;	// L2419
    for (int v1593 = 0; v1593 < v1592; v1593 += 1) {	// L2420
      ac_int<26, true> v1594 = zp2;	// L2421
      v1582.Push(v1594);	// L2422
    }
    l_S_t_1_t11: for (int t11 = 0; t11 < 55; t11++) {	// L2424
      int32_t v1596 = v1583.Pop();	// L2425
      int32_t v1597 = dcred6[0];	// L2426
      ac_int<33, true> v1598 = v1597;	// L2427
      ac_int<33, true> v1599 = v1596;	// L2428
      ac_int<33, true> v1600 = v1598 + v1599;	// L2429
      int32_t v1601 = v1600;	// L2430
      dcred6[0] = v1601;	// L2431
      ac_int<26, false> pw2;	// L2432
      pw2 = 0;	// L2433
      int32_t v1603 = sp6[0];	// L2434
      bool v1604 = v1603 < 55;	// L2435
      if (v1604) {	// L2436
        ac_int<26, false> cand2;	// L2437
        cand2 = 0;	// L2438
        int32_t v1606 = sp6[0];	// L2439
        int v1607 = v1606;	// L2440
        int32_t v1608;
        v1608 = v1580_rd((ac_int<6, false>)(((0) * 55 + (v1607))));	// L2441
        ac_int<26, false> v1609 = v1608;	// L2442
        ac_int<26, true> v1610 = cand2;	// L2443
        ac_int<26, true> v1611;
        ac_int<26, true> _bs_v1611 = v1610;
        _bs_v1611.set_slc(0, ac_int<26, false>(v1609));
        v1611 = _bs_v1611;	// L2444
        cand2 = v1611;	// L2445
        ac_int<26, true> v1612 = cand2;	// L2446
        bool v1613;
        ac_int<26, true> _bs_v1613 = v1612;
        v1613 = _bs_v1613[25];	// L2447
        int32_t v1614 = v1613;	// L2448
        bool v1615 = v1614 == 0;	// L2449
        if (v1615) {	// L2450
          int32_t v1616 = sp6[0];	// L2451
          ac_int<33, true> v1617 = v1616;	// L2452
          ac_int<33, true> v1618 = v1617 + 1;	// L2453
          int32_t v1619 = v1618;	// L2454
          sp6[0] = v1619;	// L2455
        } else {
          int32_t v1620 = dcred6[0];	// L2457
          bool v1621 = v1620 > 0;	// L2458
          if (v1621) {	// L2459
            ac_int<26, true> v1622 = cand2;	// L2460
            pw2 = v1622;	// L2461
            int32_t v1623 = dcred6[0];	// L2462
            ac_int<33, true> v1624 = v1623;	// L2463
            ac_int<33, true> v1625 = v1624 - 1;	// L2464
            int32_t v1626 = v1625;	// L2465
            dcred6[0] = v1626;	// L2466
            int32_t v1627 = sp6[0];	// L2467
            ac_int<33, true> v1628 = v1627;	// L2468
            ac_int<33, true> v1629 = v1628 + 1;	// L2469
            int32_t v1630 = v1629;	// L2470
            sp6[0] = v1630;	// L2471
          }
        }
      }
      ac_int<26, true> v1631 = pw2;	// L2475
      v1582.Push(v1631);	// L2476
    }
    done.write(true);  // RTL-observable completion (see the done port)
#ifndef __SYNTHESIS__
    __allo_done++; // csim: this kernel finished its single pass
#endif
    while (1) { wait(); }
  }
};

SC_MODULE(rdrv_s_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_out<bool> done;
  sc_out< ac_int<6, false> > v1632_radr;
  sc_out<bool> v1632_re;
  sc_in< ac_int<32, true> > v1632_q;
  sc_in<bool> v1632_rrdy;
  sc_out< ac_int<1, false> > v1633_radr;
  sc_out<bool> v1633_re;
  sc_in< ac_int<32, true> > v1633_q;
  sc_in<bool> v1633_rrdy;
  Connections::Out< ac_int<26, false> > v1634;
  Connections::In< ac_int<32, true> > v1635;
  SC_HAS_PROCESS(rdrv_s_0);
  rdrv_s_0(sc_module_name n) : sc_module(n), done("done"), v1634("v1634"), v1635("v1635") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v1632_rd(ac_int<6, false> addr) {
    v1632_radr.write(addr); v1632_re.write(true);
    wait();                    // edge N: address captured
    v1632_re.write(false);
    wait();                    // data valid on this edge
    return v1632_q.read();
  }
  #pragma design modulario <in>
  ac_int<32, true> v1633_rd(ac_int<1, false> addr) {
    v1633_radr.write(addr); v1633_re.write(true);
    wait();                    // edge N: address captured
    v1633_re.write(false);
    wait();                    // data valid on this edge
    return v1633_q.read();
  }
  void run() {
    v1634.Reset();
    v1635.Reset();
    v1632_radr.write(0);
    v1632_re.write(0);
    v1633_radr.write(0);
    v1633_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t dcred7[1];	// L2488
    for (int v1637 = 0; v1637 < 1; v1637++) {	// L2489
      dcred7[v1637] = 0;	// L2489
    }
    int32_t sp7[1];	// L2490
    for (int v1639 = 0; v1639 < 1; v1639++) {	// L2491
      sp7[v1639] = 0;	// L2491
    }
    ac_int<26, false> zp3;	// L2492
    zp3 = 0;	// L2493
    int32_t v1641;
    v1641 = v1633_rd((ac_int<1, false>)(((0) + (0))));	// L2494
    ac_int<33, true> v1642 = v1641;	// L2495
    ac_int<33, true> v1643 = v1642 - 1;	// L2496
    int v1644 = v1643;	// L2497
    for (int v1645 = 0; v1645 < v1644; v1645 += 1) {	// L2498
      ac_int<26, true> v1646 = zp3;	// L2499
      v1634.Push(v1646);	// L2500
    }
    l_S_t_1_t12: for (int t12 = 0; t12 < 55; t12++) {	// L2502
      int32_t v1648 = v1635.Pop();	// L2503
      int32_t v1649 = dcred7[0];	// L2504
      ac_int<33, true> v1650 = v1649;	// L2505
      ac_int<33, true> v1651 = v1648;	// L2506
      ac_int<33, true> v1652 = v1650 + v1651;	// L2507
      int32_t v1653 = v1652;	// L2508
      dcred7[0] = v1653;	// L2509
      ac_int<26, false> pw3;	// L2510
      pw3 = 0;	// L2511
      int32_t v1655 = sp7[0];	// L2512
      bool v1656 = v1655 < 55;	// L2513
      if (v1656) {	// L2514
        ac_int<26, false> cand3;	// L2515
        cand3 = 0;	// L2516
        int32_t v1658 = sp7[0];	// L2517
        int v1659 = v1658;	// L2518
        int32_t v1660;
        v1660 = v1632_rd((ac_int<6, false>)(((0) * 55 + (v1659))));	// L2519
        ac_int<26, false> v1661 = v1660;	// L2520
        ac_int<26, true> v1662 = cand3;	// L2521
        ac_int<26, true> v1663;
        ac_int<26, true> _bs_v1663 = v1662;
        _bs_v1663.set_slc(0, ac_int<26, false>(v1661));
        v1663 = _bs_v1663;	// L2522
        cand3 = v1663;	// L2523
        ac_int<26, true> v1664 = cand3;	// L2524
        bool v1665;
        ac_int<26, true> _bs_v1665 = v1664;
        v1665 = _bs_v1665[25];	// L2525
        int32_t v1666 = v1665;	// L2526
        bool v1667 = v1666 == 0;	// L2527
        if (v1667) {	// L2528
          int32_t v1668 = sp7[0];	// L2529
          ac_int<33, true> v1669 = v1668;	// L2530
          ac_int<33, true> v1670 = v1669 + 1;	// L2531
          int32_t v1671 = v1670;	// L2532
          sp7[0] = v1671;	// L2533
        } else {
          int32_t v1672 = dcred7[0];	// L2535
          bool v1673 = v1672 > 0;	// L2536
          if (v1673) {	// L2537
            ac_int<26, true> v1674 = cand3;	// L2538
            pw3 = v1674;	// L2539
            int32_t v1675 = dcred7[0];	// L2540
            ac_int<33, true> v1676 = v1675;	// L2541
            ac_int<33, true> v1677 = v1676 - 1;	// L2542
            int32_t v1678 = v1677;	// L2543
            dcred7[0] = v1678;	// L2544
            int32_t v1679 = sp7[0];	// L2545
            ac_int<33, true> v1680 = v1679;	// L2546
            ac_int<33, true> v1681 = v1680 + 1;	// L2547
            int32_t v1682 = v1681;	// L2548
            sp7[0] = v1682;	// L2549
          }
        }
      }
      ac_int<26, true> v1683 = pw3;	// L2553
      v1634.Push(v1683);	// L2554
    }
    done.write(true);  // RTL-observable completion (see the done port)
#ifndef __SYNTHESIS__
    __allo_done++; // csim: this kernel finished its single pass
#endif
    while (1) { wait(); }
  }
};

SC_MODULE(rclc_w_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_out<bool> done;
  sc_out< ac_int<6, false> > v1684_wadr;
  sc_out< ac_int<32, true> > v1684_d;
  sc_out<bool> v1684_we;
  sc_in<bool> v1684_wrdy;
  sc_out< ac_int<1, false> > v1685_radr;
  sc_out<bool> v1685_re;
  sc_in< ac_int<32, true> > v1685_q;
  sc_in<bool> v1685_rrdy;
  Connections::Out< ac_int<32, true> > v1686;
  Connections::In< ac_int<26, false> > v1687;
  SC_HAS_PROCESS(rclc_w_0);
  rclc_w_0(sc_module_name n) : sc_module(n), done("done"), v1686("v1686"), v1687("v1687") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <out>
  void v1684_wr(ac_int<6, false> addr, ac_int<32, true> val) {
    v1684_wadr.write(addr); v1684_d.write(val);
    v1684_we.write(true);
    wait();
    v1684_we.write(false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v1685_rd(ac_int<1, false> addr) {
    v1685_radr.write(addr); v1685_re.write(true);
    wait();                    // edge N: address captured
    v1685_re.write(false);
    wait();                    // data valid on this edge
    return v1685_q.read();
  }
  void run() {
    v1686.Reset();
    v1687.Reset();
    v1684_wadr.write(0);
    v1684_d.write(0);
    v1684_we.write(0);
    v1685_radr.write(0);
    v1685_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t k6[1];	// L2568
    for (int v1689 = 0; v1689 < 1; v1689++) {	// L2569
      k6[v1689] = 0;	// L2569
    }
    int32_t cret4[1];	// L2570
    for (int v1691 = 0; v1691 < 1; v1691++) {	// L2571
      cret4[v1691] = 0;	// L2571
    }
    int32_t zc4;	// L2572
    zc4 = 0;	// L2573
    int32_t v1693;
    v1693 = v1685_rd((ac_int<1, false>)(((0) + (0))));	// L2574
    ac_int<33, true> v1694 = v1693;	// L2575
    ac_int<33, true> v1695 = v1694 - 1;	// L2576
    int v1696 = v1695;	// L2577
    for (int v1697 = 0; v1697 < v1696; v1697 += 1) {	// L2578
      int32_t v1698 = zc4;	// L2579
      v1686.Push(v1698);	// L2580
    }
    cret4[0] = 2;	// L2582
    int32_t v1699 = cret4[0];	// L2583
    v1686.Push(v1699);	// L2584
    l_S_t_1_t13: for (int t13 = 0; t13 < 55; t13++) {	// L2585
      ac_int<26, false> v1701 = v1687.Pop();	// L2586
      ac_int<26, false> pw4;	// L2587
      pw4 = v1701;	// L2588
      cret4[0] = 0;	// L2589
      ac_int<26, true> v1703 = pw4;	// L2590
      bool v1704;
      ac_int<26, true> _bs_v1704 = v1703;
      v1704 = _bs_v1704[25];	// L2591
      int32_t v1705 = v1704;	// L2592
      bool v1706 = v1705 == 1;	// L2593
      if (v1706) {	// L2594
        cret4[0] = 1;	// L2595
        int32_t v1707 = k6[0];	// L2596
        bool v1708 = v1707 < 55;	// L2597
        if (v1708) {	// L2598
          ac_int<26, true> v1709 = pw4;	// L2599
          int32_t v1710 = v1709;	// L2600
          int32_t v1711 = v1710 & 67108863;	// L2601
          int32_t v1712 = k6[0];	// L2602
          int v1713 = v1712;	// L2603
          v1684_wr((ac_int<6, false>)(((0) * 55 + (v1713))), v1711);	// L2604
          int32_t v1714 = k6[0];	// L2605
          ac_int<33, true> v1715 = v1714;	// L2606
          ac_int<33, true> v1716 = v1715 + 1;	// L2607
          int32_t v1717 = v1716;	// L2608
          k6[0] = v1717;	// L2609
        }
      }
      int32_t v1718 = cret4[0];	// L2612
      v1686.Push(v1718);	// L2613
    }
    done.write(true);  // RTL-observable completion (see the done port)
#ifndef __SYNTHESIS__
    __allo_done++; // csim: this kernel finished its single pass
#endif
    while (1) { wait(); }
  }
};

SC_MODULE(rclc_e_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_out<bool> done;
  sc_out< ac_int<6, false> > v1719_wadr;
  sc_out< ac_int<32, true> > v1719_d;
  sc_out<bool> v1719_we;
  sc_in<bool> v1719_wrdy;
  sc_out< ac_int<1, false> > v1720_radr;
  sc_out<bool> v1720_re;
  sc_in< ac_int<32, true> > v1720_q;
  sc_in<bool> v1720_rrdy;
  Connections::Out< ac_int<32, true> > v1721;
  Connections::In< ac_int<26, false> > v1722;
  SC_HAS_PROCESS(rclc_e_0);
  rclc_e_0(sc_module_name n) : sc_module(n), done("done"), v1721("v1721"), v1722("v1722") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <out>
  void v1719_wr(ac_int<6, false> addr, ac_int<32, true> val) {
    v1719_wadr.write(addr); v1719_d.write(val);
    v1719_we.write(true);
    wait();
    v1719_we.write(false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v1720_rd(ac_int<1, false> addr) {
    v1720_radr.write(addr); v1720_re.write(true);
    wait();                    // edge N: address captured
    v1720_re.write(false);
    wait();                    // data valid on this edge
    return v1720_q.read();
  }
  void run() {
    v1721.Reset();
    v1722.Reset();
    v1719_wadr.write(0);
    v1719_d.write(0);
    v1719_we.write(0);
    v1720_radr.write(0);
    v1720_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t k7[1];	// L2627
    for (int v1724 = 0; v1724 < 1; v1724++) {	// L2628
      k7[v1724] = 0;	// L2628
    }
    int32_t cret5[1];	// L2629
    for (int v1726 = 0; v1726 < 1; v1726++) {	// L2630
      cret5[v1726] = 0;	// L2630
    }
    int32_t zc5;	// L2631
    zc5 = 0;	// L2632
    int32_t v1728;
    v1728 = v1720_rd((ac_int<1, false>)(((0) + (0))));	// L2633
    ac_int<33, true> v1729 = v1728;	// L2634
    ac_int<33, true> v1730 = v1729 - 1;	// L2635
    int v1731 = v1730;	// L2636
    for (int v1732 = 0; v1732 < v1731; v1732 += 1) {	// L2637
      int32_t v1733 = zc5;	// L2638
      v1721.Push(v1733);	// L2639
    }
    cret5[0] = 2;	// L2641
    int32_t v1734 = cret5[0];	// L2642
    v1721.Push(v1734);	// L2643
    l_S_t_1_t14: for (int t14 = 0; t14 < 55; t14++) {	// L2644
      ac_int<26, false> v1736 = v1722.Pop();	// L2645
      ac_int<26, false> pw5;	// L2646
      pw5 = v1736;	// L2647
      cret5[0] = 0;	// L2648
      ac_int<26, true> v1738 = pw5;	// L2649
      bool v1739;
      ac_int<26, true> _bs_v1739 = v1738;
      v1739 = _bs_v1739[25];	// L2650
      int32_t v1740 = v1739;	// L2651
      bool v1741 = v1740 == 1;	// L2652
      if (v1741) {	// L2653
        cret5[0] = 1;	// L2654
        int32_t v1742 = k7[0];	// L2655
        bool v1743 = v1742 < 55;	// L2656
        if (v1743) {	// L2657
          ac_int<26, true> v1744 = pw5;	// L2658
          int32_t v1745 = v1744;	// L2659
          int32_t v1746 = v1745 & 67108863;	// L2660
          int32_t v1747 = k7[0];	// L2661
          int v1748 = v1747;	// L2662
          v1719_wr((ac_int<6, false>)(((0) * 55 + (v1748))), v1746);	// L2663
          int32_t v1749 = k7[0];	// L2664
          ac_int<33, true> v1750 = v1749;	// L2665
          ac_int<33, true> v1751 = v1750 + 1;	// L2666
          int32_t v1752 = v1751;	// L2667
          k7[0] = v1752;	// L2668
        }
      }
      int32_t v1753 = cret5[0];	// L2671
      v1721.Push(v1753);	// L2672
    }
    done.write(true);  // RTL-observable completion (see the done port)
#ifndef __SYNTHESIS__
    __allo_done++; // csim: this kernel finished its single pass
#endif
    while (1) { wait(); }
  }
};

SC_MODULE(rclc_n_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_out<bool> done;
  sc_out< ac_int<6, false> > v1754_wadr;
  sc_out< ac_int<32, true> > v1754_d;
  sc_out<bool> v1754_we;
  sc_in<bool> v1754_wrdy;
  sc_out< ac_int<1, false> > v1755_radr;
  sc_out<bool> v1755_re;
  sc_in< ac_int<32, true> > v1755_q;
  sc_in<bool> v1755_rrdy;
  Connections::Out< ac_int<32, true> > v1756;
  Connections::In< ac_int<26, false> > v1757;
  SC_HAS_PROCESS(rclc_n_0);
  rclc_n_0(sc_module_name n) : sc_module(n), done("done"), v1756("v1756"), v1757("v1757") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <out>
  void v1754_wr(ac_int<6, false> addr, ac_int<32, true> val) {
    v1754_wadr.write(addr); v1754_d.write(val);
    v1754_we.write(true);
    wait();
    v1754_we.write(false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v1755_rd(ac_int<1, false> addr) {
    v1755_radr.write(addr); v1755_re.write(true);
    wait();                    // edge N: address captured
    v1755_re.write(false);
    wait();                    // data valid on this edge
    return v1755_q.read();
  }
  void run() {
    v1756.Reset();
    v1757.Reset();
    v1754_wadr.write(0);
    v1754_d.write(0);
    v1754_we.write(0);
    v1755_radr.write(0);
    v1755_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t k8[1];	// L2686
    for (int v1759 = 0; v1759 < 1; v1759++) {	// L2687
      k8[v1759] = 0;	// L2687
    }
    int32_t cret6[1];	// L2688
    for (int v1761 = 0; v1761 < 1; v1761++) {	// L2689
      cret6[v1761] = 0;	// L2689
    }
    int32_t zc6;	// L2690
    zc6 = 0;	// L2691
    int32_t v1763;
    v1763 = v1755_rd((ac_int<1, false>)(((0) + (0))));	// L2692
    ac_int<33, true> v1764 = v1763;	// L2693
    ac_int<33, true> v1765 = v1764 - 1;	// L2694
    int v1766 = v1765;	// L2695
    for (int v1767 = 0; v1767 < v1766; v1767 += 1) {	// L2696
      int32_t v1768 = zc6;	// L2697
      v1756.Push(v1768);	// L2698
    }
    cret6[0] = 2;	// L2700
    int32_t v1769 = cret6[0];	// L2701
    v1756.Push(v1769);	// L2702
    l_S_t_1_t15: for (int t15 = 0; t15 < 55; t15++) {	// L2703
      ac_int<26, false> v1771 = v1757.Pop();	// L2704
      ac_int<26, false> pw6;	// L2705
      pw6 = v1771;	// L2706
      cret6[0] = 0;	// L2707
      ac_int<26, true> v1773 = pw6;	// L2708
      bool v1774;
      ac_int<26, true> _bs_v1774 = v1773;
      v1774 = _bs_v1774[25];	// L2709
      int32_t v1775 = v1774;	// L2710
      bool v1776 = v1775 == 1;	// L2711
      if (v1776) {	// L2712
        cret6[0] = 1;	// L2713
        int32_t v1777 = k8[0];	// L2714
        bool v1778 = v1777 < 55;	// L2715
        if (v1778) {	// L2716
          ac_int<26, true> v1779 = pw6;	// L2717
          int32_t v1780 = v1779;	// L2718
          int32_t v1781 = v1780 & 67108863;	// L2719
          int32_t v1782 = k8[0];	// L2720
          int v1783 = v1782;	// L2721
          v1754_wr((ac_int<6, false>)(((0) * 55 + (v1783))), v1781);	// L2722
          int32_t v1784 = k8[0];	// L2723
          ac_int<33, true> v1785 = v1784;	// L2724
          ac_int<33, true> v1786 = v1785 + 1;	// L2725
          int32_t v1787 = v1786;	// L2726
          k8[0] = v1787;	// L2727
        }
      }
      int32_t v1788 = cret6[0];	// L2730
      v1756.Push(v1788);	// L2731
    }
    done.write(true);  // RTL-observable completion (see the done port)
#ifndef __SYNTHESIS__
    __allo_done++; // csim: this kernel finished its single pass
#endif
    while (1) { wait(); }
  }
};

SC_MODULE(rclc_s_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_out<bool> done;
  sc_out< ac_int<6, false> > v1789_wadr;
  sc_out< ac_int<32, true> > v1789_d;
  sc_out<bool> v1789_we;
  sc_in<bool> v1789_wrdy;
  sc_out< ac_int<1, false> > v1790_radr;
  sc_out<bool> v1790_re;
  sc_in< ac_int<32, true> > v1790_q;
  sc_in<bool> v1790_rrdy;
  Connections::Out< ac_int<32, true> > v1791;
  Connections::In< ac_int<26, false> > v1792;
  SC_HAS_PROCESS(rclc_s_0);
  rclc_s_0(sc_module_name n) : sc_module(n), done("done"), v1791("v1791"), v1792("v1792") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <out>
  void v1789_wr(ac_int<6, false> addr, ac_int<32, true> val) {
    v1789_wadr.write(addr); v1789_d.write(val);
    v1789_we.write(true);
    wait();
    v1789_we.write(false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v1790_rd(ac_int<1, false> addr) {
    v1790_radr.write(addr); v1790_re.write(true);
    wait();                    // edge N: address captured
    v1790_re.write(false);
    wait();                    // data valid on this edge
    return v1790_q.read();
  }
  void run() {
    v1791.Reset();
    v1792.Reset();
    v1789_wadr.write(0);
    v1789_d.write(0);
    v1789_we.write(0);
    v1790_radr.write(0);
    v1790_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t k9[1];	// L2745
    for (int v1794 = 0; v1794 < 1; v1794++) {	// L2746
      k9[v1794] = 0;	// L2746
    }
    int32_t cret7[1];	// L2747
    for (int v1796 = 0; v1796 < 1; v1796++) {	// L2748
      cret7[v1796] = 0;	// L2748
    }
    int32_t zc7;	// L2749
    zc7 = 0;	// L2750
    int32_t v1798;
    v1798 = v1790_rd((ac_int<1, false>)(((0) + (0))));	// L2751
    ac_int<33, true> v1799 = v1798;	// L2752
    ac_int<33, true> v1800 = v1799 - 1;	// L2753
    int v1801 = v1800;	// L2754
    for (int v1802 = 0; v1802 < v1801; v1802 += 1) {	// L2755
      int32_t v1803 = zc7;	// L2756
      v1791.Push(v1803);	// L2757
    }
    cret7[0] = 2;	// L2759
    int32_t v1804 = cret7[0];	// L2760
    v1791.Push(v1804);	// L2761
    l_S_t_1_t16: for (int t16 = 0; t16 < 55; t16++) {	// L2762
      ac_int<26, false> v1806 = v1792.Pop();	// L2763
      ac_int<26, false> pw7;	// L2764
      pw7 = v1806;	// L2765
      cret7[0] = 0;	// L2766
      ac_int<26, true> v1808 = pw7;	// L2767
      bool v1809;
      ac_int<26, true> _bs_v1809 = v1808;
      v1809 = _bs_v1809[25];	// L2768
      int32_t v1810 = v1809;	// L2769
      bool v1811 = v1810 == 1;	// L2770
      if (v1811) {	// L2771
        cret7[0] = 1;	// L2772
        int32_t v1812 = k9[0];	// L2773
        bool v1813 = v1812 < 55;	// L2774
        if (v1813) {	// L2775
          ac_int<26, true> v1814 = pw7;	// L2776
          int32_t v1815 = v1814;	// L2777
          int32_t v1816 = v1815 & 67108863;	// L2778
          int32_t v1817 = k9[0];	// L2779
          int v1818 = v1817;	// L2780
          v1789_wr((ac_int<6, false>)(((0) * 55 + (v1818))), v1816);	// L2781
          int32_t v1819 = k9[0];	// L2782
          ac_int<33, true> v1820 = v1819;	// L2783
          ac_int<33, true> v1821 = v1820 + 1;	// L2784
          int32_t v1822 = v1821;	// L2785
          k9[0] = v1822;	// L2786
        }
      }
      int32_t v1823 = cret7[0];	// L2789
      v1791.Push(v1823);	// L2790
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
  Connections::Combinational< ac_int<17, false> > v1845_in;
  Connections::Combinational< ac_int<17, false> > v1845_out;
  Connections::Fifo< ac_int<17, false>, 8 > v1845_fifo;
  Connections::Combinational< ac_int<17, false> > v1846_in;
  Connections::Combinational< ac_int<17, false> > v1846_out;
  Connections::Fifo< ac_int<17, false>, 8 > v1846_fifo;
  Connections::Combinational< ac_int<17, false> > v1847_in;
  Connections::Combinational< ac_int<17, false> > v1847_out;
  Connections::Fifo< ac_int<17, false>, 8 > v1847_fifo;
  Connections::Combinational< ac_int<17, false> > v1848_in;
  Connections::Combinational< ac_int<17, false> > v1848_out;
  Connections::Fifo< ac_int<17, false>, 8 > v1848_fifo;
  Connections::Combinational< ac_int<17, false> > v1849_in;
  Connections::Combinational< ac_int<17, false> > v1849_out;
  Connections::Fifo< ac_int<17, false>, 8 > v1849_fifo;
  Connections::Combinational< ac_int<17, false> > v1850_in;
  Connections::Combinational< ac_int<17, false> > v1850_out;
  Connections::Fifo< ac_int<17, false>, 8 > v1850_fifo;
  Connections::Combinational< ac_int<17, false> > v1851_in;
  Connections::Combinational< ac_int<17, false> > v1851_out;
  Connections::Fifo< ac_int<17, false>, 8 > v1851_fifo;
  Connections::Combinational< ac_int<17, false> > v1852_in;
  Connections::Combinational< ac_int<17, false> > v1852_out;
  Connections::Fifo< ac_int<17, false>, 8 > v1852_fifo;
  Connections::Combinational< ac_int<26, false> > v1853_in;
  Connections::Combinational< ac_int<26, false> > v1853_out;
  Connections::Fifo< ac_int<26, false>, 8 > v1853_fifo;
  Connections::Combinational< ac_int<26, false> > v1854_in;
  Connections::Combinational< ac_int<26, false> > v1854_out;
  Connections::Fifo< ac_int<26, false>, 8 > v1854_fifo;
  Connections::Combinational< ac_int<26, false> > v1855_in;
  Connections::Combinational< ac_int<26, false> > v1855_out;
  Connections::Fifo< ac_int<26, false>, 8 > v1855_fifo;
  Connections::Combinational< ac_int<26, false> > v1856_in;
  Connections::Combinational< ac_int<26, false> > v1856_out;
  Connections::Fifo< ac_int<26, false>, 8 > v1856_fifo;
  Connections::Combinational< ac_int<26, false> > v1857_in;
  Connections::Combinational< ac_int<26, false> > v1857_out;
  Connections::Fifo< ac_int<26, false>, 8 > v1857_fifo;
  Connections::Combinational< ac_int<26, false> > v1858_in;
  Connections::Combinational< ac_int<26, false> > v1858_out;
  Connections::Fifo< ac_int<26, false>, 8 > v1858_fifo;
  Connections::Combinational< ac_int<26, false> > v1859_in;
  Connections::Combinational< ac_int<26, false> > v1859_out;
  Connections::Fifo< ac_int<26, false>, 8 > v1859_fifo;
  Connections::Combinational< ac_int<26, false> > v1860_in;
  Connections::Combinational< ac_int<26, false> > v1860_out;
  Connections::Fifo< ac_int<26, false>, 8 > v1860_fifo;
  Connections::Combinational< ac_int<32, true> > v1861_in;
  Connections::Combinational< ac_int<32, true> > v1861_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1861_fifo;
  Connections::Combinational< ac_int<32, true> > v1862_in;
  Connections::Combinational< ac_int<32, true> > v1862_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1862_fifo;
  Connections::Combinational< ac_int<32, true> > v1863_in;
  Connections::Combinational< ac_int<32, true> > v1863_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1863_fifo;
  Connections::Combinational< ac_int<32, true> > v1864_in;
  Connections::Combinational< ac_int<32, true> > v1864_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1864_fifo;
  Connections::Combinational< ac_int<32, true> > v1865_in;
  Connections::Combinational< ac_int<32, true> > v1865_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1865_fifo;
  Connections::Combinational< ac_int<32, true> > v1866_in;
  Connections::Combinational< ac_int<32, true> > v1866_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1866_fifo;
  Connections::Combinational< ac_int<32, true> > v1867_in;
  Connections::Combinational< ac_int<32, true> > v1867_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1867_fifo;
  Connections::Combinational< ac_int<32, true> > v1868_in;
  Connections::Combinational< ac_int<32, true> > v1868_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1868_fifo;
  Connections::Combinational< ac_int<32, true> > v1869_in;
  Connections::Combinational< ac_int<32, true> > v1869_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1869_fifo;
  Connections::Combinational< ac_int<32, true> > v1870_in;
  Connections::Combinational< ac_int<32, true> > v1870_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1870_fifo;
  Connections::Combinational< ac_int<32, true> > v1871_in;
  Connections::Combinational< ac_int<32, true> > v1871_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1871_fifo;
  Connections::Combinational< ac_int<32, true> > v1872_in;
  Connections::Combinational< ac_int<32, true> > v1872_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1872_fifo;
  Connections::Combinational< ac_int<32, true> > v1873_in;
  Connections::Combinational< ac_int<32, true> > v1873_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1873_fifo;
  Connections::Combinational< ac_int<32, true> > v1874_in;
  Connections::Combinational< ac_int<32, true> > v1874_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1874_fifo;
  Connections::Combinational< ac_int<32, true> > v1875_in;
  Connections::Combinational< ac_int<32, true> > v1875_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1875_fifo;
  Connections::Combinational< ac_int<32, true> > v1876_in;
  Connections::Combinational< ac_int<32, true> > v1876_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1876_fifo;
  node_0_0 u0;
  drv_w_0 u1;
  drv_e_0 u2;
  drv_n_0 u3;
  drv_s_0 u4;
  col_w_0 u5;
  col_e_0 u6;
  col_n_0 u7;
  col_s_0 u8;
  rdrv_w_0 u9;
  rdrv_e_0 u10;
  rdrv_n_0 u11;
  rdrv_s_0 u12;
  rclc_w_0 u13;
  rclc_e_0 u14;
  rclc_n_0 u15;
  rclc_s_0 u16;
  sc_signal<bool> u0_done;
  sc_signal<bool> u1_done;
  sc_signal<bool> u2_done;
  sc_signal<bool> u3_done;
  sc_signal<bool> u4_done;
  sc_signal<bool> u5_done;
  sc_signal<bool> u6_done;
  sc_signal<bool> u7_done;
  sc_signal<bool> u8_done;
  sc_signal<bool> u9_done;
  sc_signal<bool> u10_done;
  sc_signal<bool> u11_done;
  sc_signal<bool> u12_done;
  sc_signal<bool> u13_done;
  sc_signal<bool> u14_done;
  sc_signal<bool> u15_done;
  sc_signal<bool> u16_done;
  sc_out< ac_int<6, false> > v1825_radr;
  sc_out<bool> v1825_re;
  sc_in< half > v1825_q;
  sc_in<bool> v1825_rrdy;
  sc_out< ac_int<6, false> > v1826_radr;
  sc_out<bool> v1826_re;
  sc_in< ac_int<32, true> > v1826_q;
  sc_in<bool> v1826_rrdy;
  sc_out< ac_int<6, false> > v1827_radr;
  sc_out<bool> v1827_re;
  sc_in< half > v1827_q;
  sc_in<bool> v1827_rrdy;
  sc_out< ac_int<6, false> > v1828_radr;
  sc_out<bool> v1828_re;
  sc_in< ac_int<32, true> > v1828_q;
  sc_in<bool> v1828_rrdy;
  sc_out< ac_int<6, false> > v1829_radr;
  sc_out<bool> v1829_re;
  sc_in< half > v1829_q;
  sc_in<bool> v1829_rrdy;
  sc_out< ac_int<6, false> > v1830_radr;
  sc_out<bool> v1830_re;
  sc_in< ac_int<32, true> > v1830_q;
  sc_in<bool> v1830_rrdy;
  sc_out< ac_int<6, false> > v1831_radr;
  sc_out<bool> v1831_re;
  sc_in< half > v1831_q;
  sc_in<bool> v1831_rrdy;
  sc_out< ac_int<6, false> > v1832_radr;
  sc_out<bool> v1832_re;
  sc_in< ac_int<32, true> > v1832_q;
  sc_in<bool> v1832_rrdy;
  sc_out< ac_int<6, false> > v1833_wadr;
  sc_out< half > v1833_d;
  sc_out<bool> v1833_we;
  sc_in<bool> v1833_wrdy;
  sc_out< ac_int<6, false> > v1834_wadr;
  sc_out< half > v1834_d;
  sc_out<bool> v1834_we;
  sc_in<bool> v1834_wrdy;
  sc_out< ac_int<6, false> > v1835_wadr;
  sc_out< half > v1835_d;
  sc_out<bool> v1835_we;
  sc_in<bool> v1835_wrdy;
  sc_out< ac_int<6, false> > v1836_wadr;
  sc_out< half > v1836_d;
  sc_out<bool> v1836_we;
  sc_in<bool> v1836_wrdy;
  sc_out< ac_int<6, false> > v1837_radr;
  sc_out<bool> v1837_re;
  sc_in< ac_int<32, true> > v1837_q;
  sc_in<bool> v1837_rrdy;
  sc_out< ac_int<6, false> > v1838_radr;
  sc_out<bool> v1838_re;
  sc_in< ac_int<32, true> > v1838_q;
  sc_in<bool> v1838_rrdy;
  sc_out< ac_int<6, false> > v1839_radr;
  sc_out<bool> v1839_re;
  sc_in< ac_int<32, true> > v1839_q;
  sc_in<bool> v1839_rrdy;
  sc_out< ac_int<6, false> > v1840_radr;
  sc_out<bool> v1840_re;
  sc_in< ac_int<32, true> > v1840_q;
  sc_in<bool> v1840_rrdy;
  sc_out< ac_int<6, false> > v1841_wadr;
  sc_out< ac_int<32, true> > v1841_d;
  sc_out<bool> v1841_we;
  sc_in<bool> v1841_wrdy;
  sc_out< ac_int<6, false> > v1842_wadr;
  sc_out< ac_int<32, true> > v1842_d;
  sc_out<bool> v1842_we;
  sc_in<bool> v1842_wrdy;
  sc_out< ac_int<6, false> > v1843_wadr;
  sc_out< ac_int<32, true> > v1843_d;
  sc_out<bool> v1843_we;
  sc_in<bool> v1843_wrdy;
  sc_out< ac_int<6, false> > v1844_wadr;
  sc_out< ac_int<32, true> > v1844_d;
  sc_out<bool> v1844_we;
  sc_in<bool> v1844_wrdy;
  sc_signal< ac_int<1, false> > mp0_0_radr, mp0_0_wadr;
  sc_signal<bool> mp0_0_re, mp0_0_we, mp0_0_rrdy, mp0_0_wrdy;
  sc_signal< ac_int<32, true> > mp0_0_q, mp0_0_d;
  AlloMemPins< ac_int<32, true>, 1, 1 > mp0_0_mem;  // replicated across clients: cannot be a port
  sc_signal< ac_int<1, false> > mp1_2_radr, mp1_2_wadr;
  sc_signal<bool> mp1_2_re, mp1_2_we, mp1_2_rrdy, mp1_2_wrdy;
  sc_signal< ac_int<32, true> > mp1_2_q, mp1_2_d;
  AlloMemPins< ac_int<32, true>, 1, 1 > mp1_2_mem;  // replicated across clients: cannot be a port
  sc_signal< ac_int<1, false> > mp2_2_radr, mp2_2_wadr;
  sc_signal<bool> mp2_2_re, mp2_2_we, mp2_2_rrdy, mp2_2_wrdy;
  sc_signal< ac_int<32, true> > mp2_2_q, mp2_2_d;
  AlloMemPins< ac_int<32, true>, 1, 1 > mp2_2_mem;  // replicated across clients: cannot be a port
  sc_signal< ac_int<1, false> > mp3_2_radr, mp3_2_wadr;
  sc_signal<bool> mp3_2_re, mp3_2_we, mp3_2_rrdy, mp3_2_wrdy;
  sc_signal< ac_int<32, true> > mp3_2_q, mp3_2_d;
  AlloMemPins< ac_int<32, true>, 1, 1 > mp3_2_mem;  // replicated across clients: cannot be a port
  sc_signal< ac_int<1, false> > mp4_2_radr, mp4_2_wadr;
  sc_signal<bool> mp4_2_re, mp4_2_we, mp4_2_rrdy, mp4_2_wrdy;
  sc_signal< ac_int<32, true> > mp4_2_q, mp4_2_d;
  AlloMemPins< ac_int<32, true>, 1, 1 > mp4_2_mem;  // replicated across clients: cannot be a port
  sc_signal< ac_int<1, false> > mp5_1_radr, mp5_1_wadr;
  sc_signal<bool> mp5_1_re, mp5_1_we, mp5_1_rrdy, mp5_1_wrdy;
  sc_signal< ac_int<32, true> > mp5_1_q, mp5_1_d;
  AlloMemPins< ac_int<32, true>, 1, 1 > mp5_1_mem;  // replicated across clients: cannot be a port
  sc_signal< ac_int<1, false> > mp6_1_radr, mp6_1_wadr;
  sc_signal<bool> mp6_1_re, mp6_1_we, mp6_1_rrdy, mp6_1_wrdy;
  sc_signal< ac_int<32, true> > mp6_1_q, mp6_1_d;
  AlloMemPins< ac_int<32, true>, 1, 1 > mp6_1_mem;  // replicated across clients: cannot be a port
  sc_signal< ac_int<1, false> > mp7_1_radr, mp7_1_wadr;
  sc_signal<bool> mp7_1_re, mp7_1_we, mp7_1_rrdy, mp7_1_wrdy;
  sc_signal< ac_int<32, true> > mp7_1_q, mp7_1_d;
  AlloMemPins< ac_int<32, true>, 1, 1 > mp7_1_mem;  // replicated across clients: cannot be a port
  sc_signal< ac_int<1, false> > mp8_1_radr, mp8_1_wadr;
  sc_signal<bool> mp8_1_re, mp8_1_we, mp8_1_rrdy, mp8_1_wrdy;
  sc_signal< ac_int<32, true> > mp8_1_q, mp8_1_d;
  AlloMemPins< ac_int<32, true>, 1, 1 > mp8_1_mem;  // replicated across clients: cannot be a port
  sc_signal< ac_int<1, false> > mp9_1_radr, mp9_1_wadr;
  sc_signal<bool> mp9_1_re, mp9_1_we, mp9_1_rrdy, mp9_1_wrdy;
  sc_signal< ac_int<32, true> > mp9_1_q, mp9_1_d;
  AlloMemPins< ac_int<32, true>, 1, 1 > mp9_1_mem;  // replicated across clients: cannot be a port
  sc_signal< ac_int<1, false> > mp10_1_radr, mp10_1_wadr;
  sc_signal<bool> mp10_1_re, mp10_1_we, mp10_1_rrdy, mp10_1_wrdy;
  sc_signal< ac_int<32, true> > mp10_1_q, mp10_1_d;
  AlloMemPins< ac_int<32, true>, 1, 1 > mp10_1_mem;  // replicated across clients: cannot be a port
  sc_signal< ac_int<1, false> > mp11_1_radr, mp11_1_wadr;
  sc_signal<bool> mp11_1_re, mp11_1_we, mp11_1_rrdy, mp11_1_wrdy;
  sc_signal< ac_int<32, true> > mp11_1_q, mp11_1_d;
  AlloMemPins< ac_int<32, true>, 1, 1 > mp11_1_mem;  // replicated across clients: cannot be a port
  sc_signal< ac_int<1, false> > mp12_1_radr, mp12_1_wadr;
  sc_signal<bool> mp12_1_re, mp12_1_we, mp12_1_rrdy, mp12_1_wrdy;
  sc_signal< ac_int<32, true> > mp12_1_q, mp12_1_d;
  AlloMemPins< ac_int<32, true>, 1, 1 > mp12_1_mem;  // replicated across clients: cannot be a port
  sc_signal< ac_int<1, false> > mp13_1_radr, mp13_1_wadr;
  sc_signal<bool> mp13_1_re, mp13_1_we, mp13_1_rrdy, mp13_1_wrdy;
  sc_signal< ac_int<32, true> > mp13_1_q, mp13_1_d;
  AlloMemPins< ac_int<32, true>, 1, 1 > mp13_1_mem;  // replicated across clients: cannot be a port
  sc_signal< ac_int<1, false> > mp14_1_radr, mp14_1_wadr;
  sc_signal<bool> mp14_1_re, mp14_1_we, mp14_1_rrdy, mp14_1_wrdy;
  sc_signal< ac_int<32, true> > mp14_1_q, mp14_1_d;
  AlloMemPins< ac_int<32, true>, 1, 1 > mp14_1_mem;  // replicated across clients: cannot be a port
  sc_signal< ac_int<1, false> > mp15_1_radr, mp15_1_wadr;
  sc_signal<bool> mp15_1_re, mp15_1_we, mp15_1_rrdy, mp15_1_wrdy;
  sc_signal< ac_int<32, true> > mp15_1_q, mp15_1_d;
  AlloMemPins< ac_int<32, true>, 1, 1 > mp15_1_mem;  // replicated across clients: cannot be a port
  sc_signal< ac_int<1, false> > mp16_1_radr, mp16_1_wadr;
  sc_signal<bool> mp16_1_re, mp16_1_we, mp16_1_rrdy, mp16_1_wrdy;
  sc_signal< ac_int<32, true> > mp16_1_q, mp16_1_d;
  AlloMemPins< ac_int<32, true>, 1, 1 > mp16_1_mem;  // replicated across clients: cannot be a port
  SC_CTOR(top) : v1845_in("v1845_in"), v1845_out("v1845_out"), v1845_fifo("v1845_fifo"), v1846_in("v1846_in"), v1846_out("v1846_out"), v1846_fifo("v1846_fifo"), v1847_in("v1847_in"), v1847_out("v1847_out"), v1847_fifo("v1847_fifo"), v1848_in("v1848_in"), v1848_out("v1848_out"), v1848_fifo("v1848_fifo"), v1849_in("v1849_in"), v1849_out("v1849_out"), v1849_fifo("v1849_fifo"), v1850_in("v1850_in"), v1850_out("v1850_out"), v1850_fifo("v1850_fifo"), v1851_in("v1851_in"), v1851_out("v1851_out"), v1851_fifo("v1851_fifo"), v1852_in("v1852_in"), v1852_out("v1852_out"), v1852_fifo("v1852_fifo"), v1853_in("v1853_in"), v1853_out("v1853_out"), v1853_fifo("v1853_fifo"), v1854_in("v1854_in"), v1854_out("v1854_out"), v1854_fifo("v1854_fifo"), v1855_in("v1855_in"), v1855_out("v1855_out"), v1855_fifo("v1855_fifo"), v1856_in("v1856_in"), v1856_out("v1856_out"), v1856_fifo("v1856_fifo"), v1857_in("v1857_in"), v1857_out("v1857_out"), v1857_fifo("v1857_fifo"), v1858_in("v1858_in"), v1858_out("v1858_out"), v1858_fifo("v1858_fifo"), v1859_in("v1859_in"), v1859_out("v1859_out"), v1859_fifo("v1859_fifo"), v1860_in("v1860_in"), v1860_out("v1860_out"), v1860_fifo("v1860_fifo"), v1861_in("v1861_in"), v1861_out("v1861_out"), v1861_fifo("v1861_fifo"), v1862_in("v1862_in"), v1862_out("v1862_out"), v1862_fifo("v1862_fifo"), v1863_in("v1863_in"), v1863_out("v1863_out"), v1863_fifo("v1863_fifo"), v1864_in("v1864_in"), v1864_out("v1864_out"), v1864_fifo("v1864_fifo"), v1865_in("v1865_in"), v1865_out("v1865_out"), v1865_fifo("v1865_fifo"), v1866_in("v1866_in"), v1866_out("v1866_out"), v1866_fifo("v1866_fifo"), v1867_in("v1867_in"), v1867_out("v1867_out"), v1867_fifo("v1867_fifo"), v1868_in("v1868_in"), v1868_out("v1868_out"), v1868_fifo("v1868_fifo"), v1869_in("v1869_in"), v1869_out("v1869_out"), v1869_fifo("v1869_fifo"), v1870_in("v1870_in"), v1870_out("v1870_out"), v1870_fifo("v1870_fifo"), v1871_in("v1871_in"), v1871_out("v1871_out"), v1871_fifo("v1871_fifo"), v1872_in("v1872_in"), v1872_out("v1872_out"), v1872_fifo("v1872_fifo"), v1873_in("v1873_in"), v1873_out("v1873_out"), v1873_fifo("v1873_fifo"), v1874_in("v1874_in"), v1874_out("v1874_out"), v1874_fifo("v1874_fifo"), v1875_in("v1875_in"), v1875_out("v1875_out"), v1875_fifo("v1875_fifo"), v1876_in("v1876_in"), v1876_out("v1876_out"), v1876_fifo("v1876_fifo"), u0("u0"), u1("u1"), u2("u2"), u3("u3"), u4("u4"), u5("u5"), u6("u6"), u7("u7"), u8("u8"), u9("u9"), u10("u10"), u11("u11"), u12("u12"), u13("u13"), u14("u14"), u15("u15"), u16("u16"), mp0_0_mem("mp0_0_mem"), mp1_2_mem("mp1_2_mem"), mp2_2_mem("mp2_2_mem"), mp3_2_mem("mp3_2_mem"), mp4_2_mem("mp4_2_mem"), mp5_1_mem("mp5_1_mem"), mp6_1_mem("mp6_1_mem"), mp7_1_mem("mp7_1_mem"), mp8_1_mem("mp8_1_mem"), mp9_1_mem("mp9_1_mem"), mp10_1_mem("mp10_1_mem"), mp11_1_mem("mp11_1_mem"), mp12_1_mem("mp12_1_mem"), mp13_1_mem("mp13_1_mem"), mp14_1_mem("mp14_1_mem"), mp15_1_mem("mp15_1_mem"), mp16_1_mem("mp16_1_mem") {
    u0.clk(clk);
    u0.rst(rst);
    u0.done(u0_done);
    u0.v0_radr(mp0_0_radr);
    u0.v0_re(mp0_0_re);
    u0.v0_q(mp0_0_q);
    u0.v0_rrdy(mp0_0_rrdy);
    u0.v1(v1854_in);
    u0.v2(v1855_in);
    u0.v3(v1858_in);
    u0.v4(v1859_in);
    u0.v5(v1846_in);
    u0.v6(v1847_in);
    u0.v7(v1850_in);
    u0.v8(v1851_in);
    u0.v9(v1861_in);
    u0.v10(v1864_in);
    u0.v11(v1865_in);
    u0.v12(v1868_in);
    u0.v13(v1869_in);
    u0.v14(v1872_in);
    u0.v15(v1873_in);
    u0.v16(v1876_in);
    u0.v17(v1853_out);
    u0.v18(v1856_out);
    u0.v19(v1857_out);
    u0.v20(v1860_out);
    u0.v21(v1862_out);
    u0.v22(v1863_out);
    u0.v23(v1866_out);
    u0.v24(v1867_out);
    u0.v25(v1875_out);
    u0.v26(v1874_out);
    u0.v27(v1871_out);
    u0.v28(v1870_out);
    u0.v29(v1845_out);
    u0.v30(v1848_out);
    u0.v31(v1849_out);
    u0.v32(v1852_out);
    u1.clk(clk);
    u1.rst(rst);
    u1.done(u1_done);
    u1.v1108_radr(v1825_radr);
    u1.v1108_re(v1825_re);
    u1.v1108_q(v1825_q);
    u1.v1108_rrdy(v1825_rrdy);
    u1.v1109_radr(v1826_radr);
    u1.v1109_re(v1826_re);
    u1.v1109_q(v1826_q);
    u1.v1109_rrdy(v1826_rrdy);
    u1.v1110_radr(mp1_2_radr);
    u1.v1110_re(mp1_2_re);
    u1.v1110_q(mp1_2_q);
    u1.v1110_rrdy(mp1_2_rrdy);
    u1.v1111(v1845_in);
    u1.v1112(v1869_out);
    u2.clk(clk);
    u2.rst(rst);
    u2.done(u2_done);
    u2.v1165_radr(v1827_radr);
    u2.v1165_re(v1827_re);
    u2.v1165_q(v1827_q);
    u2.v1165_rrdy(v1827_rrdy);
    u2.v1166_radr(v1828_radr);
    u2.v1166_re(v1828_re);
    u2.v1166_q(v1828_q);
    u2.v1166_rrdy(v1828_rrdy);
    u2.v1167_radr(mp2_2_radr);
    u2.v1167_re(mp2_2_re);
    u2.v1167_q(mp2_2_q);
    u2.v1167_rrdy(mp2_2_rrdy);
    u2.v1168(v1848_in);
    u2.v1169(v1872_out);
    u3.clk(clk);
    u3.rst(rst);
    u3.done(u3_done);
    u3.v1222_radr(v1829_radr);
    u3.v1222_re(v1829_re);
    u3.v1222_q(v1829_q);
    u3.v1222_rrdy(v1829_rrdy);
    u3.v1223_radr(v1830_radr);
    u3.v1223_re(v1830_re);
    u3.v1223_q(v1830_q);
    u3.v1223_rrdy(v1830_rrdy);
    u3.v1224_radr(mp3_2_radr);
    u3.v1224_re(mp3_2_re);
    u3.v1224_q(mp3_2_q);
    u3.v1224_rrdy(mp3_2_rrdy);
    u3.v1225(v1849_in);
    u3.v1226(v1873_out);
    u4.clk(clk);
    u4.rst(rst);
    u4.done(u4_done);
    u4.v1279_radr(v1831_radr);
    u4.v1279_re(v1831_re);
    u4.v1279_q(v1831_q);
    u4.v1279_rrdy(v1831_rrdy);
    u4.v1280_radr(v1832_radr);
    u4.v1280_re(v1832_re);
    u4.v1280_q(v1832_q);
    u4.v1280_rrdy(v1832_rrdy);
    u4.v1281_radr(mp4_2_radr);
    u4.v1281_re(mp4_2_re);
    u4.v1281_q(mp4_2_q);
    u4.v1281_rrdy(mp4_2_rrdy);
    u4.v1282(v1852_in);
    u4.v1283(v1876_out);
    u5.clk(clk);
    u5.rst(rst);
    u5.done(u5_done);
    u5.v1336_wadr(v1833_wadr);
    u5.v1336_d(v1833_d);
    u5.v1336_we(v1833_we);
    u5.v1336_wrdy(v1833_wrdy);
    u5.v1337_radr(mp5_1_radr);
    u5.v1337_re(mp5_1_re);
    u5.v1337_q(mp5_1_q);
    u5.v1337_rrdy(mp5_1_rrdy);
    u5.v1338(v1871_in);
    u5.v1339(v1847_out);
    u6.clk(clk);
    u6.rst(rst);
    u6.done(u6_done);
    u6.v1371_wadr(v1834_wadr);
    u6.v1371_d(v1834_d);
    u6.v1371_we(v1834_we);
    u6.v1371_wrdy(v1834_wrdy);
    u6.v1372_radr(mp6_1_radr);
    u6.v1372_re(mp6_1_re);
    u6.v1372_q(mp6_1_q);
    u6.v1372_rrdy(mp6_1_rrdy);
    u6.v1373(v1870_in);
    u6.v1374(v1846_out);
    u7.clk(clk);
    u7.rst(rst);
    u7.done(u7_done);
    u7.v1406_wadr(v1835_wadr);
    u7.v1406_d(v1835_d);
    u7.v1406_we(v1835_we);
    u7.v1406_wrdy(v1835_wrdy);
    u7.v1407_radr(mp7_1_radr);
    u7.v1407_re(mp7_1_re);
    u7.v1407_q(mp7_1_q);
    u7.v1407_rrdy(mp7_1_rrdy);
    u7.v1408(v1875_in);
    u7.v1409(v1851_out);
    u8.clk(clk);
    u8.rst(rst);
    u8.done(u8_done);
    u8.v1441_wadr(v1836_wadr);
    u8.v1441_d(v1836_d);
    u8.v1441_we(v1836_we);
    u8.v1441_wrdy(v1836_wrdy);
    u8.v1442_radr(mp8_1_radr);
    u8.v1442_re(mp8_1_re);
    u8.v1442_q(mp8_1_q);
    u8.v1442_rrdy(mp8_1_rrdy);
    u8.v1443(v1874_in);
    u8.v1444(v1850_out);
    u9.clk(clk);
    u9.rst(rst);
    u9.done(u9_done);
    u9.v1476_radr(v1837_radr);
    u9.v1476_re(v1837_re);
    u9.v1476_q(v1837_q);
    u9.v1476_rrdy(v1837_rrdy);
    u9.v1477_radr(mp9_1_radr);
    u9.v1477_re(mp9_1_re);
    u9.v1477_q(mp9_1_q);
    u9.v1477_rrdy(mp9_1_rrdy);
    u9.v1478(v1853_in);
    u9.v1479(v1861_out);
    u10.clk(clk);
    u10.rst(rst);
    u10.done(u10_done);
    u10.v1528_radr(v1838_radr);
    u10.v1528_re(v1838_re);
    u10.v1528_q(v1838_q);
    u10.v1528_rrdy(v1838_rrdy);
    u10.v1529_radr(mp10_1_radr);
    u10.v1529_re(mp10_1_re);
    u10.v1529_q(mp10_1_q);
    u10.v1529_rrdy(mp10_1_rrdy);
    u10.v1530(v1856_in);
    u10.v1531(v1864_out);
    u11.clk(clk);
    u11.rst(rst);
    u11.done(u11_done);
    u11.v1580_radr(v1839_radr);
    u11.v1580_re(v1839_re);
    u11.v1580_q(v1839_q);
    u11.v1580_rrdy(v1839_rrdy);
    u11.v1581_radr(mp11_1_radr);
    u11.v1581_re(mp11_1_re);
    u11.v1581_q(mp11_1_q);
    u11.v1581_rrdy(mp11_1_rrdy);
    u11.v1582(v1857_in);
    u11.v1583(v1865_out);
    u12.clk(clk);
    u12.rst(rst);
    u12.done(u12_done);
    u12.v1632_radr(v1840_radr);
    u12.v1632_re(v1840_re);
    u12.v1632_q(v1840_q);
    u12.v1632_rrdy(v1840_rrdy);
    u12.v1633_radr(mp12_1_radr);
    u12.v1633_re(mp12_1_re);
    u12.v1633_q(mp12_1_q);
    u12.v1633_rrdy(mp12_1_rrdy);
    u12.v1634(v1860_in);
    u12.v1635(v1868_out);
    u13.clk(clk);
    u13.rst(rst);
    u13.done(u13_done);
    u13.v1684_wadr(v1841_wadr);
    u13.v1684_d(v1841_d);
    u13.v1684_we(v1841_we);
    u13.v1684_wrdy(v1841_wrdy);
    u13.v1685_radr(mp13_1_radr);
    u13.v1685_re(mp13_1_re);
    u13.v1685_q(mp13_1_q);
    u13.v1685_rrdy(mp13_1_rrdy);
    u13.v1686(v1863_in);
    u13.v1687(v1855_out);
    u14.clk(clk);
    u14.rst(rst);
    u14.done(u14_done);
    u14.v1719_wadr(v1842_wadr);
    u14.v1719_d(v1842_d);
    u14.v1719_we(v1842_we);
    u14.v1719_wrdy(v1842_wrdy);
    u14.v1720_radr(mp14_1_radr);
    u14.v1720_re(mp14_1_re);
    u14.v1720_q(mp14_1_q);
    u14.v1720_rrdy(mp14_1_rrdy);
    u14.v1721(v1862_in);
    u14.v1722(v1854_out);
    u15.clk(clk);
    u15.rst(rst);
    u15.done(u15_done);
    u15.v1754_wadr(v1843_wadr);
    u15.v1754_d(v1843_d);
    u15.v1754_we(v1843_we);
    u15.v1754_wrdy(v1843_wrdy);
    u15.v1755_radr(mp15_1_radr);
    u15.v1755_re(mp15_1_re);
    u15.v1755_q(mp15_1_q);
    u15.v1755_rrdy(mp15_1_rrdy);
    u15.v1756(v1867_in);
    u15.v1757(v1859_out);
    u16.clk(clk);
    u16.rst(rst);
    u16.done(u16_done);
    u16.v1789_wadr(v1844_wadr);
    u16.v1789_d(v1844_d);
    u16.v1789_we(v1844_we);
    u16.v1789_wrdy(v1844_wrdy);
    u16.v1790_radr(mp16_1_radr);
    u16.v1790_re(mp16_1_re);
    u16.v1790_q(mp16_1_q);
    u16.v1790_rrdy(mp16_1_rrdy);
    u16.v1791(v1866_in);
    u16.v1792(v1858_out);
    v1845_fifo.clk(clk);
    v1845_fifo.rst(rst);
    v1845_fifo.enq(v1845_in);
    v1845_fifo.deq(v1845_out);
    v1846_fifo.clk(clk);
    v1846_fifo.rst(rst);
    v1846_fifo.enq(v1846_in);
    v1846_fifo.deq(v1846_out);
    v1847_fifo.clk(clk);
    v1847_fifo.rst(rst);
    v1847_fifo.enq(v1847_in);
    v1847_fifo.deq(v1847_out);
    v1848_fifo.clk(clk);
    v1848_fifo.rst(rst);
    v1848_fifo.enq(v1848_in);
    v1848_fifo.deq(v1848_out);
    v1849_fifo.clk(clk);
    v1849_fifo.rst(rst);
    v1849_fifo.enq(v1849_in);
    v1849_fifo.deq(v1849_out);
    v1850_fifo.clk(clk);
    v1850_fifo.rst(rst);
    v1850_fifo.enq(v1850_in);
    v1850_fifo.deq(v1850_out);
    v1851_fifo.clk(clk);
    v1851_fifo.rst(rst);
    v1851_fifo.enq(v1851_in);
    v1851_fifo.deq(v1851_out);
    v1852_fifo.clk(clk);
    v1852_fifo.rst(rst);
    v1852_fifo.enq(v1852_in);
    v1852_fifo.deq(v1852_out);
    v1853_fifo.clk(clk);
    v1853_fifo.rst(rst);
    v1853_fifo.enq(v1853_in);
    v1853_fifo.deq(v1853_out);
    v1854_fifo.clk(clk);
    v1854_fifo.rst(rst);
    v1854_fifo.enq(v1854_in);
    v1854_fifo.deq(v1854_out);
    v1855_fifo.clk(clk);
    v1855_fifo.rst(rst);
    v1855_fifo.enq(v1855_in);
    v1855_fifo.deq(v1855_out);
    v1856_fifo.clk(clk);
    v1856_fifo.rst(rst);
    v1856_fifo.enq(v1856_in);
    v1856_fifo.deq(v1856_out);
    v1857_fifo.clk(clk);
    v1857_fifo.rst(rst);
    v1857_fifo.enq(v1857_in);
    v1857_fifo.deq(v1857_out);
    v1858_fifo.clk(clk);
    v1858_fifo.rst(rst);
    v1858_fifo.enq(v1858_in);
    v1858_fifo.deq(v1858_out);
    v1859_fifo.clk(clk);
    v1859_fifo.rst(rst);
    v1859_fifo.enq(v1859_in);
    v1859_fifo.deq(v1859_out);
    v1860_fifo.clk(clk);
    v1860_fifo.rst(rst);
    v1860_fifo.enq(v1860_in);
    v1860_fifo.deq(v1860_out);
    v1861_fifo.clk(clk);
    v1861_fifo.rst(rst);
    v1861_fifo.enq(v1861_in);
    v1861_fifo.deq(v1861_out);
    v1862_fifo.clk(clk);
    v1862_fifo.rst(rst);
    v1862_fifo.enq(v1862_in);
    v1862_fifo.deq(v1862_out);
    v1863_fifo.clk(clk);
    v1863_fifo.rst(rst);
    v1863_fifo.enq(v1863_in);
    v1863_fifo.deq(v1863_out);
    v1864_fifo.clk(clk);
    v1864_fifo.rst(rst);
    v1864_fifo.enq(v1864_in);
    v1864_fifo.deq(v1864_out);
    v1865_fifo.clk(clk);
    v1865_fifo.rst(rst);
    v1865_fifo.enq(v1865_in);
    v1865_fifo.deq(v1865_out);
    v1866_fifo.clk(clk);
    v1866_fifo.rst(rst);
    v1866_fifo.enq(v1866_in);
    v1866_fifo.deq(v1866_out);
    v1867_fifo.clk(clk);
    v1867_fifo.rst(rst);
    v1867_fifo.enq(v1867_in);
    v1867_fifo.deq(v1867_out);
    v1868_fifo.clk(clk);
    v1868_fifo.rst(rst);
    v1868_fifo.enq(v1868_in);
    v1868_fifo.deq(v1868_out);
    v1869_fifo.clk(clk);
    v1869_fifo.rst(rst);
    v1869_fifo.enq(v1869_in);
    v1869_fifo.deq(v1869_out);
    v1870_fifo.clk(clk);
    v1870_fifo.rst(rst);
    v1870_fifo.enq(v1870_in);
    v1870_fifo.deq(v1870_out);
    v1871_fifo.clk(clk);
    v1871_fifo.rst(rst);
    v1871_fifo.enq(v1871_in);
    v1871_fifo.deq(v1871_out);
    v1872_fifo.clk(clk);
    v1872_fifo.rst(rst);
    v1872_fifo.enq(v1872_in);
    v1872_fifo.deq(v1872_out);
    v1873_fifo.clk(clk);
    v1873_fifo.rst(rst);
    v1873_fifo.enq(v1873_in);
    v1873_fifo.deq(v1873_out);
    v1874_fifo.clk(clk);
    v1874_fifo.rst(rst);
    v1874_fifo.enq(v1874_in);
    v1874_fifo.deq(v1874_out);
    v1875_fifo.clk(clk);
    v1875_fifo.rst(rst);
    v1875_fifo.enq(v1875_in);
    v1875_fifo.deq(v1875_out);
    v1876_fifo.clk(clk);
    v1876_fifo.rst(rst);
    v1876_fifo.enq(v1876_in);
    v1876_fifo.deq(v1876_out);
    mp0_0_mem.clk(clk);
    mp0_0_mem.rst(rst);
    mp0_0_mem.radr(mp0_0_radr);
    mp0_0_mem.re(mp0_0_re);
    mp0_0_mem.q(mp0_0_q);
    mp0_0_mem.rrdy(mp0_0_rrdy);
    mp0_0_mem.wadr(mp0_0_wadr);
    mp0_0_mem.d(mp0_0_d);
    mp0_0_mem.we(mp0_0_we);
    mp0_0_mem.wrdy(mp0_0_wrdy);
    mp1_2_mem.clk(clk);
    mp1_2_mem.rst(rst);
    mp1_2_mem.radr(mp1_2_radr);
    mp1_2_mem.re(mp1_2_re);
    mp1_2_mem.q(mp1_2_q);
    mp1_2_mem.rrdy(mp1_2_rrdy);
    mp1_2_mem.wadr(mp1_2_wadr);
    mp1_2_mem.d(mp1_2_d);
    mp1_2_mem.we(mp1_2_we);
    mp1_2_mem.wrdy(mp1_2_wrdy);
    mp2_2_mem.clk(clk);
    mp2_2_mem.rst(rst);
    mp2_2_mem.radr(mp2_2_radr);
    mp2_2_mem.re(mp2_2_re);
    mp2_2_mem.q(mp2_2_q);
    mp2_2_mem.rrdy(mp2_2_rrdy);
    mp2_2_mem.wadr(mp2_2_wadr);
    mp2_2_mem.d(mp2_2_d);
    mp2_2_mem.we(mp2_2_we);
    mp2_2_mem.wrdy(mp2_2_wrdy);
    mp3_2_mem.clk(clk);
    mp3_2_mem.rst(rst);
    mp3_2_mem.radr(mp3_2_radr);
    mp3_2_mem.re(mp3_2_re);
    mp3_2_mem.q(mp3_2_q);
    mp3_2_mem.rrdy(mp3_2_rrdy);
    mp3_2_mem.wadr(mp3_2_wadr);
    mp3_2_mem.d(mp3_2_d);
    mp3_2_mem.we(mp3_2_we);
    mp3_2_mem.wrdy(mp3_2_wrdy);
    mp4_2_mem.clk(clk);
    mp4_2_mem.rst(rst);
    mp4_2_mem.radr(mp4_2_radr);
    mp4_2_mem.re(mp4_2_re);
    mp4_2_mem.q(mp4_2_q);
    mp4_2_mem.rrdy(mp4_2_rrdy);
    mp4_2_mem.wadr(mp4_2_wadr);
    mp4_2_mem.d(mp4_2_d);
    mp4_2_mem.we(mp4_2_we);
    mp4_2_mem.wrdy(mp4_2_wrdy);
    mp5_1_mem.clk(clk);
    mp5_1_mem.rst(rst);
    mp5_1_mem.radr(mp5_1_radr);
    mp5_1_mem.re(mp5_1_re);
    mp5_1_mem.q(mp5_1_q);
    mp5_1_mem.rrdy(mp5_1_rrdy);
    mp5_1_mem.wadr(mp5_1_wadr);
    mp5_1_mem.d(mp5_1_d);
    mp5_1_mem.we(mp5_1_we);
    mp5_1_mem.wrdy(mp5_1_wrdy);
    mp6_1_mem.clk(clk);
    mp6_1_mem.rst(rst);
    mp6_1_mem.radr(mp6_1_radr);
    mp6_1_mem.re(mp6_1_re);
    mp6_1_mem.q(mp6_1_q);
    mp6_1_mem.rrdy(mp6_1_rrdy);
    mp6_1_mem.wadr(mp6_1_wadr);
    mp6_1_mem.d(mp6_1_d);
    mp6_1_mem.we(mp6_1_we);
    mp6_1_mem.wrdy(mp6_1_wrdy);
    mp7_1_mem.clk(clk);
    mp7_1_mem.rst(rst);
    mp7_1_mem.radr(mp7_1_radr);
    mp7_1_mem.re(mp7_1_re);
    mp7_1_mem.q(mp7_1_q);
    mp7_1_mem.rrdy(mp7_1_rrdy);
    mp7_1_mem.wadr(mp7_1_wadr);
    mp7_1_mem.d(mp7_1_d);
    mp7_1_mem.we(mp7_1_we);
    mp7_1_mem.wrdy(mp7_1_wrdy);
    mp8_1_mem.clk(clk);
    mp8_1_mem.rst(rst);
    mp8_1_mem.radr(mp8_1_radr);
    mp8_1_mem.re(mp8_1_re);
    mp8_1_mem.q(mp8_1_q);
    mp8_1_mem.rrdy(mp8_1_rrdy);
    mp8_1_mem.wadr(mp8_1_wadr);
    mp8_1_mem.d(mp8_1_d);
    mp8_1_mem.we(mp8_1_we);
    mp8_1_mem.wrdy(mp8_1_wrdy);
    mp9_1_mem.clk(clk);
    mp9_1_mem.rst(rst);
    mp9_1_mem.radr(mp9_1_radr);
    mp9_1_mem.re(mp9_1_re);
    mp9_1_mem.q(mp9_1_q);
    mp9_1_mem.rrdy(mp9_1_rrdy);
    mp9_1_mem.wadr(mp9_1_wadr);
    mp9_1_mem.d(mp9_1_d);
    mp9_1_mem.we(mp9_1_we);
    mp9_1_mem.wrdy(mp9_1_wrdy);
    mp10_1_mem.clk(clk);
    mp10_1_mem.rst(rst);
    mp10_1_mem.radr(mp10_1_radr);
    mp10_1_mem.re(mp10_1_re);
    mp10_1_mem.q(mp10_1_q);
    mp10_1_mem.rrdy(mp10_1_rrdy);
    mp10_1_mem.wadr(mp10_1_wadr);
    mp10_1_mem.d(mp10_1_d);
    mp10_1_mem.we(mp10_1_we);
    mp10_1_mem.wrdy(mp10_1_wrdy);
    mp11_1_mem.clk(clk);
    mp11_1_mem.rst(rst);
    mp11_1_mem.radr(mp11_1_radr);
    mp11_1_mem.re(mp11_1_re);
    mp11_1_mem.q(mp11_1_q);
    mp11_1_mem.rrdy(mp11_1_rrdy);
    mp11_1_mem.wadr(mp11_1_wadr);
    mp11_1_mem.d(mp11_1_d);
    mp11_1_mem.we(mp11_1_we);
    mp11_1_mem.wrdy(mp11_1_wrdy);
    mp12_1_mem.clk(clk);
    mp12_1_mem.rst(rst);
    mp12_1_mem.radr(mp12_1_radr);
    mp12_1_mem.re(mp12_1_re);
    mp12_1_mem.q(mp12_1_q);
    mp12_1_mem.rrdy(mp12_1_rrdy);
    mp12_1_mem.wadr(mp12_1_wadr);
    mp12_1_mem.d(mp12_1_d);
    mp12_1_mem.we(mp12_1_we);
    mp12_1_mem.wrdy(mp12_1_wrdy);
    mp13_1_mem.clk(clk);
    mp13_1_mem.rst(rst);
    mp13_1_mem.radr(mp13_1_radr);
    mp13_1_mem.re(mp13_1_re);
    mp13_1_mem.q(mp13_1_q);
    mp13_1_mem.rrdy(mp13_1_rrdy);
    mp13_1_mem.wadr(mp13_1_wadr);
    mp13_1_mem.d(mp13_1_d);
    mp13_1_mem.we(mp13_1_we);
    mp13_1_mem.wrdy(mp13_1_wrdy);
    mp14_1_mem.clk(clk);
    mp14_1_mem.rst(rst);
    mp14_1_mem.radr(mp14_1_radr);
    mp14_1_mem.re(mp14_1_re);
    mp14_1_mem.q(mp14_1_q);
    mp14_1_mem.rrdy(mp14_1_rrdy);
    mp14_1_mem.wadr(mp14_1_wadr);
    mp14_1_mem.d(mp14_1_d);
    mp14_1_mem.we(mp14_1_we);
    mp14_1_mem.wrdy(mp14_1_wrdy);
    mp15_1_mem.clk(clk);
    mp15_1_mem.rst(rst);
    mp15_1_mem.radr(mp15_1_radr);
    mp15_1_mem.re(mp15_1_re);
    mp15_1_mem.q(mp15_1_q);
    mp15_1_mem.rrdy(mp15_1_rrdy);
    mp15_1_mem.wadr(mp15_1_wadr);
    mp15_1_mem.d(mp15_1_d);
    mp15_1_mem.we(mp15_1_we);
    mp15_1_mem.wrdy(mp15_1_wrdy);
    mp16_1_mem.clk(clk);
    mp16_1_mem.rst(rst);
    mp16_1_mem.radr(mp16_1_radr);
    mp16_1_mem.re(mp16_1_re);
    mp16_1_mem.q(mp16_1_q);
    mp16_1_mem.rrdy(mp16_1_rrdy);
    mp16_1_mem.wadr(mp16_1_wadr);
    mp16_1_mem.d(mp16_1_d);
    mp16_1_mem.we(mp16_1_we);
    mp16_1_mem.wrdy(mp16_1_wrdy);
    SC_METHOD(_agg_done); sensitive << u0_done << u1_done << u2_done << u3_done << u4_done << u5_done << u6_done << u7_done << u8_done << u9_done << u10_done << u11_done << u12_done << u13_done << u14_done << u15_done << u16_done;
  }
  void _agg_done() { done.write(u0_done.read() && u1_done.read() && u2_done.read() && u3_done.read() && u4_done.read() && u5_done.read() && u6_done.read() && u7_done.read() && u8_done.read() && u9_done.read() && u10_done.read() && u11_done.read() && u12_done.read() && u13_done.read() && u14_done.read() && u15_done.read() && u16_done.read()); }
};

SC_MODULE(tb) {
  sc_clock clk;
  sc_signal<bool> rst;
  top dut;
  sc_signal<bool> done_sig;  // DUT completion (polled by sc_main)
  sc_signal< ac_int<6, false> > mp1_0_radr, mp1_0_wadr;
  sc_signal<bool> mp1_0_re, mp1_0_we, mp1_0_rrdy, mp1_0_wrdy;
  sc_signal< half > mp1_0_q, mp1_0_d;
  AlloMemPins< half, 55, 6 > mp1_0_mem;
  sc_signal< ac_int<6, false> > mp1_1_radr, mp1_1_wadr;
  sc_signal<bool> mp1_1_re, mp1_1_we, mp1_1_rrdy, mp1_1_wrdy;
  sc_signal< ac_int<32, true> > mp1_1_q, mp1_1_d;
  AlloMemPins< ac_int<32, true>, 55, 6 > mp1_1_mem;
  sc_signal< ac_int<6, false> > mp2_0_radr, mp2_0_wadr;
  sc_signal<bool> mp2_0_re, mp2_0_we, mp2_0_rrdy, mp2_0_wrdy;
  sc_signal< half > mp2_0_q, mp2_0_d;
  AlloMemPins< half, 55, 6 > mp2_0_mem;
  sc_signal< ac_int<6, false> > mp2_1_radr, mp2_1_wadr;
  sc_signal<bool> mp2_1_re, mp2_1_we, mp2_1_rrdy, mp2_1_wrdy;
  sc_signal< ac_int<32, true> > mp2_1_q, mp2_1_d;
  AlloMemPins< ac_int<32, true>, 55, 6 > mp2_1_mem;
  sc_signal< ac_int<6, false> > mp3_0_radr, mp3_0_wadr;
  sc_signal<bool> mp3_0_re, mp3_0_we, mp3_0_rrdy, mp3_0_wrdy;
  sc_signal< half > mp3_0_q, mp3_0_d;
  AlloMemPins< half, 55, 6 > mp3_0_mem;
  sc_signal< ac_int<6, false> > mp3_1_radr, mp3_1_wadr;
  sc_signal<bool> mp3_1_re, mp3_1_we, mp3_1_rrdy, mp3_1_wrdy;
  sc_signal< ac_int<32, true> > mp3_1_q, mp3_1_d;
  AlloMemPins< ac_int<32, true>, 55, 6 > mp3_1_mem;
  sc_signal< ac_int<6, false> > mp4_0_radr, mp4_0_wadr;
  sc_signal<bool> mp4_0_re, mp4_0_we, mp4_0_rrdy, mp4_0_wrdy;
  sc_signal< half > mp4_0_q, mp4_0_d;
  AlloMemPins< half, 55, 6 > mp4_0_mem;
  sc_signal< ac_int<6, false> > mp4_1_radr, mp4_1_wadr;
  sc_signal<bool> mp4_1_re, mp4_1_we, mp4_1_rrdy, mp4_1_wrdy;
  sc_signal< ac_int<32, true> > mp4_1_q, mp4_1_d;
  AlloMemPins< ac_int<32, true>, 55, 6 > mp4_1_mem;
  sc_signal< ac_int<6, false> > mp5_0_radr, mp5_0_wadr;
  sc_signal<bool> mp5_0_re, mp5_0_we, mp5_0_rrdy, mp5_0_wrdy;
  sc_signal< half > mp5_0_q, mp5_0_d;
  AlloMemPins< half, 55, 6 > mp5_0_mem;
  sc_signal< ac_int<6, false> > mp6_0_radr, mp6_0_wadr;
  sc_signal<bool> mp6_0_re, mp6_0_we, mp6_0_rrdy, mp6_0_wrdy;
  sc_signal< half > mp6_0_q, mp6_0_d;
  AlloMemPins< half, 55, 6 > mp6_0_mem;
  sc_signal< ac_int<6, false> > mp7_0_radr, mp7_0_wadr;
  sc_signal<bool> mp7_0_re, mp7_0_we, mp7_0_rrdy, mp7_0_wrdy;
  sc_signal< half > mp7_0_q, mp7_0_d;
  AlloMemPins< half, 55, 6 > mp7_0_mem;
  sc_signal< ac_int<6, false> > mp8_0_radr, mp8_0_wadr;
  sc_signal<bool> mp8_0_re, mp8_0_we, mp8_0_rrdy, mp8_0_wrdy;
  sc_signal< half > mp8_0_q, mp8_0_d;
  AlloMemPins< half, 55, 6 > mp8_0_mem;
  sc_signal< ac_int<6, false> > mp9_0_radr, mp9_0_wadr;
  sc_signal<bool> mp9_0_re, mp9_0_we, mp9_0_rrdy, mp9_0_wrdy;
  sc_signal< ac_int<32, true> > mp9_0_q, mp9_0_d;
  AlloMemPins< ac_int<32, true>, 55, 6 > mp9_0_mem;
  sc_signal< ac_int<6, false> > mp10_0_radr, mp10_0_wadr;
  sc_signal<bool> mp10_0_re, mp10_0_we, mp10_0_rrdy, mp10_0_wrdy;
  sc_signal< ac_int<32, true> > mp10_0_q, mp10_0_d;
  AlloMemPins< ac_int<32, true>, 55, 6 > mp10_0_mem;
  sc_signal< ac_int<6, false> > mp11_0_radr, mp11_0_wadr;
  sc_signal<bool> mp11_0_re, mp11_0_we, mp11_0_rrdy, mp11_0_wrdy;
  sc_signal< ac_int<32, true> > mp11_0_q, mp11_0_d;
  AlloMemPins< ac_int<32, true>, 55, 6 > mp11_0_mem;
  sc_signal< ac_int<6, false> > mp12_0_radr, mp12_0_wadr;
  sc_signal<bool> mp12_0_re, mp12_0_we, mp12_0_rrdy, mp12_0_wrdy;
  sc_signal< ac_int<32, true> > mp12_0_q, mp12_0_d;
  AlloMemPins< ac_int<32, true>, 55, 6 > mp12_0_mem;
  sc_signal< ac_int<6, false> > mp13_0_radr, mp13_0_wadr;
  sc_signal<bool> mp13_0_re, mp13_0_we, mp13_0_rrdy, mp13_0_wrdy;
  sc_signal< ac_int<32, true> > mp13_0_q, mp13_0_d;
  AlloMemPins< ac_int<32, true>, 55, 6 > mp13_0_mem;
  sc_signal< ac_int<6, false> > mp14_0_radr, mp14_0_wadr;
  sc_signal<bool> mp14_0_re, mp14_0_we, mp14_0_rrdy, mp14_0_wrdy;
  sc_signal< ac_int<32, true> > mp14_0_q, mp14_0_d;
  AlloMemPins< ac_int<32, true>, 55, 6 > mp14_0_mem;
  sc_signal< ac_int<6, false> > mp15_0_radr, mp15_0_wadr;
  sc_signal<bool> mp15_0_re, mp15_0_we, mp15_0_rrdy, mp15_0_wrdy;
  sc_signal< ac_int<32, true> > mp15_0_q, mp15_0_d;
  AlloMemPins< ac_int<32, true>, 55, 6 > mp15_0_mem;
  sc_signal< ac_int<6, false> > mp16_0_radr, mp16_0_wadr;
  sc_signal<bool> mp16_0_re, mp16_0_we, mp16_0_rrdy, mp16_0_wrdy;
  sc_signal< ac_int<32, true> > mp16_0_q, mp16_0_d;
  AlloMemPins< ac_int<32, true>, 55, 6 > mp16_0_mem;
  SC_HAS_PROCESS(tb);
  tb(sc_module_name n) : sc_module(n), clk("clk", 1, SC_NS), dut("dut"), mp1_0_mem("mp1_0_mem"), mp1_1_mem("mp1_1_mem"), mp2_0_mem("mp2_0_mem"), mp2_1_mem("mp2_1_mem"), mp3_0_mem("mp3_0_mem"), mp3_1_mem("mp3_1_mem"), mp4_0_mem("mp4_0_mem"), mp4_1_mem("mp4_1_mem"), mp5_0_mem("mp5_0_mem"), mp6_0_mem("mp6_0_mem"), mp7_0_mem("mp7_0_mem"), mp8_0_mem("mp8_0_mem"), mp9_0_mem("mp9_0_mem"), mp10_0_mem("mp10_0_mem"), mp11_0_mem("mp11_0_mem"), mp12_0_mem("mp12_0_mem"), mp13_0_mem("mp13_0_mem"), mp14_0_mem("mp14_0_mem"), mp15_0_mem("mp15_0_mem"), mp16_0_mem("mp16_0_mem") {
    dut.clk(clk); dut.rst(rst); dut.done(done_sig);
    mp1_0_mem.clk(clk); mp1_0_mem.rst(rst);
    dut.v1825_radr(mp1_0_radr); mp1_0_mem.radr(mp1_0_radr);
    dut.v1825_re(mp1_0_re); mp1_0_mem.re(mp1_0_re);
    dut.v1825_q(mp1_0_q); mp1_0_mem.q(mp1_0_q);
    dut.v1825_rrdy(mp1_0_rrdy); mp1_0_mem.rrdy(mp1_0_rrdy);
    mp1_0_mem.wadr(mp1_0_wadr);
    mp1_0_mem.d(mp1_0_d);
    mp1_0_mem.we(mp1_0_we);
    mp1_0_mem.wrdy(mp1_0_wrdy);
    mp1_1_mem.clk(clk); mp1_1_mem.rst(rst);
    dut.v1826_radr(mp1_1_radr); mp1_1_mem.radr(mp1_1_radr);
    dut.v1826_re(mp1_1_re); mp1_1_mem.re(mp1_1_re);
    dut.v1826_q(mp1_1_q); mp1_1_mem.q(mp1_1_q);
    dut.v1826_rrdy(mp1_1_rrdy); mp1_1_mem.rrdy(mp1_1_rrdy);
    mp1_1_mem.wadr(mp1_1_wadr);
    mp1_1_mem.d(mp1_1_d);
    mp1_1_mem.we(mp1_1_we);
    mp1_1_mem.wrdy(mp1_1_wrdy);
    mp2_0_mem.clk(clk); mp2_0_mem.rst(rst);
    dut.v1827_radr(mp2_0_radr); mp2_0_mem.radr(mp2_0_radr);
    dut.v1827_re(mp2_0_re); mp2_0_mem.re(mp2_0_re);
    dut.v1827_q(mp2_0_q); mp2_0_mem.q(mp2_0_q);
    dut.v1827_rrdy(mp2_0_rrdy); mp2_0_mem.rrdy(mp2_0_rrdy);
    mp2_0_mem.wadr(mp2_0_wadr);
    mp2_0_mem.d(mp2_0_d);
    mp2_0_mem.we(mp2_0_we);
    mp2_0_mem.wrdy(mp2_0_wrdy);
    mp2_1_mem.clk(clk); mp2_1_mem.rst(rst);
    dut.v1828_radr(mp2_1_radr); mp2_1_mem.radr(mp2_1_radr);
    dut.v1828_re(mp2_1_re); mp2_1_mem.re(mp2_1_re);
    dut.v1828_q(mp2_1_q); mp2_1_mem.q(mp2_1_q);
    dut.v1828_rrdy(mp2_1_rrdy); mp2_1_mem.rrdy(mp2_1_rrdy);
    mp2_1_mem.wadr(mp2_1_wadr);
    mp2_1_mem.d(mp2_1_d);
    mp2_1_mem.we(mp2_1_we);
    mp2_1_mem.wrdy(mp2_1_wrdy);
    mp3_0_mem.clk(clk); mp3_0_mem.rst(rst);
    dut.v1829_radr(mp3_0_radr); mp3_0_mem.radr(mp3_0_radr);
    dut.v1829_re(mp3_0_re); mp3_0_mem.re(mp3_0_re);
    dut.v1829_q(mp3_0_q); mp3_0_mem.q(mp3_0_q);
    dut.v1829_rrdy(mp3_0_rrdy); mp3_0_mem.rrdy(mp3_0_rrdy);
    mp3_0_mem.wadr(mp3_0_wadr);
    mp3_0_mem.d(mp3_0_d);
    mp3_0_mem.we(mp3_0_we);
    mp3_0_mem.wrdy(mp3_0_wrdy);
    mp3_1_mem.clk(clk); mp3_1_mem.rst(rst);
    dut.v1830_radr(mp3_1_radr); mp3_1_mem.radr(mp3_1_radr);
    dut.v1830_re(mp3_1_re); mp3_1_mem.re(mp3_1_re);
    dut.v1830_q(mp3_1_q); mp3_1_mem.q(mp3_1_q);
    dut.v1830_rrdy(mp3_1_rrdy); mp3_1_mem.rrdy(mp3_1_rrdy);
    mp3_1_mem.wadr(mp3_1_wadr);
    mp3_1_mem.d(mp3_1_d);
    mp3_1_mem.we(mp3_1_we);
    mp3_1_mem.wrdy(mp3_1_wrdy);
    mp4_0_mem.clk(clk); mp4_0_mem.rst(rst);
    dut.v1831_radr(mp4_0_radr); mp4_0_mem.radr(mp4_0_radr);
    dut.v1831_re(mp4_0_re); mp4_0_mem.re(mp4_0_re);
    dut.v1831_q(mp4_0_q); mp4_0_mem.q(mp4_0_q);
    dut.v1831_rrdy(mp4_0_rrdy); mp4_0_mem.rrdy(mp4_0_rrdy);
    mp4_0_mem.wadr(mp4_0_wadr);
    mp4_0_mem.d(mp4_0_d);
    mp4_0_mem.we(mp4_0_we);
    mp4_0_mem.wrdy(mp4_0_wrdy);
    mp4_1_mem.clk(clk); mp4_1_mem.rst(rst);
    dut.v1832_radr(mp4_1_radr); mp4_1_mem.radr(mp4_1_radr);
    dut.v1832_re(mp4_1_re); mp4_1_mem.re(mp4_1_re);
    dut.v1832_q(mp4_1_q); mp4_1_mem.q(mp4_1_q);
    dut.v1832_rrdy(mp4_1_rrdy); mp4_1_mem.rrdy(mp4_1_rrdy);
    mp4_1_mem.wadr(mp4_1_wadr);
    mp4_1_mem.d(mp4_1_d);
    mp4_1_mem.we(mp4_1_we);
    mp4_1_mem.wrdy(mp4_1_wrdy);
    mp5_0_mem.clk(clk); mp5_0_mem.rst(rst);
    mp5_0_mem.radr(mp5_0_radr);
    mp5_0_mem.re(mp5_0_re);
    mp5_0_mem.q(mp5_0_q);
    mp5_0_mem.rrdy(mp5_0_rrdy);
    dut.v1833_wadr(mp5_0_wadr); mp5_0_mem.wadr(mp5_0_wadr);
    dut.v1833_d(mp5_0_d); mp5_0_mem.d(mp5_0_d);
    dut.v1833_we(mp5_0_we); mp5_0_mem.we(mp5_0_we);
    dut.v1833_wrdy(mp5_0_wrdy); mp5_0_mem.wrdy(mp5_0_wrdy);
    mp6_0_mem.clk(clk); mp6_0_mem.rst(rst);
    mp6_0_mem.radr(mp6_0_radr);
    mp6_0_mem.re(mp6_0_re);
    mp6_0_mem.q(mp6_0_q);
    mp6_0_mem.rrdy(mp6_0_rrdy);
    dut.v1834_wadr(mp6_0_wadr); mp6_0_mem.wadr(mp6_0_wadr);
    dut.v1834_d(mp6_0_d); mp6_0_mem.d(mp6_0_d);
    dut.v1834_we(mp6_0_we); mp6_0_mem.we(mp6_0_we);
    dut.v1834_wrdy(mp6_0_wrdy); mp6_0_mem.wrdy(mp6_0_wrdy);
    mp7_0_mem.clk(clk); mp7_0_mem.rst(rst);
    mp7_0_mem.radr(mp7_0_radr);
    mp7_0_mem.re(mp7_0_re);
    mp7_0_mem.q(mp7_0_q);
    mp7_0_mem.rrdy(mp7_0_rrdy);
    dut.v1835_wadr(mp7_0_wadr); mp7_0_mem.wadr(mp7_0_wadr);
    dut.v1835_d(mp7_0_d); mp7_0_mem.d(mp7_0_d);
    dut.v1835_we(mp7_0_we); mp7_0_mem.we(mp7_0_we);
    dut.v1835_wrdy(mp7_0_wrdy); mp7_0_mem.wrdy(mp7_0_wrdy);
    mp8_0_mem.clk(clk); mp8_0_mem.rst(rst);
    mp8_0_mem.radr(mp8_0_radr);
    mp8_0_mem.re(mp8_0_re);
    mp8_0_mem.q(mp8_0_q);
    mp8_0_mem.rrdy(mp8_0_rrdy);
    dut.v1836_wadr(mp8_0_wadr); mp8_0_mem.wadr(mp8_0_wadr);
    dut.v1836_d(mp8_0_d); mp8_0_mem.d(mp8_0_d);
    dut.v1836_we(mp8_0_we); mp8_0_mem.we(mp8_0_we);
    dut.v1836_wrdy(mp8_0_wrdy); mp8_0_mem.wrdy(mp8_0_wrdy);
    mp9_0_mem.clk(clk); mp9_0_mem.rst(rst);
    dut.v1837_radr(mp9_0_radr); mp9_0_mem.radr(mp9_0_radr);
    dut.v1837_re(mp9_0_re); mp9_0_mem.re(mp9_0_re);
    dut.v1837_q(mp9_0_q); mp9_0_mem.q(mp9_0_q);
    dut.v1837_rrdy(mp9_0_rrdy); mp9_0_mem.rrdy(mp9_0_rrdy);
    mp9_0_mem.wadr(mp9_0_wadr);
    mp9_0_mem.d(mp9_0_d);
    mp9_0_mem.we(mp9_0_we);
    mp9_0_mem.wrdy(mp9_0_wrdy);
    mp10_0_mem.clk(clk); mp10_0_mem.rst(rst);
    dut.v1838_radr(mp10_0_radr); mp10_0_mem.radr(mp10_0_radr);
    dut.v1838_re(mp10_0_re); mp10_0_mem.re(mp10_0_re);
    dut.v1838_q(mp10_0_q); mp10_0_mem.q(mp10_0_q);
    dut.v1838_rrdy(mp10_0_rrdy); mp10_0_mem.rrdy(mp10_0_rrdy);
    mp10_0_mem.wadr(mp10_0_wadr);
    mp10_0_mem.d(mp10_0_d);
    mp10_0_mem.we(mp10_0_we);
    mp10_0_mem.wrdy(mp10_0_wrdy);
    mp11_0_mem.clk(clk); mp11_0_mem.rst(rst);
    dut.v1839_radr(mp11_0_radr); mp11_0_mem.radr(mp11_0_radr);
    dut.v1839_re(mp11_0_re); mp11_0_mem.re(mp11_0_re);
    dut.v1839_q(mp11_0_q); mp11_0_mem.q(mp11_0_q);
    dut.v1839_rrdy(mp11_0_rrdy); mp11_0_mem.rrdy(mp11_0_rrdy);
    mp11_0_mem.wadr(mp11_0_wadr);
    mp11_0_mem.d(mp11_0_d);
    mp11_0_mem.we(mp11_0_we);
    mp11_0_mem.wrdy(mp11_0_wrdy);
    mp12_0_mem.clk(clk); mp12_0_mem.rst(rst);
    dut.v1840_radr(mp12_0_radr); mp12_0_mem.radr(mp12_0_radr);
    dut.v1840_re(mp12_0_re); mp12_0_mem.re(mp12_0_re);
    dut.v1840_q(mp12_0_q); mp12_0_mem.q(mp12_0_q);
    dut.v1840_rrdy(mp12_0_rrdy); mp12_0_mem.rrdy(mp12_0_rrdy);
    mp12_0_mem.wadr(mp12_0_wadr);
    mp12_0_mem.d(mp12_0_d);
    mp12_0_mem.we(mp12_0_we);
    mp12_0_mem.wrdy(mp12_0_wrdy);
    mp13_0_mem.clk(clk); mp13_0_mem.rst(rst);
    mp13_0_mem.radr(mp13_0_radr);
    mp13_0_mem.re(mp13_0_re);
    mp13_0_mem.q(mp13_0_q);
    mp13_0_mem.rrdy(mp13_0_rrdy);
    dut.v1841_wadr(mp13_0_wadr); mp13_0_mem.wadr(mp13_0_wadr);
    dut.v1841_d(mp13_0_d); mp13_0_mem.d(mp13_0_d);
    dut.v1841_we(mp13_0_we); mp13_0_mem.we(mp13_0_we);
    dut.v1841_wrdy(mp13_0_wrdy); mp13_0_mem.wrdy(mp13_0_wrdy);
    mp14_0_mem.clk(clk); mp14_0_mem.rst(rst);
    mp14_0_mem.radr(mp14_0_radr);
    mp14_0_mem.re(mp14_0_re);
    mp14_0_mem.q(mp14_0_q);
    mp14_0_mem.rrdy(mp14_0_rrdy);
    dut.v1842_wadr(mp14_0_wadr); mp14_0_mem.wadr(mp14_0_wadr);
    dut.v1842_d(mp14_0_d); mp14_0_mem.d(mp14_0_d);
    dut.v1842_we(mp14_0_we); mp14_0_mem.we(mp14_0_we);
    dut.v1842_wrdy(mp14_0_wrdy); mp14_0_mem.wrdy(mp14_0_wrdy);
    mp15_0_mem.clk(clk); mp15_0_mem.rst(rst);
    mp15_0_mem.radr(mp15_0_radr);
    mp15_0_mem.re(mp15_0_re);
    mp15_0_mem.q(mp15_0_q);
    mp15_0_mem.rrdy(mp15_0_rrdy);
    dut.v1843_wadr(mp15_0_wadr); mp15_0_mem.wadr(mp15_0_wadr);
    dut.v1843_d(mp15_0_d); mp15_0_mem.d(mp15_0_d);
    dut.v1843_we(mp15_0_we); mp15_0_mem.we(mp15_0_we);
    dut.v1843_wrdy(mp15_0_wrdy); mp15_0_mem.wrdy(mp15_0_wrdy);
    mp16_0_mem.clk(clk); mp16_0_mem.rst(rst);
    mp16_0_mem.radr(mp16_0_radr);
    mp16_0_mem.re(mp16_0_re);
    mp16_0_mem.q(mp16_0_q);
    mp16_0_mem.rrdy(mp16_0_rrdy);
    dut.v1844_wadr(mp16_0_wadr); mp16_0_mem.wadr(mp16_0_wadr);
    dut.v1844_d(mp16_0_d); mp16_0_mem.d(mp16_0_d);
    dut.v1844_we(mp16_0_we); mp16_0_mem.we(mp16_0_we);
    dut.v1844_wrdy(mp16_0_wrdy); mp16_0_mem.wrdy(mp16_0_wrdy);
    SC_THREAD(src); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(snk); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
  }
  void src() {
    wait();
  }
  void snk() {
    wait();
  }
};

int sc_main(int, char *[]) {
  static tb t("t");
  #ifdef CONNECTIONS_ACCURATE_SIM
  Connections::set_sim_clk(&t.clk);
  #endif
  { std::ifstream _f("input0.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp0_0_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input1.data"); half _v; for (int f = 0; f < 55; ++f) { _f >> _v; t.mp1_0_mem.mem[f] = (half)_v; } }
  { std::ifstream _f("input2.data"); long long _v; for (int f = 0; f < 55; ++f) { _f >> _v; t.mp1_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input0.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp1_2_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input3.data"); half _v; for (int f = 0; f < 55; ++f) { _f >> _v; t.mp2_0_mem.mem[f] = (half)_v; } }
  { std::ifstream _f("input4.data"); long long _v; for (int f = 0; f < 55; ++f) { _f >> _v; t.mp2_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input0.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp2_2_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input5.data"); half _v; for (int f = 0; f < 55; ++f) { _f >> _v; t.mp3_0_mem.mem[f] = (half)_v; } }
  { std::ifstream _f("input6.data"); long long _v; for (int f = 0; f < 55; ++f) { _f >> _v; t.mp3_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input0.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp3_2_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input7.data"); half _v; for (int f = 0; f < 55; ++f) { _f >> _v; t.mp4_0_mem.mem[f] = (half)_v; } }
  { std::ifstream _f("input8.data"); long long _v; for (int f = 0; f < 55; ++f) { _f >> _v; t.mp4_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input0.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp4_2_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input0.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp5_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input0.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp6_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input0.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp7_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input0.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp8_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input9.data"); long long _v; for (int f = 0; f < 55; ++f) { _f >> _v; t.mp9_0_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input0.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp9_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input10.data"); long long _v; for (int f = 0; f < 55; ++f) { _f >> _v; t.mp10_0_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input0.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp10_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input11.data"); long long _v; for (int f = 0; f < 55; ++f) { _f >> _v; t.mp11_0_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input0.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp11_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input12.data"); long long _v; for (int f = 0; f < 55; ++f) { _f >> _v; t.mp12_0_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input0.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp12_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input0.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp13_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input0.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp14_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input0.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp15_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input0.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp16_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  t.rst = 0; sc_start(1, SC_NS);
  t.rst = 1;
  for (long long _c = 0; _c < 310000LL && !t.done_sig.read(); ++_c) sc_start(1, SC_NS); // until DUT done
  sc_start(256, SC_NS); // settle in-flight memory writes
  { std::ofstream _f("output0.data");
    for (int f = 0; f < 55; ++f) {
      float _s = 0;
      _s += t.mp5_0_mem.mem[f].to_float();
      _f << std::setprecision(9) << _s << "\n";
    } }
  { std::ofstream _f("output1.data");
    for (int f = 0; f < 55; ++f) {
      float _s = 0;
      _s += t.mp6_0_mem.mem[f].to_float();
      _f << std::setprecision(9) << _s << "\n";
    } }
  { std::ofstream _f("output2.data");
    for (int f = 0; f < 55; ++f) {
      float _s = 0;
      _s += t.mp7_0_mem.mem[f].to_float();
      _f << std::setprecision(9) << _s << "\n";
    } }
  { std::ofstream _f("output3.data");
    for (int f = 0; f < 55; ++f) {
      float _s = 0;
      _s += t.mp8_0_mem.mem[f].to_float();
      _f << std::setprecision(9) << _s << "\n";
    } }
  { std::ofstream _f("output4.data");
    for (int f = 0; f < 55; ++f) {
      long long _s = 0;
      _s += (long long) t.mp13_0_mem.mem[f];
      _f << _s << "\n";
    } }
  { std::ofstream _f("output5.data");
    for (int f = 0; f < 55; ++f) {
      long long _s = 0;
      _s += (long long) t.mp14_0_mem.mem[f];
      _f << _s << "\n";
    } }
  { std::ofstream _f("output6.data");
    for (int f = 0; f < 55; ++f) {
      long long _s = 0;
      _s += (long long) t.mp15_0_mem.mem[f];
      _f << _s << "\n";
    } }
  { std::ofstream _f("output7.data");
    for (int f = 0; f < 55; ++f) {
      long long _s = 0;
      _s += (long long) t.mp16_0_mem.mem[f];
      _f << _s << "\n";
    } }
  return 0;
}
