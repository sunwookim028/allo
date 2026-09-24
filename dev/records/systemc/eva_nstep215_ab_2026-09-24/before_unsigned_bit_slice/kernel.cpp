
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
    int32_t irf[8];	// L39
    for (int v33 = 0; v33 < 8; v33++) {	// L40
      irf[v33] = 0;	// L40
    }
    half drf[8];	// L41
    for (int v34 = 0; v34 < 8; v34++) {	// L42
      drf[v34] = half(0.000000f);	// L42
    }
    int32_t drf_full[8];	// L43
    for (int v35 = 0; v35 < 8; v35++) {	// L44
      drf_full[v35] = 0;	// L44
    }
    int32_t dsmask;	// L45
    dsmask = 0;	// L46
    int32_t crv_vld;	// L47
    crv_vld = 0;	// L48
    half crv_data;	// L49
    crv_data = half(0.000000f);	// L50
    int32_t crv_addr;	// L51
    crv_addr = 0;	// L52
    int32_t crv_mode;	// L53
    crv_mode = 0;	// L54
    int32_t crv_raw;	// L55
    crv_raw = 0;	// L56
    int32_t csd_vld;	// L57
    csd_vld = 0;	// L58
    ac_int<26, false> csd_pkt;	// L59
    csd_pkt = 0;	// L60
    int32_t csd_dir;	// L61
    csd_dir = 0;	// L62
    int32_t row_id;	// L63
    row_id = 0;	// L64
    int32_t col_id;	// L65
    col_id = 0;	// L66
    ac_int<26, false> oe_r;	// L67
    oe_r = 0;	// L68
    ac_int<26, false> ow_r;	// L69
    ow_r = 0;	// L70
    ac_int<26, false> on_r;	// L71
    on_r = 0;	// L72
    ac_int<26, false> os_r;	// L73
    os_r = 0;	// L74
    ac_int<17, false> txn_r;	// L75
    txn_r = 0;	// L76
    ac_int<17, false> txs_r;	// L77
    txs_r = 0;	// L78
    ac_int<17, false> txw_r;	// L79
    txw_r = 0;	// L80
    ac_int<17, false> txe_r;	// L81
    txe_r = 0;	// L82
    half hold_v[4][2];	// L83
    for (int v36 = 0; v36 < 4; v36++) {	// L84
      for (int v37 = 0; v37 < 2; v37++) {	// L84
        hold_v[v36][v37] = half(0.000000f);	// L84
      }
    }
    uint8_t hold_cnt[4];	// L85
    for (int v38 = 0; v38 < 4; v38++) {	// L86
      hold_cnt[v38] = 0;	// L86
    }
    ac_int<26, false> rbuf[4][2];	// L87
    for (int v39 = 0; v39 < 4; v39++) {	// L88
      for (int v40 = 0; v40 < 2; v40++) {	// L88
        rbuf[v39][v40] = 0;	// L88
      }
    }
    uint8_t rbcnt[4];	// L89
    for (int v41 = 0; v41 < 4; v41++) {	// L90
      rbcnt[v41] = 0;	// L90
    }
    uint8_t rcred[4];	// L91
    for (int v42 = 0; v42 < 4; v42++) {	// L92
      rcred[v42] = 0;	// L92
    }
    uint8_t cre_r;	// L93
    cre_r = 2;	// L94
    uint8_t crw_r;	// L95
    crw_r = 2;	// L96
    uint8_t crs_r;	// L97
    crs_r = 2;	// L98
    uint8_t crn_r;	// L99
    crn_r = 2;	// L100
    int32_t scred[4];	// L101
    for (int v43 = 0; v43 < 4; v43++) {	// L102
      scred[v43] = 0;	// L102
    }
    int32_t txp_v[4];	// L103
    for (int v44 = 0; v44 < 4; v44++) {	// L104
      txp_v[v44] = 0;	// L104
    }
    half txp_d[4];	// L105
    for (int v45 = 0; v45 < 4; v45++) {	// L106
      txp_d[v45] = half(0.000000f);	// L106
    }
    int32_t txp_r[4];	// L107
    for (int v46 = 0; v46 < 4; v46++) {	// L108
      txp_r[v46] = 0;	// L108
    }
    int32_t sc_r[4];	// L109
    for (int v47 = 0; v47 < 4; v47++) {	// L110
      sc_r[v47] = 2;	// L110
    }
    int32_t cfg_isz;	// L111
    cfg_isz = 0;	// L112
    int32_t cfg_itsz;	// L113
    cfg_itsz = 0;	// L114
    uint8_t fetch_en;	// L115
    fetch_en = 0;	// L116
    uint8_t instr_cnt;	// L117
    instr_cnt = 0;	// L118
    uint8_t iter_cnt;	// L119
    iter_cnt = 0;	// L120
    uint8_t condition_reg;	// L121
    condition_reg = 0;	// L122
    uint8_t sb_v[5];	// L123
    for (int v48 = 0; v48 < 5; v48++) {	// L124
      sb_v[v48] = 0;	// L124
    }
    uint8_t sb_dst[5];	// L125
    for (int v49 = 0; v49 < 5; v49++) {	// L126
      sb_dst[v49] = 0;	// L126
    }
    uint8_t sb_cmp[5];	// L127
    for (int v50 = 0; v50 < 5; v50++) {	// L128
      sb_cmp[v50] = 0;	// L128
    }
    uint8_t sb_rtr[5];	// L129
    for (int v51 = 0; v51 < 5; v51++) {	// L130
      sb_rtr[v51] = 0;	// L130
    }
    uint8_t sb_inj[5];	// L131
    for (int v52 = 0; v52 < 5; v52++) {	// L132
      sb_inj[v52] = 0;	// L132
    }
    uint8_t sb_dir[5];	// L133
    for (int v53 = 0; v53 < 5; v53++) {	// L134
      sb_dir[v53] = 0;	// L134
    }
    uint8_t sb_id[5];	// L135
    for (int v54 = 0; v54 < 5; v54++) {	// L136
      sb_id[v54] = 0;	// L136
    }
    uint8_t sb_rvld[5];	// L137
    for (int v55 = 0; v55 < 5; v55++) {	// L138
      sb_rvld[v55] = 0;	// L138
    }
    uint8_t sb_ix[5];	// L139
    for (int v56 = 0; v56 < 5; v56++) {	// L140
      sb_ix[v56] = 0;	// L140
    }
    uint8_t sb_long[5];	// L141
    for (int v57 = 0; v57 < 5; v57++) {	// L142
      sb_long[v57] = 0;	// L142
    }
    half resq[8];	// L143
    for (int v58 = 0; v58 < 8; v58++) {	// L144
      resq[v58] = half(0.000000f);	// L144
    }
    uint8_t cmpq[8];	// L145
    for (int v59 = 0; v59 < 8; v59++) {	// L146
      cmpq[v59] = 0;	// L146
    }
    uint8_t resq_wr;	// L147
    resq_wr = 0;	// L148
    ac_int<26, false> zpkt;	// L149
    zpkt = 0;	// L150
    ac_int<17, false> zsys;	// L151
    zsys = 0;	// L152
    int32_t zcr;	// L153
    zcr = 0;	// L154
    int32_t v60;
    v60 = v0_rd((ac_int<1, false>)(((0) + (0))));	// L155
    ac_int<33, true> v61 = v60;	// L156
    ac_int<33, true> v62 = v61 - 1;	// L157
    int v63 = v62;	// L158
    for (int v64 = 0; v64 < v63; v64 += 1) {	// L159
      ac_int<26, true> v65 = zpkt;	// L160
      v1.Push(v65);	// L161
      ac_int<26, true> v66 = zpkt;	// L162
      v2.Push(v66);	// L163
      ac_int<26, true> v67 = zpkt;	// L164
      v3.Push(v67);	// L165
      ac_int<26, true> v68 = zpkt;	// L166
      v4.Push(v68);	// L167
      ac_int<17, true> v69 = zsys;	// L168
      v5.Push(v69);	// L169
      ac_int<17, true> v70 = zsys;	// L170
      v6.Push(v70);	// L171
      ac_int<17, true> v71 = zsys;	// L172
      v7.Push(v71);	// L173
      ac_int<17, true> v72 = zsys;	// L174
      v8.Push(v72);	// L175
      int32_t v73 = zcr;	// L176
      v9.Push(v73);	// L177
      int32_t v74 = zcr;	// L178
      v10.Push(v74);	// L179
      int32_t v75 = zcr;	// L180
      v11.Push(v75);	// L181
      int32_t v76 = zcr;	// L182
      v12.Push(v76);	// L183
      int32_t v77 = zcr;	// L184
      v13.Push(v77);	// L185
      int32_t v78 = zcr;	// L186
      v14.Push(v78);	// L187
      int32_t v79 = zcr;	// L188
      v15.Push(v79);	// L189
      int32_t v80 = zcr;	// L190
      v16.Push(v80);	// L191
    }
    ac_int<26, true> v81 = oe_r;	// L193
    v1.Push(v81);	// L194
    ac_int<26, true> v82 = ow_r;	// L195
    v2.Push(v82);	// L196
    ac_int<26, true> v83 = os_r;	// L197
    v3.Push(v83);	// L198
    ac_int<26, true> v84 = on_r;	// L199
    v4.Push(v84);	// L200
    ac_int<17, true> v85 = txe_r;	// L201
    v5.Push(v85);	// L202
    ac_int<17, true> v86 = txw_r;	// L203
    v6.Push(v86);	// L204
    ac_int<17, true> v87 = txs_r;	// L205
    v7.Push(v87);	// L206
    ac_int<17, true> v88 = txn_r;	// L207
    v8.Push(v88);	// L208
    int8_t v89 = cre_r;	// L209
    v9.Push(v89);	// L210
    int8_t v90 = crw_r;	// L211
    v10.Push(v90);	// L212
    int8_t v91 = crs_r;	// L213
    v11.Push(v91);	// L214
    int8_t v92 = crn_r;	// L215
    v12.Push(v92);	// L216
    int32_t v93 = sc_r[0];	// L217
    v15.Push(v93);	// L218
    int32_t v94 = sc_r[1];	// L219
    v16.Push(v94);	// L220
    int32_t v95 = sc_r[2];	// L221
    v13.Push(v95);	// L222
    int32_t v96 = sc_r[3];	// L223
    v14.Push(v96);	// L224
#ifdef __SYNTHESIS__
    done.write(true);  // steady-state: no completion, so assert on entry (the post-body write is unreachable here)
    while (1) {  // steady-state loop (was `for t`): 1 iteration = 1 step
#else
    l_steady: for (int t = 0; t < 215; t += 1) {
#endif
      ac_int<26, false> v97 = v17.Pop();	// L226
      ac_int<26, false> p_w;	// L227
      p_w = v97;	// L228
      ac_int<26, false> v98 = v18.Pop();	// L229
      ac_int<26, false> p_e;	// L230
      p_e = v98;	// L231
      ac_int<26, false> v99 = v19.Pop();	// L232
      ac_int<26, false> p_n;	// L233
      p_n = v99;	// L234
      ac_int<26, false> v100 = v20.Pop();	// L235
      ac_int<26, false> p_s;	// L236
      p_s = v100;	// L237
      int32_t v101 = v21.Pop();	// L238
      uint8_t v102 = rcred[0];	// L239
      ac_int<33, true> v103 = v102;	// L240
      ac_int<33, true> v104 = v101;	// L241
      ac_int<33, true> v105 = v103 + v104;	// L242
      uint8_t v106 = v105;	// L243
      rcred[0] = v106;	// L244
      int32_t v107 = v22.Pop();	// L245
      uint8_t v108 = rcred[1];	// L246
      ac_int<33, true> v109 = v108;	// L247
      ac_int<33, true> v110 = v107;	// L248
      ac_int<33, true> v111 = v109 + v110;	// L249
      uint8_t v112 = v111;	// L250
      rcred[1] = v112;	// L251
      int32_t v113 = v23.Pop();	// L252
      uint8_t v114 = rcred[2];	// L253
      ac_int<33, true> v115 = v114;	// L254
      ac_int<33, true> v116 = v113;	// L255
      ac_int<33, true> v117 = v115 + v116;	// L256
      uint8_t v118 = v117;	// L257
      rcred[2] = v118;	// L258
      int32_t v119 = v24.Pop();	// L259
      uint8_t v120 = rcred[3];	// L260
      ac_int<33, true> v121 = v120;	// L261
      ac_int<33, true> v122 = v119;	// L262
      ac_int<33, true> v123 = v121 + v122;	// L263
      uint8_t v124 = v123;	// L264
      rcred[3] = v124;	// L265
      int32_t v125 = v25.Pop();	// L266
      int32_t v126 = scred[0];	// L267
      ac_int<33, true> v127 = v126;	// L268
      ac_int<33, true> v128 = v125;	// L269
      ac_int<33, true> v129 = v127 + v128;	// L270
      int32_t v130 = v129;	// L271
      scred[0] = v130;	// L272
      int32_t v131 = v26.Pop();	// L273
      int32_t v132 = scred[1];	// L274
      ac_int<33, true> v133 = v132;	// L275
      ac_int<33, true> v134 = v131;	// L276
      ac_int<33, true> v135 = v133 + v134;	// L277
      int32_t v136 = v135;	// L278
      scred[1] = v136;	// L279
      int32_t v137 = v27.Pop();	// L280
      int32_t v138 = scred[2];	// L281
      ac_int<33, true> v139 = v138;	// L282
      ac_int<33, true> v140 = v137;	// L283
      ac_int<33, true> v141 = v139 + v140;	// L284
      int32_t v142 = v141;	// L285
      scred[2] = v142;	// L286
      int32_t v143 = v28.Pop();	// L287
      int32_t v144 = scred[3];	// L288
      ac_int<33, true> v145 = v144;	// L289
      ac_int<33, true> v146 = v143;	// L290
      ac_int<33, true> v147 = v145 + v146;	// L291
      int32_t v148 = v147;	// L292
      scred[3] = v148;	// L293
      ac_int<26, false> fin[4];	// L294
      for (int v149 = 0; v149 < 4; v149++) {	// L295
        fin[v149] = 0;	// L295
      }
      ac_int<26, true> v150 = p_w;	// L296
      fin[0] = v150;	// L297
      ac_int<26, true> v151 = p_e;	// L298
      fin[1] = v151;	// L299
      ac_int<26, true> v152 = p_n;	// L300
      fin[2] = v152;	// L301
      ac_int<26, true> v153 = p_s;	// L302
      fin[3] = v153;	// L303
      l_S_d_1_d: for (int d = 0; d < 4; d++) {	// L304
        ac_int<26, false> v154 = fin[d];	// L305
        bool v155;
        ac_int<26, true> _bs_v155 = v154;
        v155 = _bs_v155[25];	// L306
        int32_t v156 = v155;	// L307
        bool v157 = v156 == 1;	// L308
        uint8_t v158 = rbcnt[d];	// L309
        int32_t v159 = v158;	// L310
        bool v160 = v159 < 2;	// L311
        bool v161 = v157 & v160;	// L312
        if (v161) {	// L313
          ac_int<26, false> v162 = fin[d];	// L314
          uint8_t v163 = rbcnt[d];	// L315
          int v164 = v163;	// L316
          rbuf[d][v164] = v162;	// L317
          uint8_t v165 = rbcnt[d];	// L318
          ac_int<33, true> v166 = v165;	// L319
          ac_int<33, true> v167 = v166 + 1;	// L320
          uint8_t v168 = v167;	// L321
          rbcnt[d] = v168;	// L322
        }
      }
      ac_int<26, false> hd[4];	// L325
      for (int v169 = 0; v169 < 4; v169++) {	// L326
        hd[v169] = 0;	// L326
      }
      int32_t hvld[4];	// L327
      for (int v170 = 0; v170 < 4; v170++) {	// L328
        hvld[v170] = 0;	// L328
      }
      int32_t hit[4];	// L329
      for (int v171 = 0; v171 < 4; v171++) {	// L330
        hit[v171] = 0;	// L330
      }
      int32_t axis[4];	// L331
      for (int v172 = 0; v172 < 4; v172++) {	// L332
        axis[v172] = 0;	// L332
      }
      int32_t v173 = col_id;	// L333
      axis[0] = v173;	// L334
      int32_t v174 = col_id;	// L335
      axis[1] = v174;	// L336
      int32_t v175 = row_id;	// L337
      axis[2] = v175;	// L338
      int32_t v176 = row_id;	// L339
      axis[3] = v176;	// L340
      l_S_d_2_d1: for (int d1 = 0; d1 < 4; d1++) {	// L341
        uint8_t v177 = rbcnt[d1];	// L342
        int32_t v178 = v177;	// L343
        bool v179 = v178 > 0;	// L344
        if (v179) {	// L345
          ac_int<26, false> v180 = rbuf[d1][0];	// L346
          hd[d1] = v180;	// L347
          hvld[d1] = 1;	// L348
          ac_int<26, false> v181 = hd[d1];	// L349
          ac_int<4, true> v182;
          ac_int<26, true> _bs_v182 = v181;
          v182 = _bs_v182.slc<4>(21);	// L350
          int32_t v183 = axis[d1];	// L351
          int32_t v184 = v182;	// L352
          bool v185 = v184 == v183;	// L353
          if (v185) {	// L354
            hit[d1] = 1;	// L355
          }
        }
      }
      ac_int<26, false> o_crv;	// L359
      o_crv = 0;	// L360
      int32_t crv_in;	// L361
      crv_in = -1;	// L362
      int32_t v186 = hit[3];	// L363
      bool v187 = v186 == 1;	// L364
      if (v187) {	// L365
        ac_int<26, false> v188 = hd[3];	// L366
        o_crv = v188;	// L367
        crv_in = 3;	// L368
      } else {
        int32_t v189 = hit[2];	// L370
        bool v190 = v189 == 1;	// L371
        if (v190) {	// L372
          ac_int<26, false> v191 = hd[2];	// L373
          o_crv = v191;	// L374
          crv_in = 2;	// L375
        } else {
          int32_t v192 = hit[1];	// L377
          bool v193 = v192 == 1;	// L378
          if (v193) {	// L379
            ac_int<26, false> v194 = hd[1];	// L380
            o_crv = v194;	// L381
            crv_in = 1;	// L382
          } else {
            int32_t v195 = hit[0];	// L384
            bool v196 = v195 == 1;	// L385
            if (v196) {	// L386
              ac_int<26, false> v197 = hd[0];	// L387
              o_crv = v197;	// L388
              crv_in = 0;	// L389
            }
          }
        }
      }
      ac_int<26, false> o_out[4];	// L394
      for (int v198 = 0; v198 < 4; v198++) {	// L395
        o_out[v198] = 0;	// L395
      }
      int32_t pop[4];	// L396
      for (int v199 = 0; v199 < 4; v199++) {	// L397
        pop[v199] = 0;	// L397
      }
      int32_t inj_done;	// L398
      inj_done = 0;	// L399
      int32_t idir;	// L400
      idir = -1;	// L401
      ac_int<26, true> v200 = csd_pkt;	// L402
      bool v201;
      ac_int<26, true> _bs_v201 = v200;
      v201 = _bs_v201[25];	// L403
      int32_t v202 = v201;	// L404
      bool v203 = v202 == 1;	// L405
      if (v203) {	// L406
        int32_t v204 = csd_dir;	// L407
        ac_int<33, true> v205 = v204;	// L408
        ac_int<33, true> v206 = 3 - v205;	// L409
        int32_t v207 = v206;	// L410
        idir = v207;	// L411
      }
      l_S_o_3_o: for (int o = 0; o < 4; o++) {	// L413
        uint8_t v208 = rcred[o];	// L414
        int32_t v209 = v208;	// L415
        bool v210 = v209 > 0;	// L416
        if (v210) {	// L417
          int32_t v211 = idir;	// L418
          ac_int<33, true> v212 = v211;	// L419
          ac_int<33, true> v213 = o;	// L420
          bool v214 = v212 == v213;	// L421
          if (v214) {	// L422
            ac_int<26, true> v215 = csd_pkt;	// L423
            o_out[o] = v215;	// L424
            uint8_t v216 = rcred[o];	// L425
            ac_int<33, true> v217 = v216;	// L426
            ac_int<33, true> v218 = v217 - 1;	// L427
            uint8_t v219 = v218;	// L428
            rcred[o] = v219;	// L429
            inj_done = 1;	// L430
          } else {
            int32_t v220 = hvld[o];	// L432
            bool v221 = v220 == 1;	// L433
            int32_t v222 = hit[o];	// L434
            bool v223 = v222 == 0;	// L435
            bool v224 = v221 & v223;	// L436
            if (v224) {	// L437
              ac_int<26, false> v225 = hd[o];	// L438
              o_out[o] = v225;	// L439
              uint8_t v226 = rcred[o];	// L440
              ac_int<33, true> v227 = v226;	// L441
              ac_int<33, true> v228 = v227 - 1;	// L442
              uint8_t v229 = v228;	// L443
              rcred[o] = v229;	// L444
              pop[o] = 1;	// L445
            }
          }
        }
      }
      int32_t v230 = crv_in;	// L450
      bool v231 = v230 >= 0;	// L451
      if (v231) {	// L452
        int32_t v232 = crv_in;	// L453
        int v233 = v232;	// L454
        pop[v233] = 1;	// L455
      }
      int32_t ret[4];	// L457
      for (int v234 = 0; v234 < 4; v234++) {	// L458
        ret[v234] = 0;	// L458
      }
      l_S_d_4_d2: for (int d2 = 0; d2 < 4; d2++) {	// L459
        int32_t v235 = pop[d2];	// L460
        bool v236 = v235 == 1;	// L461
        if (v236) {	// L462
          l_S_sft_4_sft: for (int sft = 0; sft < 1; sft++) {	// L463
            ac_int<26, false> v237 = rbuf[d2][(sft + 1)];	// L464
            rbuf[d2][sft] = v237;	// L465
          }
          uint8_t v238 = rbcnt[d2];	// L467
          ac_int<33, true> v239 = v238;	// L468
          ac_int<33, true> v240 = v239 - 1;	// L469
          uint8_t v241 = v240;	// L470
          rbcnt[d2] = v241;	// L471
          ret[d2] = 1;	// L472
        }
      }
      int32_t v242 = ret[0];	// L475
      uint8_t v243 = v242;	// L476
      cre_r = v243;	// L477
      int32_t v244 = ret[1];	// L478
      uint8_t v245 = v244;	// L479
      crw_r = v245;	// L480
      int32_t v246 = ret[2];	// L481
      uint8_t v247 = v246;	// L482
      crs_r = v247;	// L483
      int32_t v248 = ret[3];	// L484
      uint8_t v249 = v248;	// L485
      crn_r = v249;	// L486
      ac_int<26, false> v250 = o_out[0];	// L487
      oe_r = v250;	// L488
      ac_int<26, false> v251 = o_out[1];	// L489
      ow_r = v251;	// L490
      ac_int<26, false> v252 = o_out[2];	// L491
      os_r = v252;	// L492
      ac_int<26, false> v253 = o_out[3];	// L493
      on_r = v253;	// L494
      int32_t v254 = inj_done;	// L495
      bool v255 = v254 == 1;	// L496
      if (v255) {	// L497
        csd_pkt = 0;	// L498
      }
      ac_int<26, true> v256 = o_crv;	// L500
      bool v257;
      ac_int<26, true> _bs_v257 = v256;
      v257 = _bs_v257[25];	// L501
      int32_t v258 = v257;	// L502
      crv_vld = v258;	// L503
      ac_int<26, true> v259 = o_crv;	// L504
      int16_t v260;
      ac_int<26, true> _bs_v260 = v259;
      v260 = _bs_v260.slc<16>(0);	// L505
      half v261; v261.set_data(ac_int<16, true>(v260));	// L506
      crv_data = v261;	// L507
      ac_int<26, true> v262 = o_crv;	// L508
      ac_int<4, true> v263;
      ac_int<26, true> _bs_v263 = v262;
      v263 = _bs_v263.slc<4>(16);	// L509
      int32_t v264 = v263;	// L510
      crv_addr = v264;	// L511
      ac_int<26, true> v265 = o_crv;	// L512
      bool v266;
      ac_int<26, true> _bs_v266 = v265;
      v266 = _bs_v266[20];	// L513
      int32_t v267 = v266;	// L514
      crv_mode = v267;	// L515
      ac_int<26, true> v268 = o_crv;	// L516
      int16_t v269;
      ac_int<26, true> _bs_v269 = v268;
      v269 = _bs_v269.slc<16>(0);	// L517
      int32_t v270 = v269;	// L518
      crv_raw = v270;	// L519
      ac_int<17, false> v271 = v29.Pop();	// L520
      ac_int<17, false> rx_w;	// L521
      rx_w = v271;	// L522
      ac_int<17, false> v272 = v30.Pop();	// L523
      ac_int<17, false> rx_e;	// L524
      rx_e = v272;	// L525
      ac_int<17, false> v273 = v31.Pop();	// L526
      ac_int<17, false> rx_n;	// L527
      rx_n = v273;	// L528
      ac_int<17, false> v274 = v32.Pop();	// L529
      ac_int<17, false> rx_s;	// L530
      rx_s = v274;	// L531
      half rxv[4];	// L532
      for (int v275 = 0; v275 < 4; v275++) {	// L533
        rxv[v275] = half(0.000000f);	// L533
      }
      int32_t rxvld[4];	// L534
      for (int v276 = 0; v276 < 4; v276++) {	// L535
        rxvld[v276] = 0;	// L535
      }
      ac_int<17, true> v277 = rx_n;	// L536
      int16_t v278;
      ac_int<17, true> _bs_v278 = v277;
      v278 = _bs_v278.slc<16>(1);	// L537
      half v279; v279.set_data(ac_int<16, true>(v278));	// L538
      rxv[0] = v279;	// L539
      ac_int<17, true> v280 = rx_n;	// L540
      bool v281;
      ac_int<17, true> _bs_v281 = v280;
      v281 = _bs_v281[0];	// L541
      int32_t v282 = v281;	// L542
      rxvld[0] = v282;	// L543
      ac_int<17, true> v283 = rx_s;	// L544
      int16_t v284;
      ac_int<17, true> _bs_v284 = v283;
      v284 = _bs_v284.slc<16>(1);	// L545
      half v285; v285.set_data(ac_int<16, true>(v284));	// L546
      rxv[1] = v285;	// L547
      ac_int<17, true> v286 = rx_s;	// L548
      bool v287;
      ac_int<17, true> _bs_v287 = v286;
      v287 = _bs_v287[0];	// L549
      int32_t v288 = v287;	// L550
      rxvld[1] = v288;	// L551
      ac_int<17, true> v289 = rx_w;	// L552
      int16_t v290;
      ac_int<17, true> _bs_v290 = v289;
      v290 = _bs_v290.slc<16>(1);	// L553
      half v291; v291.set_data(ac_int<16, true>(v290));	// L554
      rxv[2] = v291;	// L555
      ac_int<17, true> v292 = rx_w;	// L556
      bool v293;
      ac_int<17, true> _bs_v293 = v292;
      v293 = _bs_v293[0];	// L557
      int32_t v294 = v293;	// L558
      rxvld[2] = v294;	// L559
      ac_int<17, true> v295 = rx_e;	// L560
      int16_t v296;
      ac_int<17, true> _bs_v296 = v295;
      v296 = _bs_v296.slc<16>(1);	// L561
      half v297; v297.set_data(ac_int<16, true>(v296));	// L562
      rxv[3] = v297;	// L563
      ac_int<17, true> v298 = rx_e;	// L564
      bool v299;
      ac_int<17, true> _bs_v299 = v298;
      v299 = _bs_v299[0];	// L565
      int32_t v300 = v299;	// L566
      rxvld[3] = v300;	// L567
      l_S_d_6_d3: for (int d3 = 0; d3 < 4; d3++) {	// L568
        int32_t v301 = rxvld[d3];	// L569
        bool v302 = v301 == 1;	// L570
        uint8_t v303 = hold_cnt[d3];	// L571
        int32_t v304 = v303;	// L572
        bool v305 = v304 < 2;	// L573
        bool v306 = v302 & v305;	// L574
        if (v306) {	// L575
          half v307 = rxv[d3];	// L576
          uint8_t v308 = hold_cnt[d3];	// L577
          int v309 = v308;	// L578
          hold_v[d3][v309] = v307;	// L579
          uint8_t v310 = hold_cnt[d3];	// L580
          ac_int<33, true> v311 = v310;	// L581
          ac_int<33, true> v312 = v311 + 1;	// L582
          uint8_t v313 = v312;	// L583
          hold_cnt[d3] = v313;	// L584
        }
      }
      int32_t retire_ok;	// L587
      retire_ok = 1;	// L588
      uint8_t v314 = sb_v[0];	// L589
      int32_t v315 = v314;	// L590
      bool v316 = v315 == 1;	// L591
      uint8_t v317 = sb_rtr[0];	// L592
      int32_t v318 = v317;	// L593
      bool v319 = v318 == 0;	// L594
      uint8_t v320 = sb_dst[0];	// L595
      int32_t v321 = v320;	// L596
      bool v322 = v321 >= 12;	// L597
      bool v323 = v316 & v319;	// L598
      bool v324 = v323 & v322;	// L599
      if (v324) {	// L600
        uint8_t v325 = sb_rvld[0];	// L601
        int32_t v326 = v325;	// L602
        bool v327 = v326 == 1;	// L603
        uint8_t v328 = sb_dst[0];	// L604
        int32_t v329 = v328;	// L605
        int32_t v330 = v329 & 3;	// L606
        int v331 = v330;	// L607
        int32_t v332 = txp_v[v331];	// L608
        bool v333 = v332 == 1;	// L609
        bool v334 = v327 & v333;	// L610
        if (v334) {	// L611
          retire_ok = 0;	// L612
        }
      }
      uint8_t v335 = sb_v[0];	// L615
      int32_t v336 = v335;	// L616
      bool v337 = v336 == 1;	// L617
      int32_t v338 = retire_ok;	// L618
      bool v339 = v338 == 1;	// L619
      bool v340 = v337 & v339;	// L620
      if (v340) {	// L621
        uint8_t v341 = sb_ix[0];	// L622
        int v342 = v341;	// L623
        half v343 = resq[v342];	// L624
        half wb;	// L625
        wb = v343;	// L626
        uint8_t v344 = sb_cmp[0];	// L627
        int32_t v345 = v344;	// L628
        bool v346 = v345 == 1;	// L629
        if (v346) {	// L630
          uint8_t v347 = sb_ix[0];	// L631
          int v348 = v347;	// L632
          uint8_t v349 = cmpq[v348];	// L633
          condition_reg = v349;	// L634
        }
        uint8_t v350 = sb_rtr[0];	// L636
        int32_t v351 = v350;	// L637
        bool v352 = v351 == 1;	// L638
        if (v352) {	// L639
          uint8_t v353 = sb_inj[0];	// L640
          int32_t v354 = v353;	// L641
          bool v355 = v354 == 1;	// L642
          ac_int<26, true> v356 = csd_pkt;	// L643
          bool v357;
          ac_int<26, true> _bs_v357 = v356;
          v357 = _bs_v357[25];	// L644
          int32_t v358 = v357;	// L645
          bool v359 = v358 == 0;	// L646
          bool v360 = v355 & v359;	// L647
          if (v360) {	// L648
            half v361 = wb;	// L649
            uint16_t v362 = (uint16_t)_fbits(v361);	// L650
            ac_int<26, true> v363 = csd_pkt;	// L651
            ac_int<26, true> v364;
            ac_int<26, true> _bs_v364 = v363;
            _bs_v364.set_slc(0, ac_int<16, false>(v362));
            v364 = _bs_v364;	// L652
            csd_pkt = v364;	// L653
            uint8_t v365 = sb_dst[0];	// L654
            ac_int<4, false> v366 = v365;	// L655
            ac_int<26, true> v367 = csd_pkt;	// L656
            ac_int<26, true> v368;
            ac_int<26, true> _bs_v368 = v367;
            _bs_v368.set_slc(16, ac_int<4, false>(v366));
            v368 = _bs_v368;	// L657
            csd_pkt = v368;	// L658
            uint8_t v369 = sb_id[0];	// L659
            ac_int<4, false> v370 = v369;	// L660
            ac_int<26, true> v371 = csd_pkt;	// L661
            ac_int<26, true> v372;
            ac_int<26, true> _bs_v372 = v371;
            _bs_v372.set_slc(21, ac_int<4, false>(v370));
            v372 = _bs_v372;	// L662
            csd_pkt = v372;	// L663
            uint8_t v373 = sb_rvld[0];	// L664
            bool v374 = v373;	// L665
            ac_int<26, true> v375 = csd_pkt;	// L666
            ac_int<26, true> v376;
            ac_int<26, true> _bs_v376 = v375;
            _bs_v376[25] = v374;
            v376 = _bs_v376;	// L667
            csd_pkt = v376;	// L668
            uint8_t v377 = sb_dir[0];	// L669
            int32_t v378 = v377;	// L670
            csd_dir = v378;	// L671
          }
        } else {
          uint8_t v379 = sb_dst[0];	// L674
          int32_t v380 = v379;	// L675
          bool v381 = v380 >= 12;	// L676
          if (v381) {	// L677
            uint8_t v382 = sb_rvld[0];	// L678
            int32_t v383 = v382;	// L679
            bool v384 = v383 == 1;	// L680
            if (v384) {	// L681
              uint8_t v385 = sb_dst[0];	// L682
              int32_t v386 = v385;	// L683
              int32_t v387 = v386 & 3;	// L684
              int v388 = v387;	// L685
              txp_v[v388] = 1;	// L686
              half v389 = wb;	// L687
              uint8_t v390 = sb_dst[0];	// L688
              int32_t v391 = v390;	// L689
              int32_t v392 = v391 & 3;	// L690
              int v393 = v392;	// L691
              txp_d[v393] = v389;	// L692
              uint8_t v394 = sb_dst[0];	// L693
              int32_t v395 = v394;	// L694
              int32_t v396 = v395 & 3;	// L695
              int v397 = v396;	// L696
              txp_r[v397] = 1;	// L697
            }
          } else {
            uint8_t v398 = sb_rvld[0];	// L700
            int32_t v399 = v398;	// L701
            bool v400 = v399 == 1;	// L702
            if (v400) {	// L703
              uint8_t v401 = sb_dst[0];	// L704
              int32_t v402 = v401;	// L705
              bool v403 = v402 < 8;	// L706
              int32_t v404 = dsmask;	// L707
              int32_t v405 = v404 >> v402;	// L710
              int32_t v406 = v405 & 1;	// L711
              bool v407 = v406 == 1;	// L712
              bool v408 = v403 & v407;	// L713
              if (v408) {	// L714
                uint8_t v409 = sb_dst[0];	// L715
                int v410 = v409;	// L716
                int32_t v411 = drf_full[v410];	// L717
                bool v412 = v411 == 0;	// L718
                if (v412) {	// L719
                  half v413 = wb;	// L720
                  uint8_t v414 = sb_dst[0];	// L721
                  int v415 = v414;	// L722
                  drf[v415] = v413;	// L723
                  uint8_t v416 = sb_dst[0];	// L724
                  int v417 = v416;	// L725
                  drf_full[v417] = 1;	// L726
                }
              } else {
                half v418 = wb;	// L729
                uint8_t v419 = sb_dst[0];	// L730
                int32_t v420 = v419;	// L731
                int32_t v421 = v420 & 7;	// L732
                int v422 = v421;	// L733
                drf[v422] = v418;	// L734
              }
            }
          }
        }
      }
      int32_t pc;	// L740
      pc = -1;	// L741
      int8_t v423 = fetch_en;	// L742
      int32_t v424 = v423;	// L743
      bool v425 = v424 == 1;	// L744
      if (v425) {	// L745
        int8_t v426 = instr_cnt;	// L746
        int32_t v427 = v426;	// L747
        pc = v427;	// L748
      }
      int32_t instr;	// L750
      instr = 0;	// L751
      int32_t v428 = pc;	// L752
      bool v429 = v428 >= 0;	// L753
      if (v429) {	// L754
        int32_t v430 = pc;	// L755
        int v431 = v430;	// L756
        int32_t v432 = irf[v431];	// L757
        instr = v432;	// L758
      }
      int32_t v433 = instr;	// L760
      int32_t v434 = v433 & 15;	// L761
      int32_t op;	// L762
      op = v434;	// L763
      int32_t v435 = instr;	// L764
      int32_t v436 = v435 >> 4;	// L765
      int32_t v437 = v436 & 15;	// L766
      int32_t dst;	// L767
      dst = v437;	// L768
      int32_t v438 = instr;	// L769
      int32_t v439 = v438 >> 8;	// L770
      int32_t v440 = v439 & 15;	// L771
      int32_t s1;	// L772
      s1 = v440;	// L773
      int32_t v441 = instr;	// L774
      int32_t v442 = v441 >> 12;	// L775
      int32_t v443 = v442 & 15;	// L776
      int32_t s2;	// L777
      s2 = v443;	// L778
      half a;	// L779
      a = half(0.000000f);	// L780
      half b;	// L781
      b = half(0.000000f);	// L782
      int32_t v444 = s1;	// L783
      bool v445 = v444 >= 12;	// L784
      if (v445) {	// L785
        int32_t v446 = s1;	// L786
        int32_t v447 = v446 & 3;	// L787
        int v448 = v447;	// L788
        half v449 = hold_v[v448][0];	// L789
        a = v449;	// L790
      } else {
        int32_t v450 = s1;	// L792
        int v451 = v450;	// L793
        half v452 = drf[v451];	// L794
        a = v452;	// L795
      }
      int32_t v453 = s2;	// L797
      bool v454 = v453 >= 12;	// L798
      if (v454) {	// L799
        int32_t v455 = s2;	// L800
        int32_t v456 = v455 & 3;	// L801
        int v457 = v456;	// L802
        half v458 = hold_v[v457][0];	// L803
        b = v458;	// L804
      } else {
        int32_t v459 = s2;	// L806
        int v460 = v459;	// L807
        half v461 = drf[v460];	// L808
        b = v461;	// L809
      }
      int32_t a_vld;	// L811
      a_vld = 1;	// L812
      int32_t b_vld;	// L813
      b_vld = 1;	// L814
      int32_t v462 = s1;	// L815
      bool v463 = v462 >= 12;	// L816
      if (v463) {	// L817
        a_vld = 0;	// L818
        int32_t v464 = s1;	// L819
        int32_t v465 = v464 & 3;	// L820
        int v466 = v465;	// L821
        uint8_t v467 = hold_cnt[v466];	// L822
        int32_t v468 = v467;	// L823
        bool v469 = v468 > 0;	// L824
        if (v469) {	// L825
          a_vld = 1;	// L826
        }
      }
      int32_t v470 = s2;	// L829
      bool v471 = v470 >= 12;	// L830
      if (v471) {	// L831
        b_vld = 0;	// L832
        int32_t v472 = s2;	// L833
        int32_t v473 = v472 & 3;	// L834
        int v474 = v473;	// L835
        uint8_t v475 = hold_cnt[v474];	// L836
        int32_t v476 = v475;	// L837
        bool v477 = v476 > 0;	// L838
        if (v477) {	// L839
          b_vld = 1;	// L840
        }
      }
      int32_t v478 = s1;	// L843
      bool v479 = v478 < 8;	// L844
      int32_t v480 = dsmask;	// L845
      int32_t v481 = v480 >> v478;	// L847
      int32_t v482 = v481 & 1;	// L848
      bool v483 = v482 == 1;	// L849
      bool v484 = v479 & v483;	// L850
      if (v484) {	// L851
        int32_t v485 = s1;	// L852
        int v486 = v485;	// L853
        int32_t v487 = drf_full[v486];	// L854
        bool v488 = v487 == 0;	// L855
        if (v488) {	// L856
          a_vld = 0;	// L857
        }
      }
      int32_t v489 = s2;	// L860
      bool v490 = v489 < 8;	// L861
      int32_t v491 = dsmask;	// L862
      int32_t v492 = v491 >> v489;	// L864
      int32_t v493 = v492 & 1;	// L865
      bool v494 = v493 == 1;	// L866
      bool v495 = v490 & v494;	// L867
      if (v495) {	// L868
        int32_t v496 = s2;	// L869
        int v497 = v496;	// L870
        int32_t v498 = drf_full[v497];	// L871
        bool v499 = v498 == 0;	// L872
        if (v499) {	// L873
          b_vld = 0;	// L874
        }
      }
      int32_t binop;	// L877
      binop = 0;	// L878
      int32_t v500 = op;	// L879
      bool v501 = v500 == 0;	// L880
      bool v502 = v500 == 1;	// L882
      bool v503 = v500 == 2;	// L884
      bool v504 = v500 == 8;	// L886
      bool v505 = v500 == 9;	// L888
      bool v506 = v501 | v502;	// L889
      bool v507 = v506 | v503;	// L890
      bool v508 = v507 | v504;	// L891
      bool v509 = v508 | v505;	// L892
      if (v509) {	// L893
        binop = 1;	// L894
      }
      int32_t raw;	// L896
      raw = 0;	// L897
      int32_t cmp_busy;	// L898
      cmp_busy = 0;	// L899
      int32_t fwd_a;	// L900
      fwd_a = 0;	// L901
      int32_t fwd_a_ix;	// L902
      fwd_a_ix = 0;	// L903
      int32_t raw_a;	// L904
      raw_a = 0;	// L905
      int32_t fwd_b;	// L906
      fwd_b = 0;	// L907
      int32_t fwd_b_ix;	// L908
      fwd_b_ix = 0;	// L909
      int32_t raw_b;	// L910
      raw_b = 0;	// L911
      l_S_k_7_k: for (int k = 0; k < 4; k++) {	// L912
        ac_int<34, true> v510 = k;	// L913
        ac_int<34, true> v511 = v510 + 1;	// L914
        int32_t v512 = v511;	// L915
        int32_t kk;	// L916
        kk = v512;	// L917
        int32_t v513 = kk;	// L918
        ac_int<34, true> v514 = v513;	// L919
        ac_int<34, true> v515 = 4 - v514;	// L920
        int32_t v516 = v515;	// L921
        int32_t inflight;	// L922
        inflight = v516;	// L923
        int32_t need;	// L924
        need = 0;	// L925
        int32_t v517 = kk;	// L926
        int v518 = v517;	// L927
        uint8_t v519 = sb_long[v518];	// L928
        int32_t v520 = v519;	// L929
        bool v521 = v520 == 1;	// L930
        if (v521) {	// L931
          need = 1;	// L932
        }
        int32_t rdy;	// L934
        rdy = 0;	// L935
        int32_t v522 = inflight;	// L936
        int32_t v523 = need;	// L937
        bool v524 = v522 >= v523;	// L938
        if (v524) {	// L939
          rdy = 1;	// L940
        }
        int32_t v525 = kk;	// L942
        int v526 = v525;	// L943
        uint8_t v527 = sb_v[v526];	// L944
        int32_t v528 = v527;	// L945
        bool v529 = v528 == 1;	// L946
        uint8_t v530 = sb_rtr[v526];	// L949
        int32_t v531 = v530;	// L950
        bool v532 = v531 == 0;	// L951
        uint8_t v533 = sb_dst[v526];	// L954
        int32_t v534 = v533;	// L955
        bool v535 = v534 < 12;	// L956
        bool v536 = v529 & v532;	// L957
        bool v537 = v536 & v535;	// L958
        if (v537) {	// L959
          int32_t v538 = s1;	// L960
          bool v539 = v538 < 12;	// L961
          int32_t v540 = kk;	// L962
          int v541 = v540;	// L963
          uint8_t v542 = sb_dst[v541];	// L964
          int32_t v543 = v542;	// L965
          int32_t v544 = v543 & 7;	// L966
          int32_t v545 = v538 & 7;	// L968
          bool v546 = v544 == v545;	// L969
          bool v547 = v539 & v546;	// L970
          if (v547) {	// L971
            int32_t v548 = rdy;	// L972
            bool v549 = v548 == 1;	// L973
            if (v549) {	// L974
              fwd_a = 1;	// L975
              int32_t v550 = kk;	// L976
              int v551 = v550;	// L977
              uint8_t v552 = sb_ix[v551];	// L978
              int32_t v553 = v552;	// L979
              fwd_a_ix = v553;	// L980
              raw_a = 0;	// L981
            } else {
              fwd_a = 0;	// L983
              raw_a = 1;	// L984
            }
          }
          int32_t v554 = binop;	// L987
          bool v555 = v554 == 1;	// L988
          int32_t v556 = s2;	// L989
          bool v557 = v556 < 12;	// L990
          int32_t v558 = kk;	// L991
          int v559 = v558;	// L992
          uint8_t v560 = sb_dst[v559];	// L993
          int32_t v561 = v560;	// L994
          int32_t v562 = v561 & 7;	// L995
          int32_t v563 = v556 & 7;	// L997
          bool v564 = v562 == v563;	// L998
          bool v565 = v555 & v557;	// L999
          bool v566 = v565 & v564;	// L1000
          if (v566) {	// L1001
            int32_t v567 = rdy;	// L1002
            bool v568 = v567 == 1;	// L1003
            if (v568) {	// L1004
              fwd_b = 1;	// L1005
              int32_t v569 = kk;	// L1006
              int v570 = v569;	// L1007
              uint8_t v571 = sb_ix[v570];	// L1008
              int32_t v572 = v571;	// L1009
              fwd_b_ix = v572;	// L1010
              raw_b = 0;	// L1011
            } else {
              fwd_b = 0;	// L1013
              raw_b = 1;	// L1014
            }
          }
        }
        int32_t v573 = kk;	// L1018
        int v574 = v573;	// L1019
        uint8_t v575 = sb_v[v574];	// L1020
        int32_t v576 = v575;	// L1021
        bool v577 = v576 == 1;	// L1022
        uint8_t v578 = sb_cmp[v574];	// L1025
        int32_t v579 = v578;	// L1026
        bool v580 = v579 == 1;	// L1027
        bool v581 = v577 & v580;	// L1028
        if (v581) {	// L1029
          cmp_busy = 1;	// L1030
        }
      }
      int32_t v582 = raw_a;	// L1033
      raw = v582;	// L1034
      int32_t v583 = binop;	// L1035
      bool v584 = v583 == 1;	// L1036
      int32_t v585 = raw_b;	// L1037
      bool v586 = v585 == 1;	// L1038
      bool v587 = v584 & v586;	// L1039
      if (v587) {	// L1040
        raw = 1;	// L1041
      }
      int32_t v588 = fwd_a;	// L1043
      bool v589 = v588 == 1;	// L1044
      if (v589) {	// L1045
        int32_t v590 = fwd_a_ix;	// L1046
        int v591 = v590;	// L1047
        half v592 = resq[v591];	// L1048
        a = v592;	// L1049
        a_vld = 1;	// L1050
      }
      int32_t v593 = fwd_b;	// L1052
      bool v594 = v593 == 1;	// L1053
      if (v594) {	// L1054
        int32_t v595 = fwd_b_ix;	// L1055
        int v596 = v595;	// L1056
        half v597 = resq[v596];	// L1057
        b = v597;	// L1058
        b_vld = 1;	// L1059
      }
      int32_t is_cond;	// L1061
      is_cond = 0;	// L1062
      int32_t v598 = op;	// L1063
      bool v599 = v598 >= 12;	// L1064
      ac_int<33, true> v600 = v598;	// L1066
      bool v601 = v600 <= 15;	// L1067
      bool v602 = v599 & v601;	// L1068
      if (v602) {	// L1069
        is_cond = 1;	// L1070
      }
      int32_t grant;	// L1072
      grant = 0;	// L1073
      int32_t v603 = pc;	// L1074
      bool v604 = v603 >= 0;	// L1075
      if (v604) {	// L1076
        grant = 1;	// L1077
      }
      int32_t v605 = pc;	// L1079
      bool v606 = v605 >= 0;	// L1080
      int32_t v607 = a_vld;	// L1081
      bool v608 = v607 == 0;	// L1082
      int32_t v609 = binop;	// L1083
      bool v610 = v609 == 1;	// L1084
      int32_t v611 = b_vld;	// L1085
      bool v612 = v611 == 0;	// L1086
      bool v613 = v610 & v612;	// L1087
      bool v614 = v608 | v613;	// L1088
      bool v615 = v606 & v614;	// L1089
      if (v615) {	// L1090
        grant = 0;	// L1091
      }
      int32_t v616 = pc;	// L1093
      bool v617 = v616 >= 0;	// L1094
      int32_t v618 = raw;	// L1095
      bool v619 = v618 == 1;	// L1096
      int32_t v620 = is_cond;	// L1097
      bool v621 = v620 == 1;	// L1098
      int32_t v622 = cmp_busy;	// L1099
      bool v623 = v622 == 1;	// L1100
      bool v624 = v621 & v623;	// L1101
      bool v625 = v619 | v624;	// L1102
      bool v626 = v617 & v625;	// L1103
      if (v626) {	// L1104
        grant = 0;	// L1105
      }
      int32_t v627 = retire_ok;	// L1107
      bool v628 = v627 == 0;	// L1108
      if (v628) {	// L1109
        grant = 0;	// L1110
      }
      int32_t v629 = grant;	// L1112
      bool v630 = v629 == 1;	// L1113
      if (v630) {	// L1114
        int8_t v631 = instr_cnt;	// L1115
        int32_t v632 = cfg_isz;	// L1116
        int32_t v633 = v631;	// L1117
        bool v634 = v633 == v632;	// L1118
        if (v634) {	// L1119
          instr_cnt = 0;	// L1120
          int8_t v635 = iter_cnt;	// L1121
          int32_t v636 = cfg_itsz;	// L1122
          ac_int<33, true> v637 = v636;	// L1123
          ac_int<33, true> v638 = v637 - 1;	// L1124
          ac_int<33, true> v639 = v635;	// L1125
          bool v640 = v639 == v638;	// L1126
          if (v640) {	// L1127
            fetch_en = 0;	// L1128
          } else {
            int8_t v641 = iter_cnt;	// L1130
            ac_int<33, true> v642 = v641;	// L1131
            ac_int<33, true> v643 = v642 + 1;	// L1132
            uint8_t v644 = v643;	// L1133
            iter_cnt = v644;	// L1134
          }
        } else {
          int8_t v645 = instr_cnt;	// L1137
          ac_int<33, true> v646 = v645;	// L1138
          ac_int<33, true> v647 = v646 + 1;	// L1139
          uint8_t v648 = v647;	// L1140
          instr_cnt = v648;	// L1141
        }
      }
      int32_t c1;	// L1144
      c1 = -1;	// L1145
      int32_t c2;	// L1146
      c2 = -1;	// L1147
      int32_t v649 = grant;	// L1148
      bool v650 = v649 == 1;	// L1149
      int32_t v651 = s1;	// L1150
      bool v652 = v651 >= 12;	// L1151
      bool v653 = v650 & v652;	// L1152
      if (v653) {	// L1153
        int32_t v654 = s1;	// L1154
        int32_t v655 = v654 & 3;	// L1155
        c1 = v655;	// L1156
      }
      int32_t v656 = grant;	// L1158
      bool v657 = v656 == 1;	// L1159
      int32_t v658 = s2;	// L1160
      bool v659 = v658 >= 12;	// L1161
      bool v660 = v657 & v659;	// L1162
      if (v660) {	// L1163
        int32_t v661 = s2;	// L1164
        int32_t v662 = v661 & 3;	// L1165
        c2 = v662;	// L1166
      }
      int32_t v663 = c1;	// L1168
      bool v664 = v663 >= 0;	// L1169
      if (v664) {	// L1170
        int32_t v665 = c1;	// L1171
        int v666 = v665;	// L1172
        half v667 = hold_v[v666][1];	// L1173
        hold_v[v666][0] = v667;	// L1176
        int32_t v668 = c1;	// L1177
        int v669 = v668;	// L1178
        uint8_t v670 = hold_cnt[v669];	// L1179
        ac_int<33, true> v671 = v670;	// L1180
        ac_int<33, true> v672 = v671 - 1;	// L1181
        uint8_t v673 = v672;	// L1182
        hold_cnt[v669] = v673;	// L1185
      }
      int32_t v674 = c2;	// L1187
      bool v675 = v674 >= 0;	// L1188
      int32_t v676 = c1;	// L1190
      bool v677 = v674 != v676;	// L1191
      bool v678 = v675 & v677;	// L1192
      if (v678) {	// L1193
        int32_t v679 = c2;	// L1194
        int v680 = v679;	// L1195
        half v681 = hold_v[v680][1];	// L1196
        hold_v[v680][0] = v681;	// L1199
        int32_t v682 = c2;	// L1200
        int v683 = v682;	// L1201
        uint8_t v684 = hold_cnt[v683];	// L1202
        ac_int<33, true> v685 = v684;	// L1203
        ac_int<33, true> v686 = v685 - 1;	// L1204
        uint8_t v687 = v686;	// L1205
        hold_cnt[v683] = v687;	// L1208
      }
      l_S_d_8_d4: for (int d4 = 0; d4 < 4; d4++) {	// L1210
        sc_r[d4] = 0;	// L1211
      }
      int32_t v688 = c1;	// L1213
      bool v689 = v688 >= 0;	// L1214
      if (v689) {	// L1215
        int32_t v690 = c1;	// L1216
        int v691 = v690;	// L1217
        sc_r[v691] = 1;	// L1218
      }
      int32_t v692 = c2;	// L1220
      bool v693 = v692 >= 0;	// L1221
      int32_t v694 = c1;	// L1223
      bool v695 = v692 != v694;	// L1224
      bool v696 = v693 & v695;	// L1225
      if (v696) {	// L1226
        int32_t v697 = c2;	// L1227
        int v698 = v697;	// L1228
        sc_r[v698] = 1;	// L1229
      }
      int32_t v699 = grant;	// L1231
      bool v700 = v699 == 1;	// L1232
      int32_t v701 = s1;	// L1233
      bool v702 = v701 < 8;	// L1234
      int32_t v703 = dsmask;	// L1235
      int32_t v704 = v703 >> v701;	// L1237
      int32_t v705 = v704 & 1;	// L1238
      bool v706 = v705 == 1;	// L1239
      bool v707 = v700 & v702;	// L1240
      bool v708 = v707 & v706;	// L1241
      if (v708) {	// L1242
        int32_t v709 = s1;	// L1243
        int v710 = v709;	// L1244
        drf_full[v710] = 0;	// L1245
      }
      int32_t v711 = grant;	// L1247
      bool v712 = v711 == 1;	// L1248
      int32_t v713 = s2;	// L1249
      bool v714 = v713 < 8;	// L1250
      int32_t v715 = dsmask;	// L1251
      int32_t v716 = v715 >> v713;	// L1253
      int32_t v717 = v716 & 1;	// L1254
      bool v718 = v717 == 1;	// L1255
      bool v719 = v712 & v714;	// L1256
      bool v720 = v719 & v718;	// L1257
      if (v720) {	// L1258
        int32_t v721 = s2;	// L1259
        int v722 = v721;	// L1260
        drf_full[v722] = 0;	// L1261
      }
      half res;	// L1263
      res = half(0.000000f);	// L1264
      int32_t v723 = op;	// L1265
      bool v724 = v723 == 0;	// L1266
      if (v724) {	// L1267
        half v725 = a;	// L1268
        half v726 = b;	// L1269
        half v727 = v725 + v726;	// L1270
        res = v727;	// L1271
      } else {
        int32_t v728 = op;	// L1273
        bool v729 = v728 == 1;	// L1274
        if (v729) {	// L1275
          half v730 = a;	// L1276
          half v731 = b;	// L1277
          half v732 = v730 - v731;	// L1278
          res = v732;	// L1279
        } else {
          int32_t v733 = op;	// L1281
          bool v734 = v733 == 2;	// L1282
          if (v734) {	// L1283
            half v735 = a;	// L1284
            half v736 = b;	// L1285
            half v737 = v735 * v736;	// L1286
            res = v737;	// L1287
          } else {
            int32_t v738 = op;	// L1289
            bool v739 = v738 == 8;	// L1290
            if (v739) {	// L1291
              half v740 = a;	// L1292
              half v741 = b;	// L1293
              bool v742 = v740 >= v741;	// L1294
              if (v742) {	// L1295
                res = half(1.000000f);	// L1296
              } else {
                res = half(-1.000000f);	// L1298
              }
            } else {
              int32_t v743 = op;	// L1301
              bool v744 = v743 == 9;	// L1302
              if (v744) {	// L1303
                half v745 = a;	// L1304
                half v746 = b;	// L1305
                bool v747 = v745 < v746;	// L1306
                if (v747) {	// L1307
                  res = half(1.000000f);	// L1308
                } else {
                  res = half(-1.000000f);	// L1310
                }
              } else {
                half v748 = a;	// L1313
                res = v748;	// L1314
              }
            }
          }
        }
      }
      int32_t v749 = a_vld;	// L1320
      int32_t res_vld;	// L1321
      res_vld = v749;	// L1322
      int32_t v750 = op;	// L1323
      bool v751 = v750 == 0;	// L1324
      bool v752 = v750 == 1;	// L1326
      bool v753 = v750 == 2;	// L1328
      bool v754 = v750 == 8;	// L1330
      bool v755 = v750 == 9;	// L1332
      bool v756 = v751 | v752;	// L1333
      bool v757 = v756 | v753;	// L1334
      bool v758 = v757 | v754;	// L1335
      bool v759 = v758 | v755;	// L1336
      if (v759) {	// L1337
        int32_t v760 = a_vld;	// L1338
        int32_t v761 = b_vld;	// L1339
        int64_t v762 = v760;	// L1340
        int64_t v763 = v761;	// L1341
        int64_t v764 = v762 * v763;	// L1342
        int32_t v765 = v764;	// L1343
        res_vld = v765;	// L1344
      }
      int32_t v766 = grant;	// L1346
      bool v767 = v766 == 0;	// L1347
      if (v767) {	// L1348
        res_vld = 0;	// L1349
      }
      int32_t is_rtr;	// L1351
      is_rtr = 0;	// L1352
      int32_t v768 = op;	// L1353
      bool v769 = v768 >= 4;	// L1354
      ac_int<33, true> v770 = v768;	// L1356
      bool v771 = v770 <= 7;	// L1357
      bool v772 = v769 & v771;	// L1358
      if (v772) {	// L1359
        is_rtr = 1;	// L1360
      }
      int32_t v773 = retire_ok;	// L1362
      bool v774 = v773 == 1;	// L1363
      if (v774) {	// L1364
        l_S_k_9_k1: for (int k1 = 0; k1 < 4; k1++) {	// L1365
          uint8_t v775 = sb_v[(k1 + 1)];	// L1366
          sb_v[k1] = v775;	// L1367
          uint8_t v776 = sb_dst[(k1 + 1)];	// L1368
          sb_dst[k1] = v776;	// L1369
          uint8_t v777 = sb_cmp[(k1 + 1)];	// L1370
          sb_cmp[k1] = v777;	// L1371
          uint8_t v778 = sb_rtr[(k1 + 1)];	// L1372
          sb_rtr[k1] = v778;	// L1373
          uint8_t v779 = sb_inj[(k1 + 1)];	// L1374
          sb_inj[k1] = v779;	// L1375
          uint8_t v780 = sb_dir[(k1 + 1)];	// L1376
          sb_dir[k1] = v780;	// L1377
          uint8_t v781 = sb_id[(k1 + 1)];	// L1378
          sb_id[k1] = v781;	// L1379
          uint8_t v782 = sb_rvld[(k1 + 1)];	// L1380
          sb_rvld[k1] = v782;	// L1381
          uint8_t v783 = sb_ix[(k1 + 1)];	// L1382
          sb_ix[k1] = v783;	// L1383
          uint8_t v784 = sb_long[(k1 + 1)];	// L1384
          sb_long[k1] = v784;	// L1385
        }
        sb_v[4] = 0;	// L1387
      }
      int32_t v785 = grant;	// L1389
      bool v786 = v785 == 1;	// L1390
      if (v786) {	// L1391
        half v787 = res;	// L1392
        int8_t v788 = resq_wr;	// L1393
        int v789 = v788;	// L1394
        resq[v789] = v787;	// L1395
        int32_t cq;	// L1396
        cq = 0;	// L1397
        int32_t v790 = op;	// L1398
        bool v791 = v790 == 8;	// L1399
        if (v791) {	// L1400
          half v792 = a;	// L1401
          half v793 = b;	// L1402
          bool v794 = v792 >= v793;	// L1403
          if (v794) {	// L1404
            cq = 1;	// L1405
          }
        }
        int32_t v795 = op;	// L1408
        bool v796 = v795 == 9;	// L1409
        if (v796) {	// L1410
          half v797 = a;	// L1411
          half v798 = b;	// L1412
          bool v799 = v797 < v798;	// L1413
          if (v799) {	// L1414
            cq = 1;	// L1415
          }
        }
        int32_t v800 = cq;	// L1418
        uint8_t v801 = v800;	// L1419
        int8_t v802 = resq_wr;	// L1420
        int v803 = v802;	// L1421
        cmpq[v803] = v801;	// L1422
        sb_v[4] = 1;	// L1423
        int32_t v804 = dst;	// L1424
        uint8_t v805 = v804;	// L1425
        sb_dst[4] = v805;	// L1426
        int8_t v806 = resq_wr;	// L1427
        sb_ix[4] = v806;	// L1428
        int32_t v807 = binop;	// L1429
        uint8_t v808 = v807;	// L1430
        sb_long[4] = v808;	// L1431
        sb_cmp[4] = 0;	// L1432
        int32_t v809 = op;	// L1433
        bool v810 = v809 == 8;	// L1434
        bool v811 = v809 == 9;	// L1436
        bool v812 = v810 | v811;	// L1437
        if (v812) {	// L1438
          sb_cmp[4] = 1;	// L1439
        }
        int32_t v813 = is_rtr;	// L1441
        int32_t rtrf;	// L1442
        rtrf = v813;	// L1443
        int32_t v814 = is_cond;	// L1444
        bool v815 = v814 == 1;	// L1445
        if (v815) {	// L1446
          rtrf = 1;	// L1447
        }
        int32_t v816 = rtrf;	// L1449
        uint8_t v817 = v816;	// L1450
        sb_rtr[4] = v817;	// L1451
        int32_t v818 = is_rtr;	// L1452
        int32_t inj;	// L1453
        inj = v818;	// L1454
        int32_t v819 = is_cond;	// L1455
        bool v820 = v819 == 1;	// L1456
        int8_t v821 = condition_reg;	// L1457
        int32_t v822 = v821;	// L1458
        bool v823 = v822 == 1;	// L1459
        bool v824 = v820 & v823;	// L1460
        if (v824) {	// L1461
          inj = 1;	// L1462
        }
        int32_t v825 = inj;	// L1464
        uint8_t v826 = v825;	// L1465
        sb_inj[4] = v826;	// L1466
        int32_t v827 = op;	// L1467
        int32_t v828 = v827 & 3;	// L1468
        uint8_t v829 = v828;	// L1469
        sb_dir[4] = v829;	// L1470
        int32_t v830 = s2;	// L1471
        uint8_t v831 = v830;	// L1472
        sb_id[4] = v831;	// L1473
        int32_t v832 = res_vld;	// L1474
        uint8_t v833 = v832;	// L1475
        sb_rvld[4] = v833;	// L1476
        int8_t v834 = resq_wr;	// L1477
        ac_int<33, true> v835 = v834;	// L1478
        ac_int<33, true> v836 = v835 + 1;	// L1479
        ac_int<33, true> v837 = v836 & 7;	// L1480
        uint8_t v838 = v837;	// L1481
        resq_wr = v838;	// L1482
      }
      txn_r = 0;	// L1484
      txs_r = 0;	// L1485
      txw_r = 0;	// L1486
      txe_r = 0;	// L1487
      int32_t v839 = txp_v[0];	// L1488
      bool v840 = v839 == 1;	// L1489
      int32_t v841 = scred[0];	// L1490
      bool v842 = v841 > 0;	// L1491
      bool v843 = v840 & v842;	// L1492
      if (v843) {	// L1493
        ac_int<17, false> twn;	// L1494
        twn = 0;	// L1495
        ac_int<17, true> v844 = twn;	// L1496
        ac_int<17, true> v845;
        ac_int<17, true> _bs_v845 = v844;
        _bs_v845[0] = 1;
        v845 = _bs_v845;	// L1497
        twn = v845;	// L1498
        half v846 = txp_d[0];	// L1499
        uint16_t v847 = (uint16_t)_fbits(v846);	// L1500
        ac_int<17, true> v848 = twn;	// L1501
        ac_int<17, true> v849;
        ac_int<17, true> _bs_v849 = v848;
        _bs_v849.set_slc(1, ac_int<16, false>(v847));
        v849 = _bs_v849;	// L1502
        twn = v849;	// L1503
        ac_int<17, true> v850 = twn;	// L1504
        txn_r = v850;	// L1505
        txp_v[0] = 0;	// L1506
        int32_t v851 = scred[0];	// L1507
        ac_int<33, true> v852 = v851;	// L1508
        ac_int<33, true> v853 = v852 - 1;	// L1509
        int32_t v854 = v853;	// L1510
        scred[0] = v854;	// L1511
      }
      int32_t v855 = txp_v[1];	// L1513
      bool v856 = v855 == 1;	// L1514
      int32_t v857 = scred[1];	// L1515
      bool v858 = v857 > 0;	// L1516
      bool v859 = v856 & v858;	// L1517
      if (v859) {	// L1518
        ac_int<17, false> tws;	// L1519
        tws = 0;	// L1520
        ac_int<17, true> v860 = tws;	// L1521
        ac_int<17, true> v861;
        ac_int<17, true> _bs_v861 = v860;
        _bs_v861[0] = 1;
        v861 = _bs_v861;	// L1522
        tws = v861;	// L1523
        half v862 = txp_d[1];	// L1524
        uint16_t v863 = (uint16_t)_fbits(v862);	// L1525
        ac_int<17, true> v864 = tws;	// L1526
        ac_int<17, true> v865;
        ac_int<17, true> _bs_v865 = v864;
        _bs_v865.set_slc(1, ac_int<16, false>(v863));
        v865 = _bs_v865;	// L1527
        tws = v865;	// L1528
        ac_int<17, true> v866 = tws;	// L1529
        txs_r = v866;	// L1530
        txp_v[1] = 0;	// L1531
        int32_t v867 = scred[1];	// L1532
        ac_int<33, true> v868 = v867;	// L1533
        ac_int<33, true> v869 = v868 - 1;	// L1534
        int32_t v870 = v869;	// L1535
        scred[1] = v870;	// L1536
      }
      int32_t v871 = txp_v[2];	// L1538
      bool v872 = v871 == 1;	// L1539
      int32_t v873 = scred[2];	// L1540
      bool v874 = v873 > 0;	// L1541
      bool v875 = v872 & v874;	// L1542
      if (v875) {	// L1543
        ac_int<17, false> tww;	// L1544
        tww = 0;	// L1545
        ac_int<17, true> v876 = tww;	// L1546
        ac_int<17, true> v877;
        ac_int<17, true> _bs_v877 = v876;
        _bs_v877[0] = 1;
        v877 = _bs_v877;	// L1547
        tww = v877;	// L1548
        half v878 = txp_d[2];	// L1549
        uint16_t v879 = (uint16_t)_fbits(v878);	// L1550
        ac_int<17, true> v880 = tww;	// L1551
        ac_int<17, true> v881;
        ac_int<17, true> _bs_v881 = v880;
        _bs_v881.set_slc(1, ac_int<16, false>(v879));
        v881 = _bs_v881;	// L1552
        tww = v881;	// L1553
        ac_int<17, true> v882 = tww;	// L1554
        txw_r = v882;	// L1555
        txp_v[2] = 0;	// L1556
        int32_t v883 = scred[2];	// L1557
        ac_int<33, true> v884 = v883;	// L1558
        ac_int<33, true> v885 = v884 - 1;	// L1559
        int32_t v886 = v885;	// L1560
        scred[2] = v886;	// L1561
      }
      int32_t v887 = txp_v[3];	// L1563
      bool v888 = v887 == 1;	// L1564
      int32_t v889 = scred[3];	// L1565
      bool v890 = v889 > 0;	// L1566
      bool v891 = v888 & v890;	// L1567
      if (v891) {	// L1568
        ac_int<17, false> twe;	// L1569
        twe = 0;	// L1570
        ac_int<17, true> v892 = twe;	// L1571
        ac_int<17, true> v893;
        ac_int<17, true> _bs_v893 = v892;
        _bs_v893[0] = 1;
        v893 = _bs_v893;	// L1572
        twe = v893;	// L1573
        half v894 = txp_d[3];	// L1574
        uint16_t v895 = (uint16_t)_fbits(v894);	// L1575
        ac_int<17, true> v896 = twe;	// L1576
        ac_int<17, true> v897;
        ac_int<17, true> _bs_v897 = v896;
        _bs_v897.set_slc(1, ac_int<16, false>(v895));
        v897 = _bs_v897;	// L1577
        twe = v897;	// L1578
        ac_int<17, true> v898 = twe;	// L1579
        txe_r = v898;	// L1580
        txp_v[3] = 0;	// L1581
        int32_t v899 = scred[3];	// L1582
        ac_int<33, true> v900 = v899;	// L1583
        ac_int<33, true> v901 = v900 - 1;	// L1584
        int32_t v902 = v901;	// L1585
        scred[3] = v902;	// L1586
      }
      int32_t v903 = crv_vld;	// L1588
      bool v904 = v903 == 1;	// L1589
      if (v904) {	// L1590
        int32_t v905 = crv_mode;	// L1591
        bool v906 = v905 == 1;	// L1592
        if (v906) {	// L1593
          int32_t v907 = crv_addr;	// L1594
          int32_t v908 = v907 >> 3;	// L1595
          int32_t v909 = v908 & 1;	// L1596
          bool v910 = v909 == 1;	// L1597
          if (v910) {	// L1598
            int32_t v911 = crv_raw;	// L1599
            int32_t v912 = crv_addr;	// L1600
            int32_t v913 = v912 & 7;	// L1601
            int v914 = v913;	// L1602
            irf[v914] = v911;	// L1603
          } else {
            int32_t v915 = crv_addr;	// L1605
            bool v916 = v915 == 0;	// L1606
            if (v916) {	// L1607
              int32_t v917 = crv_raw;	// L1608
              int32_t v918 = v917 & 255;	// L1609
              dsmask = v918;	// L1610
              int32_t v919 = crv_raw;	// L1611
              int32_t v920 = v919 >> 8;	// L1612
              int32_t v921 = v920 & 7;	// L1613
              cfg_isz = v921;	// L1614
              int32_t v922 = crv_raw;	// L1615
              int32_t v923 = v922 >> 15;	// L1616
              int32_t v924 = v923 & 1;	// L1617
              bool v925 = v924 == 1;	// L1618
              if (v925) {	// L1619
                fetch_en = 1;	// L1620
                instr_cnt = 0;	// L1621
                iter_cnt = 0;	// L1622
              }
            } else {
              int32_t v926 = crv_addr;	// L1625
              bool v927 = v926 == 1;	// L1626
              if (v927) {	// L1627
                int32_t v928 = crv_raw;	// L1628
                int32_t v929 = v928 & 255;	// L1629
                cfg_itsz = v929;	// L1630
              }
            }
          }
        } else {
          int32_t v930 = crv_addr;	// L1635
          int32_t v931 = v930 >> 2;	// L1636
          int32_t v932 = v931 & 3;	// L1637
          bool v933 = v932 == 3;	// L1638
          if (v933) {	// L1639
            int32_t v934 = crv_addr;	// L1640
            int32_t v935 = v934 & 3;	// L1641
            int v936 = v935;	// L1642
            txp_v[v936] = 1;	// L1643
            half v937 = crv_data;	// L1644
            int32_t v938 = crv_addr;	// L1645
            int32_t v939 = v938 & 3;	// L1646
            int v940 = v939;	// L1647
            txp_d[v940] = v937;	// L1648
            int32_t v941 = crv_addr;	// L1649
            int32_t v942 = v941 & 3;	// L1650
            int v943 = v942;	// L1651
            txp_r[v943] = 1;	// L1652
          } else {
            int32_t v944 = crv_addr;	// L1654
            bool v945 = v944 < 8;	// L1655
            int32_t v946 = dsmask;	// L1656
            int32_t v947 = v946 >> v944;	// L1658
            int32_t v948 = v947 & 1;	// L1659
            bool v949 = v948 == 1;	// L1660
            bool v950 = v945 & v949;	// L1661
            if (v950) {	// L1662
              int32_t v951 = crv_addr;	// L1663
              int v952 = v951;	// L1664
              int32_t v953 = drf_full[v952];	// L1665
              bool v954 = v953 == 0;	// L1666
              if (v954) {	// L1667
                half v955 = crv_data;	// L1668
                int32_t v956 = crv_addr;	// L1669
                int v957 = v956;	// L1670
                drf[v957] = v955;	// L1671
                int32_t v958 = crv_addr;	// L1672
                int v959 = v958;	// L1673
                drf_full[v959] = 1;	// L1674
              }
            } else {
              half v960 = crv_data;	// L1677
              int32_t v961 = crv_addr;	// L1678
              int v962 = v961;	// L1679
              drf[v962] = v960;	// L1680
            }
          }
        }
      }
      ac_int<26, true> v963 = oe_r;	// L1685
      v1.Push(v963);	// L1686
      ac_int<26, true> v964 = ow_r;	// L1687
      v2.Push(v964);	// L1688
      ac_int<26, true> v965 = os_r;	// L1689
      v3.Push(v965);	// L1690
      ac_int<26, true> v966 = on_r;	// L1691
      v4.Push(v966);	// L1692
      ac_int<17, true> v967 = txe_r;	// L1693
      v5.Push(v967);	// L1694
      ac_int<17, true> v968 = txw_r;	// L1695
      v6.Push(v968);	// L1696
      ac_int<17, true> v969 = txs_r;	// L1697
      v7.Push(v969);	// L1698
      ac_int<17, true> v970 = txn_r;	// L1699
      v8.Push(v970);	// L1700
      int8_t v971 = cre_r;	// L1701
      v9.Push(v971);	// L1702
      int8_t v972 = crw_r;	// L1703
      v10.Push(v972);	// L1704
      int8_t v973 = crs_r;	// L1705
      v11.Push(v973);	// L1706
      int8_t v974 = crn_r;	// L1707
      v12.Push(v974);	// L1708
      int32_t v975 = sc_r[0];	// L1709
      v15.Push(v975);	// L1710
      int32_t v976 = sc_r[1];	// L1711
      v16.Push(v976);	// L1712
      int32_t v977 = sc_r[2];	// L1713
      v13.Push(v977);	// L1714
      int32_t v978 = sc_r[3];	// L1715
      v14.Push(v978);	// L1716
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
  sc_out< ac_int<8, false> > v979_radr;
  sc_out<bool> v979_re;
  sc_in< half > v979_q;
  sc_in<bool> v979_rrdy;
  sc_out< ac_int<8, false> > v980_radr;
  sc_out<bool> v980_re;
  sc_in< ac_int<32, true> > v980_q;
  sc_in<bool> v980_rrdy;
  sc_out< ac_int<1, false> > v981_radr;
  sc_out<bool> v981_re;
  sc_in< ac_int<32, true> > v981_q;
  sc_in<bool> v981_rrdy;
  Connections::Out< ac_int<17, false> > v982;
  Connections::In< ac_int<32, true> > v983;
  SC_HAS_PROCESS(drv_w_0);
  drv_w_0(sc_module_name n) : sc_module(n), done("done"), v982("v982"), v983("v983") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <in>
  half v979_rd(ac_int<8, false> addr) {
    v979_radr.write(addr); v979_re.write(true);
    wait();                    // edge N: address captured
    v979_re.write(false);
    wait();                    // data valid on this edge
    return v979_q.read();
  }
  #pragma design modulario <in>
  ac_int<32, true> v980_rd(ac_int<8, false> addr) {
    v980_radr.write(addr); v980_re.write(true);
    wait();                    // edge N: address captured
    v980_re.write(false);
    wait();                    // data valid on this edge
    return v980_q.read();
  }
  #pragma design modulario <in>
  ac_int<32, true> v981_rd(ac_int<1, false> addr) {
    v981_radr.write(addr); v981_re.write(true);
    wait();                    // edge N: address captured
    v981_re.write(false);
    wait();                    // data valid on this edge
    return v981_q.read();
  }
  void run() {
    v982.Reset();
    v983.Reset();
    v979_radr.write(0);
    v979_re.write(0);
    v980_radr.write(0);
    v980_re.write(0);
    v981_radr.write(0);
    v981_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t dcred[1];	// L1729
    for (int v984 = 0; v984 < 1; v984++) {	// L1730
      dcred[v984] = 0;	// L1730
    }
    int32_t sp[1];	// L1731
    for (int v985 = 0; v985 < 1; v985++) {	// L1732
      sp[v985] = 0;	// L1732
    }
    ac_int<17, false> zw;	// L1733
    zw = 0;	// L1734
    int32_t v986;
    v986 = v981_rd((ac_int<1, false>)(((0) + (0))));	// L1735
    ac_int<33, true> v987 = v986;	// L1736
    ac_int<33, true> v988 = v987 - 1;	// L1737
    int v989 = v988;	// L1738
    for (int v990 = 0; v990 < v989; v990 += 1) {	// L1739
      ac_int<17, true> v991 = zw;	// L1740
      v982.Push(v991);	// L1741
    }
    l_S_t_1_t1: for (int t1 = 0; t1 < 215; t1++) {	// L1743
      int32_t v992 = v983.Pop();	// L1744
      int32_t v993 = dcred[0];	// L1745
      ac_int<33, true> v994 = v993;	// L1746
      ac_int<33, true> v995 = v992;	// L1747
      ac_int<33, true> v996 = v994 + v995;	// L1748
      int32_t v997 = v996;	// L1749
      dcred[0] = v997;	// L1750
      ac_int<17, false> w;	// L1751
      w = 0;	// L1752
      int32_t v998 = sp[0];	// L1753
      bool v999 = v998 < 215;	// L1754
      ac_int<33, true> v1000 = t1;	// L1756
      ac_int<33, true> v1001 = v998;	// L1757
      bool v1002 = v1000 >= v1001;	// L1758
      bool v1003 = v999 & v1002;	// L1759
      if (v1003) {	// L1760
        int32_t v1004 = sp[0];	// L1761
        int v1005 = v1004;	// L1762
        int32_t v1006;
        v1006 = v980_rd((ac_int<8, false>)(((0) * 215 + (v1005))));	// L1763
        bool v1007 = v1006 == 0;	// L1764
        if (v1007) {	// L1765
          int32_t v1008 = sp[0];	// L1766
          ac_int<33, true> v1009 = v1008;	// L1767
          ac_int<33, true> v1010 = v1009 + 1;	// L1768
          int32_t v1011 = v1010;	// L1769
          sp[0] = v1011;	// L1770
        } else {
          int32_t v1012 = dcred[0];	// L1772
          bool v1013 = v1012 > 0;	// L1773
          if (v1013) {	// L1774
            ac_int<17, true> v1014 = w;	// L1775
            ac_int<17, true> v1015;
            ac_int<17, true> _bs_v1015 = v1014;
            _bs_v1015[0] = 1;
            v1015 = _bs_v1015;	// L1776
            w = v1015;	// L1777
            int32_t v1016 = sp[0];	// L1778
            int v1017 = v1016;	// L1779
            half v1018;
            v1018 = v979_rd((ac_int<8, false>)(((0) * 215 + (v1017))));	// L1780
            uint16_t v1019 = (uint16_t)_fbits(v1018);	// L1781
            ac_int<17, true> v1020 = w;	// L1782
            ac_int<17, true> v1021;
            ac_int<17, true> _bs_v1021 = v1020;
            _bs_v1021.set_slc(1, ac_int<16, false>(v1019));
            v1021 = _bs_v1021;	// L1783
            w = v1021;	// L1784
            int32_t v1022 = dcred[0];	// L1785
            ac_int<33, true> v1023 = v1022;	// L1786
            ac_int<33, true> v1024 = v1023 - 1;	// L1787
            int32_t v1025 = v1024;	// L1788
            dcred[0] = v1025;	// L1789
            int32_t v1026 = sp[0];	// L1790
            ac_int<33, true> v1027 = v1026;	// L1791
            ac_int<33, true> v1028 = v1027 + 1;	// L1792
            int32_t v1029 = v1028;	// L1793
            sp[0] = v1029;	// L1794
          }
        }
      }
      ac_int<17, true> v1030 = w;	// L1798
      v982.Push(v1030);	// L1799
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
  sc_out< ac_int<8, false> > v1031_radr;
  sc_out<bool> v1031_re;
  sc_in< half > v1031_q;
  sc_in<bool> v1031_rrdy;
  sc_out< ac_int<8, false> > v1032_radr;
  sc_out<bool> v1032_re;
  sc_in< ac_int<32, true> > v1032_q;
  sc_in<bool> v1032_rrdy;
  sc_out< ac_int<1, false> > v1033_radr;
  sc_out<bool> v1033_re;
  sc_in< ac_int<32, true> > v1033_q;
  sc_in<bool> v1033_rrdy;
  Connections::Out< ac_int<17, false> > v1034;
  Connections::In< ac_int<32, true> > v1035;
  SC_HAS_PROCESS(drv_e_0);
  drv_e_0(sc_module_name n) : sc_module(n), done("done"), v1034("v1034"), v1035("v1035") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <in>
  half v1031_rd(ac_int<8, false> addr) {
    v1031_radr.write(addr); v1031_re.write(true);
    wait();                    // edge N: address captured
    v1031_re.write(false);
    wait();                    // data valid on this edge
    return v1031_q.read();
  }
  #pragma design modulario <in>
  ac_int<32, true> v1032_rd(ac_int<8, false> addr) {
    v1032_radr.write(addr); v1032_re.write(true);
    wait();                    // edge N: address captured
    v1032_re.write(false);
    wait();                    // data valid on this edge
    return v1032_q.read();
  }
  #pragma design modulario <in>
  ac_int<32, true> v1033_rd(ac_int<1, false> addr) {
    v1033_radr.write(addr); v1033_re.write(true);
    wait();                    // edge N: address captured
    v1033_re.write(false);
    wait();                    // data valid on this edge
    return v1033_q.read();
  }
  void run() {
    v1034.Reset();
    v1035.Reset();
    v1031_radr.write(0);
    v1031_re.write(0);
    v1032_radr.write(0);
    v1032_re.write(0);
    v1033_radr.write(0);
    v1033_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t dcred1[1];	// L1812
    for (int v1036 = 0; v1036 < 1; v1036++) {	// L1813
      dcred1[v1036] = 0;	// L1813
    }
    int32_t sp1[1];	// L1814
    for (int v1037 = 0; v1037 < 1; v1037++) {	// L1815
      sp1[v1037] = 0;	// L1815
    }
    ac_int<17, false> zw1;	// L1816
    zw1 = 0;	// L1817
    int32_t v1038;
    v1038 = v1033_rd((ac_int<1, false>)(((0) + (0))));	// L1818
    ac_int<33, true> v1039 = v1038;	// L1819
    ac_int<33, true> v1040 = v1039 - 1;	// L1820
    int v1041 = v1040;	// L1821
    for (int v1042 = 0; v1042 < v1041; v1042 += 1) {	// L1822
      ac_int<17, true> v1043 = zw1;	// L1823
      v1034.Push(v1043);	// L1824
    }
    l_S_t_1_t2: for (int t2 = 0; t2 < 215; t2++) {	// L1826
      int32_t v1044 = v1035.Pop();	// L1827
      int32_t v1045 = dcred1[0];	// L1828
      ac_int<33, true> v1046 = v1045;	// L1829
      ac_int<33, true> v1047 = v1044;	// L1830
      ac_int<33, true> v1048 = v1046 + v1047;	// L1831
      int32_t v1049 = v1048;	// L1832
      dcred1[0] = v1049;	// L1833
      ac_int<17, false> w1;	// L1834
      w1 = 0;	// L1835
      int32_t v1050 = sp1[0];	// L1836
      bool v1051 = v1050 < 215;	// L1837
      ac_int<33, true> v1052 = t2;	// L1839
      ac_int<33, true> v1053 = v1050;	// L1840
      bool v1054 = v1052 >= v1053;	// L1841
      bool v1055 = v1051 & v1054;	// L1842
      if (v1055) {	// L1843
        int32_t v1056 = sp1[0];	// L1844
        int v1057 = v1056;	// L1845
        int32_t v1058;
        v1058 = v1032_rd((ac_int<8, false>)(((0) * 215 + (v1057))));	// L1846
        bool v1059 = v1058 == 0;	// L1847
        if (v1059) {	// L1848
          int32_t v1060 = sp1[0];	// L1849
          ac_int<33, true> v1061 = v1060;	// L1850
          ac_int<33, true> v1062 = v1061 + 1;	// L1851
          int32_t v1063 = v1062;	// L1852
          sp1[0] = v1063;	// L1853
        } else {
          int32_t v1064 = dcred1[0];	// L1855
          bool v1065 = v1064 > 0;	// L1856
          if (v1065) {	// L1857
            ac_int<17, true> v1066 = w1;	// L1858
            ac_int<17, true> v1067;
            ac_int<17, true> _bs_v1067 = v1066;
            _bs_v1067[0] = 1;
            v1067 = _bs_v1067;	// L1859
            w1 = v1067;	// L1860
            int32_t v1068 = sp1[0];	// L1861
            int v1069 = v1068;	// L1862
            half v1070;
            v1070 = v1031_rd((ac_int<8, false>)(((0) * 215 + (v1069))));	// L1863
            uint16_t v1071 = (uint16_t)_fbits(v1070);	// L1864
            ac_int<17, true> v1072 = w1;	// L1865
            ac_int<17, true> v1073;
            ac_int<17, true> _bs_v1073 = v1072;
            _bs_v1073.set_slc(1, ac_int<16, false>(v1071));
            v1073 = _bs_v1073;	// L1866
            w1 = v1073;	// L1867
            int32_t v1074 = dcred1[0];	// L1868
            ac_int<33, true> v1075 = v1074;	// L1869
            ac_int<33, true> v1076 = v1075 - 1;	// L1870
            int32_t v1077 = v1076;	// L1871
            dcred1[0] = v1077;	// L1872
            int32_t v1078 = sp1[0];	// L1873
            ac_int<33, true> v1079 = v1078;	// L1874
            ac_int<33, true> v1080 = v1079 + 1;	// L1875
            int32_t v1081 = v1080;	// L1876
            sp1[0] = v1081;	// L1877
          }
        }
      }
      ac_int<17, true> v1082 = w1;	// L1881
      v1034.Push(v1082);	// L1882
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
  sc_out< ac_int<8, false> > v1083_radr;
  sc_out<bool> v1083_re;
  sc_in< half > v1083_q;
  sc_in<bool> v1083_rrdy;
  sc_out< ac_int<8, false> > v1084_radr;
  sc_out<bool> v1084_re;
  sc_in< ac_int<32, true> > v1084_q;
  sc_in<bool> v1084_rrdy;
  sc_out< ac_int<1, false> > v1085_radr;
  sc_out<bool> v1085_re;
  sc_in< ac_int<32, true> > v1085_q;
  sc_in<bool> v1085_rrdy;
  Connections::Out< ac_int<17, false> > v1086;
  Connections::In< ac_int<32, true> > v1087;
  SC_HAS_PROCESS(drv_n_0);
  drv_n_0(sc_module_name n) : sc_module(n), done("done"), v1086("v1086"), v1087("v1087") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <in>
  half v1083_rd(ac_int<8, false> addr) {
    v1083_radr.write(addr); v1083_re.write(true);
    wait();                    // edge N: address captured
    v1083_re.write(false);
    wait();                    // data valid on this edge
    return v1083_q.read();
  }
  #pragma design modulario <in>
  ac_int<32, true> v1084_rd(ac_int<8, false> addr) {
    v1084_radr.write(addr); v1084_re.write(true);
    wait();                    // edge N: address captured
    v1084_re.write(false);
    wait();                    // data valid on this edge
    return v1084_q.read();
  }
  #pragma design modulario <in>
  ac_int<32, true> v1085_rd(ac_int<1, false> addr) {
    v1085_radr.write(addr); v1085_re.write(true);
    wait();                    // edge N: address captured
    v1085_re.write(false);
    wait();                    // data valid on this edge
    return v1085_q.read();
  }
  void run() {
    v1086.Reset();
    v1087.Reset();
    v1083_radr.write(0);
    v1083_re.write(0);
    v1084_radr.write(0);
    v1084_re.write(0);
    v1085_radr.write(0);
    v1085_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t dcred2[1];	// L1895
    for (int v1088 = 0; v1088 < 1; v1088++) {	// L1896
      dcred2[v1088] = 0;	// L1896
    }
    int32_t sp2[1];	// L1897
    for (int v1089 = 0; v1089 < 1; v1089++) {	// L1898
      sp2[v1089] = 0;	// L1898
    }
    ac_int<17, false> zw2;	// L1899
    zw2 = 0;	// L1900
    int32_t v1090;
    v1090 = v1085_rd((ac_int<1, false>)(((0) + (0))));	// L1901
    ac_int<33, true> v1091 = v1090;	// L1902
    ac_int<33, true> v1092 = v1091 - 1;	// L1903
    int v1093 = v1092;	// L1904
    for (int v1094 = 0; v1094 < v1093; v1094 += 1) {	// L1905
      ac_int<17, true> v1095 = zw2;	// L1906
      v1086.Push(v1095);	// L1907
    }
    l_S_t_1_t3: for (int t3 = 0; t3 < 215; t3++) {	// L1909
      int32_t v1096 = v1087.Pop();	// L1910
      int32_t v1097 = dcred2[0];	// L1911
      ac_int<33, true> v1098 = v1097;	// L1912
      ac_int<33, true> v1099 = v1096;	// L1913
      ac_int<33, true> v1100 = v1098 + v1099;	// L1914
      int32_t v1101 = v1100;	// L1915
      dcred2[0] = v1101;	// L1916
      ac_int<17, false> w2;	// L1917
      w2 = 0;	// L1918
      int32_t v1102 = sp2[0];	// L1919
      bool v1103 = v1102 < 215;	// L1920
      ac_int<33, true> v1104 = t3;	// L1922
      ac_int<33, true> v1105 = v1102;	// L1923
      bool v1106 = v1104 >= v1105;	// L1924
      bool v1107 = v1103 & v1106;	// L1925
      if (v1107) {	// L1926
        int32_t v1108 = sp2[0];	// L1927
        int v1109 = v1108;	// L1928
        int32_t v1110;
        v1110 = v1084_rd((ac_int<8, false>)(((0) * 215 + (v1109))));	// L1929
        bool v1111 = v1110 == 0;	// L1930
        if (v1111) {	// L1931
          int32_t v1112 = sp2[0];	// L1932
          ac_int<33, true> v1113 = v1112;	// L1933
          ac_int<33, true> v1114 = v1113 + 1;	// L1934
          int32_t v1115 = v1114;	// L1935
          sp2[0] = v1115;	// L1936
        } else {
          int32_t v1116 = dcred2[0];	// L1938
          bool v1117 = v1116 > 0;	// L1939
          if (v1117) {	// L1940
            ac_int<17, true> v1118 = w2;	// L1941
            ac_int<17, true> v1119;
            ac_int<17, true> _bs_v1119 = v1118;
            _bs_v1119[0] = 1;
            v1119 = _bs_v1119;	// L1942
            w2 = v1119;	// L1943
            int32_t v1120 = sp2[0];	// L1944
            int v1121 = v1120;	// L1945
            half v1122;
            v1122 = v1083_rd((ac_int<8, false>)(((0) * 215 + (v1121))));	// L1946
            uint16_t v1123 = (uint16_t)_fbits(v1122);	// L1947
            ac_int<17, true> v1124 = w2;	// L1948
            ac_int<17, true> v1125;
            ac_int<17, true> _bs_v1125 = v1124;
            _bs_v1125.set_slc(1, ac_int<16, false>(v1123));
            v1125 = _bs_v1125;	// L1949
            w2 = v1125;	// L1950
            int32_t v1126 = dcred2[0];	// L1951
            ac_int<33, true> v1127 = v1126;	// L1952
            ac_int<33, true> v1128 = v1127 - 1;	// L1953
            int32_t v1129 = v1128;	// L1954
            dcred2[0] = v1129;	// L1955
            int32_t v1130 = sp2[0];	// L1956
            ac_int<33, true> v1131 = v1130;	// L1957
            ac_int<33, true> v1132 = v1131 + 1;	// L1958
            int32_t v1133 = v1132;	// L1959
            sp2[0] = v1133;	// L1960
          }
        }
      }
      ac_int<17, true> v1134 = w2;	// L1964
      v1086.Push(v1134);	// L1965
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
  sc_out< ac_int<8, false> > v1135_radr;
  sc_out<bool> v1135_re;
  sc_in< half > v1135_q;
  sc_in<bool> v1135_rrdy;
  sc_out< ac_int<8, false> > v1136_radr;
  sc_out<bool> v1136_re;
  sc_in< ac_int<32, true> > v1136_q;
  sc_in<bool> v1136_rrdy;
  sc_out< ac_int<1, false> > v1137_radr;
  sc_out<bool> v1137_re;
  sc_in< ac_int<32, true> > v1137_q;
  sc_in<bool> v1137_rrdy;
  Connections::Out< ac_int<17, false> > v1138;
  Connections::In< ac_int<32, true> > v1139;
  SC_HAS_PROCESS(drv_s_0);
  drv_s_0(sc_module_name n) : sc_module(n), done("done"), v1138("v1138"), v1139("v1139") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <in>
  half v1135_rd(ac_int<8, false> addr) {
    v1135_radr.write(addr); v1135_re.write(true);
    wait();                    // edge N: address captured
    v1135_re.write(false);
    wait();                    // data valid on this edge
    return v1135_q.read();
  }
  #pragma design modulario <in>
  ac_int<32, true> v1136_rd(ac_int<8, false> addr) {
    v1136_radr.write(addr); v1136_re.write(true);
    wait();                    // edge N: address captured
    v1136_re.write(false);
    wait();                    // data valid on this edge
    return v1136_q.read();
  }
  #pragma design modulario <in>
  ac_int<32, true> v1137_rd(ac_int<1, false> addr) {
    v1137_radr.write(addr); v1137_re.write(true);
    wait();                    // edge N: address captured
    v1137_re.write(false);
    wait();                    // data valid on this edge
    return v1137_q.read();
  }
  void run() {
    v1138.Reset();
    v1139.Reset();
    v1135_radr.write(0);
    v1135_re.write(0);
    v1136_radr.write(0);
    v1136_re.write(0);
    v1137_radr.write(0);
    v1137_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t dcred3[1];	// L1978
    for (int v1140 = 0; v1140 < 1; v1140++) {	// L1979
      dcred3[v1140] = 0;	// L1979
    }
    int32_t sp3[1];	// L1980
    for (int v1141 = 0; v1141 < 1; v1141++) {	// L1981
      sp3[v1141] = 0;	// L1981
    }
    ac_int<17, false> zw3;	// L1982
    zw3 = 0;	// L1983
    int32_t v1142;
    v1142 = v1137_rd((ac_int<1, false>)(((0) + (0))));	// L1984
    ac_int<33, true> v1143 = v1142;	// L1985
    ac_int<33, true> v1144 = v1143 - 1;	// L1986
    int v1145 = v1144;	// L1987
    for (int v1146 = 0; v1146 < v1145; v1146 += 1) {	// L1988
      ac_int<17, true> v1147 = zw3;	// L1989
      v1138.Push(v1147);	// L1990
    }
    l_S_t_1_t4: for (int t4 = 0; t4 < 215; t4++) {	// L1992
      int32_t v1148 = v1139.Pop();	// L1993
      int32_t v1149 = dcred3[0];	// L1994
      ac_int<33, true> v1150 = v1149;	// L1995
      ac_int<33, true> v1151 = v1148;	// L1996
      ac_int<33, true> v1152 = v1150 + v1151;	// L1997
      int32_t v1153 = v1152;	// L1998
      dcred3[0] = v1153;	// L1999
      ac_int<17, false> w3;	// L2000
      w3 = 0;	// L2001
      int32_t v1154 = sp3[0];	// L2002
      bool v1155 = v1154 < 215;	// L2003
      ac_int<33, true> v1156 = t4;	// L2005
      ac_int<33, true> v1157 = v1154;	// L2006
      bool v1158 = v1156 >= v1157;	// L2007
      bool v1159 = v1155 & v1158;	// L2008
      if (v1159) {	// L2009
        int32_t v1160 = sp3[0];	// L2010
        int v1161 = v1160;	// L2011
        int32_t v1162;
        v1162 = v1136_rd((ac_int<8, false>)(((0) * 215 + (v1161))));	// L2012
        bool v1163 = v1162 == 0;	// L2013
        if (v1163) {	// L2014
          int32_t v1164 = sp3[0];	// L2015
          ac_int<33, true> v1165 = v1164;	// L2016
          ac_int<33, true> v1166 = v1165 + 1;	// L2017
          int32_t v1167 = v1166;	// L2018
          sp3[0] = v1167;	// L2019
        } else {
          int32_t v1168 = dcred3[0];	// L2021
          bool v1169 = v1168 > 0;	// L2022
          if (v1169) {	// L2023
            ac_int<17, true> v1170 = w3;	// L2024
            ac_int<17, true> v1171;
            ac_int<17, true> _bs_v1171 = v1170;
            _bs_v1171[0] = 1;
            v1171 = _bs_v1171;	// L2025
            w3 = v1171;	// L2026
            int32_t v1172 = sp3[0];	// L2027
            int v1173 = v1172;	// L2028
            half v1174;
            v1174 = v1135_rd((ac_int<8, false>)(((0) * 215 + (v1173))));	// L2029
            uint16_t v1175 = (uint16_t)_fbits(v1174);	// L2030
            ac_int<17, true> v1176 = w3;	// L2031
            ac_int<17, true> v1177;
            ac_int<17, true> _bs_v1177 = v1176;
            _bs_v1177.set_slc(1, ac_int<16, false>(v1175));
            v1177 = _bs_v1177;	// L2032
            w3 = v1177;	// L2033
            int32_t v1178 = dcred3[0];	// L2034
            ac_int<33, true> v1179 = v1178;	// L2035
            ac_int<33, true> v1180 = v1179 - 1;	// L2036
            int32_t v1181 = v1180;	// L2037
            dcred3[0] = v1181;	// L2038
            int32_t v1182 = sp3[0];	// L2039
            ac_int<33, true> v1183 = v1182;	// L2040
            ac_int<33, true> v1184 = v1183 + 1;	// L2041
            int32_t v1185 = v1184;	// L2042
            sp3[0] = v1185;	// L2043
          }
        }
      }
      ac_int<17, true> v1186 = w3;	// L2047
      v1138.Push(v1186);	// L2048
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
  sc_out< ac_int<8, false> > v1187_wadr;
  sc_out< half > v1187_d;
  sc_out<bool> v1187_we;
  sc_in<bool> v1187_wrdy;
  sc_out< ac_int<1, false> > v1188_radr;
  sc_out<bool> v1188_re;
  sc_in< ac_int<32, true> > v1188_q;
  sc_in<bool> v1188_rrdy;
  Connections::Out< ac_int<32, true> > v1189;
  Connections::In< ac_int<17, false> > v1190;
  SC_HAS_PROCESS(col_w_0);
  col_w_0(sc_module_name n) : sc_module(n), done("done"), v1189("v1189"), v1190("v1190") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <out>
  void v1187_wr(ac_int<8, false> addr, half val) {
    v1187_wadr.write(addr); v1187_d.write(val);
    v1187_we.write(true);
    wait();
    v1187_we.write(false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v1188_rd(ac_int<1, false> addr) {
    v1188_radr.write(addr); v1188_re.write(true);
    wait();                    // edge N: address captured
    v1188_re.write(false);
    wait();                    // data valid on this edge
    return v1188_q.read();
  }
  void run() {
    v1189.Reset();
    v1190.Reset();
    v1187_wadr.write(0);
    v1187_d.write(half(0.0f));
    v1187_we.write(0);
    v1188_radr.write(0);
    v1188_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t k2[1];	// L2061
    for (int v1191 = 0; v1191 < 1; v1191++) {	// L2062
      k2[v1191] = 0;	// L2062
    }
    int32_t cret[1];	// L2063
    for (int v1192 = 0; v1192 < 1; v1192++) {	// L2064
      cret[v1192] = 0;	// L2064
    }
    int32_t zc;	// L2065
    zc = 0;	// L2066
    int32_t v1193;
    v1193 = v1188_rd((ac_int<1, false>)(((0) + (0))));	// L2067
    ac_int<33, true> v1194 = v1193;	// L2068
    ac_int<33, true> v1195 = v1194 - 1;	// L2069
    int v1196 = v1195;	// L2070
    for (int v1197 = 0; v1197 < v1196; v1197 += 1) {	// L2071
      int32_t v1198 = zc;	// L2072
      v1189.Push(v1198);	// L2073
    }
    cret[0] = 2;	// L2075
    int32_t v1199 = cret[0];	// L2076
    v1189.Push(v1199);	// L2077
    l_S_t_1_t5: for (int t5 = 0; t5 < 215; t5++) {	// L2078
      ac_int<17, false> v1200 = v1190.Pop();	// L2079
      ac_int<17, false> w4;	// L2080
      w4 = v1200;	// L2081
      cret[0] = 0;	// L2082
      ac_int<17, true> v1201 = w4;	// L2083
      bool v1202;
      ac_int<17, true> _bs_v1202 = v1201;
      v1202 = _bs_v1202[0];	// L2084
      int32_t v1203 = v1202;	// L2085
      bool v1204 = v1203 == 1;	// L2086
      if (v1204) {	// L2087
        cret[0] = 1;	// L2088
        int32_t v1205 = k2[0];	// L2089
        bool v1206 = v1205 < 215;	// L2090
        if (v1206) {	// L2091
          ac_int<17, true> v1207 = w4;	// L2092
          int16_t v1208;
          ac_int<17, true> _bs_v1208 = v1207;
          v1208 = _bs_v1208.slc<16>(1);	// L2093
          half v1209; v1209.set_data(ac_int<16, true>(v1208));	// L2094
          int32_t v1210 = k2[0];	// L2095
          int v1211 = v1210;	// L2096
          v1187_wr((ac_int<8, false>)(((0) * 215 + (v1211))), v1209);	// L2097
          int32_t v1212 = k2[0];	// L2098
          ac_int<33, true> v1213 = v1212;	// L2099
          ac_int<33, true> v1214 = v1213 + 1;	// L2100
          int32_t v1215 = v1214;	// L2101
          k2[0] = v1215;	// L2102
        }
      }
      int32_t v1216 = cret[0];	// L2105
      v1189.Push(v1216);	// L2106
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
  sc_out< ac_int<8, false> > v1217_wadr;
  sc_out< half > v1217_d;
  sc_out<bool> v1217_we;
  sc_in<bool> v1217_wrdy;
  sc_out< ac_int<1, false> > v1218_radr;
  sc_out<bool> v1218_re;
  sc_in< ac_int<32, true> > v1218_q;
  sc_in<bool> v1218_rrdy;
  Connections::Out< ac_int<32, true> > v1219;
  Connections::In< ac_int<17, false> > v1220;
  SC_HAS_PROCESS(col_e_0);
  col_e_0(sc_module_name n) : sc_module(n), done("done"), v1219("v1219"), v1220("v1220") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <out>
  void v1217_wr(ac_int<8, false> addr, half val) {
    v1217_wadr.write(addr); v1217_d.write(val);
    v1217_we.write(true);
    wait();
    v1217_we.write(false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v1218_rd(ac_int<1, false> addr) {
    v1218_radr.write(addr); v1218_re.write(true);
    wait();                    // edge N: address captured
    v1218_re.write(false);
    wait();                    // data valid on this edge
    return v1218_q.read();
  }
  void run() {
    v1219.Reset();
    v1220.Reset();
    v1217_wadr.write(0);
    v1217_d.write(half(0.0f));
    v1217_we.write(0);
    v1218_radr.write(0);
    v1218_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t k3[1];	// L2119
    for (int v1221 = 0; v1221 < 1; v1221++) {	// L2120
      k3[v1221] = 0;	// L2120
    }
    int32_t cret1[1];	// L2121
    for (int v1222 = 0; v1222 < 1; v1222++) {	// L2122
      cret1[v1222] = 0;	// L2122
    }
    int32_t zc1;	// L2123
    zc1 = 0;	// L2124
    int32_t v1223;
    v1223 = v1218_rd((ac_int<1, false>)(((0) + (0))));	// L2125
    ac_int<33, true> v1224 = v1223;	// L2126
    ac_int<33, true> v1225 = v1224 - 1;	// L2127
    int v1226 = v1225;	// L2128
    for (int v1227 = 0; v1227 < v1226; v1227 += 1) {	// L2129
      int32_t v1228 = zc1;	// L2130
      v1219.Push(v1228);	// L2131
    }
    cret1[0] = 2;	// L2133
    int32_t v1229 = cret1[0];	// L2134
    v1219.Push(v1229);	// L2135
    l_S_t_1_t6: for (int t6 = 0; t6 < 215; t6++) {	// L2136
      ac_int<17, false> v1230 = v1220.Pop();	// L2137
      ac_int<17, false> w5;	// L2138
      w5 = v1230;	// L2139
      cret1[0] = 0;	// L2140
      ac_int<17, true> v1231 = w5;	// L2141
      bool v1232;
      ac_int<17, true> _bs_v1232 = v1231;
      v1232 = _bs_v1232[0];	// L2142
      int32_t v1233 = v1232;	// L2143
      bool v1234 = v1233 == 1;	// L2144
      if (v1234) {	// L2145
        cret1[0] = 1;	// L2146
        int32_t v1235 = k3[0];	// L2147
        bool v1236 = v1235 < 215;	// L2148
        if (v1236) {	// L2149
          ac_int<17, true> v1237 = w5;	// L2150
          int16_t v1238;
          ac_int<17, true> _bs_v1238 = v1237;
          v1238 = _bs_v1238.slc<16>(1);	// L2151
          half v1239; v1239.set_data(ac_int<16, true>(v1238));	// L2152
          int32_t v1240 = k3[0];	// L2153
          int v1241 = v1240;	// L2154
          v1217_wr((ac_int<8, false>)(((0) * 215 + (v1241))), v1239);	// L2155
          int32_t v1242 = k3[0];	// L2156
          ac_int<33, true> v1243 = v1242;	// L2157
          ac_int<33, true> v1244 = v1243 + 1;	// L2158
          int32_t v1245 = v1244;	// L2159
          k3[0] = v1245;	// L2160
        }
      }
      int32_t v1246 = cret1[0];	// L2163
      v1219.Push(v1246);	// L2164
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
  sc_out< ac_int<8, false> > v1247_wadr;
  sc_out< half > v1247_d;
  sc_out<bool> v1247_we;
  sc_in<bool> v1247_wrdy;
  sc_out< ac_int<1, false> > v1248_radr;
  sc_out<bool> v1248_re;
  sc_in< ac_int<32, true> > v1248_q;
  sc_in<bool> v1248_rrdy;
  Connections::Out< ac_int<32, true> > v1249;
  Connections::In< ac_int<17, false> > v1250;
  SC_HAS_PROCESS(col_n_0);
  col_n_0(sc_module_name n) : sc_module(n), done("done"), v1249("v1249"), v1250("v1250") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <out>
  void v1247_wr(ac_int<8, false> addr, half val) {
    v1247_wadr.write(addr); v1247_d.write(val);
    v1247_we.write(true);
    wait();
    v1247_we.write(false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v1248_rd(ac_int<1, false> addr) {
    v1248_radr.write(addr); v1248_re.write(true);
    wait();                    // edge N: address captured
    v1248_re.write(false);
    wait();                    // data valid on this edge
    return v1248_q.read();
  }
  void run() {
    v1249.Reset();
    v1250.Reset();
    v1247_wadr.write(0);
    v1247_d.write(half(0.0f));
    v1247_we.write(0);
    v1248_radr.write(0);
    v1248_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t k4[1];	// L2177
    for (int v1251 = 0; v1251 < 1; v1251++) {	// L2178
      k4[v1251] = 0;	// L2178
    }
    int32_t cret2[1];	// L2179
    for (int v1252 = 0; v1252 < 1; v1252++) {	// L2180
      cret2[v1252] = 0;	// L2180
    }
    int32_t zc2;	// L2181
    zc2 = 0;	// L2182
    int32_t v1253;
    v1253 = v1248_rd((ac_int<1, false>)(((0) + (0))));	// L2183
    ac_int<33, true> v1254 = v1253;	// L2184
    ac_int<33, true> v1255 = v1254 - 1;	// L2185
    int v1256 = v1255;	// L2186
    for (int v1257 = 0; v1257 < v1256; v1257 += 1) {	// L2187
      int32_t v1258 = zc2;	// L2188
      v1249.Push(v1258);	// L2189
    }
    cret2[0] = 2;	// L2191
    int32_t v1259 = cret2[0];	// L2192
    v1249.Push(v1259);	// L2193
    l_S_t_1_t7: for (int t7 = 0; t7 < 215; t7++) {	// L2194
      ac_int<17, false> v1260 = v1250.Pop();	// L2195
      ac_int<17, false> w6;	// L2196
      w6 = v1260;	// L2197
      cret2[0] = 0;	// L2198
      ac_int<17, true> v1261 = w6;	// L2199
      bool v1262;
      ac_int<17, true> _bs_v1262 = v1261;
      v1262 = _bs_v1262[0];	// L2200
      int32_t v1263 = v1262;	// L2201
      bool v1264 = v1263 == 1;	// L2202
      if (v1264) {	// L2203
        cret2[0] = 1;	// L2204
        int32_t v1265 = k4[0];	// L2205
        bool v1266 = v1265 < 215;	// L2206
        if (v1266) {	// L2207
          ac_int<17, true> v1267 = w6;	// L2208
          int16_t v1268;
          ac_int<17, true> _bs_v1268 = v1267;
          v1268 = _bs_v1268.slc<16>(1);	// L2209
          half v1269; v1269.set_data(ac_int<16, true>(v1268));	// L2210
          int32_t v1270 = k4[0];	// L2211
          int v1271 = v1270;	// L2212
          v1247_wr((ac_int<8, false>)(((0) * 215 + (v1271))), v1269);	// L2213
          int32_t v1272 = k4[0];	// L2214
          ac_int<33, true> v1273 = v1272;	// L2215
          ac_int<33, true> v1274 = v1273 + 1;	// L2216
          int32_t v1275 = v1274;	// L2217
          k4[0] = v1275;	// L2218
        }
      }
      int32_t v1276 = cret2[0];	// L2221
      v1249.Push(v1276);	// L2222
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
  sc_out< ac_int<8, false> > v1277_wadr;
  sc_out< half > v1277_d;
  sc_out<bool> v1277_we;
  sc_in<bool> v1277_wrdy;
  sc_out< ac_int<1, false> > v1278_radr;
  sc_out<bool> v1278_re;
  sc_in< ac_int<32, true> > v1278_q;
  sc_in<bool> v1278_rrdy;
  Connections::Out< ac_int<32, true> > v1279;
  Connections::In< ac_int<17, false> > v1280;
  SC_HAS_PROCESS(col_s_0);
  col_s_0(sc_module_name n) : sc_module(n), done("done"), v1279("v1279"), v1280("v1280") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <out>
  void v1277_wr(ac_int<8, false> addr, half val) {
    v1277_wadr.write(addr); v1277_d.write(val);
    v1277_we.write(true);
    wait();
    v1277_we.write(false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v1278_rd(ac_int<1, false> addr) {
    v1278_radr.write(addr); v1278_re.write(true);
    wait();                    // edge N: address captured
    v1278_re.write(false);
    wait();                    // data valid on this edge
    return v1278_q.read();
  }
  void run() {
    v1279.Reset();
    v1280.Reset();
    v1277_wadr.write(0);
    v1277_d.write(half(0.0f));
    v1277_we.write(0);
    v1278_radr.write(0);
    v1278_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t k5[1];	// L2235
    for (int v1281 = 0; v1281 < 1; v1281++) {	// L2236
      k5[v1281] = 0;	// L2236
    }
    int32_t cret3[1];	// L2237
    for (int v1282 = 0; v1282 < 1; v1282++) {	// L2238
      cret3[v1282] = 0;	// L2238
    }
    int32_t zc3;	// L2239
    zc3 = 0;	// L2240
    int32_t v1283;
    v1283 = v1278_rd((ac_int<1, false>)(((0) + (0))));	// L2241
    ac_int<33, true> v1284 = v1283;	// L2242
    ac_int<33, true> v1285 = v1284 - 1;	// L2243
    int v1286 = v1285;	// L2244
    for (int v1287 = 0; v1287 < v1286; v1287 += 1) {	// L2245
      int32_t v1288 = zc3;	// L2246
      v1279.Push(v1288);	// L2247
    }
    cret3[0] = 2;	// L2249
    int32_t v1289 = cret3[0];	// L2250
    v1279.Push(v1289);	// L2251
    l_S_t_1_t8: for (int t8 = 0; t8 < 215; t8++) {	// L2252
      ac_int<17, false> v1290 = v1280.Pop();	// L2253
      ac_int<17, false> w7;	// L2254
      w7 = v1290;	// L2255
      cret3[0] = 0;	// L2256
      ac_int<17, true> v1291 = w7;	// L2257
      bool v1292;
      ac_int<17, true> _bs_v1292 = v1291;
      v1292 = _bs_v1292[0];	// L2258
      int32_t v1293 = v1292;	// L2259
      bool v1294 = v1293 == 1;	// L2260
      if (v1294) {	// L2261
        cret3[0] = 1;	// L2262
        int32_t v1295 = k5[0];	// L2263
        bool v1296 = v1295 < 215;	// L2264
        if (v1296) {	// L2265
          ac_int<17, true> v1297 = w7;	// L2266
          int16_t v1298;
          ac_int<17, true> _bs_v1298 = v1297;
          v1298 = _bs_v1298.slc<16>(1);	// L2267
          half v1299; v1299.set_data(ac_int<16, true>(v1298));	// L2268
          int32_t v1300 = k5[0];	// L2269
          int v1301 = v1300;	// L2270
          v1277_wr((ac_int<8, false>)(((0) * 215 + (v1301))), v1299);	// L2271
          int32_t v1302 = k5[0];	// L2272
          ac_int<33, true> v1303 = v1302;	// L2273
          ac_int<33, true> v1304 = v1303 + 1;	// L2274
          int32_t v1305 = v1304;	// L2275
          k5[0] = v1305;	// L2276
        }
      }
      int32_t v1306 = cret3[0];	// L2279
      v1279.Push(v1306);	// L2280
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
  sc_out< ac_int<8, false> > v1307_radr;
  sc_out<bool> v1307_re;
  sc_in< ac_int<32, true> > v1307_q;
  sc_in<bool> v1307_rrdy;
  sc_out< ac_int<1, false> > v1308_radr;
  sc_out<bool> v1308_re;
  sc_in< ac_int<32, true> > v1308_q;
  sc_in<bool> v1308_rrdy;
  Connections::Out< ac_int<26, false> > v1309;
  Connections::In< ac_int<32, true> > v1310;
  SC_HAS_PROCESS(rdrv_w_0);
  rdrv_w_0(sc_module_name n) : sc_module(n), done("done"), v1309("v1309"), v1310("v1310") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v1307_rd(ac_int<8, false> addr) {
    v1307_radr.write(addr); v1307_re.write(true);
    wait();                    // edge N: address captured
    v1307_re.write(false);
    wait();                    // data valid on this edge
    return v1307_q.read();
  }
  #pragma design modulario <in>
  ac_int<32, true> v1308_rd(ac_int<1, false> addr) {
    v1308_radr.write(addr); v1308_re.write(true);
    wait();                    // edge N: address captured
    v1308_re.write(false);
    wait();                    // data valid on this edge
    return v1308_q.read();
  }
  void run() {
    v1309.Reset();
    v1310.Reset();
    v1307_radr.write(0);
    v1307_re.write(0);
    v1308_radr.write(0);
    v1308_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t dcred4[1];	// L2292
    for (int v1311 = 0; v1311 < 1; v1311++) {	// L2293
      dcred4[v1311] = 0;	// L2293
    }
    int32_t sp4[1];	// L2294
    for (int v1312 = 0; v1312 < 1; v1312++) {	// L2295
      sp4[v1312] = 0;	// L2295
    }
    ac_int<26, false> zp;	// L2296
    zp = 0;	// L2297
    int32_t v1313;
    v1313 = v1308_rd((ac_int<1, false>)(((0) + (0))));	// L2298
    ac_int<33, true> v1314 = v1313;	// L2299
    ac_int<33, true> v1315 = v1314 - 1;	// L2300
    int v1316 = v1315;	// L2301
    for (int v1317 = 0; v1317 < v1316; v1317 += 1) {	// L2302
      ac_int<26, true> v1318 = zp;	// L2303
      v1309.Push(v1318);	// L2304
    }
    l_S_t_1_t9: for (int t9 = 0; t9 < 215; t9++) {	// L2306
      int32_t v1319 = v1310.Pop();	// L2307
      int32_t v1320 = dcred4[0];	// L2308
      ac_int<33, true> v1321 = v1320;	// L2309
      ac_int<33, true> v1322 = v1319;	// L2310
      ac_int<33, true> v1323 = v1321 + v1322;	// L2311
      int32_t v1324 = v1323;	// L2312
      dcred4[0] = v1324;	// L2313
      ac_int<26, false> pw;	// L2314
      pw = 0;	// L2315
      int32_t v1325 = sp4[0];	// L2316
      bool v1326 = v1325 < 215;	// L2317
      if (v1326) {	// L2318
        ac_int<26, false> cand;	// L2319
        cand = 0;	// L2320
        int32_t v1327 = sp4[0];	// L2321
        int v1328 = v1327;	// L2322
        int32_t v1329;
        v1329 = v1307_rd((ac_int<8, false>)(((0) * 215 + (v1328))));	// L2323
        ac_int<26, false> v1330 = v1329;	// L2324
        ac_int<26, true> v1331 = cand;	// L2325
        ac_int<26, true> v1332;
        ac_int<26, true> _bs_v1332 = v1331;
        _bs_v1332.set_slc(0, ac_int<26, false>(v1330));
        v1332 = _bs_v1332;	// L2326
        cand = v1332;	// L2327
        ac_int<26, true> v1333 = cand;	// L2328
        bool v1334;
        ac_int<26, true> _bs_v1334 = v1333;
        v1334 = _bs_v1334[25];	// L2329
        int32_t v1335 = v1334;	// L2330
        bool v1336 = v1335 == 0;	// L2331
        if (v1336) {	// L2332
          int32_t v1337 = sp4[0];	// L2333
          ac_int<33, true> v1338 = v1337;	// L2334
          ac_int<33, true> v1339 = v1338 + 1;	// L2335
          int32_t v1340 = v1339;	// L2336
          sp4[0] = v1340;	// L2337
        } else {
          int32_t v1341 = dcred4[0];	// L2339
          bool v1342 = v1341 > 0;	// L2340
          if (v1342) {	// L2341
            ac_int<26, true> v1343 = cand;	// L2342
            pw = v1343;	// L2343
            int32_t v1344 = dcred4[0];	// L2344
            ac_int<33, true> v1345 = v1344;	// L2345
            ac_int<33, true> v1346 = v1345 - 1;	// L2346
            int32_t v1347 = v1346;	// L2347
            dcred4[0] = v1347;	// L2348
            int32_t v1348 = sp4[0];	// L2349
            ac_int<33, true> v1349 = v1348;	// L2350
            ac_int<33, true> v1350 = v1349 + 1;	// L2351
            int32_t v1351 = v1350;	// L2352
            sp4[0] = v1351;	// L2353
          }
        }
      }
      ac_int<26, true> v1352 = pw;	// L2357
      v1309.Push(v1352);	// L2358
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
  sc_out< ac_int<8, false> > v1353_radr;
  sc_out<bool> v1353_re;
  sc_in< ac_int<32, true> > v1353_q;
  sc_in<bool> v1353_rrdy;
  sc_out< ac_int<1, false> > v1354_radr;
  sc_out<bool> v1354_re;
  sc_in< ac_int<32, true> > v1354_q;
  sc_in<bool> v1354_rrdy;
  Connections::Out< ac_int<26, false> > v1355;
  Connections::In< ac_int<32, true> > v1356;
  SC_HAS_PROCESS(rdrv_e_0);
  rdrv_e_0(sc_module_name n) : sc_module(n), done("done"), v1355("v1355"), v1356("v1356") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v1353_rd(ac_int<8, false> addr) {
    v1353_radr.write(addr); v1353_re.write(true);
    wait();                    // edge N: address captured
    v1353_re.write(false);
    wait();                    // data valid on this edge
    return v1353_q.read();
  }
  #pragma design modulario <in>
  ac_int<32, true> v1354_rd(ac_int<1, false> addr) {
    v1354_radr.write(addr); v1354_re.write(true);
    wait();                    // edge N: address captured
    v1354_re.write(false);
    wait();                    // data valid on this edge
    return v1354_q.read();
  }
  void run() {
    v1355.Reset();
    v1356.Reset();
    v1353_radr.write(0);
    v1353_re.write(0);
    v1354_radr.write(0);
    v1354_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t dcred5[1];	// L2370
    for (int v1357 = 0; v1357 < 1; v1357++) {	// L2371
      dcred5[v1357] = 0;	// L2371
    }
    int32_t sp5[1];	// L2372
    for (int v1358 = 0; v1358 < 1; v1358++) {	// L2373
      sp5[v1358] = 0;	// L2373
    }
    ac_int<26, false> zp1;	// L2374
    zp1 = 0;	// L2375
    int32_t v1359;
    v1359 = v1354_rd((ac_int<1, false>)(((0) + (0))));	// L2376
    ac_int<33, true> v1360 = v1359;	// L2377
    ac_int<33, true> v1361 = v1360 - 1;	// L2378
    int v1362 = v1361;	// L2379
    for (int v1363 = 0; v1363 < v1362; v1363 += 1) {	// L2380
      ac_int<26, true> v1364 = zp1;	// L2381
      v1355.Push(v1364);	// L2382
    }
    l_S_t_1_t10: for (int t10 = 0; t10 < 215; t10++) {	// L2384
      int32_t v1365 = v1356.Pop();	// L2385
      int32_t v1366 = dcred5[0];	// L2386
      ac_int<33, true> v1367 = v1366;	// L2387
      ac_int<33, true> v1368 = v1365;	// L2388
      ac_int<33, true> v1369 = v1367 + v1368;	// L2389
      int32_t v1370 = v1369;	// L2390
      dcred5[0] = v1370;	// L2391
      ac_int<26, false> pw1;	// L2392
      pw1 = 0;	// L2393
      int32_t v1371 = sp5[0];	// L2394
      bool v1372 = v1371 < 215;	// L2395
      if (v1372) {	// L2396
        ac_int<26, false> cand1;	// L2397
        cand1 = 0;	// L2398
        int32_t v1373 = sp5[0];	// L2399
        int v1374 = v1373;	// L2400
        int32_t v1375;
        v1375 = v1353_rd((ac_int<8, false>)(((0) * 215 + (v1374))));	// L2401
        ac_int<26, false> v1376 = v1375;	// L2402
        ac_int<26, true> v1377 = cand1;	// L2403
        ac_int<26, true> v1378;
        ac_int<26, true> _bs_v1378 = v1377;
        _bs_v1378.set_slc(0, ac_int<26, false>(v1376));
        v1378 = _bs_v1378;	// L2404
        cand1 = v1378;	// L2405
        ac_int<26, true> v1379 = cand1;	// L2406
        bool v1380;
        ac_int<26, true> _bs_v1380 = v1379;
        v1380 = _bs_v1380[25];	// L2407
        int32_t v1381 = v1380;	// L2408
        bool v1382 = v1381 == 0;	// L2409
        if (v1382) {	// L2410
          int32_t v1383 = sp5[0];	// L2411
          ac_int<33, true> v1384 = v1383;	// L2412
          ac_int<33, true> v1385 = v1384 + 1;	// L2413
          int32_t v1386 = v1385;	// L2414
          sp5[0] = v1386;	// L2415
        } else {
          int32_t v1387 = dcred5[0];	// L2417
          bool v1388 = v1387 > 0;	// L2418
          if (v1388) {	// L2419
            ac_int<26, true> v1389 = cand1;	// L2420
            pw1 = v1389;	// L2421
            int32_t v1390 = dcred5[0];	// L2422
            ac_int<33, true> v1391 = v1390;	// L2423
            ac_int<33, true> v1392 = v1391 - 1;	// L2424
            int32_t v1393 = v1392;	// L2425
            dcred5[0] = v1393;	// L2426
            int32_t v1394 = sp5[0];	// L2427
            ac_int<33, true> v1395 = v1394;	// L2428
            ac_int<33, true> v1396 = v1395 + 1;	// L2429
            int32_t v1397 = v1396;	// L2430
            sp5[0] = v1397;	// L2431
          }
        }
      }
      ac_int<26, true> v1398 = pw1;	// L2435
      v1355.Push(v1398);	// L2436
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
  sc_out< ac_int<8, false> > v1399_radr;
  sc_out<bool> v1399_re;
  sc_in< ac_int<32, true> > v1399_q;
  sc_in<bool> v1399_rrdy;
  sc_out< ac_int<1, false> > v1400_radr;
  sc_out<bool> v1400_re;
  sc_in< ac_int<32, true> > v1400_q;
  sc_in<bool> v1400_rrdy;
  Connections::Out< ac_int<26, false> > v1401;
  Connections::In< ac_int<32, true> > v1402;
  SC_HAS_PROCESS(rdrv_n_0);
  rdrv_n_0(sc_module_name n) : sc_module(n), done("done"), v1401("v1401"), v1402("v1402") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v1399_rd(ac_int<8, false> addr) {
    v1399_radr.write(addr); v1399_re.write(true);
    wait();                    // edge N: address captured
    v1399_re.write(false);
    wait();                    // data valid on this edge
    return v1399_q.read();
  }
  #pragma design modulario <in>
  ac_int<32, true> v1400_rd(ac_int<1, false> addr) {
    v1400_radr.write(addr); v1400_re.write(true);
    wait();                    // edge N: address captured
    v1400_re.write(false);
    wait();                    // data valid on this edge
    return v1400_q.read();
  }
  void run() {
    v1401.Reset();
    v1402.Reset();
    v1399_radr.write(0);
    v1399_re.write(0);
    v1400_radr.write(0);
    v1400_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t dcred6[1];	// L2448
    for (int v1403 = 0; v1403 < 1; v1403++) {	// L2449
      dcred6[v1403] = 0;	// L2449
    }
    int32_t sp6[1];	// L2450
    for (int v1404 = 0; v1404 < 1; v1404++) {	// L2451
      sp6[v1404] = 0;	// L2451
    }
    ac_int<26, false> zp2;	// L2452
    zp2 = 0;	// L2453
    int32_t v1405;
    v1405 = v1400_rd((ac_int<1, false>)(((0) + (0))));	// L2454
    ac_int<33, true> v1406 = v1405;	// L2455
    ac_int<33, true> v1407 = v1406 - 1;	// L2456
    int v1408 = v1407;	// L2457
    for (int v1409 = 0; v1409 < v1408; v1409 += 1) {	// L2458
      ac_int<26, true> v1410 = zp2;	// L2459
      v1401.Push(v1410);	// L2460
    }
    l_S_t_1_t11: for (int t11 = 0; t11 < 215; t11++) {	// L2462
      int32_t v1411 = v1402.Pop();	// L2463
      int32_t v1412 = dcred6[0];	// L2464
      ac_int<33, true> v1413 = v1412;	// L2465
      ac_int<33, true> v1414 = v1411;	// L2466
      ac_int<33, true> v1415 = v1413 + v1414;	// L2467
      int32_t v1416 = v1415;	// L2468
      dcred6[0] = v1416;	// L2469
      ac_int<26, false> pw2;	// L2470
      pw2 = 0;	// L2471
      int32_t v1417 = sp6[0];	// L2472
      bool v1418 = v1417 < 215;	// L2473
      if (v1418) {	// L2474
        ac_int<26, false> cand2;	// L2475
        cand2 = 0;	// L2476
        int32_t v1419 = sp6[0];	// L2477
        int v1420 = v1419;	// L2478
        int32_t v1421;
        v1421 = v1399_rd((ac_int<8, false>)(((0) * 215 + (v1420))));	// L2479
        ac_int<26, false> v1422 = v1421;	// L2480
        ac_int<26, true> v1423 = cand2;	// L2481
        ac_int<26, true> v1424;
        ac_int<26, true> _bs_v1424 = v1423;
        _bs_v1424.set_slc(0, ac_int<26, false>(v1422));
        v1424 = _bs_v1424;	// L2482
        cand2 = v1424;	// L2483
        ac_int<26, true> v1425 = cand2;	// L2484
        bool v1426;
        ac_int<26, true> _bs_v1426 = v1425;
        v1426 = _bs_v1426[25];	// L2485
        int32_t v1427 = v1426;	// L2486
        bool v1428 = v1427 == 0;	// L2487
        if (v1428) {	// L2488
          int32_t v1429 = sp6[0];	// L2489
          ac_int<33, true> v1430 = v1429;	// L2490
          ac_int<33, true> v1431 = v1430 + 1;	// L2491
          int32_t v1432 = v1431;	// L2492
          sp6[0] = v1432;	// L2493
        } else {
          int32_t v1433 = dcred6[0];	// L2495
          bool v1434 = v1433 > 0;	// L2496
          if (v1434) {	// L2497
            ac_int<26, true> v1435 = cand2;	// L2498
            pw2 = v1435;	// L2499
            int32_t v1436 = dcred6[0];	// L2500
            ac_int<33, true> v1437 = v1436;	// L2501
            ac_int<33, true> v1438 = v1437 - 1;	// L2502
            int32_t v1439 = v1438;	// L2503
            dcred6[0] = v1439;	// L2504
            int32_t v1440 = sp6[0];	// L2505
            ac_int<33, true> v1441 = v1440;	// L2506
            ac_int<33, true> v1442 = v1441 + 1;	// L2507
            int32_t v1443 = v1442;	// L2508
            sp6[0] = v1443;	// L2509
          }
        }
      }
      ac_int<26, true> v1444 = pw2;	// L2513
      v1401.Push(v1444);	// L2514
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
  sc_out< ac_int<8, false> > v1445_radr;
  sc_out<bool> v1445_re;
  sc_in< ac_int<32, true> > v1445_q;
  sc_in<bool> v1445_rrdy;
  sc_out< ac_int<1, false> > v1446_radr;
  sc_out<bool> v1446_re;
  sc_in< ac_int<32, true> > v1446_q;
  sc_in<bool> v1446_rrdy;
  Connections::Out< ac_int<26, false> > v1447;
  Connections::In< ac_int<32, true> > v1448;
  SC_HAS_PROCESS(rdrv_s_0);
  rdrv_s_0(sc_module_name n) : sc_module(n), done("done"), v1447("v1447"), v1448("v1448") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v1445_rd(ac_int<8, false> addr) {
    v1445_radr.write(addr); v1445_re.write(true);
    wait();                    // edge N: address captured
    v1445_re.write(false);
    wait();                    // data valid on this edge
    return v1445_q.read();
  }
  #pragma design modulario <in>
  ac_int<32, true> v1446_rd(ac_int<1, false> addr) {
    v1446_radr.write(addr); v1446_re.write(true);
    wait();                    // edge N: address captured
    v1446_re.write(false);
    wait();                    // data valid on this edge
    return v1446_q.read();
  }
  void run() {
    v1447.Reset();
    v1448.Reset();
    v1445_radr.write(0);
    v1445_re.write(0);
    v1446_radr.write(0);
    v1446_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t dcred7[1];	// L2526
    for (int v1449 = 0; v1449 < 1; v1449++) {	// L2527
      dcred7[v1449] = 0;	// L2527
    }
    int32_t sp7[1];	// L2528
    for (int v1450 = 0; v1450 < 1; v1450++) {	// L2529
      sp7[v1450] = 0;	// L2529
    }
    ac_int<26, false> zp3;	// L2530
    zp3 = 0;	// L2531
    int32_t v1451;
    v1451 = v1446_rd((ac_int<1, false>)(((0) + (0))));	// L2532
    ac_int<33, true> v1452 = v1451;	// L2533
    ac_int<33, true> v1453 = v1452 - 1;	// L2534
    int v1454 = v1453;	// L2535
    for (int v1455 = 0; v1455 < v1454; v1455 += 1) {	// L2536
      ac_int<26, true> v1456 = zp3;	// L2537
      v1447.Push(v1456);	// L2538
    }
    l_S_t_1_t12: for (int t12 = 0; t12 < 215; t12++) {	// L2540
      int32_t v1457 = v1448.Pop();	// L2541
      int32_t v1458 = dcred7[0];	// L2542
      ac_int<33, true> v1459 = v1458;	// L2543
      ac_int<33, true> v1460 = v1457;	// L2544
      ac_int<33, true> v1461 = v1459 + v1460;	// L2545
      int32_t v1462 = v1461;	// L2546
      dcred7[0] = v1462;	// L2547
      ac_int<26, false> pw3;	// L2548
      pw3 = 0;	// L2549
      int32_t v1463 = sp7[0];	// L2550
      bool v1464 = v1463 < 215;	// L2551
      if (v1464) {	// L2552
        ac_int<26, false> cand3;	// L2553
        cand3 = 0;	// L2554
        int32_t v1465 = sp7[0];	// L2555
        int v1466 = v1465;	// L2556
        int32_t v1467;
        v1467 = v1445_rd((ac_int<8, false>)(((0) * 215 + (v1466))));	// L2557
        ac_int<26, false> v1468 = v1467;	// L2558
        ac_int<26, true> v1469 = cand3;	// L2559
        ac_int<26, true> v1470;
        ac_int<26, true> _bs_v1470 = v1469;
        _bs_v1470.set_slc(0, ac_int<26, false>(v1468));
        v1470 = _bs_v1470;	// L2560
        cand3 = v1470;	// L2561
        ac_int<26, true> v1471 = cand3;	// L2562
        bool v1472;
        ac_int<26, true> _bs_v1472 = v1471;
        v1472 = _bs_v1472[25];	// L2563
        int32_t v1473 = v1472;	// L2564
        bool v1474 = v1473 == 0;	// L2565
        if (v1474) {	// L2566
          int32_t v1475 = sp7[0];	// L2567
          ac_int<33, true> v1476 = v1475;	// L2568
          ac_int<33, true> v1477 = v1476 + 1;	// L2569
          int32_t v1478 = v1477;	// L2570
          sp7[0] = v1478;	// L2571
        } else {
          int32_t v1479 = dcred7[0];	// L2573
          bool v1480 = v1479 > 0;	// L2574
          if (v1480) {	// L2575
            ac_int<26, true> v1481 = cand3;	// L2576
            pw3 = v1481;	// L2577
            int32_t v1482 = dcred7[0];	// L2578
            ac_int<33, true> v1483 = v1482;	// L2579
            ac_int<33, true> v1484 = v1483 - 1;	// L2580
            int32_t v1485 = v1484;	// L2581
            dcred7[0] = v1485;	// L2582
            int32_t v1486 = sp7[0];	// L2583
            ac_int<33, true> v1487 = v1486;	// L2584
            ac_int<33, true> v1488 = v1487 + 1;	// L2585
            int32_t v1489 = v1488;	// L2586
            sp7[0] = v1489;	// L2587
          }
        }
      }
      ac_int<26, true> v1490 = pw3;	// L2591
      v1447.Push(v1490);	// L2592
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
  sc_out< ac_int<8, false> > v1491_wadr;
  sc_out< ac_int<32, true> > v1491_d;
  sc_out<bool> v1491_we;
  sc_in<bool> v1491_wrdy;
  sc_out< ac_int<1, false> > v1492_radr;
  sc_out<bool> v1492_re;
  sc_in< ac_int<32, true> > v1492_q;
  sc_in<bool> v1492_rrdy;
  Connections::Out< ac_int<32, true> > v1493;
  Connections::In< ac_int<26, false> > v1494;
  SC_HAS_PROCESS(rclc_w_0);
  rclc_w_0(sc_module_name n) : sc_module(n), done("done"), v1493("v1493"), v1494("v1494") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <out>
  void v1491_wr(ac_int<8, false> addr, ac_int<32, true> val) {
    v1491_wadr.write(addr); v1491_d.write(val);
    v1491_we.write(true);
    wait();
    v1491_we.write(false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v1492_rd(ac_int<1, false> addr) {
    v1492_radr.write(addr); v1492_re.write(true);
    wait();                    // edge N: address captured
    v1492_re.write(false);
    wait();                    // data valid on this edge
    return v1492_q.read();
  }
  void run() {
    v1493.Reset();
    v1494.Reset();
    v1491_wadr.write(0);
    v1491_d.write(0);
    v1491_we.write(0);
    v1492_radr.write(0);
    v1492_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t k6[1];	// L2606
    for (int v1495 = 0; v1495 < 1; v1495++) {	// L2607
      k6[v1495] = 0;	// L2607
    }
    int32_t cret4[1];	// L2608
    for (int v1496 = 0; v1496 < 1; v1496++) {	// L2609
      cret4[v1496] = 0;	// L2609
    }
    int32_t zc4;	// L2610
    zc4 = 0;	// L2611
    int32_t v1497;
    v1497 = v1492_rd((ac_int<1, false>)(((0) + (0))));	// L2612
    ac_int<33, true> v1498 = v1497;	// L2613
    ac_int<33, true> v1499 = v1498 - 1;	// L2614
    int v1500 = v1499;	// L2615
    for (int v1501 = 0; v1501 < v1500; v1501 += 1) {	// L2616
      int32_t v1502 = zc4;	// L2617
      v1493.Push(v1502);	// L2618
    }
    cret4[0] = 2;	// L2620
    int32_t v1503 = cret4[0];	// L2621
    v1493.Push(v1503);	// L2622
    l_S_t_1_t13: for (int t13 = 0; t13 < 215; t13++) {	// L2623
      ac_int<26, false> v1504 = v1494.Pop();	// L2624
      ac_int<26, false> pw4;	// L2625
      pw4 = v1504;	// L2626
      cret4[0] = 0;	// L2627
      ac_int<26, true> v1505 = pw4;	// L2628
      bool v1506;
      ac_int<26, true> _bs_v1506 = v1505;
      v1506 = _bs_v1506[25];	// L2629
      int32_t v1507 = v1506;	// L2630
      bool v1508 = v1507 == 1;	// L2631
      if (v1508) {	// L2632
        cret4[0] = 1;	// L2633
        int32_t v1509 = k6[0];	// L2634
        bool v1510 = v1509 < 215;	// L2635
        if (v1510) {	// L2636
          ac_int<26, true> v1511 = pw4;	// L2637
          int32_t v1512 = v1511;	// L2638
          int32_t v1513 = v1512 & 67108863;	// L2639
          int32_t v1514 = k6[0];	// L2640
          int v1515 = v1514;	// L2641
          v1491_wr((ac_int<8, false>)(((0) * 215 + (v1515))), v1513);	// L2642
          int32_t v1516 = k6[0];	// L2643
          ac_int<33, true> v1517 = v1516;	// L2644
          ac_int<33, true> v1518 = v1517 + 1;	// L2645
          int32_t v1519 = v1518;	// L2646
          k6[0] = v1519;	// L2647
        }
      }
      int32_t v1520 = cret4[0];	// L2650
      v1493.Push(v1520);	// L2651
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
  sc_out< ac_int<8, false> > v1521_wadr;
  sc_out< ac_int<32, true> > v1521_d;
  sc_out<bool> v1521_we;
  sc_in<bool> v1521_wrdy;
  sc_out< ac_int<1, false> > v1522_radr;
  sc_out<bool> v1522_re;
  sc_in< ac_int<32, true> > v1522_q;
  sc_in<bool> v1522_rrdy;
  Connections::Out< ac_int<32, true> > v1523;
  Connections::In< ac_int<26, false> > v1524;
  SC_HAS_PROCESS(rclc_e_0);
  rclc_e_0(sc_module_name n) : sc_module(n), done("done"), v1523("v1523"), v1524("v1524") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <out>
  void v1521_wr(ac_int<8, false> addr, ac_int<32, true> val) {
    v1521_wadr.write(addr); v1521_d.write(val);
    v1521_we.write(true);
    wait();
    v1521_we.write(false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v1522_rd(ac_int<1, false> addr) {
    v1522_radr.write(addr); v1522_re.write(true);
    wait();                    // edge N: address captured
    v1522_re.write(false);
    wait();                    // data valid on this edge
    return v1522_q.read();
  }
  void run() {
    v1523.Reset();
    v1524.Reset();
    v1521_wadr.write(0);
    v1521_d.write(0);
    v1521_we.write(0);
    v1522_radr.write(0);
    v1522_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t k7[1];	// L2665
    for (int v1525 = 0; v1525 < 1; v1525++) {	// L2666
      k7[v1525] = 0;	// L2666
    }
    int32_t cret5[1];	// L2667
    for (int v1526 = 0; v1526 < 1; v1526++) {	// L2668
      cret5[v1526] = 0;	// L2668
    }
    int32_t zc5;	// L2669
    zc5 = 0;	// L2670
    int32_t v1527;
    v1527 = v1522_rd((ac_int<1, false>)(((0) + (0))));	// L2671
    ac_int<33, true> v1528 = v1527;	// L2672
    ac_int<33, true> v1529 = v1528 - 1;	// L2673
    int v1530 = v1529;	// L2674
    for (int v1531 = 0; v1531 < v1530; v1531 += 1) {	// L2675
      int32_t v1532 = zc5;	// L2676
      v1523.Push(v1532);	// L2677
    }
    cret5[0] = 2;	// L2679
    int32_t v1533 = cret5[0];	// L2680
    v1523.Push(v1533);	// L2681
    l_S_t_1_t14: for (int t14 = 0; t14 < 215; t14++) {	// L2682
      ac_int<26, false> v1534 = v1524.Pop();	// L2683
      ac_int<26, false> pw5;	// L2684
      pw5 = v1534;	// L2685
      cret5[0] = 0;	// L2686
      ac_int<26, true> v1535 = pw5;	// L2687
      bool v1536;
      ac_int<26, true> _bs_v1536 = v1535;
      v1536 = _bs_v1536[25];	// L2688
      int32_t v1537 = v1536;	// L2689
      bool v1538 = v1537 == 1;	// L2690
      if (v1538) {	// L2691
        cret5[0] = 1;	// L2692
        int32_t v1539 = k7[0];	// L2693
        bool v1540 = v1539 < 215;	// L2694
        if (v1540) {	// L2695
          ac_int<26, true> v1541 = pw5;	// L2696
          int32_t v1542 = v1541;	// L2697
          int32_t v1543 = v1542 & 67108863;	// L2698
          int32_t v1544 = k7[0];	// L2699
          int v1545 = v1544;	// L2700
          v1521_wr((ac_int<8, false>)(((0) * 215 + (v1545))), v1543);	// L2701
          int32_t v1546 = k7[0];	// L2702
          ac_int<33, true> v1547 = v1546;	// L2703
          ac_int<33, true> v1548 = v1547 + 1;	// L2704
          int32_t v1549 = v1548;	// L2705
          k7[0] = v1549;	// L2706
        }
      }
      int32_t v1550 = cret5[0];	// L2709
      v1523.Push(v1550);	// L2710
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
  sc_out< ac_int<8, false> > v1551_wadr;
  sc_out< ac_int<32, true> > v1551_d;
  sc_out<bool> v1551_we;
  sc_in<bool> v1551_wrdy;
  sc_out< ac_int<1, false> > v1552_radr;
  sc_out<bool> v1552_re;
  sc_in< ac_int<32, true> > v1552_q;
  sc_in<bool> v1552_rrdy;
  Connections::Out< ac_int<32, true> > v1553;
  Connections::In< ac_int<26, false> > v1554;
  SC_HAS_PROCESS(rclc_n_0);
  rclc_n_0(sc_module_name n) : sc_module(n), done("done"), v1553("v1553"), v1554("v1554") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <out>
  void v1551_wr(ac_int<8, false> addr, ac_int<32, true> val) {
    v1551_wadr.write(addr); v1551_d.write(val);
    v1551_we.write(true);
    wait();
    v1551_we.write(false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v1552_rd(ac_int<1, false> addr) {
    v1552_radr.write(addr); v1552_re.write(true);
    wait();                    // edge N: address captured
    v1552_re.write(false);
    wait();                    // data valid on this edge
    return v1552_q.read();
  }
  void run() {
    v1553.Reset();
    v1554.Reset();
    v1551_wadr.write(0);
    v1551_d.write(0);
    v1551_we.write(0);
    v1552_radr.write(0);
    v1552_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t k8[1];	// L2724
    for (int v1555 = 0; v1555 < 1; v1555++) {	// L2725
      k8[v1555] = 0;	// L2725
    }
    int32_t cret6[1];	// L2726
    for (int v1556 = 0; v1556 < 1; v1556++) {	// L2727
      cret6[v1556] = 0;	// L2727
    }
    int32_t zc6;	// L2728
    zc6 = 0;	// L2729
    int32_t v1557;
    v1557 = v1552_rd((ac_int<1, false>)(((0) + (0))));	// L2730
    ac_int<33, true> v1558 = v1557;	// L2731
    ac_int<33, true> v1559 = v1558 - 1;	// L2732
    int v1560 = v1559;	// L2733
    for (int v1561 = 0; v1561 < v1560; v1561 += 1) {	// L2734
      int32_t v1562 = zc6;	// L2735
      v1553.Push(v1562);	// L2736
    }
    cret6[0] = 2;	// L2738
    int32_t v1563 = cret6[0];	// L2739
    v1553.Push(v1563);	// L2740
    l_S_t_1_t15: for (int t15 = 0; t15 < 215; t15++) {	// L2741
      ac_int<26, false> v1564 = v1554.Pop();	// L2742
      ac_int<26, false> pw6;	// L2743
      pw6 = v1564;	// L2744
      cret6[0] = 0;	// L2745
      ac_int<26, true> v1565 = pw6;	// L2746
      bool v1566;
      ac_int<26, true> _bs_v1566 = v1565;
      v1566 = _bs_v1566[25];	// L2747
      int32_t v1567 = v1566;	// L2748
      bool v1568 = v1567 == 1;	// L2749
      if (v1568) {	// L2750
        cret6[0] = 1;	// L2751
        int32_t v1569 = k8[0];	// L2752
        bool v1570 = v1569 < 215;	// L2753
        if (v1570) {	// L2754
          ac_int<26, true> v1571 = pw6;	// L2755
          int32_t v1572 = v1571;	// L2756
          int32_t v1573 = v1572 & 67108863;	// L2757
          int32_t v1574 = k8[0];	// L2758
          int v1575 = v1574;	// L2759
          v1551_wr((ac_int<8, false>)(((0) * 215 + (v1575))), v1573);	// L2760
          int32_t v1576 = k8[0];	// L2761
          ac_int<33, true> v1577 = v1576;	// L2762
          ac_int<33, true> v1578 = v1577 + 1;	// L2763
          int32_t v1579 = v1578;	// L2764
          k8[0] = v1579;	// L2765
        }
      }
      int32_t v1580 = cret6[0];	// L2768
      v1553.Push(v1580);	// L2769
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
  sc_out< ac_int<8, false> > v1581_wadr;
  sc_out< ac_int<32, true> > v1581_d;
  sc_out<bool> v1581_we;
  sc_in<bool> v1581_wrdy;
  sc_out< ac_int<1, false> > v1582_radr;
  sc_out<bool> v1582_re;
  sc_in< ac_int<32, true> > v1582_q;
  sc_in<bool> v1582_rrdy;
  Connections::Out< ac_int<32, true> > v1583;
  Connections::In< ac_int<26, false> > v1584;
  SC_HAS_PROCESS(rclc_s_0);
  rclc_s_0(sc_module_name n) : sc_module(n), done("done"), v1583("v1583"), v1584("v1584") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  #pragma design modulario <out>
  void v1581_wr(ac_int<8, false> addr, ac_int<32, true> val) {
    v1581_wadr.write(addr); v1581_d.write(val);
    v1581_we.write(true);
    wait();
    v1581_we.write(false);
  }
  #pragma design modulario <in>
  ac_int<32, true> v1582_rd(ac_int<1, false> addr) {
    v1582_radr.write(addr); v1582_re.write(true);
    wait();                    // edge N: address captured
    v1582_re.write(false);
    wait();                    // data valid on this edge
    return v1582_q.read();
  }
  void run() {
    v1583.Reset();
    v1584.Reset();
    v1581_wadr.write(0);
    v1581_d.write(0);
    v1581_we.write(0);
    v1582_radr.write(0);
    v1582_re.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    int32_t k9[1];	// L2783
    for (int v1585 = 0; v1585 < 1; v1585++) {	// L2784
      k9[v1585] = 0;	// L2784
    }
    int32_t cret7[1];	// L2785
    for (int v1586 = 0; v1586 < 1; v1586++) {	// L2786
      cret7[v1586] = 0;	// L2786
    }
    int32_t zc7;	// L2787
    zc7 = 0;	// L2788
    int32_t v1587;
    v1587 = v1582_rd((ac_int<1, false>)(((0) + (0))));	// L2789
    ac_int<33, true> v1588 = v1587;	// L2790
    ac_int<33, true> v1589 = v1588 - 1;	// L2791
    int v1590 = v1589;	// L2792
    for (int v1591 = 0; v1591 < v1590; v1591 += 1) {	// L2793
      int32_t v1592 = zc7;	// L2794
      v1583.Push(v1592);	// L2795
    }
    cret7[0] = 2;	// L2797
    int32_t v1593 = cret7[0];	// L2798
    v1583.Push(v1593);	// L2799
    l_S_t_1_t16: for (int t16 = 0; t16 < 215; t16++) {	// L2800
      ac_int<26, false> v1594 = v1584.Pop();	// L2801
      ac_int<26, false> pw7;	// L2802
      pw7 = v1594;	// L2803
      cret7[0] = 0;	// L2804
      ac_int<26, true> v1595 = pw7;	// L2805
      bool v1596;
      ac_int<26, true> _bs_v1596 = v1595;
      v1596 = _bs_v1596[25];	// L2806
      int32_t v1597 = v1596;	// L2807
      bool v1598 = v1597 == 1;	// L2808
      if (v1598) {	// L2809
        cret7[0] = 1;	// L2810
        int32_t v1599 = k9[0];	// L2811
        bool v1600 = v1599 < 215;	// L2812
        if (v1600) {	// L2813
          ac_int<26, true> v1601 = pw7;	// L2814
          int32_t v1602 = v1601;	// L2815
          int32_t v1603 = v1602 & 67108863;	// L2816
          int32_t v1604 = k9[0];	// L2817
          int v1605 = v1604;	// L2818
          v1581_wr((ac_int<8, false>)(((0) * 215 + (v1605))), v1603);	// L2819
          int32_t v1606 = k9[0];	// L2820
          ac_int<33, true> v1607 = v1606;	// L2821
          ac_int<33, true> v1608 = v1607 + 1;	// L2822
          int32_t v1609 = v1608;	// L2823
          k9[0] = v1609;	// L2824
        }
      }
      int32_t v1610 = cret7[0];	// L2827
      v1583.Push(v1610);	// L2828
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
  Connections::Combinational< ac_int<17, false> > v1632_in;
  Connections::Combinational< ac_int<17, false> > v1632_out;
  Connections::Fifo< ac_int<17, false>, 8 > v1632_fifo;
  Connections::Combinational< ac_int<17, false> > v1633_in;
  Connections::Combinational< ac_int<17, false> > v1633_out;
  Connections::Fifo< ac_int<17, false>, 8 > v1633_fifo;
  Connections::Combinational< ac_int<17, false> > v1634_in;
  Connections::Combinational< ac_int<17, false> > v1634_out;
  Connections::Fifo< ac_int<17, false>, 8 > v1634_fifo;
  Connections::Combinational< ac_int<17, false> > v1635_in;
  Connections::Combinational< ac_int<17, false> > v1635_out;
  Connections::Fifo< ac_int<17, false>, 8 > v1635_fifo;
  Connections::Combinational< ac_int<17, false> > v1636_in;
  Connections::Combinational< ac_int<17, false> > v1636_out;
  Connections::Fifo< ac_int<17, false>, 8 > v1636_fifo;
  Connections::Combinational< ac_int<17, false> > v1637_in;
  Connections::Combinational< ac_int<17, false> > v1637_out;
  Connections::Fifo< ac_int<17, false>, 8 > v1637_fifo;
  Connections::Combinational< ac_int<17, false> > v1638_in;
  Connections::Combinational< ac_int<17, false> > v1638_out;
  Connections::Fifo< ac_int<17, false>, 8 > v1638_fifo;
  Connections::Combinational< ac_int<17, false> > v1639_in;
  Connections::Combinational< ac_int<17, false> > v1639_out;
  Connections::Fifo< ac_int<17, false>, 8 > v1639_fifo;
  Connections::Combinational< ac_int<26, false> > v1640_in;
  Connections::Combinational< ac_int<26, false> > v1640_out;
  Connections::Fifo< ac_int<26, false>, 8 > v1640_fifo;
  Connections::Combinational< ac_int<26, false> > v1641_in;
  Connections::Combinational< ac_int<26, false> > v1641_out;
  Connections::Fifo< ac_int<26, false>, 8 > v1641_fifo;
  Connections::Combinational< ac_int<26, false> > v1642_in;
  Connections::Combinational< ac_int<26, false> > v1642_out;
  Connections::Fifo< ac_int<26, false>, 8 > v1642_fifo;
  Connections::Combinational< ac_int<26, false> > v1643_in;
  Connections::Combinational< ac_int<26, false> > v1643_out;
  Connections::Fifo< ac_int<26, false>, 8 > v1643_fifo;
  Connections::Combinational< ac_int<26, false> > v1644_in;
  Connections::Combinational< ac_int<26, false> > v1644_out;
  Connections::Fifo< ac_int<26, false>, 8 > v1644_fifo;
  Connections::Combinational< ac_int<26, false> > v1645_in;
  Connections::Combinational< ac_int<26, false> > v1645_out;
  Connections::Fifo< ac_int<26, false>, 8 > v1645_fifo;
  Connections::Combinational< ac_int<26, false> > v1646_in;
  Connections::Combinational< ac_int<26, false> > v1646_out;
  Connections::Fifo< ac_int<26, false>, 8 > v1646_fifo;
  Connections::Combinational< ac_int<26, false> > v1647_in;
  Connections::Combinational< ac_int<26, false> > v1647_out;
  Connections::Fifo< ac_int<26, false>, 8 > v1647_fifo;
  Connections::Combinational< ac_int<32, true> > v1648_in;
  Connections::Combinational< ac_int<32, true> > v1648_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1648_fifo;
  Connections::Combinational< ac_int<32, true> > v1649_in;
  Connections::Combinational< ac_int<32, true> > v1649_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1649_fifo;
  Connections::Combinational< ac_int<32, true> > v1650_in;
  Connections::Combinational< ac_int<32, true> > v1650_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1650_fifo;
  Connections::Combinational< ac_int<32, true> > v1651_in;
  Connections::Combinational< ac_int<32, true> > v1651_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1651_fifo;
  Connections::Combinational< ac_int<32, true> > v1652_in;
  Connections::Combinational< ac_int<32, true> > v1652_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1652_fifo;
  Connections::Combinational< ac_int<32, true> > v1653_in;
  Connections::Combinational< ac_int<32, true> > v1653_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1653_fifo;
  Connections::Combinational< ac_int<32, true> > v1654_in;
  Connections::Combinational< ac_int<32, true> > v1654_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1654_fifo;
  Connections::Combinational< ac_int<32, true> > v1655_in;
  Connections::Combinational< ac_int<32, true> > v1655_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1655_fifo;
  Connections::Combinational< ac_int<32, true> > v1656_in;
  Connections::Combinational< ac_int<32, true> > v1656_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1656_fifo;
  Connections::Combinational< ac_int<32, true> > v1657_in;
  Connections::Combinational< ac_int<32, true> > v1657_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1657_fifo;
  Connections::Combinational< ac_int<32, true> > v1658_in;
  Connections::Combinational< ac_int<32, true> > v1658_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1658_fifo;
  Connections::Combinational< ac_int<32, true> > v1659_in;
  Connections::Combinational< ac_int<32, true> > v1659_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1659_fifo;
  Connections::Combinational< ac_int<32, true> > v1660_in;
  Connections::Combinational< ac_int<32, true> > v1660_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1660_fifo;
  Connections::Combinational< ac_int<32, true> > v1661_in;
  Connections::Combinational< ac_int<32, true> > v1661_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1661_fifo;
  Connections::Combinational< ac_int<32, true> > v1662_in;
  Connections::Combinational< ac_int<32, true> > v1662_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1662_fifo;
  Connections::Combinational< ac_int<32, true> > v1663_in;
  Connections::Combinational< ac_int<32, true> > v1663_out;
  Connections::Fifo< ac_int<32, true>, 8 > v1663_fifo;
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
  sc_out< ac_int<8, false> > v1611_radr;
  sc_out<bool> v1611_re;
  sc_in< half > v1611_q;
  sc_in<bool> v1611_rrdy;
  sc_out< ac_int<8, false> > v1627_radr;
  sc_out<bool> v1627_re;
  sc_in< ac_int<32, true> > v1627_q;
  sc_in<bool> v1627_rrdy;
  sc_out< ac_int<8, false> > v1612_radr;
  sc_out<bool> v1612_re;
  sc_in< half > v1612_q;
  sc_in<bool> v1612_rrdy;
  sc_out< ac_int<8, false> > v1628_radr;
  sc_out<bool> v1628_re;
  sc_in< ac_int<32, true> > v1628_q;
  sc_in<bool> v1628_rrdy;
  sc_out< ac_int<8, false> > v1613_radr;
  sc_out<bool> v1613_re;
  sc_in< half > v1613_q;
  sc_in<bool> v1613_rrdy;
  sc_out< ac_int<8, false> > v1629_radr;
  sc_out<bool> v1629_re;
  sc_in< ac_int<32, true> > v1629_q;
  sc_in<bool> v1629_rrdy;
  sc_out< ac_int<8, false> > v1614_radr;
  sc_out<bool> v1614_re;
  sc_in< half > v1614_q;
  sc_in<bool> v1614_rrdy;
  sc_out< ac_int<8, false> > v1630_radr;
  sc_out<bool> v1630_re;
  sc_in< ac_int<32, true> > v1630_q;
  sc_in<bool> v1630_rrdy;
  sc_out< ac_int<8, false> > v1615_wadr;
  sc_out< half > v1615_d;
  sc_out<bool> v1615_we;
  sc_in<bool> v1615_wrdy;
  sc_out< ac_int<8, false> > v1616_wadr;
  sc_out< half > v1616_d;
  sc_out<bool> v1616_we;
  sc_in<bool> v1616_wrdy;
  sc_out< ac_int<8, false> > v1617_wadr;
  sc_out< half > v1617_d;
  sc_out<bool> v1617_we;
  sc_in<bool> v1617_wrdy;
  sc_out< ac_int<8, false> > v1618_wadr;
  sc_out< half > v1618_d;
  sc_out<bool> v1618_we;
  sc_in<bool> v1618_wrdy;
  sc_out< ac_int<8, false> > v1619_radr;
  sc_out<bool> v1619_re;
  sc_in< ac_int<32, true> > v1619_q;
  sc_in<bool> v1619_rrdy;
  sc_out< ac_int<8, false> > v1620_radr;
  sc_out<bool> v1620_re;
  sc_in< ac_int<32, true> > v1620_q;
  sc_in<bool> v1620_rrdy;
  sc_out< ac_int<8, false> > v1621_radr;
  sc_out<bool> v1621_re;
  sc_in< ac_int<32, true> > v1621_q;
  sc_in<bool> v1621_rrdy;
  sc_out< ac_int<8, false> > v1622_radr;
  sc_out<bool> v1622_re;
  sc_in< ac_int<32, true> > v1622_q;
  sc_in<bool> v1622_rrdy;
  sc_out< ac_int<8, false> > v1623_wadr;
  sc_out< ac_int<32, true> > v1623_d;
  sc_out<bool> v1623_we;
  sc_in<bool> v1623_wrdy;
  sc_out< ac_int<8, false> > v1624_wadr;
  sc_out< ac_int<32, true> > v1624_d;
  sc_out<bool> v1624_we;
  sc_in<bool> v1624_wrdy;
  sc_out< ac_int<8, false> > v1625_wadr;
  sc_out< ac_int<32, true> > v1625_d;
  sc_out<bool> v1625_we;
  sc_in<bool> v1625_wrdy;
  sc_out< ac_int<8, false> > v1626_wadr;
  sc_out< ac_int<32, true> > v1626_d;
  sc_out<bool> v1626_we;
  sc_in<bool> v1626_wrdy;
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
  SC_CTOR(top) : v1632_in("v1632_in"), v1632_out("v1632_out"), v1632_fifo("v1632_fifo"), v1633_in("v1633_in"), v1633_out("v1633_out"), v1633_fifo("v1633_fifo"), v1634_in("v1634_in"), v1634_out("v1634_out"), v1634_fifo("v1634_fifo"), v1635_in("v1635_in"), v1635_out("v1635_out"), v1635_fifo("v1635_fifo"), v1636_in("v1636_in"), v1636_out("v1636_out"), v1636_fifo("v1636_fifo"), v1637_in("v1637_in"), v1637_out("v1637_out"), v1637_fifo("v1637_fifo"), v1638_in("v1638_in"), v1638_out("v1638_out"), v1638_fifo("v1638_fifo"), v1639_in("v1639_in"), v1639_out("v1639_out"), v1639_fifo("v1639_fifo"), v1640_in("v1640_in"), v1640_out("v1640_out"), v1640_fifo("v1640_fifo"), v1641_in("v1641_in"), v1641_out("v1641_out"), v1641_fifo("v1641_fifo"), v1642_in("v1642_in"), v1642_out("v1642_out"), v1642_fifo("v1642_fifo"), v1643_in("v1643_in"), v1643_out("v1643_out"), v1643_fifo("v1643_fifo"), v1644_in("v1644_in"), v1644_out("v1644_out"), v1644_fifo("v1644_fifo"), v1645_in("v1645_in"), v1645_out("v1645_out"), v1645_fifo("v1645_fifo"), v1646_in("v1646_in"), v1646_out("v1646_out"), v1646_fifo("v1646_fifo"), v1647_in("v1647_in"), v1647_out("v1647_out"), v1647_fifo("v1647_fifo"), v1648_in("v1648_in"), v1648_out("v1648_out"), v1648_fifo("v1648_fifo"), v1649_in("v1649_in"), v1649_out("v1649_out"), v1649_fifo("v1649_fifo"), v1650_in("v1650_in"), v1650_out("v1650_out"), v1650_fifo("v1650_fifo"), v1651_in("v1651_in"), v1651_out("v1651_out"), v1651_fifo("v1651_fifo"), v1652_in("v1652_in"), v1652_out("v1652_out"), v1652_fifo("v1652_fifo"), v1653_in("v1653_in"), v1653_out("v1653_out"), v1653_fifo("v1653_fifo"), v1654_in("v1654_in"), v1654_out("v1654_out"), v1654_fifo("v1654_fifo"), v1655_in("v1655_in"), v1655_out("v1655_out"), v1655_fifo("v1655_fifo"), v1656_in("v1656_in"), v1656_out("v1656_out"), v1656_fifo("v1656_fifo"), v1657_in("v1657_in"), v1657_out("v1657_out"), v1657_fifo("v1657_fifo"), v1658_in("v1658_in"), v1658_out("v1658_out"), v1658_fifo("v1658_fifo"), v1659_in("v1659_in"), v1659_out("v1659_out"), v1659_fifo("v1659_fifo"), v1660_in("v1660_in"), v1660_out("v1660_out"), v1660_fifo("v1660_fifo"), v1661_in("v1661_in"), v1661_out("v1661_out"), v1661_fifo("v1661_fifo"), v1662_in("v1662_in"), v1662_out("v1662_out"), v1662_fifo("v1662_fifo"), v1663_in("v1663_in"), v1663_out("v1663_out"), v1663_fifo("v1663_fifo"), u0("u0"), u1("u1"), u2("u2"), u3("u3"), u4("u4"), u5("u5"), u6("u6"), u7("u7"), u8("u8"), u9("u9"), u10("u10"), u11("u11"), u12("u12"), u13("u13"), u14("u14"), u15("u15"), u16("u16"), mp0_0_mem("mp0_0_mem"), mp1_2_mem("mp1_2_mem"), mp2_2_mem("mp2_2_mem"), mp3_2_mem("mp3_2_mem"), mp4_2_mem("mp4_2_mem"), mp5_1_mem("mp5_1_mem"), mp6_1_mem("mp6_1_mem"), mp7_1_mem("mp7_1_mem"), mp8_1_mem("mp8_1_mem"), mp9_1_mem("mp9_1_mem"), mp10_1_mem("mp10_1_mem"), mp11_1_mem("mp11_1_mem"), mp12_1_mem("mp12_1_mem"), mp13_1_mem("mp13_1_mem"), mp14_1_mem("mp14_1_mem"), mp15_1_mem("mp15_1_mem"), mp16_1_mem("mp16_1_mem") {
    u0.clk(clk);
    u0.rst(rst);
    u0.done(u0_done);
    u0.v0_radr(mp0_0_radr);
    u0.v0_re(mp0_0_re);
    u0.v0_q(mp0_0_q);
    u0.v0_rrdy(mp0_0_rrdy);
    u0.v1(v1641_in);
    u0.v2(v1642_in);
    u0.v3(v1645_in);
    u0.v4(v1646_in);
    u0.v5(v1633_in);
    u0.v6(v1634_in);
    u0.v7(v1637_in);
    u0.v8(v1638_in);
    u0.v9(v1648_in);
    u0.v10(v1651_in);
    u0.v11(v1652_in);
    u0.v12(v1655_in);
    u0.v13(v1656_in);
    u0.v14(v1659_in);
    u0.v15(v1660_in);
    u0.v16(v1663_in);
    u0.v17(v1640_out);
    u0.v18(v1643_out);
    u0.v19(v1644_out);
    u0.v20(v1647_out);
    u0.v21(v1649_out);
    u0.v22(v1650_out);
    u0.v23(v1653_out);
    u0.v24(v1654_out);
    u0.v25(v1662_out);
    u0.v26(v1661_out);
    u0.v27(v1658_out);
    u0.v28(v1657_out);
    u0.v29(v1632_out);
    u0.v30(v1635_out);
    u0.v31(v1636_out);
    u0.v32(v1639_out);
    u1.clk(clk);
    u1.rst(rst);
    u1.done(u1_done);
    u1.v979_radr(v1611_radr);
    u1.v979_re(v1611_re);
    u1.v979_q(v1611_q);
    u1.v979_rrdy(v1611_rrdy);
    u1.v980_radr(v1627_radr);
    u1.v980_re(v1627_re);
    u1.v980_q(v1627_q);
    u1.v980_rrdy(v1627_rrdy);
    u1.v981_radr(mp1_2_radr);
    u1.v981_re(mp1_2_re);
    u1.v981_q(mp1_2_q);
    u1.v981_rrdy(mp1_2_rrdy);
    u1.v982(v1632_in);
    u1.v983(v1656_out);
    u2.clk(clk);
    u2.rst(rst);
    u2.done(u2_done);
    u2.v1031_radr(v1612_radr);
    u2.v1031_re(v1612_re);
    u2.v1031_q(v1612_q);
    u2.v1031_rrdy(v1612_rrdy);
    u2.v1032_radr(v1628_radr);
    u2.v1032_re(v1628_re);
    u2.v1032_q(v1628_q);
    u2.v1032_rrdy(v1628_rrdy);
    u2.v1033_radr(mp2_2_radr);
    u2.v1033_re(mp2_2_re);
    u2.v1033_q(mp2_2_q);
    u2.v1033_rrdy(mp2_2_rrdy);
    u2.v1034(v1635_in);
    u2.v1035(v1659_out);
    u3.clk(clk);
    u3.rst(rst);
    u3.done(u3_done);
    u3.v1083_radr(v1613_radr);
    u3.v1083_re(v1613_re);
    u3.v1083_q(v1613_q);
    u3.v1083_rrdy(v1613_rrdy);
    u3.v1084_radr(v1629_radr);
    u3.v1084_re(v1629_re);
    u3.v1084_q(v1629_q);
    u3.v1084_rrdy(v1629_rrdy);
    u3.v1085_radr(mp3_2_radr);
    u3.v1085_re(mp3_2_re);
    u3.v1085_q(mp3_2_q);
    u3.v1085_rrdy(mp3_2_rrdy);
    u3.v1086(v1636_in);
    u3.v1087(v1660_out);
    u4.clk(clk);
    u4.rst(rst);
    u4.done(u4_done);
    u4.v1135_radr(v1614_radr);
    u4.v1135_re(v1614_re);
    u4.v1135_q(v1614_q);
    u4.v1135_rrdy(v1614_rrdy);
    u4.v1136_radr(v1630_radr);
    u4.v1136_re(v1630_re);
    u4.v1136_q(v1630_q);
    u4.v1136_rrdy(v1630_rrdy);
    u4.v1137_radr(mp4_2_radr);
    u4.v1137_re(mp4_2_re);
    u4.v1137_q(mp4_2_q);
    u4.v1137_rrdy(mp4_2_rrdy);
    u4.v1138(v1639_in);
    u4.v1139(v1663_out);
    u5.clk(clk);
    u5.rst(rst);
    u5.done(u5_done);
    u5.v1187_wadr(v1615_wadr);
    u5.v1187_d(v1615_d);
    u5.v1187_we(v1615_we);
    u5.v1187_wrdy(v1615_wrdy);
    u5.v1188_radr(mp5_1_radr);
    u5.v1188_re(mp5_1_re);
    u5.v1188_q(mp5_1_q);
    u5.v1188_rrdy(mp5_1_rrdy);
    u5.v1189(v1658_in);
    u5.v1190(v1634_out);
    u6.clk(clk);
    u6.rst(rst);
    u6.done(u6_done);
    u6.v1217_wadr(v1616_wadr);
    u6.v1217_d(v1616_d);
    u6.v1217_we(v1616_we);
    u6.v1217_wrdy(v1616_wrdy);
    u6.v1218_radr(mp6_1_radr);
    u6.v1218_re(mp6_1_re);
    u6.v1218_q(mp6_1_q);
    u6.v1218_rrdy(mp6_1_rrdy);
    u6.v1219(v1657_in);
    u6.v1220(v1633_out);
    u7.clk(clk);
    u7.rst(rst);
    u7.done(u7_done);
    u7.v1247_wadr(v1617_wadr);
    u7.v1247_d(v1617_d);
    u7.v1247_we(v1617_we);
    u7.v1247_wrdy(v1617_wrdy);
    u7.v1248_radr(mp7_1_radr);
    u7.v1248_re(mp7_1_re);
    u7.v1248_q(mp7_1_q);
    u7.v1248_rrdy(mp7_1_rrdy);
    u7.v1249(v1662_in);
    u7.v1250(v1638_out);
    u8.clk(clk);
    u8.rst(rst);
    u8.done(u8_done);
    u8.v1277_wadr(v1618_wadr);
    u8.v1277_d(v1618_d);
    u8.v1277_we(v1618_we);
    u8.v1277_wrdy(v1618_wrdy);
    u8.v1278_radr(mp8_1_radr);
    u8.v1278_re(mp8_1_re);
    u8.v1278_q(mp8_1_q);
    u8.v1278_rrdy(mp8_1_rrdy);
    u8.v1279(v1661_in);
    u8.v1280(v1637_out);
    u9.clk(clk);
    u9.rst(rst);
    u9.done(u9_done);
    u9.v1307_radr(v1619_radr);
    u9.v1307_re(v1619_re);
    u9.v1307_q(v1619_q);
    u9.v1307_rrdy(v1619_rrdy);
    u9.v1308_radr(mp9_1_radr);
    u9.v1308_re(mp9_1_re);
    u9.v1308_q(mp9_1_q);
    u9.v1308_rrdy(mp9_1_rrdy);
    u9.v1309(v1640_in);
    u9.v1310(v1648_out);
    u10.clk(clk);
    u10.rst(rst);
    u10.done(u10_done);
    u10.v1353_radr(v1620_radr);
    u10.v1353_re(v1620_re);
    u10.v1353_q(v1620_q);
    u10.v1353_rrdy(v1620_rrdy);
    u10.v1354_radr(mp10_1_radr);
    u10.v1354_re(mp10_1_re);
    u10.v1354_q(mp10_1_q);
    u10.v1354_rrdy(mp10_1_rrdy);
    u10.v1355(v1643_in);
    u10.v1356(v1651_out);
    u11.clk(clk);
    u11.rst(rst);
    u11.done(u11_done);
    u11.v1399_radr(v1621_radr);
    u11.v1399_re(v1621_re);
    u11.v1399_q(v1621_q);
    u11.v1399_rrdy(v1621_rrdy);
    u11.v1400_radr(mp11_1_radr);
    u11.v1400_re(mp11_1_re);
    u11.v1400_q(mp11_1_q);
    u11.v1400_rrdy(mp11_1_rrdy);
    u11.v1401(v1644_in);
    u11.v1402(v1652_out);
    u12.clk(clk);
    u12.rst(rst);
    u12.done(u12_done);
    u12.v1445_radr(v1622_radr);
    u12.v1445_re(v1622_re);
    u12.v1445_q(v1622_q);
    u12.v1445_rrdy(v1622_rrdy);
    u12.v1446_radr(mp12_1_radr);
    u12.v1446_re(mp12_1_re);
    u12.v1446_q(mp12_1_q);
    u12.v1446_rrdy(mp12_1_rrdy);
    u12.v1447(v1647_in);
    u12.v1448(v1655_out);
    u13.clk(clk);
    u13.rst(rst);
    u13.done(u13_done);
    u13.v1491_wadr(v1623_wadr);
    u13.v1491_d(v1623_d);
    u13.v1491_we(v1623_we);
    u13.v1491_wrdy(v1623_wrdy);
    u13.v1492_radr(mp13_1_radr);
    u13.v1492_re(mp13_1_re);
    u13.v1492_q(mp13_1_q);
    u13.v1492_rrdy(mp13_1_rrdy);
    u13.v1493(v1650_in);
    u13.v1494(v1642_out);
    u14.clk(clk);
    u14.rst(rst);
    u14.done(u14_done);
    u14.v1521_wadr(v1624_wadr);
    u14.v1521_d(v1624_d);
    u14.v1521_we(v1624_we);
    u14.v1521_wrdy(v1624_wrdy);
    u14.v1522_radr(mp14_1_radr);
    u14.v1522_re(mp14_1_re);
    u14.v1522_q(mp14_1_q);
    u14.v1522_rrdy(mp14_1_rrdy);
    u14.v1523(v1649_in);
    u14.v1524(v1641_out);
    u15.clk(clk);
    u15.rst(rst);
    u15.done(u15_done);
    u15.v1551_wadr(v1625_wadr);
    u15.v1551_d(v1625_d);
    u15.v1551_we(v1625_we);
    u15.v1551_wrdy(v1625_wrdy);
    u15.v1552_radr(mp15_1_radr);
    u15.v1552_re(mp15_1_re);
    u15.v1552_q(mp15_1_q);
    u15.v1552_rrdy(mp15_1_rrdy);
    u15.v1553(v1654_in);
    u15.v1554(v1646_out);
    u16.clk(clk);
    u16.rst(rst);
    u16.done(u16_done);
    u16.v1581_wadr(v1626_wadr);
    u16.v1581_d(v1626_d);
    u16.v1581_we(v1626_we);
    u16.v1581_wrdy(v1626_wrdy);
    u16.v1582_radr(mp16_1_radr);
    u16.v1582_re(mp16_1_re);
    u16.v1582_q(mp16_1_q);
    u16.v1582_rrdy(mp16_1_rrdy);
    u16.v1583(v1653_in);
    u16.v1584(v1645_out);
    v1632_fifo.clk(clk);
    v1632_fifo.rst(rst);
    v1632_fifo.enq(v1632_in);
    v1632_fifo.deq(v1632_out);
    v1633_fifo.clk(clk);
    v1633_fifo.rst(rst);
    v1633_fifo.enq(v1633_in);
    v1633_fifo.deq(v1633_out);
    v1634_fifo.clk(clk);
    v1634_fifo.rst(rst);
    v1634_fifo.enq(v1634_in);
    v1634_fifo.deq(v1634_out);
    v1635_fifo.clk(clk);
    v1635_fifo.rst(rst);
    v1635_fifo.enq(v1635_in);
    v1635_fifo.deq(v1635_out);
    v1636_fifo.clk(clk);
    v1636_fifo.rst(rst);
    v1636_fifo.enq(v1636_in);
    v1636_fifo.deq(v1636_out);
    v1637_fifo.clk(clk);
    v1637_fifo.rst(rst);
    v1637_fifo.enq(v1637_in);
    v1637_fifo.deq(v1637_out);
    v1638_fifo.clk(clk);
    v1638_fifo.rst(rst);
    v1638_fifo.enq(v1638_in);
    v1638_fifo.deq(v1638_out);
    v1639_fifo.clk(clk);
    v1639_fifo.rst(rst);
    v1639_fifo.enq(v1639_in);
    v1639_fifo.deq(v1639_out);
    v1640_fifo.clk(clk);
    v1640_fifo.rst(rst);
    v1640_fifo.enq(v1640_in);
    v1640_fifo.deq(v1640_out);
    v1641_fifo.clk(clk);
    v1641_fifo.rst(rst);
    v1641_fifo.enq(v1641_in);
    v1641_fifo.deq(v1641_out);
    v1642_fifo.clk(clk);
    v1642_fifo.rst(rst);
    v1642_fifo.enq(v1642_in);
    v1642_fifo.deq(v1642_out);
    v1643_fifo.clk(clk);
    v1643_fifo.rst(rst);
    v1643_fifo.enq(v1643_in);
    v1643_fifo.deq(v1643_out);
    v1644_fifo.clk(clk);
    v1644_fifo.rst(rst);
    v1644_fifo.enq(v1644_in);
    v1644_fifo.deq(v1644_out);
    v1645_fifo.clk(clk);
    v1645_fifo.rst(rst);
    v1645_fifo.enq(v1645_in);
    v1645_fifo.deq(v1645_out);
    v1646_fifo.clk(clk);
    v1646_fifo.rst(rst);
    v1646_fifo.enq(v1646_in);
    v1646_fifo.deq(v1646_out);
    v1647_fifo.clk(clk);
    v1647_fifo.rst(rst);
    v1647_fifo.enq(v1647_in);
    v1647_fifo.deq(v1647_out);
    v1648_fifo.clk(clk);
    v1648_fifo.rst(rst);
    v1648_fifo.enq(v1648_in);
    v1648_fifo.deq(v1648_out);
    v1649_fifo.clk(clk);
    v1649_fifo.rst(rst);
    v1649_fifo.enq(v1649_in);
    v1649_fifo.deq(v1649_out);
    v1650_fifo.clk(clk);
    v1650_fifo.rst(rst);
    v1650_fifo.enq(v1650_in);
    v1650_fifo.deq(v1650_out);
    v1651_fifo.clk(clk);
    v1651_fifo.rst(rst);
    v1651_fifo.enq(v1651_in);
    v1651_fifo.deq(v1651_out);
    v1652_fifo.clk(clk);
    v1652_fifo.rst(rst);
    v1652_fifo.enq(v1652_in);
    v1652_fifo.deq(v1652_out);
    v1653_fifo.clk(clk);
    v1653_fifo.rst(rst);
    v1653_fifo.enq(v1653_in);
    v1653_fifo.deq(v1653_out);
    v1654_fifo.clk(clk);
    v1654_fifo.rst(rst);
    v1654_fifo.enq(v1654_in);
    v1654_fifo.deq(v1654_out);
    v1655_fifo.clk(clk);
    v1655_fifo.rst(rst);
    v1655_fifo.enq(v1655_in);
    v1655_fifo.deq(v1655_out);
    v1656_fifo.clk(clk);
    v1656_fifo.rst(rst);
    v1656_fifo.enq(v1656_in);
    v1656_fifo.deq(v1656_out);
    v1657_fifo.clk(clk);
    v1657_fifo.rst(rst);
    v1657_fifo.enq(v1657_in);
    v1657_fifo.deq(v1657_out);
    v1658_fifo.clk(clk);
    v1658_fifo.rst(rst);
    v1658_fifo.enq(v1658_in);
    v1658_fifo.deq(v1658_out);
    v1659_fifo.clk(clk);
    v1659_fifo.rst(rst);
    v1659_fifo.enq(v1659_in);
    v1659_fifo.deq(v1659_out);
    v1660_fifo.clk(clk);
    v1660_fifo.rst(rst);
    v1660_fifo.enq(v1660_in);
    v1660_fifo.deq(v1660_out);
    v1661_fifo.clk(clk);
    v1661_fifo.rst(rst);
    v1661_fifo.enq(v1661_in);
    v1661_fifo.deq(v1661_out);
    v1662_fifo.clk(clk);
    v1662_fifo.rst(rst);
    v1662_fifo.enq(v1662_in);
    v1662_fifo.deq(v1662_out);
    v1663_fifo.clk(clk);
    v1663_fifo.rst(rst);
    v1663_fifo.enq(v1663_in);
    v1663_fifo.deq(v1663_out);
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
  sc_signal< ac_int<8, false> > mp1_0_radr, mp1_0_wadr;
  sc_signal<bool> mp1_0_re, mp1_0_we, mp1_0_rrdy, mp1_0_wrdy;
  sc_signal< half > mp1_0_q, mp1_0_d;
  AlloMemPins< half, 215, 8 > mp1_0_mem;
  sc_signal< ac_int<8, false> > mp1_1_radr, mp1_1_wadr;
  sc_signal<bool> mp1_1_re, mp1_1_we, mp1_1_rrdy, mp1_1_wrdy;
  sc_signal< ac_int<32, true> > mp1_1_q, mp1_1_d;
  AlloMemPins< ac_int<32, true>, 215, 8 > mp1_1_mem;
  sc_signal< ac_int<8, false> > mp2_0_radr, mp2_0_wadr;
  sc_signal<bool> mp2_0_re, mp2_0_we, mp2_0_rrdy, mp2_0_wrdy;
  sc_signal< half > mp2_0_q, mp2_0_d;
  AlloMemPins< half, 215, 8 > mp2_0_mem;
  sc_signal< ac_int<8, false> > mp2_1_radr, mp2_1_wadr;
  sc_signal<bool> mp2_1_re, mp2_1_we, mp2_1_rrdy, mp2_1_wrdy;
  sc_signal< ac_int<32, true> > mp2_1_q, mp2_1_d;
  AlloMemPins< ac_int<32, true>, 215, 8 > mp2_1_mem;
  sc_signal< ac_int<8, false> > mp3_0_radr, mp3_0_wadr;
  sc_signal<bool> mp3_0_re, mp3_0_we, mp3_0_rrdy, mp3_0_wrdy;
  sc_signal< half > mp3_0_q, mp3_0_d;
  AlloMemPins< half, 215, 8 > mp3_0_mem;
  sc_signal< ac_int<8, false> > mp3_1_radr, mp3_1_wadr;
  sc_signal<bool> mp3_1_re, mp3_1_we, mp3_1_rrdy, mp3_1_wrdy;
  sc_signal< ac_int<32, true> > mp3_1_q, mp3_1_d;
  AlloMemPins< ac_int<32, true>, 215, 8 > mp3_1_mem;
  sc_signal< ac_int<8, false> > mp4_0_radr, mp4_0_wadr;
  sc_signal<bool> mp4_0_re, mp4_0_we, mp4_0_rrdy, mp4_0_wrdy;
  sc_signal< half > mp4_0_q, mp4_0_d;
  AlloMemPins< half, 215, 8 > mp4_0_mem;
  sc_signal< ac_int<8, false> > mp4_1_radr, mp4_1_wadr;
  sc_signal<bool> mp4_1_re, mp4_1_we, mp4_1_rrdy, mp4_1_wrdy;
  sc_signal< ac_int<32, true> > mp4_1_q, mp4_1_d;
  AlloMemPins< ac_int<32, true>, 215, 8 > mp4_1_mem;
  sc_signal< ac_int<8, false> > mp5_0_radr, mp5_0_wadr;
  sc_signal<bool> mp5_0_re, mp5_0_we, mp5_0_rrdy, mp5_0_wrdy;
  sc_signal< half > mp5_0_q, mp5_0_d;
  AlloMemPins< half, 215, 8 > mp5_0_mem;
  sc_signal< ac_int<8, false> > mp6_0_radr, mp6_0_wadr;
  sc_signal<bool> mp6_0_re, mp6_0_we, mp6_0_rrdy, mp6_0_wrdy;
  sc_signal< half > mp6_0_q, mp6_0_d;
  AlloMemPins< half, 215, 8 > mp6_0_mem;
  sc_signal< ac_int<8, false> > mp7_0_radr, mp7_0_wadr;
  sc_signal<bool> mp7_0_re, mp7_0_we, mp7_0_rrdy, mp7_0_wrdy;
  sc_signal< half > mp7_0_q, mp7_0_d;
  AlloMemPins< half, 215, 8 > mp7_0_mem;
  sc_signal< ac_int<8, false> > mp8_0_radr, mp8_0_wadr;
  sc_signal<bool> mp8_0_re, mp8_0_we, mp8_0_rrdy, mp8_0_wrdy;
  sc_signal< half > mp8_0_q, mp8_0_d;
  AlloMemPins< half, 215, 8 > mp8_0_mem;
  sc_signal< ac_int<8, false> > mp9_0_radr, mp9_0_wadr;
  sc_signal<bool> mp9_0_re, mp9_0_we, mp9_0_rrdy, mp9_0_wrdy;
  sc_signal< ac_int<32, true> > mp9_0_q, mp9_0_d;
  AlloMemPins< ac_int<32, true>, 215, 8 > mp9_0_mem;
  sc_signal< ac_int<8, false> > mp10_0_radr, mp10_0_wadr;
  sc_signal<bool> mp10_0_re, mp10_0_we, mp10_0_rrdy, mp10_0_wrdy;
  sc_signal< ac_int<32, true> > mp10_0_q, mp10_0_d;
  AlloMemPins< ac_int<32, true>, 215, 8 > mp10_0_mem;
  sc_signal< ac_int<8, false> > mp11_0_radr, mp11_0_wadr;
  sc_signal<bool> mp11_0_re, mp11_0_we, mp11_0_rrdy, mp11_0_wrdy;
  sc_signal< ac_int<32, true> > mp11_0_q, mp11_0_d;
  AlloMemPins< ac_int<32, true>, 215, 8 > mp11_0_mem;
  sc_signal< ac_int<8, false> > mp12_0_radr, mp12_0_wadr;
  sc_signal<bool> mp12_0_re, mp12_0_we, mp12_0_rrdy, mp12_0_wrdy;
  sc_signal< ac_int<32, true> > mp12_0_q, mp12_0_d;
  AlloMemPins< ac_int<32, true>, 215, 8 > mp12_0_mem;
  sc_signal< ac_int<8, false> > mp13_0_radr, mp13_0_wadr;
  sc_signal<bool> mp13_0_re, mp13_0_we, mp13_0_rrdy, mp13_0_wrdy;
  sc_signal< ac_int<32, true> > mp13_0_q, mp13_0_d;
  AlloMemPins< ac_int<32, true>, 215, 8 > mp13_0_mem;
  sc_signal< ac_int<8, false> > mp14_0_radr, mp14_0_wadr;
  sc_signal<bool> mp14_0_re, mp14_0_we, mp14_0_rrdy, mp14_0_wrdy;
  sc_signal< ac_int<32, true> > mp14_0_q, mp14_0_d;
  AlloMemPins< ac_int<32, true>, 215, 8 > mp14_0_mem;
  sc_signal< ac_int<8, false> > mp15_0_radr, mp15_0_wadr;
  sc_signal<bool> mp15_0_re, mp15_0_we, mp15_0_rrdy, mp15_0_wrdy;
  sc_signal< ac_int<32, true> > mp15_0_q, mp15_0_d;
  AlloMemPins< ac_int<32, true>, 215, 8 > mp15_0_mem;
  sc_signal< ac_int<8, false> > mp16_0_radr, mp16_0_wadr;
  sc_signal<bool> mp16_0_re, mp16_0_we, mp16_0_rrdy, mp16_0_wrdy;
  sc_signal< ac_int<32, true> > mp16_0_q, mp16_0_d;
  AlloMemPins< ac_int<32, true>, 215, 8 > mp16_0_mem;
  SC_HAS_PROCESS(tb);
  tb(sc_module_name n) : sc_module(n), clk("clk", 1, SC_NS), dut("dut"), mp1_0_mem("mp1_0_mem"), mp1_1_mem("mp1_1_mem"), mp2_0_mem("mp2_0_mem"), mp2_1_mem("mp2_1_mem"), mp3_0_mem("mp3_0_mem"), mp3_1_mem("mp3_1_mem"), mp4_0_mem("mp4_0_mem"), mp4_1_mem("mp4_1_mem"), mp5_0_mem("mp5_0_mem"), mp6_0_mem("mp6_0_mem"), mp7_0_mem("mp7_0_mem"), mp8_0_mem("mp8_0_mem"), mp9_0_mem("mp9_0_mem"), mp10_0_mem("mp10_0_mem"), mp11_0_mem("mp11_0_mem"), mp12_0_mem("mp12_0_mem"), mp13_0_mem("mp13_0_mem"), mp14_0_mem("mp14_0_mem"), mp15_0_mem("mp15_0_mem"), mp16_0_mem("mp16_0_mem") {
    dut.clk(clk); dut.rst(rst); dut.done(done_sig);
    mp1_0_mem.clk(clk); mp1_0_mem.rst(rst);
    dut.v1611_radr(mp1_0_radr); mp1_0_mem.radr(mp1_0_radr);
    dut.v1611_re(mp1_0_re); mp1_0_mem.re(mp1_0_re);
    dut.v1611_q(mp1_0_q); mp1_0_mem.q(mp1_0_q);
    dut.v1611_rrdy(mp1_0_rrdy); mp1_0_mem.rrdy(mp1_0_rrdy);
    mp1_0_mem.wadr(mp1_0_wadr);
    mp1_0_mem.d(mp1_0_d);
    mp1_0_mem.we(mp1_0_we);
    mp1_0_mem.wrdy(mp1_0_wrdy);
    mp1_1_mem.clk(clk); mp1_1_mem.rst(rst);
    dut.v1627_radr(mp1_1_radr); mp1_1_mem.radr(mp1_1_radr);
    dut.v1627_re(mp1_1_re); mp1_1_mem.re(mp1_1_re);
    dut.v1627_q(mp1_1_q); mp1_1_mem.q(mp1_1_q);
    dut.v1627_rrdy(mp1_1_rrdy); mp1_1_mem.rrdy(mp1_1_rrdy);
    mp1_1_mem.wadr(mp1_1_wadr);
    mp1_1_mem.d(mp1_1_d);
    mp1_1_mem.we(mp1_1_we);
    mp1_1_mem.wrdy(mp1_1_wrdy);
    mp2_0_mem.clk(clk); mp2_0_mem.rst(rst);
    dut.v1612_radr(mp2_0_radr); mp2_0_mem.radr(mp2_0_radr);
    dut.v1612_re(mp2_0_re); mp2_0_mem.re(mp2_0_re);
    dut.v1612_q(mp2_0_q); mp2_0_mem.q(mp2_0_q);
    dut.v1612_rrdy(mp2_0_rrdy); mp2_0_mem.rrdy(mp2_0_rrdy);
    mp2_0_mem.wadr(mp2_0_wadr);
    mp2_0_mem.d(mp2_0_d);
    mp2_0_mem.we(mp2_0_we);
    mp2_0_mem.wrdy(mp2_0_wrdy);
    mp2_1_mem.clk(clk); mp2_1_mem.rst(rst);
    dut.v1628_radr(mp2_1_radr); mp2_1_mem.radr(mp2_1_radr);
    dut.v1628_re(mp2_1_re); mp2_1_mem.re(mp2_1_re);
    dut.v1628_q(mp2_1_q); mp2_1_mem.q(mp2_1_q);
    dut.v1628_rrdy(mp2_1_rrdy); mp2_1_mem.rrdy(mp2_1_rrdy);
    mp2_1_mem.wadr(mp2_1_wadr);
    mp2_1_mem.d(mp2_1_d);
    mp2_1_mem.we(mp2_1_we);
    mp2_1_mem.wrdy(mp2_1_wrdy);
    mp3_0_mem.clk(clk); mp3_0_mem.rst(rst);
    dut.v1613_radr(mp3_0_radr); mp3_0_mem.radr(mp3_0_radr);
    dut.v1613_re(mp3_0_re); mp3_0_mem.re(mp3_0_re);
    dut.v1613_q(mp3_0_q); mp3_0_mem.q(mp3_0_q);
    dut.v1613_rrdy(mp3_0_rrdy); mp3_0_mem.rrdy(mp3_0_rrdy);
    mp3_0_mem.wadr(mp3_0_wadr);
    mp3_0_mem.d(mp3_0_d);
    mp3_0_mem.we(mp3_0_we);
    mp3_0_mem.wrdy(mp3_0_wrdy);
    mp3_1_mem.clk(clk); mp3_1_mem.rst(rst);
    dut.v1629_radr(mp3_1_radr); mp3_1_mem.radr(mp3_1_radr);
    dut.v1629_re(mp3_1_re); mp3_1_mem.re(mp3_1_re);
    dut.v1629_q(mp3_1_q); mp3_1_mem.q(mp3_1_q);
    dut.v1629_rrdy(mp3_1_rrdy); mp3_1_mem.rrdy(mp3_1_rrdy);
    mp3_1_mem.wadr(mp3_1_wadr);
    mp3_1_mem.d(mp3_1_d);
    mp3_1_mem.we(mp3_1_we);
    mp3_1_mem.wrdy(mp3_1_wrdy);
    mp4_0_mem.clk(clk); mp4_0_mem.rst(rst);
    dut.v1614_radr(mp4_0_radr); mp4_0_mem.radr(mp4_0_radr);
    dut.v1614_re(mp4_0_re); mp4_0_mem.re(mp4_0_re);
    dut.v1614_q(mp4_0_q); mp4_0_mem.q(mp4_0_q);
    dut.v1614_rrdy(mp4_0_rrdy); mp4_0_mem.rrdy(mp4_0_rrdy);
    mp4_0_mem.wadr(mp4_0_wadr);
    mp4_0_mem.d(mp4_0_d);
    mp4_0_mem.we(mp4_0_we);
    mp4_0_mem.wrdy(mp4_0_wrdy);
    mp4_1_mem.clk(clk); mp4_1_mem.rst(rst);
    dut.v1630_radr(mp4_1_radr); mp4_1_mem.radr(mp4_1_radr);
    dut.v1630_re(mp4_1_re); mp4_1_mem.re(mp4_1_re);
    dut.v1630_q(mp4_1_q); mp4_1_mem.q(mp4_1_q);
    dut.v1630_rrdy(mp4_1_rrdy); mp4_1_mem.rrdy(mp4_1_rrdy);
    mp4_1_mem.wadr(mp4_1_wadr);
    mp4_1_mem.d(mp4_1_d);
    mp4_1_mem.we(mp4_1_we);
    mp4_1_mem.wrdy(mp4_1_wrdy);
    mp5_0_mem.clk(clk); mp5_0_mem.rst(rst);
    mp5_0_mem.radr(mp5_0_radr);
    mp5_0_mem.re(mp5_0_re);
    mp5_0_mem.q(mp5_0_q);
    mp5_0_mem.rrdy(mp5_0_rrdy);
    dut.v1615_wadr(mp5_0_wadr); mp5_0_mem.wadr(mp5_0_wadr);
    dut.v1615_d(mp5_0_d); mp5_0_mem.d(mp5_0_d);
    dut.v1615_we(mp5_0_we); mp5_0_mem.we(mp5_0_we);
    dut.v1615_wrdy(mp5_0_wrdy); mp5_0_mem.wrdy(mp5_0_wrdy);
    mp6_0_mem.clk(clk); mp6_0_mem.rst(rst);
    mp6_0_mem.radr(mp6_0_radr);
    mp6_0_mem.re(mp6_0_re);
    mp6_0_mem.q(mp6_0_q);
    mp6_0_mem.rrdy(mp6_0_rrdy);
    dut.v1616_wadr(mp6_0_wadr); mp6_0_mem.wadr(mp6_0_wadr);
    dut.v1616_d(mp6_0_d); mp6_0_mem.d(mp6_0_d);
    dut.v1616_we(mp6_0_we); mp6_0_mem.we(mp6_0_we);
    dut.v1616_wrdy(mp6_0_wrdy); mp6_0_mem.wrdy(mp6_0_wrdy);
    mp7_0_mem.clk(clk); mp7_0_mem.rst(rst);
    mp7_0_mem.radr(mp7_0_radr);
    mp7_0_mem.re(mp7_0_re);
    mp7_0_mem.q(mp7_0_q);
    mp7_0_mem.rrdy(mp7_0_rrdy);
    dut.v1617_wadr(mp7_0_wadr); mp7_0_mem.wadr(mp7_0_wadr);
    dut.v1617_d(mp7_0_d); mp7_0_mem.d(mp7_0_d);
    dut.v1617_we(mp7_0_we); mp7_0_mem.we(mp7_0_we);
    dut.v1617_wrdy(mp7_0_wrdy); mp7_0_mem.wrdy(mp7_0_wrdy);
    mp8_0_mem.clk(clk); mp8_0_mem.rst(rst);
    mp8_0_mem.radr(mp8_0_radr);
    mp8_0_mem.re(mp8_0_re);
    mp8_0_mem.q(mp8_0_q);
    mp8_0_mem.rrdy(mp8_0_rrdy);
    dut.v1618_wadr(mp8_0_wadr); mp8_0_mem.wadr(mp8_0_wadr);
    dut.v1618_d(mp8_0_d); mp8_0_mem.d(mp8_0_d);
    dut.v1618_we(mp8_0_we); mp8_0_mem.we(mp8_0_we);
    dut.v1618_wrdy(mp8_0_wrdy); mp8_0_mem.wrdy(mp8_0_wrdy);
    mp9_0_mem.clk(clk); mp9_0_mem.rst(rst);
    dut.v1619_radr(mp9_0_radr); mp9_0_mem.radr(mp9_0_radr);
    dut.v1619_re(mp9_0_re); mp9_0_mem.re(mp9_0_re);
    dut.v1619_q(mp9_0_q); mp9_0_mem.q(mp9_0_q);
    dut.v1619_rrdy(mp9_0_rrdy); mp9_0_mem.rrdy(mp9_0_rrdy);
    mp9_0_mem.wadr(mp9_0_wadr);
    mp9_0_mem.d(mp9_0_d);
    mp9_0_mem.we(mp9_0_we);
    mp9_0_mem.wrdy(mp9_0_wrdy);
    mp10_0_mem.clk(clk); mp10_0_mem.rst(rst);
    dut.v1620_radr(mp10_0_radr); mp10_0_mem.radr(mp10_0_radr);
    dut.v1620_re(mp10_0_re); mp10_0_mem.re(mp10_0_re);
    dut.v1620_q(mp10_0_q); mp10_0_mem.q(mp10_0_q);
    dut.v1620_rrdy(mp10_0_rrdy); mp10_0_mem.rrdy(mp10_0_rrdy);
    mp10_0_mem.wadr(mp10_0_wadr);
    mp10_0_mem.d(mp10_0_d);
    mp10_0_mem.we(mp10_0_we);
    mp10_0_mem.wrdy(mp10_0_wrdy);
    mp11_0_mem.clk(clk); mp11_0_mem.rst(rst);
    dut.v1621_radr(mp11_0_radr); mp11_0_mem.radr(mp11_0_radr);
    dut.v1621_re(mp11_0_re); mp11_0_mem.re(mp11_0_re);
    dut.v1621_q(mp11_0_q); mp11_0_mem.q(mp11_0_q);
    dut.v1621_rrdy(mp11_0_rrdy); mp11_0_mem.rrdy(mp11_0_rrdy);
    mp11_0_mem.wadr(mp11_0_wadr);
    mp11_0_mem.d(mp11_0_d);
    mp11_0_mem.we(mp11_0_we);
    mp11_0_mem.wrdy(mp11_0_wrdy);
    mp12_0_mem.clk(clk); mp12_0_mem.rst(rst);
    dut.v1622_radr(mp12_0_radr); mp12_0_mem.radr(mp12_0_radr);
    dut.v1622_re(mp12_0_re); mp12_0_mem.re(mp12_0_re);
    dut.v1622_q(mp12_0_q); mp12_0_mem.q(mp12_0_q);
    dut.v1622_rrdy(mp12_0_rrdy); mp12_0_mem.rrdy(mp12_0_rrdy);
    mp12_0_mem.wadr(mp12_0_wadr);
    mp12_0_mem.d(mp12_0_d);
    mp12_0_mem.we(mp12_0_we);
    mp12_0_mem.wrdy(mp12_0_wrdy);
    mp13_0_mem.clk(clk); mp13_0_mem.rst(rst);
    mp13_0_mem.radr(mp13_0_radr);
    mp13_0_mem.re(mp13_0_re);
    mp13_0_mem.q(mp13_0_q);
    mp13_0_mem.rrdy(mp13_0_rrdy);
    dut.v1623_wadr(mp13_0_wadr); mp13_0_mem.wadr(mp13_0_wadr);
    dut.v1623_d(mp13_0_d); mp13_0_mem.d(mp13_0_d);
    dut.v1623_we(mp13_0_we); mp13_0_mem.we(mp13_0_we);
    dut.v1623_wrdy(mp13_0_wrdy); mp13_0_mem.wrdy(mp13_0_wrdy);
    mp14_0_mem.clk(clk); mp14_0_mem.rst(rst);
    mp14_0_mem.radr(mp14_0_radr);
    mp14_0_mem.re(mp14_0_re);
    mp14_0_mem.q(mp14_0_q);
    mp14_0_mem.rrdy(mp14_0_rrdy);
    dut.v1624_wadr(mp14_0_wadr); mp14_0_mem.wadr(mp14_0_wadr);
    dut.v1624_d(mp14_0_d); mp14_0_mem.d(mp14_0_d);
    dut.v1624_we(mp14_0_we); mp14_0_mem.we(mp14_0_we);
    dut.v1624_wrdy(mp14_0_wrdy); mp14_0_mem.wrdy(mp14_0_wrdy);
    mp15_0_mem.clk(clk); mp15_0_mem.rst(rst);
    mp15_0_mem.radr(mp15_0_radr);
    mp15_0_mem.re(mp15_0_re);
    mp15_0_mem.q(mp15_0_q);
    mp15_0_mem.rrdy(mp15_0_rrdy);
    dut.v1625_wadr(mp15_0_wadr); mp15_0_mem.wadr(mp15_0_wadr);
    dut.v1625_d(mp15_0_d); mp15_0_mem.d(mp15_0_d);
    dut.v1625_we(mp15_0_we); mp15_0_mem.we(mp15_0_we);
    dut.v1625_wrdy(mp15_0_wrdy); mp15_0_mem.wrdy(mp15_0_wrdy);
    mp16_0_mem.clk(clk); mp16_0_mem.rst(rst);
    mp16_0_mem.radr(mp16_0_radr);
    mp16_0_mem.re(mp16_0_re);
    mp16_0_mem.q(mp16_0_q);
    mp16_0_mem.rrdy(mp16_0_rrdy);
    dut.v1626_wadr(mp16_0_wadr); mp16_0_mem.wadr(mp16_0_wadr);
    dut.v1626_d(mp16_0_d); mp16_0_mem.d(mp16_0_d);
    dut.v1626_we(mp16_0_we); mp16_0_mem.we(mp16_0_we);
    dut.v1626_wrdy(mp16_0_wrdy); mp16_0_mem.wrdy(mp16_0_wrdy);
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
  { std::ifstream _f("input12.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp0_0_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input0.data"); half _v; for (int f = 0; f < 215; ++f) { _f >> _v; t.mp1_0_mem.mem[f] = (half)_v; } }
  { std::ifstream _f("input8.data"); long long _v; for (int f = 0; f < 215; ++f) { _f >> _v; t.mp1_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input12.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp1_2_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input1.data"); half _v; for (int f = 0; f < 215; ++f) { _f >> _v; t.mp2_0_mem.mem[f] = (half)_v; } }
  { std::ifstream _f("input9.data"); long long _v; for (int f = 0; f < 215; ++f) { _f >> _v; t.mp2_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input12.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp2_2_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input2.data"); half _v; for (int f = 0; f < 215; ++f) { _f >> _v; t.mp3_0_mem.mem[f] = (half)_v; } }
  { std::ifstream _f("input10.data"); long long _v; for (int f = 0; f < 215; ++f) { _f >> _v; t.mp3_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input12.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp3_2_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input3.data"); half _v; for (int f = 0; f < 215; ++f) { _f >> _v; t.mp4_0_mem.mem[f] = (half)_v; } }
  { std::ifstream _f("input11.data"); long long _v; for (int f = 0; f < 215; ++f) { _f >> _v; t.mp4_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input12.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp4_2_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input12.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp5_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input12.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp6_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input12.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp7_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input12.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp8_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input4.data"); long long _v; for (int f = 0; f < 215; ++f) { _f >> _v; t.mp9_0_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input12.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp9_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input5.data"); long long _v; for (int f = 0; f < 215; ++f) { _f >> _v; t.mp10_0_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input12.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp10_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input6.data"); long long _v; for (int f = 0; f < 215; ++f) { _f >> _v; t.mp11_0_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input12.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp11_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input7.data"); long long _v; for (int f = 0; f < 215; ++f) { _f >> _v; t.mp12_0_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input12.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp12_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input12.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp13_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input12.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp14_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input12.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp15_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  { std::ifstream _f("input12.data"); long long _v; for (int f = 0; f < 1; ++f) { _f >> _v; t.dut.mp16_1_mem.mem[f] = (ac_int<32, true>)_v; } }
  t.rst = 0; sc_start(1, SC_NS);
  t.rst = 1;
  for (long long _c = 0; _c < 630000LL && !t.done_sig.read(); ++_c) sc_start(1, SC_NS); // until DUT done
  sc_start(256, SC_NS); // settle in-flight memory writes
  { std::ofstream _f("output0.data");
    for (int f = 0; f < 215; ++f) {
      float _s = 0;
      _s += t.mp5_0_mem.mem[f].to_float();
      _f << std::setprecision(9) << _s << "\n";
    } }
  { std::ofstream _f("output1.data");
    for (int f = 0; f < 215; ++f) {
      float _s = 0;
      _s += t.mp6_0_mem.mem[f].to_float();
      _f << std::setprecision(9) << _s << "\n";
    } }
  { std::ofstream _f("output2.data");
    for (int f = 0; f < 215; ++f) {
      float _s = 0;
      _s += t.mp7_0_mem.mem[f].to_float();
      _f << std::setprecision(9) << _s << "\n";
    } }
  { std::ofstream _f("output3.data");
    for (int f = 0; f < 215; ++f) {
      float _s = 0;
      _s += t.mp8_0_mem.mem[f].to_float();
      _f << std::setprecision(9) << _s << "\n";
    } }
  { std::ofstream _f("output4.data");
    for (int f = 0; f < 215; ++f) {
      long long _s = 0;
      _s += (long long) t.mp13_0_mem.mem[f];
      _f << _s << "\n";
    } }
  { std::ofstream _f("output5.data");
    for (int f = 0; f < 215; ++f) {
      long long _s = 0;
      _s += (long long) t.mp14_0_mem.mem[f];
      _f << _s << "\n";
    } }
  { std::ofstream _f("output6.data");
    for (int f = 0; f < 215; ++f) {
      long long _s = 0;
      _s += (long long) t.mp15_0_mem.mem[f];
      _f << _s << "\n";
    } }
  { std::ofstream _f("output7.data");
    for (int f = 0; f < 215; ++f) {
      long long _s = 0;
      _s += (long long) t.mp16_0_mem.mem[f];
      _f << _s << "\n";
    } }
  return 0;
}
