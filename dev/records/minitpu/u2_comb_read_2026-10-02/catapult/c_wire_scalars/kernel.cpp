
//===------------------------------------------------------------*- C++ -*-===//
// Automatically generated file for SystemC (Catapult HLS / MatchLib Connections).
//===----------------------------------------------------------------------===//
#include <systemc.h>
// ac::bfloat16's operators round toward zero by default; MLIR's arith.addf/mulf
// on bf16 round to nearest-even. Must precede ac_std_float.h.
#ifndef AC_STD_FLOAT_BFLOAT16_ROUND_OVERRIDE
#define AC_STD_FLOAT_BFLOAT16_ROUND_OVERRIDE AC_RND_CONV
#endif
// The ac types come BEFORE mc_connections.h: Connections' marshaller.h defines
// its Wrapped<> specializations (ac::bfloat16's AC_SPECIAL_FLOAT_WRAPPER, the
// ac_std_float one) only for the ac headers already included. With
// ac_std_float.h after it, Catapult's `go analyze` failed every bf16 port with
// CRD-135 "class ac::bfloat16 has no member Marshall"; g++ csim compiles the
// non-synthesis Connections path and never saw it.
#include <ac_int.h>
#include <ac_fixed.h>
#include <ac_std_float.h>   // IEEE floats: ac_ieee_float<binaryNN>; ac::bfloat16
#include <mc_connections.h>   // MatchLib Connections (LI valid/ready channels)
#include <connections/connections_fifo.h>  // vendor FWFT Connections::Fifo (buffered streams)
#include <ac_channel.h>     // local self-FIFO streams (one-kernel put+get+status)
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
// data_ac_int() is SIGNED: a negative 16-bit float's bits would sign-extend
// (0x8000 -> 0xffff8000), so go through the unsigned type of the float's width.
inline unsigned long long _fbits(const half &v) {
  return (unsigned long long)(unsigned short)v.data_ac_int().to_int();
}
inline unsigned long long _fbits(const ac::bfloat16 &v) {
  return (unsigned long long)(unsigned short)v.data_ac_int().to_int();
}
inline unsigned long long _fbits(const ac_ieee_float<binary32> &v) {
  return (unsigned long long)(unsigned)v.data_ac_int().to_int();
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
// raw bits -> float: the inverse of _fbits, for the testbench's data files.
// The files carry a float's IEEE bit pattern as an unsigned integer, never
// decimal text: `>> float` fails on "nan"/"inf" (failbit, after which every
// later read silently yields 0), loses a NaN's sign and payload, and
// ac::bfloat16(float) truncates toward zero, so the shortest decimal of a bf16
// value (e.g. "1.00781" for 1.0078125) parsed and converted to the bf16 BELOW.
template <class T> inline T _ffrombits(unsigned long long b) {
#ifdef __SYNTHESIS__
  return (T)b;
#else
  T v; std::memcpy(&v, &b, sizeof(T)); return v; // double (csim/tb only)
#endif
}
template <> inline half _ffrombits<half>(unsigned long long b) {
  half v; v.set_data(ac_int<16, true>((long long)b)); return v;
}
template <> inline ac::bfloat16 _ffrombits<ac::bfloat16>(unsigned long long b) {
  ac::bfloat16 v; v.set_data(ac_int<16, true>((long long)b)); return v;
}
template <>
inline ac_ieee_float<binary32> _ffrombits<ac_ieee_float<binary32> >(unsigned long long b) {
  ac_ieee_float<binary32> v; v.set_data(ac_int<32, true>((long long)b)); return v;
}
// Float conversions (casts). ac floats have explicit constructors only, and
// ac::bfloat16's hard-code round-toward-zero; arith casts round to nearest
// even. So convert through ac_std_float, whose conversions are AC_RND_CONV:
// _fstd(x) is the source as an ac_std_float, _fto/_ito round it (once) into D.
template <class D> struct _fstd_of;
template <> struct _fstd_of<ac::bfloat16> { typedef ac_std_float<16, 8> t; };
template <> struct _fstd_of<half> { typedef ac_std_float<16, 5> t; };
template <> struct _fstd_of<ac_ieee_float<binary32> > { typedef ac_std_float<32, 8> t; };
inline ac_std_float<16, 8> _fstd(const ac::bfloat16 &x) { return x.to_ac_std_float(); }
template <ac_ieee_float_format F>
inline typename ac_ieee_float<F>::ac_std_float_t _fstd(const ac_ieee_float<F> &x) {
  return x.to_ac_std_float();
}
inline ac_std_float<64, 11> _fstd(double x) { return ac_std_float<64, 11>(x); }
template <> struct _fstd_of<double> { typedef ac_std_float<64, 11> t; };
template <class D> struct _fconv {
  template <int W, int E> static D from(const ac_std_float<W, E> &s) {
    return D(typename _fstd_of<D>::t(s));
  }
  template <int WI, bool SI> static D fromi(const ac_int<WI, SI> &x) {
    return D(typename _fstd_of<D>::t(x));
  }
};
template <> struct _fconv<double> {
  template <int W, int E> static double from(const ac_std_float<W, E> &s) {
    return ac_std_float<64, 11>(s).to_double();
  }
  template <int WI, bool SI> static double fromi(const ac_int<WI, SI> &x) {
    return ac_std_float<64, 11>(x).to_double();
  }
};
template <class D, int W, int E> inline D _fto(const ac_std_float<W, E> &s) {
  return _fconv<D>::from(s);
}
template <class D, int WI, bool SI> inline D _ito(const ac_int<WI, SI> &x) {
  return _fconv<D>::fromi(x);
}
// Waveform trace of a float: Connections/sc_signal ports call sc_trace on their
// payload, unqualified, from inside sc_core and Connections. ac_sc.h (pulled in
// by mc_connections.h) supplies those overloads for every ac float, ac::bfloat16
// included, in namespace ac_tracing made visible to sc_core -- but only when
// ac_std_float.h was included first, which the include order above guarantees.
// The emitter used to write its own; with the library's in scope they are
// ambiguous, and without them a global bf16 overload was never found by ADL.
// csim: a float signal must compare BITS, not values. sc_signal::write() and
// update() skip a write whose value `==` the current one, and IEEE says
// +0 == -0: a -0 written after a +0 (or the reverse) never reaches the reader,
// on every Connections channel (sc_signal<Message>) and memory pin alike. A
// wire in RTL carries the sign bit, so csim disagreed with the RTL. Specialize
// both members for each float payload and writer policy to compare _fbits.
// Only for the OSCI kernel these members are written for (2.3.2+, sc_signal_t);
// never under synthesis, and not under Xcelium's own SystemC (NCSC).
#if !defined(__SYNTHESIS__) && !defined(NCSC) && defined(SC_VERSION_MAJOR) &&     \
    (SC_VERSION_MAJOR * 100 + SC_VERSION_MINOR) * 100 + SC_VERSION_PATCH >= 20302
namespace sc_core {
#define ALLO_SIGNAL_BITS_EQ(T, POL)                                              \
  template <> inline void sc_signal_t<T, POL>::write(const T &value_) {          \
    bool value_changed = _fbits(m_new_val) != _fbits(value_);                    \
    if (!policy_type::check_write(this, value_changed))                         \
      return;                                                                    \
    m_new_val = value_;                                                          \
    if (value_changed || policy_type::needs_update())                            \
      request_update();                                                          \
  }                                                                              \
  template <> inline void sc_signal_t<T, POL>::update() {                        \
    policy_type::update();                                                       \
    if (_fbits(m_new_val) != _fbits(m_cur_val))                                  \
      do_update();                                                               \
  }
#define ALLO_SIGNAL_BITS_EQ_ALL(T)                                               \
  ALLO_SIGNAL_BITS_EQ(T, SC_ONE_WRITER)                                          \
  ALLO_SIGNAL_BITS_EQ(T, SC_MANY_WRITERS)                                        \
  ALLO_SIGNAL_BITS_EQ(T, SC_UNCHECKED_WRITERS)
ALLO_SIGNAL_BITS_EQ_ALL(ac::bfloat16)
ALLO_SIGNAL_BITS_EQ_ALL(half)
ALLO_SIGNAL_BITS_EQ_ALL(ac_ieee_float<binary32>)
ALLO_SIGNAL_BITS_EQ_ALL(double)
#undef ALLO_SIGNAL_BITS_EQ_ALL
#undef ALLO_SIGNAL_BITS_EQ
} // namespace sc_core
#endif
// Make ac_ieee_float<Format> a valid Connections channel/Combinational payload.
// (bf16 needs nothing HERE: marshaller.h has AC_SPECIAL_FLOAT_WRAPPER(ac::bfloat16,
// 16) -- but only if ac_std_float.h was included before mc_connections.h, which
// is why that include sits above it.)
// (Since ac_std_float.h precedes mc_connections.h, marshaller.h also defines
// Wrapped<ac_ieee_float<binaryNN>> for the IEEE widths, and those explicit
// specializations win; this partial one is kept for the SCVerify TU below.)
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

SC_MODULE(src_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_out<bool> done;
  Connections::In< ac_int<5, false> > v0;
  Connections::In< ac_int<5, false> > v1;
  Connections::In< ac_int<5, false> > v2;
  Connections::In< ac_int<5, false> > v3;
  Connections::In< ac_int<16, false> > v4;
  Connections::In< ac_int<1, false> > v5;
  sc_out< ac_int<5, false> > v6;
  sc_out< ac_int<5, false> > v7;
  sc_out< ac_int<5, false> > v8;
  sc_out< ac_int<5, false> > v9;
  sc_out< ac_int<16, false> > v10;
  sc_out< ac_int<1, false> > v11;
  SC_HAS_PROCESS(src_0);
  src_0(sc_module_name n) : sc_module(n), done("done"), v0("v0"), v1("v1"), v2("v2"), v3("v3"), v4("v4"), v5("v5") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  void run() {
    v0.Reset();
    v1.Reset();
    v2.Reset();
    v3.Reset();
    v4.Reset();
    v5.Reset();
    v6.write(0);
    v7.write(0);
    v8.write(0);
    v9.write(0);
    v10.write(0);
    v11.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    l_S_t_0_t: for (int t = 0; t < 64; t++) {	// L3
      ac_int<5, false> v12 = v0.Pop();	// L4
      v6.write(v12);	// L5
      ac_int<5, false> v13 = v1.Pop();	// L6
      v7.write(v13);	// L7
      ac_int<5, false> v14 = v2.Pop();	// L8
      v8.write(v14);	// L9
      ac_int<5, false> v15 = v3.Pop();	// L10
      v9.write(v15);	// L11
      uint16_t v16 = v4.Pop();	// L12
      v10.write(v16);	// L13
      bool v17 = v5.Pop();	// L14
      v11.write(v17);	// L15
    }
    done.write(true);  // RTL-observable completion (see the done port)
#ifndef __SYNTHESIS__
    __allo_done++; // csim: this kernel finished its single pass
#endif
    while (1) { wait(); }
  }
};

SC_MODULE(rf_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_out<bool> done;
  sc_in< ac_int<5, false> > v18;
  sc_in< ac_int<5, false> > v19;
  sc_in< ac_int<5, false> > v20;
  sc_in< ac_int<5, false> > v21;
  sc_in< ac_int<16, false> > v22;
  sc_in< ac_int<1, false> > v23;
  sc_out< ac_int<16, false> > v24;
  sc_out< ac_int<16, false> > v25;
  sc_out< ac_int<16, false> > v26;
  int16_t __stateful_rf_0_r0_1;  // @ Stateful
  int16_t __stateful_rf_0_r1_2;  // @ Stateful
  int16_t __stateful_rf_0_r2_3;  // @ Stateful
  int16_t __stateful_rf_0_r3_4;  // @ Stateful
  int16_t __stateful_rf_0_r4_5;  // @ Stateful
  int16_t __stateful_rf_0_r5_6;  // @ Stateful
  int16_t __stateful_rf_0_r6_7;  // @ Stateful
  int16_t __stateful_rf_0_r7_8;  // @ Stateful
  int16_t __stateful_rf_0_r8_9;  // @ Stateful
  int16_t __stateful_rf_0_r9_10;  // @ Stateful
  int16_t __stateful_rf_0_r10_11;  // @ Stateful
  int16_t __stateful_rf_0_r11_12;  // @ Stateful
  int16_t __stateful_rf_0_r12_13;  // @ Stateful
  int16_t __stateful_rf_0_r13_14;  // @ Stateful
  int16_t __stateful_rf_0_r14_15;  // @ Stateful
  int16_t __stateful_rf_0_r15_16;  // @ Stateful
  int16_t __stateful_rf_0_r16_17;  // @ Stateful
  int16_t __stateful_rf_0_r17_18;  // @ Stateful
  int16_t __stateful_rf_0_r18_19;  // @ Stateful
  int16_t __stateful_rf_0_r19_20;  // @ Stateful
  int16_t __stateful_rf_0_r20_21;  // @ Stateful
  int16_t __stateful_rf_0_r21_22;  // @ Stateful
  int16_t __stateful_rf_0_r22_23;  // @ Stateful
  int16_t __stateful_rf_0_r23_24;  // @ Stateful
  int16_t __stateful_rf_0_r24_25;  // @ Stateful
  int16_t __stateful_rf_0_r25_26;  // @ Stateful
  int16_t __stateful_rf_0_r26_27;  // @ Stateful
  int16_t __stateful_rf_0_r27_28;  // @ Stateful
  int16_t __stateful_rf_0_r28_29;  // @ Stateful
  int16_t __stateful_rf_0_r29_30;  // @ Stateful
  int16_t __stateful_rf_0_r30_31;  // @ Stateful
  int16_t __stateful_rf_0_r31_32;  // @ Stateful
  SC_HAS_PROCESS(rf_0);
  rf_0(sc_module_name n) : sc_module(n), done("done") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  void run() {
    __stateful_rf_0_r0_1 = 0;
    __stateful_rf_0_r1_2 = 0;
    __stateful_rf_0_r2_3 = 0;
    __stateful_rf_0_r3_4 = 0;
    __stateful_rf_0_r4_5 = 0;
    __stateful_rf_0_r5_6 = 0;
    __stateful_rf_0_r6_7 = 0;
    __stateful_rf_0_r7_8 = 0;
    __stateful_rf_0_r8_9 = 0;
    __stateful_rf_0_r9_10 = 0;
    __stateful_rf_0_r10_11 = 0;
    __stateful_rf_0_r11_12 = 0;
    __stateful_rf_0_r12_13 = 0;
    __stateful_rf_0_r13_14 = 0;
    __stateful_rf_0_r14_15 = 0;
    __stateful_rf_0_r15_16 = 0;
    __stateful_rf_0_r16_17 = 0;
    __stateful_rf_0_r17_18 = 0;
    __stateful_rf_0_r18_19 = 0;
    __stateful_rf_0_r19_20 = 0;
    __stateful_rf_0_r20_21 = 0;
    __stateful_rf_0_r21_22 = 0;
    __stateful_rf_0_r22_23 = 0;
    __stateful_rf_0_r23_24 = 0;
    __stateful_rf_0_r24_25 = 0;
    __stateful_rf_0_r25_26 = 0;
    __stateful_rf_0_r26_27 = 0;
    __stateful_rf_0_r27_28 = 0;
    __stateful_rf_0_r28_29 = 0;
    __stateful_rf_0_r29_30 = 0;
    __stateful_rf_0_r30_31 = 0;
    __stateful_rf_0_r31_32 = 0;
    v24.write(0);
    v25.write(0);
    v26.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    // placeholder for const int16_t __stateful_rf_0_r0_1	// L85
    // placeholder for const int16_t __stateful_rf_0_r1_2	// L86
    // placeholder for const int16_t __stateful_rf_0_r2_3	// L87
    // placeholder for const int16_t __stateful_rf_0_r3_4	// L88
    // placeholder for const int16_t __stateful_rf_0_r4_5	// L89
    // placeholder for const int16_t __stateful_rf_0_r5_6	// L90
    // placeholder for const int16_t __stateful_rf_0_r6_7	// L91
    // placeholder for const int16_t __stateful_rf_0_r7_8	// L92
    // placeholder for const int16_t __stateful_rf_0_r8_9	// L93
    // placeholder for const int16_t __stateful_rf_0_r9_10	// L94
    // placeholder for const int16_t __stateful_rf_0_r10_11	// L95
    // placeholder for const int16_t __stateful_rf_0_r11_12	// L96
    // placeholder for const int16_t __stateful_rf_0_r12_13	// L97
    // placeholder for const int16_t __stateful_rf_0_r13_14	// L98
    // placeholder for const int16_t __stateful_rf_0_r14_15	// L99
    // placeholder for const int16_t __stateful_rf_0_r15_16	// L100
    // placeholder for const int16_t __stateful_rf_0_r16_17	// L101
    // placeholder for const int16_t __stateful_rf_0_r17_18	// L102
    // placeholder for const int16_t __stateful_rf_0_r18_19	// L103
    // placeholder for const int16_t __stateful_rf_0_r19_20	// L104
    // placeholder for const int16_t __stateful_rf_0_r20_21	// L105
    // placeholder for const int16_t __stateful_rf_0_r21_22	// L106
    // placeholder for const int16_t __stateful_rf_0_r22_23	// L107
    // placeholder for const int16_t __stateful_rf_0_r23_24	// L108
    // placeholder for const int16_t __stateful_rf_0_r24_25	// L109
    // placeholder for const int16_t __stateful_rf_0_r25_26	// L110
    // placeholder for const int16_t __stateful_rf_0_r26_27	// L111
    // placeholder for const int16_t __stateful_rf_0_r27_28	// L112
    // placeholder for const int16_t __stateful_rf_0_r28_29	// L113
    // placeholder for const int16_t __stateful_rf_0_r29_30	// L114
    // placeholder for const int16_t __stateful_rf_0_r30_31	// L115
    // placeholder for const int16_t __stateful_rf_0_r31_32	// L116
#ifdef __SYNTHESIS__
    done.write(true);  // steady-state: no completion, so assert on entry (the post-body write is unreachable here)
    #pragma hls_pipeline_init_interval 1
    while (1) {  // steady-state loop (was `for t`): 1 iteration = 1 step
#else
    #pragma hls_pipeline_init_interval 1
    l_steady: for (int t1 = 0; t1 < 64; t1 += 1) {
#endif
      ac_int<5, false> v27 = v18.read();	// L118
      ac_int<5, false> a;	// L119
      a = v27;	// L120
      ac_int<5, false> v28 = v19.read();	// L121
      ac_int<5, false> b;	// L122
      b = v28;	// L123
      ac_int<5, false> v29 = v20.read();	// L124
      ac_int<5, false> c;	// L125
      c = v29;	// L126
      ac_int<5, false> v30 = v21.read();	// L127
      ac_int<5, false> x;	// L128
      x = v30;	// L129
      uint16_t v31 = v22.read();	// L130
      uint16_t d;	// L131
      d = v31;	// L132
      bool v32 = v23.read();	// L133
      bool e;	// L134
      e = v32;	// L135
      uint16_t qa;	// L136
      qa = 0;	// L137
      uint16_t qb;	// L138
      qb = 0;	// L139
      uint16_t qc;	// L140
      qc = 0;	// L141
      ac_int<5, false> v33 = a;	// L142
      int32_t v34 = v33;	// L143
      bool v35 = v34 == 0;	// L144
      if (v35) {	// L145
        int16_t v36 = __stateful_rf_0_r0_1;	// L146
        qa = v36;	// L147
      }
      ac_int<5, false> v37 = a;	// L149
      int32_t v38 = v37;	// L150
      bool v39 = v38 == 1;	// L151
      if (v39) {	// L152
        int16_t v40 = __stateful_rf_0_r1_2;	// L153
        qa = v40;	// L154
      }
      ac_int<5, false> v41 = a;	// L156
      int32_t v42 = v41;	// L157
      bool v43 = v42 == 2;	// L158
      if (v43) {	// L159
        int16_t v44 = __stateful_rf_0_r2_3;	// L160
        qa = v44;	// L161
      }
      ac_int<5, false> v45 = a;	// L163
      int32_t v46 = v45;	// L164
      bool v47 = v46 == 3;	// L165
      if (v47) {	// L166
        int16_t v48 = __stateful_rf_0_r3_4;	// L167
        qa = v48;	// L168
      }
      ac_int<5, false> v49 = a;	// L170
      int32_t v50 = v49;	// L171
      bool v51 = v50 == 4;	// L172
      if (v51) {	// L173
        int16_t v52 = __stateful_rf_0_r4_5;	// L174
        qa = v52;	// L175
      }
      ac_int<5, false> v53 = a;	// L177
      int32_t v54 = v53;	// L178
      bool v55 = v54 == 5;	// L179
      if (v55) {	// L180
        int16_t v56 = __stateful_rf_0_r5_6;	// L181
        qa = v56;	// L182
      }
      ac_int<5, false> v57 = a;	// L184
      int32_t v58 = v57;	// L185
      bool v59 = v58 == 6;	// L186
      if (v59) {	// L187
        int16_t v60 = __stateful_rf_0_r6_7;	// L188
        qa = v60;	// L189
      }
      ac_int<5, false> v61 = a;	// L191
      int32_t v62 = v61;	// L192
      bool v63 = v62 == 7;	// L193
      if (v63) {	// L194
        int16_t v64 = __stateful_rf_0_r7_8;	// L195
        qa = v64;	// L196
      }
      ac_int<5, false> v65 = a;	// L198
      int32_t v66 = v65;	// L199
      bool v67 = v66 == 8;	// L200
      if (v67) {	// L201
        int16_t v68 = __stateful_rf_0_r8_9;	// L202
        qa = v68;	// L203
      }
      ac_int<5, false> v69 = a;	// L205
      int32_t v70 = v69;	// L206
      bool v71 = v70 == 9;	// L207
      if (v71) {	// L208
        int16_t v72 = __stateful_rf_0_r9_10;	// L209
        qa = v72;	// L210
      }
      ac_int<5, false> v73 = a;	// L212
      int32_t v74 = v73;	// L213
      bool v75 = v74 == 10;	// L214
      if (v75) {	// L215
        int16_t v76 = __stateful_rf_0_r10_11;	// L216
        qa = v76;	// L217
      }
      ac_int<5, false> v77 = a;	// L219
      int32_t v78 = v77;	// L220
      bool v79 = v78 == 11;	// L221
      if (v79) {	// L222
        int16_t v80 = __stateful_rf_0_r11_12;	// L223
        qa = v80;	// L224
      }
      ac_int<5, false> v81 = a;	// L226
      int32_t v82 = v81;	// L227
      bool v83 = v82 == 12;	// L228
      if (v83) {	// L229
        int16_t v84 = __stateful_rf_0_r12_13;	// L230
        qa = v84;	// L231
      }
      ac_int<5, false> v85 = a;	// L233
      int32_t v86 = v85;	// L234
      bool v87 = v86 == 13;	// L235
      if (v87) {	// L236
        int16_t v88 = __stateful_rf_0_r13_14;	// L237
        qa = v88;	// L238
      }
      ac_int<5, false> v89 = a;	// L240
      int32_t v90 = v89;	// L241
      bool v91 = v90 == 14;	// L242
      if (v91) {	// L243
        int16_t v92 = __stateful_rf_0_r14_15;	// L244
        qa = v92;	// L245
      }
      ac_int<5, false> v93 = a;	// L247
      int32_t v94 = v93;	// L248
      bool v95 = v94 == 15;	// L249
      if (v95) {	// L250
        int16_t v96 = __stateful_rf_0_r15_16;	// L251
        qa = v96;	// L252
      }
      ac_int<5, false> v97 = a;	// L254
      int32_t v98 = v97;	// L255
      bool v99 = v98 == 16;	// L256
      if (v99) {	// L257
        int16_t v100 = __stateful_rf_0_r16_17;	// L258
        qa = v100;	// L259
      }
      ac_int<5, false> v101 = a;	// L261
      int32_t v102 = v101;	// L262
      bool v103 = v102 == 17;	// L263
      if (v103) {	// L264
        int16_t v104 = __stateful_rf_0_r17_18;	// L265
        qa = v104;	// L266
      }
      ac_int<5, false> v105 = a;	// L268
      int32_t v106 = v105;	// L269
      bool v107 = v106 == 18;	// L270
      if (v107) {	// L271
        int16_t v108 = __stateful_rf_0_r18_19;	// L272
        qa = v108;	// L273
      }
      ac_int<5, false> v109 = a;	// L275
      int32_t v110 = v109;	// L276
      bool v111 = v110 == 19;	// L277
      if (v111) {	// L278
        int16_t v112 = __stateful_rf_0_r19_20;	// L279
        qa = v112;	// L280
      }
      ac_int<5, false> v113 = a;	// L282
      int32_t v114 = v113;	// L283
      bool v115 = v114 == 20;	// L284
      if (v115) {	// L285
        int16_t v116 = __stateful_rf_0_r20_21;	// L286
        qa = v116;	// L287
      }
      ac_int<5, false> v117 = a;	// L289
      int32_t v118 = v117;	// L290
      bool v119 = v118 == 21;	// L291
      if (v119) {	// L292
        int16_t v120 = __stateful_rf_0_r21_22;	// L293
        qa = v120;	// L294
      }
      ac_int<5, false> v121 = a;	// L296
      int32_t v122 = v121;	// L297
      bool v123 = v122 == 22;	// L298
      if (v123) {	// L299
        int16_t v124 = __stateful_rf_0_r22_23;	// L300
        qa = v124;	// L301
      }
      ac_int<5, false> v125 = a;	// L303
      int32_t v126 = v125;	// L304
      bool v127 = v126 == 23;	// L305
      if (v127) {	// L306
        int16_t v128 = __stateful_rf_0_r23_24;	// L307
        qa = v128;	// L308
      }
      ac_int<5, false> v129 = a;	// L310
      int32_t v130 = v129;	// L311
      bool v131 = v130 == 24;	// L312
      if (v131) {	// L313
        int16_t v132 = __stateful_rf_0_r24_25;	// L314
        qa = v132;	// L315
      }
      ac_int<5, false> v133 = a;	// L317
      int32_t v134 = v133;	// L318
      bool v135 = v134 == 25;	// L319
      if (v135) {	// L320
        int16_t v136 = __stateful_rf_0_r25_26;	// L321
        qa = v136;	// L322
      }
      ac_int<5, false> v137 = a;	// L324
      int32_t v138 = v137;	// L325
      bool v139 = v138 == 26;	// L326
      if (v139) {	// L327
        int16_t v140 = __stateful_rf_0_r26_27;	// L328
        qa = v140;	// L329
      }
      ac_int<5, false> v141 = a;	// L331
      int32_t v142 = v141;	// L332
      bool v143 = v142 == 27;	// L333
      if (v143) {	// L334
        int16_t v144 = __stateful_rf_0_r27_28;	// L335
        qa = v144;	// L336
      }
      ac_int<5, false> v145 = a;	// L338
      int32_t v146 = v145;	// L339
      bool v147 = v146 == 28;	// L340
      if (v147) {	// L341
        int16_t v148 = __stateful_rf_0_r28_29;	// L342
        qa = v148;	// L343
      }
      ac_int<5, false> v149 = a;	// L345
      int32_t v150 = v149;	// L346
      bool v151 = v150 == 29;	// L347
      if (v151) {	// L348
        int16_t v152 = __stateful_rf_0_r29_30;	// L349
        qa = v152;	// L350
      }
      ac_int<5, false> v153 = a;	// L352
      int32_t v154 = v153;	// L353
      bool v155 = v154 == 30;	// L354
      if (v155) {	// L355
        int16_t v156 = __stateful_rf_0_r30_31;	// L356
        qa = v156;	// L357
      }
      ac_int<5, false> v157 = a;	// L359
      int32_t v158 = v157;	// L360
      bool v159 = v158 == 31;	// L361
      if (v159) {	// L362
        int16_t v160 = __stateful_rf_0_r31_32;	// L363
        qa = v160;	// L364
      }
      ac_int<5, false> v161 = b;	// L366
      int32_t v162 = v161;	// L367
      bool v163 = v162 == 0;	// L368
      if (v163) {	// L369
        int16_t v164 = __stateful_rf_0_r0_1;	// L370
        qb = v164;	// L371
      }
      ac_int<5, false> v165 = b;	// L373
      int32_t v166 = v165;	// L374
      bool v167 = v166 == 1;	// L375
      if (v167) {	// L376
        int16_t v168 = __stateful_rf_0_r1_2;	// L377
        qb = v168;	// L378
      }
      ac_int<5, false> v169 = b;	// L380
      int32_t v170 = v169;	// L381
      bool v171 = v170 == 2;	// L382
      if (v171) {	// L383
        int16_t v172 = __stateful_rf_0_r2_3;	// L384
        qb = v172;	// L385
      }
      ac_int<5, false> v173 = b;	// L387
      int32_t v174 = v173;	// L388
      bool v175 = v174 == 3;	// L389
      if (v175) {	// L390
        int16_t v176 = __stateful_rf_0_r3_4;	// L391
        qb = v176;	// L392
      }
      ac_int<5, false> v177 = b;	// L394
      int32_t v178 = v177;	// L395
      bool v179 = v178 == 4;	// L396
      if (v179) {	// L397
        int16_t v180 = __stateful_rf_0_r4_5;	// L398
        qb = v180;	// L399
      }
      ac_int<5, false> v181 = b;	// L401
      int32_t v182 = v181;	// L402
      bool v183 = v182 == 5;	// L403
      if (v183) {	// L404
        int16_t v184 = __stateful_rf_0_r5_6;	// L405
        qb = v184;	// L406
      }
      ac_int<5, false> v185 = b;	// L408
      int32_t v186 = v185;	// L409
      bool v187 = v186 == 6;	// L410
      if (v187) {	// L411
        int16_t v188 = __stateful_rf_0_r6_7;	// L412
        qb = v188;	// L413
      }
      ac_int<5, false> v189 = b;	// L415
      int32_t v190 = v189;	// L416
      bool v191 = v190 == 7;	// L417
      if (v191) {	// L418
        int16_t v192 = __stateful_rf_0_r7_8;	// L419
        qb = v192;	// L420
      }
      ac_int<5, false> v193 = b;	// L422
      int32_t v194 = v193;	// L423
      bool v195 = v194 == 8;	// L424
      if (v195) {	// L425
        int16_t v196 = __stateful_rf_0_r8_9;	// L426
        qb = v196;	// L427
      }
      ac_int<5, false> v197 = b;	// L429
      int32_t v198 = v197;	// L430
      bool v199 = v198 == 9;	// L431
      if (v199) {	// L432
        int16_t v200 = __stateful_rf_0_r9_10;	// L433
        qb = v200;	// L434
      }
      ac_int<5, false> v201 = b;	// L436
      int32_t v202 = v201;	// L437
      bool v203 = v202 == 10;	// L438
      if (v203) {	// L439
        int16_t v204 = __stateful_rf_0_r10_11;	// L440
        qb = v204;	// L441
      }
      ac_int<5, false> v205 = b;	// L443
      int32_t v206 = v205;	// L444
      bool v207 = v206 == 11;	// L445
      if (v207) {	// L446
        int16_t v208 = __stateful_rf_0_r11_12;	// L447
        qb = v208;	// L448
      }
      ac_int<5, false> v209 = b;	// L450
      int32_t v210 = v209;	// L451
      bool v211 = v210 == 12;	// L452
      if (v211) {	// L453
        int16_t v212 = __stateful_rf_0_r12_13;	// L454
        qb = v212;	// L455
      }
      ac_int<5, false> v213 = b;	// L457
      int32_t v214 = v213;	// L458
      bool v215 = v214 == 13;	// L459
      if (v215) {	// L460
        int16_t v216 = __stateful_rf_0_r13_14;	// L461
        qb = v216;	// L462
      }
      ac_int<5, false> v217 = b;	// L464
      int32_t v218 = v217;	// L465
      bool v219 = v218 == 14;	// L466
      if (v219) {	// L467
        int16_t v220 = __stateful_rf_0_r14_15;	// L468
        qb = v220;	// L469
      }
      ac_int<5, false> v221 = b;	// L471
      int32_t v222 = v221;	// L472
      bool v223 = v222 == 15;	// L473
      if (v223) {	// L474
        int16_t v224 = __stateful_rf_0_r15_16;	// L475
        qb = v224;	// L476
      }
      ac_int<5, false> v225 = b;	// L478
      int32_t v226 = v225;	// L479
      bool v227 = v226 == 16;	// L480
      if (v227) {	// L481
        int16_t v228 = __stateful_rf_0_r16_17;	// L482
        qb = v228;	// L483
      }
      ac_int<5, false> v229 = b;	// L485
      int32_t v230 = v229;	// L486
      bool v231 = v230 == 17;	// L487
      if (v231) {	// L488
        int16_t v232 = __stateful_rf_0_r17_18;	// L489
        qb = v232;	// L490
      }
      ac_int<5, false> v233 = b;	// L492
      int32_t v234 = v233;	// L493
      bool v235 = v234 == 18;	// L494
      if (v235) {	// L495
        int16_t v236 = __stateful_rf_0_r18_19;	// L496
        qb = v236;	// L497
      }
      ac_int<5, false> v237 = b;	// L499
      int32_t v238 = v237;	// L500
      bool v239 = v238 == 19;	// L501
      if (v239) {	// L502
        int16_t v240 = __stateful_rf_0_r19_20;	// L503
        qb = v240;	// L504
      }
      ac_int<5, false> v241 = b;	// L506
      int32_t v242 = v241;	// L507
      bool v243 = v242 == 20;	// L508
      if (v243) {	// L509
        int16_t v244 = __stateful_rf_0_r20_21;	// L510
        qb = v244;	// L511
      }
      ac_int<5, false> v245 = b;	// L513
      int32_t v246 = v245;	// L514
      bool v247 = v246 == 21;	// L515
      if (v247) {	// L516
        int16_t v248 = __stateful_rf_0_r21_22;	// L517
        qb = v248;	// L518
      }
      ac_int<5, false> v249 = b;	// L520
      int32_t v250 = v249;	// L521
      bool v251 = v250 == 22;	// L522
      if (v251) {	// L523
        int16_t v252 = __stateful_rf_0_r22_23;	// L524
        qb = v252;	// L525
      }
      ac_int<5, false> v253 = b;	// L527
      int32_t v254 = v253;	// L528
      bool v255 = v254 == 23;	// L529
      if (v255) {	// L530
        int16_t v256 = __stateful_rf_0_r23_24;	// L531
        qb = v256;	// L532
      }
      ac_int<5, false> v257 = b;	// L534
      int32_t v258 = v257;	// L535
      bool v259 = v258 == 24;	// L536
      if (v259) {	// L537
        int16_t v260 = __stateful_rf_0_r24_25;	// L538
        qb = v260;	// L539
      }
      ac_int<5, false> v261 = b;	// L541
      int32_t v262 = v261;	// L542
      bool v263 = v262 == 25;	// L543
      if (v263) {	// L544
        int16_t v264 = __stateful_rf_0_r25_26;	// L545
        qb = v264;	// L546
      }
      ac_int<5, false> v265 = b;	// L548
      int32_t v266 = v265;	// L549
      bool v267 = v266 == 26;	// L550
      if (v267) {	// L551
        int16_t v268 = __stateful_rf_0_r26_27;	// L552
        qb = v268;	// L553
      }
      ac_int<5, false> v269 = b;	// L555
      int32_t v270 = v269;	// L556
      bool v271 = v270 == 27;	// L557
      if (v271) {	// L558
        int16_t v272 = __stateful_rf_0_r27_28;	// L559
        qb = v272;	// L560
      }
      ac_int<5, false> v273 = b;	// L562
      int32_t v274 = v273;	// L563
      bool v275 = v274 == 28;	// L564
      if (v275) {	// L565
        int16_t v276 = __stateful_rf_0_r28_29;	// L566
        qb = v276;	// L567
      }
      ac_int<5, false> v277 = b;	// L569
      int32_t v278 = v277;	// L570
      bool v279 = v278 == 29;	// L571
      if (v279) {	// L572
        int16_t v280 = __stateful_rf_0_r29_30;	// L573
        qb = v280;	// L574
      }
      ac_int<5, false> v281 = b;	// L576
      int32_t v282 = v281;	// L577
      bool v283 = v282 == 30;	// L578
      if (v283) {	// L579
        int16_t v284 = __stateful_rf_0_r30_31;	// L580
        qb = v284;	// L581
      }
      ac_int<5, false> v285 = b;	// L583
      int32_t v286 = v285;	// L584
      bool v287 = v286 == 31;	// L585
      if (v287) {	// L586
        int16_t v288 = __stateful_rf_0_r31_32;	// L587
        qb = v288;	// L588
      }
      ac_int<5, false> v289 = c;	// L590
      int32_t v290 = v289;	// L591
      bool v291 = v290 == 0;	// L592
      if (v291) {	// L593
        int16_t v292 = __stateful_rf_0_r0_1;	// L594
        qc = v292;	// L595
      }
      ac_int<5, false> v293 = c;	// L597
      int32_t v294 = v293;	// L598
      bool v295 = v294 == 1;	// L599
      if (v295) {	// L600
        int16_t v296 = __stateful_rf_0_r1_2;	// L601
        qc = v296;	// L602
      }
      ac_int<5, false> v297 = c;	// L604
      int32_t v298 = v297;	// L605
      bool v299 = v298 == 2;	// L606
      if (v299) {	// L607
        int16_t v300 = __stateful_rf_0_r2_3;	// L608
        qc = v300;	// L609
      }
      ac_int<5, false> v301 = c;	// L611
      int32_t v302 = v301;	// L612
      bool v303 = v302 == 3;	// L613
      if (v303) {	// L614
        int16_t v304 = __stateful_rf_0_r3_4;	// L615
        qc = v304;	// L616
      }
      ac_int<5, false> v305 = c;	// L618
      int32_t v306 = v305;	// L619
      bool v307 = v306 == 4;	// L620
      if (v307) {	// L621
        int16_t v308 = __stateful_rf_0_r4_5;	// L622
        qc = v308;	// L623
      }
      ac_int<5, false> v309 = c;	// L625
      int32_t v310 = v309;	// L626
      bool v311 = v310 == 5;	// L627
      if (v311) {	// L628
        int16_t v312 = __stateful_rf_0_r5_6;	// L629
        qc = v312;	// L630
      }
      ac_int<5, false> v313 = c;	// L632
      int32_t v314 = v313;	// L633
      bool v315 = v314 == 6;	// L634
      if (v315) {	// L635
        int16_t v316 = __stateful_rf_0_r6_7;	// L636
        qc = v316;	// L637
      }
      ac_int<5, false> v317 = c;	// L639
      int32_t v318 = v317;	// L640
      bool v319 = v318 == 7;	// L641
      if (v319) {	// L642
        int16_t v320 = __stateful_rf_0_r7_8;	// L643
        qc = v320;	// L644
      }
      ac_int<5, false> v321 = c;	// L646
      int32_t v322 = v321;	// L647
      bool v323 = v322 == 8;	// L648
      if (v323) {	// L649
        int16_t v324 = __stateful_rf_0_r8_9;	// L650
        qc = v324;	// L651
      }
      ac_int<5, false> v325 = c;	// L653
      int32_t v326 = v325;	// L654
      bool v327 = v326 == 9;	// L655
      if (v327) {	// L656
        int16_t v328 = __stateful_rf_0_r9_10;	// L657
        qc = v328;	// L658
      }
      ac_int<5, false> v329 = c;	// L660
      int32_t v330 = v329;	// L661
      bool v331 = v330 == 10;	// L662
      if (v331) {	// L663
        int16_t v332 = __stateful_rf_0_r10_11;	// L664
        qc = v332;	// L665
      }
      ac_int<5, false> v333 = c;	// L667
      int32_t v334 = v333;	// L668
      bool v335 = v334 == 11;	// L669
      if (v335) {	// L670
        int16_t v336 = __stateful_rf_0_r11_12;	// L671
        qc = v336;	// L672
      }
      ac_int<5, false> v337 = c;	// L674
      int32_t v338 = v337;	// L675
      bool v339 = v338 == 12;	// L676
      if (v339) {	// L677
        int16_t v340 = __stateful_rf_0_r12_13;	// L678
        qc = v340;	// L679
      }
      ac_int<5, false> v341 = c;	// L681
      int32_t v342 = v341;	// L682
      bool v343 = v342 == 13;	// L683
      if (v343) {	// L684
        int16_t v344 = __stateful_rf_0_r13_14;	// L685
        qc = v344;	// L686
      }
      ac_int<5, false> v345 = c;	// L688
      int32_t v346 = v345;	// L689
      bool v347 = v346 == 14;	// L690
      if (v347) {	// L691
        int16_t v348 = __stateful_rf_0_r14_15;	// L692
        qc = v348;	// L693
      }
      ac_int<5, false> v349 = c;	// L695
      int32_t v350 = v349;	// L696
      bool v351 = v350 == 15;	// L697
      if (v351) {	// L698
        int16_t v352 = __stateful_rf_0_r15_16;	// L699
        qc = v352;	// L700
      }
      ac_int<5, false> v353 = c;	// L702
      int32_t v354 = v353;	// L703
      bool v355 = v354 == 16;	// L704
      if (v355) {	// L705
        int16_t v356 = __stateful_rf_0_r16_17;	// L706
        qc = v356;	// L707
      }
      ac_int<5, false> v357 = c;	// L709
      int32_t v358 = v357;	// L710
      bool v359 = v358 == 17;	// L711
      if (v359) {	// L712
        int16_t v360 = __stateful_rf_0_r17_18;	// L713
        qc = v360;	// L714
      }
      ac_int<5, false> v361 = c;	// L716
      int32_t v362 = v361;	// L717
      bool v363 = v362 == 18;	// L718
      if (v363) {	// L719
        int16_t v364 = __stateful_rf_0_r18_19;	// L720
        qc = v364;	// L721
      }
      ac_int<5, false> v365 = c;	// L723
      int32_t v366 = v365;	// L724
      bool v367 = v366 == 19;	// L725
      if (v367) {	// L726
        int16_t v368 = __stateful_rf_0_r19_20;	// L727
        qc = v368;	// L728
      }
      ac_int<5, false> v369 = c;	// L730
      int32_t v370 = v369;	// L731
      bool v371 = v370 == 20;	// L732
      if (v371) {	// L733
        int16_t v372 = __stateful_rf_0_r20_21;	// L734
        qc = v372;	// L735
      }
      ac_int<5, false> v373 = c;	// L737
      int32_t v374 = v373;	// L738
      bool v375 = v374 == 21;	// L739
      if (v375) {	// L740
        int16_t v376 = __stateful_rf_0_r21_22;	// L741
        qc = v376;	// L742
      }
      ac_int<5, false> v377 = c;	// L744
      int32_t v378 = v377;	// L745
      bool v379 = v378 == 22;	// L746
      if (v379) {	// L747
        int16_t v380 = __stateful_rf_0_r22_23;	// L748
        qc = v380;	// L749
      }
      ac_int<5, false> v381 = c;	// L751
      int32_t v382 = v381;	// L752
      bool v383 = v382 == 23;	// L753
      if (v383) {	// L754
        int16_t v384 = __stateful_rf_0_r23_24;	// L755
        qc = v384;	// L756
      }
      ac_int<5, false> v385 = c;	// L758
      int32_t v386 = v385;	// L759
      bool v387 = v386 == 24;	// L760
      if (v387) {	// L761
        int16_t v388 = __stateful_rf_0_r24_25;	// L762
        qc = v388;	// L763
      }
      ac_int<5, false> v389 = c;	// L765
      int32_t v390 = v389;	// L766
      bool v391 = v390 == 25;	// L767
      if (v391) {	// L768
        int16_t v392 = __stateful_rf_0_r25_26;	// L769
        qc = v392;	// L770
      }
      ac_int<5, false> v393 = c;	// L772
      int32_t v394 = v393;	// L773
      bool v395 = v394 == 26;	// L774
      if (v395) {	// L775
        int16_t v396 = __stateful_rf_0_r26_27;	// L776
        qc = v396;	// L777
      }
      ac_int<5, false> v397 = c;	// L779
      int32_t v398 = v397;	// L780
      bool v399 = v398 == 27;	// L781
      if (v399) {	// L782
        int16_t v400 = __stateful_rf_0_r27_28;	// L783
        qc = v400;	// L784
      }
      ac_int<5, false> v401 = c;	// L786
      int32_t v402 = v401;	// L787
      bool v403 = v402 == 28;	// L788
      if (v403) {	// L789
        int16_t v404 = __stateful_rf_0_r28_29;	// L790
        qc = v404;	// L791
      }
      ac_int<5, false> v405 = c;	// L793
      int32_t v406 = v405;	// L794
      bool v407 = v406 == 29;	// L795
      if (v407) {	// L796
        int16_t v408 = __stateful_rf_0_r29_30;	// L797
        qc = v408;	// L798
      }
      ac_int<5, false> v409 = c;	// L800
      int32_t v410 = v409;	// L801
      bool v411 = v410 == 30;	// L802
      if (v411) {	// L803
        int16_t v412 = __stateful_rf_0_r30_31;	// L804
        qc = v412;	// L805
      }
      ac_int<5, false> v413 = c;	// L807
      int32_t v414 = v413;	// L808
      bool v415 = v414 == 31;	// L809
      if (v415) {	// L810
        int16_t v416 = __stateful_rf_0_r31_32;	// L811
        qc = v416;	// L812
      }
      uint16_t v417 = qa;	// L814
      v24.write(v417);	// L815
      uint16_t v418 = qb;	// L816
      v25.write(v418);	// L817
      uint16_t v419 = qc;	// L818
      v26.write(v419);	// L819
      bool v420 = e;	// L820
      if (v420) {	// L821
        ac_int<5, false> v421 = x;	// L822
        int32_t v422 = v421;	// L823
        bool v423 = v422 == 0;	// L824
        if (v423) {	// L825
          uint16_t v424 = d;	// L826
          __stateful_rf_0_r0_1 = v424;	// L827
        }
        ac_int<5, false> v425 = x;	// L829
        int32_t v426 = v425;	// L830
        bool v427 = v426 == 1;	// L831
        if (v427) {	// L832
          uint16_t v428 = d;	// L833
          __stateful_rf_0_r1_2 = v428;	// L834
        }
        ac_int<5, false> v429 = x;	// L836
        int32_t v430 = v429;	// L837
        bool v431 = v430 == 2;	// L838
        if (v431) {	// L839
          uint16_t v432 = d;	// L840
          __stateful_rf_0_r2_3 = v432;	// L841
        }
        ac_int<5, false> v433 = x;	// L843
        int32_t v434 = v433;	// L844
        bool v435 = v434 == 3;	// L845
        if (v435) {	// L846
          uint16_t v436 = d;	// L847
          __stateful_rf_0_r3_4 = v436;	// L848
        }
        ac_int<5, false> v437 = x;	// L850
        int32_t v438 = v437;	// L851
        bool v439 = v438 == 4;	// L852
        if (v439) {	// L853
          uint16_t v440 = d;	// L854
          __stateful_rf_0_r4_5 = v440;	// L855
        }
        ac_int<5, false> v441 = x;	// L857
        int32_t v442 = v441;	// L858
        bool v443 = v442 == 5;	// L859
        if (v443) {	// L860
          uint16_t v444 = d;	// L861
          __stateful_rf_0_r5_6 = v444;	// L862
        }
        ac_int<5, false> v445 = x;	// L864
        int32_t v446 = v445;	// L865
        bool v447 = v446 == 6;	// L866
        if (v447) {	// L867
          uint16_t v448 = d;	// L868
          __stateful_rf_0_r6_7 = v448;	// L869
        }
        ac_int<5, false> v449 = x;	// L871
        int32_t v450 = v449;	// L872
        bool v451 = v450 == 7;	// L873
        if (v451) {	// L874
          uint16_t v452 = d;	// L875
          __stateful_rf_0_r7_8 = v452;	// L876
        }
        ac_int<5, false> v453 = x;	// L878
        int32_t v454 = v453;	// L879
        bool v455 = v454 == 8;	// L880
        if (v455) {	// L881
          uint16_t v456 = d;	// L882
          __stateful_rf_0_r8_9 = v456;	// L883
        }
        ac_int<5, false> v457 = x;	// L885
        int32_t v458 = v457;	// L886
        bool v459 = v458 == 9;	// L887
        if (v459) {	// L888
          uint16_t v460 = d;	// L889
          __stateful_rf_0_r9_10 = v460;	// L890
        }
        ac_int<5, false> v461 = x;	// L892
        int32_t v462 = v461;	// L893
        bool v463 = v462 == 10;	// L894
        if (v463) {	// L895
          uint16_t v464 = d;	// L896
          __stateful_rf_0_r10_11 = v464;	// L897
        }
        ac_int<5, false> v465 = x;	// L899
        int32_t v466 = v465;	// L900
        bool v467 = v466 == 11;	// L901
        if (v467) {	// L902
          uint16_t v468 = d;	// L903
          __stateful_rf_0_r11_12 = v468;	// L904
        }
        ac_int<5, false> v469 = x;	// L906
        int32_t v470 = v469;	// L907
        bool v471 = v470 == 12;	// L908
        if (v471) {	// L909
          uint16_t v472 = d;	// L910
          __stateful_rf_0_r12_13 = v472;	// L911
        }
        ac_int<5, false> v473 = x;	// L913
        int32_t v474 = v473;	// L914
        bool v475 = v474 == 13;	// L915
        if (v475) {	// L916
          uint16_t v476 = d;	// L917
          __stateful_rf_0_r13_14 = v476;	// L918
        }
        ac_int<5, false> v477 = x;	// L920
        int32_t v478 = v477;	// L921
        bool v479 = v478 == 14;	// L922
        if (v479) {	// L923
          uint16_t v480 = d;	// L924
          __stateful_rf_0_r14_15 = v480;	// L925
        }
        ac_int<5, false> v481 = x;	// L927
        int32_t v482 = v481;	// L928
        bool v483 = v482 == 15;	// L929
        if (v483) {	// L930
          uint16_t v484 = d;	// L931
          __stateful_rf_0_r15_16 = v484;	// L932
        }
        ac_int<5, false> v485 = x;	// L934
        int32_t v486 = v485;	// L935
        bool v487 = v486 == 16;	// L936
        if (v487) {	// L937
          uint16_t v488 = d;	// L938
          __stateful_rf_0_r16_17 = v488;	// L939
        }
        ac_int<5, false> v489 = x;	// L941
        int32_t v490 = v489;	// L942
        bool v491 = v490 == 17;	// L943
        if (v491) {	// L944
          uint16_t v492 = d;	// L945
          __stateful_rf_0_r17_18 = v492;	// L946
        }
        ac_int<5, false> v493 = x;	// L948
        int32_t v494 = v493;	// L949
        bool v495 = v494 == 18;	// L950
        if (v495) {	// L951
          uint16_t v496 = d;	// L952
          __stateful_rf_0_r18_19 = v496;	// L953
        }
        ac_int<5, false> v497 = x;	// L955
        int32_t v498 = v497;	// L956
        bool v499 = v498 == 19;	// L957
        if (v499) {	// L958
          uint16_t v500 = d;	// L959
          __stateful_rf_0_r19_20 = v500;	// L960
        }
        ac_int<5, false> v501 = x;	// L962
        int32_t v502 = v501;	// L963
        bool v503 = v502 == 20;	// L964
        if (v503) {	// L965
          uint16_t v504 = d;	// L966
          __stateful_rf_0_r20_21 = v504;	// L967
        }
        ac_int<5, false> v505 = x;	// L969
        int32_t v506 = v505;	// L970
        bool v507 = v506 == 21;	// L971
        if (v507) {	// L972
          uint16_t v508 = d;	// L973
          __stateful_rf_0_r21_22 = v508;	// L974
        }
        ac_int<5, false> v509 = x;	// L976
        int32_t v510 = v509;	// L977
        bool v511 = v510 == 22;	// L978
        if (v511) {	// L979
          uint16_t v512 = d;	// L980
          __stateful_rf_0_r22_23 = v512;	// L981
        }
        ac_int<5, false> v513 = x;	// L983
        int32_t v514 = v513;	// L984
        bool v515 = v514 == 23;	// L985
        if (v515) {	// L986
          uint16_t v516 = d;	// L987
          __stateful_rf_0_r23_24 = v516;	// L988
        }
        ac_int<5, false> v517 = x;	// L990
        int32_t v518 = v517;	// L991
        bool v519 = v518 == 24;	// L992
        if (v519) {	// L993
          uint16_t v520 = d;	// L994
          __stateful_rf_0_r24_25 = v520;	// L995
        }
        ac_int<5, false> v521 = x;	// L997
        int32_t v522 = v521;	// L998
        bool v523 = v522 == 25;	// L999
        if (v523) {	// L1000
          uint16_t v524 = d;	// L1001
          __stateful_rf_0_r25_26 = v524;	// L1002
        }
        ac_int<5, false> v525 = x;	// L1004
        int32_t v526 = v525;	// L1005
        bool v527 = v526 == 26;	// L1006
        if (v527) {	// L1007
          uint16_t v528 = d;	// L1008
          __stateful_rf_0_r26_27 = v528;	// L1009
        }
        ac_int<5, false> v529 = x;	// L1011
        int32_t v530 = v529;	// L1012
        bool v531 = v530 == 27;	// L1013
        if (v531) {	// L1014
          uint16_t v532 = d;	// L1015
          __stateful_rf_0_r27_28 = v532;	// L1016
        }
        ac_int<5, false> v533 = x;	// L1018
        int32_t v534 = v533;	// L1019
        bool v535 = v534 == 28;	// L1020
        if (v535) {	// L1021
          uint16_t v536 = d;	// L1022
          __stateful_rf_0_r28_29 = v536;	// L1023
        }
        ac_int<5, false> v537 = x;	// L1025
        int32_t v538 = v537;	// L1026
        bool v539 = v538 == 29;	// L1027
        if (v539) {	// L1028
          uint16_t v540 = d;	// L1029
          __stateful_rf_0_r29_30 = v540;	// L1030
        }
        ac_int<5, false> v541 = x;	// L1032
        int32_t v542 = v541;	// L1033
        bool v543 = v542 == 30;	// L1034
        if (v543) {	// L1035
          uint16_t v544 = d;	// L1036
          __stateful_rf_0_r30_31 = v544;	// L1037
        }
        ac_int<5, false> v545 = x;	// L1039
        int32_t v546 = v545;	// L1040
        bool v547 = v546 == 31;	// L1041
        if (v547) {	// L1042
          uint16_t v548 = d;	// L1043
          __stateful_rf_0_r31_32 = v548;	// L1044
        }
      }
      wait();  // no handshake in the body: the cycle boundary, in csim and synthesis
    }
    done.write(true);  // RTL-observable completion (see the done port)
#ifndef __SYNTHESIS__
    __allo_done++; // csim: this kernel finished its single pass
#endif
    while (1) { wait(); }
  }
};

SC_MODULE(sink_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_out<bool> done;
  Connections::Out< ac_int<16, false> > v549;
  Connections::Out< ac_int<16, false> > v550;
  Connections::Out< ac_int<16, false> > v551;
  sc_in< ac_int<16, false> > v552;
  sc_in< ac_int<16, false> > v553;
  sc_in< ac_int<16, false> > v554;
  SC_HAS_PROCESS(sink_0);
  sink_0(sc_module_name n) : sc_module(n), done("done"), v549("v549"), v550("v550"), v551("v551") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  void run() {
    v549.Reset();
    v550.Reset();
    v551.Reset();
    done.write(false);  // completion flag low until the pass finishes
    wait();
    l_S_t_0_t2: for (int t2 = 0; t2 < 64; t2++) {	// L1051
      uint16_t v555 = v552.read();	// L1052
      v549.Push(v555);	// L1053
      uint16_t v556 = v553.read();	// L1054
      v550.Push(v556);	// L1055
      uint16_t v557 = v554.read();	// L1056
      v551.Push(v557);	// L1057
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
  Connections::In< ac_int<5, false> > v558;
  Connections::In< ac_int<5, false> > v559;
  Connections::In< ac_int<5, false> > v560;
  Connections::In< ac_int<5, false> > v561;
  Connections::In< ac_int<16, false> > v562;
  Connections::In< ac_int<1, false> > v563;
  Connections::Out< ac_int<16, false> > v564;
  Connections::Out< ac_int<16, false> > v565;
  Connections::Out< ac_int<16, false> > v566;
  sc_signal< ac_int<5, false> > v567;
  sc_signal< ac_int<5, false> > v568;
  sc_signal< ac_int<5, false> > v569;
  sc_signal< ac_int<5, false> > v570;
  sc_signal< ac_int<16, false> > v571;
  sc_signal< ac_int<1, false> > v572;
  sc_signal< ac_int<16, false> > v573;
  sc_signal< ac_int<16, false> > v574;
  sc_signal< ac_int<16, false> > v575;
  src_0 u0;
  rf_0 u1;
  sink_0 u2;
  sc_signal<bool> u0_done;
  sc_signal<bool> u1_done;
  sc_signal<bool> u2_done;
  SC_CTOR(top) : v558("v558"), v559("v559"), v560("v560"), v561("v561"), v562("v562"), v563("v563"), v564("v564"), v565("v565"), v566("v566"), v567("v567"), v568("v568"), v569("v569"), v570("v570"), v571("v571"), v572("v572"), v573("v573"), v574("v574"), v575("v575"), u0("u0"), u1("u1"), u2("u2") {
    u0.clk(clk);
    u0.rst(rst);
    u0.done(u0_done);
    u0.v0(v558);
    u0.v1(v559);
    u0.v2(v560);
    u0.v3(v561);
    u0.v4(v562);
    u0.v5(v563);
    u0.v6(v567);
    u0.v7(v568);
    u0.v8(v569);
    u0.v9(v570);
    u0.v10(v571);
    u0.v11(v572);
    u1.clk(clk);
    u1.rst(rst);
    u1.done(u1_done);
    u1.v18(v567);
    u1.v19(v568);
    u1.v20(v569);
    u1.v21(v570);
    u1.v22(v571);
    u1.v23(v572);
    u1.v24(v573);
    u1.v25(v574);
    u1.v26(v575);
    u2.clk(clk);
    u2.rst(rst);
    u2.done(u2_done);
    u2.v549(v564);
    u2.v550(v565);
    u2.v551(v566);
    u2.v552(v573);
    u2.v553(v574);
    u2.v554(v575);
    SC_METHOD(_agg_done); sensitive << u0_done << u1_done << u2_done;
  }
  void _agg_done() { done.write(u0_done.read() && u1_done.read() && u2_done.read()); }
};

SC_MODULE(tb) {
  sc_clock clk;
  sc_signal<bool> rst;
  top dut;
  sc_signal<bool> done_sig;  // DUT completion (polled by sc_main)
  Connections::Combinational< ac_int<5, false> > ch_v558;
  Connections::Combinational< ac_int<5, false> > ch_v559;
  Connections::Combinational< ac_int<5, false> > ch_v560;
  Connections::Combinational< ac_int<5, false> > ch_v561;
  Connections::Combinational< ac_int<16, false> > ch_v562;
  Connections::Combinational< ac_int<1, false> > ch_v563;
  Connections::Combinational< ac_int<16, false> > ch_v564;
  Connections::Combinational< ac_int<16, false> > ch_v565;
  Connections::Combinational< ac_int<16, false> > ch_v566;
  SC_HAS_PROCESS(tb);
  tb(sc_module_name n) : sc_module(n), clk("clk", 1, SC_NS), dut("dut"), ch_v558("ch_v558"), ch_v559("ch_v559"), ch_v560("ch_v560"), ch_v561("ch_v561"), ch_v562("ch_v562"), ch_v563("ch_v563"), ch_v564("ch_v564"), ch_v565("ch_v565"), ch_v566("ch_v566") {
    dut.clk(clk); dut.rst(rst); dut.done(done_sig);
    dut.v558(ch_v558);
    dut.v559(ch_v559);
    dut.v560(ch_v560);
    dut.v561(ch_v561);
    dut.v562(ch_v562);
    dut.v563(ch_v563);
    dut.v564(ch_v564);
    dut.v565(ch_v565);
    dut.v566(ch_v566);
    SC_THREAD(src_v558); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(src_v559); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(src_v560); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(src_v561); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(src_v562); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(src_v563); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(snk_v564); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(snk_v565); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(snk_v566); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
  }
  int _snk_done = 0;  // stream sinks drained; the last one sc_stop()s
  void src_v558() {
    ch_v558.ResetWrite();
    wait();
    { std::ifstream _f("input0.data"); long long _v; for (int f = 0; f < 64; ++f) { _f >> _v; ch_v558.Push((ac_int<5, false>)_v); } }
  }
  void src_v559() {
    ch_v559.ResetWrite();
    wait();
    { std::ifstream _f("input1.data"); long long _v; for (int f = 0; f < 64; ++f) { _f >> _v; ch_v559.Push((ac_int<5, false>)_v); } }
  }
  void src_v560() {
    ch_v560.ResetWrite();
    wait();
    { std::ifstream _f("input2.data"); long long _v; for (int f = 0; f < 64; ++f) { _f >> _v; ch_v560.Push((ac_int<5, false>)_v); } }
  }
  void src_v561() {
    ch_v561.ResetWrite();
    wait();
    { std::ifstream _f("input3.data"); long long _v; for (int f = 0; f < 64; ++f) { _f >> _v; ch_v561.Push((ac_int<5, false>)_v); } }
  }
  void src_v562() {
    ch_v562.ResetWrite();
    wait();
    { std::ifstream _f("input4.data"); long long _v; for (int f = 0; f < 64; ++f) { _f >> _v; ch_v562.Push((ac_int<16, false>)_v); } }
  }
  void src_v563() {
    ch_v563.ResetWrite();
    wait();
    { std::ifstream _f("input5.data"); long long _v; for (int f = 0; f < 64; ++f) { _f >> _v; ch_v563.Push((ac_int<1, false>)_v); } }
  }
  void snk_v564() {
    ch_v564.ResetRead();
    wait();
    { std::ofstream _f("output0.data"); for (int f = 0; f < 64; ++f) _f << (long long)(ch_v564.Pop()) << "\n"; }
    if (++_snk_done == 3) sc_stop();
  }
  void snk_v565() {
    ch_v565.ResetRead();
    wait();
    { std::ofstream _f("output1.data"); for (int f = 0; f < 64; ++f) _f << (long long)(ch_v565.Pop()) << "\n"; }
    if (++_snk_done == 3) sc_stop();
  }
  void snk_v566() {
    ch_v566.ResetRead();
    wait();
    { std::ofstream _f("output2.data"); for (int f = 0; f < 64; ++f) _f << (long long)(ch_v566.Pop()) << "\n"; }
    if (++_snk_done == 3) sc_stop();
  }
};

int sc_main(int, char *[]) {
  static tb t("t");
  #ifdef CONNECTIONS_ACCURATE_SIM
  Connections::set_sim_clk(&t.clk);
  #endif
  t.rst = 0; sc_start(1, SC_NS);
  t.rst = 1;
  #ifndef ALLO_TB_MAX_CYCLES
  #define ALLO_TB_MAX_CYCLES 328000LL
  #endif
  sc_start((double)ALLO_TB_MAX_CYCLES, SC_NS);
  if (sc_core::sc_get_status() != sc_core::SC_STOPPED) {
    std::cerr << "TB DEADLOCK: stream outputs not drained after " << ALLO_TB_MAX_CYCLES << " cycles (" << t._snk_done << " of 3 sinks done)" << std::endl;
    return 1;
  }
  return 0;
}
