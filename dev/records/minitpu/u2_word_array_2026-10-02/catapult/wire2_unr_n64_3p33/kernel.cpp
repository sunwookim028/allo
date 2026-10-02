
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
  Connections::In< ac_int<1, false> > v0;
  Connections::In< ac_int<1, false> > v1;
  Connections::In< ac_int<3, false> > v2;
  Connections::In< ac_int<64, false> > v3;
  Connections::In< ac_int<1, false> > v4;
  Connections::In< ac_int<1, false> > v5;
  Connections::In< ac_int<3, false> > v6;
  Connections::In< ac_int<64, false> > v7;
  sc_out< ac_int<1, false> > v8;
  sc_out< ac_int<1, false> > v9;
  sc_out< ac_int<3, false> > v10;
  sc_out< ac_int<64, false> > v11;
  sc_out< ac_int<1, false> > v12;
  sc_out< ac_int<1, false> > v13;
  sc_out< ac_int<3, false> > v14;
  sc_out< ac_int<64, false> > v15;
  SC_HAS_PROCESS(src_0);
  src_0(sc_module_name n) : sc_module(n), done("done"), v0("v0"), v1("v1"), v2("v2"), v3("v3"), v4("v4"), v5("v5"), v6("v6"), v7("v7") {
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
    v6.Reset();
    v7.Reset();
    v8.write(0);
    v9.write(0);
    v10.write(0);
    v11.write(0);
    v12.write(0);
    v13.write(0);
    v14.write(0);
    v15.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    l_S_t_0_t: for (int t = 0; t < 64; t++) {	// L4
      bool v16 = v0.Pop();	// L5
      v8.write(v16);	// L6
      bool v17 = v1.Pop();	// L7
      v9.write(v17);	// L8
      ac_int<3, false> v18 = v2.Pop();	// L9
      v10.write(v18);	// L10
      uint64_t v19 = v3.Pop();	// L11
      v11.write(v19);	// L12
      bool v20 = v4.Pop();	// L13
      v12.write(v20);	// L14
      bool v21 = v5.Pop();	// L15
      v13.write(v21);	// L16
      ac_int<3, false> v22 = v6.Pop();	// L17
      v14.write(v22);	// L18
      uint64_t v23 = v7.Pop();	// L19
      v15.write(v23);	// L20
    }
    done.write(true);  // RTL-observable completion (see the done port)
#ifndef __SYNTHESIS__
    __allo_done++; // csim: this kernel finished its single pass
#endif
    while (1) { wait(); }
  }
};

SC_MODULE(wa_0) {
  sc_in_clk clk;
  sc_in<bool> rst;
  sc_out<bool> done;
  sc_in< ac_int<1, false> > v24;
  sc_in< ac_int<1, false> > v25;
  sc_in< ac_int<3, false> > v26;
  sc_in< ac_int<64, false> > v27;
  sc_in< ac_int<1, false> > v28;
  sc_in< ac_int<1, false> > v29;
  sc_in< ac_int<3, false> > v30;
  sc_in< ac_int<64, false> > v31;
  sc_out< ac_int<64, false> > v32;
  sc_out< ac_int<64, false> > v33;
  SC_HAS_PROCESS(wa_0);
  wa_0(sc_module_name n) : sc_module(n), done("done") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  void run() {
    v32.write(0);
    v33.write(0);
    done.write(false);  // completion flag low until the pass finishes
    wait();
    uint64_t mem[8];	// L26
    uint64_t pc[3];	// L27
    uint64_t pd[2];	// L28
#ifdef __SYNTHESIS__
    done.write(true);  // steady-state: no completion, so assert on entry (the post-body write is unreachable here)
    #pragma hls_pipeline_init_interval 1
    while (1) {  // steady-state loop (was `for t`): 1 iteration = 1 step
#else
    #pragma hls_pipeline_init_interval 1
    l_steady: for (int t1 = 0; t1 < 64; t1 += 1) {
#endif
      #pragma hls_unroll
      l_S_k_0_k: for (int k = 0; k < 2; k++) {	// L30
        int v34 = (k + 1);	// L31
        uint64_t v35 = pc[((-v34) + 2)];	// L32
        pc[((-v34) + 3)] = v35;	// L33
      }
      #pragma hls_unroll
      l_S_j_1_j: for (int j = 0; j < 1; j++) {	// L35
        int v36 = (j + 1);	// L36
        uint64_t v37 = pd[((-v36) + 1)];	// L37
        pd[((-v36) + 2)] = v37;	// L38
      }
      bool v38 = v24.read();	// L40
      bool e_c;	// L41
      e_c = v38;	// L42
      bool v39 = v25.read();	// L43
      bool w_c;	// L44
      w_c = v39;	// L45
      ac_int<3, false> v40 = v26.read();	// L46
      ac_int<3, false> a_ca;	// L47
      a_ca = v40;	// L48
      ac_int<3, false> v41 = a_ca;	// L49
      int32_t v42 = v41;	// L50
      int32_t a_c;	// L51
      a_c = v42;	// L52
      uint64_t v43 = v27.read();	// L53
      uint64_t d_c;	// L54
      d_c = v43;	// L55
      bool v44 = v28.read();	// L56
      bool e_d;	// L57
      e_d = v44;	// L58
      bool v45 = v29.read();	// L59
      bool w_d;	// L60
      w_d = v45;	// L61
      ac_int<3, false> v46 = v30.read();	// L62
      ac_int<3, false> a_da;	// L63
      a_da = v46;	// L64
      ac_int<3, false> v47 = a_da;	// L65
      int32_t v48 = v47;	// L66
      int32_t a_d;	// L67
      a_d = v48;	// L68
      uint64_t v49 = v31.read();	// L69
      uint64_t d_d;	// L70
      d_d = v49;	// L71
      bool v50 = e_c;	// L72
      int32_t v51 = v50;	// L73
      bool v52 = v51 == 1;	// L74
      if (v52) {	// L75
        bool v53 = w_c;	// L76
        int32_t v54 = v53;	// L77
        bool v55 = v54 == 1;	// L78
        if (v55) {	// L79
          uint64_t v56 = d_c;	// L80
          int32_t v57 = a_c;	// L81
          int v58 = v57;	// L82
          mem[v58] = v56;	// L83
        } else {
          int32_t v59 = a_c;	// L85
          int v60 = v59;	// L86
          uint64_t v61 = mem[v60];	// L87
          pc[0] = v61;	// L88
        }
      }
      bool v62 = e_d;	// L91
      int32_t v63 = v62;	// L92
      bool v64 = v63 == 1;	// L93
      if (v64) {	// L94
        bool v65 = w_d;	// L95
        int32_t v66 = v65;	// L96
        bool v67 = v66 == 1;	// L97
        if (v67) {	// L98
          uint64_t v68 = d_d;	// L99
          int32_t v69 = a_d;	// L100
          int v70 = v69;	// L101
          mem[v70] = v68;	// L102
        } else {
          int32_t v71 = a_d;	// L104
          int v72 = v71;	// L105
          uint64_t v73 = mem[v72];	// L106
          pd[0] = v73;	// L107
        }
      }
      uint64_t v74 = pc[2];	// L110
      v32.write(v74);	// L111
      uint64_t v75 = pd[1];	// L112
      v33.write(v75);	// L113
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
  Connections::Out< ac_int<64, false> > v76;
  Connections::Out< ac_int<64, false> > v77;
  sc_in< ac_int<64, false> > v78;
  sc_in< ac_int<64, false> > v79;
  SC_HAS_PROCESS(sink_0);
  sink_0(sc_module_name n) : sc_module(n), done("done"), v76("v76"), v77("v77") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  void run() {
    v76.Reset();
    v77.Reset();
    done.write(false);  // completion flag low until the pass finishes
    wait();
    l_S_t_0_t2: for (int t2 = 0; t2 < 64; t2++) {	// L118
      uint64_t v80 = v78.read();	// L119
      v76.Push(v80);	// L120
      uint64_t v81 = v79.read();	// L121
      v77.Push(v81);	// L122
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
  Connections::In< ac_int<1, false> > v82;
  Connections::In< ac_int<1, false> > v83;
  Connections::In< ac_int<3, false> > v84;
  Connections::In< ac_int<64, false> > v85;
  Connections::In< ac_int<1, false> > v86;
  Connections::In< ac_int<1, false> > v87;
  Connections::In< ac_int<3, false> > v88;
  Connections::In< ac_int<64, false> > v89;
  Connections::Out< ac_int<64, false> > v90;
  Connections::Out< ac_int<64, false> > v91;
  sc_signal< ac_int<1, false> > v92;
  sc_signal< ac_int<1, false> > v93;
  sc_signal< ac_int<3, false> > v94;
  sc_signal< ac_int<64, false> > v95;
  sc_signal< ac_int<1, false> > v96;
  sc_signal< ac_int<1, false> > v97;
  sc_signal< ac_int<3, false> > v98;
  sc_signal< ac_int<64, false> > v99;
  sc_signal< ac_int<64, false> > v100;
  sc_signal< ac_int<64, false> > v101;
  src_0 u0;
  wa_0 u1;
  sink_0 u2;
  sc_signal<bool> u0_done;
  sc_signal<bool> u1_done;
  sc_signal<bool> u2_done;
  SC_CTOR(top) : v82("v82"), v83("v83"), v84("v84"), v85("v85"), v86("v86"), v87("v87"), v88("v88"), v89("v89"), v90("v90"), v91("v91"), v92("v92"), v93("v93"), v94("v94"), v95("v95"), v96("v96"), v97("v97"), v98("v98"), v99("v99"), v100("v100"), v101("v101"), u0("u0"), u1("u1"), u2("u2") {
    u0.clk(clk);
    u0.rst(rst);
    u0.done(u0_done);
    u0.v0(v82);
    u0.v1(v83);
    u0.v2(v84);
    u0.v3(v85);
    u0.v4(v86);
    u0.v5(v87);
    u0.v6(v88);
    u0.v7(v89);
    u0.v8(v92);
    u0.v9(v93);
    u0.v10(v94);
    u0.v11(v95);
    u0.v12(v96);
    u0.v13(v97);
    u0.v14(v98);
    u0.v15(v99);
    u1.clk(clk);
    u1.rst(rst);
    u1.done(u1_done);
    u1.v24(v92);
    u1.v25(v93);
    u1.v26(v94);
    u1.v27(v95);
    u1.v28(v96);
    u1.v29(v97);
    u1.v30(v98);
    u1.v31(v99);
    u1.v32(v100);
    u1.v33(v101);
    u2.clk(clk);
    u2.rst(rst);
    u2.done(u2_done);
    u2.v76(v90);
    u2.v77(v91);
    u2.v78(v100);
    u2.v79(v101);
    SC_METHOD(_agg_done); sensitive << u0_done << u1_done << u2_done;
  }
  void _agg_done() { done.write(u0_done.read() && u1_done.read() && u2_done.read()); }
};

SC_MODULE(tb) {
  sc_clock clk;
  sc_signal<bool> rst;
  top dut;
  sc_signal<bool> done_sig;  // DUT completion (polled by sc_main)
  Connections::Combinational< ac_int<1, false> > ch_v82;
  Connections::Combinational< ac_int<1, false> > ch_v83;
  Connections::Combinational< ac_int<3, false> > ch_v84;
  Connections::Combinational< ac_int<64, false> > ch_v85;
  Connections::Combinational< ac_int<1, false> > ch_v86;
  Connections::Combinational< ac_int<1, false> > ch_v87;
  Connections::Combinational< ac_int<3, false> > ch_v88;
  Connections::Combinational< ac_int<64, false> > ch_v89;
  Connections::Combinational< ac_int<64, false> > ch_v90;
  Connections::Combinational< ac_int<64, false> > ch_v91;
  SC_HAS_PROCESS(tb);
  tb(sc_module_name n) : sc_module(n), clk("clk", 1, SC_NS), dut("dut"), ch_v82("ch_v82"), ch_v83("ch_v83"), ch_v84("ch_v84"), ch_v85("ch_v85"), ch_v86("ch_v86"), ch_v87("ch_v87"), ch_v88("ch_v88"), ch_v89("ch_v89"), ch_v90("ch_v90"), ch_v91("ch_v91") {
    dut.clk(clk); dut.rst(rst); dut.done(done_sig);
    dut.v82(ch_v82);
    dut.v83(ch_v83);
    dut.v84(ch_v84);
    dut.v85(ch_v85);
    dut.v86(ch_v86);
    dut.v87(ch_v87);
    dut.v88(ch_v88);
    dut.v89(ch_v89);
    dut.v90(ch_v90);
    dut.v91(ch_v91);
    SC_THREAD(src_v82); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(src_v83); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(src_v84); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(src_v85); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(src_v86); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(src_v87); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(src_v88); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(src_v89); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(snk_v90); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(snk_v91); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
  }
  int _snk_done = 0;  // stream sinks drained; the last one sc_stop()s
  void src_v82() {
    ch_v82.ResetWrite();
    wait();
    { std::ifstream _f("input0.data"); long long _v; for (int f = 0; f < 64; ++f) { _f >> _v; ch_v82.Push((ac_int<1, false>)_v); } }
  }
  void src_v83() {
    ch_v83.ResetWrite();
    wait();
    { std::ifstream _f("input1.data"); long long _v; for (int f = 0; f < 64; ++f) { _f >> _v; ch_v83.Push((ac_int<1, false>)_v); } }
  }
  void src_v84() {
    ch_v84.ResetWrite();
    wait();
    { std::ifstream _f("input2.data"); long long _v; for (int f = 0; f < 64; ++f) { _f >> _v; ch_v84.Push((ac_int<3, false>)_v); } }
  }
  void src_v85() {
    ch_v85.ResetWrite();
    wait();
    { std::ifstream _f("input3.data"); long long _v; for (int f = 0; f < 64; ++f) { _f >> _v; ch_v85.Push((ac_int<64, false>)_v); } }
  }
  void src_v86() {
    ch_v86.ResetWrite();
    wait();
    { std::ifstream _f("input4.data"); long long _v; for (int f = 0; f < 64; ++f) { _f >> _v; ch_v86.Push((ac_int<1, false>)_v); } }
  }
  void src_v87() {
    ch_v87.ResetWrite();
    wait();
    { std::ifstream _f("input5.data"); long long _v; for (int f = 0; f < 64; ++f) { _f >> _v; ch_v87.Push((ac_int<1, false>)_v); } }
  }
  void src_v88() {
    ch_v88.ResetWrite();
    wait();
    { std::ifstream _f("input6.data"); long long _v; for (int f = 0; f < 64; ++f) { _f >> _v; ch_v88.Push((ac_int<3, false>)_v); } }
  }
  void src_v89() {
    ch_v89.ResetWrite();
    wait();
    { std::ifstream _f("input7.data"); long long _v; for (int f = 0; f < 64; ++f) { _f >> _v; ch_v89.Push((ac_int<64, false>)_v); } }
  }
  void snk_v90() {
    ch_v90.ResetRead();
    wait();
    { std::ofstream _f("output0.data"); for (int f = 0; f < 64; ++f) _f << (long long)(ch_v90.Pop()) << "\n"; }
    if (++_snk_done == 2) sc_stop();
  }
  void snk_v91() {
    ch_v91.ResetRead();
    wait();
    { std::ofstream _f("output1.data"); for (int f = 0; f < 64; ++f) _f << (long long)(ch_v91.Pop()) << "\n"; }
    if (++_snk_done == 2) sc_stop();
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
    std::cerr << "TB DEADLOCK: stream outputs not drained after " << ALLO_TB_MAX_CYCLES << " cycles (" << t._snk_done << " of 2 sinks done)" << std::endl;
    return 1;
  }
  return 0;
}
