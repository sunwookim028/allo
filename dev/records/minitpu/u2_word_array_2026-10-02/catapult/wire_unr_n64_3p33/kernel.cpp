
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
    uint64_t mem[8];	// L27
    uint64_t pc[3];	// L28
    uint64_t pd[2];	// L29
#ifdef __SYNTHESIS__
    done.write(true);  // steady-state: no completion, so assert on entry (the post-body write is unreachable here)
    #pragma hls_pipeline_init_interval 1
    while (1) {  // steady-state loop (was `for t`): 1 iteration = 1 step
#else
    #pragma hls_pipeline_init_interval 1
    l_steady: for (int t1 = 0; t1 < 64; t1 += 1) {
#endif
      #pragma hls_unroll
      l_S_k_0_k: for (int k = 0; k < 2; k++) {	// L31
        int v34 = (k + 1);	// L32
        uint64_t v35 = pc[((-v34) + 2)];	// L33
        pc[((-v34) + 3)] = v35;	// L34
      }
      #pragma hls_unroll
      l_S_j_1_j: for (int j = 0; j < 1; j++) {	// L36
        int v36 = (j + 1);	// L37
        uint64_t v37 = pd[((-v36) + 1)];	// L38
        pd[((-v36) + 2)] = v37;	// L39
      }
      bool v38 = v24.read();	// L41
      bool e_c;	// L42
      e_c = v38;	// L43
      bool v39 = v25.read();	// L44
      bool w_c;	// L45
      w_c = v39;	// L46
      ac_int<3, false> v40 = v26.read();	// L47
      ac_int<3, false> a_ca;	// L48
      a_ca = v40;	// L49
      ac_int<3, false> v41 = a_ca;	// L50
      int32_t v42 = v41;	// L51
      int32_t a_c;	// L52
      a_c = v42;	// L53
      uint64_t v43 = v27.read();	// L54
      uint64_t d_c;	// L55
      d_c = v43;	// L56
      bool v44 = v28.read();	// L57
      bool e_d;	// L58
      e_d = v44;	// L59
      bool v45 = v29.read();	// L60
      bool w_d;	// L61
      w_d = v45;	// L62
      ac_int<3, false> v46 = v30.read();	// L63
      ac_int<3, false> a_da;	// L64
      a_da = v46;	// L65
      ac_int<3, false> v47 = a_da;	// L66
      int32_t v48 = v47;	// L67
      int32_t a_d;	// L68
      a_d = v48;	// L69
      uint64_t v49 = v31.read();	// L70
      uint64_t d_d;	// L71
      d_d = v49;	// L72
      bool v50 = e_c;	// L73
      int32_t v51 = v50;	// L74
      bool v52 = v51 == 1;	// L75
      if (v52) {	// L76
        bool v53 = w_c;	// L77
        int32_t v54 = v53;	// L78
        bool v55 = v54 == 0;	// L79
        if (v55) {	// L80
          int32_t v56 = a_c;	// L81
          int v57 = v56;	// L82
          uint64_t v58 = mem[v57];	// L83
          pc[0] = v58;	// L84
        }
      }
      bool v59 = e_d;	// L87
      int32_t v60 = v59;	// L88
      bool v61 = v60 == 1;	// L89
      if (v61) {	// L90
        bool v62 = w_d;	// L91
        int32_t v63 = v62;	// L92
        bool v64 = v63 == 0;	// L93
        if (v64) {	// L94
          int32_t v65 = a_d;	// L95
          int v66 = v65;	// L96
          uint64_t v67 = mem[v66];	// L97
          pd[0] = v67;	// L98
        }
      }
      bool v68 = e_c;	// L101
      int32_t v69 = v68;	// L102
      bool v70 = v69 == 1;	// L103
      if (v70) {	// L104
        bool v71 = w_c;	// L105
        int32_t v72 = v71;	// L106
        bool v73 = v72 == 1;	// L107
        if (v73) {	// L108
          uint64_t v74 = d_c;	// L109
          int32_t v75 = a_c;	// L110
          int v76 = v75;	// L111
          mem[v76] = v74;	// L112
        }
      }
      bool v77 = e_d;	// L115
      int32_t v78 = v77;	// L116
      bool v79 = v78 == 1;	// L117
      if (v79) {	// L118
        bool v80 = w_d;	// L119
        int32_t v81 = v80;	// L120
        bool v82 = v81 == 1;	// L121
        if (v82) {	// L122
          uint64_t v83 = d_d;	// L123
          int32_t v84 = a_d;	// L124
          int v85 = v84;	// L125
          mem[v85] = v83;	// L126
        }
      }
      uint64_t v86 = pc[2];	// L129
      v32.write(v86);	// L130
      uint64_t v87 = pd[1];	// L131
      v33.write(v87);	// L132
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
  Connections::Out< ac_int<64, false> > v88;
  Connections::Out< ac_int<64, false> > v89;
  sc_in< ac_int<64, false> > v90;
  sc_in< ac_int<64, false> > v91;
  SC_HAS_PROCESS(sink_0);
  sink_0(sc_module_name n) : sc_module(n), done("done"), v88("v88"), v89("v89") {
    SC_THREAD(run);
    sensitive << clk.pos();
    async_reset_signal_is(rst, false);
  }
  void run() {
    v88.Reset();
    v89.Reset();
    done.write(false);  // completion flag low until the pass finishes
    wait();
    l_S_t_0_t2: for (int t2 = 0; t2 < 64; t2++) {	// L137
      uint64_t v92 = v90.read();	// L138
      v88.Push(v92);	// L139
      uint64_t v93 = v91.read();	// L140
      v89.Push(v93);	// L141
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
  Connections::In< ac_int<1, false> > v94;
  Connections::In< ac_int<1, false> > v95;
  Connections::In< ac_int<3, false> > v96;
  Connections::In< ac_int<64, false> > v97;
  Connections::In< ac_int<1, false> > v98;
  Connections::In< ac_int<1, false> > v99;
  Connections::In< ac_int<3, false> > v100;
  Connections::In< ac_int<64, false> > v101;
  Connections::Out< ac_int<64, false> > v102;
  Connections::Out< ac_int<64, false> > v103;
  sc_signal< ac_int<1, false> > v104;
  sc_signal< ac_int<1, false> > v105;
  sc_signal< ac_int<3, false> > v106;
  sc_signal< ac_int<64, false> > v107;
  sc_signal< ac_int<1, false> > v108;
  sc_signal< ac_int<1, false> > v109;
  sc_signal< ac_int<3, false> > v110;
  sc_signal< ac_int<64, false> > v111;
  sc_signal< ac_int<64, false> > v112;
  sc_signal< ac_int<64, false> > v113;
  src_0 u0;
  wa_0 u1;
  sink_0 u2;
  sc_signal<bool> u0_done;
  sc_signal<bool> u1_done;
  sc_signal<bool> u2_done;
  SC_CTOR(top) : v94("v94"), v95("v95"), v96("v96"), v97("v97"), v98("v98"), v99("v99"), v100("v100"), v101("v101"), v102("v102"), v103("v103"), v104("v104"), v105("v105"), v106("v106"), v107("v107"), v108("v108"), v109("v109"), v110("v110"), v111("v111"), v112("v112"), v113("v113"), u0("u0"), u1("u1"), u2("u2") {
    u0.clk(clk);
    u0.rst(rst);
    u0.done(u0_done);
    u0.v0(v94);
    u0.v1(v95);
    u0.v2(v96);
    u0.v3(v97);
    u0.v4(v98);
    u0.v5(v99);
    u0.v6(v100);
    u0.v7(v101);
    u0.v8(v104);
    u0.v9(v105);
    u0.v10(v106);
    u0.v11(v107);
    u0.v12(v108);
    u0.v13(v109);
    u0.v14(v110);
    u0.v15(v111);
    u1.clk(clk);
    u1.rst(rst);
    u1.done(u1_done);
    u1.v24(v104);
    u1.v25(v105);
    u1.v26(v106);
    u1.v27(v107);
    u1.v28(v108);
    u1.v29(v109);
    u1.v30(v110);
    u1.v31(v111);
    u1.v32(v112);
    u1.v33(v113);
    u2.clk(clk);
    u2.rst(rst);
    u2.done(u2_done);
    u2.v88(v102);
    u2.v89(v103);
    u2.v90(v112);
    u2.v91(v113);
    SC_METHOD(_agg_done); sensitive << u0_done << u1_done << u2_done;
  }
  void _agg_done() { done.write(u0_done.read() && u1_done.read() && u2_done.read()); }
};

SC_MODULE(tb) {
  sc_clock clk;
  sc_signal<bool> rst;
  top dut;
  sc_signal<bool> done_sig;  // DUT completion (polled by sc_main)
  Connections::Combinational< ac_int<1, false> > ch_v94;
  Connections::Combinational< ac_int<1, false> > ch_v95;
  Connections::Combinational< ac_int<3, false> > ch_v96;
  Connections::Combinational< ac_int<64, false> > ch_v97;
  Connections::Combinational< ac_int<1, false> > ch_v98;
  Connections::Combinational< ac_int<1, false> > ch_v99;
  Connections::Combinational< ac_int<3, false> > ch_v100;
  Connections::Combinational< ac_int<64, false> > ch_v101;
  Connections::Combinational< ac_int<64, false> > ch_v102;
  Connections::Combinational< ac_int<64, false> > ch_v103;
  SC_HAS_PROCESS(tb);
  tb(sc_module_name n) : sc_module(n), clk("clk", 1, SC_NS), dut("dut"), ch_v94("ch_v94"), ch_v95("ch_v95"), ch_v96("ch_v96"), ch_v97("ch_v97"), ch_v98("ch_v98"), ch_v99("ch_v99"), ch_v100("ch_v100"), ch_v101("ch_v101"), ch_v102("ch_v102"), ch_v103("ch_v103") {
    dut.clk(clk); dut.rst(rst); dut.done(done_sig);
    dut.v94(ch_v94);
    dut.v95(ch_v95);
    dut.v96(ch_v96);
    dut.v97(ch_v97);
    dut.v98(ch_v98);
    dut.v99(ch_v99);
    dut.v100(ch_v100);
    dut.v101(ch_v101);
    dut.v102(ch_v102);
    dut.v103(ch_v103);
    SC_THREAD(src_v94); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(src_v95); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(src_v96); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(src_v97); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(src_v98); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(src_v99); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(src_v100); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(src_v101); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(snk_v102); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
    SC_THREAD(snk_v103); sensitive << clk.posedge_event(); async_reset_signal_is(rst, false);
  }
  int _snk_done = 0;  // stream sinks drained; the last one sc_stop()s
  void src_v94() {
    ch_v94.ResetWrite();
    wait();
    { std::ifstream _f("input0.data"); long long _v; for (int f = 0; f < 64; ++f) { _f >> _v; ch_v94.Push((ac_int<1, false>)_v); } }
  }
  void src_v95() {
    ch_v95.ResetWrite();
    wait();
    { std::ifstream _f("input1.data"); long long _v; for (int f = 0; f < 64; ++f) { _f >> _v; ch_v95.Push((ac_int<1, false>)_v); } }
  }
  void src_v96() {
    ch_v96.ResetWrite();
    wait();
    { std::ifstream _f("input2.data"); long long _v; for (int f = 0; f < 64; ++f) { _f >> _v; ch_v96.Push((ac_int<3, false>)_v); } }
  }
  void src_v97() {
    ch_v97.ResetWrite();
    wait();
    { std::ifstream _f("input3.data"); long long _v; for (int f = 0; f < 64; ++f) { _f >> _v; ch_v97.Push((ac_int<64, false>)_v); } }
  }
  void src_v98() {
    ch_v98.ResetWrite();
    wait();
    { std::ifstream _f("input4.data"); long long _v; for (int f = 0; f < 64; ++f) { _f >> _v; ch_v98.Push((ac_int<1, false>)_v); } }
  }
  void src_v99() {
    ch_v99.ResetWrite();
    wait();
    { std::ifstream _f("input5.data"); long long _v; for (int f = 0; f < 64; ++f) { _f >> _v; ch_v99.Push((ac_int<1, false>)_v); } }
  }
  void src_v100() {
    ch_v100.ResetWrite();
    wait();
    { std::ifstream _f("input6.data"); long long _v; for (int f = 0; f < 64; ++f) { _f >> _v; ch_v100.Push((ac_int<3, false>)_v); } }
  }
  void src_v101() {
    ch_v101.ResetWrite();
    wait();
    { std::ifstream _f("input7.data"); long long _v; for (int f = 0; f < 64; ++f) { _f >> _v; ch_v101.Push((ac_int<64, false>)_v); } }
  }
  void snk_v102() {
    ch_v102.ResetRead();
    wait();
    { std::ofstream _f("output0.data"); for (int f = 0; f < 64; ++f) _f << (long long)(ch_v102.Pop()) << "\n"; }
    if (++_snk_done == 2) sc_stop();
  }
  void snk_v103() {
    ch_v103.ResetRead();
    wait();
    { std::ofstream _f("output1.data"); for (int f = 0; f < 64; ++f) _f << (long long)(ch_v103.Pop()) << "\n"; }
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
