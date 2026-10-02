/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// hls_stream.h -- Allo's *shim* for Vitis HLS's `hls::stream<T>`, used only
// when an HLS IP with stream ports is compiled for the CPU dataflow simulator
// (`df.build(top, target="simulator")`).
//
// WHY
// ---
// On the FPGA path the IP and the Allo kernels share Vitis's own
// `hls::stream`, so they are automatically connected. On the CPU there is no
// such shared primitive: Allo's kernels are JIT-compiled from MLIR and talk to
// a ring buffer in memory (see `allo_fifo.h`), while the IP is a `.cpp` we hand
// to g++. Because Allo compiles the IP itself, it also controls which
// `hls_stream.h` the IP sees -- the directory holding this file is placed
// FIRST on the include path, so `#include <hls_stream.h>` inside the IP
// resolves here instead of to Vitis's header.
//
// The result: `A.read()` inside the *unmodified* IP body performs exactly the
// ring-buffer handshake that an Allo kernel's `s.get()` performs, on exactly
// the same buffer. Only the IP side is adapted; Allo's kernels are untouched.
//
// SCOPE
// -----
// This header is deliberately not a complete reimplementation of Vitis's
// `hls::stream`. It covers the interface operations an IP performs on a port:
// blocking `read`/`write`, non-blocking `read_nb`/`write_nb`, `empty`/`full`/
// `size`, and the `>>` / `<<` operators. It is compiled only by Allo's
// generated simulator wrapper, never by Vitis.

#ifndef ALLO_SIM_HLS_STREAM_H
#define ALLO_SIM_HLS_STREAM_H

#include "allo_fifo.h"

#include <cstddef>
#include <cstdlib>

// Capacity used for a stream the IP declares *internally* (i.e. one Allo did
// not pass in and therefore did not allocate). Vitis's C simulation treats
// such a stream as unbounded; here it is a ring buffer with this many slots.
#ifndef ALLO_SIM_LOCAL_STREAM_DEPTH
#define ALLO_SIM_LOCAL_STREAM_DEPTH 1024
#endif

namespace hls {

/// Stand-in for `hls::stream<T, DEPTH>`.
///
/// Two ways to construct one:
///  * `stream(AlloFifo<T> *)` -- Allo's generated wrapper uses this to bind the
///    port to a ring buffer owned by the JIT-compiled Allo module. No storage
///    is allocated and nothing is copied: reads and writes go straight to the
///    shared buffer.
///  * `stream()` / `stream(const char *)` -- an IP-internal stream. We
///    allocate our own ring buffer so the IP still compiles and runs, but note
///    that a blocking read on an empty *internal* stream can never be
///    satisfied (nothing else is producing into it) and will spin forever,
///    just as it would deadlock in HLS C simulation.
template <typename T, int DEPTH = 0> class stream {
public:
  stream() { allocate_owned(); }
  explicit stream(const char *) { allocate_owned(); }

  /// Bind to a ring buffer allocated by the Allo simulator.
  explicit stream(AlloFifo<T> *fifo) : fifo_(*fifo), owned_(nullptr) {}

  ~stream() {
    if (owned_ != nullptr)
      std::free(owned_);
  }

  // Vitis makes `hls::stream` non-copyable; so do we, both to match and
  // because a copy would silently duplicate a FIFO view.
  stream(const stream &) = delete;
  stream &operator=(const stream &) = delete;

  /// Blocking read: waits until an element is available.
  T read() { return allo_fifo_get(fifo_); }
  void read(T &value) { value = allo_fifo_get(fifo_); }

  /// Blocking write: waits until a slot is free.
  void write(const T &value) { allo_fifo_put(fifo_, value); }

  /// Non-blocking read: returns false and leaves `value` alone if empty.
  bool read_nb(T &value) { return allo_fifo_try_get(fifo_, &value); }

  /// Non-blocking write: returns false if the FIFO is full.
  bool write_nb(const T &value) { return allo_fifo_try_put(fifo_, value); }

  bool empty() const { return allo_fifo_empty(fifo_); }
  bool full() const { return allo_fifo_full(fifo_); }
  std::size_t size() const { return (std::size_t)allo_fifo_size(fifo_); }

  /// Escape hatch for the generated wrapper / advanced uses.
  AlloFifo<T> &fifo() { return fifo_; }

private:
  void allocate_owned() {
    int32_t cap = (DEPTH > 0 ? DEPTH : ALLO_SIM_LOCAL_STREAM_DEPTH) + 1;
    owned_ = (T *)std::calloc((std::size_t)cap, sizeof(T));
    owned_head_ = 0;
    owned_tail_ = 0;
    fifo_.data = owned_;
    fifo_.cap = cap;
    fifo_.head = &owned_head_;
    fifo_.tail = &owned_tail_;
  }

  AlloFifo<T> fifo_;
  T *owned_ = nullptr;      ///< non-null only for IP-internal streams
  int32_t owned_head_ = 0;
  int32_t owned_tail_ = 0;
};

// Vitis declares these as free functions, so an IP may write `A >> x;` or
// `C << y;`. Mirror that spelling (and its `void` result).
template <typename T, int DEPTH> void operator>>(stream<T, DEPTH> &is, T &value) {
  is.read(value);
}

template <typename T, int DEPTH>
void operator<<(stream<T, DEPTH> &os, const T &value) {
  os.write(value);
}

} // namespace hls

#endif // ALLO_SIM_HLS_STREAM_H
