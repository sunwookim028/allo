/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// allo_fifo.h -- C translation of the ring-buffer FIFO protocol that the Allo
// dataflow *simulator* generates for `!allo.stream<T, depth>`.
//
// WHY THIS FILE EXISTS
// -------------------
// On `target="simulator"` every Allo stream becomes a single-producer /
// single-consumer ring buffer in shared memory, and every `stream.put` /
// `stream.get` becomes a fixed sequence of loads, stores and OpenMP fences
// (see `allo/backend/simulator.py`, `_process_function_streams`). A
// hand-written HLS IP cannot emit that sequence -- it just calls
// `hls::stream<T>::read()` / `::write()`. The functions below *are* that
// sequence, so the shim `hls_stream.h` next to this file can forward the IP's
// calls onto an Allo ring buffer with no copying and no second FIFO.
//
// Every step below mirrors one operation emitted by simulator.py. The cited
// line numbers refer to that file at the time this was written; the ordering
// they encode is what actually matters and is explained in
// `docs/IP_STREAM_SIM_SHIM.md`.
//
// MEMORY MODEL
// ------------
// The producer thread only ever writes `tail`; the consumer thread only ever
// writes `head`. Neither index is written by two threads, so no lock is
// needed -- but both are *read* by the other thread, so the reads/writes must
// be atomic (no torn or cached values) and ordered with respect to the data
// slot they protect:
//
//   * producer: store data slot, THEN publish the new tail. If the publish
//     were visible first, the consumer could read a slot that has not been
//     written yet.
//   * consumer: read data slot, THEN publish the new head. If the publish
//     were visible first, the producer could overwrite the slot while it is
//     still being read.
//
// simulator.py gets this ordering from `omp.flush` (a whole-memory fence for
// the calling thread) plus an `omp.critical` index store that
// `convert_critical_write_to_atomic_write` turns into `omp.atomic.write`. Here
// we use sequentially-consistent atomics, which imply the same fence, plus
// explicit `__atomic_thread_fence` calls at the points where simulator.py
// emits a bare `omp.flush`.

#ifndef ALLO_FIFO_H
#define ALLO_FIFO_H

#include <cstdint>
#include <unistd.h> // usleep

/// A *view* of one Allo ring buffer. Nothing here is owned: `data`, `head` and
/// `tail` point into buffers that the JIT-compiled Allo module allocated
/// (`memref.alloc` in `top`), so the IP and the Allo kernels share one FIFO.
///
/// `cap` is the number of slots, which the simulator sets to `depth + 1`: one
/// slot is always left empty so that `head == tail` unambiguously means empty
/// and `(tail + 1) % cap == head` unambiguously means full.
template <typename T> struct AlloFifo {
  T *data;      ///< cap slots of storage
  int32_t cap;  ///< depth + 1 (size of the data memref)
  int32_t *head; ///< read index; only the consumer advances it
  int32_t *tail; ///< write index; only the producer advances it
};

/// Whole-memory fence for this thread == `omp.flush` with no operands.
static inline void allo_fifo_fence() { __atomic_thread_fence(__ATOMIC_SEQ_CST); }

/// Read an index published by the *other* thread.
static inline int32_t allo_fifo_load_index(const int32_t *p) {
  return __atomic_load_n(p, __ATOMIC_SEQ_CST);
}

/// Publish an index to the *other* thread. Matches the `omp.atomic.write` that
/// `convert_critical_write_to_atomic_write` produces on the MLIR side.
static inline void allo_fifo_store_index(int32_t *p, int32_t v) {
  __atomic_store_n(p, v, __ATOMIC_SEQ_CST);
}

/// Body of the spin loop: back off so a blocked kernel does not starve the
/// peers it is waiting for. simulator.py emits `omp.taskyield` followed by
/// `usleep(1)` here. The `taskyield` is only compiled in when this file is
/// built with OpenMP enabled (see `IPModule.generate_stream_sim_wrapper`): it
/// is a scheduling hint with no effect on correctness, whereas mixing a second
/// OpenMP runtime into the process would be unsafe. `usleep(1)` is what
/// actually releases the core.
static inline void allo_fifo_backoff() {
#if defined(_OPENMP)
#pragma omp taskyield
#endif
  usleep(1);
}

/// Next index after `i`, wrapping at `cap`. Mirrors `arith.addi` + `arith.remui`
/// on the i32 index.
static inline int32_t allo_fifo_next(int32_t i, int32_t cap) {
  return (int32_t)(((uint32_t)i + 1u) % (uint32_t)cap);
}

/// True if the FIFO holds no element. Mirrors `allo.stream_empty`
/// (simulator.py: flush, load head, load tail, compare equal).
template <typename T> inline bool allo_fifo_empty(const AlloFifo<T> &f) {
  allo_fifo_fence();
  return allo_fifo_load_index(f.head) == allo_fifo_load_index(f.tail);
}

/// True if the FIFO cannot accept another element. Mirrors `allo.stream_full`
/// (simulator.py: flush, load tail, next = (tail+1)%cap, load head, compare).
template <typename T> inline bool allo_fifo_full(const AlloFifo<T> &f) {
  allo_fifo_fence();
  int32_t next = allo_fifo_next(allo_fifo_load_index(f.tail), f.cap);
  return next == allo_fifo_load_index(f.head);
}

/// Number of elements currently buffered (`hls::stream::size()`).
template <typename T> inline int32_t allo_fifo_size(const AlloFifo<T> &f) {
  allo_fifo_fence();
  int32_t head = allo_fifo_load_index(f.head);
  int32_t tail = allo_fifo_load_index(f.tail);
  int32_t diff = tail - head;
  return diff >= 0 ? diff : diff + f.cap;
}

/// Non-blocking put. Mirrors `allo.stream_try_put`: test once, and on success
/// run exactly the blocking-put body.
template <typename T>
inline bool allo_fifo_try_put(const AlloFifo<T> &f, const T &value) {
  allo_fifo_fence();
  int32_t tail = allo_fifo_load_index(f.tail);
  int32_t next = allo_fifo_next(tail, f.cap);
  if (allo_fifo_load_index(f.head) == next)
    return false; // FULL
  f.data[tail] = value;
  allo_fifo_store_index(f.tail, next);
  allo_fifo_fence();
  return true;
}

/// Non-blocking get. Mirrors `allo.stream_try_get`.
template <typename T> inline bool allo_fifo_try_get(const AlloFifo<T> &f, T *out) {
  allo_fifo_fence();
  int32_t head = allo_fifo_load_index(f.head);
  if (head == allo_fifo_load_index(f.tail))
    return false; // EMPTY
  *out = f.data[head];
  allo_fifo_store_index(f.head, allo_fifo_next(head, f.cap));
  allo_fifo_fence();
  return true;
}

/// Blocking put == `allo.stream_put`. Step-for-step translation of the
/// StreamPutOp lowering in simulator.py (lines 869-999).
template <typename T> inline void allo_fifo_put(const AlloFifo<T> &f, const T &value) {
  // (1) flush before reading the indices, so we see the consumer's latest
  //     `head` rather than a value cached in this thread (line 869).
  allo_fifo_fence();
  // (2) load our own write index once: we are the only producer, so nobody
  //     else can move it (line 870).
  int32_t tail = allo_fifo_load_index(f.tail);
  // (3) the slot we would publish next (lines 874-880).
  int32_t next = allo_fifo_next(tail, f.cap);
  // (4) spin while FULL. The loop condition re-reads `head` every iteration
  //     after a fence (lines 895-919); the loop body backs off (907-912).
  while (allo_fifo_load_index(f.head) == next)
    allo_fifo_backoff();
  // (5) write the payload into the slot *before* publishing it (line 988).
  f.data[tail] = value;
  // (6) publish the new tail atomically -- this is the point at which the
  //     element becomes visible to the consumer (lines 995-997).
  allo_fifo_store_index(f.tail, next);
  // (7) trailing flush (line 999).
  allo_fifo_fence();
}

/// Blocking get == `allo.stream_get`. Mirror of the put above; simulator.py
/// lines 884-894 and 1000-1089.
template <typename T> inline T allo_fifo_get(const AlloFifo<T> &f) {
  // (1) flush, then (2) load our own read index once (we are the only
  //     consumer) and (3) compute its successor (lines 884-894).
  allo_fifo_fence();
  int32_t head = allo_fifo_load_index(f.head);
  int32_t next = allo_fifo_next(head, f.cap);
  // (4) spin while EMPTY, re-reading `tail` after a fence each iteration
  //     (lines 902, 1002-1006).
  while (head == allo_fifo_load_index(f.tail))
    allo_fifo_backoff();
  // (5) read the payload *before* releasing the slot (line 1062).
  T value = f.data[head];
  // (6) publish the new head atomically; only now may the producer reuse the
  //     slot (lines 1087-1089).
  allo_fifo_store_index(f.head, next);
  // The MLIR get path ends at the atomic write (it emits no trailing
  // `omp.flush`); the seq-cst store already implies a full fence, so this is
  // the same barrier, written explicitly for symmetry with put.
  allo_fifo_fence();
  return value;
}

#endif // ALLO_FIFO_H
