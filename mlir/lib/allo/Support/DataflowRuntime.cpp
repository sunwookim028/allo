/*
 * Copyright Allo authors. All Rights Reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

// CPU dataflow simulator runtime.
//
// Each PE of a dataflow/systolic kernel runs as a marl fiber on a small pool of
// worker threads. Streams are bounded MPMC FIFOs whose blocking put/get suspend
// the calling FIBER, not the OS thread, so the authored depth never affects
// correctness and no PE starves a thread. The lowering emits
// allo_df_open/spawn/join/close around the PE calls and allo_sim_stream_*
// inside them.

#include "marl/conditionvariable.h"
#include "marl/mutex.h"
#include "marl/scheduler.h"
#include "marl/waitgroup.h"

#include <cassert>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#if defined(_WIN32)
#define ALLO_RUNTIME_EXPORT __declspec(dllexport)
#else
#define ALLO_RUNTIME_EXPORT __attribute__((visibility("default")))
#endif

namespace {

// A single bounded MPMC FIFO lane with inline fixed-size slots. A full put or
// empty get parks the current fiber and yields the worker thread.
struct Lane {
  marl::mutex mutex;
  marl::ConditionVariable notEmpty;
  marl::ConditionVariable notFull;
  std::vector<uint8_t> ring; // depth * itemBytes bytes
  int64_t head = 0;
  int64_t tail = 0;
  int64_t count = 0;
};

struct Stream {
  int64_t depth;
  int64_t itemBytes;
  std::vector<std::unique_ptr<Lane>> lanes;
};

Stream *asStream(uint64_t handle) {
  auto *stream = reinterpret_cast<Stream *>(handle);
  assert(stream && "invalid stream handle");
  return stream;
}

Lane &getLane(Stream *stream, int64_t lane) {
  assert(0 <= lane && lane < static_cast<int64_t>(stream->lanes.size()) &&
         "stream lane out of bounds");
  return *stream->lanes[lane];
}

void writeBytes(Stream *stream, int64_t laneId, const void *data) {
  assert(data && "invalid stream write payload");
  Lane &lane = getLane(stream, laneId);

  marl::lock lock(lane.mutex);
  lane.notFull.wait(lock, [&] { return lane.count < stream->depth; });
  std::memcpy(&lane.ring[lane.tail * stream->itemBytes], data,
              stream->itemBytes);
  lane.tail = (lane.tail + 1) % stream->depth;
  ++lane.count;
  lane.notEmpty.notify_one();
}

void readBytes(Stream *stream, int64_t laneId, void *data) {
  assert(data && "invalid stream read payload");
  Lane &lane = getLane(stream, laneId);

  marl::lock lock(lane.mutex);
  lane.notEmpty.wait(lock, [&] { return lane.count > 0; });
  std::memcpy(data, &lane.ring[lane.head * stream->itemBytes],
              stream->itemBytes);
  lane.head = (lane.head + 1) % stream->depth;
  --lane.count;
  lane.notFull.notify_one();
}

// A marl scheduler plus a WaitGroup tracking this region's live PE fibers.
// `owning` is false for a nested container, which reuses the enclosing region's
// already-bound scheduler, so close() must not tear it down.
struct DataflowScheduler {
  marl::Scheduler *scheduler;
  marl::WaitGroup pending;
  bool owning;
};

} // namespace

// ---- stream ABI (called from PE fibers; blocking is fiber-aware) ------------

extern "C" ALLO_RUNTIME_EXPORT uint64_t
allo_sim_stream_create(int64_t lanes, int64_t depth, int64_t itemBytes) {
  assert(lanes > 0 && "stream must have at least one lane");
  assert(depth > 0 && "stream depth must be positive");
  assert(itemBytes > 0 && "stream payload size must be positive");

  auto stream = std::make_unique<Stream>();
  stream->depth = depth;
  stream->itemBytes = itemBytes;
  stream->lanes.reserve(lanes);
  for (int64_t i = 0; i < lanes; ++i) {
    auto lane = std::make_unique<Lane>();
    lane->ring.resize(static_cast<size_t>(depth) * itemBytes);
    stream->lanes.push_back(std::move(lane));
  }
  return reinterpret_cast<uint64_t>(stream.release());
}

extern "C" ALLO_RUNTIME_EXPORT void
allo_sim_stream_write(uint64_t handle, int64_t lane, uint64_t value) {
  Stream *stream = asStream(handle);
  assert(stream->itemBytes <= static_cast<int64_t>(sizeof(value)) &&
         "scalar stream payload is too wide");
  writeBytes(stream, lane, &value);
}

// Preload one initial token into `lane`, seeding a feedback cycle so it does
// not deadlock. The bound (and every lane's ring) grows by one, so seeding
// never consumes the declared steady-state depth and never blocks. Called once
// per token at construction, before any PE fiber exists, so the ring is empty
// and the resize preserves the contiguous (head == 0) layout.
extern "C" ALLO_RUNTIME_EXPORT void
allo_sim_stream_seed(uint64_t handle, int64_t lane, uint64_t value) {
  Stream *stream = asStream(handle);
  assert(stream->itemBytes <= static_cast<int64_t>(sizeof(value)) &&
         "scalar stream payload is too wide");
  Lane &l = getLane(stream, lane);
  marl::lock lock(l.mutex);
  ++stream->depth;
  for (auto &lp : stream->lanes)
    lp->ring.resize(static_cast<size_t>(stream->depth) * stream->itemBytes);
  std::memcpy(&l.ring[l.tail * stream->itemBytes], &value, stream->itemBytes);
  l.tail = (l.tail + 1) % stream->depth;
  ++l.count;
  l.notEmpty.notify_one();
}

extern "C" ALLO_RUNTIME_EXPORT uint64_t allo_sim_stream_read(uint64_t handle,
                                                             int64_t lane) {
  Stream *stream = asStream(handle);
  assert(stream->itemBytes <= static_cast<int64_t>(sizeof(uint64_t)) &&
         "scalar stream payload is too wide");
  uint64_t value = 0;
  readBytes(stream, lane, &value);
  return value;
}

extern "C" ALLO_RUNTIME_EXPORT void
allo_sim_stream_write_mem(uint64_t handle, int64_t lane, uint64_t ptr) {
  writeBytes(asStream(handle), lane, reinterpret_cast<const void *>(ptr));
}

extern "C" ALLO_RUNTIME_EXPORT void
allo_sim_stream_read_mem(uint64_t handle, int64_t lane, uint64_t ptr) {
  readBytes(asStream(handle), lane, reinterpret_cast<void *>(ptr));
}

extern "C" ALLO_RUNTIME_EXPORT void allo_sim_stream_destroy(uint64_t handle) {
  delete asStream(handle);
}

// ---- dataflow scheduler ABI (called from the launcher thread) ---------------

// Open a dataflow region. At the top level this creates a marl scheduler and
// binds it to the calling thread; `numWorkers <= 0` requests one worker per
// logical core. A nested container reuses the scheduler already bound to its
// thread. Each region gets its own WaitGroup, so join blocks only for the
// fibers spawned at this level.
extern "C" ALLO_RUNTIME_EXPORT void *allo_df_open(int64_t numWorkers) {
  if (marl::Scheduler *cur = marl::Scheduler::get())
    return new DataflowScheduler{cur, marl::WaitGroup{}, /*owning=*/false};
  marl::Scheduler::Config cfg =
      numWorkers > 0 ? marl::Scheduler::Config().setWorkerThreadCount(
                           static_cast<int>(numWorkers))
                     : marl::Scheduler::Config::allCores();
  auto *scheduler = new marl::Scheduler(cfg);
  scheduler->bind();
  return new DataflowScheduler{scheduler, marl::WaitGroup{}, /*owning=*/true};
}

// Launch `fn(ctx)` as a fiber. The shared `ctx` (the PE operands) stays valid
// because allo_df_join keeps the launcher frame alive until every fiber exits.
extern "C" ALLO_RUNTIME_EXPORT void
allo_df_spawn(void *handle, void (*fn)(void *), void *ctx) {
  auto *df = static_cast<DataflowScheduler *>(handle);
  df->pending.add(1);
  marl::WaitGroup pending = df->pending;
  marl::schedule([fn, ctx, pending] {
    fn(ctx);
    pending.done();
  });
}

extern "C" ALLO_RUNTIME_EXPORT void allo_df_join(void *handle) {
  static_cast<DataflowScheduler *>(handle)->pending.wait();
}

extern "C" ALLO_RUNTIME_EXPORT void allo_df_close(void *handle) {
  auto *df = static_cast<DataflowScheduler *>(handle);
  // A nested region reuses the enclosing scheduler; leave it bound and alive.
  if (df->owning) {
    df->scheduler->unbind();
    delete df->scheduler;
  }
  delete df;
}
