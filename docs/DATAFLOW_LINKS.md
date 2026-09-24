# Kernel Interconnects: `Wire`, `Channel`, `Stream` (with non-blocking access)

An Allo dataflow design is a `@df.region` whose `@df.kernel`s talk to each other over
**links**. Previously, the only link was a buffered FIFO (`Stream`). This document covers the
three link types now available and the **non-blocking** (`try_*`) access added alongside them.

All three are declared as *type-annotated locals* inside the region body and used with
`put`/`get` from the kernels. They differ only in what sits between producer and consumer:

```python
from allo.ir.types import int32, Stream, Channel, valid_ready

@df.region()
def top(A: int32[M, N], B: int32[M, N]):
    fifo: Stream[int32, 4]              # buffered FIFO, depth 4
    chan: Channel[int32, valid_ready]   # handshake, no buffer

    @df.kernel(mapping=[1], args=[A])
    def producer(a: int32[M, N]):
        for i, j in allo.grid(M, N):
            fifo.put(a[i, j]); chan.put(a[i, j])

    @df.kernel(mapping=[1], args=[B])
    def consumer(b: int32[M, N]):
        for i, j in allo.grid(M, N):
            b[i, j] = fifo.get() + chan.get()
```

Both `Stream` and `Channel` are *ordered* links (blocking `put`/`get`), so this design is
deterministic and verifies bit-exact. A `Wire` is deliberately **not** used here; 
see "Using a `Wire` safely" below for one idiom that works.

(Runnable version: `docs/dataflow_links_examples.py` — contains this design and the
`Wire`-sideband design below, with a `main()` that builds and checks both.)

---

## The three link types

| Type | Declaration | Buffering | Handshake | Cost | Use when |
|---|---|---|---|---|---|
| **`Wire`** | `Wire[T]` | none | none | cheapest | producer/consumer are cycle-locked; you want a raw combinational connection |
| **`Channel`** | `Channel[T, proto]` | none | `valid_only` or `valid_ready` | cheap | one flow-controlled value at a time, no storage needed |
| **`Stream`** | `Stream[T, depth]` | depth ≥ 1 | credit/handshake | most | real elastic buffering / rate decoupling between kernels |

**`Wire`** — an unbuffered, zero-latency point-to-point connection. No handshake, no
back-pressure: whatever the producer drives this cycle is what the consumer reads.
**It has zero storage *and* zero alignment**. Only
sound as a sideband on an ordered link (see "Using a `Wire` safely" below).

**`Channel`** — a point-to-point link with an explicit handshake but no buffering. The protocol
is chosen at declaration:
- `Channel[T, valid_only]` — the producer asserts *valid* + data; there is no *ready* back from
  the consumer, so a value can be dropped if the consumer isn't looking. Cheapest handshake.
- `Channel[T, valid_ready]` — full valid/ready handshake with back-pressure; the transfer only
  happens when both sides agree. This is the safe default when you need flow control but not
  storage.

**`Stream`** — the buffered FIFO (the original link type). `Stream[T, depth]` gives `depth`
elements of elastic storage so producer and consumer can run at different instantaneous rates.

---

## Blocking access: `put` / `get`

The default `put`/`get` **block**: `get` waits until data is available, `put` waits until there
is room. This is the simplest model and what most designs use. (A `Wire` never blocks — it has
no notion of empty/full — so its `put`/`get` are just drive/sample.)

---

## Non-blocking access: `try_put` / `try_get` (and `empty` / `full`)

Non-blocking access lets a kernel *attempt* a transfer and keep going if it can't complete —
essential for designs that must service several links in one step (arbiters, routers, anything
that would otherwise deadlock waiting on one input).

Supported on **`Stream`** and **`Channel`** (a `Wire` has none — it never blocks):

```python
ok = S.try_put(value)      # -> bool: True if the value was accepted, False if full
val, ok = S.try_get()      # -> (data, bool): ok False means nothing was read (val is undefined)
```

`empty()` / `full()` (Stream only) query occupancy without transferring:

```python
if not S.full():  S.put(x)
if not S.empty(): y = S.get()
```

A typical non-blocking spin (poll until it succeeds, without blocking the whole thread):

```python
while not S.try_put(i * 10):   # producer: retry until accepted
    pass

ok: int1 = 0
while ok == 0:                 # consumer: retry until something arrives
    val, ok = S.try_get()
```

(Runnable: `examples/nb_stream_rtl.py`, `examples/nb_nondeterminism.py`.)

### ⚠️ Always consume the `ok` flag

The result of a `try_*` op **must be used**. A result-producing op with no uses is silently
deleted by dead-code elimination (`MemRefDCE`), so a `try_put` whose returned `ok` is never read
can be dropped entirely — the transfer just disappears with no error. Assign it to a live
variable (`ok: int1 = S.try_put(x)`) or branch on it. This is the single most common
non-blocking bug.

### Non-determinism

With non-blocking access, arrival order under contention is genuinely non-deterministic (two
producers racing for one consumer). Checkers over such designs must compare **order-tolerantly**
(multiset / per-flow order), not positionally. See `examples/nb_nondeterminism.py`.

---

## Arrays of links

Any link type can be declared as an array, so a kernel grid can be wired with one line:

```python
S: Stream[int32, 4][P]        # P independent depth-4 FIFOs; index as S[k]
sys_e: Stream[SYS_W, D][M, N] # a 2-D mesh of streams
```

Index the array to pick a link (`S[0].try_put(...)`). Arrays work for `Wire` and `Channel` too.

---

## Backend support

| | Simulator (`target="simulator"`) | Vitis HLS (`target="vitis_hls"`) | SystemC/Catapult (`target="systemc"`) |
|---|---|---|---|
| `Stream` + `put`/`get` | ✅ | ✅ | ✅ |
| `Stream` `try_*`, `empty`/`full` | ✅ | ✅ | ✅ |
| `Wire` | combinational only | ✅ | ✅ (`sc_signal`) |
| `Channel` (valid_only / valid_ready) | — | ✅ | ✅ (`Connections::Combinational`) |

For how each link lowers to SystemC (Wire→`sc_signal`, Channel→`Connections::Combinational` or
`_dat`/`_vld`, Stream→`Connections::Fifo`/`AlloFifoC`), and the RTL-level caveats, see
`docs/SYSTEMC_BACKEND.md`.

---

## Using a `Wire` safely: the sideband idiom

A `Wire` on its own is **not** a modularity boundary. Given zero storage *and* zero handshake,
a standalone Wire between two independently-paced kernels reads garbage — the consumer free-runs
and samples the signal's stale/initial value (proven by controlled experiment: a `mul→acc` split
over a `Wire` reads all-zeros, while the same split over a `Stream` or `Channel` passes — it was
the *handshake*, not the buffer, that made those correct).

A Wire *is* sound when a **blocking link supplies the ordering** and the Wire rides alongside it
as a sideband. The discipline: drive the wire **before** the ordering push, read it **after** the
ordering get.

```python
link: Stream[UInt(16), 2]      # payload: buffered, ordered
side: Wire[UInt(8)]            # derived tag: rides the wire, 0 bits stored

@df.kernel(mapping=[1], args=[A])
def gen(a: int32[N]):
    for i in range(N):
        d:  UInt(16) = a[i] & 0xFFFF
        tg: UInt(8)  = a[i] & 3
        side.put(tg)           # (1) drive the wire FIRST ...
        link.put(d)            # (2) ... THEN push. The push is the barrier.

@df.kernel(mapping=[1], args=[C])
def proc(c: int32[N]):
    for i in range(N):
        dw: UInt(16) = link.get()   # (3) BLOCKS until gen pushed
        tw: UInt(8)  = side.get()   # (4) safe: wire was driven before that push
        d: int32 = dw; tg: int32 = tw
        c[i] = d * (tg + 1)         # needs both, same instant
```

`proc` cannot reach the wire read until `gen`'s push has completed, and `gen` drove the wire
before pushing — so the value is current **by construction**. Swap either pair of
lines and the design is racy; nothing in the type system enforces this.

(Full runnable comparison: `docs/dataflow_links_examples.py`; the underlying analysis of why a
standalone `Wire` reads garbage is in `docs/noc/FINDINGS_wire_channel.md`.)

---

## Choosing a link (rule of thumb)

1. Need buffering / rate decoupling → **`Stream`**.
2. Need flow control but no storage → **`Channel[T, valid_ready]`**.
3. One-way, drop-tolerant, cheapest handshake → **`Channel[T, valid_only]`**.
4. Provably cycle-locked and want the raw wire → **`Wire`** (careful: no alignment).
