# Can Catapult synthesize a memory shared by several concurrent processes? — 2026-09-24

The question comes from MiniTPU's vector register file: one array, read and written by
three concurrently running units. The answer below comes from reading the Catapult
2024.2 install on zhang-21. Nothing was synthesized for it.

## Short answer

**Yes, in two different ways. Only one of them arbitrates.**

| | `ac_shared` + `ac_sync` (Catapult native) | MatchLib `ArbitratedScratchpad` (bundled) |
|---|---|---|
| Arbitrates? | **No.** Each process gets a dedicated physical memory port. Correctness depends on `ac_sync` handing off ownership (ping-pong style) | **Yes.** A round-robin arbiter per bank, with queued requests |
| Clients | ≤ number of physical ports. 1R1W takes one writer and one reader. A single-port RAM "Catapult will refuse to schedule" | `NumInputs` template parameter (the shipped test uses 4) |
| Contention | None by construction. Clients must not touch the same region concurrently | Each bank serves one winner per cycle. Losers wait in `LenInputBuffer` queues, and `input_ready[i]` backpressures the client |
| Declaration | `static ac_shared<T[N]> mem; static ac_sync sync;`, passed to each `CCS_BLOCK` / `hls_design` subblock | `ArbitratedScratchpad<T, CapacityBytes, NumInputs, NumBanks, LenInputBuffer> sp; sp.load_store(req, rsp, ready);` inside **one** process that owns it. Clients reach it through request/response channels |
| Binding | `directive set /Top/mem.d:rsc -MAP_TO_MODULE ccs_sample_mem.ccs_ram_sync_1R1W`, or `ram_nangate-45nm-separate_beh.RAM_separateRW`, or `…-dualport_beh.RAM_dualRW`. Port per process: `-MEMORY_USE_PORT n` | Its banks are ordinary `mem_array`s, bound like any array |
| Flow | C++ hierarchical (`hls_design` blocks) | SystemC / Connections (MatchLib); `hls/unittests/ArbitratedScratchpadTop` |
| Known limits | "true dual-port RTL memories are not handled properly, resulting in bad logic" (SharedSyncMemory app note, Known Limitations). The port-reservation note shows dual-port working through `MEMORY_USE_PORT` | The bundled MatchLib is a 2020 snapshot (`0d89d5f`). `ArbitratedScratchpadDP` adds separate read/write ports, store-forwarding, and an SPRAM mode |

**For a three-client register file:** `ac_shared` cannot express it on a two-port
memory. The clients' accesses are not handed off in phases, and there is no arbiter.
`ArbitratedScratchpad` is the construct that expresses it. Its cost is a per-bank
arbiter and crossbar plus input queues. When two readers and a writer hit the same
bank every cycle, one of them is granted per cycle in round-robin order and the
others stall on `ready`. Throughput per bank is one access per cycle. Banking spreads
conflicts across banks.

## Where it is documented in the install

- `$MGC_HOME/shared/examples/methodology/SharedSyncMemory/SharedSynchronizedMemory.pdf`
  — `ac_shared`/`ac_sync` ping-pong, six directive scripts.
- `$MGC_HOME/shared/examples/methodology/SharedMemPortReservation/SharedMemPortReservation.pdf`
  — `MEMORY_USE_PORT`, 1R1W and dual-port mapping, the single-port scheduling failure,
  and 1R1W wrapper generation.
- `$MGC_HOME/shared/pkgs/matchlib/cmod/include/ArbitratedScratchpad.h`,
  `ArbitratedScratchpadDP.h`, `Arbiter.h` (`Roundrobin` | `Static`), and
  `arbitrated_crossbar.h`. Unit tests are in `cmod/unittests/ArbitratedScratchpad{,DP}Top`
  and `hls/unittests/ArbitratedScratchpad{,DP}Top/go_hls.tcl`.
- `mutually_exclusive_multiportmem_sharing/` is **not** relevant. It shares ports
  between mutually exclusive operations inside one process.

## Not verified here

The MatchLib HLS unit test was attempted and stopped in `go analyze`: `nvhls_array.h`
needs Boost (`static_assert.hpp`). Catapult bundles only the preprocessor subset
(`shared/pkgs/boostpp/pp`), and the host has no system Boost. Running it needs a Boost
include tree as `BOOST_HOME`, plus
`make hls RUN_SCVERIFY=0 VCS_HOME=<any>` from a writable copy of `shared/pkgs/matchlib`.
Until that runs, "`ArbitratedScratchpad` synthesizes in this install" is documented,
not measured.
