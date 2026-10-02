<!--- Copyright Allo authors. All Rights Reserved. -->
<!--- SPDX-License-Identifier: Apache-2.0  -->

# IP integration

These examples show how to drop a **hand-written HLS C++ block** ("IP") into an
Allo dataflow design with `allo.IPModule`, when the IP's interface uses
`hls::stream<T>` ports rather than plain arrays.

Both examples build the same tiny instruction-driven vector machine — a
sequencer that decodes a four-instruction program and a datapath that adds two
vectors — and differ only in **which half is hand-written C++**:

| Example | Written in C++ | Written in Allo | Also shows |
|---|---|---|---|
| [`vadd_datapath_ip.py`](vadd_datapath_ip.py) | the datapath ([`vadd.cpp`](vadd.cpp)) | the sequencer | a stream-only IP interface |
| [`vadd_sequencer_ip.py`](vadd_sequencer_ip.py) | the sequencer ([`sequencer.cpp`](sequencer.cpp)) | the datapath | a **mixed** interface: 5 array ports + 4 stream ports |

## Running them

```bash
OMP_NUM_THREADS=8 python3 vadd_datapath_ip.py
OMP_NUM_THREADS=8 python3 vadd_sequencer_ip.py
```

Each runs the design on the CPU dataflow simulator and checks the result against
NumPy. If `vitis_hls` is on your `PATH`, it then also builds the HLS project;
otherwise that step is skipped and the example still passes.

`OMP_NUM_THREADS` must be at least the number of kernels in the region. The
simulator gives each `@df.kernel` its own thread, and an IP with stream ports
*blocks* on them — so if two kernels have to share a thread, the design
deadlocks rather than merely running slowly.

## What to notice

- **The IP always gets its own `@df.kernel`.** That is not cosmetic. A stream
  port is a blocking interface, so the IP has to run concurrently with whatever
  is feeding it. Calling it anywhere else is rejected with an error explaining
  this.
- **`input_idx` / `output_idx` are required for stream ports.** An
  `hls::stream<T>&` parameter looks exactly the same in C++ whether the IP reads
  or writes it, so the direction has to be declared in Python.
- **The same IP source serves both targets.** Under `target="simulator"` the
  IP is compiled against Allo's shim `hls::stream`, so its `read()`/`write()`
  drive the simulator's ring buffers directly; under `target="vitis_hls"` it is
  stitched together through Vitis's own `hls::stream`. The `.cpp` files here are
  ordinary HLS — nothing in them is Allo-specific.
- **Element types must match.** The IP writes Allo's buffer in place, so
  `hls::stream<int32_t>` has to be fed by a `Stream[int32, ...]`. A mismatch is
  a `TypeError`, not a silent conversion.

## Further reading

- [`docs/IP_STREAM_INTEGRATION.md`](../../docs/IP_STREAM_INTEGRATION.md) — how
  stream IPs are integrated for the `vitis_hls` / `vivado_hls` targets.
- [`docs/IP_STREAM_SIM_SHIM.md`](../../docs/IP_STREAM_SIM_SHIM.md) — how they
  run under the CPU dataflow simulator, including the FIFO protocol and the ABI.
- [`tests/ip_integration/`](../../tests/ip_integration/) — smaller, more focused
  cases, including IPs with plain array interfaces.

## RTL IP integration

See [RTLModule documentation](../../docs/RTL_MODULE.md) and
[`rtl_accumulator.py`](rtl_accumulator.py) for Verilator simulation and Vitis
black-box project generation with a supplied RTL wrapper.

[`rtl_adapter.py`](rtl_adapter.py) demonstrates automatic ready/valid-to-FIFO
wrapper generation, simulation of the generated wrapper, and Vitis project export.
