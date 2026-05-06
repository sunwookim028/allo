# Handoff: allo-tpu control plane redesign + shared runtime

## Goal

Replace allo-tpu's current streaming-bus control interface (one HLS kernel
invocation per 16-word chunk, host drives every word transfer) with an
industry-standard AXI-Lite + AXI-MM interface (one kernel invocation per
logical operation — H2D, D2H, PROG, RUN).  Design the register map so that
mini-tpu can converge to the same protocol when it reaches U280/XRT (P3.4),
enabling a shared `Backend` implementation in `aihw_runtime`.

The Allo prerequisite (scalar args → `s_axilite` in `@df.region()`) is
**done** — see `notes/PITFALLS_DATAFLOW_REGION.md` and `STATE.md`.

---

## Background decisions (don't re-derive)

**Why the current interface is wrong for hw:**
The current `tpu(ctrl[1], d_addr[1], inval[16], outval[16], size[1])` is
called once per 16-word BW-width chunk.  A 256-element matrix load issues
16 separate XRT `ap_start`/`ap_done` round-trips.  On real U280 hardware
this is the dominant latency source.

**The CUDA / XRT model:**
Data loading is decoupled from kernel launch.  The host allocates `pyxrt.bo`
buffers, syncs data into them (equivalent to `cudaMemcpy`), then launches the
kernel once passing bo handles as args.  XRT writes the bo device addresses
into AXI-Lite registers; the kernel's AXI-MM master bursts the data in a
single DMA transaction.  One XRT round-trip per operation, not per chunk.

**mini-tpu's existing register map (Ultra96/PYNQ):**
```
slv_reg0[3:0] = mode      (WR_DEVMEM=1, RD_DEVMEM=2, COMPUTE=3, WR_IRAM=4, ...)
slv_reg0[4]   = doorbell  (host writes 1; hardware auto-clears)
slv_reg1[0]   = ready     (read-only: all sub-FSMs idle)
slv_reg4      = d_addr    (device memory address)
slv_reg6      = dma_len   (transfer word count)
data path:  AXI-Stream slave (write) + AXI-Stream master (read), 256-bit
```
The doorbell + ready + mode/addr/len pattern is structurally identical to
HLS `ap_ctrl_hs` + s_axilite scalar args.  The only difference is the data
transport: AXI-Stream on PYNQ (PS DMA pushes), AXI-MM on U280 (PL kernel pulls).

---

## New hardware interface

### Target `tpu()` signature

```python
@df.region()
def tpu(
    ctrl:    int32,               # s_axilite: H2D=0, D2H=1, RUN=3, PROG=4
    d_addr:  int32,               # s_axilite: device memory address (spad or acc)
    n:       int32,               # s_axilite: word count (0 = skip transfer)
    dma_buf: int32[MAX_TRANSFER], # m_axi, bundle=gmem0: DRAM buffer (R/W per ctrl)
):
    spad: float32[SPAD_SIZE] @ Stateful = 0.0
    acc:  float32[ACC_SIZE]  @ Stateful = 0.0
    imem: int32[IMEM_SIZE * INSTR_W] @ Stateful = 0
    ...
```

`MAX_TRANSFER = SPAD_SIZE` (4096 words = 16 KB) is a sufficient upper bound;
`n` controls the actual burst length.

### Generated HLS port map

```
ap_ctrl_hs  0x00      ap_start (trigger) / ap_done (status)
s_axilite   0x10      ctrl
s_axilite   0x18      d_addr
s_axilite   0x20      n
m_axi       gmem0     dma_buf  (XRT passes pyxrt.bo device address here)
```

This is one m_axi port.  A single bundle is correct: only one logical transfer
happens per invocation (H2D *or* D2H *or* PROG — not simultaneously).
Separate bundles are only needed for concurrent DMA paths.

### Kernel behaviour per ctrl value

| ctrl        | action |
|-------------|--------|
| CTRL_H2D=0  | burst-read `dma_buf[0:n]` → `spad[d_addr:d_addr+n]` (bitcast int32→float32) |
| CTRL_D2H=1  | burst-write `acc[d_addr:d_addr+n]` → `dma_buf[0:n]` (bitcast float32→int32) |
| CTRL_RUN=3  | decode `imem` from `d_addr`, emit commands, execute; no DMA |
| CTRL_PROG=4 | burst-read `dma_buf[0:n*INSTR_W]` → `imem[d_addr*INSTR_W : ...]` |

Stateful arrays persist between invocations: loading weights once and running
many times still works — just don't re-issue CTRL_H2D for the weights.

---

## aihw_runtime changes

### Backend protocol (runtime.py)

Replace the current five-array `call()` with a four-arg signature that matches
the new HLS interface:

```python
class Backend(Protocol):
    def call(
        self,
        ctrl:   int,          # operation type
        d_addr: int,          # device memory address
        n:      int,          # word count
        data:   np.ndarray,   # full transfer buffer (arbitrary size, not BW-capped)
    ) -> np.ndarray | None:
        """Execute one TPU operation.

        For D2H: returns the result array (n int32 elements).
        For all others: returns None.
        """
        ...
```

### LLVMBackend

```python
def call(self, ctrl, d_addr, n, data):
    out = np.zeros(n, dtype=np.int32)
    self._module(
        np.int32(ctrl), np.int32(d_addr), np.int32(n),
        data.astype(np.int32), out,
    )
    return out if ctrl == CTRL_D2H else None
```

(The LLVM module signature changes to match the new region args.)

### XRTBackend

```python
def call(self, ctrl, d_addr, n, data):
    bo = pyxrt.bo(self._device, max(n,1)*4, pyxrt.bo.normal,
                  self._kernel.group_id(3))  # gmem0 = arg index 3
    if ctrl != CTRL_D2H and n > 0:
        np.frombuffer(bo.map(), np.int32)[:n] = data[:n]
        bo.sync(xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE, n*4, 0)
    run = self._kernel(ctrl, d_addr, n, bo)
    run.wait()
    if ctrl == CTRL_D2H and n > 0:
        bo.sync(xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE, n*4, 0)
        return np.frombuffer(bo.map(), np.int32)[:n].copy()
    return None
```

Persistent bo reuse (current pattern) can be added as an optimisation once
the baseline works — allocate a pool of max-size bos at init time.

### PynqBackend (mini-tpu / future shared)

```python
def call(self, ctrl, d_addr, n, data):
    # Write control registers
    self._mmio.write(REG_MODE,   ctrl)
    self._mmio.write(REG_D_ADDR, d_addr)
    self._mmio.write(REG_LEN,    n)
    # Fire doorbell
    self._mmio.write(REG_CTRL, self._mmio.read(REG_CTRL) | DOORBELL_BIT)
    # Stream data in (H2D / PROG)
    if ctrl in (CTRL_H2D, CTRL_PROG) and n > 0:
        self._dma.sendchannel.transfer(data[:n].astype(np.int32))
        self._dma.sendchannel.wait()
    # Poll ready
    while not (self._mmio.read(REG_STATUS) & READY_BIT):
        pass
    # Stream data out (D2H)
    if ctrl == CTRL_D2H and n > 0:
        buf = allocate(shape=(n,), dtype=np.int32)
        self._dma.recvchannel.transfer(buf)
        self._dma.recvchannel.wait()
        return np.array(buf)
    return None
```

### TPUDevice changes

Remove the BW-chunking loops in `_memcpy_h2d` and `_memcpy_d2h`.  Each
operation becomes one `_backend.call()`:

```python
def _memcpy_h2d(self, d_buf, h_array):
    flat = h_array.flatten().view(np.int32)
    self._backend.call(CTRL_H2D, d_buf.addr, flat.size, flat)

def _memcpy_d2h(self, d_buf):
    result = self._backend.call(CTRL_D2H, d_buf.addr, d_buf.size, self._empty)
    return result.reshape(d_buf.shape) if d_buf.shape else result

def launch(self, kernel, prog_addr=0):
    packed = self._pack_instructions(kernel.build())
    self._backend.call(CTRL_PROG, prog_addr, len(kernel.build()), packed)
    self._backend.call(CTRL_RUN,  prog_addr, 0, self._empty)
```

`BW` and `_empty_buf` can be removed from `aihw_runtime/config.py` and
`TPUDevice.__init__`.

---

## tpu.py changes

The `@df.region()` signature changes from the five-array streaming interface
to the four-arg interface above.  The `decoder` kernel body restructures to:

```python
@df.kernel(mapping=[1], args=[ctrl, d_addr, n, dma_buf])
def decoder(ctrl_in, d_addr_in, n_in, buf_in):
    if ctrl_in == CTRL_H2D:
        for i in range(n_in):
            spad[d_addr_in + i] = buf_in[i].bitcast()     # int32 → float32
    elif ctrl_in == CTRL_D2H:
        for i in range(n_in):
            buf_in[i] = acc[d_addr_in + i].bitcast()      # float32 → int32
    elif ctrl_in == CTRL_PROG:
        for i in range(n_in * INSTR_W):
            imem[d_addr_in * INSTR_W + i] = buf_in[i]
    elif ctrl_in == CTRL_RUN:
        # existing decode + emit loop (unchanged)
        ...
    # always emit IMEM_SIZE no-ops to mxu_driver to preserve rendezvous
    ...
```

The `mxu_driver` kernel is unchanged.  The `cmd_mxu` Stream between them
is unchanged.

### HLS codegen (generate_hls.py)

The manual pragma-patching in `generate_hls.py` can be simplified or
removed.  With Allo now generating `s_axilite` for scalar args and `m_axi`
for array args natively, only the `extern "C"` wrapping and the `#pragma HLS
dataflow` strip (vpp mode) remain relevant.

---

## mini-tpu convergence (P3.4)

When mini-tpu targets U280/XRT, its top-level module should expose the same
logical register map:

| Register | mini-tpu (current) | Proposed U280 target |
|----------|-------------------|----------------------|
| mode/ctrl | slv_reg0[3:0] | s_axilite `ctrl` |
| trigger | slv_reg0[4] doorbell | ap_start (HLS ap_ctrl_hs) |
| ready | slv_reg1[0] instr_ready | ap_done |
| d_addr | slv_reg4 | s_axilite `d_addr` |
| n / len | slv_reg6 | s_axilite `n` |
| data | AXI-Stream 256-bit | m_axi `dma_buf` |

Mode constants need alignment (mini-tpu uses 1-based; allo-tpu uses 0-based).
Pick one encoding and update both — the software `Backend` constants must
match.

The `PynqBackend` above (register map abstraction) and the future `XRTBackend`
for mini-tpu are then the same class with different transport.  Everything
above (`TPUDevice`, programs, tests) is shared.

---

## Open decisions

1. **Persistent bo pool in XRTBackend** — allocate max-size bos at `__init__`
   and reuse, or allocate per-call.  Per-call is simpler to start; pool is
   the optimisation once correctness is established.

2. **D2H write-back via `dma_buf`** — the HLS kernel writes to `dma_buf`
   for D2H.  This requires the `dma_buf` m_axi port to be both read and write
   capable (`bundle=gmem0` with read/write).  Vitis HLS supports this but
   verify the generated interface pragma (`m_axi` without `readonly`).

3. **CSimBackend (`libtpu.so`)** — ctypes call site changes from 5 array
   pointers to 4 args.  Update `_fn.argtypes` and the call site.

4. **`LLVMBackend` module signature** — the LLVM JIT module's calling
   convention must match the new region args.  Confirm with a clean
   `make sim` after the tpu.py change.

---

## Sequence

1. Update `tpu.py` — new region signature + decoder body (H2D/D2H/PROG/RUN as DMA loops)
2. Update `aihw_runtime/runtime.py` — new `Backend.call()` + all four concrete backends + `TPUDevice` cleanup
3. Update `aihw_runtime/config.py` — remove `BW`, keep `INSTR_W`, `M`, etc.
4. Simplify `hls/generate_hls.py` — remove manual pragma injection where Allo now handles it
5. Run `make sim` + full test suite; fix any LLVM sim call-site issues
6. Run `make sw_emu` to verify XRT path
7. Update `aihw_programs` if any program-level code references `BW` or the old call signature

---

## Key files

```
allo-tpu/tpu/tpu.py                    region signature + decoder kernel  (primary change)
aihw-runtime/src/aihw_runtime/runtime.py  Backend protocol + all backends + TPUDevice
aihw-runtime/src/aihw_runtime/config.py   remove BW
allo-tpu/hls/generate_hls.py           simplify pragma injection
allo-tpu/Makefile                       no changes expected
mini-tpu/tpu/src/system/tpu.sv          reference: existing register map (read-only)
```
