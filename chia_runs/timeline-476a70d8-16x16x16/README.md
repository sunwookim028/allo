# Per-process timeline of the landed design (main @ 476a70d8), 16x16x16

Measured 2026-09-19 for the CHIA run's seed hypotheses, with main's own
profiling scripts, unchanged:

    cd examples/accelerator/tinytpu_vitis/impact && source env.sh
    TPU_SHAPES=16x16x16 TPU_PRJ=$PWD/runs/landed/isa_sweep.prj \
        python pyrun.py cosim_variant.py base runs/landed
    ./profile.sh runs/landed

cosim: 686 cycles, `TB 16x16x16 mismatches = 0 / 256` (also under profiling).
Times are dataflow-monitor cycles; the region starts at 47 and the window
ends at 688. `timeline.txt`: per process start/done and run/starve/block
counts. `rle.txt`: run-length traces, R=running S=starved B=blocked,
`<state><length>@<start>`.

What it shows (read off the two files, not estimated):

* `dma_ld` runs 211 cycles back to back (68-279); no PE computes before 285
  (`pe_0_0` starved 211 cycles, 74-285). About 237 of the 641 cycles pass
  before the first MAC.
* `vru` runs 74 cycles (152-226), is BLOCKED 60 (226-286) until the array
  starts, then runs 201 (286-487), in step with `pe_0_0` (285-486).
* `pe_3_3` finishes at 624; `accu` runs 312-653 in four ~80-cycle bursts,
  each followed by a 5-cycle BLOCK (on `dma_st`); `dma_st` alternates 38
  cycles running with 47 starved and finishes at 688, 35 after `accu`.
* `sequencer` is BLOCKED 134 of its cycles.
