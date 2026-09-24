# zhang-21 run of `ppa_tinytpu` (T=4, MAXDIM=16) — 2026-09-24

**FAIL, in Catapult's C++ front end, 7 s after start.** No scheduling, no simulation,
no power. Input `../` at `main` `3aadfbf`, `run.tcl` unmodified, Catapult
2024.2/1130128 with LIC-14 checked out. `check_ppa.py` reports FAIL on arms 1, 3 and 4
(`check_ppa.out`), which is the correct verdict. Catapult exited 2.

## Cause: two defects in the Catapult C++ emitter

`go analyze` stops with 100 errors in `kernel.cpp`. They are all one of two kinds:

1. **Bit slices are emitted in the Vitis idiom**, 88 × CRD-20:
   ```cpp
   ap_int<64> v16_tmp = v15;     // "identifier ap_int is undefined"
   v16 = v16_tmp(15, 0);          // and then "v16_tmp is undefined"
   ```
   Catapult has no `ap_int`. `kernel.cpp` has 286 lines using `ap_int`/`ap_uint`.
2. **Wider-than-64-bit `ac_int` is converted to `int` implicitly**, 12 × CRD-413:
   ```cpp
   ac_int<65, true> v35 = v34 + 8;
   int v36 = v35;                 // no suitable conversion from ac_int<65,true> to int
   ```

`ppa_mac16` has neither bit slices nor arithmetic wider than 64 bits, which is why it
passed.

## A fix that compiles

`acint_fix_probe.cpp`, compiled with Catapult's g++ and `$MGC_HOME/shared/include`,
checks both replacements and their values (`def0 48 15`, exit 0):

- `ac_int<64, false> t = x; y = t.slc<16>(0);` in place of `ap_int<64> t = x; y = t(15, 0);`
  (`slc<hi-lo+1>(lo)`, with the temporary's signedness taken from the source).
- `int v = wide.to_int();` in place of `int v = wide;`

Built with `-DSHOW_DEFECT2`, the same probe reproduces defect 2 in plain g++
(`cannot convert 'ac_int<65, true>' to 'int'`). **Both defects are therefore
catchable without a licence**: compile `kernel.cpp` against the open-source hlslibs
`ac_types` on the emitting host. That catches this class of failure before a handoff.
