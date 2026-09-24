# csim result — zhang-21, 2026-09-24

Catapult 2024.2/1130128 g++ and SystemC 2.3.3, `./csim.sh` in each directory
exactly as committed. `kernel.cpp` sha256 matches both `PROVENANCE.md` files.

| Side | exit | `out_e` = `output1.data[0:8]` | wall |
|---|---|---|---|
| `before_unsigned_bit_slice/` | 0 | `1 2 3 4 5 6 0 0` | 7.6 s |
| `after_unsigned_bit_slice/` | 0 | `1 2 3 4 5 6 0 0` | 7.3 s |

**Both sides pass the ramp golden, and all eight `output*.data` are byte-identical
between them.** On this workload the signed→unsigned change to the bit slices does
not change behaviour. Since it matches the Allo `UInt` types, it reads as a
correction, not a regression. It says nothing about workloads that drive a slice to
a value with its top bit set. The 1x1 passthrough does not exercise that.

**Failure path.** A third copy of `before/` with `input0.data` line 18 (ramp value 3,
cycle 17) changed to `9.0` gives `out_e = 1 2 9 4 5 6`, so the comparison would have
caught a corrupted token. A first attempt changed `input8.data` instead. That file
holds the valid flags, where any nonzero value still means valid, so the output did
not change. That shows which file carries data, not that the check is blind.
