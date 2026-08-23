# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""LSD radix sort: histogram, prefix-scan and scatter, ping-ponging two buffers."""

import numpy as np

from allo.lang import i32, kernel

from ..spec import Benchmark

ELEMENTSPERBLOCK = 4
SIZE = 64
NUMOFBLOCKS = SIZE // ELEMENTSPERBLOCK
RADIXSIZE = 4
BUCKETSIZE = NUMOFBLOCKS * RADIXSIZE + 1
SCAN_BLOCK = 16
SCAN_RADIX = (BUCKETSIZE - 1) // SCAN_BLOCK


def build():
    @kernel
    def ss_sort(a: i32[SIZE]):
        b: i32[SIZE] = 0
        bucket: i32[BUCKETSIZE] = 0
        sm: i32[SCAN_RADIX] = 0
        bucket_indx: i32 = 0
        a_indx: i32 = 0
        valid_buffer: i32 = 0

        for exp in range(16):
            for i_init in range(BUCKETSIZE):
                bucket[i_init] = 0

            if valid_buffer == 0:
                for blockID in range(NUMOFBLOCKS):
                    for i_h in range(4):
                        a_indx = blockID * ELEMENTSPERBLOCK + i_h
                        bucket_indx = (
                            ((a[a_indx] >> (exp * 2)) & 0x3) * NUMOFBLOCKS + blockID + 1
                        )
                        bucket[bucket_indx] = bucket[bucket_indx] + 1
            else:
                for blockID2 in range(NUMOFBLOCKS):
                    for i_h2 in range(4):
                        a_indx = blockID2 * ELEMENTSPERBLOCK + i_h2
                        bucket_indx = (
                            ((b[a_indx] >> (exp * 2)) & 0x3) * NUMOFBLOCKS
                            + blockID2
                            + 1
                        )
                        bucket[bucket_indx] = bucket[bucket_indx] + 1

            for radixID in range(SCAN_RADIX):
                for i_ls in range(1, SCAN_BLOCK):
                    bucket_indx = radixID * SCAN_BLOCK + i_ls
                    bucket[bucket_indx] = bucket[bucket_indx] + bucket[bucket_indx - 1]

            sm[0] = 0
            for radixID_s in range(1, SCAN_RADIX):
                bucket_indx = radixID_s * SCAN_BLOCK - 1
                sm[radixID_s] = sm[radixID_s - 1] + bucket[bucket_indx]

            for radixID_l in range(SCAN_RADIX):
                for i_lss in range(SCAN_BLOCK):
                    bucket_indx = radixID_l * SCAN_BLOCK + i_lss
                    bucket[bucket_indx] = bucket[bucket_indx] + sm[radixID_l]

            if valid_buffer == 0:
                for blockID_u in range(NUMOFBLOCKS):
                    for i_u in range(4):
                        bucket_indx = (
                            (a[blockID_u * ELEMENTSPERBLOCK + i_u] >> (exp * 2)) & 0x3
                        ) * NUMOFBLOCKS + blockID_u
                        a_indx = blockID_u * ELEMENTSPERBLOCK + i_u
                        b[bucket[bucket_indx]] = a[a_indx]
                        bucket[bucket_indx] = bucket[bucket_indx] + 1
                valid_buffer = 1
            else:
                for blockID_u2 in range(NUMOFBLOCKS):
                    for i_u2 in range(4):
                        bucket_indx = (
                            (b[blockID_u2 * ELEMENTSPERBLOCK + i_u2] >> (exp * 2)) & 0x3
                        ) * NUMOFBLOCKS + blockID_u2
                        a_indx = blockID_u2 * ELEMENTSPERBLOCK + i_u2
                        a[bucket[bucket_indx]] = b[a_indx]
                        bucket[bucket_indx] = bucket[bucket_indx] + 1
                valid_buffer = 0

    return {"top": ss_sort}


def _none(parts):
    return parts["top"].schedule()


def inputs(rng):
    return (rng.integers(0, 100000, SIZE).astype(np.int32),)


def reference(a):
    return (np.sort(a).astype(np.int32),)


BENCHMARK = Benchmark(
    suite="machsuite",
    name="ss_sort",
    build=build,
    schedules={"none": _none},
    inputs=inputs,
    reference=reference,
    outputs=(0,),
)
