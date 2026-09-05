# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""512-point FFT built from radix-8 stages with a shared-memory transpose between."""

import numpy as np

from allo.lang import f32, i32, index, kernel
from allo.operators import math as amath

from ..spec import Benchmark

N = 512
SMEM = 8 * 8 * 9


def build():
    @kernel
    def cmplx_M_x(a_x: f32, a_y: f32, b_x: f32, b_y: f32) -> f32:
        return a_x * b_x - a_y * b_y

    @kernel
    def cmplx_M_y(a_x: f32, a_y: f32, b_x: f32, b_y: f32) -> f32:
        return a_x * b_y + a_y * b_x

    @kernel
    def cmplx_mul_x(a_x: f32, a_y: f32, b_x: f32, b_y: f32) -> f32:
        return a_x * b_x - a_y * b_y

    @kernel
    def cmplx_mul_y(a_x: f32, a_y: f32, b_x: f32, b_y: f32) -> f32:
        return a_x * b_y + a_y * b_x

    @kernel
    def cmplx_add_x(a_x: f32, b_x: f32) -> f32:
        return a_x + b_x

    @kernel
    def cmplx_add_y(a_y: f32, b_y: f32) -> f32:
        return a_y + b_y

    @kernel
    def cmplx_sub_x(a_x: f32, b_x: f32) -> f32:
        return a_x - b_x

    @kernel
    def cmplx_sub_y(a_y: f32, b_y: f32) -> f32:
        return a_y - b_y

    @kernel
    def cm_fl_mul_x(a_x: f32, b: f32) -> f32:
        return b * a_x

    @kernel
    def cm_fl_mul_y(a_y: f32, b: f32) -> f32:
        return b * a_y

    @kernel
    def twiddles8(a_x: f32[8], a_y: f32[8], i: i32, n: i32):
        PI: f32 = 3.1415926535
        reversed8: i32[8] = [0, 4, 2, 6, 1, 5, 3, 7]
        for j in range(1, 8):
            phi: f32 = (-2 * PI * reversed8[j] / n) * i
            phi_x: f32 = amath.cos(phi)
            phi_y: f32 = amath.sin(phi)
            tw: f32 = a_x[j]
            a_x[j] = cmplx_M_x(a_x[j], a_y[j], phi_x, phi_y)
            a_y[j] = cmplx_M_y(tw, a_y[j], phi_x, phi_y)

    # The out-parameter is the port's one departure from the reference: the
    # original returns f32[4], which the RTL path does not lower.
    @kernel
    def FF2(a0_x: f32, a0_y: f32, a1_x: f32, a1_y: f32, d0: f32[4]):
        d0[0] = cmplx_add_x(a0_x, a1_x)
        d0[1] = cmplx_add_y(a0_y, a1_y)
        d0[2] = cmplx_sub_x(a0_x, a1_x)
        d0[3] = cmplx_sub_y(a0_y, a1_y)

    @kernel
    def FFT4_1(a_x: f32[8], a_y: f32[8]):
        exp_1_44_x: f32 = 0.0
        exp_1_44_y: f32 = -1.0

        t1: f32[4] = 0.0
        FF2(a_x[0], a_y[0], a_x[2], a_y[2], t1)
        a_x[0] = t1[0]
        a_y[0] = t1[1]
        a_x[2] = t1[2]
        a_y[2] = t1[3]

        t2: f32[4] = 0.0
        FF2(a_x[1], a_y[1], a_x[3], a_y[3], t2)
        a_x[1] = t2[0]
        a_y[1] = t2[1]
        a_x[3] = t2[2]
        a_y[3] = t2[3]

        s3: f32 = a_x[3]
        a_x[3] = a_x[3] * exp_1_44_x - a_y[3] * exp_1_44_y
        a_y[3] = s3 * exp_1_44_y - a_y[3] * exp_1_44_x

        t4: f32[4] = 0.0
        FF2(a_x[0], a_y[0], a_x[1], a_y[1], t4)
        a_x[0] = t4[0]
        a_y[0] = t4[1]
        a_x[1] = t4[2]
        a_y[1] = t4[3]

        t5: f32[4] = 0.0
        FF2(a_x[2], a_y[2], a_x[3], a_y[3], t5)
        a_x[2] = t5[0]
        a_y[2] = t5[1]
        a_x[3] = t5[2]
        a_y[3] = t5[3]

    @kernel
    def FFT4_2(a_x: f32[8], a_y: f32[8]):
        exp_1_44_x: f32 = 0.0
        exp_1_44_y: f32 = -1.0

        u1: f32[4] = 0.0
        FF2(a_x[4], a_y[4], a_x[6], a_y[6], u1)
        a_x[4] = u1[0]
        a_y[4] = u1[1]
        a_x[6] = u1[2]
        a_y[6] = u1[3]

        u2: f32[4] = 0.0
        FF2(a_x[5], a_y[5], a_x[7], a_y[7], u2)
        a_x[5] = u2[0]
        a_y[5] = u2[1]
        a_x[7] = u2[2]
        a_y[7] = u2[3]

        w3: f32 = a_x[7]
        a_x[7] = a_x[7] * exp_1_44_x - a_y[7] * exp_1_44_y
        a_y[7] = w3 * exp_1_44_y - a_y[7] * exp_1_44_x

        u4: f32[4] = 0.0
        FF2(a_x[4], a_y[4], a_x[5], a_y[5], u4)
        a_x[4] = u4[0]
        a_y[4] = u4[1]
        a_x[5] = u4[2]
        a_y[5] = u4[3]

        u5: f32[4] = 0.0
        FF2(a_x[6], a_y[6], a_x[7], a_y[7], u5)
        a_x[6] = u5[0]
        a_y[6] = u5[1]
        a_x[7] = u5[2]
        a_y[7] = u5[3]

    @kernel
    def FFT8(a_x: f32[8], a_y: f32[8]):
        M_SQRT1_2: f32 = 0.70710678118654752440
        exp_1_8_x: f32 = 1.0
        exp_1_8_y: f32 = -1.0
        exp_1_4_x: f32 = 0.0
        exp_1_4_y: f32 = -1.0
        exp_3_8_x: f32 = -1.0
        exp_3_8_y: f32 = -1.0

        v1: f32[4] = 0.0
        FF2(a_x[0], a_y[0], a_x[4], a_y[4], v1)
        a_x[0] = v1[0]
        a_y[0] = v1[1]
        a_x[4] = v1[2]
        a_y[4] = v1[3]

        v2: f32[4] = 0.0
        FF2(a_x[1], a_y[1], a_x[5], a_y[5], v2)
        a_x[1] = v2[0]
        a_y[1] = v2[1]
        a_x[5] = v2[2]
        a_y[5] = v2[3]

        v3: f32[4] = 0.0
        FF2(a_x[2], a_y[2], a_x[6], a_y[6], v3)
        a_x[2] = v3[0]
        a_y[2] = v3[1]
        a_x[6] = v3[2]
        a_y[6] = v3[3]

        v4: f32[4] = 0.0
        FF2(a_x[3], a_y[3], a_x[7], a_y[7], v4)
        a_x[3] = v4[0]
        a_y[3] = v4[1]
        a_x[7] = v4[2]
        a_y[7] = v4[3]

        m5: f32 = a_x[5]
        a_x[5] = cm_fl_mul_x(
            cmplx_mul_x(a_x[5], a_y[5], exp_1_8_x, exp_1_8_y), M_SQRT1_2
        )
        a_y[5] = cm_fl_mul_y(cmplx_mul_y(m5, a_y[5], exp_1_8_x, exp_1_8_y), M_SQRT1_2)

        m6: f32 = a_x[6]
        a_x[6] = cmplx_mul_x(a_x[6], a_y[6], exp_1_4_x, exp_1_4_y)
        a_y[6] = cmplx_mul_y(m6, a_y[6], exp_1_4_x, exp_1_4_y)

        m7: f32 = a_x[7]
        a_x[7] = cm_fl_mul_x(
            cmplx_mul_x(a_x[7], a_y[7], exp_3_8_x, exp_3_8_y), M_SQRT1_2
        )
        a_y[7] = cm_fl_mul_y(cmplx_mul_y(m7, a_y[7], exp_3_8_x, exp_3_8_y), M_SQRT1_2)

        FFT4_1(a_x, a_y)
        FFT4_2(a_x, a_y)

    @kernel
    def loady8(a_y: f32[8], x: f32[SMEM], offset: i32, sx: i32):
        a_y[0] = x[0 * sx + offset]
        a_y[1] = x[1 * sx + offset]
        a_y[2] = x[2 * sx + offset]
        a_y[3] = x[3 * sx + offset]
        a_y[4] = x[4 * sx + offset]
        a_y[5] = x[5 * sx + offset]
        a_y[6] = x[6 * sx + offset]
        a_y[7] = x[7 * sx + offset]

    @kernel
    def fft1D_512(work_x: f32[N], work_y: f32[N]):
        stride: i32 = 64
        counter: i32 = 0
        rev8: i32[8] = [0, 4, 2, 6, 1, 5, 3, 7]

        DATA_x: f32[64 * 8] = 0.0
        DATA_y: f32[64 * 8] = 0.0
        data_x: f32[8] = 0.0
        data_y: f32[8] = 0.0
        smem: f32[SMEM] = 0.0

        for t0 in range(64):
            for e0 in range(8):
                data_x[e0] = work_x[e0 * stride + t0]
                data_y[e0] = work_y[e0 * stride + t0]
            FFT8(data_x, data_y)
            twiddles8(data_x, data_y, counter, 512)
            for s0 in range(8):
                DATA_x[t0 * 8 + s0] = data_x[s0]
                DATA_y[t0 * 8 + s0] = data_y[s0]
            counter += 1

        sx: i32 = 66
        for t1 in range(64):
            t1i: i32 = t1
            hi1: index = t1i >> 3
            lo1: index = t1i & 7
            off1: i32 = hi1 * 8 + lo1
            smem[0 * sx + off1] = DATA_x[t1 * 8 + 0]
            smem[4 * sx + off1] = DATA_x[t1 * 8 + 1]
            smem[1 * sx + off1] = DATA_x[t1 * 8 + 4]
            smem[5 * sx + off1] = DATA_x[t1 * 8 + 5]
            smem[2 * sx + off1] = DATA_x[t1 * 8 + 2]
            smem[6 * sx + off1] = DATA_x[t1 * 8 + 3]
            smem[3 * sx + off1] = DATA_x[t1 * 8 + 6]
            smem[7 * sx + off1] = DATA_x[t1 * 8 + 7]

        sx = 8
        for t2 in range(64):
            t2i: i32 = t2
            hi2: index = t2i >> 3
            lo2: index = t2i & 7
            off2: i32 = lo2 * 66 + hi2
            DATA_x[t2 * 8 + 0] = smem[0 * sx + off2]
            DATA_x[t2 * 8 + 4] = smem[4 * sx + off2]
            DATA_x[t2 * 8 + 1] = smem[1 * sx + off2]
            DATA_x[t2 * 8 + 5] = smem[5 * sx + off2]
            DATA_x[t2 * 8 + 2] = smem[2 * sx + off2]
            DATA_x[t2 * 8 + 6] = smem[6 * sx + off2]
            DATA_x[t2 * 8 + 3] = smem[3 * sx + off2]
            DATA_x[t2 * 8 + 7] = smem[7 * sx + off2]

        sx = 66
        for t3 in range(64):
            t3i: i32 = t3
            hi3: index = t3i >> 3
            lo3: index = t3i & 7
            off3: i32 = hi3 * 8 + lo3
            smem[0 * sx + off3] = DATA_y[t3 * 8 + 0]
            smem[4 * sx + off3] = DATA_y[t3 * 8 + 1]
            smem[1 * sx + off3] = DATA_y[t3 * 8 + 4]
            smem[5 * sx + off3] = DATA_y[t3 * 8 + 5]
            smem[2 * sx + off3] = DATA_y[t3 * 8 + 2]
            smem[6 * sx + off3] = DATA_y[t3 * 8 + 3]
            smem[3 * sx + off3] = DATA_y[t3 * 8 + 6]
            smem[7 * sx + off3] = DATA_y[t3 * 8 + 7]

        for t4 in range(64):
            for e4 in range(8):
                data_y[e4] = DATA_y[t4 * 8 + e4]
            t4i: i32 = t4
            hi4: index = t4i >> 3
            lo4: index = t4i & 7
            off4: i32 = lo4 * 66 + hi4
            loady8(data_y, smem, off4, 8)
            for s4 in range(8):
                DATA_y[t4 * 8 + s4] = data_y[s4]

        for t5 in range(64):
            for e5 in range(8):
                data_x[e5] = DATA_x[t5 * 8 + e5]
                data_y[e5] = DATA_y[t5 * 8 + e5]
            FFT8(data_x, data_y)
            t5i: i32 = t5
            hi5: i32 = t5i >> 3
            twiddles8(data_x, data_y, hi5, 64)
            for s5 in range(8):
                DATA_x[t5 * 8 + s5] = data_x[s5]
                DATA_y[t5 * 8 + s5] = data_y[s5]

        sx = 72
        for t6 in range(64):
            t6i: i32 = t6
            hi6: index = t6i >> 3
            lo6: index = t6i & 7
            off6: i32 = hi6 * 8 + lo6
            smem[0 * sx + off6] = DATA_x[t6 * 8 + 0]
            smem[4 * sx + off6] = DATA_x[t6 * 8 + 1]
            smem[1 * sx + off6] = DATA_x[t6 * 8 + 4]
            smem[5 * sx + off6] = DATA_x[t6 * 8 + 5]
            smem[2 * sx + off6] = DATA_x[t6 * 8 + 2]
            smem[6 * sx + off6] = DATA_x[t6 * 8 + 3]
            smem[3 * sx + off6] = DATA_x[t6 * 8 + 6]
            smem[7 * sx + off6] = DATA_x[t6 * 8 + 7]

        sx = 8
        for t7 in range(64):
            t7i: i32 = t7
            hi7: index = t7i >> 3
            lo7: index = t7i & 7
            off7: i32 = hi7 * 72 + lo7
            DATA_x[t7 * 8 + 0] = smem[0 * sx + off7]
            DATA_x[t7 * 8 + 4] = smem[4 * sx + off7]
            DATA_x[t7 * 8 + 1] = smem[1 * sx + off7]
            DATA_x[t7 * 8 + 5] = smem[5 * sx + off7]
            DATA_x[t7 * 8 + 2] = smem[2 * sx + off7]
            DATA_x[t7 * 8 + 6] = smem[6 * sx + off7]
            DATA_x[t7 * 8 + 3] = smem[3 * sx + off7]
            DATA_x[t7 * 8 + 7] = smem[7 * sx + off7]

        sx = 72
        for t8 in range(64):
            t8i: i32 = t8
            hi8: index = t8i >> 3
            lo8: index = t8i & 7
            off8: i32 = hi8 * 8 + lo8
            smem[0 * sx + off8] = DATA_y[t8 * 8 + 0]
            smem[4 * sx + off8] = DATA_y[t8 * 8 + 1]
            smem[1 * sx + off8] = DATA_y[t8 * 8 + 4]
            smem[5 * sx + off8] = DATA_y[t8 * 8 + 5]
            smem[2 * sx + off8] = DATA_y[t8 * 8 + 2]
            smem[6 * sx + off8] = DATA_y[t8 * 8 + 3]
            smem[3 * sx + off8] = DATA_y[t8 * 8 + 6]
            smem[7 * sx + off8] = DATA_y[t8 * 8 + 7]

        for t9 in range(64):
            for e9 in range(8):
                data_y[e9] = DATA_y[t9 * 8 + e9]
            t9i: i32 = t9
            hi9: index = t9i >> 3
            lo9: index = t9i & 7
            off9: i32 = hi9 * 72 + lo9
            loady8(data_y, smem, off9, 8)
            for s9 in range(8):
                DATA_y[t9 * 8 + s9] = data_y[s9]

        for ta in range(64):
            for ea in range(8):
                data_y[ea] = DATA_y[ta * 8 + ea]
                data_x[ea] = DATA_x[ta * 8 + ea]
            FFT8(data_x, data_y)
            for sa in range(8):
                work_x[sa * stride + ta] = data_x[rev8[sa]]
                work_y[sa * stride + ta] = data_y[rev8[sa]]

    return {"top": fft1D_512, "FFT8": FFT8, "twiddles8": twiddles8}


def _none(parts):
    return parts["top"].schedule()


def inputs(rng):
    work_x = rng.uniform(0.01, 0.25, N).astype(np.float32)
    work_y = rng.uniform(0.01, 0.25, N).astype(np.float32)
    return work_x, work_y


def reference(work_x, work_y):
    out = np.fft.fft(work_x.astype(np.complex64) + 1j * work_y.astype(np.complex64))
    return out.real.astype(np.float32), out.imag.astype(np.float32)


BENCHMARK = Benchmark(
    suite="machsuite",
    name="fft1D_512",
    build=build,
    schedules={"none": _none},
    inputs=inputs,
    reference=reference,
    outputs=(0, 1),
    tolerance=(1e-2, 1e-2),
    skip={"none": "math.cos / math.sin have no RTL lowering on the default device"},
)
