# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lucas-Kanade optical flow: gradients, separable smoothing, then a 2x2 solve."""

import numpy as np

from allo.lang import f32, kernel

from ..spec import Benchmark

HEIGHT, WIDTH = 32, 32

GRAD_WEIGHTS = np.array([1, -8, 0, 8, -1], np.float32)
GRAD_FILTER = np.array(
    [0.0755, 0.133, 0.1869, 0.2903, 0.1869, 0.133, 0.0755], np.float32
)
TENSOR_FILTER = np.array([0.3243, 0.3513, 0.3243], np.float32)


def build():
    @kernel
    def gradient_xy_calc(
        frame: f32[HEIGHT, WIDTH],
        gradient_x: f32[HEIGHT, WIDTH],
        gradient_y: f32[HEIGHT, WIDTH],
    ):
        weights: f32[5] = GRAD_WEIGHTS
        for r in range(HEIGHT + 2):
            for c in range(WIDTH + 2):
                x_grad: f32 = 0.0
                y_grad: f32 = 0.0
                if r >= 4 and r < HEIGHT and c >= 4 and c < WIDTH:
                    for i in range(5):
                        x_grad += frame[r - 2, c - i] * weights[4 - i]
                        y_grad += frame[r - i, c - 2] * weights[4 - i]
                    gradient_x[r - 2, c - 2] = x_grad / 12.0
                    gradient_y[r - 2, c - 2] = y_grad / 12.0
                elif r >= 2 and c >= 2:
                    gradient_x[r - 2, c - 2] = 0.0
                    gradient_y[r - 2, c - 2] = 0.0

    @kernel
    def gradient_z_calc(
        frame0: f32[HEIGHT, WIDTH],
        frame1: f32[HEIGHT, WIDTH],
        frame2: f32[HEIGHT, WIDTH],
        frame3: f32[HEIGHT, WIDTH],
        frame4: f32[HEIGHT, WIDTH],
        gradient_z: f32[HEIGHT, WIDTH],
    ):
        weights: f32[5] = GRAD_WEIGHTS
        for rz in range(HEIGHT):
            for cz in range(WIDTH):
                acc_z: f32 = (
                    frame0[rz, cz] * weights[0]
                    + frame1[rz, cz] * weights[1]
                    + frame2[rz, cz] * weights[2]
                    + frame3[rz, cz] * weights[3]
                    + frame4[rz, cz] * weights[4]
                )
                gradient_z[rz, cz] = acc_z / 12.0

    @kernel
    def gradient_weight_y(
        gradient_x: f32[HEIGHT, WIDTH],
        gradient_y: f32[HEIGHT, WIDTH],
        gradient_z: f32[HEIGHT, WIDTH],
        filt_grad: f32[HEIGHT, WIDTH, 3],
    ):
        gfilter: f32[7] = GRAD_FILTER
        for ry in range(HEIGHT + 3):
            for cy in range(WIDTH):
                ax: f32 = 0.0
                ay: f32 = 0.0
                az: f32 = 0.0
                if ry >= 6 and ry < HEIGHT:
                    for i in range(7):
                        ax += gradient_x[ry - i, cy] * gfilter[i]
                        ay += gradient_y[ry - i, cy] * gfilter[i]
                        az += gradient_z[ry - i, cy] * gfilter[i]
                    filt_grad[ry - 3, cy, 0] = ax
                    filt_grad[ry - 3, cy, 1] = ay
                    filt_grad[ry - 3, cy, 2] = az
                elif ry >= 3:
                    filt_grad[ry - 3, cy, 0] = 0.0
                    filt_grad[ry - 3, cy, 1] = 0.0
                    filt_grad[ry - 3, cy, 2] = 0.0

    @kernel
    def gradient_weight_x(
        y_filt: f32[HEIGHT, WIDTH, 3], filt_grad: f32[HEIGHT, WIDTH, 3]
    ):
        gfilter: f32[7] = GRAD_FILTER
        for rx in range(HEIGHT):
            for cx in range(WIDTH + 3):
                for k in range(3):
                    acc: f32 = 0.0
                    if cx >= 6 and cx < WIDTH:
                        for i in range(7):
                            acc += y_filt[rx, cx - i, k] * gfilter[i]
                        filt_grad[rx, cx - 3, k] = acc
                    elif cx >= 3:
                        filt_grad[rx, cx - 3, k] = 0.0

    @kernel
    def outer_product(
        gradient: f32[HEIGHT, WIDTH, 3], out_product: f32[HEIGHT, WIDTH, 6]
    ):
        for ro in range(HEIGHT):
            for co in range(WIDTH):
                gx: f32 = gradient[ro, co, 0]
                gy: f32 = gradient[ro, co, 1]
                gz: f32 = gradient[ro, co, 2]
                out_product[ro, co, 0] = gx * gx
                out_product[ro, co, 1] = gy * gy
                out_product[ro, co, 2] = gz * gz
                out_product[ro, co, 3] = gx * gy
                out_product[ro, co, 4] = gx * gz
                out_product[ro, co, 5] = gy * gz

    @kernel
    def tensor_weight_y(outer: f32[HEIGHT, WIDTH, 6], tensor_y: f32[HEIGHT, WIDTH, 6]):
        tfilter: f32[3] = TENSOR_FILTER
        for rty in range(HEIGHT + 1):
            for cty in range(WIDTH):
                for k in range(6):
                    acc: f32 = 0.0
                    if rty >= 2 and rty < HEIGHT:
                        for i in range(3):
                            acc += outer[rty - i, cty, k] * tfilter[i]
                    if rty >= 1:
                        tensor_y[rty - 1, cty, k] = acc

    @kernel
    def tensor_weight_x(tensor_y: f32[HEIGHT, WIDTH, 6], tensor: f32[HEIGHT, WIDTH, 6]):
        tfilter: f32[3] = TENSOR_FILTER
        for rtx in range(HEIGHT):
            for ctx in range(WIDTH + 1):
                for k in range(6):
                    acc: f32 = 0.0
                    if ctx >= 2 and ctx < WIDTH:
                        for i in range(3):
                            acc += tensor_y[rtx, ctx - i, k] * tfilter[i]
                    if ctx >= 1:
                        tensor[rtx, ctx - 1, k] = acc

    @kernel
    def flow_calc(tensors: f32[HEIGHT, WIDTH, 6], output: f32[HEIGHT, WIDTH, 2]):
        for rf in range(HEIGHT):
            for cf in range(WIDTH):
                if rf >= 2 and rf < HEIGHT - 2 and cf >= 2 and cf < WIDTH - 2:
                    denom: f32 = (
                        tensors[rf, cf, 0] * tensors[rf, cf, 1]
                        - tensors[rf, cf, 3] * tensors[rf, cf, 3]
                    )
                    output[rf, cf, 0] = (
                        tensors[rf, cf, 5] * tensors[rf, cf, 3]
                        - tensors[rf, cf, 4] * tensors[rf, cf, 1]
                    ) / denom
                    output[rf, cf, 1] = (
                        tensors[rf, cf, 4] * tensors[rf, cf, 3]
                        - tensors[rf, cf, 5] * tensors[rf, cf, 0]
                    ) / denom
                else:
                    output[rf, cf, 0] = 0.0
                    output[rf, cf, 1] = 0.0

    @kernel
    def optical_flow(
        frame0: f32[HEIGHT, WIDTH],
        frame1: f32[HEIGHT, WIDTH],
        frame2: f32[HEIGHT, WIDTH],
        frame3: f32[HEIGHT, WIDTH],
        frame4: f32[HEIGHT, WIDTH],
        outputs: f32[HEIGHT, WIDTH, 2],
    ):
        gradient_x: f32[HEIGHT, WIDTH] = 0.0
        gradient_y: f32[HEIGHT, WIDTH] = 0.0
        gradient_z: f32[HEIGHT, WIDTH] = 0.0
        y_filtered: f32[HEIGHT, WIDTH, 3] = 0.0
        filtered_gradient: f32[HEIGHT, WIDTH, 3] = 0.0
        out_product: f32[HEIGHT, WIDTH, 6] = 0.0
        tensor_y: f32[HEIGHT, WIDTH, 6] = 0.0
        tensor: f32[HEIGHT, WIDTH, 6] = 0.0

        gradient_xy_calc(frame2, gradient_x, gradient_y)
        gradient_z_calc(frame0, frame1, frame2, frame3, frame4, gradient_z)
        gradient_weight_y(gradient_x, gradient_y, gradient_z, y_filtered)
        gradient_weight_x(y_filtered, filtered_gradient)
        outer_product(filtered_gradient, out_product)
        tensor_weight_y(out_product, tensor_y)
        tensor_weight_x(tensor_y, tensor)
        flow_calc(tensor, outputs)

    return {
        "top": optical_flow,
        "gradient_xy_calc": gradient_xy_calc,
        "gradient_z_calc": gradient_z_calc,
        "gradient_weight_y": gradient_weight_y,
        "gradient_weight_x": gradient_weight_x,
        "outer_product": outer_product,
        "tensor_weight_y": tensor_weight_y,
        "tensor_weight_x": tensor_weight_x,
        "flow_calc": flow_calc,
    }


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    stages = []
    for name, loop in (
        ("gradient_z_calc", "cz"),
        ("outer_product", "co"),
        ("tensor_weight_y", "cty"),
        ("tensor_weight_x", "ctx"),
        ("flow_calc", "cf"),
    ):
        s = parts[name].schedule()
        s.pipeline(s.loop(loop), ii=1)
        stages.append(s)
    top = parts["top"].schedule()
    top.compose(*stages)
    return top


def inputs(rng):
    # A smooth translating pattern, so the 2x2 structure tensor stays well
    # conditioned and the flow solve does not divide by a near-zero determinant.
    y, x = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float32)
    frames = []
    for t in range(5):
        base = np.sin(0.3 * (x - 0.5 * t)) * np.cos(0.25 * (y - 0.3 * t))
        noise = rng.uniform(-0.01, 0.01, (HEIGHT, WIDTH))
        frames.append((base + noise).astype(np.float32))
    return (*frames, np.zeros((HEIGHT, WIDTH, 2), np.float32))


def reference(frame0, frame1, frame2, frame3, frame4, outputs):
    frames = [frame0, frame1, frame2, frame3, frame4]
    gx = np.zeros((HEIGHT, WIDTH), np.float32)
    gy = np.zeros((HEIGHT, WIDTH), np.float32)
    for r in range(HEIGHT + 2):
        for c in range(WIDTH + 2):
            if 4 <= r < HEIGHT and 4 <= c < WIDTH:
                gx[r - 2, c - 2] = (
                    sum(frame2[r - 2, c - i] * GRAD_WEIGHTS[4 - i] for i in range(5))
                    / 12.0
                )
                gy[r - 2, c - 2] = (
                    sum(frame2[r - i, c - 2] * GRAD_WEIGHTS[4 - i] for i in range(5))
                    / 12.0
                )
    gz = sum(frames[i] * GRAD_WEIGHTS[i] for i in range(5)) / 12.0

    y_filt = np.zeros((HEIGHT, WIDTH, 3), np.float32)
    for r in range(6, HEIGHT):
        for k, g in enumerate((gx, gy, gz)):
            y_filt[r - 3, :, k] = sum(g[r - i, :] * GRAD_FILTER[i] for i in range(7))

    filt = np.zeros((HEIGHT, WIDTH, 3), np.float32)
    for c in range(6, WIDTH):
        filt[:, c - 3, :] = sum(y_filt[:, c - i, :] * GRAD_FILTER[i] for i in range(7))

    outer = np.stack(
        [
            filt[:, :, 0] ** 2,
            filt[:, :, 1] ** 2,
            filt[:, :, 2] ** 2,
            filt[:, :, 0] * filt[:, :, 1],
            filt[:, :, 0] * filt[:, :, 2],
            filt[:, :, 1] * filt[:, :, 2],
        ],
        axis=2,
    )

    ten_y = np.zeros((HEIGHT, WIDTH, 6), np.float32)
    for r in range(2, HEIGHT):
        ten_y[r - 1] = sum(outer[r - i] * TENSOR_FILTER[i] for i in range(3))
    ten = np.zeros((HEIGHT, WIDTH, 6), np.float32)
    for c in range(2, WIDTH):
        ten[:, c - 1] = sum(ten_y[:, c - i] * TENSOR_FILTER[i] for i in range(3))

    out = np.zeros((HEIGHT, WIDTH, 2), np.float32)
    t = ten[2 : HEIGHT - 2, 2 : WIDTH - 2]
    denom = t[:, :, 0] * t[:, :, 1] - t[:, :, 3] ** 2
    out[2 : HEIGHT - 2, 2 : WIDTH - 2, 0] = (
        t[:, :, 5] * t[:, :, 3] - t[:, :, 4] * t[:, :, 1]
    ) / denom
    out[2 : HEIGHT - 2, 2 : WIDTH - 2, 1] = (
        t[:, :, 4] * t[:, :, 3] - t[:, :, 5] * t[:, :, 0]
    ) / denom
    return (out,)


BENCHMARK = Benchmark(
    suite="rosetta",
    name="optical_flow",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(5,),
    tolerance=(1e-2, 1e-2),
)
