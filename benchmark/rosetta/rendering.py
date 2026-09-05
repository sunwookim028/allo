# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Triangle rasterization: project, bound, test each pixel, z-cull, then shade."""

import numpy as np

from allo.lang import i32, kernel

from ..spec import Benchmark

NUM_3D_TRI = 24
MAX_X, MAX_Y = 32, 32
MAX_FRAGMENT = 256
BACKGROUND_Z = 255
COLOR = 100


def build():
    @kernel
    def projection(tri3d: i32[NUM_3D_TRI, 9], idx: i32, angle: i32, tri2d: i32[7]):
        if angle == 0:
            tri2d[0] = tri3d[idx, 0]
            tri2d[1] = tri3d[idx, 1]
            tri2d[2] = tri3d[idx, 3]
            tri2d[3] = tri3d[idx, 4]
            tri2d[4] = tri3d[idx, 6]
            tri2d[5] = tri3d[idx, 7]
            tri2d[6] = tri3d[idx, 2] // 3 + tri3d[idx, 5] // 3 + tri3d[idx, 8] // 3
        elif angle == 1:
            tri2d[0] = tri3d[idx, 0]
            tri2d[1] = tri3d[idx, 2]
            tri2d[2] = tri3d[idx, 3]
            tri2d[3] = tri3d[idx, 5]
            tri2d[4] = tri3d[idx, 6]
            tri2d[5] = tri3d[idx, 8]
            tri2d[6] = tri3d[idx, 1] // 3 + tri3d[idx, 4] // 3 + tri3d[idx, 7] // 3
        else:
            tri2d[0] = tri3d[idx, 2]
            tri2d[1] = tri3d[idx, 1]
            tri2d[2] = tri3d[idx, 5]
            tri2d[3] = tri3d[idx, 4]
            tri2d[4] = tri3d[idx, 8]
            tri2d[5] = tri3d[idx, 7]
            tri2d[6] = tri3d[idx, 0] // 3 + tri3d[idx, 3] // 3 + tri3d[idx, 6] // 3

    @kernel
    def rasterization1(tri2d: i32[7], max_min: i32[5], max_index: i32[1]) -> i32:
        cw: i32 = (tri2d[4] - tri2d[0]) * (tri2d[3] - tri2d[1]) - (
            tri2d[5] - tri2d[1]
        ) * (tri2d[2] - tri2d[0])
        flag: i32 = 0
        if cw == 0:
            flag = 1
        else:
            # The reference takes the triangle BY VALUE, so its clockwise swap is
            # local to the bounding box and never reaches the pixel test below.
            x0: i32 = tri2d[0]
            y0: i32 = tri2d[1]
            x1: i32 = tri2d[2]
            y1: i32 = tri2d[3]
            if cw < 0:
                x0 = tri2d[2]
                y0 = tri2d[3]
                x1 = tri2d[0]
                y1 = tri2d[1]
            max_min[0] = min(min(x0, x1), tri2d[4])
            max_min[1] = max(max(x0, x1), tri2d[4])
            max_min[2] = min(min(y0, y1), tri2d[5])
            max_min[3] = max(max(y0, y1), tri2d[5])
            max_min[4] = max_min[1] - max_min[0]
            max_index[0] = (max_min[1] - max_min[0]) * (max_min[3] - max_min[2])
        return flag

    @kernel
    def rasterization2(
        flag: i32,
        max_min: i32[5],
        max_index: i32[1],
        tri2d: i32[7],
        fragment: i32[MAX_FRAGMENT, 4],
    ) -> i32:
        size: i32 = 0
        if flag == 0:
            for k in range(max_index[0]):
                x: i32 = max_min[0] + k % max_min[4]
                y: i32 = max_min[2] + k // max_min[4]
                pi0: i32 = (x - tri2d[0]) * (tri2d[3] - tri2d[1]) - (y - tri2d[1]) * (
                    tri2d[2] - tri2d[0]
                )
                pi1: i32 = (x - tri2d[2]) * (tri2d[5] - tri2d[3]) - (y - tri2d[3]) * (
                    tri2d[4] - tri2d[2]
                )
                pi2: i32 = (x - tri2d[4]) * (tri2d[1] - tri2d[5]) - (y - tri2d[5]) * (
                    tri2d[0] - tri2d[4]
                )
                if pi0 >= 0 and pi1 >= 0 and pi2 >= 0 and size < MAX_FRAGMENT:
                    fragment[size, 0] = x
                    fragment[size, 1] = y
                    fragment[size, 2] = tri2d[6]
                    fragment[size, 3] = COLOR
                    size += 1
        return size

    @kernel
    def zculling(
        counter: i32,
        fragment: i32[MAX_FRAGMENT, 4],
        size: i32,
        pixels: i32[MAX_FRAGMENT, 3],
        z_buffer: i32[MAX_X, MAX_Y],
    ) -> i32:
        if counter == 0:
            for zi in range(MAX_X):
                for zj in range(MAX_Y):
                    z_buffer[zi, zj] = BACKGROUND_Z
        pixel_cntr: i32 = 0
        for n in range(size):
            fx: i32 = fragment[n, 0]
            fy: i32 = fragment[n, 1]
            fz: i32 = fragment[n, 2]
            if fz < z_buffer[fy, fx]:
                pixels[pixel_cntr, 0] = fx
                pixels[pixel_cntr, 1] = fy
                pixels[pixel_cntr, 2] = fragment[n, 3]
                pixel_cntr += 1
                z_buffer[fy, fx] = fz
        return pixel_cntr

    @kernel
    def coloring_fb(
        counter: i32,
        size_pixels: i32,
        pixels: i32[MAX_FRAGMENT, 3],
        frame_buffer: i32[MAX_X, MAX_Y],
    ):
        if counter == 0:
            for ci in range(MAX_X):
                for cj in range(MAX_Y):
                    frame_buffer[ci, cj] = 0
        for p in range(size_pixels):
            frame_buffer[pixels[p, 0], pixels[p, 1]] = pixels[p, 2]

    @kernel
    def rendering(tri3d: i32[NUM_3D_TRI, 9], output: i32[MAX_X, MAX_Y]):
        tri2d: i32[7] = 0
        max_min: i32[5] = 0
        max_index: i32[1] = 0
        fragment: i32[MAX_FRAGMENT, 4] = 0
        pixels: i32[MAX_FRAGMENT, 3] = 0
        z_buffer: i32[MAX_X, MAX_Y] = 0
        for i in range(NUM_3D_TRI):
            projection(tri3d, i, 0, tri2d)
            flag: i32 = rasterization1(tri2d, max_min, max_index)
            size_fragment: i32 = rasterization2(
                flag, max_min, max_index, tri2d, fragment
            )
            size_pixels: i32 = zculling(i, fragment, size_fragment, pixels, z_buffer)
            coloring_fb(i, size_pixels, pixels, output)

    return {
        "top": rendering,
        "projection": projection,
        "rasterization1": rasterization1,
        "rasterization2": rasterization2,
        "zculling": zculling,
        "coloring_fb": coloring_fb,
    }


def _none(parts):
    return parts["top"].schedule()


def _v1(parts):
    rast2 = parts["rasterization2"].schedule()
    rast2.unroll(rast2.loop("k"), factor=2)
    top = parts["top"].schedule()
    top.compose(rast2)
    return top


def inputs(rng):
    tri3d = np.zeros((NUM_3D_TRI, 9), np.int32)
    for t in range(NUM_3D_TRI):
        ox = int(rng.integers(0, MAX_X - 8))
        oy = int(rng.integers(0, MAX_Y - 8))
        for v in range(3):
            tri3d[t, 3 * v + 0] = ox + int(rng.integers(0, 8))
            tri3d[t, 3 * v + 1] = oy + int(rng.integers(0, 8))
            tri3d[t, 3 * v + 2] = int(rng.integers(0, 240))
    return tri3d, np.zeros((MAX_X, MAX_Y), np.int32)


def reference(tri3d, output):
    frame = np.zeros((MAX_X, MAX_Y), np.int32)
    z_buffer = np.full((MAX_X, MAX_Y), BACKGROUND_Z, np.int32)
    for i in range(NUM_3D_TRI):
        x0, y0 = int(tri3d[i, 0]), int(tri3d[i, 1])
        x1, y1 = int(tri3d[i, 3]), int(tri3d[i, 4])
        x2, y2 = int(tri3d[i, 6]), int(tri3d[i, 7])
        z = tri3d[i, 2] // 3 + tri3d[i, 5] // 3 + tri3d[i, 8] // 3
        cw = (x2 - x0) * (y1 - y0) - (y2 - y0) * (x1 - x0)
        if cw == 0:
            continue
        bx0, by0, bx1, by1 = (x1, y1, x0, y0) if cw < 0 else (x0, y0, x1, y1)
        min_x, max_x = min(bx0, bx1, x2), max(bx0, bx1, x2)
        min_y, max_y = min(by0, by1, y2), max(by0, by1, y2)
        span = max_x - min_x
        fragments = []
        for k in range((max_x - min_x) * (max_y - min_y)):
            x, y = min_x + k % span, min_y + k // span
            pi0 = (x - x0) * (y1 - y0) - (y - y0) * (x1 - x0)
            pi1 = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
            pi2 = (x - x2) * (y0 - y2) - (y - y2) * (x0 - x2)
            if pi0 >= 0 and pi1 >= 0 and pi2 >= 0 and len(fragments) < MAX_FRAGMENT:
                fragments.append((x, y, z))
        for x, y, fz in fragments:
            if fz < z_buffer[y, x]:
                z_buffer[y, x] = fz
                frame[x, y] = COLOR
    return (frame,)


BENCHMARK = Benchmark(
    suite="rosetta",
    name="rendering",
    build=build,
    schedules={"none": _none, "v1": _v1},
    inputs=inputs,
    reference=reference,
    outputs=(1,),
)
