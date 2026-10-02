"""vpu_bf16_add.sv @ b3ba0a4d, re-expressed as integer field manipulation in
RTLGen's frontend (kkkaishao/allo allo-rtlgen @ 13b55a63). Mirrors the .sv
statement by statement; the 17-bit leading-zero count is a 5-step priority
search instead of the .sv's for-loop with a found flag."""
import os
from allo import kernel
from allo.lang import i32, u16

N = int(os.environ.get("BF_N", "16"))


@kernel
def bf16_add_bits_loop(A: u16[N], B: u16[N], C: u16[N]):
    for k in range(N, name="k"):
        H16: i32 = 65536
        ONE: i32 = 1
        a: i32 = A[k]
        b: i32 = B[k]
        sa: i32 = (a >> 15) & 1
        sb: i32 = (b >> 15) & 1
        ea: i32 = (a >> 7) & 255
        eb: i32 = (b >> 7) & 255
        fa: i32 = a & 127
        fb: i32 = b & 127
        ha: i32 = H16 if ea != 0 else 0
        hb: i32 = H16 if eb != 0 else 0
        ma: i32 = (fa << 9) | ha
        mb: i32 = (fb << 9) | hb
        a_large: i32 = ONE if ((ea << 17) | ma) >= ((eb << 17) | mb) else 0
        ea1: i32 = 1 if ea == 0 else ea
        eb1: i32 = 1 if eb == 0 else eb
        sl: i32 = sa if a_large == 1 else sb
        el: i32 = ea1 if a_large == 1 else eb1
        es: i32 = eb1 if a_large == 1 else ea1
        ml: i32 = ma if a_large == 1 else mb
        ms: i32 = mb if a_large == 1 else ma
        ed: i32 = el - es
        sh: i32 = 10 if ed >= 10 else ed
        sm: i32 = ms >> sh
        mag: i32 = (ml + sm) if sa == sb else (ml - sm)
        er: i32 = el
        res: i32 = 0
        if (ea == 255 and fa != 0) or (eb == 255 and fb != 0) or (ea == 255 and eb == 255 and sa != sb):
            res = 0x7FC0
        elif ea == 255:
            res = (sa << 15) | 0x7F80
        elif eb == 255:
            res = (sb << 15) | 0x7F80
        elif (a & 0x7FFF) == 0:
            res = b
        elif (b & 0x7FFF) == 0:
            res = a
        elif ed >= 10:
            res = a if a_large == 1 else b
        elif mag == 0:
            res = 0
        else:
            if mag >= 131072:
                mag = (mag | ((mag & 1) << 1)) >> 1
                er = er + 1
            elif mag < 65536:
                lz: i32 = 17
                found: i32 = 0
                for j in range(17, name="j"):
                    if found == 0 and ((mag >> (16 - j)) & 1) == 1:
                        lz = j
                        found = 1
                mx: i32 = 16 if er > 17 else er - 1
                ns: i32 = lz if lz < mx else mx
                mag = mag << ns
                er = er - ns
            g: i32 = (mag >> 8) & 1
            r: i32 = (mag >> 7) & 1
            s: i32 = ONE if (mag & 127) != 0 else 0
            ru: i32 = g & (r | s | ((mag >> 9) & 1))
            rnd: i32 = ((mag >> 9) & 127) + ru
            if rnd >= 128:
                rnd = 0
                er = er + 1
            if er >= 255:
                res = (sl << 15) | 0x7F80
            elif er <= 1 and mag < 65536:
                res = (sl << 15) | rnd
            else:
                res = (sl << 15) | ((er & 255) << 7) | rnd
        C[k] = res
