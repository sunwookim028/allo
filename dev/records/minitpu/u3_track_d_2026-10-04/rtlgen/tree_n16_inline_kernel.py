from allo import kernel
from allo.lang import i32, u8, u16
N = 400

@kernel
def tree(RST: u8[N], VLD: u8[N], OP: u8[N], D: u16[N, 16], VO: u8[N], RO: u16[N], LVO: u8[N], LRO: u16[N, 4]):
    vq0: i32 = 0
    vq1: i32 = 0
    vq2: i32 = 0
    vq3: i32 = 0
    vq4: i32 = 0
    vq5: i32 = 0
    vq6: i32 = 0
    vq7: i32 = 0
    vq8: i32 = 0
    rq0: i32 = 0
    rq1: i32 = 0
    rq2: i32 = 0
    rq3: i32 = 0
    rq4: i32 = 0
    rq5: i32 = 0
    rq6: i32 = 0
    rq7: i32 = 0
    rq8: i32 = 0
    lvq0: i32 = 0
    lvq1: i32 = 0
    lvq2: i32 = 0
    lvq3: i32 = 0
    lvq4: i32 = 0
    lrq0_0: i32 = 0
    lrq0_1: i32 = 0
    lrq0_2: i32 = 0
    lrq0_3: i32 = 0
    lrq1_0: i32 = 0
    lrq1_1: i32 = 0
    lrq1_2: i32 = 0
    lrq1_3: i32 = 0
    lrq2_0: i32 = 0
    lrq2_1: i32 = 0
    lrq2_2: i32 = 0
    lrq2_3: i32 = 0
    lrq3_0: i32 = 0
    lrq3_1: i32 = 0
    lrq3_2: i32 = 0
    lrq3_3: i32 = 0
    lrq4_0: i32 = 0
    lrq4_1: i32 = 0
    lrq4_2: i32 = 0
    lrq4_3: i32 = 0
    for t in range(N, name="t"):
        r: i32 = RST[t]
        v: i32 = VLD[t]
        o: i32 = OP[t]
        H16: i32 = 65536
        ONE: i32 = 1
        ZERO: i32 = 0
        TEN: i32 = 10
        K16: i32 = 65535
        S15: i32 = 32768
        a_large: i32 = 0
        ea: i32 = 0
        ea1: i32 = 0
        eb: i32 = 0
        eb1: i32 = 0
        ed: i32 = 0
        el: i32 = 0
        er: i32 = 0
        es: i32 = 0
        fa: i32 = 0
        fb: i32 = 0
        g: i32 = 0
        gt: i32 = 0
        ha: i32 = 0
        hb: i32 = 0
        ka: i32 = 0
        kb: i32 = 0
        lz: i32 = 0
        ma: i32 = 0
        mag: i32 = 0
        mb: i32 = 0
        ml: i32 = 0
        ms: i32 = 0
        mx: i32 = 0
        ns: i32 = 0
        rb: i32 = 0
        res: i32 = 0
        rnd: i32 = 0
        ru: i32 = 0
        s: i32 = 0
        sa: i32 = 0
        sb: i32 = 0
        sh: i32 = 0
        sl: i32 = 0
        sm: i32 = 0
        x: i32 = 0
        a: i32 = 0
        b: i32 = 0
        n0: i32 = D[t, 0]
        n1: i32 = D[t, 1]
        n2: i32 = D[t, 2]
        n3: i32 = D[t, 3]
        n4: i32 = D[t, 4]
        n5: i32 = D[t, 5]
        n6: i32 = D[t, 6]
        n7: i32 = D[t, 7]
        n8: i32 = D[t, 8]
        n9: i32 = D[t, 9]
        n10: i32 = D[t, 10]
        n11: i32 = D[t, 11]
        n12: i32 = D[t, 12]
        n13: i32 = D[t, 13]
        n14: i32 = D[t, 14]
        n15: i32 = D[t, 15]
        a = n0
        b = n1
        sa = (a >> 15) & 1
        sb = (b >> 15) & 1
        ea = (a >> 7) & 255
        eb = (b >> 7) & 255
        fa = a & 127
        fb = b & 127
        ha = H16 if ea != 0 else ZERO
        hb = H16 if eb != 0 else ZERO
        ma = (fa << 9) | ha
        mb = (fb << 9) | hb
        a_large = ONE if ((ea << 17) | ma) >= ((eb << 17) | mb) else ZERO
        ea1 = ONE if ea == 0 else ea
        eb1 = ONE if eb == 0 else eb
        sl = sa if a_large == 1 else sb
        el = ea1 if a_large == 1 else eb1
        es = eb1 if a_large == 1 else ea1
        ml = ma if a_large == 1 else mb
        ms = mb if a_large == 1 else ma
        ed = el - es
        sh = TEN if ed >= 10 else ed
        sm = ms >> sh
        mag = (ml + sm) if sa == sb else (ml - sm)
        er = el
        res = 0
        # max: bf16_gt(a, b) ? a : b -- vpu_pkg::bf16_gt, an unsigned compare of
        # the sign-flipped key (a total order on bit patterns: -NaN < -Inf < ..
        # < -0 < +0 < .. < +Inf < +NaN, NaNs by payload)
        ka = a ^ S15
        kb = b ^ S15
        if sa == 1:
            ka = K16 - a
        if sb == 1:
            kb = K16 - b
        gt = ONE if ka > kb else ZERO
        if o != 0:
            res = a if gt == 1 else b
        elif (ea == 255 and fa != 0) or (eb == 255 and fb != 0) or (ea == 255 and eb == 255 and sa != sb):
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
                x = mag
                lz = 0
                if x < 512:
                    lz = lz + 8
                    x = x << 8
                if x < 8192:
                    lz = lz + 4
                    x = x << 4
                if x < 32768:
                    lz = lz + 2
                    x = x << 2
                if x < 65536:
                    lz = lz + 1
                    x = x << 1
                if x < 65536:
                    lz = lz + 1
                mx = 16 if er > 17 else er - 1
                ns = lz if lz < mx else mx
                mag = mag << ns
                er = er - ns
            g = (mag >> 8) & 1
            rb = (mag >> 7) & 1
            s = ONE if (mag & 127) != 0 else ZERO
            ru = g & (rb | s | ((mag >> 9) & 1))
            rnd = ((mag >> 9) & 127) + ru
            if rnd >= 128:
                rnd = 0
                er = er + 1
            if er >= 255:
                res = (sl << 15) | 0x7F80
            elif er <= 1 and mag < 65536:
                res = (sl << 15) | rnd
            else:
                res = (sl << 15) | ((er & 255) << 7) | rnd
        n16: i32 = res
        a = n2
        b = n3
        sa = (a >> 15) & 1
        sb = (b >> 15) & 1
        ea = (a >> 7) & 255
        eb = (b >> 7) & 255
        fa = a & 127
        fb = b & 127
        ha = H16 if ea != 0 else ZERO
        hb = H16 if eb != 0 else ZERO
        ma = (fa << 9) | ha
        mb = (fb << 9) | hb
        a_large = ONE if ((ea << 17) | ma) >= ((eb << 17) | mb) else ZERO
        ea1 = ONE if ea == 0 else ea
        eb1 = ONE if eb == 0 else eb
        sl = sa if a_large == 1 else sb
        el = ea1 if a_large == 1 else eb1
        es = eb1 if a_large == 1 else ea1
        ml = ma if a_large == 1 else mb
        ms = mb if a_large == 1 else ma
        ed = el - es
        sh = TEN if ed >= 10 else ed
        sm = ms >> sh
        mag = (ml + sm) if sa == sb else (ml - sm)
        er = el
        res = 0
        # max: bf16_gt(a, b) ? a : b -- vpu_pkg::bf16_gt, an unsigned compare of
        # the sign-flipped key (a total order on bit patterns: -NaN < -Inf < ..
        # < -0 < +0 < .. < +Inf < +NaN, NaNs by payload)
        ka = a ^ S15
        kb = b ^ S15
        if sa == 1:
            ka = K16 - a
        if sb == 1:
            kb = K16 - b
        gt = ONE if ka > kb else ZERO
        if o != 0:
            res = a if gt == 1 else b
        elif (ea == 255 and fa != 0) or (eb == 255 and fb != 0) or (ea == 255 and eb == 255 and sa != sb):
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
                x = mag
                lz = 0
                if x < 512:
                    lz = lz + 8
                    x = x << 8
                if x < 8192:
                    lz = lz + 4
                    x = x << 4
                if x < 32768:
                    lz = lz + 2
                    x = x << 2
                if x < 65536:
                    lz = lz + 1
                    x = x << 1
                if x < 65536:
                    lz = lz + 1
                mx = 16 if er > 17 else er - 1
                ns = lz if lz < mx else mx
                mag = mag << ns
                er = er - ns
            g = (mag >> 8) & 1
            rb = (mag >> 7) & 1
            s = ONE if (mag & 127) != 0 else ZERO
            ru = g & (rb | s | ((mag >> 9) & 1))
            rnd = ((mag >> 9) & 127) + ru
            if rnd >= 128:
                rnd = 0
                er = er + 1
            if er >= 255:
                res = (sl << 15) | 0x7F80
            elif er <= 1 and mag < 65536:
                res = (sl << 15) | rnd
            else:
                res = (sl << 15) | ((er & 255) << 7) | rnd
        n17: i32 = res
        a = n4
        b = n5
        sa = (a >> 15) & 1
        sb = (b >> 15) & 1
        ea = (a >> 7) & 255
        eb = (b >> 7) & 255
        fa = a & 127
        fb = b & 127
        ha = H16 if ea != 0 else ZERO
        hb = H16 if eb != 0 else ZERO
        ma = (fa << 9) | ha
        mb = (fb << 9) | hb
        a_large = ONE if ((ea << 17) | ma) >= ((eb << 17) | mb) else ZERO
        ea1 = ONE if ea == 0 else ea
        eb1 = ONE if eb == 0 else eb
        sl = sa if a_large == 1 else sb
        el = ea1 if a_large == 1 else eb1
        es = eb1 if a_large == 1 else ea1
        ml = ma if a_large == 1 else mb
        ms = mb if a_large == 1 else ma
        ed = el - es
        sh = TEN if ed >= 10 else ed
        sm = ms >> sh
        mag = (ml + sm) if sa == sb else (ml - sm)
        er = el
        res = 0
        # max: bf16_gt(a, b) ? a : b -- vpu_pkg::bf16_gt, an unsigned compare of
        # the sign-flipped key (a total order on bit patterns: -NaN < -Inf < ..
        # < -0 < +0 < .. < +Inf < +NaN, NaNs by payload)
        ka = a ^ S15
        kb = b ^ S15
        if sa == 1:
            ka = K16 - a
        if sb == 1:
            kb = K16 - b
        gt = ONE if ka > kb else ZERO
        if o != 0:
            res = a if gt == 1 else b
        elif (ea == 255 and fa != 0) or (eb == 255 and fb != 0) or (ea == 255 and eb == 255 and sa != sb):
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
                x = mag
                lz = 0
                if x < 512:
                    lz = lz + 8
                    x = x << 8
                if x < 8192:
                    lz = lz + 4
                    x = x << 4
                if x < 32768:
                    lz = lz + 2
                    x = x << 2
                if x < 65536:
                    lz = lz + 1
                    x = x << 1
                if x < 65536:
                    lz = lz + 1
                mx = 16 if er > 17 else er - 1
                ns = lz if lz < mx else mx
                mag = mag << ns
                er = er - ns
            g = (mag >> 8) & 1
            rb = (mag >> 7) & 1
            s = ONE if (mag & 127) != 0 else ZERO
            ru = g & (rb | s | ((mag >> 9) & 1))
            rnd = ((mag >> 9) & 127) + ru
            if rnd >= 128:
                rnd = 0
                er = er + 1
            if er >= 255:
                res = (sl << 15) | 0x7F80
            elif er <= 1 and mag < 65536:
                res = (sl << 15) | rnd
            else:
                res = (sl << 15) | ((er & 255) << 7) | rnd
        n18: i32 = res
        a = n6
        b = n7
        sa = (a >> 15) & 1
        sb = (b >> 15) & 1
        ea = (a >> 7) & 255
        eb = (b >> 7) & 255
        fa = a & 127
        fb = b & 127
        ha = H16 if ea != 0 else ZERO
        hb = H16 if eb != 0 else ZERO
        ma = (fa << 9) | ha
        mb = (fb << 9) | hb
        a_large = ONE if ((ea << 17) | ma) >= ((eb << 17) | mb) else ZERO
        ea1 = ONE if ea == 0 else ea
        eb1 = ONE if eb == 0 else eb
        sl = sa if a_large == 1 else sb
        el = ea1 if a_large == 1 else eb1
        es = eb1 if a_large == 1 else ea1
        ml = ma if a_large == 1 else mb
        ms = mb if a_large == 1 else ma
        ed = el - es
        sh = TEN if ed >= 10 else ed
        sm = ms >> sh
        mag = (ml + sm) if sa == sb else (ml - sm)
        er = el
        res = 0
        # max: bf16_gt(a, b) ? a : b -- vpu_pkg::bf16_gt, an unsigned compare of
        # the sign-flipped key (a total order on bit patterns: -NaN < -Inf < ..
        # < -0 < +0 < .. < +Inf < +NaN, NaNs by payload)
        ka = a ^ S15
        kb = b ^ S15
        if sa == 1:
            ka = K16 - a
        if sb == 1:
            kb = K16 - b
        gt = ONE if ka > kb else ZERO
        if o != 0:
            res = a if gt == 1 else b
        elif (ea == 255 and fa != 0) or (eb == 255 and fb != 0) or (ea == 255 and eb == 255 and sa != sb):
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
                x = mag
                lz = 0
                if x < 512:
                    lz = lz + 8
                    x = x << 8
                if x < 8192:
                    lz = lz + 4
                    x = x << 4
                if x < 32768:
                    lz = lz + 2
                    x = x << 2
                if x < 65536:
                    lz = lz + 1
                    x = x << 1
                if x < 65536:
                    lz = lz + 1
                mx = 16 if er > 17 else er - 1
                ns = lz if lz < mx else mx
                mag = mag << ns
                er = er - ns
            g = (mag >> 8) & 1
            rb = (mag >> 7) & 1
            s = ONE if (mag & 127) != 0 else ZERO
            ru = g & (rb | s | ((mag >> 9) & 1))
            rnd = ((mag >> 9) & 127) + ru
            if rnd >= 128:
                rnd = 0
                er = er + 1
            if er >= 255:
                res = (sl << 15) | 0x7F80
            elif er <= 1 and mag < 65536:
                res = (sl << 15) | rnd
            else:
                res = (sl << 15) | ((er & 255) << 7) | rnd
        n19: i32 = res
        a = n8
        b = n9
        sa = (a >> 15) & 1
        sb = (b >> 15) & 1
        ea = (a >> 7) & 255
        eb = (b >> 7) & 255
        fa = a & 127
        fb = b & 127
        ha = H16 if ea != 0 else ZERO
        hb = H16 if eb != 0 else ZERO
        ma = (fa << 9) | ha
        mb = (fb << 9) | hb
        a_large = ONE if ((ea << 17) | ma) >= ((eb << 17) | mb) else ZERO
        ea1 = ONE if ea == 0 else ea
        eb1 = ONE if eb == 0 else eb
        sl = sa if a_large == 1 else sb
        el = ea1 if a_large == 1 else eb1
        es = eb1 if a_large == 1 else ea1
        ml = ma if a_large == 1 else mb
        ms = mb if a_large == 1 else ma
        ed = el - es
        sh = TEN if ed >= 10 else ed
        sm = ms >> sh
        mag = (ml + sm) if sa == sb else (ml - sm)
        er = el
        res = 0
        # max: bf16_gt(a, b) ? a : b -- vpu_pkg::bf16_gt, an unsigned compare of
        # the sign-flipped key (a total order on bit patterns: -NaN < -Inf < ..
        # < -0 < +0 < .. < +Inf < +NaN, NaNs by payload)
        ka = a ^ S15
        kb = b ^ S15
        if sa == 1:
            ka = K16 - a
        if sb == 1:
            kb = K16 - b
        gt = ONE if ka > kb else ZERO
        if o != 0:
            res = a if gt == 1 else b
        elif (ea == 255 and fa != 0) or (eb == 255 and fb != 0) or (ea == 255 and eb == 255 and sa != sb):
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
                x = mag
                lz = 0
                if x < 512:
                    lz = lz + 8
                    x = x << 8
                if x < 8192:
                    lz = lz + 4
                    x = x << 4
                if x < 32768:
                    lz = lz + 2
                    x = x << 2
                if x < 65536:
                    lz = lz + 1
                    x = x << 1
                if x < 65536:
                    lz = lz + 1
                mx = 16 if er > 17 else er - 1
                ns = lz if lz < mx else mx
                mag = mag << ns
                er = er - ns
            g = (mag >> 8) & 1
            rb = (mag >> 7) & 1
            s = ONE if (mag & 127) != 0 else ZERO
            ru = g & (rb | s | ((mag >> 9) & 1))
            rnd = ((mag >> 9) & 127) + ru
            if rnd >= 128:
                rnd = 0
                er = er + 1
            if er >= 255:
                res = (sl << 15) | 0x7F80
            elif er <= 1 and mag < 65536:
                res = (sl << 15) | rnd
            else:
                res = (sl << 15) | ((er & 255) << 7) | rnd
        n20: i32 = res
        a = n10
        b = n11
        sa = (a >> 15) & 1
        sb = (b >> 15) & 1
        ea = (a >> 7) & 255
        eb = (b >> 7) & 255
        fa = a & 127
        fb = b & 127
        ha = H16 if ea != 0 else ZERO
        hb = H16 if eb != 0 else ZERO
        ma = (fa << 9) | ha
        mb = (fb << 9) | hb
        a_large = ONE if ((ea << 17) | ma) >= ((eb << 17) | mb) else ZERO
        ea1 = ONE if ea == 0 else ea
        eb1 = ONE if eb == 0 else eb
        sl = sa if a_large == 1 else sb
        el = ea1 if a_large == 1 else eb1
        es = eb1 if a_large == 1 else ea1
        ml = ma if a_large == 1 else mb
        ms = mb if a_large == 1 else ma
        ed = el - es
        sh = TEN if ed >= 10 else ed
        sm = ms >> sh
        mag = (ml + sm) if sa == sb else (ml - sm)
        er = el
        res = 0
        # max: bf16_gt(a, b) ? a : b -- vpu_pkg::bf16_gt, an unsigned compare of
        # the sign-flipped key (a total order on bit patterns: -NaN < -Inf < ..
        # < -0 < +0 < .. < +Inf < +NaN, NaNs by payload)
        ka = a ^ S15
        kb = b ^ S15
        if sa == 1:
            ka = K16 - a
        if sb == 1:
            kb = K16 - b
        gt = ONE if ka > kb else ZERO
        if o != 0:
            res = a if gt == 1 else b
        elif (ea == 255 and fa != 0) or (eb == 255 and fb != 0) or (ea == 255 and eb == 255 and sa != sb):
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
                x = mag
                lz = 0
                if x < 512:
                    lz = lz + 8
                    x = x << 8
                if x < 8192:
                    lz = lz + 4
                    x = x << 4
                if x < 32768:
                    lz = lz + 2
                    x = x << 2
                if x < 65536:
                    lz = lz + 1
                    x = x << 1
                if x < 65536:
                    lz = lz + 1
                mx = 16 if er > 17 else er - 1
                ns = lz if lz < mx else mx
                mag = mag << ns
                er = er - ns
            g = (mag >> 8) & 1
            rb = (mag >> 7) & 1
            s = ONE if (mag & 127) != 0 else ZERO
            ru = g & (rb | s | ((mag >> 9) & 1))
            rnd = ((mag >> 9) & 127) + ru
            if rnd >= 128:
                rnd = 0
                er = er + 1
            if er >= 255:
                res = (sl << 15) | 0x7F80
            elif er <= 1 and mag < 65536:
                res = (sl << 15) | rnd
            else:
                res = (sl << 15) | ((er & 255) << 7) | rnd
        n21: i32 = res
        a = n12
        b = n13
        sa = (a >> 15) & 1
        sb = (b >> 15) & 1
        ea = (a >> 7) & 255
        eb = (b >> 7) & 255
        fa = a & 127
        fb = b & 127
        ha = H16 if ea != 0 else ZERO
        hb = H16 if eb != 0 else ZERO
        ma = (fa << 9) | ha
        mb = (fb << 9) | hb
        a_large = ONE if ((ea << 17) | ma) >= ((eb << 17) | mb) else ZERO
        ea1 = ONE if ea == 0 else ea
        eb1 = ONE if eb == 0 else eb
        sl = sa if a_large == 1 else sb
        el = ea1 if a_large == 1 else eb1
        es = eb1 if a_large == 1 else ea1
        ml = ma if a_large == 1 else mb
        ms = mb if a_large == 1 else ma
        ed = el - es
        sh = TEN if ed >= 10 else ed
        sm = ms >> sh
        mag = (ml + sm) if sa == sb else (ml - sm)
        er = el
        res = 0
        # max: bf16_gt(a, b) ? a : b -- vpu_pkg::bf16_gt, an unsigned compare of
        # the sign-flipped key (a total order on bit patterns: -NaN < -Inf < ..
        # < -0 < +0 < .. < +Inf < +NaN, NaNs by payload)
        ka = a ^ S15
        kb = b ^ S15
        if sa == 1:
            ka = K16 - a
        if sb == 1:
            kb = K16 - b
        gt = ONE if ka > kb else ZERO
        if o != 0:
            res = a if gt == 1 else b
        elif (ea == 255 and fa != 0) or (eb == 255 and fb != 0) or (ea == 255 and eb == 255 and sa != sb):
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
                x = mag
                lz = 0
                if x < 512:
                    lz = lz + 8
                    x = x << 8
                if x < 8192:
                    lz = lz + 4
                    x = x << 4
                if x < 32768:
                    lz = lz + 2
                    x = x << 2
                if x < 65536:
                    lz = lz + 1
                    x = x << 1
                if x < 65536:
                    lz = lz + 1
                mx = 16 if er > 17 else er - 1
                ns = lz if lz < mx else mx
                mag = mag << ns
                er = er - ns
            g = (mag >> 8) & 1
            rb = (mag >> 7) & 1
            s = ONE if (mag & 127) != 0 else ZERO
            ru = g & (rb | s | ((mag >> 9) & 1))
            rnd = ((mag >> 9) & 127) + ru
            if rnd >= 128:
                rnd = 0
                er = er + 1
            if er >= 255:
                res = (sl << 15) | 0x7F80
            elif er <= 1 and mag < 65536:
                res = (sl << 15) | rnd
            else:
                res = (sl << 15) | ((er & 255) << 7) | rnd
        n22: i32 = res
        a = n14
        b = n15
        sa = (a >> 15) & 1
        sb = (b >> 15) & 1
        ea = (a >> 7) & 255
        eb = (b >> 7) & 255
        fa = a & 127
        fb = b & 127
        ha = H16 if ea != 0 else ZERO
        hb = H16 if eb != 0 else ZERO
        ma = (fa << 9) | ha
        mb = (fb << 9) | hb
        a_large = ONE if ((ea << 17) | ma) >= ((eb << 17) | mb) else ZERO
        ea1 = ONE if ea == 0 else ea
        eb1 = ONE if eb == 0 else eb
        sl = sa if a_large == 1 else sb
        el = ea1 if a_large == 1 else eb1
        es = eb1 if a_large == 1 else ea1
        ml = ma if a_large == 1 else mb
        ms = mb if a_large == 1 else ma
        ed = el - es
        sh = TEN if ed >= 10 else ed
        sm = ms >> sh
        mag = (ml + sm) if sa == sb else (ml - sm)
        er = el
        res = 0
        # max: bf16_gt(a, b) ? a : b -- vpu_pkg::bf16_gt, an unsigned compare of
        # the sign-flipped key (a total order on bit patterns: -NaN < -Inf < ..
        # < -0 < +0 < .. < +Inf < +NaN, NaNs by payload)
        ka = a ^ S15
        kb = b ^ S15
        if sa == 1:
            ka = K16 - a
        if sb == 1:
            kb = K16 - b
        gt = ONE if ka > kb else ZERO
        if o != 0:
            res = a if gt == 1 else b
        elif (ea == 255 and fa != 0) or (eb == 255 and fb != 0) or (ea == 255 and eb == 255 and sa != sb):
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
                x = mag
                lz = 0
                if x < 512:
                    lz = lz + 8
                    x = x << 8
                if x < 8192:
                    lz = lz + 4
                    x = x << 4
                if x < 32768:
                    lz = lz + 2
                    x = x << 2
                if x < 65536:
                    lz = lz + 1
                    x = x << 1
                if x < 65536:
                    lz = lz + 1
                mx = 16 if er > 17 else er - 1
                ns = lz if lz < mx else mx
                mag = mag << ns
                er = er - ns
            g = (mag >> 8) & 1
            rb = (mag >> 7) & 1
            s = ONE if (mag & 127) != 0 else ZERO
            ru = g & (rb | s | ((mag >> 9) & 1))
            rnd = ((mag >> 9) & 127) + ru
            if rnd >= 128:
                rnd = 0
                er = er + 1
            if er >= 255:
                res = (sl << 15) | 0x7F80
            elif er <= 1 and mag < 65536:
                res = (sl << 15) | rnd
            else:
                res = (sl << 15) | ((er & 255) << 7) | rnd
        n23: i32 = res
        a = n16
        b = n17
        sa = (a >> 15) & 1
        sb = (b >> 15) & 1
        ea = (a >> 7) & 255
        eb = (b >> 7) & 255
        fa = a & 127
        fb = b & 127
        ha = H16 if ea != 0 else ZERO
        hb = H16 if eb != 0 else ZERO
        ma = (fa << 9) | ha
        mb = (fb << 9) | hb
        a_large = ONE if ((ea << 17) | ma) >= ((eb << 17) | mb) else ZERO
        ea1 = ONE if ea == 0 else ea
        eb1 = ONE if eb == 0 else eb
        sl = sa if a_large == 1 else sb
        el = ea1 if a_large == 1 else eb1
        es = eb1 if a_large == 1 else ea1
        ml = ma if a_large == 1 else mb
        ms = mb if a_large == 1 else ma
        ed = el - es
        sh = TEN if ed >= 10 else ed
        sm = ms >> sh
        mag = (ml + sm) if sa == sb else (ml - sm)
        er = el
        res = 0
        # max: bf16_gt(a, b) ? a : b -- vpu_pkg::bf16_gt, an unsigned compare of
        # the sign-flipped key (a total order on bit patterns: -NaN < -Inf < ..
        # < -0 < +0 < .. < +Inf < +NaN, NaNs by payload)
        ka = a ^ S15
        kb = b ^ S15
        if sa == 1:
            ka = K16 - a
        if sb == 1:
            kb = K16 - b
        gt = ONE if ka > kb else ZERO
        if o != 0:
            res = a if gt == 1 else b
        elif (ea == 255 and fa != 0) or (eb == 255 and fb != 0) or (ea == 255 and eb == 255 and sa != sb):
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
                x = mag
                lz = 0
                if x < 512:
                    lz = lz + 8
                    x = x << 8
                if x < 8192:
                    lz = lz + 4
                    x = x << 4
                if x < 32768:
                    lz = lz + 2
                    x = x << 2
                if x < 65536:
                    lz = lz + 1
                    x = x << 1
                if x < 65536:
                    lz = lz + 1
                mx = 16 if er > 17 else er - 1
                ns = lz if lz < mx else mx
                mag = mag << ns
                er = er - ns
            g = (mag >> 8) & 1
            rb = (mag >> 7) & 1
            s = ONE if (mag & 127) != 0 else ZERO
            ru = g & (rb | s | ((mag >> 9) & 1))
            rnd = ((mag >> 9) & 127) + ru
            if rnd >= 128:
                rnd = 0
                er = er + 1
            if er >= 255:
                res = (sl << 15) | 0x7F80
            elif er <= 1 and mag < 65536:
                res = (sl << 15) | rnd
            else:
                res = (sl << 15) | ((er & 255) << 7) | rnd
        n24: i32 = res
        a = n18
        b = n19
        sa = (a >> 15) & 1
        sb = (b >> 15) & 1
        ea = (a >> 7) & 255
        eb = (b >> 7) & 255
        fa = a & 127
        fb = b & 127
        ha = H16 if ea != 0 else ZERO
        hb = H16 if eb != 0 else ZERO
        ma = (fa << 9) | ha
        mb = (fb << 9) | hb
        a_large = ONE if ((ea << 17) | ma) >= ((eb << 17) | mb) else ZERO
        ea1 = ONE if ea == 0 else ea
        eb1 = ONE if eb == 0 else eb
        sl = sa if a_large == 1 else sb
        el = ea1 if a_large == 1 else eb1
        es = eb1 if a_large == 1 else ea1
        ml = ma if a_large == 1 else mb
        ms = mb if a_large == 1 else ma
        ed = el - es
        sh = TEN if ed >= 10 else ed
        sm = ms >> sh
        mag = (ml + sm) if sa == sb else (ml - sm)
        er = el
        res = 0
        # max: bf16_gt(a, b) ? a : b -- vpu_pkg::bf16_gt, an unsigned compare of
        # the sign-flipped key (a total order on bit patterns: -NaN < -Inf < ..
        # < -0 < +0 < .. < +Inf < +NaN, NaNs by payload)
        ka = a ^ S15
        kb = b ^ S15
        if sa == 1:
            ka = K16 - a
        if sb == 1:
            kb = K16 - b
        gt = ONE if ka > kb else ZERO
        if o != 0:
            res = a if gt == 1 else b
        elif (ea == 255 and fa != 0) or (eb == 255 and fb != 0) or (ea == 255 and eb == 255 and sa != sb):
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
                x = mag
                lz = 0
                if x < 512:
                    lz = lz + 8
                    x = x << 8
                if x < 8192:
                    lz = lz + 4
                    x = x << 4
                if x < 32768:
                    lz = lz + 2
                    x = x << 2
                if x < 65536:
                    lz = lz + 1
                    x = x << 1
                if x < 65536:
                    lz = lz + 1
                mx = 16 if er > 17 else er - 1
                ns = lz if lz < mx else mx
                mag = mag << ns
                er = er - ns
            g = (mag >> 8) & 1
            rb = (mag >> 7) & 1
            s = ONE if (mag & 127) != 0 else ZERO
            ru = g & (rb | s | ((mag >> 9) & 1))
            rnd = ((mag >> 9) & 127) + ru
            if rnd >= 128:
                rnd = 0
                er = er + 1
            if er >= 255:
                res = (sl << 15) | 0x7F80
            elif er <= 1 and mag < 65536:
                res = (sl << 15) | rnd
            else:
                res = (sl << 15) | ((er & 255) << 7) | rnd
        n25: i32 = res
        a = n20
        b = n21
        sa = (a >> 15) & 1
        sb = (b >> 15) & 1
        ea = (a >> 7) & 255
        eb = (b >> 7) & 255
        fa = a & 127
        fb = b & 127
        ha = H16 if ea != 0 else ZERO
        hb = H16 if eb != 0 else ZERO
        ma = (fa << 9) | ha
        mb = (fb << 9) | hb
        a_large = ONE if ((ea << 17) | ma) >= ((eb << 17) | mb) else ZERO
        ea1 = ONE if ea == 0 else ea
        eb1 = ONE if eb == 0 else eb
        sl = sa if a_large == 1 else sb
        el = ea1 if a_large == 1 else eb1
        es = eb1 if a_large == 1 else ea1
        ml = ma if a_large == 1 else mb
        ms = mb if a_large == 1 else ma
        ed = el - es
        sh = TEN if ed >= 10 else ed
        sm = ms >> sh
        mag = (ml + sm) if sa == sb else (ml - sm)
        er = el
        res = 0
        # max: bf16_gt(a, b) ? a : b -- vpu_pkg::bf16_gt, an unsigned compare of
        # the sign-flipped key (a total order on bit patterns: -NaN < -Inf < ..
        # < -0 < +0 < .. < +Inf < +NaN, NaNs by payload)
        ka = a ^ S15
        kb = b ^ S15
        if sa == 1:
            ka = K16 - a
        if sb == 1:
            kb = K16 - b
        gt = ONE if ka > kb else ZERO
        if o != 0:
            res = a if gt == 1 else b
        elif (ea == 255 and fa != 0) or (eb == 255 and fb != 0) or (ea == 255 and eb == 255 and sa != sb):
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
                x = mag
                lz = 0
                if x < 512:
                    lz = lz + 8
                    x = x << 8
                if x < 8192:
                    lz = lz + 4
                    x = x << 4
                if x < 32768:
                    lz = lz + 2
                    x = x << 2
                if x < 65536:
                    lz = lz + 1
                    x = x << 1
                if x < 65536:
                    lz = lz + 1
                mx = 16 if er > 17 else er - 1
                ns = lz if lz < mx else mx
                mag = mag << ns
                er = er - ns
            g = (mag >> 8) & 1
            rb = (mag >> 7) & 1
            s = ONE if (mag & 127) != 0 else ZERO
            ru = g & (rb | s | ((mag >> 9) & 1))
            rnd = ((mag >> 9) & 127) + ru
            if rnd >= 128:
                rnd = 0
                er = er + 1
            if er >= 255:
                res = (sl << 15) | 0x7F80
            elif er <= 1 and mag < 65536:
                res = (sl << 15) | rnd
            else:
                res = (sl << 15) | ((er & 255) << 7) | rnd
        n26: i32 = res
        a = n22
        b = n23
        sa = (a >> 15) & 1
        sb = (b >> 15) & 1
        ea = (a >> 7) & 255
        eb = (b >> 7) & 255
        fa = a & 127
        fb = b & 127
        ha = H16 if ea != 0 else ZERO
        hb = H16 if eb != 0 else ZERO
        ma = (fa << 9) | ha
        mb = (fb << 9) | hb
        a_large = ONE if ((ea << 17) | ma) >= ((eb << 17) | mb) else ZERO
        ea1 = ONE if ea == 0 else ea
        eb1 = ONE if eb == 0 else eb
        sl = sa if a_large == 1 else sb
        el = ea1 if a_large == 1 else eb1
        es = eb1 if a_large == 1 else ea1
        ml = ma if a_large == 1 else mb
        ms = mb if a_large == 1 else ma
        ed = el - es
        sh = TEN if ed >= 10 else ed
        sm = ms >> sh
        mag = (ml + sm) if sa == sb else (ml - sm)
        er = el
        res = 0
        # max: bf16_gt(a, b) ? a : b -- vpu_pkg::bf16_gt, an unsigned compare of
        # the sign-flipped key (a total order on bit patterns: -NaN < -Inf < ..
        # < -0 < +0 < .. < +Inf < +NaN, NaNs by payload)
        ka = a ^ S15
        kb = b ^ S15
        if sa == 1:
            ka = K16 - a
        if sb == 1:
            kb = K16 - b
        gt = ONE if ka > kb else ZERO
        if o != 0:
            res = a if gt == 1 else b
        elif (ea == 255 and fa != 0) or (eb == 255 and fb != 0) or (ea == 255 and eb == 255 and sa != sb):
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
                x = mag
                lz = 0
                if x < 512:
                    lz = lz + 8
                    x = x << 8
                if x < 8192:
                    lz = lz + 4
                    x = x << 4
                if x < 32768:
                    lz = lz + 2
                    x = x << 2
                if x < 65536:
                    lz = lz + 1
                    x = x << 1
                if x < 65536:
                    lz = lz + 1
                mx = 16 if er > 17 else er - 1
                ns = lz if lz < mx else mx
                mag = mag << ns
                er = er - ns
            g = (mag >> 8) & 1
            rb = (mag >> 7) & 1
            s = ONE if (mag & 127) != 0 else ZERO
            ru = g & (rb | s | ((mag >> 9) & 1))
            rnd = ((mag >> 9) & 127) + ru
            if rnd >= 128:
                rnd = 0
                er = er + 1
            if er >= 255:
                res = (sl << 15) | 0x7F80
            elif er <= 1 and mag < 65536:
                res = (sl << 15) | rnd
            else:
                res = (sl << 15) | ((er & 255) << 7) | rnd
        n27: i32 = res
        a = n24
        b = n25
        sa = (a >> 15) & 1
        sb = (b >> 15) & 1
        ea = (a >> 7) & 255
        eb = (b >> 7) & 255
        fa = a & 127
        fb = b & 127
        ha = H16 if ea != 0 else ZERO
        hb = H16 if eb != 0 else ZERO
        ma = (fa << 9) | ha
        mb = (fb << 9) | hb
        a_large = ONE if ((ea << 17) | ma) >= ((eb << 17) | mb) else ZERO
        ea1 = ONE if ea == 0 else ea
        eb1 = ONE if eb == 0 else eb
        sl = sa if a_large == 1 else sb
        el = ea1 if a_large == 1 else eb1
        es = eb1 if a_large == 1 else ea1
        ml = ma if a_large == 1 else mb
        ms = mb if a_large == 1 else ma
        ed = el - es
        sh = TEN if ed >= 10 else ed
        sm = ms >> sh
        mag = (ml + sm) if sa == sb else (ml - sm)
        er = el
        res = 0
        # max: bf16_gt(a, b) ? a : b -- vpu_pkg::bf16_gt, an unsigned compare of
        # the sign-flipped key (a total order on bit patterns: -NaN < -Inf < ..
        # < -0 < +0 < .. < +Inf < +NaN, NaNs by payload)
        ka = a ^ S15
        kb = b ^ S15
        if sa == 1:
            ka = K16 - a
        if sb == 1:
            kb = K16 - b
        gt = ONE if ka > kb else ZERO
        if o != 0:
            res = a if gt == 1 else b
        elif (ea == 255 and fa != 0) or (eb == 255 and fb != 0) or (ea == 255 and eb == 255 and sa != sb):
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
                x = mag
                lz = 0
                if x < 512:
                    lz = lz + 8
                    x = x << 8
                if x < 8192:
                    lz = lz + 4
                    x = x << 4
                if x < 32768:
                    lz = lz + 2
                    x = x << 2
                if x < 65536:
                    lz = lz + 1
                    x = x << 1
                if x < 65536:
                    lz = lz + 1
                mx = 16 if er > 17 else er - 1
                ns = lz if lz < mx else mx
                mag = mag << ns
                er = er - ns
            g = (mag >> 8) & 1
            rb = (mag >> 7) & 1
            s = ONE if (mag & 127) != 0 else ZERO
            ru = g & (rb | s | ((mag >> 9) & 1))
            rnd = ((mag >> 9) & 127) + ru
            if rnd >= 128:
                rnd = 0
                er = er + 1
            if er >= 255:
                res = (sl << 15) | 0x7F80
            elif er <= 1 and mag < 65536:
                res = (sl << 15) | rnd
            else:
                res = (sl << 15) | ((er & 255) << 7) | rnd
        n28: i32 = res
        a = n26
        b = n27
        sa = (a >> 15) & 1
        sb = (b >> 15) & 1
        ea = (a >> 7) & 255
        eb = (b >> 7) & 255
        fa = a & 127
        fb = b & 127
        ha = H16 if ea != 0 else ZERO
        hb = H16 if eb != 0 else ZERO
        ma = (fa << 9) | ha
        mb = (fb << 9) | hb
        a_large = ONE if ((ea << 17) | ma) >= ((eb << 17) | mb) else ZERO
        ea1 = ONE if ea == 0 else ea
        eb1 = ONE if eb == 0 else eb
        sl = sa if a_large == 1 else sb
        el = ea1 if a_large == 1 else eb1
        es = eb1 if a_large == 1 else ea1
        ml = ma if a_large == 1 else mb
        ms = mb if a_large == 1 else ma
        ed = el - es
        sh = TEN if ed >= 10 else ed
        sm = ms >> sh
        mag = (ml + sm) if sa == sb else (ml - sm)
        er = el
        res = 0
        # max: bf16_gt(a, b) ? a : b -- vpu_pkg::bf16_gt, an unsigned compare of
        # the sign-flipped key (a total order on bit patterns: -NaN < -Inf < ..
        # < -0 < +0 < .. < +Inf < +NaN, NaNs by payload)
        ka = a ^ S15
        kb = b ^ S15
        if sa == 1:
            ka = K16 - a
        if sb == 1:
            kb = K16 - b
        gt = ONE if ka > kb else ZERO
        if o != 0:
            res = a if gt == 1 else b
        elif (ea == 255 and fa != 0) or (eb == 255 and fb != 0) or (ea == 255 and eb == 255 and sa != sb):
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
                x = mag
                lz = 0
                if x < 512:
                    lz = lz + 8
                    x = x << 8
                if x < 8192:
                    lz = lz + 4
                    x = x << 4
                if x < 32768:
                    lz = lz + 2
                    x = x << 2
                if x < 65536:
                    lz = lz + 1
                    x = x << 1
                if x < 65536:
                    lz = lz + 1
                mx = 16 if er > 17 else er - 1
                ns = lz if lz < mx else mx
                mag = mag << ns
                er = er - ns
            g = (mag >> 8) & 1
            rb = (mag >> 7) & 1
            s = ONE if (mag & 127) != 0 else ZERO
            ru = g & (rb | s | ((mag >> 9) & 1))
            rnd = ((mag >> 9) & 127) + ru
            if rnd >= 128:
                rnd = 0
                er = er + 1
            if er >= 255:
                res = (sl << 15) | 0x7F80
            elif er <= 1 and mag < 65536:
                res = (sl << 15) | rnd
            else:
                res = (sl << 15) | ((er & 255) << 7) | rnd
        n29: i32 = res
        a = n28
        b = n29
        sa = (a >> 15) & 1
        sb = (b >> 15) & 1
        ea = (a >> 7) & 255
        eb = (b >> 7) & 255
        fa = a & 127
        fb = b & 127
        ha = H16 if ea != 0 else ZERO
        hb = H16 if eb != 0 else ZERO
        ma = (fa << 9) | ha
        mb = (fb << 9) | hb
        a_large = ONE if ((ea << 17) | ma) >= ((eb << 17) | mb) else ZERO
        ea1 = ONE if ea == 0 else ea
        eb1 = ONE if eb == 0 else eb
        sl = sa if a_large == 1 else sb
        el = ea1 if a_large == 1 else eb1
        es = eb1 if a_large == 1 else ea1
        ml = ma if a_large == 1 else mb
        ms = mb if a_large == 1 else ma
        ed = el - es
        sh = TEN if ed >= 10 else ed
        sm = ms >> sh
        mag = (ml + sm) if sa == sb else (ml - sm)
        er = el
        res = 0
        # max: bf16_gt(a, b) ? a : b -- vpu_pkg::bf16_gt, an unsigned compare of
        # the sign-flipped key (a total order on bit patterns: -NaN < -Inf < ..
        # < -0 < +0 < .. < +Inf < +NaN, NaNs by payload)
        ka = a ^ S15
        kb = b ^ S15
        if sa == 1:
            ka = K16 - a
        if sb == 1:
            kb = K16 - b
        gt = ONE if ka > kb else ZERO
        if o != 0:
            res = a if gt == 1 else b
        elif (ea == 255 and fa != 0) or (eb == 255 and fb != 0) or (ea == 255 and eb == 255 and sa != sb):
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
                x = mag
                lz = 0
                if x < 512:
                    lz = lz + 8
                    x = x << 8
                if x < 8192:
                    lz = lz + 4
                    x = x << 4
                if x < 32768:
                    lz = lz + 2
                    x = x << 2
                if x < 65536:
                    lz = lz + 1
                    x = x << 1
                if x < 65536:
                    lz = lz + 1
                mx = 16 if er > 17 else er - 1
                ns = lz if lz < mx else mx
                mag = mag << ns
                er = er - ns
            g = (mag >> 8) & 1
            rb = (mag >> 7) & 1
            s = ONE if (mag & 127) != 0 else ZERO
            ru = g & (rb | s | ((mag >> 9) & 1))
            rnd = ((mag >> 9) & 127) + ru
            if rnd >= 128:
                rnd = 0
                er = er + 1
            if er >= 255:
                res = (sl << 15) | 0x7F80
            elif er <= 1 and mag < 65536:
                res = (sl << 15) | rnd
            else:
                res = (sl << 15) | ((er & 255) << 7) | rnd
        n30: i32 = res
        vq8 = vq7
        rq8 = rq7
        vq7 = vq6
        rq7 = rq6
        vq6 = vq5
        rq6 = rq5
        vq5 = vq4
        rq5 = rq4
        vq4 = vq3
        rq4 = rq3
        vq3 = vq2
        rq3 = rq2
        vq2 = vq1
        rq2 = rq1
        vq1 = vq0
        rq1 = rq0
        vq0 = v
        rq0 = n30
        lvq4 = lvq3
        lrq4_0 = lrq3_0
        lrq4_1 = lrq3_1
        lrq4_2 = lrq3_2
        lrq4_3 = lrq3_3
        lvq3 = lvq2
        lrq3_0 = lrq2_0
        lrq3_1 = lrq2_1
        lrq3_2 = lrq2_2
        lrq3_3 = lrq2_3
        lvq2 = lvq1
        lrq2_0 = lrq1_0
        lrq2_1 = lrq1_1
        lrq2_2 = lrq1_2
        lrq2_3 = lrq1_3
        lvq1 = lvq0
        lrq1_0 = lrq0_0
        lrq1_1 = lrq0_1
        lrq1_2 = lrq0_2
        lrq1_3 = lrq0_3
        lvq0 = v
        lrq0_0 = n24
        lrq0_1 = n25
        lrq0_2 = n26
        lrq0_3 = n27
        if r == 0:
            vq0 = 0
            vq1 = 0
            vq2 = 0
            vq3 = 0
            vq4 = 0
            vq5 = 0
            vq6 = 0
            vq7 = 0
            vq8 = 0
            lvq0 = 0
            lvq1 = 0
            lvq2 = 0
            lvq3 = 0
            lvq4 = 0
        LRO[t, 0] = lrq4_0
        LRO[t, 1] = lrq4_1
        LRO[t, 2] = lrq4_2
        LRO[t, 3] = lrq4_3
        VO[t] = vq8
        RO[t] = rq8
        LVO[t] = lvq4
