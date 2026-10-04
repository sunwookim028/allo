@df.region()
def rf_d12(
    RA: UInt(5)[N],
    RB: UInt(5)[N],
    RC: UInt(5)[N],
    WA: UInt(5)[N],
    WD: UInt(W)[N],
    WE: uint1[N],
    QA: UInt(W)[N],
    QB: UInt(W)[N],
    QC: UInt(W)[N],
):
    ra: Stream[UInt(5), 2]
    rb: Stream[UInt(5), 2]
    rc: Stream[UInt(5), 2]
    wa: Stream[UInt(5), 2]
    wd: Stream[UInt(W), 2]
    we: Stream[uint1, 2]
    qa: Stream[UInt(W), 2]
    qb: Stream[UInt(W), 2]
    qc: Stream[UInt(W), 2]
    vreg_ra_a: Stream[UInt(5), 2]
    vreg_ra_q: Stream[UInt(W), 2]
    vreg_rb_a: Stream[UInt(5), 2]
    vreg_rb_q: Stream[UInt(W), 2]
    vreg_rc_a: Stream[UInt(5), 2]
    vreg_rc_q: Stream[UInt(W), 2]
    vreg_w_a: Stream[UInt(5), 2]
    vreg_w_d: Stream[UInt(W), 2]
    vreg_w_e: Stream[uint1, 2]

    @df.kernel(mapping=[1], args=[RA, RB, RC, WA, WD, WE])
    def src(xa: UInt(5)[N], xb: UInt(5)[N], xc: UInt(5)[N], xwa: UInt(5)[N],
            xwd: UInt(W)[N], xwe: uint1[N]):
        for t in range(N):
            ra.put(xa[t])
            rb.put(xb[t])
            rc.put(xc[t])
            wa.put(xwa[t])
            wd.put(xwd[t])
            we.put(xwe[t])

    @df.kernel(mapping=[1])
    def rd_a():
        for _ in range(N):
            a5: UInt(5) = ra.get()
            a: int32 = a5
            _vreg_ra_a: UInt(5) = a
            vreg_ra_a.put(_vreg_ra_a)
            _vreg_ra_q: UInt(W) = vreg_ra_q.get()
            qa.put(_vreg_ra_q)

    @df.kernel(mapping=[1])
    def rd_b():
        for _ in range(N):
            b5: UInt(5) = rb.get()
            b: int32 = b5
            _vreg_rb_a: UInt(5) = b
            vreg_rb_a.put(_vreg_rb_a)
            _vreg_rb_q: UInt(W) = vreg_rb_q.get()
            qb.put(_vreg_rb_q)

    @df.kernel(mapping=[1])
    def rd_c():
        for _ in range(N):
            c5: UInt(5) = rc.get()
            c: int32 = c5
            _vreg_rc_a: UInt(5) = c
            vreg_rc_a.put(_vreg_rc_a)
            _vreg_rc_q: UInt(W) = vreg_rc_q.get()
            qc.put(_vreg_rc_q)

    @df.kernel(mapping=[1])
    def wb():
        for _ in range(N):
            x5: UInt(5) = wa.get()
            x: int32 = x5
            d: UInt(W) = wd.get()
            e: uint1 = we.get()
            _vreg_w_a: UInt(5) = x
            vreg_w_a.put(_vreg_w_a)
            _vreg_w_d: UInt(W) = d
            vreg_w_d.put(_vreg_w_d)
            _vreg_w_e: uint1 = e
            vreg_w_e.put(_vreg_w_e)

    @df.kernel(mapping=[1], args=[QA, QB, QC])
    def sink(ya: UInt(W)[N], yb: UInt(W)[N], yc: UInt(W)[N]):
        for t in range(N):
            ya[t] = qa.get()
            yb[t] = qb.get()
            yc[t] = qc.get()

    @df.kernel(mapping=[1])
    def vreg_mem():
        mem: UInt(W)[32]
        for _ in range(N):
            _ra_a: UInt(5) = vreg_ra_a.get()
            _ra_i: int32 = _ra_a
            vreg_ra_q.put(mem[_ra_i])
            _rb_a: UInt(5) = vreg_rb_a.get()
            _rb_i: int32 = _rb_a
            vreg_rb_q.put(mem[_rb_i])
            _rc_a: UInt(5) = vreg_rc_a.get()
            _rc_i: int32 = _rc_a
            vreg_rc_q.put(mem[_rc_i])
            _w_a: UInt(5) = vreg_w_a.get()
            _w_i: int32 = _w_a
            _w_d: UInt(W) = vreg_w_d.get()
            _w_e: uint1 = vreg_w_e.get()
            if _w_e:
                mem[_w_i] = _w_d
