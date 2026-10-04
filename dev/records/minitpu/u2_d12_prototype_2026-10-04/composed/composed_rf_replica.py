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
    ra: Wire[UInt(5)]
    rb: Wire[UInt(5)]
    rc: Wire[UInt(5)]
    wa: Wire[UInt(5)]
    wd: Wire[UInt(W)]
    we: Wire[uint1]
    qa: Wire[UInt(W), comb]
    qb: Wire[UInt(W), comb]
    qc: Wire[UInt(W), comb]
    vreg_w_a_ra: Wire[UInt(5), comb]
    vreg_w_d_ra: Wire[UInt(W), comb]
    vreg_w_e_ra: Wire[uint1, comb]
    vreg_w_a_rb: Wire[UInt(5), comb]
    vreg_w_d_rb: Wire[UInt(W), comb]
    vreg_w_e_rb: Wire[uint1, comb]
    vreg_w_a_rc: Wire[UInt(5), comb]
    vreg_w_d_rc: Wire[UInt(W), comb]
    vreg_w_e_rc: Wire[uint1, comb]

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
        _vreg_ra: UInt(W)[32] @ Stateful(reset=False)
        for _ in range(N):
            a5: UInt(5) = ra.get()
            a: int32 = a5
            qa.put(_vreg_ra[a])
            _vreg_ra_wa: UInt(5) = vreg_w_a_ra.get()
            _vreg_ra_wi: int32 = _vreg_ra_wa
            _vreg_ra_wd: UInt(W) = vreg_w_d_ra.get()
            _vreg_ra_we: uint1 = vreg_w_e_ra.get()
            if _vreg_ra_we:
                _vreg_ra[_vreg_ra_wi] = _vreg_ra_wd

    @df.kernel(mapping=[1])
    def rd_b():
        _vreg_rb: UInt(W)[32] @ Stateful(reset=False)
        for _ in range(N):
            b5: UInt(5) = rb.get()
            b: int32 = b5
            qb.put(_vreg_rb[b])
            _vreg_rb_wa: UInt(5) = vreg_w_a_rb.get()
            _vreg_rb_wi: int32 = _vreg_rb_wa
            _vreg_rb_wd: UInt(W) = vreg_w_d_rb.get()
            _vreg_rb_we: uint1 = vreg_w_e_rb.get()
            if _vreg_rb_we:
                _vreg_rb[_vreg_rb_wi] = _vreg_rb_wd

    @df.kernel(mapping=[1])
    def rd_c():
        _vreg_rc: UInt(W)[32] @ Stateful(reset=False)
        for _ in range(N):
            c5: UInt(5) = rc.get()
            c: int32 = c5
            qc.put(_vreg_rc[c])
            _vreg_rc_wa: UInt(5) = vreg_w_a_rc.get()
            _vreg_rc_wi: int32 = _vreg_rc_wa
            _vreg_rc_wd: UInt(W) = vreg_w_d_rc.get()
            _vreg_rc_we: uint1 = vreg_w_e_rc.get()
            if _vreg_rc_we:
                _vreg_rc[_vreg_rc_wi] = _vreg_rc_wd

    @df.kernel(mapping=[1])
    def wb():
        for _ in range(N):
            x5: UInt(5) = wa.get()
            x: int32 = x5
            d: UInt(W) = wd.get()
            e: uint1 = we.get()
            _vreg_w_a: UInt(5) = x
            _vreg_w_d: UInt(W) = d
            _vreg_w_e: uint1 = e
            vreg_w_a_ra.put(_vreg_w_a)
            vreg_w_d_ra.put(_vreg_w_d)
            vreg_w_e_ra.put(_vreg_w_e)
            vreg_w_a_rb.put(_vreg_w_a)
            vreg_w_d_rb.put(_vreg_w_d)
            vreg_w_e_rb.put(_vreg_w_e)
            vreg_w_a_rc.put(_vreg_w_a)
            vreg_w_d_rc.put(_vreg_w_d)
            vreg_w_e_rc.put(_vreg_w_e)

    @df.kernel(mapping=[1], args=[QA, QB, QC])
    def sink(ya: UInt(W)[N], yb: UInt(W)[N], yc: UInt(W)[N]):
        for t in range(N):
            ya[t] = qa.get()
            yb[t] = qb.get()
            yc[t] = qc.get()
