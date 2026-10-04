@df.region()
def wa_d12(
    CE: uint1[N],
    CW: uint1[N],
    CA: UInt(AW)[N],
    CD: UInt(W)[N],
    DE: uint1[N],
    DW: uint1[N],
    DA: UInt(AW)[N],
    DD: UInt(W)[N],
    QC: UInt(W)[N],
    QD: UInt(W)[N],
):
    ce: Wire[uint1]
    cw: Wire[uint1]
    ca: Wire[UInt(AW)]
    cd: Wire[UInt(W)]
    de: Wire[uint1]
    dw: Wire[uint1]
    da: Wire[UInt(AW)]
    dd: Wire[UInt(W)]
    qc: Wire[UInt(W), comb]
    qd: Wire[UInt(W), comb]
    vmem_c_a: Wire[UInt(3), comb]
    vmem_c_q: Wire[UInt(W)]
    vmem_c_d: Wire[UInt(W), comb]
    vmem_c_e: Wire[uint1, comb]
    vmem_d_a: Wire[UInt(3), comb]
    vmem_d_q: Wire[UInt(W)]
    vmem_d_d: Wire[UInt(W), comb]
    vmem_d_e: Wire[uint1, comb]

    @df.kernel(mapping=[1], args=[CE, CW, CA, CD, DE, DW, DA, DD])
    def src(xce: uint1[N], xcw: uint1[N], xca: UInt(AW)[N], xcd: UInt(W)[N],
            xde: uint1[N], xdw: uint1[N], xda: UInt(AW)[N], xdd: UInt(W)[N]):
        for t in range(N):
            ce.put(xce[t])
            cw.put(xcw[t])
            ca.put(xca[t])
            cd.put(xcd[t])
            de.put(xde[t])
            dw.put(xdw[t])
            da.put(xda[t])
            dd.put(xdd[t])

    @df.kernel(mapping=[1])
    def port_c():
        for _ in range(N):
            e: uint1 = ce.get()
            w: uint1 = cw.get()
            a_: UInt(AW) = ca.get()
            a: int32 = a_
            d: UInt(W) = cd.get()
            _vmem_c_a: UInt(3) = a
            vmem_c_a.put(_vmem_c_a)
            _vmem_c_q: UInt(W) = vmem_c_q.get()
            qc.put(_vmem_c_q)
            ew: uint1 = e & w
            _vmem_c_d: UInt(W) = d
            vmem_c_d.put(_vmem_c_d)
            _vmem_c_e: uint1 = ew
            vmem_c_e.put(_vmem_c_e)

    @df.kernel(mapping=[1])
    def port_d():
        for _ in range(N):
            e: uint1 = de.get()
            w: uint1 = dw.get()
            a_: UInt(AW) = da.get()
            a: int32 = a_
            d: UInt(W) = dd.get()
            _vmem_d_a: UInt(3) = a
            vmem_d_a.put(_vmem_d_a)
            _vmem_d_q: UInt(W) = vmem_d_q.get()
            qd.put(_vmem_d_q)
            ew: uint1 = e & w
            _vmem_d_d: UInt(W) = d
            vmem_d_d.put(_vmem_d_d)
            _vmem_d_e: uint1 = ew
            vmem_d_e.put(_vmem_d_e)

    @df.kernel(mapping=[1], args=[QC, QD])
    def sink(yc: UInt(W)[N], yd: UInt(W)[N]):
        for t in range(N):
            yc[t] = qc.get()
            yd[t] = qd.get()

    @df.kernel(mapping=[1])
    def vmem_mem():
        mem: UInt(W)[8] @ Stateful(reset=False)
        _c_p: UInt(W)[3]
        _d_p: UInt(W)[2]
        for _ in range(N):
            _c_a: UInt(3) = vmem_c_a.get()
            _c_i: int32 = _c_a
            _c_p[2] = _c_p[1]
            _c_p[1] = _c_p[0]
            _c_p[0] = mem[_c_i]
            vmem_c_q.put(_c_p[2])
            _d_a: UInt(3) = vmem_d_a.get()
            _d_i: int32 = _d_a
            _d_p[1] = _d_p[0]
            _d_p[0] = mem[_d_i]
            vmem_d_q.put(_d_p[1])
            _c_d: UInt(W) = vmem_c_d.get()
            _c_e: uint1 = vmem_c_e.get()
            if _c_e:
                mem[_c_i] = _c_d
            _d_d: UInt(W) = vmem_d_d.get()
            _d_e: uint1 = vmem_d_e.get()
            if _d_e:
                mem[_d_i] = _d_d
