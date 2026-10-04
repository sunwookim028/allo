@df.region()
def pe_two(
    A_a: UInt(32)[N_WORK],
    W_a: UInt(32)[N_WORK],
    P_a: UInt(32)[N_WORK],
    OUT_a: UInt(32)[N_WORK],
    A_b: UInt(32)[N_WORK],
    W_b: UInt(32)[N_WORK],
    P_b: UInt(32)[N_WORK],
    OUT_b: UInt(32)[N_WORK],
):
    lhs_a: Stream[MAC_IN__a, 4]
    wq_a: Stream[MAC_IN__a, 4]
    psum_in_a: Stream[MAC_ACC__a, 4]
    psum_out_a: Stream[MAC_ACC__a, 4]
    lhs_b: Stream[MAC_IN__b, 4]
    wq_b: Stream[MAC_IN__b, 4]
    psum_in_b: Stream[MAC_ACC__b, 4]
    psum_out_b: Stream[MAC_ACC__b, 4]

    @df.kernel(mapping=[1], args=[A_a, W_a, P_a])
    def pe_feed_a(a_mem: UInt(32)[N_WORK], w_mem: UInt(32)[N_WORK], p_mem: UInt(32)[N_WORK]):
        for work in range(N_WORK):
            a_word: UInt(32) = a_mem[work]
            w_word: UInt(32) = w_mem[work]
            p_word: UInt(32) = p_mem[work]
            a: MAC_IN__a = a_word
            w: MAC_IN__a = w_word
            p: MAC_ACC__a = p_word
            lhs_a.put(a)
            wq_a.put(w)
            psum_in_a.put(p)

    @df.kernel(mapping=[1])
    def mac_pe_a():
        for work in range(N_WORK):
            a: MAC_IN__a = lhs_a.get()
            w: MAC_IN__a = wq_a.get()
            north: MAC_ACC__a = psum_in_a.get()
            product: MAC_ACC__a = MAC_MUL__a(a, w)
            south: MAC_ACC__a = MAC_ADD__a(product, north)
            psum_out_a.put(south)

    @df.kernel(mapping=[1], args=[OUT_a])
    def pe_sink_a(out_mem: UInt(32)[N_WORK]):
        for work in range(N_WORK):
            south: MAC_ACC__a = psum_out_a.get()
            word: UInt(32) = south
            out_mem[work] = word

    @df.kernel(mapping=[1], args=[A_b, W_b, P_b])
    def pe_feed_b(a_mem: UInt(32)[N_WORK], w_mem: UInt(32)[N_WORK], p_mem: UInt(32)[N_WORK]):
        for work in range(N_WORK):
            a_word: UInt(32) = a_mem[work]
            w_word: UInt(32) = w_mem[work]
            p_word: UInt(32) = p_mem[work]
            a: MAC_IN__b = a_word
            w: MAC_IN__b = w_word
            p: MAC_ACC__b = p_word
            lhs_b.put(a)
            wq_b.put(w)
            psum_in_b.put(p)

    @df.kernel(mapping=[1])
    def mac_pe_b():
        for work in range(N_WORK):
            a: MAC_IN__b = lhs_b.get()
            w: MAC_IN__b = wq_b.get()
            north: MAC_ACC__b = psum_in_b.get()
            product: MAC_ACC__b = MAC_MUL__b(a, w)
            south: MAC_ACC__b = MAC_ADD__b(product, north)
            psum_out_b.put(south)

    @df.kernel(mapping=[1], args=[OUT_b])
    def pe_sink_b(out_mem: UInt(32)[N_WORK]):
        for work in range(N_WORK):
            south: MAC_ACC__b = psum_out_b.get()
            word: UInt(32) = south
            out_mem[work] = word
