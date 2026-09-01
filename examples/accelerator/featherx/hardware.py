import allo
from allo import kernel, consteval
from allo.lang import i32, Stream, i8
import math


def compute_birrd_params(AW: int) -> tuple[int, int]:
    """Compute BIRRD network parameters from array width.

    Returns:
        P0: Number of stages in BIRRD network
        P1: Number of switches per stage
    """
    LOG2_AW = int(math.log2(AW))
    P0 = 2 * LOG2_AW if AW > 4 else 2 * LOG2_AW - 1
    P1 = AW // 2
    return P0, P1


@consteval(lazy=True)
def reverse_bits(data: i32, bit_range: i32) -> i32:
    """Reverse the lower bit_range bits of data.

    Used for butterfly network routing in BIRRD.
    """
    mask = (1 << bit_range) - 1
    reversed_bits: i32 = 0
    for i in range(0, bit_range):
        i_32: i32 = i
        if data & (1 << i_32):
            reversed_bits |= 1 << (bit_range - 1 - i_32)
    return (data & ~mask) | reversed_bits


def make_featherx(M, K, N, AW, AH, Ty, num_inst, n_inner=1, k_passes=1, Nt_local=None):

    if Nt_local is None:
        Nt_local = AH
    TyOut = i32
    LOG2_AW = int(math.log2(AW))
    LOG2_AH = int(math.log2(AH))
    P0, P1 = compute_birrd_params(AW)
    num_tiles = num_inst - 3
    total_ops = num_tiles * n_inner
    num_blocks = num_tiles // k_passes

    num_accum_params = 2 + num_tiles  # quant_scale, quant_zp, sr[0..num_tiles-1]

    @kernel
    def a_loader(
        A_buf: i32[M, K],
        inst: i32[num_inst, 13],
        loader_m_start: i32[total_ops],
        col_a_in: Stream[i32, AH][AW],
    ):
        iacts_zp: i32 = inst[0, 6]

        for tile in range(num_tiles):
            inst_idx: i32 = tile + 3
            Gr: i32 = inst[inst_idx, 3]
            k_start_tile: i32 = inst[inst_idx, 11]

            # Compute log2_Gr via comparison chain (Gr is power of 2)
            log2_Gr: i32 = 0
            if Gr >= 2:
                log2_Gr = 1
            if Gr >= 4:
                log2_Gr = 2
            if Gr >= 8:
                log2_Gr = 3
            if Gr >= 16:
                log2_Gr = 4
            mask_Gr: i32 = Gr - 1

            for inner in range(n_inner):
                op_idx: i32 = tile * n_inner + inner
                m_start: i32 = loader_m_start[op_idx]

                # Send A values (1 per column per nk cycle)
                for nk in range(AH):
                    for nj in range(AW):  # unrolled: static stream index
                        m_idx: i32 = m_start + (allo.cast(nj, i32) & mask_Gr)
                        k_idx: i32 = (
                            k_start_tile + nk + (allo.cast(nj, i32) >> log2_Gr) * AH
                        )
                        a_val: i32 = A_buf[m_idx, k_idx] - iacts_zp
                        col_a_in[nj].put(a_val)

    @kernel
    def w_loader(
        B_buf: i32[K, N],
        inst: i32[num_inst, 13],
        loader_n_start: i32[total_ops],
        col_w_in: Stream[i32, total_ops][AH, AW],
    ):
        weights_zp: i32 = inst[1, 6]

        for tile in range(num_tiles):
            inst_idx: i32 = tile + 3
            Gr: i32 = inst[inst_idx, 3]
            Gc: i32 = inst[inst_idx, 4]
            sr: i32 = inst[inst_idx, 5]
            sc: i32 = inst[inst_idx, 6]
            k_start_tile: i32 = inst[inst_idx, 11]

            # Compute log2_Gr via comparison chain (Gr is power of 2)
            log2_Gr: i32 = 0
            if Gr >= 2:
                log2_Gr = 1
            if Gr >= 4:
                log2_Gr = 2
            if Gr >= 8:
                log2_Gr = 3
            if Gr >= 16:
                log2_Gr = 4
            mask_Gc: i32 = Gc - 1

            for inner in range(n_inner):
                op_idx: i32 = tile * n_inner + inner
                n_start: i32 = loader_n_start[op_idx]

                # Send W values (1 per column per nk cycle)
                for nk in range(AH):
                    for nj in range(AW):  # fully unroll
                        for pe_row in range(AH):  # fully unroll
                            k_idx: i32 = (
                                k_start_tile + nk + (allo.cast(nj, i32) >> log2_Gr) * AH
                            )
                            wn_idx: i32 = (
                                n_start
                                + sr * pe_row
                                + sc * (allo.cast(nj, i32) & mask_Gc)
                            )
                            w_val: i32 = B_buf[k_idx, wn_idx] - weights_zp
                            col_w_in[pe_row, nj].put(w_val)

    @kernel(mapping=[AW])
    def w_broadcast(
        col_w_in: Stream[i32, total_ops][AH, AW],
        pe_w_in: Stream[i32, total_ops][AH, AW],
    ):
        nj = allo.get_wid(0)
        for _op in range(total_ops):
            for nk in range(AH):
                for row in range(AH):  # fully unroll
                    w_val: i32 = col_w_in[row, nj].get()
                    pe_w_in[row, nj].put(w_val)

    @kernel(mapping=[AH + 1, AW])
    def pe_array(
        pe_out: Stream[i32, total_ops][AH, AW],
        connection: Stream[TyOut, AH][P0 + 1, P1 * 2],
        col_a_in: Stream[i32, AH][AW],
        pe_w_in: Stream[i32, total_ops][AH, AW],
        pe_a_down: Stream[i32, AH][AH, AW],
    ):
        ni, nj = allo.get_wid(0), allo.get_wid(1)

        if ni == AH:
            for _op in range(total_ops):
                buf: i32[AH]
                for pe_row in range(AH):  # fully unroll
                    buf[pe_row] = pe_out[pe_row, nj].get()
                for row in range(AH):
                    connection[0, nj].put(buf[row])
        else:
            for _op in range(total_ops):
                tile_accum: i32 = 0
                for nk in range(AH):
                    a_val: i32 = 0
                    if ni == 0:
                        a_val = col_a_in[nj].get()
                    else:
                        a_val = pe_a_down[ni - 1, nj].get()

                    w_val: i32 = pe_w_in[ni, nj].get()
                    if ni < AH - 1:
                        pe_a_down[ni, nj].put(a_val)

                    tile_accum += a_val * w_val
                pe_out[ni, nj].put(tile_accum)

    @kernel
    def inst_rw(
        inst_input: Stream[i8, total_ops][P0, P1], birrd_inst: i8[num_tiles, P0, P1]
    ):
        for tile in range(num_tiles):
            for _rep in range(n_inner):
                for i in range(P0):  # fully unroll
                    for j in range(P1):  # fully unroll
                        inst_input[i, j].put(birrd_inst[tile, i, j])

    @kernel(mapping=[P0, P1])
    def BIRRD(
        inst_input: Stream[i8, total_ops][P0, P1],
        connection: Stream[TyOut, AH][P0 + 1, P1 * 2],
    ):
        i, j = allo.get_wid(0), allo.get_wid(1)

        for _op in range(total_ops):
            inst_val = inst_input[i, j].get()

            for _ in range(AH):
                in_left: TyOut = connection[i, j * 2].get()
                in_right: TyOut = connection[i, j * 2 + 1].get()

                out_left: TyOut = 0
                out_right: TyOut = 0

                if inst_val == 0:
                    out_left = in_left
                    out_right = in_right
                elif inst_val == 1:
                    out_left = in_left
                    out_right = in_left + in_right
                elif inst_val == 2:
                    out_left = in_left + in_right
                    out_right = in_right
                else:
                    out_left = in_right
                    out_right = in_left

                if i != P0 - 1:
                    connection[
                        i + 1,
                        reverse_bits(
                            2 * j,
                            2 if i == 0 else min(min(LOG2_AH, 2 + i), 2 * LOG2_AW - i),
                        ),
                    ].put(out_left)
                    connection[
                        i + 1,
                        reverse_bits(
                            2 * j + 1,
                            2 if i == 0 else min(min(LOG2_AH, 2 + i), 2 * LOG2_AW - i),
                        ),
                    ].put(out_right)
                else:
                    connection[i + 1, j * 2].put(out_left)
                    connection[i + 1, j * 2 + 1].put(out_right)

    @kernel
    def output_accum(
        local_output_col_map: i32[num_tiles, AW],
        local_output_num_m: i32[num_tiles],
        local_output_n_base: i32[num_tiles, AW],
        local_accum_m_start: i32[total_ops],
        local_accum_n_start: i32[total_ops],
        local_accum_params: i32[num_accum_params],
        C_buf: i32[M, N],
        connection: Stream[TyOut, AH][P0 + 1, P1 * 2],
    ):
        quant_scale: i32 = local_accum_params[0]
        quant_zp: i32 = local_accum_params[1]

        # Accumulator indexed by (col, d) — fixed indices for II=1
        tile_acc: i32[AW, AH]

        for block in range(num_blocks):
            # Zero tile_acc (meta_for on outer dim for AW-parallel zeroing)
            for i0 in range(AW):  # fully unroll
                for _j0 in range(AH):
                    tile_acc[i0, _j0] = 0

            base_tile: i32 = block * k_passes

            # Fused: read from BIRRD streams + accumulate with fixed indices
            for k in range(k_passes):
                for inner in range(n_inner):
                    for d in range(AH):
                        for col in range(
                            AW, name="oacc"
                        ):  # unrolled: static stream index
                            tile_acc[col, d] = (
                                tile_acc[col, d] + connection[P0, col].get()
                            )

            # Writeback: apply col→m mapping and write to C
            num_m: i32 = local_output_num_m[base_tile]
            sr_val: i32 = local_accum_params[2 + base_tile]
            m_start: i32 = local_accum_m_start[base_tile * n_inner]
            n_start: i32 = local_accum_n_start[base_tile * n_inner]
            for col in range(AW):
                m_pos: i32 = local_output_col_map[base_tile, col]
                n_base_col: i32 = local_output_n_base[base_tile, col]
                col_mask: i32 = 0
                m_safe: i32 = 0
                if m_pos < num_m:
                    col_mask = 1
                    m_safe = m_pos
                for on in range(AH):
                    sr_mask: i32 = 0
                    if sr_val != 0:
                        sr_mask = 1
                    if on == 0:
                        sr_mask = 1
                    n_off: i32 = sr_val * on + n_base_col
                    val: i32 = tile_acc[col, on] * col_mask * sr_mask
                    if quant_scale != 0:
                        val = (val * quant_scale + quant_zp) & 255
                    C_buf[m_start + m_safe, n_start + n_off] = val

    @kernel
    def store_C(C_buf: i32[M, N], local_C: i32[M, N]):
        # Dedicated DMA process: on-chip C_buf -> DRAM C (own dataflow stage).
        for fi in range(M):
            for fj in range(N):
                local_C[fi, fj] = C_buf[fi, fj]

    @kernel
    def featherx(
        A_pe: i32[M, K],
        B_pe: i32[K, N],
        inst_pe: i32[num_inst, 13],
        loader_m_start: i32[total_ops],
        inst_w: i32[num_inst, 13],
        loader_n_start: i32[total_ops],
        birrd_inst: i8[num_tiles, P0, P1],
        output_col_map: i32[num_tiles, AW],
        output_num_m: i32[num_tiles],
        output_n_base: i32[num_tiles, AW],
        accum_m_start: i32[total_ops],
        accum_n_start: i32[total_ops],
        accum_params: i32[num_accum_params],
        C: i32[M, N],
    ):
        # Streams for A and W values
        col_a_in: Stream[i32, AH][AW]
        col_w_in: Stream[i32, total_ops][AH, AW]
        pe_w_in: Stream[i32, total_ops][AH, AW]
        pe_out: Stream[i32, total_ops][AH, AW]
        pe_a_down: Stream[i32, AH][AH, AW]

        # Streams for BIRRD network
        inst_input: Stream[i8, total_ops][P0, P1]
        connection: Stream[TyOut, AH][P0 + 1, P1 * 2]

        # bufferize all
        local_A = A_pe.bufferize()
        local_B = B_pe.bufferize()
        inst_pe = inst_pe.bufferize()
        loader_m_start = loader_m_start.bufferize()
        inst_w = inst_w.bufferize()
        loader_n_start = loader_n_start.bufferize()
        birrd_inst = birrd_inst.bufferize()
        output_col_map = output_col_map.bufferize()
        output_num_m = output_num_m.bufferize()
        output_n_base = output_n_base.bufferize()
        accum_m_start = accum_m_start.bufferize()
        accum_n_start = accum_n_start.bufferize()
        accum_params = accum_params.bufferize()
        local_C: i32[M, N]

        a_loader(local_A, inst_pe, loader_m_start, col_a_in)
        w_loader(local_B, inst_w, loader_n_start, col_w_in)
        w_broadcast(col_w_in, pe_w_in)
        pe_array(pe_out, connection, col_a_in, pe_w_in, pe_a_down)
        inst_rw(inst_input, birrd_inst)
        BIRRD(inst_input, connection)
        output_accum(
            output_col_map,
            output_num_m,
            output_n_base,
            accum_m_start,
            accum_n_start,
            accum_params,
            local_C,
            connection,
        )
        store_C(local_C, C)

    def build_schedule():
        """Compose per-kernel schedules into a dataflow top, ready to export.

        Every loop that indexes a stream array by a loop variable is unrolled so
        the index becomes a compile-time constant (required for Vitis stream-array
        codegen); the rest of the parallelism is structural (PE-array mappings).
        """

        al = a_loader.schedule()
        al.unroll("nj")
        al.pipeline("tile")  # flatten inner+nk, stream one tile per cycle

        wl = w_loader.schedule()
        wl.unroll(["nj", "pe_row"])
        wl.pipeline("tile")

        wb = w_broadcast.schedule()
        wb.unroll("row")

        pa = pe_array.schedule()
        pa.unroll("pe_row")

        ir = inst_rw.schedule()
        ir.unroll(["i", "j"])

        oa = output_accum.schedule()
        oa.unroll("oacc")
        oa.unroll("i0")
        oa.partition(oa.buffer("tile_acc"), dim=1, kind=oa.Complete)
        oa.partition(oa.buffer("tile_acc"), dim=2, kind=oa.Complete)

        sc = store_C.schedule()
        sc.pipeline("fj")  # burst C_buf -> DRAM C

        sch = featherx.schedule()
        sch.partition(sch.buffer("local_A"), dim=2, kind=sch.Complete)
        sch.partition(sch.buffer("local_A"), dim=1, kind=sch.Cyclic, factor=AW)
        sch.partition(sch.buffer("local_B"), dim=1, kind=sch.Complete)
        sch.partition(sch.buffer("local_B"), dim=2, kind=sch.Cyclic, factor=AH)
        sch.partition(sch.buffer("local_C"), dim=2, kind=sch.Complete)
        sch.dataflow()
        sch.compose(al, wl, wb, pa, ir, oa, sc)

        return sch

    return featherx, build_schedule
