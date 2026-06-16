import struct

import numpy as np

N = 8
OP_MATMUL_TILE = 0x01
FLUSH_INSTR = 0x5200000000000000
ACC_DST = 2 * N
SPM_A = N
SPM_B = 0


def fp16_bits(value):
    return int(np.float16(value).view(np.uint16))


def fp32_bits(value):
    return struct.unpack(">I", struct.pack(">f", float(value)))[0]


def encode_instr(op, flags=0, op0=0, op1=0, op2=0):
    return (
        ((op & 0xFF) << 56)
        | ((flags & 0xFF) << 48)
        | ((op0 & 0xFFFF) << 32)
        | ((op1 & 0xFFFF) << 16)
        | (op2 & 0xFFFF)
    )


def pack_program_word(word):
    return f"{word & ((1 << 64) - 1):064x}"


def pack_spad_row_fp16(vals):
    result = 0
    for i, value in enumerate(vals):
        result |= (fp16_bits(value) & 0xFFFF) << (i * 16)
    return f"{result:064x}"


def pack_spad_row_fp32(vals):
    result = 0
    for i, value in enumerate(vals):
        result |= (fp32_bits(value) & 0xFFFFFFFF) << (i * 32)
    return f"{result:064x}"


def main():
    rng = np.random.default_rng(0xC011EC7)

    # Use small fp16-exact integer values so the ASIC FP units should produce
    # deterministic fp32 rows while still toggling a nontrivial matrix path.
    w_mat = rng.integers(-2, 3, size=(N, N)).astype(np.float16).astype(np.float32)
    x_mat = rng.integers(-2, 3, size=(N, N)).astype(np.float16).astype(np.float32)
    expected = (x_mat @ w_mat.T).astype(np.float32)

    program = [
        encode_instr(OP_MATMUL_TILE, op0=ACC_DST, op1=SPM_A, op2=SPM_B),
        FLUSH_INSTR,
    ]

    rows = []
    rows.extend(pack_program_word(word) for word in program)
    rows.extend(pack_spad_row_fp16(w_mat[i]) for i in range(N))
    rows.extend(pack_spad_row_fp16(x_mat[i]) for i in range(N))
    rows.extend(pack_spad_row_fp32(expected[i]) for i in range(N))

    with open("test_vectors.txt", "w", encoding="ascii") as f:
        for row in rows:
            f.write(row + "\n")


if __name__ == "__main__":
    main()
