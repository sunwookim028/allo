# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""AES-256 ECB encryption: byte-wise table lookup, GF(2^8) mixing and key expansion."""

import numpy as np

from allo.lang import kernel, u8, u32

from ..spec import Benchmark

SBOX = np.array(
    [
        0x63,
        0x7C,
        0x77,
        0x7B,
        0xF2,
        0x6B,
        0x6F,
        0xC5,
        0x30,
        0x01,
        0x67,
        0x2B,
        0xFE,
        0xD7,
        0xAB,
        0x76,
        0xCA,
        0x82,
        0xC9,
        0x7D,
        0xFA,
        0x59,
        0x47,
        0xF0,
        0xAD,
        0xD4,
        0xA2,
        0xAF,
        0x9C,
        0xA4,
        0x72,
        0xC0,
        0xB7,
        0xFD,
        0x93,
        0x26,
        0x36,
        0x3F,
        0xF7,
        0xCC,
        0x34,
        0xA5,
        0xE5,
        0xF1,
        0x71,
        0xD8,
        0x31,
        0x15,
        0x04,
        0xC7,
        0x23,
        0xC3,
        0x18,
        0x96,
        0x05,
        0x9A,
        0x07,
        0x12,
        0x80,
        0xE2,
        0xEB,
        0x27,
        0xB2,
        0x75,
        0x09,
        0x83,
        0x2C,
        0x1A,
        0x1B,
        0x6E,
        0x5A,
        0xA0,
        0x52,
        0x3B,
        0xD6,
        0xB3,
        0x29,
        0xE3,
        0x2F,
        0x84,
        0x53,
        0xD1,
        0x00,
        0xED,
        0x20,
        0xFC,
        0xB1,
        0x5B,
        0x6A,
        0xCB,
        0xBE,
        0x39,
        0x4A,
        0x4C,
        0x58,
        0xCF,
        0xD0,
        0xEF,
        0xAA,
        0xFB,
        0x43,
        0x4D,
        0x33,
        0x85,
        0x45,
        0xF9,
        0x02,
        0x7F,
        0x50,
        0x3C,
        0x9F,
        0xA8,
        0x51,
        0xA3,
        0x40,
        0x8F,
        0x92,
        0x9D,
        0x38,
        0xF5,
        0xBC,
        0xB6,
        0xDA,
        0x21,
        0x10,
        0xFF,
        0xF3,
        0xD2,
        0xCD,
        0x0C,
        0x13,
        0xEC,
        0x5F,
        0x97,
        0x44,
        0x17,
        0xC4,
        0xA7,
        0x7E,
        0x3D,
        0x64,
        0x5D,
        0x19,
        0x73,
        0x60,
        0x81,
        0x4F,
        0xDC,
        0x22,
        0x2A,
        0x90,
        0x88,
        0x46,
        0xEE,
        0xB8,
        0x14,
        0xDE,
        0x5E,
        0x0B,
        0xDB,
        0xE0,
        0x32,
        0x3A,
        0x0A,
        0x49,
        0x06,
        0x24,
        0x5C,
        0xC2,
        0xD3,
        0xAC,
        0x62,
        0x91,
        0x95,
        0xE4,
        0x79,
        0xE7,
        0xC8,
        0x37,
        0x6D,
        0x8D,
        0xD5,
        0x4E,
        0xA9,
        0x6C,
        0x56,
        0xF4,
        0xEA,
        0x65,
        0x7A,
        0xAE,
        0x08,
        0xBA,
        0x78,
        0x25,
        0x2E,
        0x1C,
        0xA6,
        0xB4,
        0xC6,
        0xE8,
        0xDD,
        0x74,
        0x1F,
        0x4B,
        0xBD,
        0x8B,
        0x8A,
        0x70,
        0x3E,
        0xB5,
        0x66,
        0x48,
        0x03,
        0xF6,
        0x0E,
        0x61,
        0x35,
        0x57,
        0xB9,
        0x86,
        0xC1,
        0x1D,
        0x9E,
        0xE1,
        0xF8,
        0x98,
        0x11,
        0x69,
        0xD9,
        0x8E,
        0x94,
        0x9B,
        0x1E,
        0x87,
        0xE9,
        0xCE,
        0x55,
        0x28,
        0xDF,
        0x8C,
        0xA1,
        0x89,
        0x0D,
        0xBF,
        0xE6,
        0x42,
        0x68,
        0x41,
        0x99,
        0x2D,
        0x0F,
        0xB0,
        0x54,
        0xBB,
        0x16,
    ],
    np.uint8,
)


def build():
    @kernel
    def F(x: u8) -> u8:
        return (x << 1) ^ ((x >> 7 & 1) * 0x1B)

    @kernel
    def rj_sbox(x: u8) -> u8:
        sbox: u8[256] = SBOX
        i: u32 = x
        return sbox[i]

    @kernel
    def rj_xtime(x: u8) -> u8:
        return (x << 1) ^ 0x1B if (x & 0x80) != 0x00 else (x << 1)

    @kernel
    def sub_bytes(buf: u8[16]):
        i: u8 = 16
        while i > 0:
            i -= 1
            buf[i] = rj_sbox(buf[i])

    @kernel
    def add_round_key(buf: u8[16], key: u8[16]):
        for i in range(16):
            buf[i] ^= key[i]

    @kernel
    def add_round_key_cpy(buf: u8[16], key: u8[32], cpk: u8[32]):
        i: u8 = 16
        while i > 0:
            i -= 1
            buf[i] ^= key[i]
            cpk[i] = key[i]
            cpk[16 + i] = key[16 + i]

    @kernel
    def shift_rows(buf: u8[16]):
        t: u8[1] = 0
        t[0] = buf[1]
        buf[1] = buf[5]
        buf[5] = buf[9]
        buf[9] = buf[13]
        buf[13] = t[0]

        t[0] = buf[10]
        buf[10] = buf[2]
        buf[2] = t[0]

        t[0] = buf[3]
        buf[3] = buf[15]
        buf[15] = buf[11]
        buf[11] = buf[7]
        buf[7] = t[0]

        t[0] = buf[14]
        buf[14] = buf[6]
        buf[6] = t[0]

    @kernel
    def mix_columns(buf: u8[16]):
        v: u8[5] = 0
        for i in range(0, 16, 4):
            v[0] = buf[i]
            v[1] = buf[i + 1]
            v[2] = buf[i + 2]
            v[3] = buf[i + 3]
            v[4] = v[0] ^ v[1] ^ v[2] ^ v[3]
            buf[i] ^= v[4] ^ rj_xtime(v[0] ^ v[1])
            buf[i + 1] ^= v[4] ^ rj_xtime(v[1] ^ v[2])
            buf[i + 2] ^= v[4] ^ rj_xtime(v[2] ^ v[3])
            buf[i + 3] ^= v[4] ^ rj_xtime(v[3] ^ v[0])

    @kernel
    def expand_enc_key(k: u8[32], rc: u8[1]):
        k[0] ^= rj_sbox(k[29]) ^ (rc[0])
        k[1] ^= rj_sbox(k[30])
        k[2] ^= rj_sbox(k[31])
        k[3] ^= rj_sbox(k[28])
        rc[0] = F(rc[0])

        for i in range(4, 16, 4):
            k[i] ^= k[i - 4]
            k[i + 1] ^= k[i - 3]
            k[i + 2] ^= k[i - 2]
            k[i + 3] ^= k[i - 1]

        k[16] ^= rj_sbox(k[12])
        k[17] ^= rj_sbox(k[13])
        k[18] ^= rj_sbox(k[14])
        k[19] ^= rj_sbox(k[15])

        for i2 in range(20, 32, 4):
            k[i2] ^= k[i2 - 4]
            k[i2 + 1] ^= k[i2 - 3]
            k[i2 + 2] ^= k[i2 - 2]
            k[i2 + 3] ^= k[i2 - 1]

    @kernel
    def encrypt_ecb(k: u8[32], buf: u8[16]):
        key: u8[32]
        enc_key: u8[32]
        dec_key: u8[32]
        rcon: u8[1] = [1]

        for i0 in range(32):
            enc_key[i0] = k[i0]
            dec_key[i0] = k[i0]

        for i1 in range(7):
            expand_enc_key(dec_key, rcon)

        add_round_key_cpy(buf, enc_key, key)

        rcon[0] = 1
        for i2 in range(1, 14):
            sub_bytes(buf)
            shift_rows(buf)
            mix_columns(buf)
            i2_u8: u8 = i2
            if (i2_u8 & 1) != 0:
                temp_key: u8[16]
                for j in range(16):
                    temp_key[j] = key[16 + j]
                add_round_key(buf, temp_key)
            else:
                expand_enc_key(key, rcon)
                temp_key: u8[16]
                for j in range(16):
                    temp_key[j] = key[j]
                add_round_key(buf, temp_key)

        sub_bytes(buf)
        shift_rows(buf)
        expand_enc_key(key, rcon)
        temp_key: u8[16]
        for j in range(16):
            temp_key[j] = key[j]
        add_round_key(buf, temp_key)

    return {
        "top": encrypt_ecb,
        "add_round_key": add_round_key,
        "mix_columns": mix_columns,
        "expand_enc_key": expand_enc_key,
    }


def _none(parts):
    return parts["top"].schedule()


def inputs(rng):
    k = rng.integers(0, 256, 32).astype(np.uint8)
    buf = rng.integers(0, 256, 16).astype(np.uint8)
    return k, buf


def _u8(x):
    return x & 0xFF


def _f(x):
    return _u8((x << 1) ^ (((x >> 7) & 1) * 0x1B))


def _xtime(x):
    return _u8((x << 1) ^ 0x1B) if (x & 0x80) != 0x00 else _u8(x << 1)


def _sub_bytes(buf):
    for i in range(16):
        buf[i] = int(SBOX[buf[i]])


def _add_round_key(buf, key):
    for i in range(16):
        buf[i] = _u8(buf[i] ^ key[i])


def _add_round_key_cpy(buf, key, cpk):
    for i in range(16):
        buf[i] = _u8(buf[i] ^ key[i])
        cpk[i] = key[i]
        cpk[16 + i] = key[16 + i]


def _shift_rows(buf):
    t = buf[1]
    buf[1], buf[5], buf[9], buf[13] = buf[5], buf[9], buf[13], t
    t = buf[10]
    buf[10], buf[2] = buf[2], t
    t = buf[3]
    buf[3], buf[15], buf[11], buf[7] = buf[15], buf[11], buf[7], t
    t = buf[14]
    buf[14], buf[6] = buf[6], t


def _mix_columns(buf):
    for i in range(0, 16, 4):
        a, b, c, d = buf[i], buf[i + 1], buf[i + 2], buf[i + 3]
        e = a ^ b ^ c ^ d
        buf[i] = _u8(a ^ e ^ _xtime(a ^ b))
        buf[i + 1] = _u8(b ^ e ^ _xtime(b ^ c))
        buf[i + 2] = _u8(c ^ e ^ _xtime(c ^ d))
        buf[i + 3] = _u8(d ^ e ^ _xtime(d ^ a))


def _expand_enc_key(k, rc):
    k[0] = _u8(k[0] ^ int(SBOX[k[29]]) ^ rc[0])
    k[1] = _u8(k[1] ^ int(SBOX[k[30]]))
    k[2] = _u8(k[2] ^ int(SBOX[k[31]]))
    k[3] = _u8(k[3] ^ int(SBOX[k[28]]))
    rc[0] = _f(rc[0])
    for i in range(4, 16, 4):
        for o in range(4):
            k[i + o] = _u8(k[i + o] ^ k[i + o - 4])
    k[16] = _u8(k[16] ^ int(SBOX[k[12]]))
    k[17] = _u8(k[17] ^ int(SBOX[k[13]]))
    k[18] = _u8(k[18] ^ int(SBOX[k[14]]))
    k[19] = _u8(k[19] ^ int(SBOX[k[15]]))
    for i in range(20, 32, 4):
        for o in range(4):
            k[i + o] = _u8(k[i + o] ^ k[i + o - 4])


def reference(k, buf):
    key = [0] * 32
    enc_key = [int(x) for x in k]
    dec_key = [int(x) for x in k]
    block = [int(x) for x in buf]
    rcon = [1]
    for _ in range(7):
        _expand_enc_key(dec_key, rcon)
    _add_round_key_cpy(block, enc_key, key)
    rcon[0] = 1
    for i in range(1, 14):
        _sub_bytes(block)
        _shift_rows(block)
        _mix_columns(block)
        if (i & 1) != 0:
            _add_round_key(block, key[16:32])
        else:
            _expand_enc_key(key, rcon)
            _add_round_key(block, key[0:16])
    _sub_bytes(block)
    _shift_rows(block)
    _expand_enc_key(key, rcon)
    _add_round_key(block, key[0:16])
    return (np.array(block, np.uint8),)


BENCHMARK = Benchmark(
    suite="machsuite",
    name="encrypt_ecb",
    build=build,
    schedules={"none": _none},
    inputs=inputs,
    reference=reference,
    outputs=(1,),
)
