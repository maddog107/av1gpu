// W4: register-resident exact inverse transforms for k_recon.
//
// Templated copies of av1core dsp.h av1g_idct / av1g_iadst4/8/16 / av1g_iidentity / av1g_inv_1d with the
// transform size as a template parameter and every loop unrolled, so the working array lives in registers
// instead of local memory (the generic versions index int32_t T[64] / copyT[64] dynamically: ~45 % of
// k_recon's local stores).  The integer operations, their order and the conformance checks are the same as
// in av1core, so the residual and the error flags are bit-identical.
//
// 32-bit arithmetic: av1core computes the butterflies in int64, but with in-range inputs nothing here needs
// more than 31 bits.  The working values are within r bits (r = BitDepth + 8 <= 18 for rows, 16 for columns):
// the dequantized inputs are clamped to BitDepth + 8 bits, the column inputs are int16, H() clamps to r bits and
// every B() / iadst4 result is range-checked.  So a product T * cos128 is below 2^(r - 1 + 12) and every sum
// below 2^(r + 12) <= 2^30; iadst4's sums are checked at r + 12 bits.  Until the first failed check all values
// are therefore exactly the int64 ones, the first failed check fails here too, and a transform with any failed
// check is discarded (cta_inverse_add halves the levels and redoes it), so the residual and the flags are the
// same.  The arithmetic wraps through uint32_t, so values after a failed check are defined (and unused).
#pragma once
#include <stdint.h>
#include "av1g/dsp.h"

namespace rci {

__host__ __device__ constexpr int brev(int numBits, int x) {
    int t = 0;
    for (int i = 0; i < numBits; i++) t |= ((x >> i) & 1) << (numBits - 1 - i);
    return t;
}
__device__ __forceinline__ int cos128(int angle) { return av1g_cos128(angle); }
__device__ __forceinline__ int sin128(int angle) { return av1g_cos128(angle - 64); }
// wrapping 32-bit operations (see the header: exact for the in-range values that matter)
__device__ __forceinline__ int32_t wadd(int32_t a, int32_t b) { return (int32_t)((uint32_t)a + (uint32_t)b); }
__device__ __forceinline__ int32_t wsub(int32_t a, int32_t b) { return (int32_t)((uint32_t)a - (uint32_t)b); }
__device__ __forceinline__ int32_t wmul(int32_t a, int32_t b) { return (int32_t)((uint32_t)a * (uint32_t)b); }
// av1g_round2_64(x, n) for n >= 1
__device__ __forceinline__ int32_t rnd2(int32_t x, int n) { return wadd(x, 1 << (n - 1)) >> n; }
__device__ __forceinline__ void chk(int32_t v, int bits, uint32_t &err, uint32_t flag) {   // bits <= 31
    if (v < -(1 << (bits - 1)) || v > (1 << (bits - 1)) - 1) err |= flag;
}
template <int L>
__device__ __forceinline__ void B(int32_t (&T)[L], int a, int b, int angle, int flip, int r, uint32_t &err) {
    const int32_t c = cos128(angle), s = sin128(angle);
    const int32_t x = wsub(wmul(T[a], c), wmul(T[b], s));
    const int32_t y = wadd(wmul(T[a], s), wmul(T[b], c));
    const int32_t xa = rnd2(x, 12), yb = rnd2(y, 12);
    chk(xa, r, err, AV1G_ERR_B_RANGE);
    chk(yb, r, err, AV1G_ERR_B_RANGE);
    if (flip) { T[a] = yb; T[b] = xa; }
    else      { T[a] = xa; T[b] = yb; }
}
template <int L>
__device__ __forceinline__ void H(int32_t (&T)[L], int a, int b, int flip, int r) {
    if (flip) { const int t = a; a = b; b = t; }
    const int32_t x = T[a], y = T[b];
    const int32_t lo = -(1 << (r - 1)), hi = (1 << (r - 1)) - 1;
    const int32_t s = wadd(x, y), d = wsub(x, y);
    T[a] = s < lo ? lo : (s > hi ? hi : s);
    T[b] = d < lo ? lo : (d > hi ? hi : d);
}

template <int n>
__device__ __forceinline__ void idct(int32_t (&T)[1 << n], int r, uint32_t &err) {
    constexpr int n0 = 1 << n;
    int32_t copyT[n0];
#pragma unroll
    for (int i = 0; i < n0; i++) copyT[i] = T[i];
#pragma unroll
    for (int i = 0; i < n0; i++) T[i] = copyT[brev(n, i)];                                     /* step 1 */
    if constexpr (n == 6) {
#pragma unroll
        for (int i = 0; i < 16; i++) B(T, 32 + i, 63 - i, 63 - 4 * brev(4, i), 0, r, err);
    }
    if constexpr (n >= 5) {
#pragma unroll
        for (int i = 0; i < 8; i++) B(T, 16 + i, 31 - i, 6 + (brev(3, 7 - i) << 3), 0, r, err);
    }
    if constexpr (n == 6) {
#pragma unroll
        for (int i = 0; i < 16; i++) H(T, 32 + i * 2, 33 + i * 2, i & 1, r);
    }
    if constexpr (n >= 4) {
#pragma unroll
        for (int i = 0; i < 4; i++) B(T, 8 + i, 15 - i, 12 + (brev(2, 3 - i) << 4), 0, r, err);
    }
    if constexpr (n >= 5) {
#pragma unroll
        for (int i = 0; i < 8; i++) H(T, 16 + 2 * i, 17 + 2 * i, i & 1, r);
    }
    if constexpr (n == 6) {
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 2; j++) B(T, 62 - i * 4 - j, 33 + i * 4 + j, 60 - 16 * brev(2, i) + 64 * j, 1, r, err);
    }
    if constexpr (n >= 3) {
#pragma unroll
        for (int i = 0; i < 2; i++) B(T, 4 + i, 7 - i, 56 - 32 * i, 0, r, err);
    }
    if constexpr (n >= 4) {
#pragma unroll
        for (int i = 0; i < 4; i++) H(T, 8 + 2 * i, 9 + 2 * i, i & 1, r);
    }
    if constexpr (n >= 5) {
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++) B(T, 30 - 4 * i - j, 17 + 4 * i + j, 24 + (j << 6) + ((1 - i) << 5), 1, r, err);
    }
    if constexpr (n == 6) {
#pragma unroll
        for (int i = 0; i < 8; i++)
#pragma unroll
            for (int j = 0; j < 2; j++) H(T, 32 + i * 4 + j, 35 + i * 4 - j, i & 1, r);
    }
#pragma unroll
    for (int i = 0; i < 2; i++) B(T, 2 * i, 2 * i + 1, 32 + 16 * i, 1 - i, r, err);              /* step 12 */
    if constexpr (n >= 3) {
#pragma unroll
        for (int i = 0; i < 2; i++) H(T, 4 + 2 * i, 5 + 2 * i, i, r);
    }
    if constexpr (n >= 4) {
#pragma unroll
        for (int i = 0; i < 2; i++) B(T, 14 - i, 9 + i, 48 + 64 * i, 1, r, err);
    }
    if constexpr (n >= 5) {
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 2; j++) H(T, 16 + 4 * i + j, 19 + 4 * i - j, i & 1, r);
    }
    if constexpr (n == 6) {
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) B(T, 61 - i * 8 - j, 34 + i * 8 + j, 56 - i * 32 + (j >> 1) * 64, 1, r, err);
    }
#pragma unroll
    for (int i = 0; i < 2; i++) H(T, i, 3 - i, 0, r);                                            /* step 17 */
    if constexpr (n >= 3) B(T, 6, 5, 32, 1, r, err);
    if constexpr (n >= 4) {
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 2; j++) H(T, 8 + 4 * i + j, 11 + 4 * i - j, i, r);
    }
    if constexpr (n >= 5) {
#pragma unroll
        for (int i = 0; i < 4; i++) B(T, 29 - i, 18 + i, 48 + (i >> 1) * 64, 1, r, err);
    }
    if constexpr (n == 6) {
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) H(T, 32 + 8 * i + j, 39 + 8 * i - j, i & 1, r);
    }
    if constexpr (n >= 3) {
#pragma unroll
        for (int i = 0; i < 4; i++) H(T, i, 7 - i, 0, r);                                       /* step 22 */
    }
    if constexpr (n >= 4) {
#pragma unroll
        for (int i = 0; i < 2; i++) B(T, 13 - i, 10 + i, 32, 1, r, err);
    }
    if constexpr (n >= 5) {
#pragma unroll
        for (int i = 0; i < 2; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) H(T, 16 + i * 8 + j, 23 + i * 8 - j, i, r);
    }
    if constexpr (n == 6) {
#pragma unroll
        for (int i = 0; i < 8; i++) B(T, 59 - i, 36 + i, i < 4 ? 48 : 112, 1, r, err);
    }
    if constexpr (n >= 4) {
#pragma unroll
        for (int i = 0; i < 8; i++) H(T, i, 15 - i, 0, r);                                      /* step 26 */
    }
    if constexpr (n >= 5) {
#pragma unroll
        for (int i = 0; i < 4; i++) B(T, 27 - i, 20 + i, 32, 1, r, err);
    }
    if constexpr (n == 6) {
#pragma unroll
        for (int i = 0; i < 8; i++) { H(T, 32 + i, 47 - i, 0, r); H(T, 48 + i, 63 - i, 1, r); }
    }
    if constexpr (n >= 5) {
#pragma unroll
        for (int i = 0; i < 16; i++) H(T, i, 31 - i, 0, r);
    }
    if constexpr (n == 6) {
#pragma unroll
        for (int i = 0; i < 8; i++) B(T, 55 - i, 40 + i, 32, 1, r, err);
    }
    if constexpr (n == 6) {
#pragma unroll
        for (int i = 0; i < 32; i++) H(T, i, 63 - i, 0, r);                                     /* step 31 */
    }
}

template <int n>
__device__ __forceinline__ void iadst_in_perm(int32_t (&T)[1 << n]) {
    constexpr int n0 = 1 << n;
    int32_t copyT[n0];
#pragma unroll
    for (int i = 0; i < n0; i++) copyT[i] = T[i];
#pragma unroll
    for (int i = 0; i < n0; i++) T[i] = copyT[(i & 1) ? (i - 1) : (n0 - i - 1)];
}
template <int n>
__device__ __forceinline__ void iadst_out_perm(int32_t (&T)[1 << n]) {
    constexpr int n0 = 1 << n;
    int32_t copyT[n0];
#pragma unroll
    for (int i = 0; i < n0; i++) copyT[i] = T[i];
#pragma unroll
    for (int i = 0; i < n0; i++) {
        const int a = (i >> 3) & 1;
        const int b = ((i >> 2) & 1) ^ ((i >> 3) & 1);
        const int c = ((i >> 1) & 1) ^ ((i >> 2) & 1);
        const int d = (i & 1) ^ ((i >> 1) & 1);
        const int idx = ((d << 3) | (c << 2) | (b << 1) | a) >> (4 - n);
        T[i] = (i & 1) ? -copyT[idx] : copyT[idx];
    }
}

__device__ __forceinline__ void iadst4(int32_t (&T)[4], int r, uint32_t &err) {
    const int32_t SINPI_1_9 = 1321, SINPI_2_9 = 2482, SINPI_3_9 = 3344, SINPI_4_9 = 3803;
    int32_t s[7], x[4];
    s[0] = wmul(SINPI_1_9, T[0]);
    s[1] = wmul(SINPI_2_9, T[0]);
    s[2] = wmul(SINPI_3_9, T[1]);
    s[3] = wmul(SINPI_4_9, T[2]);
    s[4] = wmul(SINPI_1_9, T[2]);
    s[5] = wmul(SINPI_2_9, T[3]);
    s[6] = wmul(SINPI_4_9, T[3]);
    const int32_t a7 = wsub(T[0], T[2]);
    const int32_t b7 = wadd(a7, T[3]);
    chk(a7, r + 1, err, AV1G_ERR_ADST4_RANGE);
    chk(b7, r, err, AV1G_ERR_ADST4_RANGE);
#pragma unroll
    for (int i = 0; i < 7; i++) chk(s[i], r + 12, err, AV1G_ERR_ADST4_RANGE);
    s[0] = wadd(s[0], s[3]);
    s[1] = wsub(s[1], s[4]);
    s[3] = s[2];
    s[2] = wmul(SINPI_3_9, b7);
    chk(s[0], r + 12, err, AV1G_ERR_ADST4_RANGE);
    chk(s[1], r + 12, err, AV1G_ERR_ADST4_RANGE);
    chk(s[2], r + 12, err, AV1G_ERR_ADST4_RANGE);
    s[0] = wadd(s[0], s[5]);
    s[1] = wsub(s[1], s[6]);
    chk(s[0], r + 12, err, AV1G_ERR_ADST4_RANGE);
    chk(s[1], r + 12, err, AV1G_ERR_ADST4_RANGE);
    x[0] = wadd(s[0], s[3]);
    x[1] = wadd(s[1], s[3]);
    x[2] = s[2];
    x[3] = wadd(s[0], s[1]);
    chk(x[3], r + 12, err, AV1G_ERR_ADST4_RANGE);
    x[3] = wsub(x[3], s[3]);
#pragma unroll
    for (int i = 0; i < 4; i++) chk(x[i], r + 12, err, AV1G_ERR_ADST4_RANGE);
#pragma unroll
    for (int i = 0; i < 4; i++) T[i] = rnd2(x[i], 12);
}
__device__ __forceinline__ void iadst8(int32_t (&T)[8], int r, uint32_t &err) {
    iadst_in_perm<3>(T);
#pragma unroll
    for (int i = 0; i < 4; i++) B(T, 2 * i, 2 * i + 1, 60 - 16 * i, 1, r, err);
#pragma unroll
    for (int i = 0; i < 4; i++) H(T, i, 4 + i, 0, r);
#pragma unroll
    for (int i = 0; i < 2; i++) B(T, 4 + 3 * i, 5 + i, 48 - 32 * i, 1, r, err);
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) H(T, 4 * j + i, 2 + 4 * j + i, 0, r);
#pragma unroll
    for (int i = 0; i < 2; i++) B(T, 2 + 4 * i, 3 + 4 * i, 32, 1, r, err);
    iadst_out_perm<3>(T);
}
__device__ __forceinline__ void iadst16(int32_t (&T)[16], int r, uint32_t &err) {
    iadst_in_perm<4>(T);
#pragma unroll
    for (int i = 0; i < 8; i++) B(T, 2 * i, 2 * i + 1, 62 - 8 * i, 1, r, err);
#pragma unroll
    for (int i = 0; i < 8; i++) H(T, i, 8 + i, 0, r);
#pragma unroll
    for (int i = 0; i < 2; i++) {
        B(T, 8 + 2 * i, 9 + 2 * i, 56 - 32 * i, 1, r, err);
        B(T, 13 + 2 * i, 12 + 2 * i, 8 + 32 * i, 1, r, err);
    }
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) H(T, 8 * j + i, 4 + 8 * j + i, 0, r);
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 2; j++) B(T, 4 + 8 * j + 3 * i, 5 + 8 * j + i, 48 - 32 * i, 1, r, err);
#pragma unroll
    for (int i = 0; i < 2; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) H(T, 4 * j + i, 2 + 4 * j + i, 0, r);
#pragma unroll
    for (int i = 0; i < 4; i++) B(T, 2 + 4 * i, 3 + 4 * i, 32, 1, r, err);
    iadst_out_perm<4>(T);
}
template <int n>
__device__ __forceinline__ void iidentity(int32_t (&T)[1 << n]) {
#pragma unroll
    for (int i = 0; i < (1 << n); i++) {
        // |T| <= 2^17 (rows, 10 bits): T * 11586 < 2^31
        if constexpr (n == 2) T[i] = rnd2(wmul(T[i], 5793), 12);
        else if constexpr (n == 3) T[i] = wmul(T[i], 2);
        else if constexpr (n == 4) T[i] = rnd2(wmul(T[i], 11586), 12);
        else T[i] = wmul(T[i], 4);
    }
}
// av1g_inv_1d for a compile-time size (kinds that do not exist at a size are never requested)
template <int n>
__device__ __forceinline__ void inv_1d(int32_t (&T)[1 << n], int kind, int r, uint32_t &err) {
    if (kind == AV1G_K_DCT) idct<n>(T, r, err);
    else if (kind == AV1G_K_ADST) {
        if constexpr (n == 2) iadst4(T, r, err);
        else if constexpr (n == 3) iadst8(T, r, err);
        else if constexpr (n == 4) iadst16(T, r, err);
    } else {
        if constexpr (n <= 5) iidentity<n>(T);
    }
}

// one row of the 2-D inverse (dequantization + optional rect scaling + 1-D + row shift + clip); returns err.
// BD: dequantization clamp +-2^(7 + BitDepth) and rowClampRange BitDepth + 8; colClampRange is 16 at 8 and
// 10 bits, so the int16 residual buffer and col() are the same for both.
template <int LW, int BD = 8>
__device__ __forceinline__ uint32_t row(int i, int th, int tw, const int16_t *q, int dcQ, int acQ, const uint8_t *qm, int den, bool rect,
                                        int rk, int rowShift, int16_t *res) {
    constexpr int W = 1 << LW;
    int32_t Tv[W];
    int nz = 0;
    const int dsh = den >> 1;   // den = 1, 2, 4: the division of av1g_dequant_coef (a non-negative value) is this shift
#pragma unroll
    for (int j = 0; j < W; j++) {
        int32_t d = 0;
        if (j < 32 && i < th && j < tw) {
            const int idx = i * tw + j;
            if (q[idx]) {   // av1g_dequant_coef<BD>(q[idx], step, den)
                const int32_t dq = q[idx] * av1g_qm_step(idx == 0 ? dcQ : acQ, qm, idx);
                int32_t a = ((dq < 0 ? -dq : dq) & 0xFFFFFF) >> dsh;
                if (dq < 0) a = -a;
                d = av1g_clip3(-(1 << (7 + BD)), (1 << (7 + BD)) - 1, a);
            }
        }
        Tv[j] = d;
        nz |= d;
    }
    if (!nz) {
#pragma unroll
        for (int j = 0; j < W; j++) res[i * W + j] = 0;
        return 0;
    }
    if (rect) {
#pragma unroll
        for (int j = 0; j < W; j++) Tv[j] = rnd2(wmul(Tv[j], 2896), 12);
    }
    uint32_t err = 0;
    inv_1d<LW>(Tv, rk, BD + 8, err);
#pragma unroll
    for (int j = 0; j < W; j++)
        res[i * W + j] = (int16_t)av1g_clip3(-(1 << 15), (1 << 15) - 1, rowShift ? rnd2(Tv[j], rowShift) : Tv[j]);
    return err;
}
// one column of the 2-D inverse (1-D + round + range flag 0x100); returns err
template <int LH>
__device__ __forceinline__ uint32_t col(int j, int W, int ck, int16_t *res) {
    constexpr int H = 1 << LH;
    int32_t Tv[H];
#pragma unroll
    for (int i = 0; i < H; i++) Tv[i] = res[i * W + j];
    uint32_t err = 0;
    inv_1d<LH>(Tv, ck, 16, err);
#pragma unroll
    for (int i = 0; i < H; i++) {
        const int32_t v = rnd2(Tv[i], 4);
        res[i * W + j] = (int16_t)av1g_clip3(-(1 << 15), (1 << 15) - 1, v);
        if (v < -(1 << 15) || v > (1 << 15) - 1) err |= 0x100u;
    }
    return err;
}

}  // namespace rci
