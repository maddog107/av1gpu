// open-loop luma analysis, one warp per (image, block position) of one block shape.
//   1. edges from SOURCE pixels (open loop), spec availability approximated by Z-order
//   2. SATD screening (orthonormal 4x4 Hadamard) of the 13 modes at delta 0, then angle deltas
//      +-1..3 around the two best directional modes -> top-K by SATD + lambda_s * mode bits
//   3. exact-rate RD (float forward transform, dead-zone quantizer, AV1 default-CDF coefficient
//      rate) of the top-K with DCT_DCT, tx-type search for the winner
//   -> GpBlockRes {J, candidates best-first, tx type}
#pragma once
#include "dev_tx.cuh"

template <int W, int H, typename P = uint8_t>
struct ASmem {
    // 64x64 and 32x64 (64-sample columns, DCT_DCT only): the residual is stored by column, RS samples apart (an odd
    // number of 4-byte words: conflict-free), and the transform runs in place (warp_eval_cm): a column's 64 samples (128
    // bytes) are replaced by its 32 outputs, then each row of those by its outputs.  No tmp: 64x64 needs 9.4 instead of
    // 17.4 KB per warp (9 instead of 5 warps per SM), 32x64 14.4 instead of 18.5 KB per CTA (6 instead of 5 CTAs); the
    // levels of 64x64 go in the residual's upper half (columns 32..63, free once the row pass has read them).  (16x64:
    // register-limited, measured no faster this way.)
    static constexpr bool cm = H == 64 && W >= 32;
    static constexpr int RS = H + 2;
    static constexpr bool qInUpper = cm && W == 64;
    // shapes with a 32- or 64-sample side have only DCT_DCT for intra luma (no tx-type search), so every evaluation
    // follows a fresh warp_residual: the quantized levels reuse the residual's storage (the column pass has read all of
    // it before warp_quant_rate_rt writes a level).  Less shared memory per warp: 32x32 fits 3 CTAs per SM instead of 2.
    static constexpr bool qInRes = !cm && (W >= 32 || H >= 32);
    int16_t res[cm ? W * RS : W * H];
    float tmp[cm ? 1 : TxD<W, H>::th * TxD<W, H>::TS];
    int16_t qs[(qInRes || qInUpper) ? 2 : TxD<W, H>::n];
    __device__ __forceinline__ int16_t *q() { return qInUpper ? res + 32 * RS : qInRes ? res : qs; }
    P ea[AV1G_EDGE_LEN], el[AV1G_EDGE_LEN];
    P pa[AV1G_EDGE_LEN], pl[AV1G_EDGE_LEN];   // edge-filter prepared edges (RD stage)
    int up[2];
    float satd[32];
    int cand[32];
};

// sum |coefficients| of the 4x4 Hadamard transform of d (the SATD is 0.25 x this; at most 64 x the largest |d|, so
// the sums over a 64x64 block stay exact in float)
__device__ __forceinline__ int hadamard4_sum(int *d) {
    // rows
#pragma unroll
    for (int i = 0; i < 4; i++) {
        int *r = d + 4 * i;
        int s0 = r[0] + r[1], s1 = r[0] - r[1], s2 = r[2] + r[3], s3 = r[2] - r[3];
        r[0] = s0 + s2; r[1] = s1 + s3; r[2] = s0 - s2; r[3] = s1 - s3;
    }
    int sum = 0;
#pragma unroll
    for (int j = 0; j < 4; j++) {
        int s0 = d[j] + d[4 + j], s1 = d[j] - d[4 + j], s2 = d[8 + j] + d[12 + j], s3 = d[8 + j] - d[12 + j];
        sum += abs(s0 + s2) + abs(s1 + s3) + abs(s0 - s2) + abs(s1 - s3);
    }
    return sum;
}
__device__ __forceinline__ float hadamard4_abs(int *d) { return (float)hadamard4_sum(d) * 0.25f; }

__device__ __forceinline__ int pangle_of(int mode, int delta) {
    return av1g_is_directional(mode) ? AV1G_T(Mode_To_Angle)[mode] + delta * AV1G_ANGLE_STEP : 0;
}

// prediction of the 4x4 sub-block at (sy, sx) for candidate c (edges A/L, dc value dcv) into p[i * 4 + j]: the
// values of av1g_pred_pixel_up (spec 7.11.2; upA / upL: upsampling flags of edge-filtered edges), with the
// per-row / per-column terms and the mode switch hoisted
template <typename P>
__device__ __forceinline__ void pred_sub4(int c, int sy, int sx, int lw, int lh, const P *A, const P *L, int dcv, int *p,
                                          int upA = 0, int upL = 0) {
    const int m = cand_mode(c);
    const int w = 1 << lw, h = 1 << lh;
    if (av1g_is_directional(m)) {
        const int pAngle = pangle_of(m, cand_delta(c));
        if (pAngle == 90 || pAngle == 180) {
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) p[i * 4 + j] = pAngle == 90 ? A[sx + j] : L[sy + i];
        } else if (pAngle < 90) {
            const int dx = AV1G_T(Dr_Intra_Derivative)[pAngle], maxBaseX = (w + h - 1) << upA;
#pragma unroll
            for (int i = 0; i < 4; i++) {
                const int idx = (sy + i + 1) * dx, b = (idx >> (6 - upA)) + (sx << upA), s = ((idx << upA) >> 1) & 0x1F;
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    const int k = b + (j << upA);
                    p[i * 4 + j] = k < maxBaseX ? (A[k] * (32 - s) + A[k + 1] * s + 16) >> 5 : A[maxBaseX];
                }
            }
        } else if (pAngle > 180) {
            const int dy = AV1G_T(Dr_Intra_Derivative)[270 - pAngle];
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int idx = (sx + j + 1) * dy, b = (idx >> (6 - upL)) + (sy << upL), s = ((idx << upL) >> 1) & 0x1F;
#pragma unroll
                for (int i = 0; i < 4; i++) {
                    const int k = b + (i << upL);
                    p[i * 4 + j] = (L[k] * (32 - s) + L[k + 1] * s + 16) >> 5;
                }
            }
        } else {
            const int dx = AV1G_T(Dr_Intra_Derivative)[180 - pAngle], dy = AV1G_T(Dr_Intra_Derivative)[pAngle - 90];
#pragma unroll
            for (int i = 0; i < 4; i++)
#pragma unroll
                for (int j = 0; j < 4; j++) {
                    int idx = ((sx + j) << 6) - (sy + i + 1) * dx, up = upA;
                    const P *E = A;
                    if ((idx >> (6 - upA)) < -(1 << upA)) { idx = ((sy + i) << 6) - (sx + j + 1) * dy; up = upL; E = L; }
                    const int b = idx >> (6 - up), s = ((idx * (1 << up)) >> 1) & 0x1F;
                    p[i * 4 + j] = (E[b] * (32 - s) + E[b + 1] * s + 16) >> 5;
                }
        }
        return;
    }
    if (m == AV1G_DC_PRED) {
#pragma unroll
        for (int k = 0; k < 16; k++) p[k] = dcv;
        return;
    }
    if (m == AV1G_PAETH_PRED) {
        const int tl = A[-1];
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++) {
                const int a = A[sx + j], l = L[sy + i], base = a + l - tl;
                const int pL = abs(base - l), pT = abs(base - a), pTL = abs(base - tl);
                p[i * 4 + j] = (pL <= pT && pL <= pTL) ? l : pT <= pTL ? a : tl;
            }
        return;
    }
    // SMOOTH, SMOOTH_V, SMOOTH_H: s = v + h with the vertical / horizontal terms of the modes that use them
    const uint8_t *wx = av1g_sm_weights(lw) + sx, *wy = av1g_sm_weights(lh) + sy;
    const int bl = L[h - 1], tr = A[w - 1];
    const int sh = m == AV1G_SMOOTH_PRED ? 9 : 8, useV = m != AV1G_SMOOTH_H_PRED, useH = m != AV1G_SMOOTH_V_PRED;
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const int v = useV ? wy[i] * A[sx + j] + (256 - wy[i]) * bl : 0;
            const int hz = useH ? wx[j] * L[sy + i] + (256 - wx[j]) * tr : 0;
            p[i * 4 + j] = (v + hz + (1 << (sh - 1))) >> sh;
        }
}

// index of the smallest (v, i) over the warp (-1 if every v is GP_INF): the pick of a serial scan in index order
// that keeps the first of equal values
__device__ __forceinline__ int warp_argmin(float v, int i) {
    for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, v, o);
        const int oi = __shfl_xor_sync(0xffffffffu, i, o);
        if (ov < v || (ov == v && oi < i)) { v = ov; i = oi; }
    }
    return v < GP_INF ? i : -1;
}

// av1g_dc_value with the edge sums taken over the warp (all lanes get it)
template <int BD, typename P>
__device__ __forceinline__ int warp_dc_value(const P *A, const P *L, int haveLeft, int haveAbove, int lw, int lh) {
    const int lane = threadIdx.x & 31, w = 1 << lw, h = 1 << lh;
    int sa = 0, sl = 0;
    for (int k = lane; k < w; k += 32) sa += A[k];
    for (int k = lane; k < h; k += 32) sl += L[k];
    sa = warp_sumi(sa);
    sl = warp_sumi(sl);
    if (haveLeft && haveAbove) return (sa + sl + ((w + h) >> 1)) / (w + h);
    if (haveLeft) return av1g_clip1<BD>((sl + (h >> 1)) >> lh);
    if (haveAbove) return av1g_clip1<BD>((sa + (w >> 1)) >> lw);
    return 1 << (BD - 1);
}

// 4 consecutive source samples (row offsets of the 4x4 sub-blocks are 4-sample aligned, luma rows 8-byte aligned)
__device__ __forceinline__ void load4(const uint8_t *s, int *v) {
    const uint32_t w = *(const uint32_t *)s;
    v[0] = w & 255; v[1] = (w >> 8) & 255; v[2] = (w >> 16) & 255; v[3] = w >> 24;
}
__device__ __forceinline__ void load4(const uint16_t *s, int *v) {
    const uint2 w = *(const uint2 *)s;
    v[0] = w.x & 0xFFFF; v[1] = w.x >> 16; v[2] = w.y & 0xFFFF; v[3] = w.y >> 16;
}

// SATD of candidates cands[c0..c1) (all 4x4 sub-blocks), accumulated into satd[]
template <int W, int H, typename P>
__device__ __forceinline__ void warp_screen(const P *src, int sp, const int *cands, int c0, int c1, float *satd,
                                            const P *A, const P *L, int dcv, int sub = 1) {
    typedef TxD<W, H> T;
    const int lane = threadIdx.x & 31;
    constexpr int nsx = W / 4, nsub0 = (W / 4) * (H / 4);
    // sub == 2: checkerboard of 4x4 sub-blocks (nsx is even for every shape >= 256 pixels)
    const int nsub = sub == 2 ? nsub0 / 2 : nsub0;
    const int items = (c1 - c0) * nsub;
    // nsub is a power of two: a pass of 32 items covers one candidate, or 32 / nsub candidates in aligned groups of
    // nsub lanes; the group's integer sum goes to satd (exact, so the order of the additions does not matter)
    for (int it0 = 0; it0 < items; it0 += 32) {
        const int it = it0 + lane;
        const int ci = c0 + it / nsub;
        int t = 0;
        if (it < items) {
            int sb = it % nsub;
            if (sub == 2) { const int r = (2 * sb) / nsx; sb = 2 * sb + (r & 1); }
            const int sx = (sb % nsx) * 4, sy = (sb / nsx) * 4;
            int d[16], p[16];
            pred_sub4(cands[ci], sy, sx, T::lw, T::lh, A, L, dcv, p);
#pragma unroll
            for (int i = 0; i < 4; i++) load4(src + (sy + i) * sp + sx, d + 4 * i);
#pragma unroll
            for (int k = 0; k < 16; k++) d[k] -= p[k];
            t = hadamard4_sum(d);
        }
        for (int o = 16; o > 0; o >>= 1)
            if (o < nsub) t += __shfl_xor_sync(0xffffffffu, t, o);
        if (it < items && (lane & (nsub - 1)) == 0) satd[ci] += (float)t * 0.25f;
    }
    __syncwarp();
}

// warp-parallel av1g_edge_filter / av1g_edge_upsample / av1g_edge_prepare (dsp.h), the same integer arithmetic
// (the k_recon.cuh versions, under their own names here): every lane reads its outputs' inputs (the unfiltered edge)
// into registers, __syncwarp, then writes.  Called by full warps with warp-uniform arguments.
template <typename P>
__device__ __forceinline__ void an_edge_filter(P *buf, int sz, int strength) {   // sz <= 129: outputs 1 .. 128
    if (!strength) return;
    const int lane = threadIdx.x & 31;
    const uint8_t *K = AV1G_T(Intra_Edge_Kernel)[strength - 1];
    int v[4];
#pragma unroll
    for (int k = 0; k < 4; k++) {
        const int i = lane + 1 + 32 * k;
        int s = 0;
        if (i < sz) {
#pragma unroll
            for (int j = 0; j < AV1G_INTRA_EDGE_TAPS; j++) s += K[j] * buf[av1g_clip3(0, sz - 1, i - 2 + j) - 1];
        }
        v[k] = (s + 8) >> 4;
    }
    __syncwarp();
#pragma unroll
    for (int k = 0; k < 4; k++)
        if (lane + 1 + 32 * k < sz) buf[lane + 32 * k] = (P)v[k];
    __syncwarp();
}
template <int BD, typename P>
__device__ __forceinline__ void an_edge_upsample(P *buf, int numPx) {   // numPx <= 16 (upsampling: w + h <= 16)
    const int lane = threadIdx.x & 31;
    // dup[0] = buf[-1], dup[k] = buf[k - 2] (k = 1 .. numPx + 1), dup[numPx + 2] = buf[numPx - 1]
    auto dup = [&](int k) -> int { return k == 0 ? buf[-1] : k == numPx + 2 ? buf[numPx - 1] : buf[k - 2]; };
    int s = 0, e = 0;
    if (lane < numPx) {
        s = av1g_clip1<BD>(av1g_round2(-dup(lane) + 9 * dup(lane + 1) + 9 * dup(lane + 2) - dup(lane + 3), 4));
        e = dup(lane + 2);
    }
    const P c = buf[-1];
    __syncwarp();
    if (lane < numPx) { buf[2 * lane - 1] = (P)s; buf[2 * lane] = (P)e; }
    if (lane == 0) buf[-2] = c;
    __syncwarp();
}
template <int BD, typename P>
__device__ __forceinline__ void an_edge_prepare(int pAngle, int log2W, int log2H, int filterType, int haveLeft, int haveAbove, int x,
                                                int y, int maxX, int maxY, P *above, P *left, int &upA, int &upL) {
    const int w = 1 << log2W, h = 1 << log2H;
    if (pAngle != 90 && pAngle != 180) {
        if (pAngle > 90 && pAngle < 180 && (w + h) >= 24) {
            if ((threadIdx.x & 31) == 0) {
                const P c = (P)av1g_round2(left[0] * 5 + above[-1] * 6 + above[0] * 5, 4);
                left[-1] = c;
                above[-1] = c;
            }
            __syncwarp();
        }
        if (haveAbove)
            an_edge_filter(above, (w < maxX - x + 1 ? w : maxX - x + 1) + (pAngle < 90 ? h : 0) + 1,
                           av1g_edge_strength(w, h, filterType, pAngle - 90));
        if (haveLeft)
            an_edge_filter(left, (h < maxY - y + 1 ? h : maxY - y + 1) + (pAngle > 180 ? w : 0) + 1,
                           av1g_edge_strength(w, h, filterType, pAngle - 180));
    }
    upA = av1g_edge_use_upsample(w, h, filterType, pAngle - 90);
    if (upA) an_edge_upsample<BD>(above, w + (pAngle < 90 ? h : 0));
    upL = av1g_edge_use_upsample(w, h, filterType, pAngle - 180);
    if (upL) an_edge_upsample<BD>(left, h + (pAngle > 180 ? w : 0));
}

// residual of candidate c into res (W x H; by column, ASmem::RS apart, for ASmem::cm), returns sum r^2 in raster lane
// order (all lanes; 0 without a 64-sample side, where EvalIn::resEnergy is unused).  With the intra edge filter,
// directional candidates use edges prepared by the warp into pa/pl (filter type 0: neighbour modes unknown).
template <int W, int H, int BD = 8, typename P>
__device__ __forceinline__ float warp_residual(const P *src, int sp, int c, int16_t *res, const P *A,
                                               const P *L, int dcv, P *pa = nullptr, P *pl = nullptr,
                                               int *up = nullptr, int edgeFilter = 0, int haveL = 0, int haveA = 0, int x = 0,
                                               int y = 0, int maxX = 0, int maxY = 0) {
    typedef TxD<W, H> T;
    const int lane = threadIdx.x & 31;
    const int m = cand_mode(c);
    const bool ef = edgeFilter && av1g_is_directional(m);
    int upA = 0, upL = 0;
    if (ef) {
        for (int k = lane; k < AV1G_EDGE_LEN; k += 32) { pa[k] = A[k - AV1G_EDGE_OFF]; pl[k] = L[k - AV1G_EDGE_OFF]; }
        __syncwarp();
        an_edge_prepare<BD>(pangle_of(m, cand_delta(c)), T::lw, T::lh, 0, haveL, haveA, x, y, maxX, maxY, pa + AV1G_EDGE_OFF,
                            pl + AV1G_EDGE_OFF, upA, upL);
    }
    const P *EA = ef ? pa + AV1G_EDGE_OFF : A, *EL = ef ? pl + AV1G_EDGE_OFF : L;
    // res may hold the previous evaluation's quantized levels (ASmem::qInRes, qInUpper) or transform outputs
    // (ASmem::cm): every lane is done reading them
    __syncwarp();
    constexpr bool cm = H == 64 && W >= 32;   // ASmem::cm
    constexpr int RS = H + 2;
    constexpr int nsx = W / 4;
    for (int sb = lane; sb < (W / 4) * (H / 4); sb += 32) {
        const int sx = (sb % nsx) * 4, sy = (sb / nsx) * 4;
        int pr[16];
        pred_sub4(c, sy, sx, T::lw, T::lh, EA, EL, dcv, pr, upA, upL);
#pragma unroll
        for (int i = 0; i < 4; i++)
#pragma unroll
            for (int j = 0; j < 4; j++)
                res[cm ? (sx + j) * RS + sy + i : (sy + i) * W + sx + j] = (int16_t)((int)src[(sy + i) * sp + sx + j] - pr[i * 4 + j]);
    }
    __syncwarp();
    if (W < 64 && H < 64) return 0.f;
    float e = 0.f;
    for (int p = lane; p < W * H; p += 32) {
        const int r = res[cm ? (p % W) * RS + p / W : p];
        e += (float)(r * r);
    }
    return warp_sum(e);
}

// warp_eval for the column-major residual of ASmem::cm (H == 64, DCT_DCT: no flips, no identity kind), the transform
// in place: the column pass replaces column j's 64 samples by its 32 outputs (the same 128 bytes, read by the lanes of
// that column before any of them writes), the row pass row i's 64 inputs by its tw outputs (one lane per row), all at
// column stride RS / 2 floats.  The same fmaf chain per output and the same quantization as warp_eval: identical
// results.  q: the levels (natural order, tw x 32), outside the first tw columns.
template <int W, int H, int BD = 8>
__device__ __forceinline__ void warp_eval_cm(int16_t *res, int16_t *q, const EvalIn &e, EvalOut &o) {
    static_assert(H == 64 && W >= 32, "column-major storage: 64-sample columns, one lane per column");
    constexpr int TH = 32, TW = W > 32 ? 32 : W, RS = H + 2, RF = RS / 2;
    const int lane = threadIdx.x & 31;
    float *col = (float *)res;   // column j's outputs: col[j * RF + i]
    {
        // warp_fwd_ct's column pass with one item per column (GC = 1 for W >= 32): lane j owns columns j and j + 32
        const float *FcT = g_fwdT + g_fwdOff[AV1G_K_DCT][log2i(H)];
        for (int j = lane; j < W; j += 32) {
            const int16_t *x2 = res + j * RS;
            float acc[TH];
#pragma unroll
            for (int i = 0; i < TH; i++) acc[i] = 0.f;
            for (int k = 0; k < H; k++) {
                const float x = (float)x2[k];
                const float4 *f4 = (const float4 *)(FcT + k * H);
#pragma unroll
                for (int i4 = 0; i4 < TH / 4; i4++) {
                    const float4 f = __ldg(&f4[i4]);
                    acc[4 * i4 + 0] = fmaf(f.x, x, acc[4 * i4 + 0]);
                    acc[4 * i4 + 1] = fmaf(f.y, x, acc[4 * i4 + 1]);
                    acc[4 * i4 + 2] = fmaf(f.z, x, acc[4 * i4 + 2]);
                    acc[4 * i4 + 3] = fmaf(f.w, x, acc[4 * i4 + 3]);
                }
            }
#pragma unroll
            for (int i = 0; i < TH; i++) col[j * RF + i] = acc[i];
        }
    }
    __syncwarp();
    {
        const float *FrT = g_fwdT + g_fwdOff[AV1G_K_DCT][log2i(W)];
        const float scale = g_txScale[e.txSz];
        const int i = lane;   // one row per lane (TH == 32)
        float acc[TW];
#pragma unroll
        for (int jj = 0; jj < TW; jj++) acc[jj] = 0.f;
        for (int k = 0; k < W; k++) {
            const float x = col[k * RF + i];
            const float4 *f4 = (const float4 *)(FrT + k * W);
#pragma unroll
            for (int j4 = 0; j4 < TW / 4; j4++) {
                const float4 f = __ldg(&f4[j4]);
                acc[4 * j4 + 0] = fmaf(f.x, x, acc[4 * j4 + 0]);
                acc[4 * j4 + 1] = fmaf(f.y, x, acc[4 * j4 + 1]);
                acc[4 * j4 + 2] = fmaf(f.z, x, acc[4 * j4 + 2]);
                acc[4 * j4 + 3] = fmaf(f.w, x, acc[4 * j4 + 3]);
            }
        }
#pragma unroll
        for (int jj = 0; jj < TW; jj++) col[jj * RF + i] = acc[jj] * scale;   // only this lane's row
    }
    __syncwarp();
    // warp_quant_rate_rt with the coefficient (i, j) at col[j * RF + i]
    constexpr int n = TW * TH, lt = TW == 32 ? 5 : TW == 16 ? 4 : 3;
    const int den = av1g_dq_denom(e.txSz);
    float dist = 0.f, kept = 0.f;
    for (int idx = lane; idx < n; idx += 32) {
        const int i = idx >> lt, j = idx & (TW - 1);
        const float v = col[j * RF + i];
        const int step = av1g_qm_step(idx == 0 ? e.dcQ : e.acQ, e.qm, idx);
        const float a = fabsf(v) * (float)den / (float)step;
        const float wr = e.qm ? 32.f / (float)e.qm[idx] : 1.f;
        int l = a < e.zbin ? 0 : (int)(a + (e.qmRnd ? e.rnd * wr : e.rnd));
        l = min(l, 16383);
        const int qv = v < 0 ? -l : l;
        q[idx] = (int16_t)qv;
        const float dq = l ? (float)av1g_dequant_coef<BD>(qv, step, den) : 0.f;
        const float d = v - dq;
        dist = fmaf(d * d, e.qmDist ? wr * wr : 1.f, dist);
        kept = fmaf(v, v, kept);
    }
    dist = warp_sum(dist);
    __syncwarp();
    const uint16_t *scan = av1g_get_scan(e.txSz, e.txType);
    int last = 0;
    for (int c = lane; c < n; c += 32)
        if (q[scan[c]]) last = c + 1;
    const int eob = warp_max(last);
    float bits = lane == 0 ? base_bits(e, eob) : 0.f;
    if (eob > 0) {
        const int txClass = av1g_tx_class(e.txType);
        for (int c = lane; c < eob; c += 32) bits += coeff_bits(c, eob, q, scan, e.txSz, txClass, e.plane > 0, e.dcCtx, e.C);
    }
    bits = warp_sum(bits);
    const float g = g_txg[e.txSz][e.txType];
    float D = g * dist;
    kept = warp_sum(kept);
    D += fmaxf(0.f, e.resEnergy - g * kept);
    o.D = D;
    o.R = bits;
    o.eob = eob;
}

// Threads per SM that k_analysis<W, H> is compiled for (__launch_bounds__: registers <= 64 K / budget).  Measured per
// shape against 512 (128 registers) on an RTX PRO 2000 Blackwell, kernel time with the default preset / --preset base:
// 768 (80 registers, 6 CTAs of 4 warps) for shapes of up to 256 pixels, whose shared memory allows 6 CTAs (16x8 -12 /
// -11 %, 8x16 -10 / -9 %, 16x16 -6 / -9 %, 8x32 -3 / -8 %, 32x8 -2 / -5 %); 640 for 16x32 (96 registers, 5 CTAs: -3 /
// -1 %; 32x16 gained nothing that way); 384 for 32x32 (168 registers, 3 CTAs, the limit of its 28 KB anyway: -7 / -7 %).
template <int W, int H>
constexpr int an_thread_budget() {
    return W * H <= 256 ? 768 : (W == 16 && H == 32) ? 640 : (W == 32 && H == 32) ? 384 : 512;
}

// BD: bit depth of the batch (distortions and lambda in native units, see gp_lambda)
template <int W, int H, int BS, int NW, int BD = 8>
__global__ void __launch_bounds__(NW * 32, an_thread_budget<W, H>() / (NW * 32)) k_analysis(GpBatch B, GpParams P, uint32_t nTotal) {
    typedef TxD<W, H> T;
    typedef av1g_pixel_t(BD) Px;
    __shared__ ASmem<W, H, Px> smem[NW];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    ASmem<W, H, Px> &S = smem[warp];
    Px *A = S.ea + AV1G_EDGE_OFF, *L = S.el + AV1G_EDGE_OFF;
    const uint32_t *starts = B.shImgStart + (size_t)BS * (B.nimg + 1);
    constexpr int txSz = W == H ? (W == 4 ? AV1G_TX_4X4 : W == 8 ? AV1G_TX_8X8 : W == 16 ? AV1G_TX_16X16 : W == 32 ? AV1G_TX_32X32 : AV1G_TX_64X64)
                       : (W == 4 && H == 8) ? AV1G_TX_4X8 : (W == 8 && H == 4) ? AV1G_TX_8X4 : (W == 8 && H == 16) ? AV1G_TX_8X16
                       : (W == 16 && H == 8) ? AV1G_TX_16X8 : (W == 16 && H == 32) ? AV1G_TX_16X32 : (W == 32 && H == 16) ? AV1G_TX_32X16
                       : (W == 32 && H == 64) ? AV1G_TX_32X64 : (W == 64 && H == 32) ? AV1G_TX_64X32 : (W == 4 && H == 16) ? AV1G_TX_4X16
                       : (W == 16 && H == 4) ? AV1G_TX_16X4 : (W == 8 && H == 32) ? AV1G_TX_8X32 : (W == 32 && H == 8) ? AV1G_TX_32X8
                       : (W == 16 && H == 64) ? AV1G_TX_16X64 : AV1G_TX_64X16;
    const int set = av1g_get_tx_set_intra(txSz, 0);

    for (uint32_t gb = blockIdx.x * NW + warp; gb < nTotal; gb += gridDim.x * NW) {
        // image of this block (binary search over the per-image starts of this shape)
        int lo = 0, hi = B.nimg - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (starts[mid] <= gb) lo = mid; else hi = mid - 1;
        }
        const GpImage &I = B.img[lo];
        const uint32_t li = gb - starts[lo];
        const int nbx = I.shNbx[BS];
        const int x = (int)(li % nbx) * W, y = (int)(li / nbx) * H;
        GpBlockRes &out = B.blk[I.shOff[BS] + li];
        if (x >= I.MiCols * 4 || y >= I.MiRows * 4) {
            if (lane == 0) { out.J = GP_INF; out.ncand = 0; }
            continue;
        }
        const TileBox tb = tile_of(I, x, y);
        const int haveLeft = x > tb.x0, haveAbove = y > tb.y0;
        const int haveAR = ol_decoded(x + W, y - 1, x, y, 0, tb);
        const int haveBL = ol_decoded(x - 1, y + H, x, y, 0, tb);
        const int maxX = I.MiCols * 4 - 1, maxY = I.MiRows * 4 - 1;
        const Px *src = gp_src<BD>(I, 0) + (size_t)y * I.spitch[0] + x;
        const int sp = I.spitch[0];
        const int qidx = sb_qindex(I, x, y);
        const float lambda = gp_lambda<BD>(qidx, P.lambdaMul) * ssim_scale(I, x, y, W, H);
        const float lamS = P.satdLambda * sqrtf(lambda);
        const float *C = g_cost[I.qctx];
        if (prune_flat<BD>(I, P, x, y, W, H, qidx)) {
            if (lane == 0) { out.J = GP_INF; out.ncand = 0; }
            __syncwarp();
            continue;
        }
        if (P.pruneVar > 0.f && W * H >= P.pruneMinArea) {
            float s1 = 0.f, s2 = 0.f;
            for (int p = lane; p < W * H; p += 32) {
                const float v = (float)src[(p / W) * sp + p % W];
                s1 += v;
                s2 += v * v;
            }
            s1 = warp_sum(s1);
            s2 = warp_sum(s2);
            const float var = s2 / (W * H) - (s1 / (W * H)) * (s1 / (W * H));
            const float st = av1g_ac_q<BD>(qidx) / 8.f;
            if (var > P.pruneVar * st * st) {
                if (lane == 0) { out.J = GP_INF; out.ncand = 0; }
                __syncwarp();
                continue;
            }
        }
        build_edges_par<BD>(gp_src<BD>(I, 0), sp, x, y, T::lw, T::lh, haveLeft, haveAbove, haveAR, haveBL, maxX, maxY, A, L, lane, 32);
        __syncwarp();
        const int dcv = warp_dc_value<BD>(A, L, haveLeft, haveAbove, T::lw, T::lh);
        const GpExtCand *ext = (const GpExtCand *)B.extCand[BS];
        int K = min((W * H >= 1024 && P.kLarge > 0) ? P.kLarge : P.K, GP_MAXK);
        int top[GP_MAXK];
        auto mbits = [&](int c) -> float {
            const int m = cand_mode(c);
            float b = g_avgYMode[m];
            if (BS >= AV1G_BLOCK_8X8 && av1g_is_directional(m)) b += C[CID(angle_delta[m - AV1G_V_PRED]) + cand_delta(c) + 3];
            return b;
        };
        if (ext) {
            // candidates from an external screener (optional hook, unused by default), best first
            const GpExtCand *ec = ext + (size_t)(B.extOff[BS * B.nimg + lo] + li) * B.extTopK;
            int n = 0;
            for (int k = 0; k < B.extTopK && n < GP_MAXK; k++) {
                const int m = ec[k].mode;
                if (m >= 13) continue;
                const int d = (BS >= AV1G_BLOCK_8X8 && av1g_is_directional(m)) ? ec[k].delta : 0;
                if (lane == 0) S.cand[n] = cand_pack(m, d);
                top[n] = n;
                n++;
            }
            K = min(K, n);
            __syncwarp();
        } else {
        if (lane < 32) { S.satd[lane] = 0.f; S.cand[lane] = 0; }
        if (lane < 13) S.cand[lane] = cand_pack(lane, 0);
        __syncwarp();
        // ---- screening: 13 modes at delta 0 ----
        const int ssub = (P.screenSub == 2 && W * H >= 256) ? 2 : 1;
        warp_screen<W, H>(src, sp, S.cand, 0, 13, S.satd, A, L, dcv, ssub);
        // screening cost of candidate `lane`
        auto jsOf = [&](int i) -> float { return S.satd[i] + lamS * mbits(S.cand[i]); };
        float js = lane < 13 ? jsOf(lane) : GP_INF;
        const bool allowDelta = P.deltas && BS >= AV1G_BLOCK_8X8 && (P.deltasMaxArea <= 0 || W * H <= P.deltasMaxArea);
        if (allowDelta) {
            // best two directional modes by screening cost
            const float jd = lane >= AV1G_V_PRED && lane <= AV1G_D67_PRED ? js : GP_INF;
            const int b0 = warp_argmin(jd, lane);
            const int b1 = warp_argmin(lane == b0 ? GP_INF : jd, lane);
            if (lane < 12) {
                const int m = lane < 6 ? b0 : b1;
                const int k = lane % 6;
                const int d = k < 3 ? -(k + 1) : (k - 2);
                S.cand[13 + lane] = cand_pack(m, d);
            }
            __syncwarp();
            warp_screen<W, H>(src, sp, S.cand, 13, 25, S.satd, A, L, dcv, ssub);
            if (lane >= 13 && lane < 25) js = jsOf(lane);
        }
        // ---- top-K by screening cost ----
        for (int k = 0; k < K; k++) {
            top[k] = warp_argmin(js, lane);
            if (lane == top[k]) js = GP_INF;
        }
        }
        // ---- RD of the top-K with DCT_DCT ----
        const int hasC = av1g_has_chroma(BS, y >> 2, x >> 2, I.mono, I.ssx, I.ssy);
        const float overhead = (hasC ? P.uvOverheadBits : 0.f) + g_avgSkip0;
        EvalIn e;
        e.txSz = txSz; e.plane = 0;
        e.dcQ = av1g_dc_q<BD>(qidx); e.acQ = av1g_ac_q<BD>(qidx);
        e.rnd = P.rnd; e.zbin = P.zbin;
        e.skipCtx = 0; e.dcCtx = 0; e.reduced = 0; e.C = C;
        e.qmDist = P.qmDist; e.qmRnd = P.qmRnd;
        // one evaluation site (code size): steps 0 .. nK-1 rate the top-K with DCT_DCT, the next ones the other tx
        // types of the set for the winner
        float bestJ = GP_INF;
        int bestK = 0, bestTx = AV1G_DCT_DCT, resK = -1;
        int nK = 0;
        while (nK < K && top[nK] >= 0) nK++;
        const int nSteps = nK + ((P.txSearch && set != AV1G_TX_SET_DCTONLY) ? 15 : 0);
        for (int st = 0; st < nSteps; st++) {
            const int k = st < nK ? st : bestK, t = st < nK ? AV1G_DCT_DCT : st - nK + 1;
            if (t != AV1G_DCT_DCT && !av1g_tx_type_in_set_intra(set, t)) continue;
            const int c = S.cand[top[k]];
            if (k != resK) {
                e.resEnergy = warp_residual<W, H, BD>(src, sp, c, S.res, A, L, dcv, S.pa, S.pl, S.up, I.edgeFilter, haveLeft, haveAbove,
                                                      x, y, maxX, maxY);
                resK = k;
            }
            e.txType = t;
            e.intraDir = cand_mode(c);
            e.qm = av1g_qm(I.usingQm, I.qm[0], 0, txSz, t);
            EvalOut o;
            if constexpr (ASmem<W, H, Px>::cm) warp_eval_cm<W, H, BD>(S.res, S.q(), e, o);
            else warp_eval<W, H, BD>(S.res, S.tmp, S.q(), e, o);
            const float J = o.D + lambda * (o.R + mbits(c) + overhead);
            if (J < bestJ) {
                bestJ = J;
                if (st < nK) bestK = k; else bestTx = t;
            }
        }
        if (lane == 0) {
            out.J = bestJ;
            out.tx = (uint8_t)bestTx;
            int n = 0;
            out.cand[n++] = (uint8_t)S.cand[top[bestK]];
            for (int k = 0; k < K && top[k] >= 0; k++)
                if (k != bestK) out.cand[n++] = (uint8_t)S.cand[top[k]];
            out.ncand = (uint8_t)n;
        }
        __syncwarp();
    }
}
