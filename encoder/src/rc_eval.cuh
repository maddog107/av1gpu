// k_recon's evaluation of one transform block (forward transform, quantization, coefficient rate) by a group of G
// lanes of a warp: the whole warp (G = 32), or one half-warp (G = 16, tx blocks up to 8x8: two tasks per warp).
// The values are those of dev_tx.cuh's warp_eval_rt, bit for bit:
// - forward transform: each output keeps its own fmaf chain in the same k order (warp_fwd_ct's argument, 53e044f);
//   only which lane computes it changes.  Up to 16x16 each column / row is split over GC / GR lanes (F2), as the
//   analysis kernels' warp_fwd_ct does; larger sizes (G = 32 only) run dev_tx.cuh's runtime passes.
// - quantization and rate: a half-warp lane l runs the work of the warp's lanes l and l + 16 ("virtual lanes", each
//   with its own accumulator in the same element order); the first butterfly step of warp_sum (xor 16) is the sum
//   of the two, the rest (xor 8 .. 1) runs on the half-warp's shuffles.  Float addition is commutative, so every
//   lane ends with the bits of warp_sum.  eob is a maximum (order-free).
// - rounding: the quantizer's offset, D and the callers' J = D + lambda * R are written with fmaf / __fmul_rn /
//   __fadd_rn as the compiler contracted them in warp_eval_rt inlined into the caller (one FFMA for J and for the
//   64-point energy term, separate FMUL / FADD everywhere else; SASS of b834f3f), so inlining or outlining rc_eval
//   cannot change a value.
#pragma once
#include "dev_tx.cuh"

// a lane group: lane l of G lanes, mask m (G = 16: the half-warp's 16 lanes)
template <int G>
struct RcLanes {
    int l;
    unsigned m;
    __device__ __forceinline__ void sync() const { __syncwarp(m); }
};
__device__ __forceinline__ RcLanes<32> rc_warp_lanes() { return RcLanes<32>{(int)threadIdx.x & 31, 0xffffffffu}; }
__device__ __forceinline__ RcLanes<16> rc_half_lanes() {
    const int lane = threadIdx.x & 31;
    return RcLanes<16>{lane & 15, 0xffffu << (lane & 16)};
}

// sum over the group with warp_sum's bits; v[k] = virtual lane l + 16 k's partial sum (V = 32 / G of them)
template <int G>
__device__ __forceinline__ float rc_grp_sum(const float *v, const RcLanes<G> &L) {
    float s;
    if constexpr (G == 32) {
        s = v[0] + __shfl_xor_sync(L.m, v[0], 16);
    } else {
        s = v[0] + v[1];
    }
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) s += __shfl_xor_sync(L.m, s, o);
    return s;
}
template <int G>
__device__ __forceinline__ int rc_grp_max(int v, const RcLanes<G> &L) {
#pragma unroll
    for (int o = G / 2; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(L.m, v, o));
    return v;
}

// forward transform of a W x H tx block (W, H <= 16) by G lanes; tmp: th x tw, row stride W + 1 (dev_tx.cuh layout)
template <int W, int H, int G>
__device__ __forceinline__ void rc_fwd_ct(const int16_t *res, float *tmp, int txSz, int txType, const RcLanes<G> &L) {
    constexpr int TH = H, TW = W, TS = W + 1;
    constexpr int GC = TH <= 4 ? 1 : (G / W < TH / 4 ? G / W : TH / 4), OC = TH / GC;
    constexpr int GR = TW <= 4 ? 1 : (G / TH < TW / 4 ? G / TH : TW / 4), OR = TW / GR;
    static_assert(W * GC <= G && TH * GR <= G, "one item per lane");
    const int rk = av1g_row_kind(txType), ck = av1g_col_kind(txType);
    const int fud = av1g_flip_ud(txType), flr = av1g_flip_lr(txType);
    if (L.l < W * GC) {
        const int it = L.l, j = it % W, i0 = (it / W) * OC;
        const int jj = flr ? W - 1 - j : j;
        float acc[OC];
        if (ck == AV1G_K_IDT) {
            const float *Fc = g_fwd + g_fwdOff[ck][TxD<W, H>::lh];
#pragma unroll
            for (int i = 0; i < OC; i++)
                acc[i] = __ldg(&Fc[(i0 + i) * H + i0 + i]) * (float)res[(fud ? H - 1 - (i0 + i) : i0 + i) * W + jj];
        } else {
            const float *FcT = g_fwdT + g_fwdOff[ck][TxD<W, H>::lh];
#pragma unroll
            for (int i = 0; i < OC; i++) acc[i] = 0.f;
#pragma unroll
            for (int k = 0; k < H; k++) {
                const float x = (float)res[(fud ? H - 1 - k : k) * W + jj];
                const float4 *f4 = (const float4 *)(FcT + k * H + i0);
#pragma unroll
                for (int i4 = 0; i4 < OC / 4; i4++) {
                    const float4 f = __ldg(&f4[i4]);
                    acc[4 * i4 + 0] = fmaf(f.x, x, acc[4 * i4 + 0]);
                    acc[4 * i4 + 1] = fmaf(f.y, x, acc[4 * i4 + 1]);
                    acc[4 * i4 + 2] = fmaf(f.z, x, acc[4 * i4 + 2]);
                    acc[4 * i4 + 3] = fmaf(f.w, x, acc[4 * i4 + 3]);
                }
            }
        }
#pragma unroll
        for (int i = 0; i < OC; i++) tmp[(i0 + i) * TS + j] = acc[i];
    }
    L.sync();
    // rows, in place: every lane of a row reads it before any of them writes
    const float scale = g_txScale[txSz];
    const bool act = L.l < TH * GR;
    const int i = L.l % TH, j0 = (L.l / TH) * OR;
    float acc[OR];
    if (act) {
        if (rk == AV1G_K_IDT) {
            const float *Fr = g_fwd + g_fwdOff[rk][TxD<W, H>::lw];
#pragma unroll
            for (int jj = 0; jj < OR; jj++) acc[jj] = __ldg(&Fr[(j0 + jj) * W + j0 + jj]) * tmp[i * TS + j0 + jj];
        } else {
            const float *FrT = g_fwdT + g_fwdOff[rk][TxD<W, H>::lw];
#pragma unroll
            for (int jj = 0; jj < OR; jj++) acc[jj] = 0.f;
#pragma unroll
            for (int k = 0; k < W; k++) {
                const float x = tmp[i * TS + k];
                const float4 *f4 = (const float4 *)(FrT + k * W + j0);
#pragma unroll
                for (int j4 = 0; j4 < OR / 4; j4++) {
                    const float4 f = __ldg(&f4[j4]);
                    acc[4 * j4 + 0] = fmaf(f.x, x, acc[4 * j4 + 0]);
                    acc[4 * j4 + 1] = fmaf(f.y, x, acc[4 * j4 + 1]);
                    acc[4 * j4 + 2] = fmaf(f.z, x, acc[4 * j4 + 2]);
                    acc[4 * j4 + 3] = fmaf(f.w, x, acc[4 * j4 + 3]);
                }
            }
        }
    }
    L.sync();
    if (act) {
#pragma unroll
        for (int jj = 0; jj < OR; jj++) tmp[i * TS + j0 + jj] = acc[jj] * scale;
    }
    L.sync();
}

// runtime size: up to 16x16 the split compile-time passes; 32 and 64 (G = 32 only) dev_tx.cuh's warp_fwd_rt
template <int G>
__device__ __forceinline__ void rc_fwd(const int16_t *res, float *tmp, int W, int H, int txSz, int txType, const RcLanes<G> &L) {
    switch (txSz) {
#define RC_FWD(w, h) rc_fwd_ct<w, h, G>(res, tmp, txSz, txType, L); return
    case AV1G_TX_4X4: RC_FWD(4, 4);
    case AV1G_TX_8X8: RC_FWD(8, 8);
    case AV1G_TX_4X8: RC_FWD(4, 8);
    case AV1G_TX_8X4: RC_FWD(8, 4);
    default: break;
    }
    if constexpr (G == 32) {
        switch (txSz) {
        case AV1G_TX_16X16: RC_FWD(16, 16);
        case AV1G_TX_8X16: RC_FWD(8, 16);
        case AV1G_TX_16X8: RC_FWD(16, 8);
        case AV1G_TX_4X16: RC_FWD(4, 16);
        case AV1G_TX_16X4: RC_FWD(16, 4);
        default: warp_fwd_rt(res, tmp, W, H, txSz, txType); return;
        }
    }
#undef RC_FWD
}

// quantization + distortion + rate of tmp (warp_quant_rate_rt): levels into q, results in every lane of the group.
// G = 16 only for n <= 64 (no 64-point transforms there).
template <int BD, int G>
__device__ __forceinline__ void rc_quant_rate(const float *tmp, int16_t *q, const EvalIn &e, EvalOut &o, int W, int H, const RcLanes<G> &L) {
    constexpr int V = 32 / G;
    const int th = H > 32 ? 32 : H, tw = W > 32 ? 32 : W, TS = W + 1, n = tw * th;
    const int lt = log2i(tw);
    const int den = av1g_dq_denom(e.txSz), dsh = den >> 1;
    const bool k64 = G == 32 && (W == 64 || H == 64);   // the kept energy is only read for 64-point sizes
    float dist[V], kept[V];
#pragma unroll
    for (int v = 0; v < V; v++) { dist[v] = 0.f; kept[v] = 0.f; }
    for (int k = 0; k < n; k += 32) {
#pragma unroll
        for (int v = 0; v < V; v++) {
            const int idx = k + L.l + v * G;
            if (idx < n) {
                const int i = idx >> lt, j = idx & (tw - 1);
                const float x = tmp[i * TS + j];
                const int step = av1g_qm_step(idx == 0 ? e.dcQ : e.acQ, e.qm, idx);
                const float a = fabsf(x) * (float)den / (float)step;
                const float wr = e.qm ? 32.f / (float)e.qm[idx] : 1.f;
                int l = a < e.zbin ? 0 : (int)__fadd_rn(a, e.qmRnd ? __fmul_rn(e.rnd, wr) : e.rnd);
                l = min(l, 16383);
                const int qv = x < 0 ? -l : l;
                q[idx] = (int16_t)qv;
                float dq = 0.f;
                if (l) {   // av1g_dequant_coef<BD>(qv, step, den): its division of a non-negative value by den = 1, 2, 4 is a shift (F6)
                    const int32_t dqi = qv * step;
                    int32_t a2 = ((dqi < 0 ? -dqi : dqi) & 0xFFFFFF) >> dsh;
                    if (dqi < 0) a2 = -a2;
                    dq = (float)av1g_clip3(-(1 << (7 + BD)), (1 << (7 + BD)) - 1, a2);
                }
                const float d = x - dq;
                dist[v] = fmaf(__fmul_rn(d, d), e.qmDist ? __fmul_rn(wr, wr) : 1.f, dist[v]);
                if (k64) kept[v] = fmaf(x, x, kept[v]);
            }
        }
    }
    const float distS = rc_grp_sum<G>(dist, L);
    L.sync();
    const uint16_t *scan = av1g_get_scan(e.txSz, e.txType);
    int last = 0;
    for (int k = 0; k < n; k += 32) {
#pragma unroll
        for (int v = 0; v < V; v++) {
            const int c = k + L.l + v * G;
            if (c < n && q[scan[c]]) last = c + 1;
        }
    }
    const int eob = rc_grp_max<G>(last, L);
    float bits[V];
#pragma unroll
    for (int v = 0; v < V; v++) bits[v] = L.l + v * G == 0 ? base_bits(e, eob) : 0.f;
    if (eob > 0) {
        const int txClass = av1g_tx_class(e.txType);
        for (int k = 0; k < eob; k += 32) {
#pragma unroll
            for (int v = 0; v < V; v++) {
                const int c = k + L.l + v * G;
                if (c < eob) bits[v] += coeff_bits(c, eob, q, scan, e.txSz, txClass, e.plane > 0, e.dcCtx, e.C);
            }
        }
    }
    const float bitsS = rc_grp_sum<G>(bits, L);
    const float g = g_txg[e.txSz][e.txType];
    float D = __fmul_rn(g, distS);
    if constexpr (G == 32) {
        if (W == 64 || H == 64) {
            const float keptS = rc_grp_sum<G>(kept, L);
            D = __fadd_rn(D, fmaxf(0.f, fmaf(-g, keptS, e.resEnergy)));
        }
    }
    o.D = D;
    o.R = bitsS;
    o.eob = eob;
}

// Not inlined: inlined at every call site the evaluation ran with the caller's live state around it, and ptxas spilled
// the quantizer's inputs (EvalIn fields, pointers) and reloaded them from local memory in every loop iteration (the hot
// local-memory traffic of the closed loop).  As its own function it has the register file to itself; the call saves the
// caller's registers once per task.  Arguments by value: a reference to a caller's local would be read from local memory.
template <int BD, int G>
__device__ __noinline__ EvalOut rc_eval(const int16_t *res, float *tmp, int16_t *q, const EvalIn e, int W, int H, const RcLanes<G> L) {
    EvalOut o;
    rc_fwd<G>(res, tmp, W, H, e.txSz, e.txType, L);
    rc_quant_rate<BD, G>(tmp, q, e, o, W, H, L);
    return o;
}
