// open-loop analysis for small block shapes (W*H <= 64: 4x4, 4x8, 8x4, 8x8, 4x16, 16x4),
// ONE THREAD PER BLOCK: the same algorithm as k_analysis (SATD screening of 13 modes + angle deltas,
// top-K, exact-rate RD with DCT_DCT, tx-type search for the winner) in scalar code - no shuffles, no
// syncs, 32 blocks per warp instead of one.
#pragma once
#include "dev_tx.cuh"
#include "k_analysis.cuh"

template <int W, int H>
struct SmallTx {
    static constexpr int lw = TxD<W, H>::lw, lh = TxD<W, H>::lh;
    static constexpr int n = W * H;
};

// scalar forward transform: out[i][j] (H x W, row-major, no 32-clipping needed for these sizes).  The passes take
// the transposed matrices four outputs per load; each output keeps the fmaf order of the matrix product.
template <int W, int H>
__device__ __forceinline__ void fwd_scalar(const int16_t *res, float *out, int txSz, int txType) {
    const int rk = av1g_row_kind(txType), ck = av1g_col_kind(txType);
    const int fud = av1g_flip_ud(txType), flr = av1g_flip_lr(txType);
    const float *Fc = g_fwd + g_fwdOff[ck][SmallTx<W, H>::lh];
    const float *Fr = g_fwd + g_fwdOff[rk][SmallTx<W, H>::lw];
    const float *FcT = g_fwdT + g_fwdOff[ck][SmallTx<W, H>::lh];
    const float *FrT = g_fwdT + g_fwdOff[rk][SmallTx<W, H>::lw];
    float tmp[H * W];
    // columns
#pragma unroll
    for (int j = 0; j < W; j++) {
        const int jj = flr ? W - 1 - j : j;
        float x[H];
#pragma unroll
        for (int k = 0; k < H; k++) x[k] = (float)res[(fud ? H - 1 - k : k) * W + jj];
        if (ck == AV1G_K_IDT) {
#pragma unroll
            for (int i = 0; i < H; i++) tmp[i * W + j] = __ldg(&Fc[i * H + i]) * x[i];
        } else {
            float acc[H];
#pragma unroll
            for (int i = 0; i < H; i++) acc[i] = 0.f;
#pragma unroll
            for (int k = 0; k < H; k++) {
                const float4 *f4 = (const float4 *)(FcT + k * H);   // FcT[k][i] = Fc[i][k]
#pragma unroll
                for (int i4 = 0; i4 < H / 4; i4++) {
                    const float4 f = __ldg(&f4[i4]);
                    acc[4 * i4 + 0] = fmaf(f.x, x[k], acc[4 * i4 + 0]);
                    acc[4 * i4 + 1] = fmaf(f.y, x[k], acc[4 * i4 + 1]);
                    acc[4 * i4 + 2] = fmaf(f.z, x[k], acc[4 * i4 + 2]);
                    acc[4 * i4 + 3] = fmaf(f.w, x[k], acc[4 * i4 + 3]);
                }
            }
#pragma unroll
            for (int i = 0; i < H; i++) tmp[i * W + j] = acc[i];
        }
    }
    // rows
    const float scale = g_txScale[txSz];
#pragma unroll
    for (int i = 0; i < H; i++) {
        if (rk == AV1G_K_IDT) {
#pragma unroll
            for (int j = 0; j < W; j++) out[i * W + j] = __ldg(&Fr[j * W + j]) * tmp[i * W + j] * scale;
        } else {
            float acc[W];
#pragma unroll
            for (int j = 0; j < W; j++) acc[j] = 0.f;
#pragma unroll
            for (int k = 0; k < W; k++) {
                const float4 *f4 = (const float4 *)(FrT + k * W);   // FrT[k][j] = Fr[j][k]
#pragma unroll
                for (int j4 = 0; j4 < W / 4; j4++) {
                    const float4 f = __ldg(&f4[j4]);
                    acc[4 * j4 + 0] = fmaf(f.x, tmp[i * W + k], acc[4 * j4 + 0]);
                    acc[4 * j4 + 1] = fmaf(f.y, tmp[i * W + k], acc[4 * j4 + 1]);
                    acc[4 * j4 + 2] = fmaf(f.z, tmp[i * W + k], acc[4 * j4 + 2]);
                    acc[4 * j4 + 3] = fmaf(f.w, tmp[i * W + k], acc[4 * j4 + 3]);
                }
            }
#pragma unroll
            for (int j = 0; j < W; j++) out[i * W + j] = acc[j] * scale;
        }
    }
}

template <int W, int H, int BD = 8>
__device__ __forceinline__ EvalOut eval_scalar(const int16_t *res, const EvalIn &e) {
    constexpr int n = W * H;
    float c[n];
    int16_t q[n];
    fwd_scalar<W, H>(res, c, e.txSz, e.txType);
    const int den = av1g_dq_denom(e.txSz);
    float dist = 0.f;
    for (int idx = 0; idx < n; idx++) {
        const float v = c[idx];
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
    }
    const uint16_t *scan = av1g_get_scan(e.txSz, e.txType);
    int eob = 0;
    for (int k = n - 1; k >= 0; k--)
        if (q[scan[k]]) { eob = k + 1; break; }
    float bits = base_bits(e, eob);
    if (eob > 0) {
        const int txClass = av1g_tx_class(e.txType);
        for (int k = 0; k < eob; k++) bits += coeff_bits(k, eob, q, scan, e.txSz, txClass, 0, e.dcCtx, e.C);
    }
    EvalOut o;
    o.D = g_txg[e.txSz][e.txType] * dist;
    o.R = bits;
    o.eob = eob;
    return o;
}

// MODE as k_analysis: 0 the default, 1 the opt-in variant (X: block list)
template <int W, int H, int BS, int BD, int MODE = 0>
__device__ __forceinline__ void analysis_small(const GpBatch &B, const GpParams &P, uint32_t nTotal, const GpLive &LC, const GpAnX &X) {
    static_assert(!an_cellwise<W, H>() || an_live_filled(BS), "k_live16 must fill this cell-wise shape (AN_LIVE_SHAPES)");
    typedef SmallTx<W, H> T;
    typedef av1g_pixel_t(BD) Px;
    uint32_t gb = blockIdx.x * blockDim.x + threadIdx.x;
    if (gb >= nTotal) return;
    // a block list (k_an_list) holds only blocks inside the frame that prune_flat keeps; nTotal is its worst case
    const bool fromList = MODE != 0 && X.list;
    if constexpr (MODE != 0)
        if (fromList) {
            if (gb >= __ldg(X.count)) return;
            gb = __ldg(X.list + gb);
        }
    // with a live-cell list, thread gb takes block gb % bpc of live cell gb / bpc (a warp: 2 to 8 whole cells, all of
    // them kept by prune_flat), else block gb of the shape's raster grid
    const bool cellwise = an_cellwise<W, H>() && LC.cell;
    int lo;
    uint32_t li;
    int x, y;
    if constexpr (an_cellwise<W, H>()) {
        if (cellwise && !live_block<W, H>(B, LC, BS, gb, lo, li, x, y)) return;
    }
    if (!cellwise) {
        const uint32_t *starts = B.shImgStart + (size_t)BS * (B.nimg + 1);
        lo = 0;
        int hi = B.nimg - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (starts[mid] <= gb) lo = mid; else hi = mid - 1;
        }
        li = gb - starts[lo];
        const int nbx = B.img[lo].shNbx[BS];
        x = (int)(li % nbx) * W;
        y = (int)(li / nbx) * H;
    }
    const GpImage &I = B.img[lo];
    GpBlockRes &out = B.blk[I.shOff[BS] + li];
    if (x >= I.MiCols * 4 || y >= I.MiRows * 4) { out.J = GP_INF; out.ncand = 0; return; }
    constexpr int txSz = (W == 4 && H == 4) ? AV1G_TX_4X4 : (W == 8 && H == 8) ? AV1G_TX_8X8 : (W == 4 && H == 8) ? AV1G_TX_4X8
                       : (W == 8 && H == 4) ? AV1G_TX_8X4 : (W == 4 && H == 16) ? AV1G_TX_4X16 : AV1G_TX_16X4;
    const int set = av1g_get_tx_set_intra(txSz, 0);
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
    if (!cellwise && !fromList && prune_flat<BD>(I, P, x, y, W, H, qidx)) { out.J = GP_INF; out.ncand = 0; return; }
    constexpr int EL = AV1G_EDGE_OFF + 2 * (W + H) + 8;
    Px ea[EL], el[EL];
    Px *A = ea + AV1G_EDGE_OFF, *L = el + AV1G_EDGE_OFF;
    build_edges_par<BD>(gp_src<BD>(I, 0), sp, x, y, T::lw, T::lh, haveLeft, haveAbove, haveAR, haveBL, maxX, maxY, A, L, 0, 1);
    const int dcv = av1g_dc_value<BD>(A, L, haveLeft, haveAbove, T::lw, T::lh);
    Px s[W * H];
#pragma unroll
    for (int i = 0; i < H; i++)
#pragma unroll
        for (int j = 0; j < W; j++) s[i * W + j] = src[i * sp + j];
    auto mbits = [&](int c) -> float {
        const int m = cand_mode(c);
        float b = g_avgYMode[m];
        if (BS >= AV1G_BLOCK_8X8 && av1g_is_directional(m)) b += C[CID(angle_delta[m - AV1G_V_PRED]) + cand_delta(c) + 3];
        return b;
    };
    auto satd_of = [&](int c) -> float {
        int t = 0;
        for (int sy = 0; sy < H; sy += 4)
            for (int sx = 0; sx < W; sx += 4) {
                int d[16];
                pred_sub4(c, sy, sx, T::lw, T::lh, A, L, dcv, d);
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int j = 0; j < 4; j++) d[i * 4 + j] = (int)s[(sy + i) * W + sx + j] - d[i * 4 + j];
                t += hadamard4_sum(d);
            }
        return (float)t * 0.25f;
    };
    const GpExtCand *ext = (const GpExtCand *)B.extCand[BS];
    int K = min(P.K, GP_MAXK);
    int cands[GP_MAXK];
    int nc = 0;
    if (ext) {
        const GpExtCand *ec = ext + (size_t)(B.extOff[BS * B.nimg + lo] + li) * B.extTopK;
        for (int k = 0; k < B.extTopK && nc < GP_MAXK; k++) {
            const int m = ec[k].mode;
            if (m >= 13) continue;
            cands[nc++] = cand_pack(m, (BS >= AV1G_BLOCK_8X8 && av1g_is_directional(m)) ? ec[k].delta : 0);
        }
        K = min(K, nc);
    } else {
        float topJ[GP_MAXK];
        for (int k = 0; k < K; k++) { cands[k] = -1; topJ[k] = GP_INF; }
        float dirJ[8];
        auto insert = [&](int c, float j) {
            if (j < topJ[K - 1]) {
                int k = K - 1;
                while (k > 0 && topJ[k - 1] > j) { topJ[k] = topJ[k - 1]; cands[k] = cands[k - 1]; k--; }
                topJ[k] = j; cands[k] = c;
            }
        };
        for (int m = 0; m < 13; m++) {
            const int c = cand_pack(m, 0);
            const float j = satd_of(c) + lamS * mbits(c);
            if (m >= AV1G_V_PRED && m <= AV1G_D67_PRED) dirJ[m - AV1G_V_PRED] = j;
            insert(c, j);
        }
        if (P.deltas && BS >= AV1G_BLOCK_8X8) {
            int b0 = 0, b1 = 1;
            if (dirJ[b1] < dirJ[b0]) { b0 = 1; b1 = 0; }
            for (int k = 2; k < 8; k++) {
                if (dirJ[k] < dirJ[b0]) { b1 = b0; b0 = k; }
                else if (dirJ[k] < dirJ[b1]) b1 = k;
            }
            for (int t = 0; t < 2; t++)
                for (int d = -3; d <= 3; d++) {
                    if (!d) continue;
                    const int c = cand_pack(AV1G_V_PRED + (t ? b1 : b0), d);
                    insert(c, satd_of(c) + lamS * mbits(c));
                }
        }
        nc = 0;
        while (nc < K && cands[nc] >= 0) nc++;
    }
    // ---- RD ----
    const int hasC = av1g_has_chroma(BS, y >> 2, x >> 2, I.mono, I.ssx, I.ssy);
    const float overhead = (hasC ? P.uvOverheadBits : 0.f) + g_avgSkip0;
    EvalIn e;
    e.txSz = txSz; e.plane = 0;
    e.dcQ = av1g_dc_q<BD>(qidx); e.acQ = av1g_ac_q<BD>(qidx);
    e.rnd = P.rnd; e.zbin = P.zbin;
    e.skipCtx = 0; e.dcCtx = 0; e.reduced = 0; e.C = C;
    e.qmDist = P.qmDist; e.qmRnd = P.qmRnd; e.resEnergy = 0;
    int16_t res[W * H];
    Px pa[EL], pl[EL];
    auto residual = [&](int c) {
        const int m = cand_mode(c);
        const bool ef = I.edgeFilter && av1g_is_directional(m);
        int upA = 0, upL = 0;
        if (ef) {
            for (int k = 0; k < EL; k++) { pa[k] = ea[k]; pl[k] = el[k]; }
            av1g_edge_prepare<BD>(pangle_of(m, cand_delta(c)), T::lw, T::lh, 0, haveLeft, haveAbove, x, y, maxX, maxY,
                                  pa + AV1G_EDGE_OFF, pl + AV1G_EDGE_OFF, &upA, &upL);
        }
        for (int sy = 0; sy < H; sy += 4)
            for (int sx = 0; sx < W; sx += 4) {
                int pr[16];
                pred_sub4(c, sy, sx, T::lw, T::lh, ef ? pa + AV1G_EDGE_OFF : A, ef ? pl + AV1G_EDGE_OFF : L, dcv, pr, upA, upL);
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int j = 0; j < 4; j++) res[(sy + i) * W + sx + j] = (int16_t)((int)s[(sy + i) * W + sx + j] - pr[i * 4 + j]);
            }
    };
    // one evaluation site (code size): steps 0 .. nK-1 rate the candidates with DCT_DCT, the next ones the other tx
    // types of the set for the winner
    float bestJ = GP_INF;
    int bestK = 0, bestTx = AV1G_DCT_DCT, resK = -1;
    const int nK = min(nc, K);
    const int nSteps = nK + ((nc > 0 && P.txSearch && set != AV1G_TX_SET_DCTONLY) ? 15 : 0);
    for (int st = 0; st < nSteps; st++) {
        const int k = st < nK ? st : bestK, t = st < nK ? AV1G_DCT_DCT : st - nK + 1;
        if (t != AV1G_DCT_DCT && !av1g_tx_type_in_set_intra(set, t)) continue;
        const int c = cands[k];
        if (k != resK) { residual(c); resK = k; }
        e.txType = t;
        e.intraDir = cand_mode(c);
        e.qm = av1g_qm(I.usingQm, I.qm[0], 0, txSz, t);
        const EvalOut o = eval_scalar<W, H, BD>(res, e);
        const float J = o.D + lambda * (o.R + mbits(c) + overhead);
        if (J < bestJ) {
            bestJ = J;
            if (st < nK) bestK = k; else bestTx = t;
        }
    }
    out.J = nc ? bestJ : GP_INF;
    out.tx = (uint8_t)bestTx;
    int m = 0;
    if (nc) out.cand[m++] = (uint8_t)cands[bestK];
    for (int k = 0; k < nc && k < K; k++)
        if (k != bestK) out.cand[m++] = (uint8_t)cands[k];
    out.ncand = (uint8_t)m;
}

// Register budget per shape, measured on an RTX PRO 2000 Blackwell (one thread per block, so the budget is the
// thread's): 4x4 and 4x8 run 10-15 % faster at 6 CTAs per SM (80 registers, k_analysis_small6); 8x4 and 8x8 gain
// nothing there and 4x16 / 16x4 lose 2-3 % (their spills grow), so they keep the compiler's choice (126-168 registers).
template <int W, int H, int BS, int BD = 8, int MODE = 0>
__global__ void __launch_bounds__(128) k_analysis_small(GpBatch B, GpParams P, uint32_t nTotal, GpLive LC, GpAnX X) {
    analysis_small<W, H, BS, BD, MODE>(B, P, nTotal, LC, X);
}
template <int W, int H, int BS, int BD = 8, int MODE = 0>
__global__ void __launch_bounds__(128, 6) k_analysis_small6(GpBatch B, GpParams P, uint32_t nTotal, GpLive LC, GpAnX X) {
    analysis_small<W, H, BS, BD, MODE>(B, P, nTotal, LC, X);
}
