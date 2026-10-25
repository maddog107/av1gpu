// optional chroma term of the open-loop analysis (GpParams::uvInDp): for every block of a
// shape that carries chroma, the best exact-rate U+V RD cost over {DC_PRED, the luma winner's mode}
// (source edges, delta 0, no CfL) is added to the block's J, so the partition DP decides on
// luma + chroma cost (measured -0.9 % total J for chroma-aware partitioning).
// One warp per block, launched after the luma analysis of the shape.
#pragma once
#include "dev_tx.cuh"
#include "k_analysis.cuh"

template <int W, int H>
struct UvDims {
    static constexpr int w = W / 2 < 4 ? 4 : (W / 2 > 32 ? 32 : W / 2);
    static constexpr int h = H / 2 < 4 ? 4 : (H / 2 > 32 ? 32 : H / 2);
};

template <int UW, int UH>
struct UvSmem {
    int16_t res[UW * UH];
    float tmp[TxD<UW, UH>::th * TxD<UW, UH>::TS];
    int16_t q[TxD<UW, UH>::n];
    uint8_t ea[2][AV1G_EDGE_LEN], el[2][AV1G_EDGE_LEN];
    int dcv[2];
};

template <int W, int H, int BS, int NW>
__global__ void __launch_bounds__(NW * 32) k_analysis_uv(GpBatch B, GpParams P, uint32_t nTotal) {
    constexpr int UW = UvDims<W, H>::w, UH = UvDims<W, H>::h;
    typedef TxD<UW, UH> T;
    __shared__ UvSmem<UW, UH> smem[NW];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    UvSmem<UW, UH> &S = smem[warp];
    const uint32_t *starts = B.shImgStart + (size_t)BS * (B.nimg + 1);
    constexpr int uvTx = (UW == 4 && UH == 4) ? AV1G_TX_4X4 : (UW == 8 && UH == 8) ? AV1G_TX_8X8 : (UW == 16 && UH == 16) ? AV1G_TX_16X16
                       : (UW == 32 && UH == 32) ? AV1G_TX_32X32 : (UW == 4 && UH == 8) ? AV1G_TX_4X8 : (UW == 8 && UH == 4) ? AV1G_TX_8X4
                       : (UW == 8 && UH == 16) ? AV1G_TX_8X16 : (UW == 16 && UH == 8) ? AV1G_TX_16X8 : (UW == 16 && UH == 32) ? AV1G_TX_16X32
                       : (UW == 32 && UH == 16) ? AV1G_TX_32X16 : (UW == 4 && UH == 16) ? AV1G_TX_4X16 : (UW == 16 && UH == 4) ? AV1G_TX_16X4
                       : (UW == 8 && UH == 32) ? AV1G_TX_8X32 : AV1G_TX_32X8;
    for (uint32_t gb = blockIdx.x * NW + warp; gb < nTotal; gb += gridDim.x * NW) {
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
        if (x >= I.MiCols * 4 || y >= I.MiRows * 4 || out.ncand == 0 || out.J >= GP_INF) continue;
        if (!av1g_has_chroma(BS, y >> 2, x >> 2, I.mono)) continue;
        const int cx = ((x >> 2) >> 1) * 4, cy = ((y >> 2) >> 1) * 4;   // chroma block origin (sub-8x8: the 8x8 area)
        const TileBox tb = tile_of(I, x, y);
        const int haveLeft = cx > (tb.x0 >> 1), haveAbove = cy > (tb.y0 >> 1);
        const int haveAR = ol_decoded(cx + UW, cy - 1, cx, cy, 1, tb);
        const int haveBL = ol_decoded(cx - 1, cy + UH, cx, cy, 1, tb);
        const int maxX = ((I.MiCols * 4) >> 1) - 1, maxY = ((I.MiRows * 4) >> 1) - 1;
        const int qidx = sb_qindex(I, x, y);
        const float lambda = gp_lambda(qidx, P.lambdaMul) * ssim_scale(I, x, y, W, H) * P.lambdaCMul;
        const BlockQ bq = block_q(I, qidx);
        const float *C = g_cost[I.qctx];
        for (int pl = 0; pl < 2; pl++) {
            uint8_t *A = S.ea[pl] + AV1G_EDGE_OFF, *L = S.el[pl] + AV1G_EDGE_OFF;
            build_edges_par(I.src[1 + pl], I.spitch[1 + pl], cx, cy, T::lw, T::lh, haveLeft, haveAbove, haveAR, haveBL, maxX, maxY,
                            A, L, lane, 32);
        }
        __syncwarp();
        if (lane < 2) S.dcv[lane] = av1g_dc_value(S.ea[lane] + AV1G_EDGE_OFF, S.el[lane] + AV1G_EDGE_OFF, haveLeft, haveAbove, T::lw, T::lh);
        __syncwarp();
        const int yMode = cand_mode(out.cand[0]);
        const bool cflAllowed = av1g_cfl_allowed(BS);
        float best = GP_INF;
        const int nc = yMode == AV1G_DC_PRED ? 1 : 2;
        for (int k = 0; k < nc; k++) {
            const int uv = k == 0 ? AV1G_DC_PRED : yMode;
            const int c = cand_pack(uv, 0);
            float J = lambda * (cflAllowed ? C[CID(uv_mode_cfl_allowed[yMode]) + uv] : C[CID(uv_mode_cfl_not_allowed[yMode]) + uv]);
            if (BS >= AV1G_BLOCK_8X8 && av1g_is_directional(uv)) J += lambda * C[CID(angle_delta[uv - AV1G_V_PRED]) + 3];
            for (int pl = 0; pl < 2; pl++) {
                const uint8_t *src = I.src[1 + pl] + (size_t)cy * I.spitch[1 + pl] + cx;
                const float e2 = warp_residual<UW, UH>(src, I.spitch[1 + pl], c, S.res, S.ea[pl] + AV1G_EDGE_OFF,
                                                       S.el[pl] + AV1G_EDGE_OFF, S.dcv[pl]);
                EvalIn e;
                e.txSz = uvTx; e.plane = 1 + pl;
                e.txType = av1g_chroma_tx_type(uvTx, uv, 0);
                e.dcQ = bq.dcQ[1 + pl]; e.acQ = bq.acQ[1 + pl];
                e.qm = av1g_qm(I.usingQm, I.qm[1 + pl], 1 + pl, uvTx, e.txType);
                e.rnd = P.rndC; e.zbin = P.zbinC;
                e.skipCtx = 7; e.dcCtx = 0; e.intraDir = 0; e.reduced = 0; e.C = C;
                e.resEnergy = e2; e.qmDist = P.qmDist; e.qmRnd = P.qmRnd;
                EvalOut o;
                warp_eval<UW, UH>(S.res, S.tmp, S.q, e, o);
                J += o.D + lambda * o.R;
            }
            best = fminf(best, J);
        }
        if (lane == 0) out.J += best;
        __syncwarp();
    }
}
