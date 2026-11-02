// palette candidates of the open-loop analysis (screen-content frames, GpImage::sct; launched after the analysis kernels
// only for batches with such frames): one warp per (image, block position) of a palette shape (8x8 .. 64x64, 4x16,
// 16x4).  The block's luma palettes of 8 .. 2 colours (pal_dev.cuh), the residual against each through the analysis'
// exact-rate evaluation with DCT_DCT and IDTX, J = D + lambda * (R + palette bits + DC_PRED bits + overhead).  The
// cheapest, if cheaper than the best mode, becomes candidate 0 of the block (PAL_CAND + size - 2) for the partition DP
// and the closed loop.
#include "kernels.h"
#include "dev_common.cuh"
#include "dev_tx.cuh"
#include "pal_dev.cuh"

GP_DEFINE_TABLE_UPLOAD(gp_upload_tables_pal)

template <int BD>
struct PalSmem {
    int16_t res[64 * 64];
    union {
        uint32_t hist[1 << BD];
        float tmp[32 * 65];
    };
    int16_t q[32 * 32];
    uint8_t map[64 * 64];
    uint16_t dv[64], dc[64], col[8];
};

// block gb of shape bs (one warp)
template <int BD>
__device__ __forceinline__ void pal_block(const GpBatch &B, const GpParams &P, int bs, uint32_t gb, PalSmem<BD> &S) {
    typedef av1g_pixel_t(BD) Px;
    const int lane = threadIdx.x;
    const uint32_t *starts = B.shImgStart + (size_t)bs * (B.nimg + 1);
    int lo = 0, hi = B.nimg - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (starts[mid] <= gb) lo = mid; else hi = mid - 1;
    }
    const GpImage &I = B.img[lo];
    if (!I.sct) return;
    const uint32_t li = gb - starts[lo];
    const int W = 4 * av1g_bw4(bs), H = 4 * av1g_bh4(bs);
    const int x = (int)(li % I.shNbx[bs]) * W, y = (int)(li / I.shNbx[bs]) * H;
    if (x >= I.MiCols * 4 || y >= I.MiRows * 4) return;
    GpBlockRes &out = B.blk[I.shOff[bs] + li];
    const int ow = min(W, I.MiCols * 4 - x), oh = min(H, I.MiRows * 4 - y);
    const Px *src = gp_src<BD>(I, 0) + (size_t)y * I.spitch[0] + x;
    const int sp = I.spitch[0];
    const int nd = pal_hist<BD>(src, sp, ow, oh, S.hist, S.dv, S.dc);
    if (nd < 2 || nd > 64) return;
    const int qidx = sb_qindex(I, x, y);
    const float lambda = gp_lambda<BD>(qidx, P.lambdaMul) * ssim_scale(I, x, y, W, H);
    const float *C = g_cost[I.qctx];
    const int txSz = AV1G_T(Max_Tx_Size_Rect)[bs];
    const int set = av1g_get_tx_set_intra(txSz, 0);
    const int hasC = av1g_has_chroma(bs, y >> 2, x >> 2, I.mono, I.ssx, I.ssy);
    const float overhead = (hasC ? P.uvOverheadBits : 0.f) + g_avgSkip0;
    EvalIn e;
    e.txSz = txSz; e.plane = 0;
    e.dcQ = av1g_dc_q<BD>(qidx); e.acQ = av1g_ac_q<BD>(qidx);
    e.rnd = P.rnd; e.zbin = P.zbin;
    e.skipCtx = 0; e.dcCtx = 0; e.reduced = 0; e.C = C;
    e.qmDist = P.qmDist; e.qmRnd = P.qmRnd;
    e.intraDir = AV1G_DC_PRED;
    // palette sizes from the most colours down (libaom searches 8 .. 2 too), the top colours and k-means for each; stop
    // after two sizes in a row that did not improve
    float bestJ = GP_INF;
    int bestCode = 0, bestT = AV1G_DCT_DCT, worse = 0;
    bool improved = false;
    for (int step = 0; step < 14 && worse < 2; step++) {
        const int nT = min(nd, 8) - (step >> 1);
        const bool top = !(step & 1);
        if (nT < 2) break;
        if (!top && nd <= nT) continue;   // exact palette: k-means gives the same
        int n = 0;
        if (lane == 0) n = pal_kmeans(S.dv, S.dc, nd, nT, top, BD, S.col);
        n = __shfl_sync(0xffffffffu, n, 0);
        __syncwarp();
        if (n < 2) continue;
        pal_map(src, sp, W, H, ow, oh, S.col, n, S.map, lane, 32);
        __syncwarp();
        float e2 = 0.f;
        for (int i = lane; i < W * H; i += 32) {
            const int r = (int)src[(i / W) * sp + i % W] - (int)S.col[S.map[i]];
            S.res[i] = (int16_t)r;
            e2 += (float)(r * r);
        }
        __syncwarp();
        e.resEnergy = warp_sum(e2);
        float best = GP_INF;
        int bt = AV1G_DCT_DCT;
        for (int k = 0; k < 2; k++) {
            const int t = k ? AV1G_IDTX : AV1G_DCT_DCT;
            if (k && !av1g_tx_type_in_set_intra(set, t)) continue;
            e.txType = t;
            e.qm = av1g_qm(I.usingQm, I.qm[0], 0, txSz, t);
            EvalOut o;
            warp_eval_rt<BD>(S.res, S.tmp, S.q, e, o, W, H);
            const float J = o.D + lambda * o.R;
            if (J < best) { best = J; bt = t; }
            __syncwarp();
        }
        const float J = best + lambda * (pal_bits(n, S.col, S.map, W, ow, oh, bs, BD, C) + g_avgYMode[AV1G_DC_PRED] + overhead);
        if (J < bestJ) { bestJ = J; bestCode = pal_code(nT, top); bestT = bt; improved = true; }
        __syncwarp();
        if (!top || nd <= nT) { worse = improved ? 0 : worse + 1; improved = false; }   // end of this size
    }
    if (lane == 0 && bestCode && bestJ < out.J) {
        // the palette first, then the analysis' candidates (the last one drops out when the list is full)
        const int nc = min((int)out.ncand + 1, GP_MAXK);
        for (int k = nc - 1; k > 0; k--) out.cand[k] = out.cand[k - 1];
        out.cand[0] = (uint8_t)bestCode;
        out.ncand = (uint8_t)nc;
        out.J = bestJ;
        out.tx = (uint8_t)bestT;
    }
}

// the palette shapes of a batch: shape bs[k] covers the combined block indices start[k] .. start[k + 1]
struct PalShapes {
    int n;
    int bs[AV1G_BLOCK_SIZES];
    uint32_t start[AV1G_BLOCK_SIZES + 1];
};

// one launch, a fixed grid that walks every palette shape's blocks: in batches without screen-content frames (the
// detection's flag, set on the GPU) every CTA returns at once
template <int BD>
__global__ void __launch_bounds__(32) k_pal_an(GpBatch B, GpParams P, PalShapes sh) {
    __shared__ PalSmem<BD> S;
    if (!*B.palAny) return;
    for (uint32_t g = blockIdx.x; g < sh.start[sh.n]; g += gridDim.x) {
        int k = 0;
        while (g >= sh.start[k + 1]) k++;
        pal_block<BD>(B, P, sh.bs[k], g - sh.start[k], S);
        __syncwarp();
    }
}

void gp_launch_pal_analysis(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st, int bd) {
    if (!B.oPal || !B.palAny) return;
    PalShapes sh;
    sh.n = 0;
    sh.start[0] = 0;
    for (int s = AV1G_BLOCK_8X8; s < AV1G_BLOCK_SIZES; s++) {
        if (!shTotal[s] || !av1g_palette_allowed(s) || (s >= 13 && s <= 15)) continue;
        sh.bs[sh.n] = s;
        sh.start[sh.n + 1] = sh.start[sh.n] + shTotal[s];
        sh.n++;
    }
    if (!sh.n) return;
    const uint32_t grid = sh.start[sh.n] < 2048 ? sh.start[sh.n] : 2048;   // the L4 holds 232 of these CTAs at once (4 per SM)
    if (bd == 10) k_pal_an<10><<<grid, 32, 0, st>>>(B, P, sh);
    else k_pal_an<8><<<grid, 32, 0, st>>>(B, P, sh);
}
