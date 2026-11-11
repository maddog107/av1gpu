// Open-loop shortlists for the closed loop ("open-loop shortlists, closed-loop picks"): between the partition DP and the
// wavefront, on the encoder stream (no host synchronisation), in parallel over every leaf of the batch.  k_recon runs per
// wavefront step and a step lasts as long as its slowest superblock, so work moved from a superblock's serial chain to
// here is paid at the price of parallel work.  Only with GpParams::uvShortlist (default off: nothing here runs and
// nothing is allocated).
//
//   chroma (B.uvShort)  the screening rc_chroma runs - SATD of the 13 UV modes over U and V, then of the 6 angle deltas
//                       around the best directional mode, cost SATD + lambda_s * mode bits (uv_mode context: the luma
//                       mode of the analysis winner) - on SOURCE edges built with the closed loop's exact availability
//                       (left / above from the tile, above-right / below-left from its BlockDecoded flags, which the
//                       partition fixes: k_lp_index); chroma blocks of 16 or more 4x4 sub-blocks screen a checkerboard of
//                       them.  Its best uvShortlist candidates, best first, are written per leaf; the closed loop skips
//                       its own screening and runs the exact RD of those plus CfL (whose alphas need the reconstructed
//                       luma) on the real edges.
//
// Two kernels: k_lp_index (one CTA per superblock) works out the availability flags and lists the leaves with work;
// k_lp_uv runs one warp per listed leaf (so a superblock of one 64x64 leaf holds no idle warps).  The list's order
// depends on the CTAs' timing, the work of an entry does not: every entry writes only its own leaf's slot from the
// image, its partition and fixed arithmetic (integer SATD sums), so the results are deterministic and the same in
// any batch.  A shortlist entry is a legal candidate whatever the edges: only decisions change, the stream stays valid.
#include "kernels.h"
#include "dev_common.cuh"
#include "k_analysis.cuh"

#include <algorithm>
#include <mutex>

GP_DEFINE_TABLE_UPLOAD(gp_upload_tables_leafprep)
const GpHostTables *gp_host_tables();   // k_misc.cu

#define LP_IX_NT 64    // k_lp_index: threads per superblock
#define LP_NW 4        // k_lp_uv: warps per CTA, one leaf each
#define LP_NT (LP_NW * 32)

// scratch of k_lp_index / k_lp_uv (stream-ordered, gp_launch_leafprep)
struct LpWork {
    uint32_t *cnt;     // [1] entries of uvItem
    uint32_t *uvItem;  // [nSb * GP_MAX_LEAVES] sb << 8 | leaf: leaves with chroma (B.uvShort)
    uint8_t *flags;    // [nSb * GP_MAX_LEAVES] per gp_short_slot: 4 chroma above-right, 8 chroma below-left
};

// the closed loop's BlockDecoded flag of 4x4 unit (r, c) of plane p (SB-relative, plane units; r, c in -1 .. 16) when
// leaf li starts: k_recon's clear_block_decoded_flags state outside the superblock, its tx blocks' marks inside it (own:
// the leaf that marks each unit, 0xFFFF none)
__device__ __forceinline__ int lp_decoded(const uint16_t *own, int p, int r, int c, int li, int sub, int sbW4, int sbH4) {
    const int S = 16 >> (p ? sub : 0);
    if (r >= 0 && c >= 0 && r < S && c < S) return own[r * 16 + c] < li;
    int v = (r < 0 && c < sbW4) || (c < 0 && r < sbH4);
    if (r == S && c == -1) v = 0;
    return v;
}

// the analysis result of the leaf (bsize at x, y): rc_luma's br
__device__ __forceinline__ const GpBlockRes &lp_br(const GpBatch &B, const GpImage &I, int bsize, int x, int y) {
    const int txSz = AV1G_T(Max_Tx_Size_Rect)[bsize];
    return B.blk[I.shOff[bsize] + (uint32_t)(y / AV1G_T(Tx_Height)[txSz]) * I.shNbx[bsize] + x / AV1G_T(Tx_Width)[txSz]];
}

// per superblock: BlockDecoded as k_recon builds it (rc_record_txb_at marks each recorded tx block's units - luma: the
// block (tx depth 0), beyond the frame edge too; chroma: its tx blocks, 4:4:4 ones outside the coded area are not coded),
// each chroma leaf's above-right / below-left flags, and the work list
__global__ void __launch_bounds__(LP_IX_NT) k_lp_index(GpBatch B, GpParams P, LpWork W) {
    __shared__ uint16_t own[2][16 * 16];   // luma / chroma: the leaf whose tx block marks each 4x4 unit of the SB
    const uint32_t sb = blockIdx.x;
    const GpImage &I = B.img[B.sbImg[sb]];
    const uint32_t local = sb - I.sbBase;
    const int sbc = (int)(local % I.sbCols), sbr = (int)(local / I.sbCols);
    const int sbx = sbc * 64, sby = sbr * 64;
    const int tid = threadIdx.x, lane = tid & 31;
    const TileBox tb = tile_of(I, sbx, sby);
    const int miRowEnd = tb.y1 >> 2, miColEnd = tb.x1 >> 2;
    const int ss = I.ssx;   // chroma subsampling (ssx == ssy)
    const int nLeaves = B.leafCountW[sb];
    const GpLeaf *lv = B.leaves + (size_t)sb * GP_MAX_LEAVES;
    for (int i = tid; i < 2 * 16 * 16; i += LP_IX_NT) own[i >> 8][i & 255] = 0xFFFF;
    __syncthreads();
    for (int li = tid; li < nLeaves; li += LP_IX_NT) {
        const GpLeaf lf = lv[li];
        const int bw4 = av1g_bw4(lf.bsize), bh4 = av1g_bh4(lf.bsize);
        for (int k = 0; k < bw4 * bh4; k++) own[0][(lf.y4 + k / bw4) * 16 + lf.x4 + k % bw4] = (uint16_t)li;
        const int mi_row = (sby >> 2) + lf.y4, mi_col = (sbx >> 2) + lf.x4;
        if (av1g_has_chroma(lf.bsize, mi_row, mi_col, I.mono, ss, ss)) {
            const int uvTx = av1g_uv_tx_size(lf.bsize, ss, ss), ub = av1g_uv_bsize(lf.bsize, ss, ss);
            const int w4 = AV1G_T(Tx_Width)[uvTx] >> 2, h4 = AV1G_T(Tx_Height)[uvTx] >> 2;
            const int nTxX = ss ? 1 : max(1, av1g_bw4(ub) / w4), nTxY = ss ? 1 : max(1, av1g_bh4(ub) / h4);
            const int r0 = (mi_row & 15) >> ss, c0 = (mi_col & 15) >> ss;
            const int x4 = mi_col >> ss, y4 = mi_row >> ss;
            for (int ty = 0; ty < nTxY; ty++)
                for (int tx = 0; tx < nTxX; tx++) {
                    if (!ss && (x4 + tx * w4 >= (I.MiCols >> ss) || y4 + ty * h4 >= (I.MiRows >> ss))) continue;
                    for (int k = 0; k < w4 * h4; k++) own[1][(r0 + ty * h4 + k / w4) * 16 + c0 + tx * w4 + k % w4] = (uint16_t)li;
                }
        }
    }
    __syncthreads();
    // per leaf (whole warps: the list entries are reserved with one atomic per warp)
    for (int l0 = tid - lane; l0 < nLeaves; l0 += LP_IX_NT) {
        const int li = l0 + lane;
        bool uv = false;
        if (li < nLeaves) {
            const GpLeaf lf = lv[li];
            const int bsize = lf.bsize, x = sbx + lf.x4 * 4, y = sby + lf.y4 * 4, mi_row = y >> 2, mi_col = x >> 2;
            if (av1g_has_chroma(bsize, mi_row, mi_col, I.mono, ss, ss)) {
                const int sbW4 = miColEnd - sbc * 16, sbH4 = miRowEnd - sbr * 16;
                const int uvTx = av1g_uv_tx_size(bsize, ss, ss);
                const int r4 = (mi_row & 15) >> ss, c4 = (mi_col & 15) >> ss;
                int f = lp_decoded(own[1], 1, r4 - 1, c4 + (AV1G_T(Tx_Width)[uvTx] >> 2), li, ss, sbW4 >> ss, sbH4 >> ss) << 2;
                f |= lp_decoded(own[1], 1, r4 + (AV1G_T(Tx_Height)[uvTx] >> 2), c4 - 1, li, ss, sbW4 >> ss, sbH4 >> ss) << 3;
                W.flags[gp_short_slot(sb, mi_row, mi_col)] = (uint8_t)f;
                uv = true;
            }
        }
        const uint32_t item = sb << 8 | (uint32_t)li;
        const unsigned m = __ballot_sync(0xffffffffu, uv);
        if (m) {
            uint32_t base = 0;
            if (lane == 0) base = atomicAdd(W.cnt, (uint32_t)__popc(m));
            base = __shfl_sync(0xffffffffu, base, 0);
            if (uv) W.uvItem[base + __popc(m & ((1u << lane) - 1))] = item;
        }
    }
}

// the leaf of a list entry and its block geometry
struct LpLeaf {
    uint32_t sb;
    int li, bsize, x, y, mi_row, mi_col, ss, flags, availU, availL, availUC, availLC, qidx;
    TileBox tb;
};
__device__ __forceinline__ LpLeaf lp_leaf(const GpBatch &B, const LpWork &W, uint32_t item, const GpImage *&Ip) {
    LpLeaf L;
    L.sb = item >> 8;
    L.li = (int)(item & 255);
    const GpImage &I = B.img[B.sbImg[L.sb]];
    Ip = &I;
    const uint32_t local = L.sb - I.sbBase;
    const int sbx = (int)(local % I.sbCols) * 64, sby = (int)(local / I.sbCols) * 64;
    const GpLeaf lf = B.leaves[(size_t)L.sb * GP_MAX_LEAVES + L.li];
    L.bsize = lf.bsize;
    L.x = sbx + lf.x4 * 4;
    L.y = sby + lf.y4 * 4;
    L.mi_row = L.y >> 2;
    L.mi_col = L.x >> 2;
    L.ss = I.ssx;
    L.tb = tile_of(I, sbx, sby);
    L.flags = W.flags[gp_short_slot(L.sb, L.mi_row, L.mi_col)];
    av1g_block_avail(L.bsize, L.mi_row, L.mi_col, I.mono, L.ss, L.ss, L.tb.y0 >> 2, L.tb.y1 >> 2, L.tb.x0 >> 2, L.tb.x1 >> 2, &L.availU,
                     &L.availL, &L.availUC, &L.availLC);
    L.qidx = sb_qindex(I, sbx, sby);
    return L;
}

template <typename Px>
struct LpUvScr {
    __align__(16) Px ea[2][AV1G_EDGE_LEN];   // U / V edges
    __align__(16) Px el[2][AV1G_EDGE_LEN];
    int satdI[32];
    int cand[32];
};

// chroma: rc_chroma's SATD screening on the source, one warp per listed leaf
template <int BD>
__global__ void __launch_bounds__(LP_NT) k_lp_uv(GpBatch B, GpParams P, LpWork W) {
    typedef av1g_pixel_t(BD) Px;
    __shared__ LpUvScr<Px> scr[LP_NW];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    LpUvScr<Px> &S = scr[warp];
    const uint32_t n = *W.cnt;
    for (uint32_t e = blockIdx.x * LP_NW + warp; e < n; e += gridDim.x * LP_NW) {
        const GpImage *Ip;
        const LpLeaf L = lp_leaf(B, W, W.uvItem[e], Ip);
        const GpImage &I = *Ip;
        const int bsize = L.bsize, ss = L.ss;
        const int c0 = lp_br(B, I, bsize, L.x, L.y).cand[0];
        const int yMode = c0 >= 128 ? AV1G_DC_PRED : cand_mode(c0);   // k_recon's luma_mode (palette, filter intra: DC_PRED)
        const float *C = g_cost[I.qctx];
        const float lambdaB = gp_lambda<BD>(L.qidx, P.lambdaMul) * ssim_scale(I, L.x, L.y, bw_px(bsize), bh_px(bsize));   // k_recon's
        const int uvTx = av1g_uv_tx_size(bsize, ss, ss);
        const int Wc = AV1G_T(Tx_Width)[uvTx], Hc = AV1G_T(Tx_Height)[uvTx], lw = log2i(Wc), lh = log2i(Hc);
        const int cx = (L.mi_col >> ss) * 4, cy = (L.mi_row >> ss) * 4;
        const int maxX = ((I.MiCols * 4) >> ss) - 1, maxY = ((I.MiRows * 4) >> ss) - 1;
        for (int k = 0; k < 2; k++)
            build_edges_par<BD>(gp_src<BD>(I, 1 + k), I.spitch[1 + k], cx, cy, lw, lh, L.availLC, L.availUC, (L.flags >> 2) & 1,
                                (L.flags >> 3) & 1, maxX, maxY, S.ea[k] + AV1G_EDGE_OFF, S.el[k] + AV1G_EDGE_OFF, lane, 32);
        S.satdI[lane] = 0;
        if (lane < 13) S.cand[lane] = cand_pack(lane, 0);
        __syncwarp();
        int dcv[2];
        for (int k = 0; k < 2; k++) dcv[k] = warp_dc_value<BD>(S.ea[k] + AV1G_EDGE_OFF, S.el[k] + AV1G_EDGE_OFF, L.availLC, L.availUC, lw, lh);
        const Px *srcU = gp_src<BD>(I, 1) + (size_t)cy * I.spitch[1] + cx, *srcV = gp_src<BD>(I, 2) + (size_t)cy * I.spitch[2] + cx;
        const float lamS = P.satdLambda * sqrtf(lambdaB * P.lambdaCMul);
        const bool cflAllowed = av1g_cfl_allowed(bsize);
        auto uvcbits = [&](int c) -> float {   // rc_chroma's mode bits of candidate code c
            const int m = cand_mode(c);
            float b = cflAllowed ? C[CID(uv_mode_cfl_allowed[yMode]) + m] : C[CID(uv_mode_cfl_not_allowed[yMode]) + m];
            if (bsize >= AV1G_BLOCK_8X8 && av1g_is_directional(m)) b += C[CID(angle_delta[m - AV1G_V_PRED]) + cand_delta(c) + 3];
            return b;
        };
        // 16 or more 4x4 sub-blocks: a checkerboard of them (the analysis' --screen-sub 2 for luma), half the work
        const int nsx = Wc >> 2, nsub = (Wc >> 2) * (Hc >> 2), cb = nsub >= 16, ln = lw + lh - 4 - cb;
        int nuv = 13;
        for (int pass = 0; pass < 2; pass++) {
            const int ca = pass ? 13 : 0, nc = pass ? 6 : 13;
            if (pass) {
                if (!(P.uvDeltas && bsize >= AV1G_BLOCK_8X8)) break;
                // best directional mode at delta 0 (the first minimum), its deltas -1 .. -3, +1 .. +3
                const float j = lane >= AV1G_V_PRED && lane <= AV1G_D67_PRED ? fmaf(lamS, uvcbits(cand_pack(lane, 0)), (float)S.satdI[lane] * 0.25f)
                                                                             : GP_INF;
                const int bm = warp_argmin(j, lane);
                if (lane < 6) S.cand[13 + lane] = cand_pack(bm, lane < 3 ? -(lane + 1) : (lane - 2));
                __syncwarp();
                nuv = 19;
            }
            // one 4x4 sub-block of one (candidate, plane) per lane: U items, then V; integer sums (order-free)
            const int ns = nsub >> cb;   // sub-blocks screened per candidate and plane
            for (int it = lane; it < nc * ns * 2; it += 32) {
                const int pl = it < nc * ns ? 0 : 1, rr = it - pl * nc * ns;
                const int ci = ca + (rr >> ln);
                int sbk = rr & (ns - 1);
                if (cb) sbk = 2 * sbk + (((2 * sbk) >> (lw - 2)) & 1);   // row r: the columns of r's parity
                const int sx = (sbk & (nsx - 1)) * 4, sy = (sbk >> (lw - 2)) * 4;
                const Px *s = pl ? srcV : srcU;
                const int sp = I.spitch[1 + pl];
                int d[16], p[16];
                pred_sub4(S.cand[ci], sy, sx, lw, lh, S.ea[pl] + AV1G_EDGE_OFF, S.el[pl] + AV1G_EDGE_OFF, dcv[pl], p);
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int jj = 0; jj < 4; jj++) d[i * 4 + jj] = (int)s[(sy + i) * sp + sx + jj] - p[i * 4 + jj];
                atomicAdd(&S.satdI[ci], hadamard4_sum(d));
            }
            __syncwarp();
        }
        // the best uvShortlist by screening cost, in (cost, index) order (rc_chroma's top-K)
        float js = lane < nuv ? fmaf(lamS, uvcbits(S.cand[lane]), (float)S.satdI[lane] * 0.25f) : GP_INF;
        uint8_t *out = B.uvShort + gp_short_slot(L.sb, L.mi_row, L.mi_col) * P.uvShortlist;
        for (int k = 0; k < P.uvShortlist; k++) {
            const int b = warp_argmin(js, lane);
            if (lane == 0) out[k] = (uint8_t)(b < 0 ? 0xFF : S.cand[b]);
            if (lane == b) js = GP_INF;
        }
        __syncwarp();
    }
}

int gp_launch_leafprep(const GpBatch &B, const GpParams &P, cudaStream_t st, int bd) {
    if (!B.nSb || !B.uvShort) return 0;
    const size_t slots = (size_t)B.nSb * GP_MAX_LEAVES;
    LpWork W;
    char *mem = nullptr;
    cudaError_t e = cudaMallocAsync((void **)&mem, 16 + slots * 4 + slots, st);
    if (e != cudaSuccess) return (int)e;
    W.cnt = (uint32_t *)mem;
    W.uvItem = (uint32_t *)(mem + 16);
    W.flags = (uint8_t *)(W.uvItem + slots);
    e = cudaMemsetAsync(W.cnt, 0, 4, st);
    if (e == cudaSuccess) {
        // one warp per list entry, grid-strided; about 8 leaves per superblock: one warp per superblock
        const unsigned grid = (unsigned)std::max<uint32_t>(1, (B.nSb + LP_NW - 1) / LP_NW);
        k_lp_index<<<B.nSb, LP_IX_NT, 0, st>>>(B, P, W);
        if (bd == 10) k_lp_uv<10><<<grid, LP_NT, 0, st>>>(B, P, W);
        else k_lp_uv<8><<<grid, LP_NT, 0, st>>>(B, P, W);
        e = cudaGetLastError();
    }
    const cudaError_t f = cudaFreeAsync(mem, st);
    return (int)(e != cudaSuccess ? e : f);
}

// tables of this module on first use (as gp_recon_prepare): a process without shortlists never loads it lazily
int gp_leafprep_prepare() {
    static std::once_flag once;
    static int rc = 0;
    std::call_once(once, []() {
        rc = gp_upload_tables_leafprep(gp_host_tables());
        if (!rc && cudaStreamSynchronize(cudaStreamPerThread) != cudaSuccess) rc = -1;
    });
    return rc;
}
