// Lossless encoding (base_q_idx 0, GpConfig::lossless).  Every tx block is a 4x4 WHT in every plane and the
// reconstruction IS the source, so every intra prediction can be computed from source pixels: there is no
// closed loop and no wavefront, and the only thing that depends on the decisions is which neighbours a 4x4
// may use (above-right / below-left availability, decode order).  Distortion is 0: decisions minimise an
// estimate of the rate (default-CDF symbol costs at q-context 0, dev_tx.cuh coeff_bits).
//
//   k_ll_cost    thread per (plane, 4x4, candidate entry): bits of the WHT residual of one prediction; one
//                entry per candidate (mode, angle delta, filter-intra mode) and per above-right (modes < 90 deg)
//                / below-left (> 180 deg) availability it depends on.  Tables: position-major uint16, 1/16 bit.
//   k_ll_shape   warp per (shape, block): lanes over the 66 luma candidates, J = sum of the block's 4x4 costs
//                (in-block availability exact, the block's corners by the open-loop Z-order guess) + mode bits,
//                plus the best chroma mode's J (4:2:0 sub-8x8 blocks: their share of the 4x4 chroma block)
//                -> GpBlockRes.J in lambda units, so the lossy k_dp builds the partition unchanged.
//   k_ll_decide  CTA per SB: tx layout of the leaves, exact availability (BlockDecoded via an order map), luma
//                and chroma modes from the tables, CfL with a full alpha search (4x4 chroma blocks only)
//                -> blocks, tx blocks, mode maps.
//   k_ll_coeffs  CTA per SB: exact prediction of every tx block (edge filter types from the final mode maps,
//                CfL from source luma), levels = av1g_fwht4x4(source - prediction); blocks without residual
//                get skip = 1.
#include "kernels.h"
#include "dev_common.cuh"
#include "dev_tx.cuh"

#include <cstdio>
#include <cstring>
#include <vector>

GP_DEFINE_TABLE_UPLOAD(gp_upload_tables_ll_dev)

// ---------------------------------------------------------------------------
// candidates and cost-table entries
// ---------------------------------------------------------------------------
// candidates 0..60: (mode, angle delta) of the 13 modes (directional ones with deltas -3..3), shared by luma and
// chroma; 61..65: filter intra modes 0..4 (luma).  Entries: one per candidate, two for candidates whose prediction
// reads the above-right (pAngle < 90) or below-left (pAngle > 180) edge extension.
struct LLCand { int8_t mode, delta, fi, var; uint8_t ent; uint8_t pad[3]; };   // var: 0 none, 1 AR, 2 BL
struct LLEnt { int8_t mode, delta, fi, ar, bl, pad[3]; };
static __constant__ LLCand c_llCand[LL_NCAND_Y];
static __constant__ LLEnt c_llEnt[LL_NE_Y];

static void ll_build_tables(LLCand *cand, LLEnt *ent) {
    int nc = 0, ne = 0;
    auto add = [&](int m, int d, int fi) {
        LLCand &c = cand[nc++];
        c.mode = (int8_t)m; c.delta = (int8_t)d; c.fi = (int8_t)fi; c.ent = (uint8_t)ne; c.var = 0;
        const int pa = av1g_is_directional(m) ? AV1G_T(Mode_To_Angle)[m] + d * AV1G_ANGLE_STEP : 90;
        if (fi < 0 && pa < 90) c.var = 1;
        if (fi < 0 && pa > 180) c.var = 2;
        for (int v = 0; v < (c.var ? 2 : 1); v++) {
            LLEnt &e = ent[ne++];
            e.mode = (int8_t)m; e.delta = (int8_t)d; e.fi = (int8_t)fi;
            e.ar = (int8_t)(c.var == 1 ? v : 0); e.bl = (int8_t)(c.var == 2 ? v : 0);
        }
    };
    for (int m = 0; m < 13; m++) {
        if (av1g_is_directional(m)) for (int d = -3; d <= 3; d++) add(m, d, -1);
        else add(m, 0, -1);
    }
    for (int f = 0; f < 5; f++) add(AV1G_DC_PRED, 0, f);
}

int gp_upload_tables_ll(const GpHostTables *h) {
    if (gp_upload_tables_ll_dev(h)) return -1;
    LLCand cand[LL_NCAND_Y + 8];
    LLEnt ent[LL_NE_Y + 16];
    memset(cand, 0, sizeof cand);
    memset(ent, 0, sizeof ent);
    ll_build_tables(cand, ent);
    // the table sizes the kernels assume: 61 + 5 candidates, 88 (chroma) + 5 entries
    if (cand[LL_NCAND_UV].ent != LL_NE_UV || cand[LL_NCAND_Y - 1].ent != LL_NE_Y - 1 || cand[LL_NCAND_UV - 1].fi >= 0 ||
        cand[LL_NCAND_UV].fi != 0) {
        fprintf(stderr, "lossless candidate tables: unexpected layout\n");
        return -1;
    }
    if (cudaMemcpyToSymbol(c_llCand, cand, sizeof(LLCand) * LL_NCAND_Y) != cudaSuccess) return -1;
    if (cudaMemcpyToSymbol(c_llEnt, ent, sizeof(LLEnt) * LL_NE_Y) != cudaSuccess) return -1;
    return 0;
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
// estimated bits of one lossless 4x4 tx block (levels q, natural order): txb_skip, eob, every coefficient
// (no transform type: base_q_idx 0).  Neighbour contexts are unknown here: typical ones (luma ctx 5, chroma 12).
__device__ float ll_rate(const int16_t *q, int ptype, const float *C) {
    const uint16_t *scan = AV1G_T(Default_Scan_4x4);
    int eob = 0;
    for (int c = 15; c >= 0; c--)
        if (q[scan[c]]) { eob = c + 1; break; }
    const int skipCtx = ptype ? 12 : 5;
    if (!eob) return C[CID(txb_skip[0][skipCtx]) + 1];
    float b = C[CID(txb_skip[0][skipCtx]) + 0];
    int eobPt;
    if (eob <= 2) eobPt = eob;
    else { eobPt = 2; int v = eob - 1; while (v > 1) { v >>= 1; eobPt++; } }
    b += C[CID(eob_pt_16[ptype][0]) + eobPt - 1];
    if (eobPt >= 3) {
        const int rem = eob - ((1 << (eobPt - 2)) + 1);
        b += C[CID(eob_extra[0][ptype][eobPt - 3]) + ((rem >> (eobPt - 3)) & 1)];
        b += (float)(eobPt - 3);
    }
    for (int c = 0; c < eob; c++) b += coeff_bits(c, eob, q, scan, AV1G_TX_4X4, AV1G_TX_CLASS_2D, ptype, 0, C);
    return b;
}

// BlockDecoded of the above-right / below-left 4x4 of the 4x4 tx block t at (x4, y4) (plane units), spec
// transform_block with av1g_txb_edges' rules on an SB-local order map (ord[yy * 16 + xx] = SB txb index or 0xFFFF).
__device__ __forceinline__ void ll_ar_bl(int plane, int ssx, int ssy, int x4, int y4, int t, int tileR1, int tileC1,
                                         const uint16_t *ord, int &ar, int &bl) {
    const int subX = plane ? ssx : 0, subY = plane ? ssy : 0;
    const int row = y4 << subY, col = x4 << subX;
    const int sbR = row & ~15, sbC = col & ~15;
    const int by = (row & 15) >> subY, bx = (col & 15) >> subX;
    const int sbSizeX = 16 >> subX, sbSizeY = 16 >> subY;
    const int sbW4 = (tileC1 - sbC) >> subX, sbH4 = (tileR1 - sbR) >> subY;
    auto dec = [&](int yy, int xx) -> int {
        if (yy < 0) return xx < sbW4;
        if (xx < 0) return yy < sbH4 && yy != sbSizeY;
        if (yy >= sbSizeY || xx >= sbSizeX) return 0;
        return ord[yy * 16 + xx] < t;
    };
    ar = dec(by - 1, bx + 1);
    bl = dec(by + 1, bx - 1);
}

// tile bounds (mi units) of luma mi (r, c)
struct LLTile { int r0, r1, c0, c1; };
__device__ __forceinline__ LLTile ll_tile(const GpImage &I, int r, int c) {
    int tc = 0, tr = 0;
    while (tc + 1 < I.tileCols && c >= I.tileColStart[tc + 1]) tc++;
    while (tr + 1 < I.tileRows && r >= I.tileRowStart[tr + 1]) tr++;
    LLTile t = {I.tileRowStart[tr], I.tileRowStart[tr + 1], I.tileColStart[tc], I.tileColStart[tc + 1]};
    return t;
}

// prediction of one 4x4 tx block of plane `plane` at plane pixel (x, y) from the source (= reconstruction)
template <int BD>
__device__ void ll_predict(const GpImage &I, int plane, int x, int y, int mode, int delta, int fi, int haveL, int haveA, int ar,
                           int bl, int edgeFilter, int filterType, av1g_pixel_t(BD) *pred) {
    typedef av1g_pixel_t(BD) P;
    const int sx = plane ? I.ssx : 0, sy = plane ? I.ssy : 0;
    const int maxX = ((I.MiCols * 4) >> sx) - 1, maxY = ((I.MiRows * 4) >> sy) - 1;
    P edgeA[AV1G_EDGE_OFF + 24], edgeL[AV1G_EDGE_OFF + 24];   // 4x4: indices -2 .. 2 * (4 + 4) are used
    P *A = edgeA + AV1G_EDGE_OFF, *L = edgeL + AV1G_EDGE_OFF;
    av1g_build_edges<BD>(gp_src<BD>(I, plane), I.spitch[plane], x, y, 2, 2, haveL, haveA, ar, bl, maxX, maxY, A, L);
    av1g_predict_intra_ex<BD>(mode, delta, fi, edgeFilter, filterType, haveL, haveA, 2, 2, x, y, maxX, maxY, A, L, pred, 4);
}

// levels of a 4x4: WHT of (source - pred)
template <int BD>
__device__ __forceinline__ void ll_levels(const GpImage &I, int plane, int x, int y, const av1g_pixel_t(BD) *pred, int16_t *q) {
    const av1g_pixel_t(BD) *s = gp_src<BD>(I, plane) + (size_t)y * I.spitch[plane] + x;
    int32_t r[16];
    for (int i = 0; i < 4; i++)
        for (int j = 0; j < 4; j++) r[i * 4 + j] = (int)s[(size_t)i * I.spitch[plane] + j] - (int)pred[i * 4 + j];
    av1g_fwht4x4(r, q);
}

// CfL (spec 7.11.5) on a 4x4 chroma tx block at chroma (startX, startY): pred holds the DC prediction
template <int BD>
__device__ __forceinline__ void ll_cfl(int alpha, const int *ac, av1g_pixel_t(BD) *pred) {
    for (int i = 0; i < 16; i++) pred[i] = (av1g_pixel_t(BD))av1g_clip1<BD>(pred[i] + av1g_round2signed(alpha * ac[i], 6));
}
// the "v - lumaAvg" terms of av1g_cfl_predict for a 4x4 chroma tx block (luma = source)
template <int BD>
__device__ void ll_cfl_ac(const GpImage &I, int startX, int startY, int maxLumaW, int maxLumaH, int *ac) {
    const int subX = I.ssx, subY = I.ssy;
    const av1g_pixel_t(BD) *luma = gp_src<BD>(I, 0);
    const int lp = I.spitch[0];
    int sum = 0;
    for (int i = 0; i < 4; i++) {
        int lumaY = (startY + i) << subY;
        if (lumaY > maxLumaH - (1 << subY)) lumaY = maxLumaH - (1 << subY);
        for (int j = 0; j < 4; j++) {
            int lumaX = (startX + j) << subX;
            if (lumaX > maxLumaW - (1 << subX)) lumaX = maxLumaW - (1 << subX);
            int t = 0;
            for (int dy = 0; dy <= subY; dy++)
                for (int dx = 0; dx <= subX; dx++) t += luma[(size_t)(lumaY + dy) * lp + lumaX + dx];
            ac[i * 4 + j] = t << (3 - subX - subY);
            sum += ac[i * 4 + j];
        }
    }
    const int avg = av1g_round2(sum, 4);
    for (int i = 0; i < 16; i++) ac[i] -= avg;
}

// luma mode bits (average contexts): y_mode, angle delta, filter intra flag / mode
__device__ __forceinline__ float ll_ymode_bits(int bs, const LLCand &c, int fiOn, const float *C) {
    float b = g_avgYMode[c.mode];
    if (bs >= AV1G_BLOCK_8X8 && av1g_is_directional(c.mode)) b += C[CID(angle_delta[c.mode - AV1G_V_PRED]) + c.delta + 3];
    if (fiOn && c.mode == AV1G_DC_PRED && av1g_bw4(bs) <= 8 && av1g_bh4(bs) <= 8) {
        b += C[CID(filter_intra[bs]) + (c.fi >= 0)];
        if (c.fi >= 0) b += C[CID(filter_intra_mode) + c.fi];
    }
    return b;
}
__device__ __forceinline__ bool ll_cand_ok(int bs, const LLCand &c, int fiOn) {
    if (c.delta && bs < AV1G_BLOCK_8X8) return false;
    if (c.fi >= 0 && !(fiOn && av1g_bw4(bs) <= 8 && av1g_bh4(bs) <= 8)) return false;
    return true;
}
__device__ __forceinline__ float ll_uvmode_bits(int bs, int yMode, int uv, int delta, int cflAllowed, const float *C) {
    float b = cflAllowed ? C[CID(uv_mode_cfl_allowed[yMode]) + uv] : C[CID(uv_mode_cfl_not_allowed[yMode]) + uv];
    if (bs >= AV1G_BLOCK_8X8 && uv < 13 && av1g_is_directional(uv)) b += C[CID(angle_delta[uv - AV1G_V_PRED]) + delta + 3];
    return b;
}

__device__ __forceinline__ int ll_find_image(const uint32_t *starts, int nimg, uint32_t g) {
    int lo = 0, hi = nimg - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (starts[mid] <= g) lo = mid; else hi = mid - 1;
    }
    return lo;
}

// ---------------------------------------------------------------------------
// k_ll_cost
// ---------------------------------------------------------------------------
template <int BD>
__global__ void __launch_bounds__(128) k_ll_cost(GpBatch B, GpLossless L, int edgeFilter) {
    const uint64_t g = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= L.segStart[L.nSeg]) return;
    int lo = 0, hi = L.nSeg - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (L.segStart[mid] <= g) lo = mid; else hi = mid - 1;
    }
    const int img = lo / 3, plane = lo % 3;
    const GpImage &I = B.img[img];
    const int ne = plane ? LL_NE_UV : LL_NE_Y;
    const uint64_t local = g - L.segStart[lo];
    const uint32_t pos = (uint32_t)(local / ne);
    const int e = (int)(local % ne);
    const int sx = plane ? I.ssx : 0, sy = plane ? I.ssy : 0;
    const int w4 = I.MiCols >> sx;
    const int x4 = (int)(pos % w4), y4 = (int)(pos / w4);
    const LLTile t = ll_tile(I, y4 << sy, x4 << sx);
    const int haveL = x4 > (t.c0 >> sx), haveA = y4 > (t.r0 >> sy);
    const LLEnt E = c_llEnt[e];
    av1g_pixel_t(BD) pred[16];
    ll_predict<BD>(I, plane, x4 * 4, y4 * 4, E.mode, E.delta, E.fi, haveL, haveA, E.ar, E.bl, edgeFilter, 0, pred);
    int16_t q[16];
    ll_levels<BD>(I, plane, x4 * 4, y4 * 4, pred, q);
    const float bits = ll_rate(q, plane > 0, g_cost[0]);
    L.cost[plane][(size_t)L.costOff[img][plane] + (size_t)pos * ne + e] = (uint16_t)min(65535.f, bits * 16.f + 0.5f);
}

// ---------------------------------------------------------------------------
// k_ll_shape
// ---------------------------------------------------------------------------
// availability of the above-right / below-left 4x4 of in-block 4x4 (i, j) (rows / cols in plane 4x4 units) of a
// block of bw4 x bh4 4x4s at plane pixel (x0, y0), as the analysis assumes it: exact inside the block, the
// open-loop guess (ol_decoded) at its corners
__device__ __forceinline__ void ll_shape_arbl(int i, int j, int bw4, int bh4, int x0, int y0, int sub, const TileBox &tb, int pw4,
                                              int ph4, int &ar, int &bl) {
    const int x = x0 + 4 * j, y = y0 + 4 * i;
    const int tx0 = tb.x0 >> sub, ty0 = tb.y0 >> sub, tx1 = tb.x1 >> sub, ty1 = tb.y1 >> sub;
    if (i > 0) ar = j + 1 < bw4 && (x >> 2) + 1 < pw4;
    else if (j + 1 < bw4) ar = y > ty0 && x + 4 < tx1 && (x >> 2) + 1 < pw4;
    else ar = ol_decoded(x + 4, y - 4, x0, y0, sub, tb);
    if (j > 0) bl = 0;
    else if (i + 1 < bh4) bl = x > tx0 && y + 4 < ty1 && (y >> 2) + 1 < ph4;
    else bl = ol_decoded(x - 4, y + 4, x0, y0, sub, tb);
}

template <int BD>
__global__ void __launch_bounds__(128) k_ll_shape(GpBatch B, GpLossless L, int bs, uint32_t nTotal, int fiOn) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const uint32_t gb = blockIdx.x * 4 + warp;
    if (gb >= nTotal) return;
    const int img = ll_find_image(B.shImgStart + (size_t)bs * (B.nimg + 1), B.nimg, gb);
    const GpImage &I = B.img[img];
    const uint32_t li = gb - B.shImgStart[(size_t)bs * (B.nimg + 1) + img];
    const int W = 4 * av1g_bw4(bs), H = 4 * av1g_bh4(bs);
    const int nbx = I.shNbx[bs];
    const int x0 = (int)(li % nbx) * W, y0 = (int)(li / nbx) * H;
    GpBlockRes &out = B.blk[I.shOff[bs] + li];
    if (x0 >= I.MiCols * 4 || y0 >= I.MiRows * 4) {
        if (lane == 0) { out.J = GP_INF; out.ncand = 0; }
        return;
    }
    const float *C = g_cost[0];
    const TileBox tb = tile_of(I, x0, y0);
    const int bw4 = W >> 2, bh4 = H >> 2;
    // ---- luma: lanes over the 66 candidates; the best (lowest index on ties) by warp reduction ----
    float bestJ = GP_INF;
    int bestC = 0;
    {
        const int pw4 = I.MiCols, ph4 = I.MiRows;
        const int nj = min(bw4, pw4 - (x0 >> 2)), ni = min(bh4, ph4 - (y0 >> 2));
        const uint16_t *ct = L.cost[0] + L.costOff[img][0];
        for (int c = lane; c < LL_NCAND_Y; c += 32) {
            const LLCand cd = c_llCand[c];
            float J = GP_INF;
            if (ll_cand_ok(bs, cd, fiOn)) {
                uint32_t s = 0;
                for (int i = 0; i < ni; i++)
                    for (int j = 0; j < nj; j++) {
                        int ar = 0, bl = 0;
                        if (cd.var) ll_shape_arbl(i, j, bw4, bh4, x0, y0, 0, tb, pw4, ph4, ar, bl);
                        const int e = cd.ent + (cd.var == 1 ? ar : cd.var == 2 ? bl : 0);
                        s += ct[((size_t)((y0 >> 2) + i) * pw4 + (x0 >> 2) + j) * LL_NE_Y + e];
                    }
                J = s * (1.f / 16.f) + ll_ymode_bits(bs, cd, fiOn, C);
            }
            if (J < bestJ) { bestJ = J; bestC = c; }
        }
        for (int o = 16; o > 0; o >>= 1) {
            const float oj = __shfl_xor_sync(0xffffffffu, bestJ, o);
            const int oc = __shfl_xor_sync(0xffffffffu, bestC, o);
            if (oj < bestJ || (oj == bestJ && oc < bestC)) { bestJ = oj; bestC = oc; }
        }
    }
    // ---- chroma: best uv mode (no CfL) over the chroma footprint, U + V ----
    float uvJ = 0.f;
    if (!I.mono) {
        const int sx = I.ssx, sy = I.ssy;
        // 4:2:0 4xN / Nx4: the chroma 4x4 column / row is shared by two blocks; count its cost by area
        const int cw = max(W >> sx, 4), ch = max(H >> sy, 4);
        const int cx0 = (x0 >> sx) & ~(cw - 1), cy0 = (y0 >> sy) & ~(ch - 1);
        const float frac = (float)(W * H) / (float)((cw << sx) * (ch << sy));
        const int pw4 = I.MiCols >> sx, ph4 = I.MiRows >> sy;
        const int cbw4 = cw >> 2, cbh4 = ch >> 2;
        const int nj = min(cbw4, pw4 - (cx0 >> 2)), ni = min(cbh4, ph4 - (cy0 >> 2));
        const int cflOk = L.cfl && av1g_cfl_allowed_ex(bs, 1, sx, sy);
        float best = GP_INF;
        for (int c = lane; c < LL_NCAND_UV; c += 32) {
            const LLCand cd = c_llCand[c];
            if (cd.delta && bs < AV1G_BLOCK_8X8) continue;   // angle_delta_uv uses the luma block size
            uint32_t s = 0;
            for (int p = 1; p < 3; p++) {
                const uint16_t *ct = L.cost[p] + L.costOff[img][p];
                for (int i = 0; i < ni; i++)
                    for (int j = 0; j < nj; j++) {
                        int ar = 0, bl = 0;
                        if (cd.var) ll_shape_arbl(i, j, cbw4, cbh4, cx0, cy0, sx, tb, pw4, ph4, ar, bl);
                        const int e = cd.ent + (cd.var == 1 ? ar : cd.var == 2 ? bl : 0);
                        s += ct[((size_t)((cy0 >> 2) + i) * pw4 + (cx0 >> 2) + j) * LL_NE_UV + e];
                    }
            }
            const float J = s * (1.f / 16.f) + ll_uvmode_bits(bs, AV1G_DC_PRED, cd.mode, cd.delta, cflOk, C);
            best = fminf(best, J);
        }
        if (cflOk) {
            // CfL (allowed for one 4x4 chroma tx block per plane): DC prediction + the best alpha per plane, luma = the
            // source of the block the chroma covers (4:2:0 sub-8x8: the 8x8)
            const LLTile tl = ll_tile(I, (cy0 << sy) >> 2, (cx0 << sx) >> 2);
            const int haveL = (cx0 >> 2) > (tl.c0 >> sx), haveA = (cy0 >> 2) > (tl.r0 >> sy);
            const int maxLumaW = min((cx0 + cw) << sx, I.MiCols * 4), maxLumaH = min((cy0 + ch) << sy, I.MiRows * 4);
            int ac[16];
            ll_cfl_ac<BD>(I, cx0, cy0, maxLumaW, maxLumaH, ac);
            float jc = ll_uvmode_bits(bs, AV1G_DC_PRED, AV1G_UV_CFL_PRED, 0, 1, C) + 3.f;   // + ~3 bits of joint signs
            for (int p = 1; p < 3; p++) {
                av1g_pixel_t(BD) dc[16];
                ll_predict<BD>(I, p, cx0, cy0, AV1G_DC_PRED, 0, -1, haveL, haveA, 0, 0, 0, 0, dc);
                float bp = GP_INF;
                for (int a = lane - 16; a <= 16; a += 32) {
                    av1g_pixel_t(BD) pr[16];
                    for (int i = 0; i < 16; i++) pr[i] = dc[i];
                    ll_cfl<BD>(a, ac, pr);
                    int16_t q[16];
                    ll_levels<BD>(I, p, cx0, cy0, pr, q);
                    float J = ll_rate(q, 1, C);
                    if (a) J += C[CID(cfl_alpha[0]) + (a < 0 ? -a : a) - 1];
                    bp = fminf(bp, J);
                }
                for (int o = 16; o > 0; o >>= 1) bp = fminf(bp, __shfl_xor_sync(0xffffffffu, bp, o));
                jc += bp;
            }
            best = fminf(best, jc);
        }
        for (int o = 16; o > 0; o >>= 1) best = fminf(best, __shfl_xor_sync(0xffffffffu, best, o));
        uvJ = best * frac;
    }
    if (lane == 0) {
        // the block's J in lambda units (k_dp adds lambda * partition bits); the best luma candidate (informational:
        // k_ll_decide decides again with exact availability)
        const float lambda = gp_lambda<8>(0, 1.f) * L.lambdaScale;
        out.J = (bestJ + uvJ + g_avgSkip0) * lambda;
        const LLCand cd = c_llCand[bestC];
        out.cand[0] = (uint8_t)(cd.fi >= 0 ? 0x68 + cd.fi : cand_pack(cd.mode, cd.delta));
        out.ncand = 1;
        out.tx = 0;
    }
}

// ---------------------------------------------------------------------------
// k_ll_decide: CTA per SB
// ---------------------------------------------------------------------------
#define LL_NT 128
struct LLSbShared {
    uint16_t ord[3][256];          // SB-local order maps (plane 4x4 units, row stride 16)
    uint16_t txb0[GP_MAX_LEAVES + 1];
    uint8_t lbs[GP_MAX_LEAVES], lx[GP_MAX_LEAVES], ly[GP_MAX_LEAVES];
    uint16_t nTxbLeaf[GP_MAX_LEAVES];
    float red[4][LL_NCAND_Y];
    int nLeaves;
};

// counts of the coded 4x4 tx blocks of a leaf, per plane
__device__ __forceinline__ void ll_leaf_counts(const GpImage &I, int bs, int miRow, int miCol, int &nY, int &nC) {
    nY = min(av1g_bw4(bs), I.MiCols - miCol) * min(av1g_bh4(bs), I.MiRows - miRow);
    nC = 0;
    if (av1g_has_chroma(bs, miRow, miCol, I.mono, I.ssx, I.ssy)) {
        const int cbs = av1g_uv_bsize(bs, I.ssx, I.ssy);
        const int bx = miCol >> I.ssx, by = miRow >> I.ssy;
        nC = min(av1g_bw4(cbs), (I.MiCols >> I.ssx) - bx) * min(av1g_bh4(cbs), (I.MiRows >> I.ssy) - by);
    }
}

template <int BD>
__global__ void __launch_bounds__(LL_NT) k_ll_decide(GpBatch B, GpLossless L, int fiOn) {
    __shared__ LLSbShared S;
    const uint32_t sb = blockIdx.x;
    const GpImage &I = B.img[B.sbImg[sb]];
    const int img = B.sbImg[sb];
    const uint32_t loc = sb - I.sbBase;
    const int sbc = (int)(loc % I.sbCols), sbr = (int)(loc / I.sbCols);
    const int sbMiR = sbr * 16, sbMiC = sbc * 16;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const float *C = g_cost[0];
    const int nl = B.leafCountW[sb];
    const GpLeaf *leaves = B.leaves + (size_t)sb * GP_MAX_LEAVES;
    for (int i = tid; i < 3 * 256; i += LL_NT) S.ord[i >> 8][i & 255] = 0xFFFF;
    for (int k = tid; k < nl; k += LL_NT) {
        const GpLeaf lf = leaves[k];
        S.lbs[k] = lf.bsize; S.lx[k] = lf.x4; S.ly[k] = lf.y4;
        int nY, nC;
        ll_leaf_counts(I, lf.bsize, sbMiR + lf.y4, sbMiC + lf.x4, nY, nC);
        S.nTxbLeaf[k] = (uint16_t)(nY + 2 * nC);
    }
    __syncthreads();
    if (tid == 0) {
        int t = 0;
        for (int k = 0; k < nl; k++) { S.txb0[k] = (uint16_t)t; t += S.nTxbLeaf[k]; }
        S.txb0[nl] = (uint16_t)t;
        S.nLeaves = nl;
        GpSbOut so;
        so.nBlocks = (uint32_t)nl; so.nTxb = (uint32_t)t; so.nCoeffs = (uint32_t)t * 16; so.pad = 0;
        B.sbOut[sb] = so;
    }
    __syncthreads();
    // ---- tx layout + order maps ----
    av1g_txb *oT = B.oTxb + (size_t)sb * B.sbTxbCap;
    for (int k = tid; k < nl; k += LL_NT) {
        const int bs = S.lbs[k], r = sbMiR + S.ly[k], c = sbMiC + S.lx[k];
        uint32_t t = S.txb0[k];
        B.oBlockTxb[(size_t)sb * GP_MAX_LEAVES + k] = (uint16_t)t;
        av1g_for_each_txb_ex(bs, r, c, 0, 1, I.mono, I.ssx, I.ssy, I.MiRows, I.MiCols, [&](int plane, int, int x4, int y4, int, int) {
            av1g_txb x;
            x.coeff_off = t * 16;
            x.x4 = (uint16_t)x4; x.y4 = (uint16_t)y4;
            x.plane = (uint8_t)plane; x.tx_size = AV1G_TX_4X4; x.tx_type = AV1G_DCT_DCT; x.flags = 0; x.eob = 0; x.reserved = 0;
            oT[t] = x;
            const int sx = plane ? I.ssx : 0, sy = plane ? I.ssy : 0;
            S.ord[plane][(y4 - (sbMiR >> sy)) * 16 + x4 - (sbMiC >> sx)] = (uint16_t)t;
            t++;
        });
    }
    __syncthreads();
    // ---- decisions: warp per leaf ----
    for (int k = warp; k < nl; k += LL_NT / 32) {
        const int bs = S.lbs[k], r = sbMiR + S.ly[k], c = sbMiC + S.lx[k];
        const LLTile tl = ll_tile(I, r, c);
        av1g_block blk;
        memset(&blk, 0, sizeof blk);
        blk.mi_row = (uint16_t)r; blk.mi_col = (uint16_t)c; blk.bsize = (uint8_t)bs;
        // luma: every candidate with exact availability
        const int nj = min(av1g_bw4(bs), I.MiCols - c), ni = min(av1g_bh4(bs), I.MiRows - r);
        const uint16_t *ctY = L.cost[0] + L.costOff[img][0];
        const int t0 = S.txb0[k];
        for (int cc = lane; cc < LL_NCAND_Y; cc += 32) {
            const LLCand cd = c_llCand[cc];
            float J = GP_INF;
            if (ll_cand_ok(bs, cd, fiOn)) {
                uint32_t s = 0;
                for (int i = 0; i < ni; i++)
                    for (int j = 0; j < nj; j++) {
                        int ar = 0, bl = 0;
                        if (cd.var) ll_ar_bl(0, I.ssx, I.ssy, c + j, r + i, t0 + i * nj + j, tl.r1, tl.c1, S.ord[0], ar, bl);
                        const int e = cd.ent + (cd.var == 1 ? ar : cd.var == 2 ? bl : 0);
                        s += ctY[((size_t)(r + i) * I.MiCols + c + j) * LL_NE_Y + e];
                    }
                J = s * (1.f / 16.f) + ll_ymode_bits(bs, cd, fiOn, C);
            }
            S.red[warp][cc] = J;
        }
        __syncwarp();
        int bestY = 0;
        {
            float bj = GP_INF;
            int bc = 0;
            for (int cc = lane; cc < LL_NCAND_Y; cc += 32)
                if (S.red[warp][cc] < bj) { bj = S.red[warp][cc]; bc = cc; }
            for (int o = 16; o > 0; o >>= 1) {
                const float oj = __shfl_xor_sync(0xffffffffu, bj, o);
                const int oc = __shfl_xor_sync(0xffffffffu, bc, o);
                if (oj < bj || (oj == bj && oc < bc)) { bj = oj; bc = oc; }
            }
            bestY = bc;
        }
        __syncwarp();
        const LLCand yc = c_llCand[bestY];
        blk.y_mode = (uint8_t)yc.mode;
        blk.angle_delta_y = (int8_t)yc.delta;
        blk.filter_intra = (uint8_t)(yc.fi >= 0 ? yc.fi + 1 : 0);
        // chroma
        const int hasChroma = av1g_has_chroma(bs, r, c, I.mono, I.ssx, I.ssy);
        if (hasChroma) {
            const int sx = I.ssx, sy = I.ssy;
            const int cbs = av1g_uv_bsize(bs, sx, sy);
            const int bx4 = c >> sx, by4 = r >> sy;
            const int pw4 = I.MiCols >> sx, ph4 = I.MiRows >> sy;
            const int cnj = min(av1g_bw4(cbs), pw4 - bx4), cni = min(av1g_bh4(cbs), ph4 - by4);
            const int tU = t0 + ni * nj, tV = tU + cni * cnj;
            const int cflOk = av1g_cfl_allowed_ex(bs, 1, sx, sy);
            float bj = GP_INF;
            int bc = 0;
            for (int cc = lane; cc < LL_NCAND_UV; cc += 32) {
                const LLCand cd = c_llCand[cc];
                if (cd.delta && bs < AV1G_BLOCK_8X8) continue;
                uint32_t s = 0;
                for (int p = 1; p < 3; p++) {
                    const uint16_t *ct = L.cost[p] + L.costOff[img][p];
                    for (int i = 0; i < cni; i++)
                        for (int j = 0; j < cnj; j++) {
                            int ar = 0, bl = 0;
                            if (cd.var)
                                ll_ar_bl(p, sx, sy, bx4 + j, by4 + i, (p == 1 ? tU : tV) + i * cnj + j, tl.r1, tl.c1, S.ord[p], ar, bl);
                            const int e = cd.ent + (cd.var == 1 ? ar : cd.var == 2 ? bl : 0);
                            s += ct[((size_t)(by4 + i) * pw4 + bx4 + j) * LL_NE_UV + e];
                        }
                }
                const float J = s * (1.f / 16.f) + ll_uvmode_bits(bs, yc.mode, cd.mode, cd.delta, cflOk, C);
                if (J < bj) { bj = J; bc = cc; }
            }
            for (int o = 16; o > 0; o >>= 1) {
                const float oj = __shfl_xor_sync(0xffffffffu, bj, o);
                const int oc = __shfl_xor_sync(0xffffffffu, bc, o);
                if (oj < bj || (oj == bj && oc < bc)) { bj = oj; bc = oc; }
            }
            blk.uv_mode = (uint8_t)c_llCand[bc].mode;
            blk.angle_delta_uv = (int8_t)c_llCand[bc].delta;
            if (cflOk && L.cfl) {
                // CfL on the single 4x4 chroma tx block per plane: exact DC prediction, alpha -16..16 per plane
                int aU, aL, aUC, aLC;
                av1g_block_avail(bs, r, c, I.mono, sx, sy, tl.r0, tl.r1, tl.c0, tl.c1, &aU, &aL, &aUC, &aLC);
                const int maxLumaW = min(c + av1g_bw4(bs), I.MiCols) * 4, maxLumaH = min(r + av1g_bh4(bs), I.MiRows) * 4;
                int ac[16];
                ll_cfl_ac<BD>(I, bx4 * 4, by4 * 4, maxLumaW, maxLumaH, ac);
                float bp[2];
                int ba[2];
                for (int p = 1; p < 3; p++) {
                    av1g_pixel_t(BD) dc[16];
                    ll_predict<BD>(I, p, bx4 * 4, by4 * 4, AV1G_DC_PRED, 0, -1, aLC, aUC, 0, 0, 0, 0, dc);
                    float best = GP_INF;
                    int ba1 = 0;
                    for (int a = lane - 16; a <= 16; a += 32) {
                        av1g_pixel_t(BD) pr[16];
                        for (int i = 0; i < 16; i++) pr[i] = dc[i];
                        ll_cfl<BD>(a, ac, pr);
                        int16_t q[16];
                        ll_levels<BD>(I, p, bx4 * 4, by4 * 4, pr, q);
                        float J = ll_rate(q, 1, C);
                        if (a) J += C[CID(cfl_alpha[0]) + (a < 0 ? -a : a) - 1];   // magnitude (context approximated)
                        if (J < best) { best = J; ba1 = a; }
                    }
                    for (int o = 16; o > 0; o >>= 1) {
                        const float oj = __shfl_xor_sync(0xffffffffu, best, o);
                        const int oa = __shfl_xor_sync(0xffffffffu, ba1, o);
                        if (oj < best || (oj == best && abs(oa) < abs(ba1))) { best = oj; ba1 = oa; }
                    }
                    bp[p - 1] = best;
                    ba[p - 1] = ba1;
                }
                if (ba[0] || ba[1]) {
                    const int su = ba[0] == 0 ? 0 : ba[0] < 0 ? 1 : 2, sv = ba[1] == 0 ? 0 : ba[1] < 0 ? 1 : 2;
                    const float Jc = bp[0] + bp[1] + ll_uvmode_bits(bs, yc.mode, AV1G_UV_CFL_PRED, 0, 1, C) +
                                     C[CID(cfl_sign) + su * 3 + sv - 1];
                    if (Jc < bj) {
                        blk.uv_mode = AV1G_UV_CFL_PRED;
                        blk.angle_delta_uv = 0;
                        blk.cfl_alpha_u = (int8_t)ba[0];
                        blk.cfl_alpha_v = (int8_t)ba[1];
                    }
                }
            }
        }
        if (lane == 0) {
            B.oBlocks[(size_t)sb * GP_MAX_LEAVES + k] = blk;
            const int rEnd = min(r + av1g_bh4(bs), I.MiRows), cEnd = min(c + av1g_bw4(bs), I.MiCols);
            for (int y = r; y < rEnd; y++)
                for (int x = c; x < cEnd; x++) {
                    I.mYMode[(size_t)y * I.MiCols + x] = blk.y_mode;
                    if (hasChroma) I.mUVMode[(size_t)y * I.MiCols + x] = blk.uv_mode;
                }
        }
        __syncwarp();
    }
}

// ---------------------------------------------------------------------------
// k_ll_coeffs: CTA per SB
// ---------------------------------------------------------------------------
template <int BD>
__global__ void __launch_bounds__(LL_NT) k_ll_coeffs(GpBatch B, int edgeFilter) {
    __shared__ uint16_t ord[3][256];
    __shared__ uint8_t txbBlk[GP_MAX_TXB_444];
    __shared__ int nz[GP_MAX_LEAVES];
    const uint32_t sb = blockIdx.x;
    const GpImage &I = B.img[B.sbImg[sb]];
    const uint32_t loc = sb - I.sbBase;
    const int sbMiR = (int)(loc / I.sbCols) * 16, sbMiC = (int)(loc % I.sbCols) * 16;
    const int tid = threadIdx.x;
    const GpSbOut so = B.sbOut[sb];
    const av1g_block *blocks = B.oBlocks + (size_t)sb * GP_MAX_LEAVES;
    const uint16_t *bt = B.oBlockTxb + (size_t)sb * GP_MAX_LEAVES;
    const av1g_txb *T = B.oTxb + (size_t)sb * B.sbTxbCap;
    int16_t *Q = B.oCoeffs + (size_t)sb * B.sbCoefCap;
    for (int i = tid; i < 3 * 256; i += LL_NT) ord[i >> 8][i & 255] = 0xFFFF;
    for (uint32_t k = tid; k < so.nBlocks; k += LL_NT) {
        nz[k] = 0;
        const uint32_t e = k + 1 < so.nBlocks ? bt[k + 1] : so.nTxb;
        for (uint32_t t = bt[k]; t < e; t++) txbBlk[t] = (uint8_t)k;
    }
    __syncthreads();
    for (uint32_t t = tid; t < so.nTxb; t += LL_NT) {
        const av1g_txb x = T[t];
        const int sx = x.plane ? I.ssx : 0, sy = x.plane ? I.ssy : 0;
        ord[x.plane][(x.y4 - (sbMiR >> sy)) * 16 + x.x4 - (sbMiC >> sx)] = (uint16_t)t;
    }
    __syncthreads();
    for (uint32_t t = tid; t < so.nTxb; t += LL_NT) {
        const av1g_txb x = T[t];
        const int k = txbBlk[t];
        const av1g_block b = blocks[k];
        const int plane = x.plane, sx = plane ? I.ssx : 0, sy = plane ? I.ssy : 0;
        const LLTile tl = ll_tile(I, b.mi_row, b.mi_col);
        int aU, aL, aUC, aLC;
        av1g_block_avail(b.bsize, b.mi_row, b.mi_col, I.mono, I.ssx, I.ssy, tl.r0, tl.r1, tl.c0, tl.c1, &aU, &aL, &aUC, &aLC);
        const int xi = x.x4 - (b.mi_col >> sx), yi = x.y4 - (b.mi_row >> sy);
        const int haveL = (plane ? aLC : aL) || xi > 0, haveA = (plane ? aUC : aU) || yi > 0;
        int ar, bl;
        ll_ar_bl(plane, I.ssx, I.ssy, x.x4, x.y4, (int)t, tl.r1, tl.c1, ord[plane], ar, bl);
        const bool cfl = plane && b.uv_mode == AV1G_UV_CFL_PRED;
        const int mode = plane == 0 ? b.y_mode : cfl ? AV1G_DC_PRED : b.uv_mode;
        const int delta = plane == 0 ? b.angle_delta_y : b.angle_delta_uv;
        const int fi = plane == 0 && b.filter_intra ? b.filter_intra - 1 : -1;
        const int ft = (edgeFilter && av1g_is_directional(mode))
                           ? av1g_get_filter_type(plane, b.mi_row, b.mi_col, plane ? aUC : aU, plane ? aLC : aL, I.mYMode, I.mUVMode,
                                                  I.MiCols, I.ssx, I.ssy)
                           : 0;
        av1g_pixel_t(BD) pred[16];
        ll_predict<BD>(I, plane, x.x4 * 4, x.y4 * 4, mode, delta, fi, haveL, haveA, ar, bl, edgeFilter, ft, pred);
        if (cfl) {
            const int maxLumaW = min(b.mi_col + av1g_bw4(b.bsize), I.MiCols) * 4;
            const int maxLumaH = min(b.mi_row + av1g_bh4(b.bsize), I.MiRows) * 4;
            int ac[16];
            ll_cfl_ac<BD>(I, x.x4 * 4, x.y4 * 4, maxLumaW, maxLumaH, ac);
            ll_cfl<BD>(plane == 1 ? b.cfl_alpha_u : b.cfl_alpha_v, ac, pred);
        }
        int16_t q[16];
        ll_levels<BD>(I, plane, x.x4 * 4, x.y4 * 4, pred, q);
        int any = 0;
        for (int i = 0; i < 16; i++) { Q[(size_t)t * 16 + i] = q[i]; any |= q[i]; }
        if (any) nz[k] = 1;   // benign race: every writer stores 1
    }
    __syncthreads();
    for (uint32_t k = tid; k < so.nBlocks; k += LL_NT) {
        const int skip = !nz[k];
        B.oBlocks[(size_t)sb * GP_MAX_LEAVES + k].skip = (uint8_t)skip;
        atomicAdd(&B.stats[0], 1u);
        if (skip) atomicAdd(&B.stats[1], 1u);
        if (blocks[k].uv_mode == AV1G_UV_CFL_PRED &&
            av1g_has_chroma(blocks[k].bsize, blocks[k].mi_row, blocks[k].mi_col, I.mono, I.ssx, I.ssy))
            atomicAdd(&B.stats[2], 1u);
    }
}

// ---------------------------------------------------------------------------
// launchers
// ---------------------------------------------------------------------------
void gp_launch_ll_analysis(const GpBatch &B, const GpLossless &L, const uint32_t *shTotal, int edgeFilter, int fiOn, cudaStream_t st,
                           int bd) {
    const uint64_t total = L.hostTotal;
    if (total) {
        const unsigned grid = (unsigned)((total + 127) / 128);
        if (bd == 10) k_ll_cost<10><<<grid, 128, 0, st>>>(B, L, edgeFilter);
        else k_ll_cost<8><<<grid, 128, 0, st>>>(B, L, edgeFilter);
    }
    for (int s = 0; s < GP_NSHAPES; s++) {
        if (!shTotal[s] || s >= AV1G_BLOCK_INVALID || (s >= 13 && s <= 15)) continue;
        if (bd == 10) k_ll_shape<10><<<(shTotal[s] + 3) / 4, 128, 0, st>>>(B, L, s, shTotal[s], fiOn);
        else k_ll_shape<8><<<(shTotal[s] + 3) / 4, 128, 0, st>>>(B, L, s, shTotal[s], fiOn);
    }
}

void gp_launch_ll_recon(const GpBatch &B, const GpLossless &L, int edgeFilter, int fiOn, cudaStream_t st, int bd) {
    if (!B.nSb) return;
    if (bd == 10) {
        k_ll_decide<10><<<B.nSb, LL_NT, 0, st>>>(B, L, fiOn);
        k_ll_coeffs<10><<<B.nSb, LL_NT, 0, st>>>(B, edgeFilter);
    } else {
        k_ll_decide<8><<<B.nSb, LL_NT, 0, st>>>(B, L, fiOn);
        k_ll_coeffs<8><<<B.nSb, LL_NT, 0, st>>>(B, edgeFilter);
    }
}
