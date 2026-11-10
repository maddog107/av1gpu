// closed-loop decision + reconstruction, one CTA (4 warps) per 64x64 superblock,
// launched in SB-wavefront steps (step = sbCol + 2 * sbRow: left and above-right neighbours are
// complete).  Inside the SB the blocks are processed in decode order exactly as the decoder does:
// BlockDecoded flags, real reconstructed edges, exact contexts from the 4x4 maps.
//
// Per block:
//   luma    re-evaluate the analysis top-K (DCT_DCT) with the REAL edges and exact mode/coefficient
//           contexts, tx-type search for the winner, final quantization, exact dequant + inverse
//           transform (av1core dsp.h) + reconstruction
//   chroma  SATD screening of the 13 UV modes with real edges -> top uvK + CfL (least-squares alpha
//           from the reconstructed luma) -> exact RD of U+V -> final quantization + reconstruction
//           (4:2:0 or 4:4:4: the kernel and its chroma helpers are instantiated per subsampling SS - 1 for
//           4:2:0 (and monochrome), 0 for 4:4:4 - so the 4:2:0 code has no runtime layout checks; 4:4:4
//           blocks with a 64-sample side have several chroma tx blocks per plane, decided on the first
//           one, then coded in raster order)
//   record  av1g_block / av1g_txb / coefficients into the SB's output slots, maps + contexts
//
// Bit depth: the kernel and every helper touching samples are templates on BD (8 or 10; samples
// av1g_pixel_t(BD)); the 8-bit instantiations are the 8-bit code.  Distortions are in native units with a
// native-unit lambda (gp_lambda<BD>).
#pragma once
#include <cstddef>
#include "dev_tx.cuh"
#include "k_analysis.cuh"
#include "pal_dev.cuh"
#include "rc_eval.cuh"
#include "rc_inv.cuh"

#define RC_NT 128
#define RC_NW 4

struct TxR {
    int w, h, tw, th, n, lw, lh;
    __device__ TxR(int W, int H) : w(W), h(H), tw(W > 32 ? 32 : W), th(H > 32 ? 32 : H), n(0), lw(log2i(W)), lh(log2i(H)) { n = tw * th; }
};

// Per-warp evaluation scratch for tx blocks up to 16x16 (all sizes with a tx-type search); larger
// tx blocks (32/64-point, DCT only) are evaluated one at a time by warp 0 in the shared big scratch.
struct RcWarpScr {
    int16_t res[16 * 16];
    float tmp[16 * 17];
    int16_t q[16 * 16];
};
struct RcBigScr {
    int16_t res[64 * 64];
    float tmp[32 * 65];
    int16_t q[32 * 32];
};
struct RcInvScr {
    int16_t res[64 * 64];   // residual after the row pass / final residual (both fit 16 bits, spec 7.13.3)
};
#define RC_TASKQ_OFF_C (sizeof(RcWarpScr) * RC_NW)
#define RC_TAIL_OFF (RC_TASKQ_OFF_C + 14 * 512)
// shared-memory union (warp scratches / inverse transform buffer / kept levels); the rare 32/64-point
// evaluations use a per-CTA slot of B.bigScratch in global memory (GP_RC_BIG_BYTES each)
#define RC_UNION_BYTES (RC_TAIL_OFF + 2048)
union RcUnion {
    RcWarpScr w[RC_NW];
    RcInvScr inv;
    uint8_t raw[RC_UNION_BYTES];
};
static_assert(sizeof(RcBigScr) <= GP_RC_BIG_BYTES, "GP_RC_BIG_BYTES too small");
// W5/W3: per-CTA global scratch = one big slot per warp (big tasks run on all warps) + kept
// levels of up to RC_BIGQ_MAX big chroma RD tasks (the final reuses them instead of re-quantizing)
#define RC_BIGQ_MAX 8
#define RC_CTA_BIG_BYTES ((size_t)RC_NW * sizeof(RcBigScr) + (size_t)RC_BIGQ_MAX * 32 * 32 * 2)
__device__ __forceinline__ bool rc_big(int W, int H) { return W > 16 || H > 16; }
// chroma RD task levels (tx <= 16x16) are kept in the union after the warp scratches: above the inverse
// transform's int16 buffer (8192 B) and below the union's end
#define RC_TASKQ_OFF (sizeof(RcWarpScr) * RC_NW)
#define RC_TASKQ_MAX ((int)((RC_TAIL_OFF - RC_TASKQ_OFF) / 512))
__device__ __forceinline__ int16_t *rc_taskq(RcUnion &u, int t) { return (int16_t *)((char *)&u + RC_TASKQ_OFF) + t * 256; }
// [16384, 18432) of the union: luma per-warp best levels (luma passes, tx <= 16x16) or CfL luma values
// (CfL-capable blocks: chroma tx <= 16x16 in 4:2:0, <= 32x32 in 4:4:4 = 1024 values, the full 2 KB).  Phase-disjoint uses.
__device__ __forceinline__ int16_t *rc_bestq(RcUnion &u, int w) { return (int16_t *)((char *)&u + RC_TAIL_OFF) + w * 256; }
__device__ __forceinline__ const int16_t *rc_cflL(const RcUnion &u) { return (const int16_t *)((const char *)&u + RC_TAIL_OFF); }
__device__ __forceinline__ int16_t *rc_cflLw(RcUnion &u) { return (int16_t *)((char *)&u + RC_TAIL_OFF); }

template <int BD>
struct RcShared {
    typedef av1g_pixel_t(BD) Px;
    RcUnion u;
    int16_t qfin[32 * 32];
    RcBigScr *bigG;                 // this CTA's global big scratch: one RcBigScr per warp (W5)
    int16_t *bigQ;                  // kept levels of big chroma RD tasks [RC_BIGQ_MAX][32 * 32] (W3)
    uint32_t bdm[3][18];            // BlockDecoded bitmask: row yy+1, bit xx+1 (yy, xx in -1..16)
    __align__(16) Px ea[3][AV1G_EDGE_LEN];   // 16-byte aligned rows: rc_prepare_edges copies them in 16-byte words
    __align__(16) Px el[3][AV1G_EDGE_LEN];
    int dcv[3];
    // intra edge filter: per-warp prepared (filtered / upsampled) edges of the candidate being evaluated
    __align__(16) Px wea[RC_NW][AV1G_EDGE_LEN];
    __align__(16) Px wel[RC_NW][AV1G_EDGE_LEN];
    int wup[RC_NW][2];
    int ft[3], hvL[3], hvA[3], px[3], py[3], mx[3], my[3];
    int edgeFilter;
    float taskJ[32];
    union { float satd[32]; int satdI[32]; };   // satdI: integer totals of the >= 16x16 SATD screening
    int uvc[32];
    int taskC[32], taskT[32], taskP[32], taskE[32];
    int nTask;
    float finD, finR;               // rc_final_quant: winner's distortion / rate (tx-size search)
    int errFlag[2], nzFlag[2];      // cta_inverse_add, per RcGrp half
    int eobUV[2];                   // concurrent chroma finals: eob of U / V
    int nTxb, nCoef;
    int finEob;
    int cflAvg, cflAlpha[2];
    // per-warp best luma task (tx blocks <= 16x16): its levels are the final ones, no re-evaluation
    float bestJw[RC_NW];
    int bestCw[RC_NW], bestTw[RC_NW], bestEw[RC_NW];
    // luma palette candidate of the current block (screen-content frames; pal_cand): colours, size (0: none), bits, its
    // index map (in the SB's colour-map slot, at nMap), and the SB's palettes so far
    uint16_t palCol[8];
    int palN;
    float palBits;
    const uint8_t *palMap;
    int nPal, nMap;
};

// ---------------------------------------------------------------------------
// threads of a CTA-wide helper: the CTA (barrier 0), or one half of it for the concurrent U / V chroma finals
// (threads 0-63 / 64-127 on the named barriers 2 / 3, the ids of rc_edges' chroma halves; immediates, see rc_edges)
struct RcGrp {
    int t, nt, k, bar;   // thread index in the group, group size, half (0: CTA or U, 1: V), barrier id
    __device__ __forceinline__ void sync() const {
        if (bar == 0) __syncthreads();
        else if (bar == 2) asm volatile("bar.sync 2, 64;" ::: "memory");
        else asm volatile("bar.sync 3, 64;" ::: "memory");
    }
};
__device__ __forceinline__ RcGrp rc_cta() { return RcGrp{(int)threadIdx.x, RC_NT, 0, 0}; }
__device__ __forceinline__ RcGrp rc_half() { const int k = threadIdx.x >> 6; return RcGrp{(int)threadIdx.x & 63, 64, k, 2 + k}; }

// exact inverse transform + reconstruction (group g), mirrors av1g_inv_txfm2d / av1g_add_residual.
// q may be modified (halved) when the coefficients violate the conformance ranges
// (= AV1G_ENC_FIX_RANGE).  Returns the number of halvings (all threads of g).  Half k uses the residual
// buffer S.u.inv.res + k * 2048 (halves: chroma tx blocks, <= 32x32).
template <int BD>
__device__ int cta_inverse_add(int W, int H, RcShared<BD> &S, int16_t *q, int txSz, int txType, int dcQ, int acQ, const uint8_t *qm,
                               av1g_pixel_t(BD) *dst, int pitch, const RcGrp g) {   // g by value (a non-inlined call)
    const TxR T(W, H);
    const int tid = g.t, nt = g.nt;
    const int den = av1g_dq_denom(txSz);
    const int rk = av1g_row_kind(txType), ck = av1g_col_kind(txType);
    const int rowShift = AV1G_T(Transform_Row_Shift)[txSz];
    const bool rect = (T.lw - T.lh == 1 || T.lh - T.lw == 1);
    int16_t *res = S.u.inv.res + g.k * 2048;
    int &errFlag = S.errFlag[g.k], &nzFlag = S.nzFlag[g.k];
    int fixes = 0;
    for (;;) {
        if (tid == 0) { errFlag = 0; nzFlag = 0; }
        g.sync();
        // W4: register-resident templated 1-D transforms (rc_inv.cuh), same arithmetic as av1g_inv_1d
        uint32_t err = 0;
        switch (T.lw) {
#define RC_ROWS(LW) for (int i = tid; i < H; i += nt) err |= rci::row<LW, BD>(i, T.th, T.tw, q, dcQ, acQ, qm, den, rect, rk, rowShift, res)
        case 2: RC_ROWS(2); break;
        case 3: RC_ROWS(3); break;
        case 4: RC_ROWS(4); break;
        case 5: RC_ROWS(5); break;
        default: RC_ROWS(6); break;
#undef RC_ROWS
        }
        if (err) atomicOr(&errFlag, (int)err);
        g.sync();
        err = 0;
        switch (T.lh) {
#define RC_COLS(LH) for (int j = tid; j < W; j += nt) err |= rci::col<LH>(j, W, ck, res)
        case 2: RC_COLS(2); break;
        case 3: RC_COLS(3); break;
        case 4: RC_COLS(4); break;
        case 5: RC_COLS(5); break;
        default: RC_COLS(6); break;
#undef RC_COLS
        }
        if (err) atomicOr(&errFlag, (int)err);
        g.sync();
        if (!errFlag) break;
        fixes++;
        for (int idx = tid; idx < T.n; idx += nt) {
            q[idx] = (int16_t)(q[idx] / 2);
            if (q[idx]) nzFlag = 1;
        }
        g.sync();
        if (!nzFlag) return fixes;  // all zero now: no residual
    }
    const int fud = av1g_flip_ud(txType), flr = av1g_flip_lr(txType);
    for (int p = tid; p < W * H; p += nt) {
        const int i = p >> T.lw, j = p & (W - 1);
        const int xx = flr ? W - 1 - j : j, yy = fud ? H - 1 - i : i;
        av1g_pixel_t(BD) *d = dst + yy * pitch + xx;
        *d = (av1g_pixel_t(BD))av1g_clip1<BD>((int)*d + res[i * W + j]);
    }
    g.sync();
    return fixes;
}

// ---------------------------------------------------------------------------
// coefficient contexts from the 4x4 maps (mirror of av1core k_txsym).  Every warp (full) computes them: lane i
// loads above / left entry i (w4, h4 <= 16) at once, and one packed sum gives the counts the contexts depend on
// (entries != 0 and > 3 per side, dc categories 1 and 2), so the serial loops' values in all lanes.
template <int SS>   // chroma subsampling (1: 4:2:0, 0: 4:4:4); plane 0 does not depend on it
__device__ __forceinline__ void txb_ctx(const GpImage &I, int plane, int txSz, int x4, int y4, int bsize, const TileBox &tb,
                                        int *skipCtx, int *dcCtx) {
    const int subX = plane ? SS : 0, subY = plane ? SS : 0;
    const int stride = I.MiCols >> subX;
    const int maxX4 = I.MiCols >> subX, maxY4 = I.MiRows >> subY;
    const uint8_t *L = I.cLvl[plane], *D = I.cDc[plane];
    const bool hasAbove = y4 > ((tb.y0 >> 2) >> subY), hasLeft = x4 > ((tb.x0 >> 2) >> subX);
    const int w = AV1G_T(Tx_Width)[txSz], h = AV1G_T(Tx_Height)[txSz], w4 = w >> 2, h4 = h >> 2;
    const int lane = threadIdx.x & 31;
    const bool a = hasAbove && lane < w4 && x4 + lane < maxX4, l = hasLeft && lane < h4 && y4 + lane < maxY4;
    const int la = a ? L[(size_t)(y4 - 1) * stride + x4 + lane] : 0, da = a ? D[(size_t)(y4 - 1) * stride + x4 + lane] : 0;
    const int ll = l ? L[(size_t)(y4 + lane) * stride + x4 - 1] : 0, dl = l ? D[(size_t)(y4 + lane) * stride + x4 - 1] : 0;
    // [0,5) above != 0, [5,10) above > 3, [10,15) left != 0, [15,20) left > 3, [20,26) dc 2, [26,32) dc 1 (<= 16, 32)
    uint32_t f = plane ? ((la | da) != 0) | (uint32_t)((ll | dl) != 0) << 10
                       : (la != 0) | (uint32_t)(la > 3) << 5 | (uint32_t)(ll != 0) << 10 | (uint32_t)(ll > 3) << 15;
    f |= (uint32_t)((da == 2) + (dl == 2)) << 20 | (uint32_t)((da == 1) + (dl == 1)) << 26;
    for (int o = 16; o > 0; o >>= 1) f += __shfl_xor_sync(0xffffffffu, f, o);
    const bool nzA = f & 31, nzL = (f >> 10) & 31;
    if (plane == 0) {
        const int bw = bw_px(bsize), bh = bh_px(bsize);
        const bool gtA = (f >> 5) & 31, gtL = (f >> 15) & 31;
        if (bw == w && bh == h) *skipCtx = 0;
        else if (!nzA && !nzL) *skipCtx = 1;
        else if (!nzA || !nzL) *skipCtx = 2 + (gtA || gtL);
        else if (!gtA && !gtL) *skipCtx = 4;
        else if (!gtA || !gtL) *skipCtx = 5;
        else *skipCtx = 6;
    } else {
        const int ub = av1g_uv_bsize(bsize, SS, SS);
        const int bw = bw_px(ub), bh = bh_px(ub);
        *skipCtx = nzA + nzL + 7 + ((bw * bh > w * h) ? 3 : 0);
    }
    const int dcSign = (int)((f >> 20) & 63) - (int)(f >> 26);
    *dcCtx = dcSign < 0 ? 1 : dcSign > 0 ? 2 : 0;
}

// prediction of one pixel: mode candidate c (or CfL with alpha when cfl), plane edges.  slot >= 0:
// directional modes use the edges prepared (intra edge filter) by warp `slot`.
template <int BD>
__device__ __forceinline__ int rc_pred(const RcShared<BD> &S, int plane, int c, bool cfl, int alpha, int i, int j, int lw, int lh,
                                       int cw, int slot = -1) {
    if (cfl) {
        const int v = rc_cflL(S.u)[i * cw + j];
        return av1g_clip1<BD>(S.dcv[plane] + av1g_round2signed(alpha * (v - S.cflAvg), 6));
    }
    const int m = cand_mode(c);
    if (m == AV1G_DC_PRED) return S.dcv[plane];
    if (slot >= 0 && S.edgeFilter && av1g_is_directional(m))
        return av1g_pred_pixel_up(m, pangle_of(m, cand_delta(c)), i, j, lw, lh, S.wea[slot] + AV1G_EDGE_OFF,
                                  S.wel[slot] + AV1G_EDGE_OFF, S.wup[slot][0], S.wup[slot][1]);
    return av1g_pred_pixel(m, pangle_of(m, cand_delta(c)), i, j, lw, lh, S.ea[plane] + AV1G_EDGE_OFF, S.el[plane] + AV1G_EDGE_OFF);
}

// filter-intra candidates are coded as 128 + filter_intra_mode (y_mode DC_PRED), a luma palette as PAL_CAND + size - 2
// (DC_PRED; pal_dev.cuh)
__device__ __forceinline__ bool is_fi(int c) { return c >= 128 && c < 133; }
__device__ __forceinline__ int luma_mode(int c) { return c >= 128 ? AV1G_DC_PRED : cand_mode(c); }
__device__ __forceinline__ int intra_dir(int c) {
    return is_fi(c) ? AV1G_T(Filter_Intra_Mode_To_Intra_Dir)[c - 128] : pal_cand(c) ? AV1G_DC_PRED : cand_mode(c);
}

// warp: prepare the edge-filtered edges of (plane, candidate) into this warp's slot (lane 0)
static_assert(AV1G_EDGE_LEN % 16 == 0 && offsetof(RcShared<8>, ea) % 16 == 0 && offsetof(RcShared<8>, el) % 16 == 0 &&
              offsetof(RcShared<8>, wea) % 16 == 0 && offsetof(RcShared<8>, wel) % 16 == 0, "edge buffers must be 16-byte aligned");
static_assert(offsetof(RcShared<10>, ea) % 16 == 0 && offsetof(RcShared<10>, el) % 16 == 0 && offsetof(RcShared<10>, wea) % 16 == 0 &&
              offsetof(RcShared<10>, wel) % 16 == 0, "edge buffers must be 16-byte aligned");
// lane-group-parallel av1g_edge_filter / av1g_edge_upsample (dsp.h), the same integer arithmetic: every lane reads its
// outputs' inputs (the unfiltered edge) into registers, syncs the group, then writes.  Called by whole groups (a warp,
// or a half-warp for tx blocks up to 8x8: sz <= 17 there).
template <int G, typename P>
__device__ __forceinline__ void grp_edge_filter(P *buf, int sz, int strength, const RcLanes<G> &L) {   // sz <= 4 G + 1
    if (!strength) return;
    constexpr int NK = G == 32 ? 4 : 2;   // G = 32: outputs 1 .. 128; G = 16: 1 .. 32
    const int lane = L.l;
    const uint8_t *K = AV1G_T(Intra_Edge_Kernel)[strength - 1];
    int v[NK];
#pragma unroll
    for (int k = 0; k < NK; k++) {
        const int i = lane + 1 + G * k;
        int s = 0;
        if (i < sz) {
#pragma unroll
            for (int j = 0; j < AV1G_INTRA_EDGE_TAPS; j++) s += K[j] * buf[av1g_clip3(0, sz - 1, i - 2 + j) - 1];
        }
        v[k] = (s + 8) >> 4;
    }
    L.sync();
#pragma unroll
    for (int k = 0; k < NK; k++)
        if (lane + 1 + G * k < sz) buf[lane + G * k] = (P)v[k];
    L.sync();
}
template <int BD, int G, typename P>
__device__ __forceinline__ void grp_edge_upsample(P *buf, int numPx, const RcLanes<G> &L) {   // numPx <= 16 (upsampling: w + h <= 16)
    const int lane = L.l;
    // dup[0] = buf[-1], dup[k] = buf[k - 2] (k = 1 .. numPx + 1), dup[numPx + 2] = buf[numPx - 1]
    auto dup = [&](int k) -> int { return k == 0 ? buf[-1] : k == numPx + 2 ? buf[numPx - 1] : buf[k - 2]; };
    int s = 0, e = 0;
    if (lane < numPx) {
        s = av1g_clip1<BD>(av1g_round2(-dup(lane) + 9 * dup(lane + 1) + 9 * dup(lane + 2) - dup(lane + 3), 4));
        e = dup(lane + 2);
    }
    const P c = buf[-1];
    L.sync();
    if (lane < numPx) { buf[2 * lane - 1] = (P)s; buf[2 * lane] = (P)e; }
    if (lane == 0) buf[-2] = c;
    L.sync();
}
// lane-group-parallel av1g_edge_prepare (the corner by lane 0); every lane gets the upsampling flags
template <int BD, int G, typename P>
__device__ __forceinline__ void grp_edge_prepare(int pAngle, int log2W, int log2H, int filterType, int haveLeft, int haveAbove, int x, int y,
                                                 int maxX, int maxY, P *above, P *left, int *upA, int *upL, const RcLanes<G> &L) {
    const int w = 1 << log2W, h = 1 << log2H;
    if (pAngle != 90 && pAngle != 180) {
        if (pAngle > 90 && pAngle < 180 && (w + h) >= 24) {
            if (L.l == 0) {
                const P c = (P)av1g_round2(left[0] * 5 + above[-1] * 6 + above[0] * 5, 4);
                left[-1] = c;
                above[-1] = c;
            }
            L.sync();
        }
        if (haveAbove)
            grp_edge_filter<G>(above, (w < maxX - x + 1 ? w : maxX - x + 1) + (pAngle < 90 ? h : 0) + 1,
                               av1g_edge_strength(w, h, filterType, pAngle - 90), L);
        if (haveLeft)
            grp_edge_filter<G>(left, (h < maxY - y + 1 ? h : maxY - y + 1) + (pAngle > 180 ? w : 0) + 1,
                               av1g_edge_strength(w, h, filterType, pAngle - 180), L);
    }
    *upA = av1g_edge_use_upsample(w, h, filterType, pAngle - 90);
    if (*upA) grp_edge_upsample<BD, G>(above, w + (pAngle < 90 ? h : 0), L);
    *upL = av1g_edge_use_upsample(w, h, filterType, pAngle - 180);
    if (*upL) grp_edge_upsample<BD, G>(left, h + (pAngle > 180 ? w : 0), L);
}

template <int BD>
__device__ __forceinline__ void rc_prepare_edges(RcShared<BD> &S, int plane, int c, bool cfl, int lw, int lh) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int m = cand_mode(c);
    if (cfl || !S.edgeFilter || !av1g_is_directional(m)) return;
    // copy in 16-byte words (same bytes): nvcc's vectorization of the byte loop depends on the surrounding
    // code; compiled byte by byte it cost the closed loop ~5 % (measured on an L4)
    constexpr int NW = AV1G_EDGE_LEN * (int)sizeof(av1g_pixel_t(BD)) / 16;
    const uint4 *sa = reinterpret_cast<const uint4 *>(S.ea[plane]), *sl = reinterpret_cast<const uint4 *>(S.el[plane]);
    uint4 *da = reinterpret_cast<uint4 *>(S.wea[warp]), *dl = reinterpret_cast<uint4 *>(S.wel[warp]);
    for (int k = lane; k < 2 * NW; k += 32) {
        if (k < NW) da[k] = sa[k];
        else dl[k - NW] = sl[k - NW];
    }
    __syncwarp();
    int ua = 0, ul = 0;
    grp_edge_prepare<BD, 32>(pangle_of(m, cand_delta(c)), lw, lh, S.ft[plane], S.hvL[plane], S.hvA[plane], S.px[plane], S.py[plane], S.mx[plane],
                             S.my[plane], S.wea[warp] + AV1G_EDGE_OFF, S.wel[warp] + AV1G_EDGE_OFF, &ua, &ul, rc_warp_lanes());
    if (lane == 0) {
        S.wup[warp][0] = ua;
        S.wup[warp][1] = ul;
    }
    __syncwarp();
}

// ---------------------------------------------------------------------------
// two tasks per warp (tx blocks up to 8x8, more tasks than warps): half-warp h of warp w works in the first or second
// half of S.u.w[w] (residual, levels: 64 values each; transform: 8 x 9 floats each) and prepares its edges in the
// rest of w's tmp: 64 samples per edge, enough for blocks up to 8x8 (prepared edges reach index 2 (w + h) - 2 <= 30,
// the filter and the upsampling index -2 .. 30; a copy starts AV1G_EDGE_OFF below index 0)
#define RC_HALF_EDGE 64
struct RcHalfScr {
    int16_t *res, *q;
    float *tmp;
    void *ea, *el;   // av1g_pixel_t(BD)[RC_HALF_EDGE] each
};
template <int BD>
__device__ __forceinline__ RcHalfScr rc_half_scr(RcShared<BD> &S) {
    RcWarpScr &w = S.u.w[threadIdx.x >> 5];
    const int h = (threadIdx.x >> 4) & 1;
    constexpr int EB = RC_HALF_EDGE * (int)sizeof(av1g_pixel_t(BD));   // bytes per edge
    static_assert(2 * 8 * 9 * 4 + 2 * 2 * RC_HALF_EDGE * 2 <= (int)sizeof(w.tmp) && 2 * 64 <= 16 * 16, "half-warp scratch");
    char *e = (char *)(w.tmp + 2 * 8 * 9) + h * 2 * EB;   // 16-byte aligned: RcWarpScr is 2112 B, tmp at 512, + 576
    return RcHalfScr{w.res + h * 64, w.q + h * 64, w.tmp + h * 8 * 9, e, e + EB};
}
static_assert(sizeof(RcWarpScr) % 16 == 0 && offsetof(RcWarpScr, tmp) % 16 == 0, "half-warp edge slots must be 16-byte aligned");

// half-warp: residual of (plane, cand / CfL) for a W x H tx block (up to 8x8) into Hs.res, the prediction of rc_residual
// (rc_pred: DC, CfL, filtered directional edges in the half's own slot, the unfiltered edges otherwise; luma filter
// intra and palette as rc_residual).  No residual energy: it only matters for 64-point transforms.
template <int BD>
__device__ void rc_residual_half(int W, int H, RcShared<BD> &S, int plane, int c, bool cfl, int alpha, const av1g_pixel_t(BD) *src, int sp,
                                 const RcHalfScr &Hs, const RcLanes<16> &L) {
    typedef av1g_pixel_t(BD) Px;
    const int lw = log2i(W), lh = log2i(H);
    int16_t *res = Hs.res;
    if (plane == 0 && is_fi(c)) {
        Px *fiBuf = (Px *)Hs.tmp;
        if (L.l == 0) av1g_predict_filter_intra<BD>(c - 128, lw, lh, S.ea[0] + AV1G_EDGE_OFF, S.el[0] + AV1G_EDGE_OFF, fiBuf, W);
        L.sync();
        for (int p = L.l; p < W * H; p += 16) res[p] = (int16_t)((int)src[(p >> lw) * sp + (p & (W - 1))] - (int)fiBuf[p]);
        L.sync();
        return;
    }
    if (plane == 0 && pal_cand(c)) {
        for (int p = L.l; p < W * H; p += 16) res[p] = (int16_t)((int)src[(p >> lw) * sp + (p & (W - 1))] - (int)S.palCol[S.palMap[p]]);
        L.sync();
        return;
    }
    const int m = cand_mode(c);
    const bool filt = !cfl && S.edgeFilter && av1g_is_directional(m);
    const Px *EA = S.ea[plane] + AV1G_EDGE_OFF, *EL = S.el[plane] + AV1G_EDGE_OFF;
    int upA = 0, upL = 0;
    if (filt) {
        constexpr int NW = RC_HALF_EDGE * (int)sizeof(Px) / 16;
        const uint4 *sa = reinterpret_cast<const uint4 *>(S.ea[plane]), *sl = reinterpret_cast<const uint4 *>(S.el[plane]);
        uint4 *da = reinterpret_cast<uint4 *>(Hs.ea), *dl = reinterpret_cast<uint4 *>(Hs.el);
        for (int k = L.l; k < 2 * NW; k += 16) {
            if (k < NW) da[k] = sa[k];
            else dl[k - NW] = sl[k - NW];
        }
        L.sync();
        grp_edge_prepare<BD, 16>(pangle_of(m, cand_delta(c)), lw, lh, S.ft[plane], S.hvL[plane], S.hvA[plane], S.px[plane], S.py[plane],
                                 S.mx[plane], S.my[plane], (Px *)Hs.ea + AV1G_EDGE_OFF, (Px *)Hs.el + AV1G_EDGE_OFF, &upA, &upL, L);
        EA = (const Px *)Hs.ea + AV1G_EDGE_OFF;
        EL = (const Px *)Hs.el + AV1G_EDGE_OFF;
    }
    const int pa = pangle_of(m, cand_delta(c));
    for (int p = L.l; p < W * H; p += 16) {
        const int i = p >> lw, j = p & (W - 1);
        int pr;
        if (cfl) pr = av1g_clip1<BD>(S.dcv[plane] + av1g_round2signed(alpha * (rc_cflL(S.u)[i * W + j] - S.cflAvg), 6));
        else if (m == AV1G_DC_PRED) pr = S.dcv[plane];
        else pr = av1g_pred_pixel_up(m, pa, i, j, lw, lh, EA, EL, upA, upL);
        res[p] = (int16_t)((int)src[i * sp + j] - pr);
    }
    L.sync();
}

// warp: residual of (plane, cand/cfl) for a W x H tx block into res; returns sum r^2 for 64-point sizes (the only ones
// whose distortion reads it, EvalIn::resEnergy), else 0
template <int BD>
__device__ float rc_residual(int W, int H, RcShared<BD> &S, int plane, int c, bool cfl, int alpha, const av1g_pixel_t(BD) *src, int sp,
                             int16_t *res, av1g_pixel_t(BD) *fiBuf = nullptr) {
    const TxR T(W, H);
    const int lane = threadIdx.x & 31;
    const int slot = threadIdx.x >> 5;
    const bool en = W == 64 || H == 64;
    float e = 0.f;
    if (plane == 0 && is_fi(c)) {
        // recursive filter-intra prediction (spec 7.11.2.3), lane 0, into fiBuf (stride W)
        if (lane == 0)
            av1g_predict_filter_intra<BD>(c - 128, T.lw, T.lh, S.ea[0] + AV1G_EDGE_OFF, S.el[0] + AV1G_EDGE_OFF, fiBuf, W);
        __syncwarp();
        for (int p = lane; p < W * H; p += 32) {
            const int i = p >> T.lw, j = p & (W - 1);
            const int r = (int)src[i * sp + j] - (int)fiBuf[p];
            res[p] = (int16_t)r;
            if (en) e += (float)(r * r);
        }
        __syncwarp();
        return en ? warp_sum(e) : 0.f;
    }
    if (plane == 0 && pal_cand(c)) {   // luma palette (one tx block of the block's size): the colours of the map
        for (int p = lane; p < W * H; p += 32) {
            const int i = p >> T.lw, j = p & (W - 1);
            const int r = (int)src[i * sp + j] - (int)S.palCol[S.palMap[p]];
            res[p] = (int16_t)r;
            if (en) e += (float)(r * r);
        }
        __syncwarp();
        return en ? warp_sum(e) : 0.f;
    }
    rc_prepare_edges(S, plane, c, cfl, T.lw, T.lh);
    for (int p = lane; p < W * H; p += 32) {
        const int i = p >> T.lw, j = p & (W - 1);
        const int r = (int)src[i * sp + j] - rc_pred(S, plane, c, cfl, alpha, i, j, T.lw, T.lh, W, slot);
        res[p] = (int16_t)r;
        if (en) e += (float)(r * r);
    }
    __syncwarp();
    return en ? warp_sum(e) : 0.f;
}

// final quantization of (plane, cand, txType) into S.qfin by warp 0 with distortion and rate (S.finEob, S.finD, S.finR):
// the tx-size search compares J; everything else uses the CTA-wide rc_final_quant below (levels only)
template <int BD>
__device__ void rc_final_quant_rd(int W, int H, RcShared<BD> &S, const EvalIn &e, int plane, int c, bool cfl, int alpha,
                                  const av1g_pixel_t(BD) *src, int sp) {
    const int warp = threadIdx.x >> 5;
    const bool big = rc_big(W, H);
    if (warp == 0) {
        RcBigScr &BG = *S.bigG;
        int16_t *res = big ? BG.res : S.u.w[0].res;
        float *tmp = big ? BG.tmp : S.u.w[0].tmp;
        EvalIn ee = e;
        ee.resEnergy = rc_residual<BD>(W, H, S, plane, c, cfl, alpha, src, sp, res, (av1g_pixel_t(BD) *)tmp);
        const EvalOut o = rc_eval<BD, 32>(res, tmp, S.qfin, ee, W, H, rc_warp_lanes());
        if ((threadIdx.x & 31) == 0) { S.finEob = o.eob; S.finD = o.D; S.finR = o.R; }
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// W1: CTA-wide final quantization (levels + eob only).  warp_eval_rt also computes D and R,
// which the final quantization never uses, and it runs on warp 0 alone; here the forward transform and
// the quantizer run on all 128 threads and the rate pass is skipped.  Every output keeps the arithmetic
// of fwd_col_pass / fwd_row_pass / warp_quant_rate_rt (same fmaf order per output, same quantizer
// expression), so the levels are bit-identical.
template <int TC>
__device__ __forceinline__ void cta_fwd_col(const int16_t *res, float *tmp, int W, int H, int TH, int ck, int fud, int flr,
                                            const float *Fc, const float *FcT, int nChunk) {
    const int TS = W + 1;
    const int it = threadIdx.x;
    if (it >= W * nChunk) return;
    const int j = it % W, i0 = (it / W) * TC;
    const int jj = flr ? W - 1 - j : j;
    if (ck == AV1G_K_IDT) {   // identity: TH == H
        for (int i = i0; i < i0 + TC && i < TH; i++) tmp[i * TS + j] = __ldg(&Fc[i * H + i]) * (float)res[(fud ? H - 1 - i : i) * W + jj];
        return;
    }
    float acc[TC];
#pragma unroll
    for (int i = 0; i < TC; i++) acc[i] = 0.f;
    for (int k = 0; k < H; k++) {
        const float x = (float)res[(fud ? H - 1 - k : k) * W + jj];
        const float4 *f4 = (const float4 *)(FcT + k * H + i0);
#pragma unroll
        for (int i4 = 0; i4 < TC / 4; i4++) {
            const float4 f = __ldg(&f4[i4]);
            acc[4 * i4 + 0] = fmaf(f.x, x, acc[4 * i4 + 0]);
            acc[4 * i4 + 1] = fmaf(f.y, x, acc[4 * i4 + 1]);
            acc[4 * i4 + 2] = fmaf(f.z, x, acc[4 * i4 + 2]);
            acc[4 * i4 + 3] = fmaf(f.w, x, acc[4 * i4 + 3]);
        }
    }
#pragma unroll
    for (int i = 0; i < TC; i++) tmp[(i0 + i) * TS + j] = acc[i];
}
// row pass in place: all items read their row first, CTA barrier, then write (callers: every thread)
template <int TC>
__device__ __forceinline__ void cta_fwd_row(float *tmp, int W, int th, int rk, const float *Fr, const float *FrT, float scale, int nChunk) {
    const int TS = W + 1;
    const int it = threadIdx.x;
    const bool act = it < th * nChunk;
    const int i = act ? it % th : 0, j0 = act ? (it / th) * TC : 0;
    float acc[TC];
#pragma unroll
    for (int jj = 0; jj < TC; jj++) acc[jj] = 0.f;
    if (act) {
        if (rk == AV1G_K_IDT) {
#pragma unroll
            for (int jj = 0; jj < TC; jj++) acc[jj] = __ldg(&Fr[(j0 + jj) * W + j0 + jj]) * tmp[i * TS + j0 + jj];
        } else {
            for (int k = 0; k < W; k++) {
                const float x = tmp[i * TS + k];
                const float4 *f4 = (const float4 *)(FrT + k * W + j0);
#pragma unroll
                for (int j4 = 0; j4 < TC / 4; j4++) {
                    const float4 f = __ldg(&f4[j4]);
                    acc[4 * j4 + 0] = fmaf(f.x, x, acc[4 * j4 + 0]);
                    acc[4 * j4 + 1] = fmaf(f.y, x, acc[4 * j4 + 1]);
                    acc[4 * j4 + 2] = fmaf(f.z, x, acc[4 * j4 + 2]);
                    acc[4 * j4 + 3] = fmaf(f.w, x, acc[4 * j4 + 3]);
                }
            }
        }
    }
    __syncthreads();
    if (act) {
#pragma unroll
        for (int jj = 0; jj < TC; jj++) tmp[i * TS + j0 + jj] = acc[jj] * scale;
    }
}
__device__ __forceinline__ int rc_chunks(int outs, int lanesPerChunk) { return max(1, min(outs / 4, RC_NT / lanesPerChunk)); }

// residual (whole CTA, edges prepared by warp 0 into slot 0) -> forward transform -> levels into S.qfin, S.finEob.
// res (and the prepared slot-0 edges) stay valid for rc_write_from_res / rc_write_pred.
template <int BD>
__device__ void rc_final_quant(int W, int H, RcShared<BD> &S, const EvalIn &e, int plane, int c, bool cfl, int alpha,
                               const av1g_pixel_t(BD) *src, int sp) {
    const TxR T(W, H);
    const int tid = threadIdx.x;
    const bool big = rc_big(W, H);
    RcBigScr &BG = *S.bigG;
    int16_t *res = big ? BG.res : S.u.w[0].res;
    float *tmp = big ? BG.tmp : S.u.w[0].tmp;
    if (plane == 0 && is_fi(c)) {
        if (tid < 32) rc_residual<BD>(W, H, S, plane, c, cfl, alpha, src, sp, res, (av1g_pixel_t(BD) *)tmp);
    } else if (plane == 0 && pal_cand(c)) {
        for (int p = tid; p < W * H; p += RC_NT) res[p] = (int16_t)((int)src[(p >> T.lw) * sp + (p & (W - 1))] - (int)S.palCol[S.palMap[p]]);
    } else {
        if (tid < 32) rc_prepare_edges<BD>(S, plane, c, cfl, T.lw, T.lh);
        __syncthreads();
        for (int p = tid; p < W * H; p += RC_NT) {
            const int i = p >> T.lw, j = p & (W - 1);
            res[p] = (int16_t)((int)src[i * sp + j] - rc_pred<BD>(S, plane, c, cfl, alpha, i, j, T.lw, T.lh, W, 0));
        }
    }
    __syncthreads();
    // forward transform (mirrors warp_fwd_rt)
    const int th = T.th, tw = T.tw;
    const int rk = av1g_row_kind(e.txType), ck = av1g_col_kind(e.txType);
    const int fud = av1g_flip_ud(e.txType), flr = av1g_flip_lr(e.txType);
    {
        const float *Fc = g_fwd + g_fwdOff[ck][T.lh];
        const float *FcT = g_fwdT + g_fwdOff[ck][T.lh];
        const int nc = rc_chunks(th, W), TC = th / nc;
        if (TC == 4) cta_fwd_col<4>(res, tmp, W, H, th, ck, fud, flr, Fc, FcT, nc);
        else if (TC == 8) cta_fwd_col<8>(res, tmp, W, H, th, ck, fud, flr, Fc, FcT, nc);
        else cta_fwd_col<16>(res, tmp, W, H, th, ck, fud, flr, Fc, FcT, nc);
    }
    __syncthreads();
    {
        const float *Fr = g_fwd + g_fwdOff[rk][T.lw];
        const float *FrT = g_fwdT + g_fwdOff[rk][T.lw];
        const float scale = g_txScale[e.txSz];
        const int nc = rc_chunks(tw, th), TC = tw / nc;
        if (TC == 4) cta_fwd_row<4>(tmp, W, th, rk, Fr, FrT, scale, nc);
        else if (TC == 8) cta_fwd_row<8>(tmp, W, th, rk, Fr, FrT, scale, nc);
        else cta_fwd_row<16>(tmp, W, th, rk, Fr, FrT, scale, nc);
    }
    __syncthreads();
    // quantization (mirrors warp_quant_rate_rt's level computation)
    const int TS = W + 1, n = tw * th, lt = log2i(tw);
    const int den = av1g_dq_denom(e.txSz);
    int16_t *q = S.qfin;
    for (int idx = tid; idx < n; idx += RC_NT) {
        const int i = idx >> lt, j = idx & (tw - 1);
        const float v = tmp[i * TS + j];
        const int step = av1g_qm_step(idx == 0 ? e.dcQ : e.acQ, e.qm, idx);
        const float a = fabsf(v) * (float)den / (float)step;
        const float wr = e.qm ? 32.f / (float)e.qm[idx] : 1.f;
        int l = a < e.zbin ? 0 : (int)__fadd_rn(a, e.qmRnd ? __fmul_rn(e.rnd, wr) : e.rnd);   // as rc_quant_rate
        l = min(l, 16383);
        const int qv = v < 0 ? -l : l;
        q[idx] = (int16_t)qv;
    }
    if (tid == 0) S.finEob = 0;
    __syncthreads();
    const uint16_t *scan = av1g_get_scan(e.txSz, e.txType);
    int last = 0;
    for (int cc = tid; cc < n; cc += RC_NT)
        if (q[scan[cc]]) last = cc + 1;
    last = warp_max(last);
    if ((tid & 31) == 0 && last) atomicMax(&S.finEob, last);
    __syncthreads();
}

// reconstruction base = prediction = src - residual (the residual of the final candidate is exact:
// res = src - pred in integers), instead of evaluating the prediction again (CTA)
template <typename Px>
__device__ void rc_write_from_res(int W, int H, const Px *src, int sp, const int16_t *res, Px *dst, int pitch) {
    const int lw = log2i(W);
    for (int p = threadIdx.x; p < W * H; p += RC_NT) {
        const int i = p >> lw, j = p & (W - 1);
        dst[i * pitch + j] = (Px)((int)src[i * sp + j] - (int)res[p]);
    }
    __syncthreads();
}

// write prediction into the reconstruction (group g; directional modes: edges prepared in slot)
template <int BD>
__device__ void rc_write_pred(int W, int H, const RcShared<BD> &S, int plane, int c, bool cfl, int alpha, av1g_pixel_t(BD) *dst, int pitch,
                              const RcGrp &g, int slot = 0) {
    const TxR T(W, H);
    if (plane == 0 && is_fi(c)) {
        if (g.t == 0)
            av1g_predict_filter_intra<BD>(c - 128, T.lw, T.lh, S.ea[0] + AV1G_EDGE_OFF, S.el[0] + AV1G_EDGE_OFF, dst, pitch);
        g.sync();
        return;
    }
    if (plane == 0 && pal_cand(c)) {
        for (int p = g.t; p < W * H; p += g.nt) dst[(p >> T.lw) * pitch + (p & (W - 1))] = (av1g_pixel_t(BD))S.palCol[S.palMap[p]];
        g.sync();
        return;
    }
    for (int p = g.t; p < W * H; p += g.nt) {
        const int i = p >> T.lw, j = p & (W - 1);
        dst[i * pitch + j] = (av1g_pixel_t(BD))rc_pred<BD>(S, plane, c, cfl, alpha, i, j, T.lw, T.lh, W, slot);
    }
    g.sync();
}

// record a finished tx block of levels q (group g): txb entry txbIdx, coefficients at cOff, context maps,
// BlockDecoded.  Returns eob.  The caller advances S.nTxb / S.nCoef.
template <int SS, int BD>
__device__ int rc_record_txb_at(int W, int H, RcShared<BD> &S, const GpBatch &B, uint32_t sb, const GpImage &I, int plane, int txSz,
                                int txType, int x4, int y4, int sbr4, int sbc4, const int16_t *q, const RcGrp &g, uint32_t cOff,
                                int txbIdx) {
    const TxR T(W, H);
    const int tid = g.t;
    // eob / culLevel / dc category from the final levels (first warp of g)
    __shared__ int sEob[2], sCul[2];
    if (tid < 32) {
        const uint16_t *scan = av1g_get_scan(txSz, txType);
        int last = 0;
        for (int c = tid; c < T.n; c += 32) if (q[scan[c]]) last = c + 1;
        const int eob = warp_max(last);
        int cul = 0;
        for (int c = tid; c < eob; c += 32) { const int v = q[scan[c]]; cul += v < 0 ? -v : v; }
        cul = warp_sumi(cul);
        if (tid == 0) { sEob[g.k] = eob; sCul[g.k] = min(cul, 63); }
    }
    g.sync();
    const int eob = sEob[g.k], cul = sCul[g.k];
    const int dcCat = q[0] < 0 ? 1 : q[0] > 0 ? 2 : 0;
    int16_t *oc = B.oCoeffs + (size_t)sb * B.sbCoefCap + cOff;
    for (int i = tid; i < T.n; i += g.nt) oc[i] = q[i];
    // context maps (eob == 0 -> 0, the same as the decoder's skip / all-zero handling)
    const int stride = I.MiCols >> (plane ? SS : 0), rows = I.MiRows >> (plane ? SS : 0);
    const int w4 = W >> 2, h4 = H >> 2, lw4 = T.lw - 2;
    for (int k = tid; k < w4 * h4; k += g.nt) {
        const int yy = y4 + (k >> lw4), xx = x4 + (k & (w4 - 1));
        if (yy < rows && xx < stride) {
            I.cLvl[plane][(size_t)yy * stride + xx] = (uint8_t)(eob ? cul : 0);
            I.cDc[plane][(size_t)yy * stride + xx] = (uint8_t)(eob ? dcCat : 0);
        }
    }
    // BlockDecoded (full tx block area, also beyond the frame edge)
    for (int k = tid; k < w4 * h4; k += g.nt) atomicOr(&S.bdm[plane][sbr4 + (k >> lw4) + 1], 1u << (sbc4 + (k & (w4 - 1)) + 1));
    if (tid == 0) {
        av1g_txb t;
        t.coeff_off = cOff;
        t.x4 = (uint16_t)x4; t.y4 = (uint16_t)y4;
        t.plane = (uint8_t)plane; t.tx_size = (uint8_t)txSz;
        t.tx_type = (uint8_t)(plane == 0 && eob == 0 ? AV1G_DCT_DCT : txType);
        t.flags = 0; t.eob = (uint16_t)eob; t.reserved = 0;
        B.oTxb[(size_t)sb * B.sbTxbCap + txbIdx] = t;
    }
    return eob;
}
// record the tx block in S.qfin as the SB's next one (CTA)
template <int SS, int BD>
__device__ int rc_record_txb(int W, int H, RcShared<BD> &S, const GpBatch &B, uint32_t sb, const GpImage &I, int plane, int txSz, int txType,
                             int x4, int y4, int sbr4, int sbc4) {
    // coefficient offset: read BEFORE the barriers - thread 0 advances S.nCoef after them, and a warp
    // reading it late wrote its levels one tx block too far, leaving stale memory in the record
    // (racecheck 2026-10-02; the 2-encoder batch-mode V-plane decoder mismatches)
    const uint32_t cOff = (uint32_t)S.nCoef;
    const int eob = rc_record_txb_at<SS, BD>(W, H, S, B, sb, I, plane, txSz, txType, x4, y4, sbr4, sbc4, S.qfin, rc_cta(), cOff, S.nTxb);
    if (threadIdx.x == 0) {
        S.nTxb++;
        S.nCoef += TxR(W, H).n;
    }
    __syncthreads();
    return eob;
}

// edges of a tx block from the reconstruction, built by threads tid0 .. tid0+nt-1; then one thread
// records the edge-filter inputs and the DC value.  Callers __syncthreads() afterwards.
template <int SS, int BD>
__device__ __forceinline__ void rc_edges(RcShared<BD> &S, const GpImage &I, int plane, int px, int py, int lw, int lh, int haveL,
                                         int haveA, int haveAR, int haveBL, int mi_row, int mi_col, int tid0, int nt,
                                         int blkA = -1, int blkL = -1) {
    const int maxX = ((I.MiCols * 4) >> (plane ? SS : 0)) - 1, maxY = ((I.MiRows * 4) >> (plane ? SS : 0)) - 1;
    av1g_pixel_t(BD) *A = S.ea[plane] + AV1G_EDGE_OFF, *L = S.el[plane] + AV1G_EDGE_OFF;
    const int t = (int)threadIdx.x - tid0;
    if (t < 0 || t >= nt) return;
    build_edges_par<BD>(gp_rec<BD>(I, plane), I.rpitch[plane], px, py, lw, lh, haveL, haveA, haveAR, haveBL, maxX, maxY, A, L, t, nt);
    // the group of nt threads is warp-aligned (nt = 32 or 64): sync the warps of the group
    // named barrier ids must be immediates: with a register id ptxas reserves all 16 hardware barriers
    // (EIATTR_NUM_BARRIERS 16), which limits k_recon to ONE resident CTA per SM instead of 4
    if (nt <= 32) __syncwarp();
    else if (plane == 0) asm volatile("bar.sync 1, %0;" ::"r"(nt));
    else if (plane == 1) asm volatile("bar.sync 2, %0;" ::"r"(nt));
    else asm volatile("bar.sync 3, %0;" ::"r"(nt));
    if (t < 32) {   // F6: the DC sums over the group's first warp (warp_dc_value: av1g_dc_value's sums and rounding), not one thread
        const int dcv = warp_dc_value<BD>(A, L, haveL, haveA, lw, lh);
        if (t == 0) {
            S.dcv[plane] = dcv;
            S.hvL[plane] = haveL; S.hvA[plane] = haveA; S.px[plane] = px; S.py[plane] = py; S.mx[plane] = maxX; S.my[plane] = maxY;
            S.ft[plane] = I.edgeFilter ? av1g_get_filter_type(plane, mi_row, mi_col, blkA >= 0 ? blkA : haveA, blkL >= 0 ? blkL : haveL,
                                                              I.mYMode, I.mUVMode, I.MiCols, SS, SS) : 0;
        }
    }
}

// phase accounting (GpParams::profile): thread 0 adds the cycles since its last tick to B.prof[phase];
// phases end at CTA-wide barriers, so this is the CTA's critical path per phase.
__device__ __forceinline__ void rc_tick(const GpBatch &B, const GpParams &P, int phase) {
    if (!P.profile || threadIdx.x != 0) return;
    __shared__ long long t0;
    const long long n = clock64();
    if (phase >= 0) atomicAdd(&B.prof[phase], (unsigned long long)(n - t0));
    t0 = n;
}

// ---------------------------------------------------------------------------
// luma of one block (tx size = block size, tx_depth 0)
// the luma palette of candidate c (pal_dev.cuh) for the block at (x, y): colours, its map in the SB's colour-map slot (at
// S.nMap) and its bits, rebuilt from the source exactly as k_pal built it (warp 0; histogram scratch in the union).  Not
// inlined: it stays out of rc_luma's register allocation.  The analysis only proposes palettes of 2 or more colours, and
// this rebuilds the same one; should it ever come out smaller, a 2-colour palette keeps the block valid.
template <int BD>
__device__ __noinline__ void rc_pal_prepare(RcShared<BD> &S, const GpBatch &B, const GpImage &I, uint32_t sb, const av1g_pixel_t(BD) *src,
                                            int x, int y, int W, int H, int bsize, int c, const float *C) {
    const int tid = threadIdx.x;
    if (tid < 32) {
        const int ow = min(W, I.MiCols * 4 - x), oh = min(H, I.MiRows * 4 - y);
        uint32_t *hist = (uint32_t *)S.u.raw;
        uint16_t *dv = (uint16_t *)(S.u.raw + 4096), *dc = dv + 64;
        const int nd = pal_hist<BD>(src, I.spitch[0], ow, oh, hist, dv, dc);
        int n = 0;
        if (tid == 0) {
            if (nd >= 2 && nd <= 64) n = pal_kmeans(dv, dc, nd, pal_target(c), pal_top(c), BD, S.palCol);
            if (n < 2) {
                S.palCol[0] = (uint16_t)min((int)src[0], (1 << BD) - 2);
                S.palCol[1] = (uint16_t)(S.palCol[0] + 1);
                n = 2;
            }
        }
        n = __shfl_sync(0xffffffffu, n, 0);
        __syncwarp();
        uint8_t *map = B.oMap + (size_t)sb * GP_SB_MAP + S.nMap;
        pal_map(src, I.spitch[0], W, H, ow, oh, S.palCol, n, map, tid, 32);
        __syncwarp();
        const float bits = pal_bits(n, S.palCol, map, W, ow, oh, bsize, BD, C);
        if (tid == 0) { S.palBits = bits; S.palN = n; S.palMap = map; }
    }
    __syncthreads();
}

template <int BD>
__device__ __noinline__ void rc_luma(int W, int H, RcShared<BD> &S, const GpBatch &B, const GpParams &P, uint32_t sb, const GpImage &I, int bsize, int x,
                        int y, int availU, int availL, const TileBox &tb, const BlockQ &bq, float lambda, const float *C,
                        int *outMode, int *outDelta, int *outEob, int *outFi, int *outTxDepth, int *outPal) {
    const TxR T(W, H);
    const int tid = threadIdx.x;
    const int txSz = AV1G_T(Max_Tx_Size_Rect)[bsize];
    const int sbr4 = (y >> 2) & 15, sbc4 = (x >> 2) & 15;
    const int stepX = W >> 2, stepY = H >> 2;
    const int haveAR0 = (S.bdm[0][sbr4 - 1 + 1] >> (sbc4 + stepX + 1)) & 1;
    const int haveBL0 = (S.bdm[0][sbr4 + stepY + 1] >> (sbc4 - 1 + 1)) & 1;
    *outTxDepth = 0;
    typedef av1g_pixel_t(BD) Px;
    rc_edges<1, BD>(S, I, 0, x, y, T.lw, T.lh, availL, availU, haveAR0, haveBL0, y >> 2, x >> 2, 0, 64);
    const GpBlockRes &br = B.blk[I.shOff[bsize] + (uint32_t)(y / H) * I.shNbx[bsize] + x / W];
    int skipCtx, dcCtx;
    txb_ctx<1>(I, 0, txSz, x >> 2, y >> 2, bsize, tb, &skipCtx, &dcCtx);
    const int mi_row = y >> 2, mi_col = x >> 2;
    const int am = AV1G_T(Intra_Mode_Context)[availU ? I.mYMode[(size_t)(mi_row - 1) * I.MiCols + mi_col] : AV1G_DC_PRED];
    const int lm = AV1G_T(Intra_Mode_Context)[availL ? I.mYMode[(size_t)mi_row * I.MiCols + mi_col - 1] : AV1G_DC_PRED];
    const bool fiOk = P.filterIntra && av1g_bw4(bsize) <= 8 && av1g_bh4(bsize) <= 8;
    auto mbits = [&](int c) -> float {
        const int m = luma_mode(c);
        float b = C[CID(y_mode[am][lm]) + m];
        if (pal_cand(c)) return b + S.palBits;   // no filter intra with a palette
        if (!is_fi(c) && bsize >= AV1G_BLOCK_8X8 && av1g_is_directional(m)) b += C[CID(angle_delta[m - AV1G_V_PRED]) + cand_delta(c) + 3];
        if (fiOk && m == AV1G_DC_PRED) {
            b += C[CID(filter_intra[bsize]) + (is_fi(c) ? 1 : 0)];
            if (is_fi(c)) b += C[CID(filter_intra_mode) + c - 128];
        }
        return b;
    };
    EvalIn e;
    e.txSz = txSz; e.plane = 0; e.dcQ = bq.dcQ[0]; e.acQ = bq.acQ[0];
    e.rnd = P.rnd; e.zbin = P.zbin; e.skipCtx = skipCtx; e.dcCtx = dcCtx; e.reduced = 0; e.C = C;
    e.intraDir = 0; e.txType = 0; e.qm = 0; e.resEnergy = 0;
    e.qmDist = P.qmDist; e.qmRnd = P.qmRnd;
    const Px *src = gp_src<BD>(I, 0) + (size_t)y * I.spitch[0] + x;
    const Px *srcP[3] = {src, src, src};
    const int spP[3] = {I.spitch[0], I.spitch[0], I.spitch[0]};
    const int nc = max(1, min((int)br.ncand, P.Krecheck));
    const int set = av1g_get_tx_set_intra(txSz, 0);
    // luma palette candidate (screen-content frames; k_pal makes it candidate 0): its palette, map and bits
    if (B.oPal && I.sct && pal_cand(br.cand[0])) rc_pal_prepare<BD>(S, B, I, sb, src, x, y, W, H, bsize, br.cand[0], C);
    int bestC = br.cand[0], bestT = br.tx;
    __syncthreads();
    rc_tick(B, P, 0);  // luma setup: edges, contexts
    const bool big = rc_big(W, H);
    const bool pass2 = P.txSearchCL && set != AV1G_TX_SET_DCTONLY;
    bool evaluated = false;
    if (tid < RC_NW) S.bestJw[tid] = GP_INF;
    // run S.taskC/T on the warps (big tx: warp 0 only); taskJ per task, per-warp best (levels kept)
    // W2: when every task of a pass has the same (non-filter-intra) candidate (pass 2: tx types of
    // the winner), the residual is computed once by the CTA into S.u.w[0].res and shared by all warps
    // (read-only during the evaluation); it also gives the final prediction (src - res) for free.
    int sharedResC = -1;
    float sharedResE = 0.f;
    auto runTasks = [&](bool shared) {
        const int warp = tid >> 5;
        if (W <= 8 && H <= 8 && S.nTask > RC_NW) {
            // tx up to 8x8 with more tasks than warps (pass 2: 5-7 tx types): one task per half-warp, rounds of 8
            // instead of 4 (rc_eval.cuh: the same J and levels).  The per-warp best is then updated in the order of the
            // loop below - warp w compares tasks w, w + 4, w + 8, ... - so ties resolve as there.
            const RcLanes<16> L = rc_half_lanes();
            const RcHalfScr Hs = rc_half_scr(S);
            for (int r0 = 0; r0 < S.nTask; r0 += 2 * RC_NW) {
                const int t = r0 + (tid >> 4);
                if (t < S.nTask) {
                    const int c = S.taskC[t];
                    EvalIn ee = e;
                    ee.txType = S.taskT[t];
                    ee.intraDir = intra_dir(c);
                    ee.qm = av1g_qm(I.usingQm, I.qm[0], 0, txSz, ee.txType);
                    if (!shared) rc_residual_half<BD>(W, H, S, 0, c, false, 0, src, I.spitch[0], Hs, L);
                    const EvalOut o = rc_eval<BD, 16>(shared ? S.u.w[0].res : Hs.res, Hs.tmp, Hs.q, ee, W, H, L);
                    const float J = fmaf(lambda, __fadd_rn(o.R, mbits(c)), o.D);   // explicit rounding: rc_eval.cuh
                    if (L.l == 0) { S.taskJ[t] = J; S.taskE[t] = o.eob; }
                }
                __syncthreads();
#pragma unroll
                for (int k = 0; k < 2; k++) {
                    const int tt = r0 + warp + k * RC_NW, hs = warp + k * RC_NW;   // task, its half-warp slot
                    if (tt < S.nTask && S.taskJ[tt] < S.bestJw[warp]) {   // warp-uniform
                        const int16_t *qs = S.u.w[hs >> 1].q + (hs & 1) * 64;
                        for (int i = tid & 31; i < W * H; i += 32) rc_bestq(S.u, warp)[i] = qs[i];
                        __syncwarp();
                        if ((tid & 31) == 0) {
                            S.bestJw[warp] = S.taskJ[tt]; S.bestCw[warp] = S.taskC[tt]; S.bestTw[warp] = S.taskT[tt]; S.bestEw[warp] = S.taskE[tt];
                        }
                    }
                    __syncwarp();
                }
                __syncthreads();
            }
            return;
        }
        for (int t = warp; t < S.nTask; t += RC_NW) {   // W5: big tasks on every warp (own global slot)
            const int c = S.taskC[t];
            RcBigScr &BG = S.bigG[warp];
            int16_t *res = big ? BG.res : (shared ? S.u.w[0].res : S.u.w[warp].res);
            float *tmp = big ? BG.tmp : S.u.w[warp].tmp;
            int16_t *q = big ? BG.q : S.u.w[warp].q;
            EvalIn ee = e;
            ee.txType = S.taskT[t];
            ee.intraDir = intra_dir(c);
            ee.qm = av1g_qm(I.usingQm, I.qm[0], 0, txSz, ee.txType);
            ee.resEnergy = shared ? sharedResE : rc_residual<BD>(W, H, S, 0, c, false, 0, src, I.spitch[0], res, (Px *)tmp);
            const EvalOut o = rc_eval<BD, 32>(res, tmp, q, ee, W, H, rc_warp_lanes());
            const float J = fmaf(lambda, __fadd_rn(o.R, mbits(c)), o.D);
            if ((tid & 31) == 0) S.taskJ[t] = J;
            if (!big) {
                const bool better = J < S.bestJw[warp];   // identical in all lanes (warp-reduced values)
                __syncwarp();
                if (better) {
                    for (int i = tid & 31; i < W * H; i += 32) rc_bestq(S.u, warp)[i] = q[i];
                    if ((tid & 31) == 0) { S.bestJw[warp] = J; S.bestCw[warp] = c; S.bestTw[warp] = ee.txType; S.bestEw[warp] = o.eob; }
                }
                __syncwarp();
            }
        }
        __syncthreads();
    };
    if (P.txAllCL && set != AV1G_TX_SET_DCTONLY) {
        // every re-checked candidate (+ filter-intra modes) x every intra tx type of the set (libaom s3's
        // "tx search in mode evaluation"), in chunks of 32 tasks; per-warp best levels kept across chunks
        int types[16], nTypes = 0;
        for (int t = 0; t < 16; t++)
            if (av1g_tx_type_in_set_intra(set, t)) types[nTypes++] = t;
        const int total = (nc + (fiOk ? 5 : 0)) * nTypes;
        float bJ = GP_INF;
        for (int base = 0; base < total; base += 32) {
            if (tid == 0) {
                int n = 0;
                for (int i = base; i < total && n < 32; i++, n++) {
                    const int k = i / nTypes;
                    S.taskC[n] = k < nc ? br.cand[k] : 128 + (k - nc);
                    S.taskT[n] = types[i % nTypes];
                    S.taskP[n] = 0;
                }
                S.nTask = n;
            }
            __syncthreads();
            runTasks(false);
            for (int t = 0; t < S.nTask; t++)
                if (S.taskJ[t] < bJ) { bJ = S.taskJ[t]; bestC = S.taskC[t]; bestT = S.taskT[t]; }
            __syncthreads();   // all threads read the chunk before thread 0 writes the next one
        }
        evaluated = true;
        rc_tick(B, P, 1);
    } else if (nc > 1 || pass2 || fiOk) {
        // pass 1: candidates with DCT_DCT (+ the analysis winner with its tx type, + filter-intra modes)
        if (tid == 0) {
            int n = 0;
            for (int k = 0; k < nc; k++) { S.taskC[n] = br.cand[k]; S.taskT[n] = AV1G_DCT_DCT; S.taskP[n] = 0; n++; }
            if (br.tx != AV1G_DCT_DCT) { S.taskC[n] = br.cand[0]; S.taskT[n] = br.tx; S.taskP[n] = 0; n++; }
            if (fiOk)
                for (int f = 0; f < 5; f++) { S.taskC[n] = 128 + f; S.taskT[n] = AV1G_DCT_DCT; S.taskP[n] = 0; n++; }
            S.nTask = n;
        }
        __syncthreads();
        const bool skip1 = S.nTask == 1;   // nothing to compare: bestC = that candidate
        if (!skip1) {
            runTasks(false);
            evaluated = true;
            float bJ = GP_INF;
            for (int t = 0; t < S.nTask; t++)
                if (S.taskJ[t] < bJ) { bJ = S.taskJ[t]; bestC = S.taskC[t]; bestT = S.taskT[t]; }
            __syncthreads();
        } else {
            bestC = S.taskC[0];
            bestT = S.taskT[0];
        }
        // every thread must have read S.nTask / taskC / taskT before thread 0 rewrites them for pass 2:
        // without this barrier a late warp saw pass 2's nTask, took the !skip1 branch (runTasks and its
        // __syncthreads) and misaligned the CTA's barriers - rare, timing-dependent corrupt records
        // (racecheck 2026-10-02; seen as V-plane decoder mismatches in 2-encoder batch runs)
        __syncthreads();
        rc_tick(B, P, 1);  // pass 1
        // pass 2: tx-type search for the winner with the real residual (all types if pass 1 was skipped)
        if (pass2) {
            if (tid == 0) {
                int n = 0;
                for (int t = 0; t < 16; t++) {
                    if (!av1g_tx_type_in_set_intra(set, t)) continue;
                    if (!skip1 && (t == AV1G_DCT_DCT || (bestC == br.cand[0] && t == br.tx))) continue;
                    S.taskC[n] = bestC; S.taskT[n] = t; S.taskP[n] = 0; n++;
                }
                S.nTask = n;
            }
            __syncthreads();
            if (S.nTask > 0) {
                if (!big && !is_fi(bestC)) {
                    // shared residual of bestC: edges by warp 0 (slot 0), residual by the CTA; energy with the
                    // lane order of rc_residual (warp 0) so EvalIn.resEnergy is bit-identical
                    if (tid < 32) rc_prepare_edges<BD>(S, 0, bestC, false, T.lw, T.lh);
                    __syncthreads();
                    int16_t *res = S.u.w[0].res;
                    if (pal_cand(bestC))
                        for (int p = tid; p < W * H; p += RC_NT)
                            res[p] = (int16_t)((int)src[(p >> T.lw) * I.spitch[0] + (p & (W - 1))] - (int)S.palCol[S.palMap[p]]);
                    else
                        for (int p = tid; p < W * H; p += RC_NT) {
                            const int i = p >> T.lw, j = p & (W - 1);
                            res[p] = (int16_t)((int)src[i * I.spitch[0] + j] - rc_pred<BD>(S, 0, bestC, false, 0, i, j, T.lw, T.lh, W, 0));
                        }
                    __syncthreads();
                    if (W == 64 || H == 64) {   // only the 64-point distortion uses the energy (never in pass 2)
                        float en = 0.f;
                        for (int p = tid & 31; p < W * H; p += 32) en += (float)((int)res[p] * (int)res[p]);
                        sharedResE = warp_sum(en);
                    }
                    sharedResC = bestC;
                    runTasks(true);
                } else {
                    runTasks(false);
                }
                evaluated = true;
                if (big) {
                    float bJ = GP_INF;
                    for (int t = 0; t < S.nTask; t++)
                        if (S.taskJ[t] < bJ) { bJ = S.taskJ[t]; bestT = S.taskT[t]; }
                }
            }
            __syncthreads();
        }
    }
    int bestW = -1;
    if (evaluated && !big) {   // overall winner = best of the per-warp minima (levels already quantized)
        float bJ = GP_INF;
        for (int w = 0; w < RC_NW; w++)
            if (S.bestJw[w] < bJ) { bJ = S.bestJw[w]; bestW = w; }
        if (bestW >= 0) { bestC = S.bestCw[bestW]; bestT = S.bestTw[bestW]; }
    }
    (void)srcP; (void)spP;
    rc_tick(B, P, 2);  // pass 2
    // final quantization + reconstruction
    e.txType = bestT;
    e.intraDir = intra_dir(bestC);
    e.qm = av1g_qm(I.usingQm, I.qm[0], 0, txSz, bestT);
    float J0;
    const bool trySplit = P.txSizeSearch && av1g_max_tx_depth(bsize) >= 1 && !pal_cand(bestC);   // palette: one tx block
    // W1/W2: without the tx-size trial the final prediction is src - residual (the shared pass-2
    // residual of bestC, or the final quantization's), no second prediction pass
    const int16_t *predRes = nullptr;
    if (bestW >= 0) {
        // reuse the winner's levels
        for (int i = tid; i < W * H; i += RC_NT) S.qfin[i] = rc_bestq(S.u, bestW)[i];
        if (tid == 0) S.finEob = S.bestEw[bestW];
        if (!trySplit && sharedResC == bestC) predRes = S.u.w[0].res;
        else if (tid < 32) rc_prepare_edges<BD>(S, 0, bestC, false, T.lw, T.lh);
        __syncthreads();
        J0 = S.bestJw[bestW];
    } else if (trySplit) {
        rc_final_quant_rd<BD>(W, H, S, e, 0, bestC, false, 0, src, I.spitch[0]);
        J0 = fmaf(lambda, __fadd_rn(S.finR, mbits(bestC)), S.finD);
    } else {
        rc_final_quant<BD>(W, H, S, e, 0, bestC, false, 0, src, I.spitch[0]);
        predRes = big ? S.bigG->res : S.u.w[0].res;
        J0 = 0.f;   // only the tx-size trial uses it
    }
    int txDepth = 0;
    if (trySplit) {
        // ---- tx-size search: largest tx (above) vs one split level, closed loop: every sub tx block is
        //      predicted from the reconstruction of the previous ones (raster order), tx type searched per
        //      sub block, recorded as it goes; rolled back when depth 0 wins ----
        const int eob0 = S.finEob;
        int16_t *keep0 = bestW >= 0 ? rc_bestq(S.u, bestW) : rc_bestq(S.u, 0);   // union tail, 2 KB
        const int nLv = min(W, 32) * min(H, 32);   // 64-point transforms keep 32x32 levels
        if (bestW < 0)
            for (int i = tid; i < nLv; i += RC_NT) keep0[i] = S.qfin[i];
        const int nTxb0 = S.nTxb, nCoef0 = S.nCoef;
        __syncthreads();
        const int txSub = AV1G_T(Split_Tx_Size)[txSz];
        const int Ws = AV1G_T(Tx_Width)[txSub], Hs = AV1G_T(Tx_Height)[txSub];
        const TxR Ts(Ws, Hs);
        const int setS = av1g_get_tx_set_intra(txSub, 0);
        int types[16], nTypes = 0;
        if ((P.txSearchCL || P.txAllCL) && setS != AV1G_TX_SET_DCTONLY) {
            for (int t = 0; t < 16; t++)
                if (av1g_tx_type_in_set_intra(setS, t)) types[nTypes++] = t;
        } else {
            types[nTypes++] = AV1G_DCT_DCT;
        }
        const bool bigS = rc_big(Ws, Hs);
        const int maxX4 = I.MiCols, maxY4 = I.MiRows;
        const int sqUp = AV1G_T(Tx_Size_Sqr_Up)[txSz];
        // tx_depth symbol, averaged over its 3 contexts
        auto depthBits = [&](int d) -> float {
            float s = 0.f;
            for (int ctx = 0; ctx < 3; ctx++)
                s += sqUp == AV1G_TX_8X8 ? C[CID(tx_8x8[ctx]) + d] : sqUp == AV1G_TX_16X16 ? C[CID(tx_16x16[ctx]) + d]
                   : sqUp == AV1G_TX_32X32 ? C[CID(tx_32x32[ctx]) + d] : C[CID(tx_64x64[ctx]) + d];
            return s * (1.f / 3.f);
        };
        float J1 = lambda * (mbits(bestC) + depthBits(1));
        const float limit = J0 + lambda * depthBits(0);
        int eobAny = 0;
        const int warp = tid >> 5;
        for (int iy = 0; iy < H / Hs && J1 < limit; iy++)
            for (int ix = 0; ix < W / Ws && J1 < limit; ix++) {
                const int xs = x + ix * Ws, ys = y + iy * Hs;
                if ((xs >> 2) >= maxX4 || (ys >> 2) >= maxY4) continue;   // tx blocks outside the frame are not coded
                const int sr4 = (ys >> 2) & 15, sc4 = (xs >> 2) & 15;
                const int hAR = (S.bdm[0][sr4 - 1 + 1] >> (sc4 + (Ws >> 2) + 1)) & 1;
                const int hBL = (S.bdm[0][sr4 + (Hs >> 2) + 1] >> (sc4 - 1 + 1)) & 1;
                rc_edges<1, BD>(S, I, 0, xs, ys, Ts.lw, Ts.lh, availL || ix > 0, availU || iy > 0, hAR, hBL, y >> 2, x >> 2, 0, 64, availU,
                                availL);
                __syncthreads();
                int sCtx, dCtx;
                txb_ctx<1>(I, 0, txSub, xs >> 2, ys >> 2, bsize, tb, &sCtx, &dCtx);
                EvalIn es = e;
                es.txSz = txSub; es.skipCtx = sCtx; es.dcCtx = dCtx; es.intraDir = intra_dir(bestC);
                const Px *srcS = gp_src<BD>(I, 0) + (size_t)ys * I.spitch[0] + xs;
                int tSel;
                float Jk;
                if (bigS) {   // 32x32 sub blocks of a 64x64: DCT only, warp 0 into S.qfin
                    es.txType = AV1G_DCT_DCT;
                    es.qm = av1g_qm(I.usingQm, I.qm[0], 0, txSub, AV1G_DCT_DCT);
                    rc_final_quant_rd<BD>(Ws, Hs, S, es, 0, bestC, false, 0, srcS, I.spitch[0]);
                    tSel = AV1G_DCT_DCT;
                    Jk = fmaf(lambda, S.finR, S.finD);
                } else {
                    float myJ = GP_INF;
                    int myT = AV1G_DCT_DCT, myE = 0;
                    for (int ti = warp; ti < nTypes; ti += RC_NW) {
                        int16_t *res = S.u.w[warp].res, *q = S.u.w[warp].q;
                        float *tmp = S.u.w[warp].tmp;
                        EvalIn ee = es;
                        ee.txType = types[ti];
                        ee.qm = av1g_qm(I.usingQm, I.qm[0], 0, txSub, ee.txType);
                        ee.resEnergy = rc_residual<BD>(Ws, Hs, S, 0, bestC, false, 0, srcS, I.spitch[0], res, (Px *)tmp);
                        const EvalOut o = rc_eval<BD, 32>(res, tmp, q, ee, Ws, Hs, rc_warp_lanes());
                        const float J = fmaf(lambda, o.R, o.D);
                        if (J < myJ) {   // warp-uniform
                            __syncwarp();
                            for (int i = tid & 31; i < Ws * Hs; i += 32) rc_taskq(S.u, warp)[i] = q[i];
                            myJ = J; myT = ee.txType; myE = o.eob;
                        }
                        __syncwarp();
                    }
                    if ((tid & 31) == 0) { S.taskJ[warp] = myJ; S.taskT[warp] = myT; S.taskE[warp] = myE; }
                    __syncthreads();
                    int bw = 0;
                    for (int w = 1; w < RC_NW; w++)
                        if (S.taskJ[w] < S.taskJ[bw]) bw = w;
                    for (int i = tid; i < Ws * Hs; i += RC_NT) S.qfin[i] = rc_taskq(S.u, bw)[i];
                    if (tid == 0) S.finEob = S.taskE[bw];
                    tSel = S.taskT[bw];
                    Jk = S.taskJ[bw];
                    if (tid < 32) rc_prepare_edges<BD>(S, 0, bestC, false, Ts.lw, Ts.lh);
                    __syncthreads();
                }
                J1 += Jk;
                Px *dstS = gp_rec<BD>(I, 0) + (size_t)ys * I.rpitch[0] + xs;
                rc_write_pred<BD>(Ws, Hs, S, 0, bestC, false, 0, dstS, I.rpitch[0], rc_cta());
                if (S.finEob > 0) {
                    const int fixes = cta_inverse_add(Ws, Hs, S, S.qfin, txSub, tSel, e.dcQ, e.acQ,
                                                      av1g_qm(I.usingQm, I.qm[0], 0, txSub, tSel), dstS, I.rpitch[0], rc_cta());
                    if (fixes && tid == 0) atomicAdd(&B.stats[3], 1u);
                }
                eobAny |= rc_record_txb<1, BD>(Ws, Hs, S, B, sb, I, 0, txSub, tSel, xs >> 2, ys >> 2, sr4, sc4);
            }
        if (J1 < limit) {
            txDepth = 1;
            *outEob = eobAny;
            if (tid == 0) atomicAdd(&B.stats[4], 1u);
        } else {
            // depth 0 wins: drop the trial's tx records, rebuild the block's edges and levels
            __syncthreads();
            if (tid == 0) { S.nTxb = nTxb0; S.nCoef = nCoef0; }
            rc_edges<1, BD>(S, I, 0, x, y, T.lw, T.lh, availL, availU, haveAR0, haveBL0, y >> 2, x >> 2, 0, 64);
            __syncthreads();
            for (int i = tid; i < nLv; i += RC_NT) S.qfin[i] = keep0[i];
            if (tid == 0) S.finEob = eob0;
            if (tid < 32) rc_prepare_edges<BD>(S, 0, bestC, false, T.lw, T.lh);
            __syncthreads();
        }
    }
    *outTxDepth = txDepth;
    if (txDepth == 0) {
        Px *dst = gp_rec<BD>(I, 0) + (size_t)y * I.rpitch[0] + x;
        if (predRes) rc_write_from_res(W, H, src, I.spitch[0], predRes, dst, I.rpitch[0]);
        else rc_write_pred<BD>(W, H, S, 0, bestC, false, 0, dst, I.rpitch[0], rc_cta());
        if (S.finEob > 0) {
            const int fixes = cta_inverse_add(W, H, S, S.qfin, txSz, bestT, e.dcQ, e.acQ, e.qm, dst, I.rpitch[0], rc_cta());
            if (fixes && tid == 0) atomicAdd(&B.stats[3], 1u);
        }
        *outEob = rc_record_txb<1, BD>(W, H, S, B, sb, I, 0, txSz, bestT, x >> 2, y >> 2, sbr4, sbc4);
    }
    rc_tick(B, P, 3);  // final quant + prediction + inverse + record
    *outMode = luma_mode(bestC);
    *outDelta = is_fi(bestC) || pal_cand(bestC) ? 0 : cand_delta(bestC);
    *outPal = pal_cand(bestC) ? S.palN : 0;
    *outFi = is_fi(bestC) ? bestC - 128 + 1 : 0;
}

// ---------------------------------------------------------------------------
// chroma of one block: U and V tx blocks, W x H = chroma tx size.  The mode decision (SATD screening,
// CfL, exact RD) uses the first tx block of each plane - the only one in 4:2:0 and for 4:4:4 blocks up to
// 32x32; 4:4:4 blocks with a 64-sample side (no CfL) code their further tx blocks after it in raster
// order, each predicted from the reconstruction of the previous ones (closed loop, as the decoder does).
template <int SS, int BD>
__device__ __noinline__ void rc_chroma(int W, int H, RcShared<BD> &S, const GpBatch &B, const GpParams &P, uint32_t sb, const GpImage &I, int bsize,
                          int mi_row, int mi_col, int yMode, int yTxDepth, int availUC, int availLC, const TileBox &tb, const BlockQ &bq,
                          float lambda, const float *C, int *outUv, int *outDuv, int *outAu, int *outAv, int *outEobU, int *outEobV) {
    const TxR T(W, H);
    const int tid = threadIdx.x;
    constexpr int ssx = SS, ssy = SS;
    const int txSz = av1g_uv_tx_size(bsize, ssx, ssy);
    const int cx = (mi_col >> ssx) * 4, cy = (mi_row >> ssy) * 4;     // chroma pixels (4:2:0 sub-8x8: the 8x8 area)
    const int x4 = cx >> 2, y4 = cy >> 2;
    const int sbr4 = (mi_row & 15) >> ssy, sbc4 = (mi_col & 15) >> ssx;
    const int stepX = W >> 2, stepY = H >> 2;
    typedef av1g_pixel_t(BD) Px;
    {
        const int pl = tid < 64 ? 1 : 2;
        const int haveAR = (S.bdm[pl][sbr4 - 1 + 1] >> (sbc4 + stepX + 1)) & 1;
        const int haveBL = (S.bdm[pl][sbr4 + stepY + 1] >> (sbc4 - 1 + 1)) & 1;
        rc_edges<SS, BD>(S, I, pl, cx, cy, T.lw, T.lh, availLC, availUC, haveAR, haveBL, mi_row, mi_col, pl == 1 ? 0 : 64, 64);
    }
    if (tid < 32) S.satd[tid] = 0.f;
    const Px *srcU = gp_src<BD>(I, 1) + (size_t)cy * I.spitch[1] + cx;
    const Px *srcV = gp_src<BD>(I, 2) + (size_t)cy * I.spitch[2] + cx;
    const bool cflOk = P.cfl && av1g_cfl_allowed(bsize);
    const float lamS = P.satdLambda * sqrtf(lambda);
    auto uvbits = [&](int m) -> float {
        if (av1g_cfl_allowed(bsize)) return C[CID(uv_mode_cfl_allowed[yMode]) + m];
        return C[CID(uv_mode_cfl_not_allowed[yMode]) + m];
    };
    const bool uvDeltaOk = P.uvDeltas && bsize >= AV1G_BLOCK_8X8;
    auto uvcbits = [&](int c) -> float {   // candidate code (mode, delta)
        const int m = cand_mode(c);
        float b = uvbits(m);
        if (bsize >= AV1G_BLOCK_8X8 && av1g_is_directional(m)) b += C[CID(angle_delta[m - AV1G_V_PRED]) + cand_delta(c) + 3];
        return b;
    };
    __syncthreads();
    // ---- CfL luma values (reconstructed luma, spec 7.11.5) ----
    if (cflOk) {
        // MaxLumaW/H (spec transform_block): right / bottom edge of the last luma tx block coded, in raster
        // order.  Tx blocks whose top-left lies outside the frame are not coded, so with a split luma tx
        // (tx_depth 1) on the right / bottom frame edge this can stop short of the block.
        const int lumaTx = av1g_luma_tx_size(bsize, yTxDepth);
        const int tw = AV1G_T(Tx_Width)[lumaTx], th = AV1G_T(Tx_Height)[lumaTx];
        int maxLumaW = 0, maxLumaH = 0;
        for (int ly = mi_row * 4; ly < mi_row * 4 + bh_px(bsize); ly += th)
            for (int lx = mi_col * 4; lx < mi_col * 4 + bw_px(bsize); lx += tw)
                if (lx < I.MiCols * 4 && ly < I.MiRows * 4) { maxLumaW = lx + tw; maxLumaH = ly + th; }
        const Px *luma = gp_rec<BD>(I, 0);
        const int lp = I.rpitch[0];
        int sum = 0;
        for (int p = tid; p < W * H; p += RC_NT) {
            const int i = p >> T.lw, j = p & (W - 1);
            int v;
            if (ssx) {   // 2x2 luma sum << 1 (spec 7.11.5, subX = subY = 1)
                int lumaY = (cy + i) << 1, lumaX = (cx + j) << 1;
                if (lumaY > maxLumaH - 2) lumaY = maxLumaH - 2;
                if (lumaX > maxLumaW - 2) lumaX = maxLumaW - 2;
                const int t = luma[lumaY * lp + lumaX] + luma[lumaY * lp + lumaX + 1] + luma[(lumaY + 1) * lp + lumaX] +
                              luma[(lumaY + 1) * lp + lumaX + 1];
                v = t << 1;
            } else {     // co-sited luma << 3 (subX = subY = 0)
                const int lumaY = min(cy + i, maxLumaH - 1), lumaX = min(cx + j, maxLumaW - 1);
                v = luma[lumaY * lp + lumaX] << 3;
            }
            rc_cflLw(S.u)[p] = (int16_t)v;
            sum += v;
        }
        sum = warp_sumi(sum);
        if (tid == 0) S.cflAvg = 0;
        __syncthreads();
        if ((tid & 31) == 0) atomicAdd(&S.cflAvg, sum);
        __syncthreads();
        if (tid == 0) S.cflAvg = av1g_round2(S.cflAvg, T.lw + T.lh);
        __syncthreads();
        // least-squares alpha per plane: r = src - DC, l = v - avg; alpha ~ 64 * sum(r l) / sum(l^2)
        float num[2] = {0.f, 0.f}, den = 0.f;
        for (int p = tid; p < W * H; p += RC_NT) {
            const int i = p >> T.lw, j = p & (W - 1);
            const float l = (float)(rc_cflL(S.u)[p] - S.cflAvg);
            num[0] += l * (float)((int)srcU[i * I.spitch[1] + j] - S.dcv[1]);
            num[1] += l * (float)((int)srcV[i * I.spitch[2] + j] - S.dcv[2]);
            den += l * l;
        }
        num[0] = warp_sum(num[0]); num[1] = warp_sum(num[1]); den = warp_sum(den);
        // The four warps' sums are added in warp order, 0 + w0 + w1 + w2 + w3.  They were added with a float atomicAdd
        // per warp, in arrival order: once a sum is not exact (10-bit, large high-contrast blocks) the rounding, and
        // through a tie in the +-1 refinement the alpha itself, could depend on which warp came first.  The fixed
        // order is one of those arrival orders.  S.taskJ is free until the chroma RD tasks.
        float *part = S.taskJ;
        const int w = tid >> 5;
        if ((tid & 31) == 0) { part[3 * w] = num[0]; part[3 * w + 1] = num[1]; part[3 * w + 2] = den; }
        __syncthreads();
        if (tid < 2) {
            float n = 0.f, d = 0.f;
            for (int k = 0; k < RC_NW; k++) { n += part[3 * k + tid]; d += part[3 * k + 2]; }
            int a = d > 0.f ? (int)rintf(64.f * n / d) : 0;
            S.cflAlpha[tid] = max(-16, min(16, a));
        }
        __syncthreads();
        // refine alpha +-1 by pixel SSE
        {
            float sse[2][3] = {{0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}};
            for (int p = tid; p < W * H; p += RC_NT) {
                const int i = p >> T.lw, j = p & (W - 1);
                const int l = rc_cflL(S.u)[p] - S.cflAvg;
                for (int pl = 0; pl < 2; pl++) {
                    const int s = pl == 0 ? srcU[i * I.spitch[1] + j] : srcV[i * I.spitch[2] + j];
                    for (int k = 0; k < 3; k++) {
                        const int a = max(-16, min(16, S.cflAlpha[pl] + k - 1));
                        const int pr = av1g_clip1<BD>(S.dcv[pl + 1] + av1g_round2signed(a * l, 6));
                        const float d = (float)(s - pr);
                        sse[pl][k] += d * d;
                    }
                }
            }
            for (int pl = 0; pl < 2; pl++)
                for (int k = 0; k < 3; k++) {
                    const float v = warp_sum(sse[pl][k]);
                    if ((tid & 31) == 0) part[6 * w + 3 * pl + k] = v;
                }
        }
        __syncthreads();
        if (tid == 0) {
            for (int pl = 0; pl < 2; pl++) {
                float sse[3];
                for (int k = 0; k < 3; k++) {
                    sse[k] = 0.f;
                    for (int ww = 0; ww < RC_NW; ww++) sse[k] += part[6 * ww + 3 * pl + k];   // warp order, as above
                }
                int bk = 1;
                for (int k = 0; k < 3; k++) if (sse[k] < sse[bk]) bk = k;
                S.cflAlpha[pl] = max(-16, min(16, S.cflAlpha[pl] + bk - 1));
            }
        }
        __syncthreads();
    }
    rc_tick(B, P, 4);  // chroma edges + CfL
    // ---- SATD screening of the 13 UV modes (real edges), then angle deltas +-1..3 around the best
    //      directional UV mode ----
    if (tid < 13) S.uvc[tid] = cand_pack(tid, 0);
    __syncthreads();
    int nuv = 13;
    for (int pass = 0; pass < 2; pass++) {
        const int c0 = pass ? 13 : 0, c1 = pass ? 19 : 13;
        if (pass) {
            if (!uvDeltaOk) break;
            if (tid < 32) {   // best directional mode at delta 0: lane m's cost, the first minimum as the serial scan
                const float j = tid >= AV1G_V_PRED && tid <= AV1G_D67_PRED ? fmaf(lamS, uvcbits(cand_pack(tid, 0)), S.satd[tid]) : GP_INF;
                const int bm = warp_argmin(j, tid);
                if (tid < 6) S.uvc[13 + tid] = cand_pack(bm, tid < 3 ? -(tid + 1) : (tid - 2));
            }
            __syncthreads();
            nuv = 19;
        }
        const int nsx = W / 4, nsub = (W / 4) * (H / 4);
        const int lx = T.lw - 2, ln = T.lw + T.lh - 4;   // log2 nsx, nsub (powers of two: shifts, no divisions)
        const int nc = c1 - c0;
        if (nsub >= 16) {   // 16x16 and up: one 4x4 sub-block per thread (enough of them for the CTA), pred_sub4
            // items = whole warps; a half-warp's sub-blocks share (plane, candidate): their integer sum goes into the
            // candidate's integer total with one shared-memory atomic, converted to float once below.  (A float
            // atomicAdd per half-warp rounded in arrival order once a total passed 2^22, which 10-bit 4:4:4 64x64
            // blocks can reach: 512 sub-blocks of up to 65472 / 4.)
            const int items = nc * nsub * 2;
            for (int it = tid; it < items; it += RC_NT) {
                const int pl = it < nc * nsub ? 1 : 2;
                const int rr = it - (pl - 1) * nc * nsub;
                const int ci = c0 + (rr >> ln), sbk = rr & (nsub - 1);
                const int sx = (sbk & (nsx - 1)) * 4, sy = (sbk >> lx) * 4;
                const Px *s = pl == 1 ? srcU : srcV;
                const int sp = I.spitch[pl];
                int d[16], p[16];
                pred_sub4(S.uvc[ci], sy, sx, T.lw, T.lh, S.ea[pl] + AV1G_EDGE_OFF, S.el[pl] + AV1G_EDGE_OFF, S.dcv[pl], p);
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int j = 0; j < 4; j++) d[i * 4 + j] = (int)s[(sy + i) * sp + sx + j] - p[i * 4 + j];
                int v = hadamard4_sum(d);
#pragma unroll
                for (int o = 1; o < 16; o <<= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
                if ((tid & 15) == 0) atomicAdd(&S.satdI[ci], v);
            }
            __syncthreads();
            // (float) is exact below 2^24, so this is the float sums' value wherever they were exact (0.f, 0: same bits)
            if (tid >= c0 && tid < c1) S.satd[tid] = (float)S.satdI[tid] * 0.25f;
            __syncthreads();
            continue;
        }
        // smaller blocks (F3): one 4x4 sub-block per thread with pred_sub4, as above.  A candidate's ipc = 2 * nsub
        // items (U then V sub-blocks) are adjacent lanes of one warp, 4 candidates per warp (2 for nsub = 8), so the 13
        // modes take one round (two for nsub = 8) instead of 13-26 rounds of one pixel per lane.  The 13 modes run in
        // an order that groups pred_sub4's branches: a warp sees at most two of them ({D113, D135, D157 | D203},
        // {D45, D67 | V, H}, {SMOOTH, SMOOTH_V, SMOOTH_H | PAETH}, {DC}).  The SATD of a candidate is an integer sum
        // (exact in float, see hadamard4_sum), reduced over its lanes and stored once: the same S.satd value.
        const int lane = tid & 31, warp = tid >> 5;
        const int lip = ln + 1, lcpw = lip <= 3 ? 2 : 1;   // log2 items per candidate (1..4), log2 candidates per warp
        const int nws = (nc + (1 << lcpw) - 1) >> lcpw;    // warp-sized slots of candidates
        for (int ws = warp; ws < nws; ws += RC_NW) {
            const int s = (ws << lcpw) + (lane >> lip), item = lane & ((1 << lip) - 1);
            const bool act = (lane >> lip) < (1 << lcpw) && s < nc;
            int v = 0;
            int ci = 0;
            if (act) {
                // pass 0: 5 4 6 7 | 3 8 1 2 | 9 10 11 12 | 0 (mode = candidate index); pass 1: the deltas in order
                ci = c0 + (pass ? s : (int)((0xCBA921837645ull >> (4 * s)) & 15));
                const int pl = 1 + (item >> ln), sbk = item & (nsub - 1);
                const int sx = (sbk & (nsx - 1)) * 4, sy = (sbk >> lx) * 4;
                const Px *sp0 = pl == 1 ? srcU : srcV;
                const int sp = I.spitch[pl];
                int d[16], p[16];
                pred_sub4(S.uvc[ci], sy, sx, T.lw, T.lh, S.ea[pl] + AV1G_EDGE_OFF, S.el[pl] + AV1G_EDGE_OFF, S.dcv[pl], p);
#pragma unroll
                for (int i = 0; i < 4; i++)
#pragma unroll
                    for (int j = 0; j < 4; j++) d[i * 4 + j] = (int)sp0[(sy + i) * sp + sx + j] - p[i * 4 + j];
                v = hadamard4_sum(d);
            }
#pragma unroll
            for (int o = 1; o < 16; o <<= 1)
                if (o < (1 << lip)) v += __shfl_xor_sync(0xffffffffu, v, o);
            if (act && item == 0) S.satd[ci] = (float)v * 0.25f;
        }
        __syncthreads();
    }
    rc_tick(B, P, 5);  // chroma SATD screening
    // ---- exact RD: top uvK modes (+ CfL) for U and V ----
    if (tid < 32) {
        // top K by SATD cost (candidate ci on lane ci): K rounds of warp_argmin give the K smallest in (cost, index) order,
        // the order of the serial insertion (strict <: an equal cost goes after the earlier candidate)
        const int K = max(1, min(P.uvK, 6));
        float js = tid < nuv ? fmaf(lamS, uvcbits(S.uvc[tid]), S.satd[tid]) : GP_INF;
        int n = 0;
        for (int k = 0; k < K; k++) {
            const int b = warp_argmin(js, tid);
            if (b < 0) break;
            if (tid == 0) {
                const int c = S.uvc[b];
                for (int pl = 1; pl <= 2; pl++) {
                    S.taskC[n + pl - 1] = c;
                    S.taskT[n + pl - 1] = av1g_chroma_tx_type(txSz, cand_mode(c), 0);
                    S.taskP[n + pl - 1] = pl;
                }
            }
            n += 2;
            if (tid == b) js = GP_INF;
        }
        if (tid == 0) S.nTask = n;
    }
    if (tid == 0) {
        int n = S.nTask;   // thread 0's own store above
        if (cflOk && (S.cflAlpha[0] || S.cflAlpha[1])) {
            for (int pl = 1; pl <= 2; pl++) {
                S.taskC[n] = 255;
                S.taskT[n] = av1g_chroma_tx_type(txSz, AV1G_UV_CFL_PRED, 0);
                S.taskP[n] = pl;
                n++;
            }
        }
        S.nTask = n;
    }
    __syncthreads();
    int skipCtxU, dcCtxU, skipCtxV, dcCtxV;
    txb_ctx<SS>(I, 1, txSz, x4, y4, bsize, tb, &skipCtxU, &dcCtxU);
    txb_ctx<SS>(I, 2, txSz, x4, y4, bsize, tb, &skipCtxV, &dcCtxV);
    EvalIn e;
    e.txSz = txSz; e.plane = 1; e.dcQ = bq.dcQ[1]; e.acQ = bq.acQ[1];
    e.rnd = P.rndC; e.zbin = P.zbinC; e.skipCtx = skipCtxU; e.dcCtx = dcCtxU; e.reduced = 0; e.C = C;
    e.intraDir = 0; e.txType = 0; e.qm = 0; e.resEnergy = 0;
    e.qmDist = P.qmDist; e.qmRnd = P.qmRnd;
    if (W <= 8 && H <= 8 && S.nTask > RC_NW) {
        // tx up to 8x8 with more tasks than warps (uvK 2 + CfL: 6): one task per half-warp, one round instead of two;
        // the same values (rc_eval.cuh), levels kept as below
        const RcLanes<16> L = rc_half_lanes();
        const RcHalfScr Hs = rc_half_scr(S);
        for (int t = tid >> 4; t < S.nTask; t += 2 * RC_NW) {
            const int pl = S.taskP[t], c = S.taskC[t];
            const bool cfl = c == 255;
            EvalIn ee = e;
            ee.plane = pl;
            ee.dcQ = bq.dcQ[pl]; ee.acQ = bq.acQ[pl];
            ee.skipCtx = pl == 1 ? skipCtxU : skipCtxV;
            ee.dcCtx = pl == 1 ? dcCtxU : dcCtxV;
            ee.txType = S.taskT[t];
            ee.qm = av1g_qm(I.usingQm, I.qm[pl], pl, txSz, ee.txType);
            rc_residual_half<BD>(W, H, S, pl, c, cfl, cfl ? S.cflAlpha[pl - 1] : 0, pl == 1 ? srcU : srcV, I.spitch[pl], Hs, L);
            const EvalOut o = rc_eval<BD, 16>(Hs.res, Hs.tmp, Hs.q, ee, W, H, L);
            int16_t *dq = rc_taskq(S.u, t);   // nTask <= 6 * 2 + 2 = RC_TASKQ_MAX: all kept
            for (int i = L.l; i < W * H; i += 16) dq[i] = Hs.q[i];
            if (L.l == 0) { S.taskE[t] = o.eob; S.taskJ[t] = fmaf(lambda, o.R, o.D); }
        }
        __syncthreads();
    } else {
        const int warp = tid >> 5;
        const bool big = rc_big(W, H);
        for (int t = warp; t < S.nTask; t += RC_NW) {   // W5: big tasks on every warp (own global slot)
            const int pl = S.taskP[t], c = S.taskC[t];
            const bool cfl = c == 255;
            RcBigScr &BG = S.bigG[warp];
            int16_t *res = big ? BG.res : S.u.w[warp].res;
            float *tmp = big ? BG.tmp : S.u.w[warp].tmp;
            int16_t *qq = big ? BG.q : S.u.w[warp].q;
            EvalIn ee = e;
            ee.plane = pl;
            ee.dcQ = bq.dcQ[pl]; ee.acQ = bq.acQ[pl];
            ee.skipCtx = pl == 1 ? skipCtxU : skipCtxV;
            ee.dcCtx = pl == 1 ? dcCtxU : dcCtxV;
            ee.txType = S.taskT[t];
            ee.qm = av1g_qm(I.usingQm, I.qm[pl], pl, txSz, ee.txType);
            ee.resEnergy = rc_residual<BD>(W, H, S, pl, c, cfl, cfl ? S.cflAlpha[pl - 1] : 0, pl == 1 ? srcU : srcV, I.spitch[pl], res);
            const EvalOut o = rc_eval<BD, 32>(res, tmp, qq, ee, W, H, rc_warp_lanes());
            if (!big && t < RC_TASKQ_MAX) {   // keep the levels: the final of the winner reuses them
                int16_t *dq = rc_taskq(S.u, t);
                for (int i = tid & 31; i < W * H; i += 32) dq[i] = qq[i];
                if ((tid & 31) == 0) S.taskE[t] = o.eob;
            } else if (big && t < RC_BIGQ_MAX) {   // W3: big tasks keep theirs in global memory
                int16_t *dq = S.bigQ + t * 1024;
                for (int i = tid & 31; i < T.n; i += 32) dq[i] = qq[i];
                if ((tid & 31) == 0) S.taskE[t] = o.eob;
            }
            if ((tid & 31) == 0) S.taskJ[t] = fmaf(lambda, o.R, o.D);
        }
        __syncthreads();
    }
    // pick the best mode (sum over U and V + mode bits)
    int bestUv = AV1G_DC_PRED, bestUvC = cand_pack(AV1G_DC_PRED, 0);
    {
        float bJ = GP_INF;
        for (int t = 0; t + 1 < S.nTask; t += 2) {
            const int c = S.taskC[t];
            float J = S.taskJ[t] + S.taskJ[t + 1];
            int uv;
            if (c == 255) {
                uv = AV1G_UV_CFL_PRED;
                const int aU = S.cflAlpha[0], aV = S.cflAlpha[1];
                const int sU = aU == 0 ? 0 : aU < 0 ? 1 : 2, sV = aV == 0 ? 0 : aV < 0 ? 1 : 2;
                float b = uvbits(AV1G_UV_CFL_PRED) + C[CID(cfl_sign) + sU * 3 + sV - 1];
                if (sU) b += C[CID(cfl_alpha[(sU - 1) * 3 + sV]) + abs(aU) - 1];
                if (sV) b += C[CID(cfl_alpha[(sV - 1) * 3 + sU]) + abs(aV) - 1];
                J += lambda * b;
            } else {
                uv = cand_mode(c);
                J += lambda * uvcbits(c);
            }
            if (J < bJ) { bJ = J; bestUv = uv; bestUvC = c; }
        }
    }
    __syncthreads();
    rc_tick(B, P, 6);  // chroma exact RD tasks
    const bool cfl = bestUv == AV1G_UV_CFL_PRED;
    const int uvTxType = av1g_chroma_tx_type(txSz, bestUv, 0);
    // chroma tx blocks per plane: 1 (4:2:0, and 4:4:4 up to 32x32), 2 or 4 for 4:4:4 blocks with a 64-sample side
    const int ub = av1g_uv_bsize(bsize, ssx, ssy);
    const int nTxX = SS ? 1 : max(1, bw_px(ub) / W), nTxY = SS ? 1 : max(1, bh_px(ub) / H);
    const int cCols4 = I.MiCols >> ssx, cRows4 = I.MiRows >> ssy;
    int eobs[2] = {0, 0};
    const bool bigC = rc_big(W, H);
    int keptUV[2] = {-1, -1};
    for (int t = 0; t < S.nTask && t < (bigC ? RC_BIGQ_MAX : RC_TASKQ_MAX); t++)
        for (int k = 0; k < 2; k++)
            if (keptUV[k] < 0 && S.taskP[t] == 1 + k && S.taskC[t] == (cfl ? 255 : bestUvC)) keptUV[k] = t;
    if (SS && keptUV[0] >= 0 && keptUV[1] >= 0) {
        // 4:2:0 (one tx block per plane), both winners' levels kept: U on threads 0-63 and V on 64-127 at once,
        // each with its own levels, residual buffer and edge slot (0 / 2); recorded in the loop's order (U, V).
        // Big tx: U's levels in S.qfin, V's in S.u.inv.res[3072, 4096) (above both residual buffers)
        const RcGrp g = rc_half();
        const int pl = 1 + g.k, kept = keptUV[g.k], c = cfl ? 255 : bestUvC;
        int16_t *q = !bigC ? rc_taskq(S.u, kept) : g.k ? S.u.inv.res + 3072 : S.qfin;
        if (bigC) for (int i = g.t; i < T.n; i += g.nt) q[i] = S.bigQ[kept * 1024 + i];
        Px *dst = gp_rec<BD>(I, pl) + (size_t)cy * I.rpitch[pl] + cx;
        if (g.t < 32) rc_prepare_edges<BD>(S, pl, c, cfl, T.lw, T.lh);
        g.sync();
        rc_write_pred<BD>(W, H, S, pl, c, cfl, cfl ? S.cflAlpha[pl - 1] : 0, dst, I.rpitch[pl], g, 2 * g.k);
        if (S.taskE[kept] > 0) {
            const int fixes = cta_inverse_add(W, H, S, q, txSz, uvTxType, bq.dcQ[pl], bq.acQ[pl], av1g_qm(I.usingQm, I.qm[pl], pl, txSz, uvTxType),
                                              dst, I.rpitch[pl], g);
            if (fixes && g.t == 0) atomicAdd(&B.stats[3], 1u);
        }
        const int eob = rc_record_txb_at<SS, BD>(W, H, S, B, sb, I, pl, txSz, uvTxType, x4, y4, sbr4, sbc4, q, g,
                                                 (uint32_t)S.nCoef + g.k * T.n, S.nTxb + g.k);
        if (g.t == 0) S.eobUV[g.k] = eob;
        __syncthreads();
        // one word each: merged into a wider load they would also read S.nTxb, which thread 0 updates below
        eobs[0] = *(const volatile int *)&S.eobUV[0];
        eobs[1] = *(const volatile int *)&S.eobUV[1];
        if (tid == 0) { S.nTxb += 2; S.nCoef += 2 * T.n; }
        __syncthreads();
    } else
    for (int pl = 1; pl <= 2; pl++) {
        EvalIn ee = e;
        ee.plane = pl;
        ee.dcQ = bq.dcQ[pl]; ee.acQ = bq.acQ[pl];
        ee.skipCtx = pl == 1 ? skipCtxU : skipCtxV;
        ee.dcCtx = pl == 1 ? dcCtxU : dcCtxV;
        ee.txType = uvTxType;
        ee.qm = av1g_qm(I.usingQm, I.qm[pl], pl, txSz, uvTxType);
        const int alpha = cfl ? S.cflAlpha[pl - 1] : 0;
        const int c = cfl ? 255 : bestUvC;
        const Px *s = pl == 1 ? srcU : srcV;
        for (int ty = 0; ty < nTxY; ty++)
            for (int tx = 0; tx < nTxX; tx++) {
                const int ox = tx * W, oy = ty * H;   // offset inside the chroma block
                if (!SS && (x4 + (ox >> 2) >= cCols4 || y4 + (oy >> 2) >= cRows4)) continue;   // outside the coded area: not coded
                Px *dst = gp_rec<BD>(I, pl) + (size_t)(cy + oy) * I.rpitch[pl] + cx + ox;
                if (SS || (tx == 0 && ty == 0)) {
                    const int kept = keptUV[pl - 1];
                    if (kept >= 0) {
                        if (bigC) for (int i = tid; i < T.n; i += RC_NT) S.qfin[i] = S.bigQ[kept * 1024 + i];
                        else for (int i = tid; i < W * H; i += RC_NT) S.qfin[i] = rc_taskq(S.u, kept)[i];
                        if (tid == 0) S.finEob = S.taskE[kept];
                        if (tid < 32) rc_prepare_edges<BD>(S, pl, c, cfl, T.lw, T.lh);
                        __syncthreads();
                        rc_write_pred<BD>(W, H, S, pl, c, cfl, alpha, dst, I.rpitch[pl], rc_cta());
                    } else {
                        rc_final_quant<BD>(W, H, S, ee, pl, c, cfl, alpha, s, I.spitch[pl]);
                        rc_write_from_res(W, H, s, I.spitch[pl], rc_big(W, H) ? S.bigG->res : S.u.w[0].res, dst, I.rpitch[pl]);
                    }
                } else {
                    // a further 4:4:4 tx block: its edges from the reconstruction (the previous tx blocks are done),
                    // its contexts, closed-loop final quantization; AvailU/LChroma of the block for the edge filter type
                    const int r4 = sbr4 + (oy >> 2), c4 = sbc4 + (ox >> 2);   // SB-relative, plane 4x4 units
                    const int haveAR = (S.bdm[pl][r4 - 1 + 1] >> (c4 + (W >> 2) + 1)) & 1;
                    const int haveBL = (S.bdm[pl][r4 + (H >> 2) + 1] >> (c4 - 1 + 1)) & 1;
                    rc_edges<SS, BD>(S, I, pl, cx + ox, cy + oy, T.lw, T.lh, availLC || tx > 0, availUC || ty > 0, haveAR, haveBL, mi_row,
                                     mi_col, 0, 64, availUC, availLC);
                    __syncthreads();
                    int sc, dc;
                    txb_ctx<SS>(I, pl, txSz, x4 + (ox >> 2), y4 + (oy >> 2), bsize, tb, &sc, &dc);
                    EvalIn et = ee;
                    et.skipCtx = sc; et.dcCtx = dc;
                    const Px *so = s + (size_t)oy * I.spitch[pl] + ox;
                    rc_final_quant<BD>(W, H, S, et, pl, c, cfl, alpha, so, I.spitch[pl]);
                    rc_write_from_res(W, H, so, I.spitch[pl], rc_big(W, H) ? S.bigG->res : S.u.w[0].res, dst, I.rpitch[pl]);
                }
                if (S.finEob > 0) {
                    const int fixes = cta_inverse_add(W, H, S, S.qfin, txSz, uvTxType, ee.dcQ, ee.acQ, ee.qm, dst, I.rpitch[pl], rc_cta());
                    if (fixes && tid == 0) atomicAdd(&B.stats[3], 1u);
                }
                eobs[pl - 1] |= rc_record_txb<SS, BD>(W, H, S, B, sb, I, pl, txSz, uvTxType, x4 + (ox >> 2), y4 + (oy >> 2), sbr4 + (oy >> 2),
                                                  sbc4 + (ox >> 2));
            }
    }
    rc_tick(B, P, 7);  // chroma final quant + recon
    *outUv = bestUv;
    *outDuv = cfl ? 0 : cand_delta(bestUvC);
    *outAu = cfl ? S.cflAlpha[0] : 0;
    *outAv = cfl ? S.cflAlpha[1] : 0;
    *outEobU = eobs[0];
    *outEobV = eobs[1];
}

// ---------------------------------------------------------------------------
// SS: chroma subsampling of the frames in sbList (1: 4:2:0 and monochrome, 0: 4:4:4 - monochrome frames may be
// in either list, they have no chroma); gp_launch_recon launches one instantiation per layout.  BD: bit depth of
// the batch.
template <int SS, int BD>
__global__ void __launch_bounds__(RC_NT, 4) k_recon(GpBatch B, GpParams P, const uint32_t *sbList) {
    __shared__ RcShared<BD> S;
    const uint32_t sb = sbList[blockIdx.x];
    const GpImage &I = B.img[B.sbImg[sb]];
    const uint32_t local = sb - I.sbBase;
    const int sbc = (int)(local % I.sbCols), sbr = (int)(local / I.sbCols);
    const int sbx = sbc * 64, sby = sbr * 64;
    const int tid = threadIdx.x;
    const TileBox tb = tile_of(I, sbx, sby);
    const int miRowEnd = tb.y1 >> 2, miColEnd = tb.x1 >> 2;
    const long long sbT0 = clock64();
    if (P.poisonSmem) {   // debug: uninitialised shared-memory reads become a fixed pattern (0x7F7F levels)
        for (int i = tid; i < (int)(sizeof(RcShared<BD>) / 4); i += RC_NT) ((uint32_t *)&S)[i] = 0x7F7F7F7Fu;
        __syncthreads();
    }
    // clear_block_decoded_flags (spec): one thread per (plane, row) word
    if (tid < 3 * 18) {
        const int p = tid / 18, yy = tid % 18 - 1;
        const int subX = p ? SS : 0, subY = p ? SS : 0;
        const int sbW4 = (miColEnd - sbc * 16) >> subX, sbH4 = (miRowEnd - sbr * 16) >> subY;
        uint32_t w = 0;
        for (int xx = -1; xx <= 16; xx++) {
            int v;
            if (yy < 0 && xx < sbW4) v = 1;
            else if (xx < 0 && yy < sbH4) v = 1;
            else v = 0;
            if (yy == (16 >> subY) && xx == -1) v = 0;
            w |= (uint32_t)v << (xx + 1);
        }
        S.bdm[p][yy + 1] = w;
    }
    const int nLeaves = B.leafCountW[sb];
    if (tid == 0) { S.nTxb = 0; S.nCoef = 0; S.nPal = 0; S.nMap = 0; S.edgeFilter = I.edgeFilter; S.bigG = (RcBigScr *)((char *)B.bigScratch + (size_t)blockIdx.x * RC_CTA_BIG_BYTES); S.bigQ = (int16_t *)(S.bigG + RC_NW); }
    const int qidx = sb_qindex(I, sbx, sby);
    const BlockQ bq = block_q<BD>(I, qidx);
    const float lambda = gp_lambda<BD>(qidx, P.lambdaMul);
    const float *C = g_cost[I.qctx];
    __syncthreads();
    rc_tick(B, P, -1);
    for (int li = 0; li < nLeaves; li++) {
        const GpLeaf lf = B.leaves[(size_t)sb * GP_MAX_LEAVES + li];
        const int bsize = lf.bsize;
        const int x = sbx + lf.x4 * 4, y = sby + lf.y4 * 4;
        const int mi_row = y >> 2, mi_col = x >> 2;
        int availU, availL, availUC, availLC;
        av1g_block_avail(bsize, mi_row, mi_col, I.mono, SS, SS, tb.y0 >> 2, miRowEnd, tb.x0 >> 2, miColEnd, &availU, &availL,
                         &availUC, &availLC);
        const int hasChroma = av1g_has_chroma(bsize, mi_row, mi_col, I.mono, SS, SS);
        const uint32_t firstTxb = (uint32_t)S.nTxb;
        const float lambdaB = lambda * ssim_scale(I, x, y, bw_px(bsize), bh_px(bsize));
        int yMode = 0, yDelta = 0, eobY = 0, yFi = 0, yTxDepth = 0, yPal = 0;
        const int txSz = AV1G_T(Max_Tx_Size_Rect)[bsize];
rc_luma<BD>(AV1G_T(Tx_Width)[txSz], AV1G_T(Tx_Height)[txSz], S, B, P, sb, I, bsize, x, y, availU, availL, tb, bq, lambdaB, C, &yMode, &yDelta, &eobY, &yFi, &yTxDepth, &yPal);
        int uv = 0, duv = 0, aU = 0, aV = 0, eobU = 0, eobV = 0;
        if (hasChroma) {
            const int uvTx = av1g_uv_tx_size(bsize, SS, SS);
rc_chroma<SS, BD>(AV1G_T(Tx_Width)[uvTx], AV1G_T(Tx_Height)[uvTx], S, B, P, sb, I, bsize, mi_row, mi_col, yMode, yTxDepth, availUC, availLC, tb, bq, lambdaB * P.lambdaCMul, C, &uv, &duv, &aU, &aV, &eobU, &eobV);
        }
        const int skip = (eobY == 0 && eobU == 0 && eobV == 0) ? 1 : 0;
        // maps
        const int bw4 = av1g_bw4(bsize), bh4 = av1g_bh4(bsize), lbw4 = log2i(bw4);   // F6: powers of two, fields read once
        {
            const int miRows = I.MiRows, miCols = I.MiCols;
            uint8_t *mY = I.mYMode, *mUV = I.mUVMode, *mS = I.mSkip, *mM = I.mMiSize, *mT = I.mTx;
            const uint8_t txv = (uint8_t)(yTxDepth ? AV1G_T(Split_Tx_Size)[txSz] : txSz);
            for (int k = tid; k < bw4 * bh4; k += RC_NT) {
                const int r = mi_row + (k >> lbw4), c = mi_col + (k & (bw4 - 1));
                if (r < miRows && c < miCols) {
                    const size_t o = (size_t)r * miCols + c;
                    mY[o] = (uint8_t)yMode;
                    if (hasChroma) mUV[o] = (uint8_t)uv;
                    mS[o] = (uint8_t)skip;
                    mM[o] = (uint8_t)bsize;
                    mT[o] = txv;
                }
            }
        }
        if (tid == 0) {
            av1g_block b;
            b.mi_row = (uint16_t)mi_row; b.mi_col = (uint16_t)mi_col;
            b.bsize = (uint8_t)bsize; b.skip = (uint8_t)skip; b.tx_depth = (uint8_t)yTxDepth;
            b.y_mode = (uint8_t)yMode; b.angle_delta_y = (int8_t)(bsize >= AV1G_BLOCK_8X8 && av1g_is_directional(yMode) ? yDelta : 0);
            b.uv_mode = (uint8_t)(hasChroma ? uv : 0);
            b.angle_delta_uv = (int8_t)(hasChroma && bsize >= AV1G_BLOCK_8X8 && av1g_is_directional(uv) ? duv : 0);
            b.cfl_alpha_u = (int8_t)aU; b.cfl_alpha_v = (int8_t)aV;
            b.filter_intra = (uint8_t)yFi; b.cdef_idx = 0;
            b.sb_qindex = (uint8_t)(I.sbQ ? qidx : 0);
            B.oBlocks[(size_t)sb * GP_MAX_LEAVES + li] = b;
            B.oBlockTxb[(size_t)sb * GP_MAX_LEAVES + li] = (uint16_t)firstTxb;
            if (yPal) {   // the luma palette: its map is already at S.nMap of the SB's slot
                av1g_palette p;
                memset(&p, 0, sizeof p);
                p.block = (uint32_t)li;
                p.map_off = (uint32_t)S.nMap;
                p.size_y = (uint8_t)yPal;
                for (int k = 0; k < yPal; k++) p.colors[0][k] = S.palCol[k];
                B.oPal[(size_t)sb * GP_SB_PAL + S.nPal] = p;
                S.nPal++;
                S.nMap += bw_px(bsize) * bh_px(bsize);
            }
            if (skip) atomicAdd(&B.stats[1], 1u);
            if (hasChroma && uv == AV1G_UV_CFL_PRED) atomicAdd(&B.stats[2], 1u);
            atomicAdd(&B.stats[0], 1u);
        }
        __syncthreads();
        rc_tick(B, P, 8);  // maps + block record
    }
    if (tid == 0) {
        GpSbOut o;
        memset(&o, 0, sizeof o);
        o.nBlocks = (uint32_t)nLeaves; o.nTxb = (uint32_t)S.nTxb; o.nCoeffs = (uint32_t)S.nCoef;
        o.nPal = (uint32_t)S.nPal; o.nMap = (uint32_t)S.nMap;
        B.sbOut[sb] = o;
        if (P.profile && B.sbCyc) B.sbCyc[sb] = (uint32_t)min(clock64() - sbT0, 0xffffffffLL);
    }
}
