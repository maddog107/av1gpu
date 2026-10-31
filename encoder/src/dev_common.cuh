// device-side tables and helpers (included once, by kernels.cu).
#pragma once
#include <cuda_runtime.h>
#include <stdint.h>
#include "av1g/hd.h"
#include "av1g/enums_gen.h"
#include "av1g/spec_tables.h"
#include "av1g/dsp.h"
#include "av1g/blockinfo.h"
#include "av1g/cdfctx.h"
#include "gp.h"

#define GP_INF 3.0e38f

// ---------------------------------------------------------------------------
// forward transform matrices (float inverse of av1core's exact 1-D inverse transforms)
//   g_fwd + g_fwdOff[kind][log2n]: n x n row-major, coef[i] = sum_k F[i][k] * x[k]
//   kinds: AV1G_K_DCT (n 4..64), AV1G_K_ADST (4..16), AV1G_K_IDT (4..32)
// g_txg[txSz][txType]: pixel-domain energy per unit coefficient energy (orthogonal-up-to-scale)
// g_txScale[txSz]: 2^(rowShift + 4) / rect   (forward output scale, dequant domain)
// ---------------------------------------------------------------------------
#define GP_FWD_FLOATS ((16 + 64 + 256 + 1024 + 4096) + (16 + 64 + 256) + (16 + 64 + 256 + 1024))

// Host copy of all tables (filled once by gp_build_tables in k_misc.cu); every translation unit keeps
// its own static __device__ copies (no relocatable device code needed) and uploads them in its
// gp_upload_tables_<tu>() function (see GP_DEFINE_TABLE_UPLOAD).
struct GpHostTables {
    float fwd[GP_FWD_FLOATS];
    float fwdT[GP_FWD_FLOATS];   // each matrix transposed (column-major F): float4 loads along the output index
    int fwdOff[3][7];
    float txg[19][16];
    float txScale[19];
    float cost[4][GP_NCDF];
    float avgYMode[13];
    float avgPart[5][10];
    float avgSkip0;
};

static __device__ float g_fwd[GP_FWD_FLOATS];
static __device__ __align__(16) float g_fwdT[GP_FWD_FLOATS];
static __device__ int g_fwdOff[3][7];
static __device__ float g_txg[19][16];
static __device__ float g_txScale[19];
// symbol costs (bits) parallel to av1g_cdfs, one table per coefficient-CDF q-context
static __device__ float g_cost[4][GP_NCDF];
// open-loop averages over unknown neighbour contexts
static __device__ float g_avgYMode[13];
static __device__ float g_avgPart[5][10];   // [bsl 1..4][partition]
static __device__ float g_avgSkip0;         // skip = 0, averaged over contexts

#define GP_DEFINE_TABLE_UPLOAD(name)                                                              \
    int name(const GpHostTables *h) {                                                             \
        cudaError_t e = cudaSuccess;                                                              \
        if (e == cudaSuccess) e = cudaMemcpyToSymbol(g_fwd, h->fwd, sizeof h->fwd);               \
        if (e == cudaSuccess) e = cudaMemcpyToSymbol(g_fwdT, h->fwdT, sizeof h->fwdT);            \
        if (e == cudaSuccess) e = cudaMemcpyToSymbol(g_fwdOff, h->fwdOff, sizeof h->fwdOff);      \
        if (e == cudaSuccess) e = cudaMemcpyToSymbol(g_txg, h->txg, sizeof h->txg);               \
        if (e == cudaSuccess) e = cudaMemcpyToSymbol(g_txScale, h->txScale, sizeof h->txScale);   \
        if (e == cudaSuccess) e = cudaMemcpyToSymbol(g_cost, h->cost, sizeof h->cost);            \
        if (e == cudaSuccess) e = cudaMemcpyToSymbol(g_avgYMode, h->avgYMode, sizeof h->avgYMode); \
        if (e == cudaSuccess) e = cudaMemcpyToSymbol(g_avgPart, h->avgPart, sizeof h->avgPart);   \
        if (e == cudaSuccess) e = cudaMemcpyToSymbol(g_avgSkip0, &h->avgSkip0, sizeof h->avgSkip0); \
        return e == cudaSuccess ? 0 : -1;                                                         \
    }

// dummy instance: its field addresses define the cdf ids (same convention as av1core gpu_entropy)
static __device__ av1g_cdfs g_cdf_layout;
#define CID(expr) ((uint32_t)((const uint16_t *)&(g_cdf_layout.expr) - (const uint16_t *)&g_cdf_layout))

__device__ __forceinline__ float cost_of(const float *C, uint32_t cid, int s) { return C[cid + s]; }

// ---------------------------------------------------------------------------
// small helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ int warp_max(int v) {
    for (int o = 16; o > 0; o >>= 1) v = max(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
__device__ __forceinline__ int warp_sumi(int v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

__device__ __forceinline__ int bw_px(int bs) { return 4 * av1g_bw4(bs); }
__device__ __forceinline__ int bh_px(int bs) { return 4 * av1g_bh4(bs); }

// libaom-style lambda (per bit, in 8-bit SSE units): rdmult = dc_q^2 * (3.3 + 0.0015 dc_q) * w/128 (tune=iq
// key frame), lambda = rdmult / 2048 (RDCOST: R in 1/512 bit units * rdmult >> 9, D = SSE * 16 << 7).
// 10 bits (av1_compute_rd_mult_based_on_qindex): dc_q of the 10-bit table, and libaom divides rdmult by 16
// (ROUND_POWER_OF_TWO(rdmult, 4)) while it normalizes every distortion by 2^(2 (BitDepth - 8)); this search
// keeps distortions (SSE, SATD) in native 10-bit units, so the lambda here is that rdmult times 16 (= the
// expression without the division): the same J order, every distortion / lambda scale consistent.
template <int BD = 8>
__host__ __device__ inline float gp_lambda(int qindex, float mul) {
    const int q = AV1G_T(Dc_Qlookup)[(BD - 8) >> 1][qindex < 0 ? 0 : qindex > 255 ? 255 : qindex];
    double rd = (double)q * q * (3.3 + 0.0015 * q);
    int w = ((255 - qindex) * 3) / 4;
    w = w < 0 ? 0 : w > 72 ? 72 : w;
    rd = rd * (128 + w) / 128.0;
    return (float)(rd / 2048.0 * mul);
}

// sample planes of an image (10-bit: uint16_t; pitches are in samples)
template <int BD> __device__ __forceinline__ const av1g_pixel_t(BD) *gp_src(const GpImage &I, int p) {
    return (const av1g_pixel_t(BD) *)I.src[p];
}
template <int BD> __device__ __forceinline__ av1g_pixel_t(BD) *gp_rec(const GpImage &I, int p) { return (av1g_pixel_t(BD) *)I.rec[p]; }

__device__ __forceinline__ int sb_qindex(const GpImage &I, int x, int y) {
    if (!I.sbQ) return I.baseQ;
    return I.sbQ[(y >> 6) * I.sbCols + (x >> 6)];
}

// quantizers of one block (spec: set_block_q)
struct BlockQ {
    int dcQ[3], acQ[3];
};
template <int BD = 8>
__device__ __forceinline__ BlockQ block_q(const GpImage &I, int qidx) {
    BlockQ q;
    q.dcQ[0] = av1g_dc_q<BD>(qidx); q.acQ[0] = av1g_ac_q<BD>(qidx);
    q.dcQ[1] = av1g_dc_q<BD>(qidx + I.dqUDc); q.acQ[1] = av1g_ac_q<BD>(qidx + I.dqUAc);
    q.dcQ[2] = av1g_dc_q<BD>(qidx + I.dqVDc); q.acQ[2] = av1g_ac_q<BD>(qidx + I.dqVAc);
    return q;
}

// tile containing luma pixel (x, y): bounds in pixels
struct TileBox { int x0, y0, x1, y1; };
__device__ __forceinline__ TileBox tile_of(const GpImage &I, int x, int y) {
    TileBox t;
    int c = 0, r = 0;
    while (c + 1 < I.tileCols && x >= I.tileColStart[c + 1] * 4) c++;
    while (r + 1 < I.tileRows && y >= I.tileRowStart[r + 1] * 4) r++;
    t.x0 = I.tileColStart[c] * 4; t.x1 = I.tileColStart[c + 1] * 4;
    t.y0 = I.tileRowStart[r] * 4; t.y1 = I.tileRowStart[r + 1] * 4;
    return t;
}

__device__ __forceinline__ int zorder16(int x, int y) {  // x, y in 0..15
    int z = 0;
    for (int b = 0; b < 4; b++) z |= (((x >> b) & 1) << (2 * b)) | (((y >> b) & 1) << (2 * b + 1));
    return z;
}

// Open-loop guess of "pixel (px,py) (plane units, sub = chroma shift) is decoded before the block at
// (x,y)": previous SB row and left SB as the spec's BlockDecoded init, Z-order inside the SB.
__device__ __forceinline__ bool ol_decoded(int px, int py, int x, int y, int sub, const TileBox &t) {
    const int tx0 = t.x0 >> sub, ty0 = t.y0 >> sub, tx1 = t.x1 >> sub, ty1 = t.y1 >> sub;
    if (px < tx0 || py < ty0 || px >= tx1 || py >= ty1) return false;
    const int S = 64 >> sub;
    const int sbx = (x / S) * S, sby = (y / S) * S;
    if (py < sby) return true;
    if (px < sbx) return py < sby + S;
    if (px >= sbx + S || py >= sby + S) return false;
    const int u = 4;  // 4x4 units of the plane
    return zorder16(((px - sbx) / u) & 15, ((py - sby) / u) & 15) < zorder16(((x - sbx) / u) & 15, ((y - sby) / u) & 15);
}

// candidate packing: mode << 3 | (delta + 3)
__device__ __forceinline__ int cand_mode(int c) { return c >> 3; }
__device__ __forceinline__ int cand_delta(int c) { return (c & 7) - 3; }
__device__ __forceinline__ int cand_pack(int m, int d) { return (m << 3) | (d + 3); }

// Parallel replica of av1g_build_edges (spec 7.11.2 edge preparation): threads tid = 0..nt-1 of a group
// fill AboveRow[-1 .. w+h-1] / LeftCol[-1 .. w+h-1].  Identical output (checked against the serial one).
template <int BD = 8, typename P>
__device__ __forceinline__ void build_edges_par(const P *plane, int stride, int x, int y, int lw, int lh, int haveL,
                                                int haveA, int haveAR, int haveBL, int maxX, int maxY, P *above,
                                                P *left, int tid, int nt) {
    AV1G_PIX_CHECK(BD, P);
    const int w = 1 << lw, h = 1 << lh, n = w + h;
    int alim = x + (haveAR ? 2 * w : w) - 1;
    if (alim > maxX) alim = maxX;
    int llim = y + (haveBL ? 2 * h : h) - 1;
    if (llim > maxY) llim = maxY;
    for (int i = tid; i < n; i += nt) {
        P a, l;
        if (!haveA && haveL) a = plane[(size_t)y * stride + x - 1];
        else if (!haveA && !haveL) a = (1 << (BD - 1)) - 1;
        else a = plane[(size_t)(y - 1) * stride + min(x + i, alim)];
        if (!haveL && haveA) l = plane[(size_t)(y - 1) * stride + x];
        else if (!haveL && !haveA) l = (1 << (BD - 1)) + 1;
        else l = plane[(size_t)min(y + i, llim) * stride + x - 1];
        above[i] = a;
        left[i] = l;
    }
    if (tid == 0) {
        P c;
        if (haveA && haveL) c = plane[(size_t)(y - 1) * stride + x - 1];
        else if (haveA) c = plane[(size_t)(y - 1) * stride + x];
        else if (haveL) c = plane[(size_t)y * stride + x - 1];
        else c = 1 << (BD - 1);
        above[-1] = c;
        left[-1] = c;
    }
}

// tune=iq per-block rdmult scale: geometric mean of the per-16x16 SSIM factors the block covers
// (libaom av1_set_ssim_rdmult)
__device__ __forceinline__ float ssim_scale(const GpImage &I, int x, int y, int w, int h) {
    if (!I.ssimLog) return 1.f;
    const int c0 = x >> 4, r0 = y >> 4;
    const int nc = (w + 15) >> 4, nr = (h + 15) >> 4;
    float s = 0.f;
    int n = 0;
    for (int r = r0; r < r0 + nr && r < I.rows16; r++)
        for (int c = c0; c < c0 + nc && c < I.cols16; c++) { s += I.ssimLog[r * I.cols16 + c]; n++; }
    return n ? __expf(s / n) : 1.f;
}

// flat-area pruning of shapes below 16x16 (GpParams::pruneFlat); var16 and the step are in native units
template <int BD = 8>
__device__ __forceinline__ bool prune_flat(const GpImage &I, const GpParams &P, int x, int y, int W, int H, int qidx) {
    if (P.pruneFlat <= 0.f || !I.var16 || W * H >= 256) return false;
    const int cx = x >> 4, cy = y >> 4;
    if ((cx + 1) * 16 > I.MiCols * 4 || (cy + 1) * 16 > I.MiRows * 4) return false;   // incomplete region: keep
    const float st = av1g_ac_q<BD>(qidx) / 8.f;
    return I.var16[cy * I.cols16 + cx] < P.pruneFlat * st * st;
}
