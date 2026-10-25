// lfsearch.cu - GPU deblocking-level + CDEF strength search.  See av1f/lfsearch_gpu.h.
//
// Deblocking: candidate levels are applied with av1core's bit-exact GPU deblocker
// (av1g_gpu_deblock) to scratch copies of the pre-filter frame, SSE vs source per plane over the
// visible picture (k_sse); FROM_Q reproduces libaom s6.
// CDEF: directions once (av1g_gpu_cdef_dir), then k_cdef_sse, one CTA per 64x64: for every pixel of
// a filtered 8x8 the primary contribution per primary strength (variance-adjusted for luma) and
// the secondary contribution per secondary strength are computed once for the two tap sets
// (block direction, used when pri != 0; direction 0, used when pri == 0), so all npri x nsec
// candidates cost ~npri*4 + nsec*16 constrain() calls per pixel instead of 12 per candidate.  The
// per-SB SSE table feeds libaom's joint greedy preset search (search_one_dual + refinement) on the
// GPU (k_cur / k_pairs / k_argmin) and the RD choice of cdef_bits (host, a few numbers).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "av1f/lfsearch_gpu.h"
#include "av1f/tuneiq.h"
#include "av1g/gpu_postfilter.h"
#include "av1g/postfilter.h"

#define CK(x)                                    \
    do {                                         \
        const cudaError_t e_ = (cudaError_t)(x); \
        if (e_ != cudaSuccess) return (int)e_;   \
    } while (0)

namespace {

constexpr int LF_BATCH = 8;       // deblocking candidates evaluated per batch
constexpr int SEL_CHUNK = 256;    // SBs per k_pairs thread chunk

size_t al(size_t x) { return (x + 255) & ~(size_t)255; }
}  // namespace
static bool lf_fused(const av1f_lf_opts *o);
namespace {

struct Geo {
    int MiRows, MiCols, sbRows, sbCols, nsb, mono;
    int lw, lh, cw, ch;  // padded plane sizes
    size_t lbytes, cbytes;
};
Geo geo(const av1g_frame_params *fp) {
    Geo g;
    g.MiCols = 2 * ((fp->width + 7) >> 3);
    g.MiRows = 2 * ((fp->height + 7) >> 3);
    g.sbCols = (g.MiCols + 15) >> 4;
    g.sbRows = (g.MiRows + 15) >> 4;
    g.nsb = g.sbCols * g.sbRows;
    g.mono = fp->monochrome;
    g.lw = g.MiCols * 4; g.lh = g.MiRows * 4;
    g.cw = g.MiCols * 2; g.ch = g.MiRows * 2;
    g.lbytes = (size_t)g.lw * g.lh;
    g.cbytes = (size_t)g.cw * g.ch;
    return g;
}

struct Layout {
    size_t pf, cand, sse, msey, mseuv, tot, cur, lev, res, wmap, wsum, prune, async, total;
};
Layout layout(const av1g_frame_params *fp, const av1f_lf_opts *o) {
    const Geo g = geo(fp);
    Layout L;
    size_t x = 0;
    L.pf = x; x = al(x + av1g_gpu_pf_scratch_size(fp));
    L.cand = x;
    if (o && o->lf_mode == AV1F_LF_SEARCH && (!lf_fused(o) || getenv("AV1F_LF_CHECK"))) x = al(x + LF_BATCH * (g.lbytes + 2 * g.cbytes));
    L.sse = x; x = al(x + 64 * 3 * sizeof(unsigned long long));
    L.msey = x; x = al(x + (size_t)g.nsb * 64 * 4);
    L.mseuv = x; x = al(x + (size_t)g.nsb * 64 * 4);
    L.tot = x; x = al(x + 64 * 64 * sizeof(unsigned long long));
    L.cur = x; x = al(x + (size_t)g.nsb * sizeof(unsigned long long));
    L.lev = x; x = al(x + 4 * 2 * 8 * sizeof(int) + 8 * sizeof(unsigned long long));
    L.res = x; x = al(x + 64);
    L.wmap = x; x = al(x + (size_t)(g.MiRows / 2) * (g.MiCols / 2) * sizeof(float));
    L.wsum = x; x = al(x + 16);
    L.prune = x; x = al(x + (size_t)g.nsb * 16 * 4 * 2 + 64 * 4);
    L.async = x; x = al(x + 256);   // lv[4], nact, bits, tots[8] (async path)
    L.total = x;
    return L;
}

// ---------------------------------------------------------------- deblocking SSE
struct SseJob {
    const uint8_t *a[LF_BATCH * 3];
    int astride[3];
    const uint8_t *src[3];
    int sstride[3];
    int w[3], h[3];
    int ncand;
    const float *wmap;   // per luma 8x8 (NULL = plain SSE)
    int n8c;
};
__global__ void k_sse(SseJob j, unsigned long long *out /* [cand][3] */, int cand0) {
    const int z = blockIdx.z, cand = z / 3, p = z % 3;
    if (cand >= j.ncand || !j.a[z]) return;
    const int w = j.w[p], h = j.h[p];
    unsigned long long s = 0;
    const int sh = p ? 2 : 3;  // plane samples per luma-8x8 weight cell: 8 (luma) / 4 (chroma)
    for (int y = blockIdx.y; y < h; y += gridDim.y)
        for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < w; x += gridDim.x * blockDim.x) {
            const int d = (int)j.a[z][(size_t)y * j.astride[p] + x] - (int)j.src[p][(size_t)y * j.sstride[p] + x];
            if (j.wmap) s += (unsigned long long)(16.f * d * d * j.wmap[(size_t)(y >> sh) * j.n8c + (x >> sh)] + 0.5f);
            else s += (unsigned)(d * d);
        }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffff, s, o);
    __shared__ unsigned long long r[32];
    if ((threadIdx.x & 31) == 0) r[threadIdx.x >> 5] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned long long t = 0;
        for (int i = 0; i < (int)(blockDim.x >> 5); i++) t += r[i];
        atomicAdd(&out[(cand0 + cand) * 3 + p], t);
    }
}

// ---------------------------------------------------------------- SSIMULACRA2 proxy weights
// w[8x8] = 1 / (var_src(8x8 luma) + c), later normalised to a mean of 1 over the picture
__global__ void k_wmap(const uint8_t *src, int stride, int n8c, int n8r, float c, float *w, double *sum) {
    const int bx = blockIdx.x * blockDim.x + threadIdx.x, by = blockIdx.y;
    float v = 0.f;
    if (bx < n8c && by < n8r) {
        int s = 0, q = 0;
        for (int i = 0; i < 8; i++)
            for (int j = 0; j < 8; j++) {
                const int x = src[(size_t)(by * 8 + i) * stride + bx * 8 + j];
                s += x;
                q += x * x;
            }
        const float var = (q - (float)s * s / 64.f) / 64.f;
        v = 1.f / (var + c);
        w[(size_t)by * n8c + bx] = v;
    }
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(sum, (double)v);
}
__global__ void k_wnorm(float *w, size_t n, const double *sum) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < n) w[i] = (float)(w[i] * (double)n / *sum);
}

// ---------------------------------------------------------------- fused deblocking-level search
// One CTA per 64x64 SB: stage the pre-filter luma (96x96 = SB + 16-sample halo) and chroma (2 x
// 48x48 = SB + 8) in shared memory; for every level 0..63: copy to a working buffer, run pass 0
// (vertical edges; all halo rows, because pass 1 reads them) and pass 1 (horizontal edges) with
// av1core's av1g_lf_line, then accumulate SSE vs the source over the SB's visible samples.  The
// dependency reach of one output sample is < 12 luma / < 8 chroma samples per direction, so the
// halo makes the SB result identical to whole-frame filtering (checked: AV1F_LF_CHECK).
constexpr int LT = 96, LH = 16, CT = 48, CH = 8;
// k_lf_tiles2: k_lf_tiles with the SB's edge list built once (positions + filter sizes of the actual
// transform edges, exactly av1g_lf_edge's tests) so that each level only runs av1g_lf_line per
// (edge, line), and one reduction for all planes.  Same results (AV1F_LF_CHECK).
constexpr int LF_E0 = 704, LF_E1 = 576;  // max edges per pass in one SB region (luma + 2 chroma)
__global__ void __launch_bounds__(256) k_lf_tiles2(av1g_pf_frame f, const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, int ss0,
                                                   int ss1, int ss2, const float *wmap, unsigned long long *sse, int grad) {
    __shared__ uint8_t inL[LT * LT], wkL[LT * LT];
    __shared__ uint8_t inC[2][CT * CT], wkC[2][CT * CT];
    __shared__ uint32_t edges[2][LF_E0 > LF_E1 ? LF_E0 : LF_E1];
    __shared__ int nE[2];
    __shared__ unsigned long long red[8][3];
    const int sb = blockIdx.x, sbr = sb / f.sbCols, sbc = sb % f.sbCols;
    const int tid = threadIdx.x;
    const int x0 = sbc * 64, y0 = sbr * 64, cx0 = sbc * 32, cy0 = sbr * 32;
    const int np = f.mono ? 1 : 3;
    if (tid < 2) nE[tid] = 0;
    for (int i = tid; i < LT * LT; i += 256) {
        const int X = x0 - LH + i % LT, Y = y0 - LH + i / LT;
        inL[i] = (X >= 0 && Y >= 0 && X < f.MiCols * 4 && Y < f.MiRows * 4) ? f.plane[0][(size_t)Y * f.stride[0] + X] : 0;
    }
    if (np > 1)
        for (int i = tid; i < 2 * CT * CT; i += 256) {
            const int pl = i / (CT * CT), j = i % (CT * CT);
            const int X = cx0 - CH + j % CT, Y = cy0 - CH + j / CT;
            inC[pl][j] = (X >= 0 && Y >= 0 && X < f.MiCols * 2 && Y < f.MiRows * 2) ? f.plane[1 + pl][(size_t)Y * f.stride[1 + pl] + X] : 0;
        }
    __syncthreads();
    // ---- edge lists: entry = plane (2 bits) | filterSize code (2 bits) | local x (8) | local y (8)
    for (int pass = 0; pass < 2; pass++) {
        // luma ranges
        const int lr0 = pass ? max(0, (y0 - 4) >> 2) : max(0, (y0 - LH + 4) >> 2);
        const int lr1 = pass ? min(f.MiRows, ((y0 + 68) >> 2) + 1) : min(f.MiRows, (y0 + 64 + LH - 4) >> 2);
        const int lc0 = pass ? x0 >> 2 : max(0, (x0 - 4) >> 2);
        const int lc1 = pass ? min(f.MiCols, (x0 + 64) >> 2) : min(f.MiCols, ((x0 + 68) >> 2) + 1);
        const int nl = max(0, (lr1 - lr0) * (lc1 - lc0));
        int nc = 0, cr0 = 0, cc0 = 0, ncc = 1;
        if (np > 1) {
            cr0 = pass ? max(0, (cy0 - 4) >> 2) : max(0, (cy0 - CH + 4) >> 2);
            const int cr1 = pass ? min(f.MiRows >> 1, ((cy0 + 36) >> 2) + 1) : min(f.MiRows >> 1, (cy0 + 32 + CH - 4) >> 2);
            cc0 = pass ? (cx0 >> 2) : max(0, (cx0 - 4) >> 2);
            const int cc1 = pass ? min(f.MiCols >> 1, (cx0 + 32) >> 2) : min(f.MiCols >> 1, ((cx0 + 36) >> 2) + 1);
            ncc = max(1, cc1 - cc0);
            nc = max(0, (cr1 - cr0) * (cc1 - cc0));
        }
        for (int e = tid; e < nl + 2 * nc; e += 256) {
            int plane, row, col;
            if (e < nl) { plane = 0; row = lr0 + e / (lc1 - lc0); col = lc0 + e % (lc1 - lc0); }
            else { const int k = (e - nl) % nc; plane = 1 + (e - nl) / nc; row = (cr0 + k / ncc) * 2; col = (cc0 + k % ncc) * 2; }
            // == av1g_lf_edge's tests (everything but the level)
            const int sub = plane ? 1 : 0, x = col * 4, y = row * 4;
            if (x >= f.W || y >= f.H) continue;
            if (pass == 0 && x == 0) continue;
            if (pass == 1 && y == 0) continue;
            const int r = row | sub, c = col | sub;
            const int xP = x >> sub, yP = y >> sub;
            const int pr = r - ((pass == 1) << sub), pc = c - ((pass == 0) << sub);
            const int st = f.MiCols >> sub;
            const int txSz = f.lfTx[plane][(size_t)(r >> sub) * st + (c >> sub)];
            const int prevTxSz = f.lfTx[plane][(size_t)(pr >> sub) * st + (pc >> sub)];
            const int cur = pass == 0 ? AV1G_T(Tx_Width)[txSz] : AV1G_T(Tx_Height)[txSz];
            const int prev = pass == 0 ? AV1G_T(Tx_Width)[prevTxSz] : AV1G_T(Tx_Height)[prevTxSz];
            if ((pass == 0 ? xP : yP) & (cur - 1)) continue;
            const int base = prev < cur ? prev : cur, cap = plane ? 8 : 16;
            const int fs = base < cap ? base : cap;  // 4, 8 or 16
            const int lx = xP - (plane ? cx0 - CH : x0 - LH), ly = yP - (plane ? cy0 - CH : y0 - LH);
            const int slot = atomicAdd(&nE[pass], 1);
            if (slot < (pass ? LF_E1 : LF_E0))
                edges[pass][slot] = (uint32_t)plane << 30 | (uint32_t)(fs == 4 ? 0 : fs == 8 ? 1 : 2) << 28 | (uint32_t)lx << 8 | (uint32_t)ly;
        }
    }
    __syncthreads();
    const int shift = f.sharpness > 4 ? 2 : f.sharpness > 0 ? 1 : 0;
    for (int lvl = 0; lvl < 64; lvl++) {
        for (int i = tid; i < LT * LT / 16; i += 256) reinterpret_cast<uint4 *>(wkL)[i] = reinterpret_cast<const uint4 *>(inL)[i];
        if (np > 1)
            for (int i = tid; i < 2 * CT * CT / 16; i += 256)
                reinterpret_cast<uint4 *>(&wkC[0][0])[i] = reinterpret_cast<const uint4 *>(&inC[0][0])[i];
        __syncthreads();
        if (lvl) {
            const int limit = f.sharpness > 0 ? av1g_clip3(1, 9 - f.sharpness, lvl >> shift) : ((lvl >> shift) > 1 ? lvl >> shift : 1);
            const int blimit = 2 * (lvl + 2) + limit, thresh = lvl >> 4;
            for (int pass = 0; pass < 2; pass++) {
                const int n = min(nE[pass], pass ? LF_E1 : LF_E0);
                for (int t = tid; t < 4 * n; t += 256) {
                    const uint32_t e = edges[pass][t >> 2];
                    const int plane = e >> 30, fs = 4 << ((e >> 28) & 3), lx = (e >> 8) & 255, ly = e & 255, line = t & 3;
                    uint8_t *buf = plane ? wkC[plane - 1] : wkL;
                    const int T = plane ? CT : LT;
                    uint8_t *q0 = buf + (ly + (pass == 0) * line) * T + lx + (pass == 1) * line;
                    av1g_lf_line(q0, pass == 0 ? 1 : T, plane, fs, limit, blimit, thresh);
                }
                __syncthreads();
            }
        }
        unsigned long long acc[3] = {0, 0, 0};
        for (int pl = 0; pl < np; pl++) {
            const int sub = pl ? 1 : 0, n = pl ? 32 : 64;
            const int W = pl ? (f.W + 1) >> 1 : f.W, H = pl ? (f.H + 1) >> 1 : f.H;
            const int bx = pl ? cx0 : x0, by = pl ? cy0 : y0;
            const uint8_t *src = pl == 0 ? s0 : pl == 1 ? s1 : s2;
            const int sst = pl == 0 ? ss0 : pl == 1 ? ss1 : ss2;
            const uint8_t *wk = pl ? wkC[pl - 1] : wkL;
            const int T = pl ? CT : LT, Hh = pl ? CH : LH;
            for (int i = tid; i < n * n; i += 256) {
                const int X = bx + (i % n), Y = by + (i / n);
                if (X >= W || Y >= H) continue;
                const int d = (int)wk[(Y - by + Hh) * T + (X - bx + Hh)] - (int)src[(size_t)Y * sst + X];
                if (grad) {
                    const int lx = i % n, ly = i / n;
                    if (lx + 1 < n && X + 1 < W) {
                        const int d2 = (int)wk[(Y - by + Hh) * T + (X + 1 - bx + Hh)] - (int)src[(size_t)Y * sst + X + 1];
                        acc[pl] += (unsigned)((d2 - d) * (d2 - d));
                    }
                    if (ly + 1 < n && Y + 1 < H) {
                        const int d2 = (int)wk[(Y + 1 - by + Hh) * T + (X - bx + Hh)] - (int)src[(size_t)(Y + 1) * sst + X];
                        acc[pl] += (unsigned)((d2 - d) * (d2 - d));
                    }
                } else if (wmap) acc[pl] += (unsigned long long)(16.f * d * d * wmap[(size_t)((Y << sub) >> 3) * (f.MiCols >> 1) + ((X << sub) >> 3)] + 0.5f);
                else acc[pl] += (unsigned)(d * d);
            }
        }
        for (int pl = 0; pl < 3; pl++) {
            unsigned long long v = acc[pl];
            for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
            if ((tid & 31) == 0) red[tid >> 5][pl] = v;
        }
        __syncthreads();
        if (tid < np) {
            unsigned long long t = 0;
            for (int w = 0; w < 8; w++) t += red[w][tid];
            atomicAdd(&sse[lvl * 3 + tid], t);
        }
        // the next level's copy only writes wk*; red is rewritten after its own __syncthreads
    }
}

__global__ void k_cdef_idx_norm(int8_t *idx, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) idx[i] = idx[i] >= 0 ? 0 : -1;
}

// ---------------------------------------------------------------- CDEF SSE per SB
struct CdefCand {
    int pri[16];   // signalled primary strengths (0..15)
    int sec[4];    // applied secondary strengths (0, 1, 2, 4)
};

__constant__ int8_t c_dirs[8][2][2] = {{{-1, 1}, {-2, 2}}, {{0, 1}, {-1, 2}}, {{0, 1}, {0, 2}}, {{0, 1}, {1, 2}},
                                       {{1, 1}, {2, 2}},   {{1, 0}, {2, 1}},  {{1, 0}, {2, 0}}, {{1, 0}, {2, -1}}};

// Taps of one direction set: primary at dir, secondary at dir -/+ 2; availability = frame bounds.
struct Taps {
    int p[4];   // primary samples (k = 0, 1) x (sign -1, +1); -1 = unavailable
    int s[8];   // secondary samples (k) x (sign) x (dir - 2, dir + 2)
    int mn, mx;
};
__device__ __forceinline__ int tap(const uint8_t *src, int stride, int x, int y, int dy, int dx, int sub, int MiRows,
                                   int MiCols) {
    const int yy = y + dy, xx = x + dx;
    if (yy < 0 || xx < 0 || ((yy << sub) >> 2) >= MiRows || ((xx << sub) >> 2) >= MiCols) return -1;
    return src[(size_t)yy * stride + xx];
}
__device__ __forceinline__ void gather(Taps &t, const uint8_t *src, int stride, int x, int y, int dir, int sub, int MiRows,
                                       int MiCols, int x0) {
    t.mn = t.mx = x0;
#pragma unroll
    for (int k = 0; k < 2; k++)
#pragma unroll
        for (int sg = 0; sg < 2; sg++) {
            const int sign = sg ? 1 : -1;
            int v = tap(src, stride, x, y, sign * c_dirs[dir][k][0], sign * c_dirs[dir][k][1], sub, MiRows, MiCols);
            t.p[k * 2 + sg] = v;
            if (v >= 0) { t.mn = min(t.mn, v); t.mx = max(t.mx, v); }
#pragma unroll
            for (int o = 0; o < 2; o++) {
                const int dd = (dir + (o ? 2 : -2)) & 7;
                v = tap(src, stride, x, y, sign * c_dirs[dd][k][0], sign * c_dirs[dd][k][1], sub, MiRows, MiCols);
                t.s[(k * 2 + sg) * 2 + o] = v;
                if (v >= 0) { t.mn = min(t.mn, v); t.mx = max(t.mx, v); }
            }
        }
}
__device__ __forceinline__ int sec_sum(const Taps &t, int x0, int sec, int damping) {
    if (!sec) return 0;
    int s = 0;
#pragma unroll
    for (int i = 0; i < 8; i++)
        if (t.s[i] >= 0) s += (i < 4 ? 2 : 1) * av1g_cdef_constrain(t.s[i] - x0, sec, damping);  // Cdef_Sec_Taps[k], k = i >> 2
    return s;
}
__device__ __forceinline__ int pri_sum(const Taps &t, int x0, int adj, int damping) {
    if (!adj) return 0;
    const int w0 = (adj & 1) ? 3 : 4, w1 = (adj & 1) ? 3 : 2;  // Cdef_Pri_Taps[adj & 1]
    int s = 0;
#pragma unroll
    for (int i = 0; i < 4; i++)
        if (t.p[i] >= 0) s += (i < 2 ? w0 : w1) * av1g_cdef_constrain(t.p[i] - x0, adj, damping);
    return s;
}

template <int NPRI, int NSEC, typename ACC>
__device__ __forceinline__ void cdef_pixel_acc(const uint8_t *src, int stride, int x, int y, int sub, int MiRows, int MiCols,
                                               int yDir, int varStr, int var, int damping, int target, const CdefCand &cd,
                                               ACC *acc, float wt) {
    const int x0 = src[(size_t)y * stride + x];
    Taps tA, t0;
    gather(tA, src, stride, x, y, yDir, sub, MiRows, MiCols, x0);
    gather(t0, src, stride, x, y, 0, sub, MiRows, MiCols, x0);
    int SA[NSEC], S0[NSEC];
#pragma unroll
    for (int si = 0; si < NSEC; si++) {
        SA[si] = sec_sum(tA, x0, cd.sec[si], damping);
        S0[si] = sec_sum(t0, x0, cd.sec[si], damping);
    }
#pragma unroll
    for (int pi = 0; pi < NPRI; pi++) {
        const int pri = cd.pri[pi];
        const int adj = sub ? pri : (var ? (pri * (4 + varStr) + 8) >> 4 : 0);
        const int P = pri ? pri_sum(tA, x0, adj, damping) : 0;
        const int mn = pri ? tA.mn : t0.mn, mx = pri ? tA.mx : t0.mx;
#pragma unroll
        for (int si = 0; si < NSEC; si++) {
            const int sum = P + (pri ? SA[si] : S0[si]);
            int v = x0 + ((8 + sum - (sum < 0)) >> 4);
            v = v < mn ? mn : v > mx ? mx : v;
            const int e = v - target;
            if (sizeof(ACC) == 4 && (ACC)0.5f == 0) acc[pi * NSEC + si] += (ACC)(e * e);  // integer SSE (libaom)
            else acc[pi * NSEC + si] += (ACC)(wt * (float)(e * e));
        }
    }
}

template <int NPRI, int NSEC, typename ACC>
__global__ void __launch_bounds__(256) k_cdef_sse(av1g_pf_frame f, const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, int ss0,
                                                  int ss1, int ss2, CdefCand cd, const float *wmap, uint32_t *mse_y,
                                                  uint32_t *mse_uv) {
    constexpr int NC = NPRI * NSEC;
    __shared__ ACC red[8][NC];
    const int sb = blockIdx.x, sbr = sb / f.sbCols, sbc = sb % f.sbCols;
    const bool active = f.cdefIdx[sb] >= 0;
    const int n8c = f.MiCols >> 1;
    for (int pass = 0; pass < (f.mono ? 1 : 2); pass++) {
        ACC acc[NC];
#pragma unroll
        for (int j = 0; j < NC; j++) acc[j] = 0;
        if (active) {
            const int npx = pass == 0 ? 4096 : 2048;
            for (int i = threadIdx.x; i < npx; i += 256) {
                int plane, x, y;
                if (pass == 0) { plane = 0; y = sbr * 64 + (i >> 6); x = sbc * 64 + (i & 63); }
                else { plane = 1 + (i >> 10); const int j = i & 1023; y = sbr * 32 + (j >> 5); x = sbc * 32 + (j & 31); }
                const int sub = plane ? 1 : 0;
                if (((x << sub) >> 2) >= f.MiCols || ((y << sub) >> 2) >= f.MiRows) continue;
                const int ly = y << sub, lx = x << sub;  // luma position of the sample's 8x8
                const int r = (ly >> 3) * 2, c = (lx >> 3) * 2;
                if (av1g_cdef_active_idx(&f, r, c) < 0) continue;
                const int dv = f.cdefDirVar[(size_t)(ly >> 3) * n8c + (lx >> 3)];
                const int yDir = dv & 7, var = dv >> 3;
                int varStr = 0;
                if (var >> 6) { varStr = av1g_floor_log2((uint32_t)(var >> 6)); if (varStr > 12) varStr = 12; }
                const uint8_t *src = f.plane[plane];
                const int st = f.stride[plane];
                const uint8_t *tg = plane == 0 ? s0 : plane == 1 ? s1 : s2;
                const int tst = plane == 0 ? ss0 : plane == 1 ? ss1 : ss2;
                const float wt = wmap ? wmap[(size_t)(ly >> 3) * n8c + (lx >> 3)] : 1.f;
                cdef_pixel_acc<NPRI, NSEC, ACC>(src, st, x, y, sub, f.MiRows, f.MiCols, yDir, varStr, var,
                                                plane ? f.cdefDamping - 1 : f.cdefDamping, tg[(size_t)y * tst + x], cd, acc, wt);
            }
        }
#pragma unroll
        for (int j = 0; j < NC; j++) {
            ACC v = acc[j];
            for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
            if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5][j] = v;
        }
        __syncthreads();
        for (int j = threadIdx.x; j < NC; j += blockDim.x) {
            ACC t = 0;
            for (int w = 0; w < 8; w++) t += red[w][j];
            // weighted: x16 fixed point (mean weight 1), saturating; plain SSE is exact
            const uint32_t u = (sizeof(ACC) == 4 && (ACC)0.5f == 0) ? (uint32_t)t
                                                                    : (uint32_t)fminf(16.f * (float)t + 0.5f, 4.0e9f);
            (pass == 0 ? mse_y : mse_uv)[(size_t)sb * NC + j] = u;
        }
        __syncthreads();
    }
}

// Shared-memory version: the SB's deblocked samples + 2-sample halo are staged once (int16, -1 =
// outside the frame = CdefAvailable 0), per-8x8 direction/variance/activity/weight too; every tap is
// then a shared-memory read.  Same arithmetic as k_cdef_sse (AV1F_CDEF_CHECK=1 compares both).
__device__ __forceinline__ void gather_t(Taps &t, const int16_t *tile, int tw, int lx, int ly, int dir, int x0) {
    t.mn = t.mx = x0;
#pragma unroll
    for (int k = 0; k < 2; k++)
#pragma unroll
        for (int sg = 0; sg < 2; sg++) {
            const int sign = sg ? 1 : -1;
            int v = tile[(ly + sign * c_dirs[dir][k][0]) * tw + lx + sign * c_dirs[dir][k][1]];
            t.p[k * 2 + sg] = v;
            if (v >= 0) { t.mn = min(t.mn, v); t.mx = max(t.mx, v); }
#pragma unroll
            for (int o = 0; o < 2; o++) {
                const int dd = (dir + (o ? 2 : -2)) & 7;
                v = tile[(ly + sign * c_dirs[dd][k][0]) * tw + lx + sign * c_dirs[dd][k][1]];
                t.s[(k * 2 + sg) * 2 + o] = v;
                if (v >= 0) { t.mn = min(t.mn, v); t.mx = max(t.mx, v); }
            }
        }
}

template <int NPRI, int NSEC, typename ACC>
__device__ __forceinline__ void cdef_pixel_acc_t(const int16_t *tile, int tw, int lx, int ly, int sub, int yDir, int varStr, int var,
                                                 int damping, int target, const CdefCand &cd, ACC *acc, float wt) {
    const int x0 = tile[ly * tw + lx];
    Taps tA, t0;
    gather_t(tA, tile, tw, lx, ly, yDir, x0);
    gather_t(t0, tile, tw, lx, ly, 0, x0);
    int SA[NSEC], S0[NSEC];
#pragma unroll
    for (int si = 0; si < NSEC; si++) {
        SA[si] = sec_sum(tA, x0, cd.sec[si], damping);
        S0[si] = sec_sum(t0, x0, cd.sec[si], damping);
    }
#pragma unroll
    for (int pi = 0; pi < NPRI; pi++) {
        const int pri = cd.pri[pi];
        const int adj = sub ? pri : (var ? (pri * (4 + varStr) + 8) >> 4 : 0);
        const int P = pri ? pri_sum(tA, x0, adj, damping) : 0;
        const int mn = pri ? tA.mn : t0.mn, mx = pri ? tA.mx : t0.mx;
#pragma unroll
        for (int si = 0; si < NSEC; si++) {
            const int sum = P + (pri ? SA[si] : S0[si]);
            int v = x0 + ((8 + sum - (sum < 0)) >> 4);
            v = v < mn ? mn : v > mx ? mx : v;
            const int e = v - target;
            if (sizeof(ACC) == 4 && (ACC)0.5f == 0) acc[pi * NSEC + si] += (ACC)(e * e);
            else acc[pi * NSEC + si] += (ACC)(wt * (float)(e * e));
        }
    }
}

template <int NPRI, int NSEC, typename ACC>
__global__ void __launch_bounds__(256) k_cdef_sse_t(av1g_pf_frame f, const uint8_t *s0, const uint8_t *s1, const uint8_t *s2, int ss0,
                                                    int ss1, int ss2, CdefCand cd, const float *wmap, uint32_t *mse_y,
                                                    uint32_t *mse_uv) {
    constexpr int NC = NPRI * NSEC, TL = 68, TC = 36;
    __shared__ int16_t tl[TL * TL];
    __shared__ int16_t tc[2][TC * TC];
    __shared__ int32_t dvs[64];
    __shared__ float wts[64];
    __shared__ uint8_t act[64];
    __shared__ ACC red[8][NC];
    const int sb = blockIdx.x, sbr = sb / f.sbCols, sbc = sb % f.sbCols;
    const int tid = threadIdx.x;
    if (f.cdefIdx[sb] < 0) {
        for (int j = tid; j < NC; j += blockDim.x) {
            mse_y[(size_t)sb * NC + j] = 0;
            if (!f.mono) mse_uv[(size_t)sb * NC + j] = 0;
        }
        return;
    }
    const int n8c = f.MiCols >> 1;
    {   // stage luma (+ chroma) with a 2-sample halo
        const int x0 = sbc * 64 - 2, y0 = sbr * 64 - 2;
        for (int i = tid; i < TL * TL; i += blockDim.x) {
            const int X = x0 + i % TL, Y = y0 + i / TL;
            const bool av = X >= 0 && Y >= 0 && (X >> 2) < f.MiCols && (Y >> 2) < f.MiRows;
            tl[i] = av ? f.plane[0][(size_t)Y * f.stride[0] + X] : -1;
        }
        if (!f.mono) {
            const int cx0 = sbc * 32 - 2, cy0 = sbr * 32 - 2;
            for (int i = tid; i < 2 * TC * TC; i += blockDim.x) {
                const int pl = i / (TC * TC), j = i % (TC * TC);
                const int X = cx0 + j % TC, Y = cy0 + j / TC;
                const bool av = X >= 0 && Y >= 0 && ((X << 1) >> 2) < f.MiCols && ((Y << 1) >> 2) < f.MiRows;
                tc[pl][j] = av ? f.plane[1 + pl][(size_t)Y * f.stride[1 + pl] + X] : -1;
            }
        }
        if (tid < 64) {
            const int by = sbr * 8 + (tid >> 3), bx = sbc * 8 + (tid & 7);
            const bool in = by * 2 < f.MiRows && bx * 2 < f.MiCols;
            act[tid] = in && av1g_cdef_active_idx(&f, by * 2, bx * 2) >= 0;
            dvs[tid] = in ? f.cdefDirVar[(size_t)by * n8c + bx] : 0;
            wts[tid] = (in && wmap) ? wmap[(size_t)by * n8c + bx] : 1.f;
        }
    }
    __syncthreads();
    for (int pass = 0; pass < (f.mono ? 1 : 2); pass++) {
        ACC acc[NC];
#pragma unroll
        for (int j = 0; j < NC; j++) acc[j] = 0;
        const int npx = pass == 0 ? 4096 : 2048;
        for (int i = tid; i < npx; i += 256) {
            int plane, lx, ly;
            if (pass == 0) { plane = 0; ly = i >> 6; lx = i & 63; }
            else { plane = 1 + (i >> 10); const int j = i & 1023; ly = j >> 5; lx = j & 31; }
            const int sub = plane ? 1 : 0;
            const int b = ((ly << sub) >> 3) * 8 + ((lx << sub) >> 3);
            if (!act[b]) continue;
            const int x = (sub ? sbc * 32 : sbc * 64) + lx, y = (sub ? sbr * 32 : sbr * 64) + ly;
            if (((x << sub) >> 2) >= f.MiCols || ((y << sub) >> 2) >= f.MiRows) continue;
            const int dv = dvs[b], yDir = dv & 7, var = dv >> 3;
            int varStr = 0;
            if (var >> 6) { varStr = av1g_floor_log2((uint32_t)(var >> 6)); if (varStr > 12) varStr = 12; }
            const uint8_t *tg = plane == 0 ? s0 : plane == 1 ? s1 : s2;
            const int tst = plane == 0 ? ss0 : plane == 1 ? ss1 : ss2;
            const int16_t *tile = plane == 0 ? tl : tc[plane - 1];
            cdef_pixel_acc_t<NPRI, NSEC, ACC>(tile, plane ? TC : TL, lx + 2, ly + 2, sub, yDir, varStr, var,
                                              plane ? f.cdefDamping - 1 : f.cdefDamping, tg[(size_t)y * tst + x], cd, acc, wts[b]);
        }
#pragma unroll
        for (int j = 0; j < NC; j++) {
            ACC v = acc[j];
            for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
            if ((tid & 31) == 0) red[tid >> 5][j] = v;
        }
        __syncthreads();
        for (int j = tid; j < NC; j += blockDim.x) {
            ACC t = 0;
            for (int w = 0; w < 8; w++) t += red[w][j];
            const uint32_t u = (sizeof(ACC) == 4 && (ACC)0.5f == 0) ? (uint32_t)t
                                                                    : (uint32_t)fminf(16.f * (float)t + 0.5f, 4.0e9f);
            (pass == 0 ? mse_y : mse_uv)[(size_t)sb * NC + j] = u;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------- preset selection (libaom joint search)
struct Sel {
    int *lev;                 // [2][8]: lev0 (luma strength index), lev1 (chroma)
    unsigned long long *best; // [2]: tot of the last search_one, argmin index
};
__global__ void k_cur(const uint32_t *my, const uint32_t *muv, int nc, int nsb, const int *lev, int nb, int mono,
                      unsigned long long *cur) {
    const int sb = blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= nsb) return;
    unsigned long long b = ~0ull;
    for (int g = 0; g < nb; g++) {
        unsigned long long c = my[(size_t)sb * nc + lev[g]];
        if (!mono) c += muv[(size_t)sb * nc + lev[8 + g]];
        b = c < b ? c : b;
    }
    cur[sb] = b;
}
__global__ void k_pairs(const uint32_t *my, const uint32_t *muv, int nc, int nsb, const unsigned long long *cur, int mono,
                        unsigned long long *tot) {
    const int pairs = mono ? nc : nc * nc;
    const int p = blockIdx.x * blockDim.x + threadIdx.x;
    if (p >= pairs) return;
    const int j = mono ? p : p / nc, k = mono ? 0 : p % nc;
    const int s0 = blockIdx.y * SEL_CHUNK, s1 = min(nsb, s0 + SEL_CHUNK);
    unsigned long long t = 0;
    for (int sb = s0; sb < s1; sb++) {
        unsigned long long c = my[(size_t)sb * nc + j];
        if (!mono) c += muv[(size_t)sb * nc + k];
        const unsigned long long b = cur[sb];
        t += c < b ? c : b;
    }
    atomicAdd(&tot[p], t);
}
// argmin (first index on ties = libaom's strict '<' in row-major (j, k) order); writes lev[pos]
__global__ void k_argmin(const unsigned long long *tot, int nc, int mono, int *lev, int pos, unsigned long long *best) {
    const int pairs = mono ? nc : nc * nc;
    unsigned long long bv = ~0ull;
    int bi = 0x7fffffff;
    for (int p = threadIdx.x; p < pairs; p += blockDim.x)
        if (tot[p] < bv || (tot[p] == bv && p < bi)) { bv = tot[p]; bi = p; }
    __shared__ unsigned long long sv[256];
    __shared__ int si[256];
    sv[threadIdx.x] = bv;
    si[threadIdx.x] = bi;
    __syncthreads();
    for (int o = blockDim.x / 2; o; o >>= 1) {
        if (threadIdx.x < o) {
            const unsigned long long v2 = sv[threadIdx.x + o];
            const int i2 = si[threadIdx.x + o];
            if (v2 < sv[threadIdx.x] || (v2 == sv[threadIdx.x] && i2 < si[threadIdx.x])) { sv[threadIdx.x] = v2; si[threadIdx.x] = i2; }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        lev[pos] = mono ? si[0] : si[0] / nc;
        lev[8 + pos] = mono ? 0 : si[0] % nc;
        best[0] = sv[0];
    }
}
__global__ void k_shift(int *lev, int nb) {  // drop the oldest preset (refinement)
    if (threadIdx.x == 0)
        for (int j = 0; j < nb - 1; j++) { lev[j] = lev[j + 1]; lev[8 + j] = lev[8 + j + 1]; }
}
__global__ void k_assign(const uint32_t *my, const uint32_t *muv, int nc, int nsb, const int *lev, int nb, int mono,
                         const int8_t *active, int8_t *idx, unsigned long long *tot) {
    const int sb = blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= nsb) return;
    if (active[sb] < 0) { idx[sb] = -1; return; }
    unsigned long long b = ~0ull;
    int bg = 0;
    for (int g = 0; g < nb; g++) {
        unsigned long long c = my[(size_t)sb * nc + lev[g]];
        if (!mono) c += muv[(size_t)sb * nc + lev[8 + g]];
        if (c < b) { b = c; bg = g; }
    }
    idx[sb] = (int8_t)bg;
    atomicAdd(tot, b);
}

// Whole joint preset search (all cdef_bits levels, greedy + refinement) in ONE 1024-thread CTA, for
// small candidate sets (pairs <= 256, e.g. libaom LVL4: 16 pairs).  Same arithmetic and tie rules as
// k_cur / k_pairs / k_argmin; avoids ~300 tiny launches.
__global__ void __launch_bounds__(1024) k_select_small(const uint32_t *my, const uint32_t *muv, int nc, int nsb, int mono,
                                                      int maxbits, int refine, int fast_mono, int *lev_out,
                                                      unsigned long long *tot_out, unsigned long long *cur) {
    __shared__ unsigned long long tot[256];
    __shared__ int lev[16];
    const int pairs = mono ? nc : nc * nc;
    const int tid = threadIdx.x, nslice = 1024 / pairs;
    const int p = tid % pairs, slice = tid / pairs;
    const int pj = mono ? p : p / nc, pk = mono ? 0 : p % nc;
    for (int i = 0; i <= maxbits; i++) {
        const int nb = 1 << i;
        const int steps = nb + (fast_mono ? 0 : refine * nb);
        for (int st = 0; st < steps; st++) {
            const int k = st < nb ? st : nb - 1;  // number of presets kept while searching the next one
            if (st >= nb && tid == 0)
                for (int j = 0; j < nb - 1; j++) { lev[j] = lev[j + 1]; lev[8 + j] = lev[8 + j + 1]; }
            if (tid < pairs) tot[tid] = 0;
            __syncthreads();
            for (int sb = tid; sb < nsb; sb += 1024) {
                unsigned long long b = ~0ull;
                for (int g = 0; g < k; g++) {
                    unsigned long long c = my[(size_t)sb * nc + lev[g]];
                    if (!mono) c += muv[(size_t)sb * nc + lev[8 + g]];
                    b = c < b ? c : b;
                }
                cur[sb] = b;
            }
            __syncthreads();
            if (slice < nslice) {
                unsigned long long t = 0;
                for (int sb = slice; sb < nsb; sb += nslice) {
                    unsigned long long c = my[(size_t)sb * nc + pj];
                    if (!mono) c += muv[(size_t)sb * nc + pk];
                    const unsigned long long b = cur[sb];
                    t += c < b ? c : b;
                }
                atomicAdd(&tot[p], t);
            }
            __syncthreads();
            if (tid == 0) {
                int bi = 0;
                for (int q = 1; q < pairs; q++) if (tot[q] < tot[bi]) bi = q;
                lev[k] = mono ? bi : bi / nc;
                lev[8 + k] = mono ? 0 : bi % nc;
                tot_out[i] = tot[bi];
            }
            __syncthreads();
        }
        if (tid < 16) lev_out[i * 16 + tid] = lev[tid];
        __syncthreads();
    }
}

// Candidate pruning for the exhaustive search: per plane, greedily pick K strengths that best cover
// the SBs (libaom's single-plane search_one criterion), then gather the K columns of each table.
__global__ void __launch_bounds__(1024) k_prune(const uint32_t *m, int nc, int nsb, int K, unsigned long long *cur, int *sel) {
    __shared__ unsigned long long tot[64];
    const int tid = threadIdx.x, j = tid & 63, slice = tid >> 6, nslice = 1024 / 64;
    for (int sb = tid; sb < nsb; sb += 1024) cur[sb] = ~0ull;
    __syncthreads();
    for (int k = 0; k < K; k++) {
        if (tid < 64) tot[tid] = 0;
        __syncthreads();
        if (j < nc) {
            unsigned long long t = 0;
            for (int sb = slice; sb < nsb; sb += nslice) {
                const unsigned long long c = m[(size_t)sb * nc + j], b = cur[sb];
                t += c < b ? c : b;
            }
            atomicAdd(&tot[j], t);
        }
        __syncthreads();
        if (tid == 0) {
            int bi = -1;
            for (int q = 0; q < nc; q++) {
                bool used = false;
                for (int u = 0; u < k; u++) used |= sel[u] == q;
                if (!used && (bi < 0 || tot[q] < tot[bi])) bi = q;
            }
            sel[k] = bi;
        }
        __syncthreads();
        const int b = sel[k];
        for (int sb = tid; sb < nsb; sb += 1024) {
            const unsigned long long c = m[(size_t)sb * nc + b];
            if (c < cur[sb]) cur[sb] = c;
        }
        __syncthreads();
    }
}
__global__ void k_gather_cols(const uint32_t *m, int nc, int nsb, const int *sel, int K, uint32_t *out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nsb * K) return;
    out[i] = m[(size_t)(i / K) * nc + sel[i % K]];
}

// ---------------------------------------------------------------- async path helpers (no host sync)
// deblock with levels read from device memory (wraps av1core's bit-exact av1g_lf_edge)
__global__ void k_deblock_lv(av1g_pf_frame f, int pass, const int *lv) {
    const int plane = blockIdx.z, sub = plane ? 1 : 0;
    const int col = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= (f.MiCols >> sub) || row >= (f.MiRows >> sub)) return;
    if (!lv[0] && !lv[1]) return;
    av1g_pf_frame fl = f;
    for (int i = 0; i < 4; i++) fl.lfLevel[i] = lv[i];
    av1g_lf_edge(&fl, plane, pass, row << sub, col << sub);
}
// argmin over the 64 deblocking levels per plane (lowest level on ties); chroma 0 if luma is 0
__global__ void k_lf_pick(const unsigned long long *sse, int nplanes, int *lv) {
    if (threadIdx.x) return;
    int b[3] = {0, 0, 0};
    for (int p = 0; p < nplanes; p++)
        for (int l = 1; l < 64; l++) if (sse[l * 3 + p] < sse[b[p] * 3 + p]) b[p] = l;
    lv[0] = lv[1] = b[0];
    lv[2] = b[0] ? b[1] : 0;
    lv[3] = b[0] ? b[2] : 0;
}
__global__ void k_count_active(const int8_t *idx, int n, int *cnt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int v = i < n && idx[i] >= 0;
    const int w = __syncthreads_count(v);
    if (threadIdx.x == 0 && w) atomicAdd(cnt, w);
}
struct RdArgs {
    int lastBits, mono, rdmult;
    double rdscale;   // rdmult multiplier (1 = libaom integer RDCOST)
    double dscale;    // distortion scale (2048, or 2048/16 for weighted tables)
};
__global__ void k_rd_pick(const unsigned long long *tots, const int *nact, RdArgs a, int *bits) {
    if (threadIdx.x) return;
    int bb = 0;
    double best = 0;
    for (int i = 0; i <= a.lastBits; i++) {
        const long long nbits = (long long)(*nact) * i + (1 << i) * 6 * (a.mono ? 1 : 2);
        double rd;
        if (a.rdscale == 1.0 && a.dscale == 2048.0) rd = (double)((nbits * 512 * (long long)a.rdmult + 256) >> 9) + (double)tots[i] * 2048.0;
        else rd = (double)nbits * a.rdmult * a.rdscale + (double)tots[i] * a.dscale;
        if (i == 0 || rd < best) { best = rd; bb = i; }
    }
    *bits = bb;
}
__global__ void k_assign_dev(const uint32_t *my, const uint32_t *muv, int nc, int nsb, const int *lev, const int *bits, int mono,
                             const int8_t *active, int8_t *idx) {
    const int sb = blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= nsb) return;
    if (active[sb] < 0) { idx[sb] = -1; return; }
    const int *l = lev + *bits * 16, nb = 1 << *bits;
    unsigned long long b = ~0ull;
    int bg = 0;
    for (int g = 0; g < nb; g++) {
        unsigned long long c = my[(size_t)sb * nc + l[g]];
        if (!mono) c += muv[(size_t)sb * nc + l[8 + g]];
        if (c < b) { b = c; bg = g; }
    }
    idx[sb] = (int8_t)bg;
}
struct FinArgs {
    int lfFixed[4], useLvDev, sharp, cdefEnable, cdefOff, damping, nsec, halve, mono, pruned;
    CdefCand cd;
};
__global__ void k_finalize(FinArgs a, const int *lvDev, const int *bits, const int *lev, const int *sel, av1f_lf_result *r) {
    if (threadIdx.x) return;
    for (int i = 0; i < 4; i++) r->lf_level[i] = a.useLvDev ? lvDev[i] : a.lfFixed[i];
    r->sharpness = a.sharp;
    r->cdef_enable = a.cdefEnable;
    r->cdef_damping = a.damping;
    for (int j = 0; j < 8; j++) r->y_pri[j] = r->y_sec[j] = r->uv_pri[j] = r->uv_sec[j] = 0;
    if (a.cdefOff) { r->cdef_bits = 0; return; }
    const int b = *bits, nb = 1 << b;
    r->cdef_bits = b;
    for (int j = 0; j < nb; j++) {
        int jy = lev[b * 16 + j], juv = lev[b * 16 + 8 + j];
        if (a.pruned) { jy = sel[jy]; juv = sel[16 + juv]; }
        int yp = a.cd.pri[jy / a.nsec], ys = a.cd.sec[jy % a.nsec], up = a.cd.pri[juv / a.nsec], us = a.cd.sec[juv % a.nsec];
        if (a.mono) up = us = 0;
        if (a.halve) {
            yp >>= 1; up >>= 1;
            ys = ((ys == 4 ? 3 : ys) >> 1); ys = ys == 3 ? 4 : ys;
            us = ((us == 4 ? 3 : us) >> 1); us = us == 3 ? 4 : us;
        }
        r->y_pri[j] = yp; r->y_sec[j] = ys; r->uv_pri[j] = up; r->uv_sec[j] = us;
    }
}

struct Timer {
    cudaEvent_t e[2];
    cudaStream_t s;
    explicit Timer(cudaStream_t st) : s(st) { cudaEventCreate(&e[0]); cudaEventCreate(&e[1]); cudaEventRecord(e[0], s); }
    float stop() {
        cudaEventRecord(e[1], s);
        cudaEventSynchronize(e[1]);
        float ms = 0;
        cudaEventElapsedTime(&ms, e[0], e[1]);
        cudaEventDestroy(e[0]);
        cudaEventDestroy(e[1]);
        return ms;
    }
};

}  // namespace


// Cross-check of k_lf_tiles: the straightforward evaluation (64 x frame copy + av1core deblock + SSE)
// into a second table, compared with `sse` (AV1F_LF_CHECK=1; needs the candidate scratch).
// The deblocking-level search by frame copies (default): 64 x (copy + av1core deblock) in batches of
// LF_BATCH + SSE.  compare = true: evaluate into a temporary table and compare with `out` (written
// by the fused kernel) - AV1F_LF_CHECK; compare = false: write the table to `out`.
static bool lf_fused(const av1f_lf_opts *o) { return getenv("AV1F_LF_COPIES") == nullptr || o->dist == AV1F_DIST_GRAD; }
static int lf_search_copies(const av1g_frame_params *fp, const Geo &g, const av1g_pf_frame &f, uint8_t *const d_recon[3],
                            const int recon_stride[3], const uint8_t *const d_src[3], const int src_stride[3], const float *wmap,
                            uint8_t *cand, unsigned long long *out, cudaStream_t s, bool compare) {
    const int nplanes = g.mono ? 1 : 3;
    unsigned long long *sse = out;
    if (compare) CK(cudaMalloc(&sse, 64 * 3 * sizeof(unsigned long long)));
    CK(cudaMemsetAsync(sse, 0, 64 * 3 * sizeof(unsigned long long), s));
    const size_t cs = g.lbytes + 2 * g.cbytes;
    SseJob job;
    memset(&job, 0, sizeof job);
    job.wmap = wmap;
    job.n8c = g.MiCols / 2;
    for (int p = 0; p < 3; p++) {
        job.astride[p] = p ? g.cw : g.lw;
        job.src[p] = d_src[p];
        job.sstride[p] = src_stride[p];
        job.w[p] = p ? (fp->width + 1) >> 1 : fp->width;
        job.h[p] = p ? (fp->height + 1) >> 1 : fp->height;
    }
    for (int l0 = 0; l0 < 64; l0 += LF_BATCH) {
        const int nb = l0 + LF_BATCH <= 64 ? LF_BATCH : 64 - l0;
        for (int k = 0; k < nb; k++) {
            uint8_t *c = cand + k * cs;
            av1g_pf_frame fk = f;
            for (int p = 0; p < nplanes; p++) {
                uint8_t *dst = p == 0 ? c : c + g.lbytes + (p - 1) * g.cbytes;
                const int w = p ? g.cw : g.lw, h = p ? g.ch : g.lh;
                CK(cudaMemcpy2DAsync(dst, w, d_recon[p], recon_stride[p], w, h, cudaMemcpyDeviceToDevice, s));
                fk.plane[p] = dst;
                fk.stride[p] = w;
                job.a[k * 3 + p] = dst;
            }
            for (int p = nplanes; p < 3; p++) job.a[k * 3 + p] = nullptr;
            fk.lfLevel[0] = fk.lfLevel[1] = fk.lfLevel[2] = fk.lfLevel[3] = l0 + k;
            CK(av1g_gpu_deblock(&fk, s));
        }
        job.ncand = nb;
        k_sse<<<dim3(4, 64, nb * 3), 256, 0, s>>>(job, sse, l0);
    }
    if (!compare) return (int)cudaGetLastError();
    unsigned long long a[192], b[192];
    CK(cudaMemcpyAsync(a, sse, sizeof a, cudaMemcpyDeviceToHost, s));
    CK(cudaMemcpyAsync(b, out, sizeof b, cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    int bad = 0;
    for (int i = 0; i < 64 * nplanes; i++) bad += a[(i / nplanes) * 3 + i % nplanes] != b[(i / nplanes) * 3 + i % nplanes];
    fprintf(stderr, "[av1f] deblock search: fused tile kernel vs frame copies: %d of %d SSE entries differ\n", bad, 64 * nplanes);
    cudaFree(sse);
    return 0;
}

extern "C" size_t av1f_lf_scratch_size(const av1g_frame_params *fp, const av1f_lf_opts *o) { return layout(fp, o).total; }


extern "C" int av1f_lf_search(const av1g_frame_params *fp, const av1g_block *d_blocks, uint32_t n_blocks, uint8_t *const d_recon[3],
                              const int recon_stride[3], const uint8_t *const d_src[3], const int src_stride[3],
                              const av1f_lf_opts *o, av1f_lf_result *res, int8_t *d_sb_cdef_idx, void *d_scratch,
                              cudaStream_t s) {
    const Geo g = geo(fp);
    const Layout L = layout(fp, o);
    uint8_t *base = (uint8_t *)d_scratch;
    memset(res, 0, sizeof *res);
    const int bq = fp->base_q_idx;
    av1f_tuneiq_frame tf;
    av1f_tuneiq_frame_init(bq, 1, 1, g.mono, &tf);
    const int sharp = o->sharpness >= 0 ? o->sharpness : tf.lf_sharpness;
    const int damping = o->cdef_damping >= 3 ? o->cdef_damping : tf.cdef_damping;
    const int nplanes = g.mono ? 1 : 3;

    // ---- maps (skip / tx sizes / CDEF activity) via av1core
    av1g_frame_params fpc = *fp;
    fpc.enable_cdef = 1;
    fpc.cdef_bits = 0;
    fpc.cdef_damping_minus_3 = (uint8_t)(damping - 3);
    av1g_pf_frame f;
    CK(av1g_gpu_pf_prepare(&fpc, d_blocks, n_blocks, base + L.pf, &f, s));
    k_cdef_idx_norm<<<(g.nsb + 255) / 256, 256, 0, s>>>(f.cdefIdx, g.nsb);
    f.sharpness = sharp;
    float *wmap = nullptr;
    if (o->dist == AV1F_DIST_VARW) {
        wmap = (float *)(base + L.wmap);
        double *wsum = (double *)(base + L.wsum);
        const int n8c = g.MiCols / 2, n8r = g.MiRows / 2;
        CK(cudaMemsetAsync(wsum, 0, sizeof(double), s));
        k_wmap<<<dim3((n8c + 127) / 128, n8r), 128, 0, s>>>(d_src[0], src_stride[0], n8c, n8r, o->dist_c > 0 ? o->dist_c : 58.5f,
                                                           wmap, wsum);
        k_wnorm<<<(unsigned)(((size_t)n8c * n8r + 255) / 256), 256, 0, s>>>(wmap, (size_t)n8c * n8r, wsum);
    }

    // ---- deblocking
    Timer tlf(s);
    int lv[4] = {0, 0, 0, 0};
    if (o->lf_mode == AV1F_LF_FROM_Q) {
        lv[0] = lv[1] = lv[2] = lv[3] = tf.lf_level;
    } else if (o->lf_mode == AV1F_LF_SEARCH) {
        unsigned long long *sse = (unsigned long long *)(base + L.sse);
        CK(cudaMemsetAsync(sse, 0, 64 * 3 * sizeof(unsigned long long), s));
        av1g_pf_frame ft = f;
        for (int p = 0; p < nplanes; p++) { ft.plane[p] = d_recon[p]; ft.stride[p] = recon_stride[p]; }
        if (lf_fused(o)) {  // default: one fused kernel, no frame copies (AV1F_LF_COPIES=1: copy-based path)
            k_lf_tiles2<<<g.nsb, 256, 0, s>>>(ft, d_src[0], d_src[1], d_src[2], src_stride[0], src_stride[1], src_stride[2], wmap, sse,
                                              o->dist == AV1F_DIST_GRAD);
            CK(cudaGetLastError());
            if (getenv("AV1F_LF_CHECK") && o->dist != AV1F_DIST_GRAD)
                CK(lf_search_copies(fp, g, f, d_recon, recon_stride, d_src, src_stride, wmap, base + L.cand, sse, s, true));
        } else {
            CK(lf_search_copies(fp, g, f, d_recon, recon_stride, d_src, src_stride, wmap, base + L.cand, sse, s, false));
        }
        unsigned long long hs[64 * 3];
        CK(cudaMemcpyAsync(hs, sse, sizeof hs, cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        for (int p = 0; p < nplanes; p++) {
            int b = 0;
            for (int l = 1; l < 64; l++) if (hs[l * 3 + p] < hs[b * 3 + p]) b = l;
            if (p == 0) lv[0] = lv[1] = b;
            else lv[p + 1] = b;
            res->lf_sse[p] = hs[b * 3 + p];
        }
        if (!lv[0]) lv[2] = lv[3] = 0;  // chroma levels are only coded with a nonzero luma level
    }
    for (int i = 0; i < 4; i++) { f.lfLevel[i] = lv[i]; res->lf_level[i] = lv[i]; }
    res->sharpness = sharp;  // coded even when the levels are 0 (libaom writes its value)
    f.sharpness = res->sharpness;
    for (int p = 0; p < nplanes; p++) { f.plane[p] = d_recon[p]; f.stride[p] = recon_stride[p]; }
    CK(av1g_gpu_deblock(&f, s));
    res->ms_lf = tlf.stop();

    // ---- CDEF
    res->cdef_enable = o->cdef_mode != AV1F_CDEF_OFF;
    res->cdef_damping = damping;
    if (o->cdef_mode == AV1F_CDEF_OFF || (o->cdef_mode == AV1F_CDEF_AOM && tf.cdef_mode == 0)) {
        // libaom CDEF_ADAPTIVE at qindex <= 32: one all-zero preset (CDEF signalled but inactive)
        res->cdef_bits = 0;
        if (d_sb_cdef_idx) {
            k_cdef_idx_norm<<<(g.nsb + 255) / 256, 256, 0, s>>>(f.cdefIdx, g.nsb);
            CK(cudaMemcpyAsync(d_sb_cdef_idx, f.cdefIdx, g.nsb, cudaMemcpyDeviceToDevice, s));
        }
        CK(cudaStreamSynchronize(s));
        return (int)cudaGetLastError();
    }
    Timer tdir(s);
    f.enableCdef = 1;
    f.cdefDamping = damping;
    CK(av1g_gpu_cdef_dir(&f, s));
    res->ms_cdef_dir = tdir.stop();

    double rdOut = 0;
    auto run_cdef = [&](int damp, bool final) -> int {
    f.cdefDamping = damp;
    Timer tsse(s);
    CdefCand cd;
    memset(&cd, 0, sizeof cd);
    int npri, nsec;
    if (o->cdef_mode == AV1F_CDEF_AOM) {  // CDEF_FAST_SEARCH_LVL4: pri {0, 11} x sec {0, 2}
        npri = 2; nsec = 2;
        cd.pri[0] = 0; cd.pri[1] = 11;
        cd.sec[0] = 0; cd.sec[1] = 2;
    } else {
        npri = 16; nsec = 4;
        for (int i = 0; i < 16; i++) cd.pri[i] = i;
        cd.sec[0] = 0; cd.sec[1] = 1; cd.sec[2] = 2; cd.sec[3] = 4;
    }
    const int nc = npri * nsec;
    uint32_t *my = (uint32_t *)(base + L.msey), *muv = (uint32_t *)(base + L.mseuv);
    if (g.mono) CK(cudaMemsetAsync(muv, 0, (size_t)g.nsb * nc * 4, s));
#define AV1F_LAUNCH_SSE(KERNEL, MY, MUV)                                                                                     \
    do {                                                                                                                     \
        if (nc == 4 && !wmap)                                                                                                \
            KERNEL<2, 2, uint32_t><<<g.nsb, 256, 0, s>>>(f, d_src[0], d_src[1], d_src[2], src_stride[0], src_stride[1],       \
                                                         src_stride[2], cd, nullptr, MY, MUV);                                \
        else if (nc == 4)                                                                                                    \
            KERNEL<2, 2, float><<<g.nsb, 256, 0, s>>>(f, d_src[0], d_src[1], d_src[2], src_stride[0], src_stride[1],          \
                                                      src_stride[2], cd, wmap, MY, MUV);                                      \
        else if (!wmap)                                                                                                      \
            KERNEL<16, 4, uint32_t><<<g.nsb, 256, 0, s>>>(f, d_src[0], d_src[1], d_src[2], src_stride[0], src_stride[1],      \
                                                          src_stride[2], cd, nullptr, MY, MUV);                               \
        else                                                                                                                 \
            KERNEL<16, 4, float><<<g.nsb, 256, 0, s>>>(f, d_src[0], d_src[1], d_src[2], src_stride[0], src_stride[1],         \
                                                       src_stride[2], cd, wmap, MY, MUV);                                     \
    } while (0)
    AV1F_LAUNCH_SSE(k_cdef_sse_t, my, muv);
    if (getenv("AV1F_CDEF_CHECK")) {  // debug: the global-memory kernel must give identical tables
        uint32_t *cy, *cuv;
        CK(cudaMalloc(&cy, (size_t)g.nsb * nc * 4));
        CK(cudaMalloc(&cuv, (size_t)g.nsb * nc * 4));
        CK(cudaMemsetAsync(cuv, 0, (size_t)g.nsb * nc * 4, s));
        AV1F_LAUNCH_SSE(k_cdef_sse, cy, cuv);
        std::vector<uint32_t> a((size_t)g.nsb * nc), b2((size_t)g.nsb * nc), a2((size_t)g.nsb * nc), b3((size_t)g.nsb * nc);
        CK(cudaMemcpyAsync(a.data(), my, a.size() * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(b2.data(), cy, a.size() * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(a2.data(), muv, a.size() * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaMemcpyAsync(b3.data(), cuv, a.size() * 4, cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        size_t bad = 0;
        for (size_t i = 0; i < a.size(); i++) bad += (a[i] != b2[i]) + (!g.mono && a2[i] != b3[i]);
        fprintf(stderr, "[av1f] CDEF SSE tile vs global kernel: %zu of %zu entries differ\n", bad, 2 * a.size());
        cudaFree(cy);
        cudaFree(cuv);
    }
#undef AV1F_LAUNCH_SSE
    CK(cudaGetLastError());
    res->ms_cdef_sse += tsse.stop();

    // ---- joint preset search for cdef_bits 0..max, RD choice (libaom av1_cdef_search)
    Timer tsel(s);
    std::vector<int8_t> act(g.nsb);
    CK(cudaMemcpyAsync(act.data(), f.cdefIdx, g.nsb, cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    int nact = 0;
    for (int8_t a : act) nact += a >= 0;
    unsigned long long *tot = (unsigned long long *)(base + L.tot), *cur = (unsigned long long *)(base + L.cur);
    int *lev = (int *)(base + L.lev);
    unsigned long long *best = (unsigned long long *)(base + L.lev + 4 * 2 * 8 * sizeof(int));
    // exhaustive mode: prune to K strengths per plane (K x K joint pairs, single-CTA selection)
    int selY[64], selUV[64], ncs = nc;  // pruned column -> strength index (identity when not pruned)
    const uint32_t *smy = my, *smuv = muv;
    const int K = (o->cdef_mode == AV1F_CDEF_SEARCH && o->cdef_prune > 0) ? (o->cdef_prune < 16 ? o->cdef_prune : 16) : 0;
    if (K) {
        uint32_t *py = (uint32_t *)(base + L.prune), *puv = py + (size_t)g.nsb * 16;
        int *dsel = (int *)(puv + (size_t)g.nsb * 16);
        k_prune<<<1, 1024, 0, s>>>(my, nc, g.nsb, K, cur, dsel);
        k_gather_cols<<<(g.nsb * K + 255) / 256, 256, 0, s>>>(my, nc, g.nsb, dsel, K, py);
        if (!g.mono) {
            k_prune<<<1, 1024, 0, s>>>(muv, nc, g.nsb, K, cur, dsel + 16);
            k_gather_cols<<<(g.nsb * K + 255) / 256, 256, 0, s>>>(muv, nc, g.nsb, dsel + 16, K, puv);
        } else {
            CK(cudaMemsetAsync(puv, 0, (size_t)g.nsb * K * 4, s));
        }
        int hs[32];
        CK(cudaMemcpyAsync(hs, dsel, sizeof hs, cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        for (int i = 0; i < 16; i++) { selY[i] = i < K ? hs[i] : 0; selUV[i] = (i < K && !g.mono) ? hs[16 + i] : 0; }
        smy = py; smuv = puv; ncs = K;
        res->n_cand = K;
        for (int i = 0; i < 16; i++) { res->cand_y[i] = selY[i]; res->cand_uv[i] = selUV[i]; }
    } else {
        for (int i = 0; i < 64; i++) selY[i] = selUV[i] = i;
    }
    const int pairs = g.mono ? ncs : ncs * ncs;
    const bool fast_mono = g.mono && o->cdef_mode == AV1F_CDEF_AOM;  // libaom: no refinement for fast single-plane
    const int refine = o->cdef_mode == AV1F_CDEF_AOM ? 4 : (o->cdef_refine >= 0 ? o->cdef_refine : 4);
    const int joint = g.mono ? ncs : ncs * ncs;
    const int maxsig = joint == 1 ? 0 : 32 - __builtin_clz((unsigned)(joint - 1));
    int hl[4][16];
    unsigned long long htot[4];
    const int maxbits = o->cdef_max_bits < 3 ? o->cdef_max_bits : 3;
    auto search_one = [&](int *levp, int k) -> int {
        k_cur<<<(g.nsb + 255) / 256, 256, 0, s>>>(smy, smuv, ncs, g.nsb, levp, k, g.mono, cur);
        CK(cudaMemsetAsync(tot, 0, pairs * sizeof(unsigned long long), s));
        k_pairs<<<dim3((pairs + 127) / 128, (g.nsb + SEL_CHUNK - 1) / SEL_CHUNK), 128, 0, s>>>(smy, smuv, ncs, g.nsb, cur, g.mono, tot);
        k_argmin<<<1, 256, 0, s>>>(tot, ncs, g.mono, levp, k, best);
        return (int)cudaGetLastError();
    };
    const int lastBits = maxbits < maxsig ? maxbits : maxsig;
    if (pairs <= 256) {
        unsigned long long *dtots = best + 2;
        k_select_small<<<1, 1024, 0, s>>>(smy, smuv, ncs, g.nsb, g.mono, lastBits, refine, fast_mono, lev, dtots, cur);
        CK(cudaGetLastError());
        CK(cudaMemcpyAsync(htot, dtots, sizeof(unsigned long long) * (lastBits + 1), cudaMemcpyDeviceToHost, s));
    } else {
        for (int i = 0; i <= lastBits; i++) {
            int *lp = lev + i * 16;
            const int nb = 1 << i;
            for (int k = 0; k < nb; k++) CK(search_one(lp, k));
            if (!fast_mono)
                for (int r = 0; r < refine * nb; r++) {
                    k_shift<<<1, 32, 0, s>>>(lp, nb);
                    CK(search_one(lp, nb - 1));
                }
            CK(cudaMemcpyAsync(&htot[i], best, sizeof(unsigned long long), cudaMemcpyDeviceToHost, s));
        }
    }
    CK(cudaMemcpyAsync(hl, lev, sizeof hl, cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    const double rdmult = (double)tf.rdmult * o->cdef_rd_scale;
    int bestBits = 0;
    double bestRd = 0;
    for (int i = 0; i <= maxbits && i <= maxsig; i++) {
        const int nb = 1 << i;
        const long long bits = (long long)nact * i + nb * 6 * (g.mono ? 1 : 2);
        double rd;
        const double dsc = wmap ? 2048.0 / 16.0 : 2048.0;  // weighted SSE is stored x16
        if (o->cdef_rd_scale == 1.0f && !wmap)  // libaom integer RDCOST: ((bits * 512 * rdmult + 256) >> 9) + tot * 16 * 128
            rd = (double)(((bits * 512 * (long long)tf.rdmult + 256) >> 9)) + (double)htot[i] * 2048.0;
        else
            rd = bits * rdmult + (double)htot[i] * dsc;
        if (i == 0 || rd < bestRd) { bestRd = rd; bestBits = i; }
    }
    const int nb = 1 << bestBits;
    unsigned long long *dtot = best + 1;
    CK(cudaMemsetAsync(dtot, 0, sizeof(unsigned long long), s));
    int8_t *outIdx = d_sb_cdef_idx ? d_sb_cdef_idx : f.cdefIdx;
    k_assign<<<(g.nsb + 255) / 256, 256, 0, s>>>(smy, smuv, ncs, g.nsb, lev + bestBits * 16, nb, g.mono, f.cdefIdx, outIdx, dtot);
    CK(cudaMemcpyAsync(&res->cdef_sse, dtot, sizeof(unsigned long long), cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    res->ms_cdef_select += tsel.stop();
    rdOut = bestRd;
    if (!final) return 0;
    res->cdef_damping = damp;
    res->cdef_bits = bestBits;
    for (int j = 0; j < nb; j++) {
        const int jy = selY[hl[bestBits][j]], juv = selUV[hl[bestBits][8 + j]];
        int yp = cd.pri[jy / nsec], ys = cd.sec[jy % nsec];
        int up = cd.pri[juv / nsec], us = cd.sec[juv % nsec];
        if (g.mono) up = us = 0;
        if (o->cdef_mode == AV1F_CDEF_AOM && tf.cdef_mode == 1) {  // CDEF_ADAPTIVE: halve the signalled values
            auto sig = [](int s2) { return s2 == 4 ? 3 : s2; };
            auto app = [](int s2) { return s2 == 3 ? 4 : s2; };
            yp >>= 1; up >>= 1;
            ys = app(sig(ys) >> 1);
            us = app(sig(us) >> 1);
        }
        res->y_pri[j] = yp; res->y_sec[j] = ys;
        res->uv_pri[j] = up; res->uv_sec[j] = us;
    }
    return (int)cudaGetLastError();
    };
    if (o->cdef_damping == -2 && o->cdef_mode == AV1F_CDEF_SEARCH) {  // also search the damping (3..6)
        double bestRd = 0;
        int bestD = damping;
        for (int d = 3; d <= 6; d++) {
            CK(run_cdef(d, false));
            if (d == 3 || rdOut < bestRd) { bestRd = rdOut; bestD = d; }
        }
        CK(run_cdef(bestD, true));
    } else {
        CK(run_cdef(damping, true));
    }
    return (int)cudaGetLastError();
}

// Host convenience wrapper (tests / Python): host blocks + padded host planes in, results out.
extern "C" int av1f_lf_search_host(const av1g_frame_params *fp, const av1g_block *blocks, uint32_t n_blocks, uint8_t *const recon[3],
                                   const int recon_stride[3], const uint8_t *const src[3], const int src_stride[3],
                                   const av1f_lf_opts *o, av1f_lf_result *res, int8_t *sb_cdef_idx) {
    const Geo g = geo(fp);
    av1g_block *db;
    uint8_t *dr[3] = {0, 0, 0}, *ds[3] = {0, 0, 0};
    int8_t *di;
    void *scr;
    const size_t ns = av1f_lf_scratch_size(fp, o);
    CK(cudaMalloc(&db, sizeof(av1g_block) * (n_blocks ? n_blocks : 1)));
    CK(cudaMalloc(&di, g.nsb));
    CK(cudaMalloc(&scr, ns));
    CK(cudaMemcpy(db, blocks, sizeof(av1g_block) * n_blocks, cudaMemcpyHostToDevice));
    int rs[3], ss[3];
    for (int p = 0; p < (g.mono ? 1 : 3); p++) {
        const int w = p ? g.cw : g.lw, h = p ? g.ch : g.lh;
        CK(cudaMalloc(&dr[p], (size_t)w * h));
        CK(cudaMalloc(&ds[p], (size_t)w * h));
        CK(cudaMemcpy2D(dr[p], w, recon[p], recon_stride[p], w, h, cudaMemcpyHostToDevice));
        CK(cudaMemcpy2D(ds[p], w, src[p], src_stride[p], w, h, cudaMemcpyHostToDevice));
        rs[p] = ss[p] = w;
    }
    int rc;
    if (getenv("AV1F_ASYNC")) {  // test hook: route through the asynchronous variant
        av1f_lf_result *dres;
        CK(cudaMalloc(&dres, sizeof *dres));
        rc = av1f_lf_search_async(fp, db, n_blocks, dr, rs, ds, ss, o, dres, di, scr, 0);
        cudaDeviceSynchronize();
        cudaMemcpy(res, dres, sizeof *res, cudaMemcpyDeviceToHost);
        cudaFree(dres);
    } else {
        rc = av1f_lf_search(fp, db, n_blocks, dr, rs, ds, ss, o, res, di, scr, 0);
    }
    cudaDeviceSynchronize();
    if (sb_cdef_idx) cudaMemcpy(sb_cdef_idx, di, g.nsb, cudaMemcpyDeviceToHost);
    for (int p = 0; p < (g.mono ? 1 : 3); p++) {
        const int w = p ? g.cw : g.lw, h = p ? g.ch : g.lh;
        cudaMemcpy2D(recon[p], recon_stride[p], dr[p], w, w, h, cudaMemcpyDeviceToHost);  // deblocked
        cudaFree(dr[p]);
        cudaFree(ds[p]);
    }
    cudaFree(db);
    cudaFree(di);
    cudaFree(scr);
    return rc ? rc : (int)cudaGetLastError();
}

// Fully asynchronous variant: same decisions as av1f_lf_search (identical results), no host
// synchronisation and no timers; writes the result struct to DEVICE memory d_res.  The caller
// copies it (and d_sb_cdef_idx) after its own sync point.  d_recon is deblocked as above.
extern "C" int av1f_lf_search_async(const av1g_frame_params *fp, const av1g_block *d_blocks, uint32_t n_blocks,
                                    uint8_t *const d_recon[3], const int recon_stride[3], const uint8_t *const d_src[3],
                                    const int src_stride[3], const av1f_lf_opts *o, av1f_lf_result *d_res, int8_t *d_sb_cdef_idx,
                                    void *d_scratch, cudaStream_t s) {
    const Geo g = geo(fp);
    const Layout L = layout(fp, o);
    uint8_t *base = (uint8_t *)d_scratch;
    const int bq = fp->base_q_idx;
    av1f_tuneiq_frame tf;
    av1f_tuneiq_frame_init(bq, 1, 1, g.mono, &tf);
    const int sharp = o->sharpness >= 0 ? o->sharpness : tf.lf_sharpness;
    const int damping = o->cdef_damping >= 3 ? o->cdef_damping : tf.cdef_damping;
    const int nplanes = g.mono ? 1 : 3;
    int *lvDev = (int *)(base + L.async), *nactDev = lvDev + 4, *bitsDev = lvDev + 5;
    unsigned long long *tots = (unsigned long long *)(base + L.async + 64);
    av1g_frame_params fpc = *fp;
    fpc.enable_cdef = 1;
    fpc.cdef_bits = 0;
    fpc.cdef_damping_minus_3 = (uint8_t)(damping - 3);
    av1g_pf_frame f;
    CK(av1g_gpu_pf_prepare(&fpc, d_blocks, n_blocks, base + L.pf, &f, s));
    k_cdef_idx_norm<<<(g.nsb + 255) / 256, 256, 0, s>>>(f.cdefIdx, g.nsb);
    f.sharpness = sharp;
    float *wmap = nullptr;
    if (o->dist == AV1F_DIST_VARW) {
        wmap = (float *)(base + L.wmap);
        double *wsum = (double *)(base + L.wsum);
        const int n8c = g.MiCols / 2, n8r = g.MiRows / 2;
        CK(cudaMemsetAsync(wsum, 0, sizeof(double), s));
        k_wmap<<<dim3((n8c + 127) / 128, n8r), 128, 0, s>>>(d_src[0], src_stride[0], n8c, n8r, o->dist_c > 0 ? o->dist_c : 58.5f,
                                                           wmap, wsum);
        k_wnorm<<<(unsigned)(((size_t)n8c * n8r + 255) / 256), 256, 0, s>>>(wmap, (size_t)n8c * n8r, wsum);
    }
    // ---- deblocking
    FinArgs fa;
    memset(&fa, 0, sizeof fa);
    fa.sharp = sharp;
    for (int p = 0; p < nplanes; p++) { f.plane[p] = d_recon[p]; f.stride[p] = recon_stride[p]; }
    if (o->lf_mode == AV1F_LF_SEARCH) {
        unsigned long long *sse = (unsigned long long *)(base + L.sse);
        CK(cudaMemsetAsync(sse, 0, 64 * 3 * sizeof(unsigned long long), s));
        if (lf_fused(o))
            k_lf_tiles2<<<g.nsb, 256, 0, s>>>(f, d_src[0], d_src[1], d_src[2], src_stride[0], src_stride[1], src_stride[2], wmap, sse,
                                              o->dist == AV1F_DIST_GRAD);
        else
            CK(lf_search_copies(fp, g, f, d_recon, recon_stride, d_src, src_stride, wmap, base + L.cand, sse, s, false));
        k_lf_pick<<<1, 32, 0, s>>>(sse, nplanes, lvDev);
        const dim3 blk(32, 4), grid((g.MiCols + 31) / 32, (g.MiRows + 3) / 4, nplanes);
        for (int pass = 0; pass < 2; pass++) k_deblock_lv<<<grid, blk, 0, s>>>(f, pass, lvDev);
        fa.useLvDev = 1;
    } else {
        const int l = o->lf_mode == AV1F_LF_FROM_Q ? tf.lf_level : 0;
        for (int i = 0; i < 4; i++) { f.lfLevel[i] = l; fa.lfFixed[i] = l; }
        CK(av1g_gpu_deblock(&f, s));
    }
    // ---- CDEF
    fa.cdefEnable = o->cdef_mode != AV1F_CDEF_OFF;
    fa.damping = damping;
    fa.mono = g.mono;
    if (o->cdef_mode == AV1F_CDEF_OFF || (o->cdef_mode == AV1F_CDEF_AOM && tf.cdef_mode == 0)) {
        fa.cdefOff = 1;
        if (d_sb_cdef_idx) CK(cudaMemcpyAsync(d_sb_cdef_idx, f.cdefIdx, g.nsb, cudaMemcpyDeviceToDevice, s));
        k_finalize<<<1, 32, 0, s>>>(fa, lvDev, bitsDev, nullptr, nullptr, d_res);
        return (int)cudaGetLastError();
    }
    f.enableCdef = 1;
    f.cdefDamping = damping;
    CK(av1g_gpu_cdef_dir(&f, s));
    CdefCand cd;
    memset(&cd, 0, sizeof cd);
    int npri, nsec;
    if (o->cdef_mode == AV1F_CDEF_AOM) {
        npri = 2; nsec = 2;
        cd.pri[1] = 11; cd.sec[1] = 2;
    } else {
        npri = 16; nsec = 4;
        for (int i = 0; i < 16; i++) cd.pri[i] = i;
        cd.sec[0] = 0; cd.sec[1] = 1; cd.sec[2] = 2; cd.sec[3] = 4;
    }
    const int nc = npri * nsec;
    uint32_t *my = (uint32_t *)(base + L.msey), *muv = (uint32_t *)(base + L.mseuv);
    if (g.mono) CK(cudaMemsetAsync(muv, 0, (size_t)g.nsb * nc * 4, s));
    if (nc == 4 && !wmap)
        k_cdef_sse_t<2, 2, uint32_t><<<g.nsb, 256, 0, s>>>(f, d_src[0], d_src[1], d_src[2], src_stride[0], src_stride[1], src_stride[2],
                                                           cd, nullptr, my, muv);
    else if (nc == 4)
        k_cdef_sse_t<2, 2, float><<<g.nsb, 256, 0, s>>>(f, d_src[0], d_src[1], d_src[2], src_stride[0], src_stride[1], src_stride[2],
                                                        cd, wmap, my, muv);
    else if (!wmap)
        k_cdef_sse_t<16, 4, uint32_t><<<g.nsb, 256, 0, s>>>(f, d_src[0], d_src[1], d_src[2], src_stride[0], src_stride[1],
                                                            src_stride[2], cd, nullptr, my, muv);
    else
        k_cdef_sse_t<16, 4, float><<<g.nsb, 256, 0, s>>>(f, d_src[0], d_src[1], d_src[2], src_stride[0], src_stride[1], src_stride[2],
                                                         cd, wmap, my, muv);
    CK(cudaMemsetAsync(nactDev, 0, sizeof(int), s));
    k_count_active<<<(g.nsb + 255) / 256, 256, 0, s>>>(f.cdefIdx, g.nsb, nactDev);
    unsigned long long *tot = (unsigned long long *)(base + L.tot), *cur = (unsigned long long *)(base + L.cur);
    int *lev = (int *)(base + L.lev);
    unsigned long long *best = (unsigned long long *)(base + L.lev + 4 * 2 * 8 * sizeof(int));
    int ncs = nc;
    const uint32_t *smy = my, *smuv = muv;
    int *dsel = nullptr;
    const int K = (o->cdef_mode == AV1F_CDEF_SEARCH && o->cdef_prune > 0) ? (o->cdef_prune < 16 ? o->cdef_prune : 16) : 0;
    if (K) {
        uint32_t *py = (uint32_t *)(base + L.prune), *puv = py + (size_t)g.nsb * 16;
        dsel = (int *)(puv + (size_t)g.nsb * 16);
        k_prune<<<1, 1024, 0, s>>>(my, nc, g.nsb, K, cur, dsel);
        k_gather_cols<<<(g.nsb * K + 255) / 256, 256, 0, s>>>(my, nc, g.nsb, dsel, K, py);
        if (!g.mono) {
            k_prune<<<1, 1024, 0, s>>>(muv, nc, g.nsb, K, cur, dsel + 16);
            k_gather_cols<<<(g.nsb * K + 255) / 256, 256, 0, s>>>(muv, nc, g.nsb, dsel + 16, K, puv);
        } else {
            CK(cudaMemsetAsync(puv, 0, (size_t)g.nsb * K * 4, s));
            CK(cudaMemsetAsync(dsel + 16, 0, 16 * sizeof(int), s));
        }
        smy = py; smuv = puv; ncs = K;
    }
    const int pairs = g.mono ? ncs : ncs * ncs;
    const bool fast_mono = g.mono && o->cdef_mode == AV1F_CDEF_AOM;
    const int refine = o->cdef_mode == AV1F_CDEF_AOM ? 4 : (o->cdef_refine >= 0 ? o->cdef_refine : 4);
    const int joint = pairs;
    const int maxsig = joint == 1 ? 0 : 32 - __builtin_clz((unsigned)(joint - 1));
    const int maxbits = o->cdef_max_bits < 3 ? o->cdef_max_bits : 3;
    const int lastBits = maxbits < maxsig ? maxbits : maxsig;
    if (pairs <= 256) {
        k_select_small<<<1, 1024, 0, s>>>(smy, smuv, ncs, g.nsb, g.mono, lastBits, refine, fast_mono, lev, tots, cur);
    } else {
        for (int i = 0; i <= lastBits; i++) {
            int *lp = lev + i * 16;
            const int nb = 1 << i;
            auto one = [&](int k) {
                k_cur<<<(g.nsb + 255) / 256, 256, 0, s>>>(smy, smuv, ncs, g.nsb, lp, k, g.mono, cur);
                cudaMemsetAsync(tot, 0, pairs * sizeof(unsigned long long), s);
                k_pairs<<<dim3((pairs + 127) / 128, (g.nsb + SEL_CHUNK - 1) / SEL_CHUNK), 128, 0, s>>>(smy, smuv, ncs, g.nsb, cur,
                                                                                                     g.mono, tot);
                k_argmin<<<1, 256, 0, s>>>(tot, ncs, g.mono, lp, k, best);
            };
            for (int k = 0; k < nb; k++) one(k);
            if (!fast_mono)
                for (int r = 0; r < refine * nb; r++) {
                    k_shift<<<1, 32, 0, s>>>(lp, nb);
                    one(nb - 1);
                }
            CK(cudaMemcpyAsync(&tots[i], best, sizeof(unsigned long long), cudaMemcpyDeviceToDevice, s));
        }
    }
    RdArgs ra = {lastBits, g.mono, tf.rdmult, (double)o->cdef_rd_scale, wmap ? 2048.0 / 16.0 : 2048.0};
    k_rd_pick<<<1, 32, 0, s>>>(tots, nactDev, ra, bitsDev);
    int8_t *outIdx = d_sb_cdef_idx ? d_sb_cdef_idx : f.cdefIdx;
    k_assign_dev<<<(g.nsb + 255) / 256, 256, 0, s>>>(smy, smuv, ncs, g.nsb, lev, bitsDev, g.mono, f.cdefIdx, outIdx);
    fa.nsec = nsec;
    fa.cd = cd;
    fa.halve = o->cdef_mode == AV1F_CDEF_AOM && tf.cdef_mode == 1;
    fa.pruned = K > 0;
    k_finalize<<<1, 32, 0, s>>>(fa, lvDev, bitsDev, lev, dsel, d_res);
    return (int)cudaGetLastError();
}

extern "C" int av1f_cdef_select_tables(const uint32_t *d_my, const uint32_t *d_muv, int nc, int nsb, const int8_t *d_active, int nact,
                                       int mono, int max_bits, int refine, int prune_k, double lambda, av1f_cdef_sel *out,
                                       int8_t *d_sb_idx, cudaStream_t s) {
    unsigned long long *cur, *tot, *best;
    int *lev, *dsel;
    uint32_t *py, *puv;
    const int K = prune_k > 0 ? (prune_k < 16 ? prune_k : 16) : 0;
    CK(cudaMallocAsync(&cur, (size_t)nsb * 8, s));
    CK(cudaMallocAsync(&tot, 64 * 64 * 8, s));
    CK(cudaMallocAsync(&best, 16 * 8, s));
    CK(cudaMallocAsync(&lev, 4 * 16 * sizeof(int), s));
    CK(cudaMallocAsync(&dsel, 32 * sizeof(int), s));
    CK(cudaMallocAsync(&py, (size_t)nsb * 16 * 4 * 2 + 4, s));
    puv = py + (size_t)nsb * 16;
    int sel[2][64];
    const uint32_t *smy = d_my, *smuv = d_muv;
    int ncs = nc;
    if (K && K < nc) {
        k_prune<<<1, 1024, 0, s>>>(d_my, nc, nsb, K, cur, dsel);
        k_gather_cols<<<(nsb * K + 255) / 256, 256, 0, s>>>(d_my, nc, nsb, dsel, K, py);
        if (!mono) {
            k_prune<<<1, 1024, 0, s>>>(d_muv, nc, nsb, K, cur, dsel + 16);
            k_gather_cols<<<(nsb * K + 255) / 256, 256, 0, s>>>(d_muv, nc, nsb, dsel + 16, K, puv);
        } else {
            CK(cudaMemsetAsync(puv, 0, (size_t)nsb * K * 4, s));
        }
        int hs[32];
        CK(cudaMemcpyAsync(hs, dsel, sizeof hs, cudaMemcpyDeviceToHost, s));
        CK(cudaStreamSynchronize(s));
        for (int i = 0; i < 64; i++) { sel[0][i] = i < K ? hs[i] : 0; sel[1][i] = (i < K && !mono) ? hs[16 + i] : 0; }
        smy = py; smuv = puv; ncs = K;
    } else {
        for (int i = 0; i < 64; i++) sel[0][i] = sel[1][i] = i;
    }
    const int pairs = mono ? ncs : ncs * ncs;
    const int maxsig = pairs == 1 ? 0 : 32 - __builtin_clz((unsigned)(pairs - 1));
    const int lastBits = (max_bits < 3 ? max_bits : 3) < maxsig ? (max_bits < 3 ? max_bits : 3) : maxsig;
    unsigned long long htot[4];
    int hl[4][16];
    if (pairs <= 256) {
        k_select_small<<<1, 1024, 0, s>>>(smy, smuv, ncs, nsb, mono, lastBits, refine, 0, lev, best + 2, cur);
        CK(cudaMemcpyAsync(htot, best + 2, sizeof(unsigned long long) * (lastBits + 1), cudaMemcpyDeviceToHost, s));
    } else {
        for (int i = 0; i <= lastBits; i++) {
            int *lp = lev + i * 16;
            const int nb = 1 << i;
            auto one = [&](int k) {
                k_cur<<<(nsb + 255) / 256, 256, 0, s>>>(smy, smuv, ncs, nsb, lp, k, mono, cur);
                cudaMemsetAsync(tot, 0, pairs * sizeof(unsigned long long), s);
                k_pairs<<<dim3((pairs + 127) / 128, (nsb + SEL_CHUNK - 1) / SEL_CHUNK), 128, 0, s>>>(smy, smuv, ncs, nsb, cur, mono, tot);
                k_argmin<<<1, 256, 0, s>>>(tot, ncs, mono, lp, k, best);
            };
            for (int k = 0; k < nb; k++) one(k);
            for (int r = 0; r < refine * nb; r++) {
                k_shift<<<1, 32, 0, s>>>(lp, nb);
                one(nb - 1);
            }
            CK(cudaMemcpyAsync(&htot[i], best, sizeof(unsigned long long), cudaMemcpyDeviceToHost, s));
        }
    }
    CK(cudaMemcpyAsync(hl, lev, sizeof hl, cudaMemcpyDeviceToHost, s));
    CK(cudaStreamSynchronize(s));
    int bb = 0;
    double brd = 0;
    for (int i = 0; i <= lastBits; i++) {
        const double rd = (double)htot[i] + lambda * ((double)nact * i + (1 << i) * 6 * (mono ? 1 : 2));
        if (i == 0 || rd < brd) { brd = rd; bb = i; }
    }
    out->bits = bb;
    out->rd = brd;
    out->cost = (double)htot[bb];
    for (int j = 0; j < 8; j++) {
        out->lev_y[j] = j < (1 << bb) ? sel[0][hl[bb][j]] : 0;
        out->lev_uv[j] = j < (1 << bb) ? sel[1][hl[bb][8 + j]] : 0;
    }
    if (d_sb_idx) {
        unsigned long long *dtot = best + 1;
        CK(cudaMemsetAsync(dtot, 0, 8, s));
        k_assign<<<(nsb + 255) / 256, 256, 0, s>>>(smy, smuv, ncs, nsb, lev + bb * 16, 1 << bb, mono, d_active, d_sb_idx, dtot);
    }
    cudaFreeAsync(cur, s); cudaFreeAsync(tot, s); cudaFreeAsync(best, s); cudaFreeAsync(lev, s); cudaFreeAsync(dsel, s);
    cudaFreeAsync(py, s);
    CK(cudaStreamSynchronize(s));
    return (int)cudaGetLastError();
}
