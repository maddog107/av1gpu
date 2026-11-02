// luma palettes of the lossy path (screen-content frames, GpImage::sct): the palette of a block, its colour index
// map and an estimate of their bits, shared by the analysis (k_pal.cu) and the closed loop (k_recon.cuh) so both
// derive the same palette from the source.
//   pal_hist   histogram of the block's coded samples -> its distinct values (at most 64 are kept)
//   pal_kmeans a palette of at most nT colours: the distinct values themselves if there are at most nT, else the nT
//              most frequent values (top: exact colours, libaom's perform_top_color_palette_search) or 1-D k-means
//              with nT centres started from them (libaom's perform_k_means_palette_search), rounded and merged
//   pal_map    each sample's nearest colour (the coded part; outside it the copies of the last coded column / row,
//              as the decoder makes them)
//   pal_bits   has_palette_y + palette_size + the colours as literals (cache not counted) + every index at its exact
//              context with the default CDFs
// The analysis tries nT = 8 .. 2, top colours and k-means, and keeps the cheapest; its candidate code tells the closed
// loop which palette to rebuild: PAL_CAND + nT - 2 (k-means), + 8 (top colours).
#pragma once
#include "dev_common.cuh"

#define PAL_CAND 0xF0   // candidate codes 0xF0 .. 0xF6 / 0xF8 .. 0xFE: a luma palette of at most 2 .. 8 colours, k-means /
                        // top colours (GpBlockRes::cand)
__device__ __forceinline__ bool pal_cand(int c) { return (c & 0xF0) == PAL_CAND && (c & 7) != 7; }
__device__ __forceinline__ int pal_target(int c) { return (c & 7) + 2; }
__device__ __forceinline__ bool pal_top(int c) { return (c & 8) != 0; }
__device__ __forceinline__ int pal_code(int nT, bool top) { return PAL_CAND + nT - 2 + (top ? 8 : 0); }

// warp-wide; hist: shared scratch of 1 << BD counters; dv / dc (shared, 64 each): the distinct values ascending and
// their counts.  Returns the number of distinct values (more than 64: only the first 64 are listed)
template <int BD, typename Px>
__device__ int pal_hist(const Px *src, int sp, int ow, int oh, uint32_t *hist, uint16_t *dv, uint16_t *dc) {
    const int lane = threadIdx.x & 31;
    constexpr int NB = 1 << BD;
    for (int v = lane; v < NB; v += 32) hist[v] = 0;
    __syncwarp();
    for (int i = lane; i < ow * oh; i += 32) atomicAdd(&hist[src[(i / ow) * sp + i % ow]], 1u);
    __syncwarp();
    int nd = 0;
    for (int base = 0; base < NB; base += 32) {
        const int v = base + lane;
        const bool has = hist[v] != 0;
        const uint32_t b = __ballot_sync(0xffffffffu, has);
        const int pos = nd + __popc(b & ((1u << lane) - 1u));
        if (has && pos < 64) { dv[pos] = (uint16_t)v; dc[pos] = (uint16_t)min(hist[v], 65535u); }
        nd += __popc(b);
    }
    __syncwarp();
    return nd;
}

// one thread; nd (2..64) distinct values dv with counts dc; col out.  Returns the palette size (<= nT)
static __device__ __noinline__ int pal_kmeans(const uint16_t *dv, const uint16_t *dc, int nd, int nT, bool top, int bd, uint16_t *col) {
    if (nd <= nT) {   // every value: exact
        for (int i = 0; i < nd; i++) col[i] = dv[i];
        return nd;
    }
    // start: the nT most frequent values (ties: the lower value), ascending
    uint32_t used = 0, used2 = 0;
    float c[8];
    for (int k = 0; k < nT; k++) {
        int best = -1;
        for (int i = 0; i < nd; i++) {
            const bool u = i < 32 ? (used >> i) & 1 : (used2 >> (i - 32)) & 1;
            if (!u && (best < 0 || dc[i] > dc[best])) best = i;
        }
        if (best < 32) used |= 1u << best; else used2 |= 1u << (best - 32);
        c[k] = (float)dv[best];
    }
    for (int a = 1; a < nT; a++) {
        const float t = c[a];
        int j = a - 1;
        while (j >= 0 && c[j] > t) { c[j + 1] = c[j]; j--; }
        c[j + 1] = t;
    }
    if (top) {   // the most frequent values themselves (distinct, ascending)
        for (int k = 0; k < nT; k++) col[k] = (uint16_t)c[k];
        return nT;
    }
    // Lloyd iterations on the histogram (nearest centre, the lower one on ties), at most 10
    for (int it = 0; it < 10; it++) {
        float s[8] = {0, 0, 0, 0, 0, 0, 0, 0}, w[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (int i = 0; i < nd; i++) {
            int bk = 0;
            float bd2 = fabsf((float)dv[i] - c[0]);
            for (int k = 1; k < nT; k++) {
                const float d = fabsf((float)dv[i] - c[k]);
                if (d < bd2) { bd2 = d; bk = k; }
            }
            s[bk] += (float)dv[i] * dc[i];
            w[bk] += dc[i];
        }
        bool moved = false;
        for (int k = 0; k < nT; k++)
            if (w[k] > 0.f) {
                const float m = s[k] / w[k];
                moved |= fabsf(m - c[k]) > 0.01f;
                c[k] = m;
            }
        if (!moved) break;
    }
    // round, sort, merge equal colours
    int r[8], n = 0;
    for (int k = 0; k < nT; k++) r[k] = min((1 << bd) - 1, max(0, (int)lrintf(c[k])));
    for (int a = 1; a < nT; a++) {
        const int t = r[a];
        int j = a - 1;
        while (j >= 0 && r[j] > t) { r[j + 1] = r[j]; j--; }
        r[j + 1] = t;
    }
    for (int k = 0; k < nT; k++)
        if (!n || r[k] != col[n - 1]) col[n++] = (uint16_t)r[k];
    return n;
}

// nearest colour of v (the lower index on ties)
__device__ __forceinline__ int pal_nearest(int v, const uint16_t *col, int n) {
    int bk = 0, bd = abs(v - (int)col[0]);
    for (int k = 1; k < n; k++) {
        const int d = abs(v - (int)col[k]);
        if (d < bd) { bd = d; bk = k; }
    }
    return bk;
}

// the W x H index map (stride W): coded part ow x oh, the rest copies of its last column / row.  Lanes [l0, l0 + nl).
template <typename Px>
__device__ void pal_map(const Px *src, int sp, int W, int H, int ow, int oh, const uint16_t *col, int n, uint8_t *map, int l, int nl) {
    for (int i = l; i < W * H; i += nl) {
        const int r = min(i / W, oh - 1), c = min(i % W, ow - 1);
        map[i] = (uint8_t)pal_nearest(src[r * sp + c], col, n);
    }
}

// bits of a luma palette and its map (warp-wide, every lane returns the same)
static __device__ float pal_bits(int n, const uint16_t *col, const uint8_t *map, int W, int ow, int oh, int bsize, int bd, const float *C) {
    const int lane = threadIdx.x & 31;
    float b = 0.f;
    for (int i = lane; i < ow * oh; i += 32) {
        if (!i) continue;
        const int r = i / ow, c = i % ow;
        uint8_t order[8];
        const int ctx = av1g_palette_color_context(map, W, r, c, n, order);
        int s = 0;
        while (s < n - 1 && order[s] != map[r * W + c]) s++;
        b += C[(uint32_t)(av1g_palette_color_cdf(&g_cdf_layout, 0, n, ctx) - (uint16_t *)&g_cdf_layout) + s];
    }
    b = warp_sum(b);
    const int bctx = av1g_palette_bsize_ctx(bsize);
    b += C[CID(palette_y_mode[bctx][0]) + 1] + C[CID(palette_y_size[bctx]) + n - 2];
    // literals: the first colour, palette_num_extra_bits and the deltas (bits shrink with the remaining range)
    uint16_t lits[8], cache[1] = {0};
    uint8_t found[1];
    int nf, extra, bits;
    av1g_palette_split(col, n, cache, 0, bd, 0, found, &nf, lits, &extra, &bits);
    b += (float)(bd + 2);
    for (int k = 1; k < n; k++) {
        b += (float)bits;
        bits = min(bits, av1g_ceil_log2((1 << bd) - lits[k] - 1));
    }
    int nb = 0;
    for (int x = n; x; x >>= 1) nb++;
    return b + (float)(nb - 1);   // the first index (NS)
}
