// varboost.cu - GPU variance boost (libaom tune=iq deltaq-mode 6).  See av1f/varboost_gpu.h.
//
// k_sb_stats: 4 SBs per 256-thread block; thread = one 8x8 block (8 x 8-byte loads), integer
//             aom_variance8x8 / 64; the octile ranks 31/39/47 are found by counting (stable rank),
//             then target = lut[base][var] (var < 1024) else base.
// k_sb_round: delta_q_res rounding in libaom's order: one thread per (image, tile) or
//             (image, tile, SB row); the nearest mode is fully parallel.
#include <cstdio>
#include <vector>

#include "av1f/tuneiq.h"
#include "av1f/varboost_gpu.h"

__global__ void k_vb_luts(uint8_t *luts, int strength) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 256 * 1024) return;
    const int base = i >> 10, var = i & 1023;
    luts[i] = (uint8_t)(base == 0 ? 0 : av1f_varboost_qindex(base, (unsigned)var, strength));
}

static __device__ __forceinline__ int find_image(const av1f_vb_desc *d, int n, uint32_t sb) {
    int lo = 0, hi = n - 1;
    while (lo < hi) {
        const int mid = (lo + hi + 1) >> 1;
        if (d[mid].sb_off <= sb) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

__global__ void __launch_bounds__(256) k_sb_stats(const av1f_vb_desc *desc, int n, uint32_t total, const uint8_t *luts,
                                                  uint32_t *sb_var, uint8_t *sb_target) {
    __shared__ unsigned v[4][64];
    __shared__ unsigned sel[4][3];
    const int g = threadIdx.x >> 6, t = threadIdx.x & 63;
    const uint32_t sb = blockIdx.x * 4 + g;
    const bool live = sb < total;
    int img = 0, base = 0;
    if (live) {
        img = find_image(desc, n, sb);
        const av1f_vb_desc &d = desc[img];
        const int cols = ((((d.width + 7) >> 3) * 8) + 63) >> 6;
        const uint32_t local = sb - d.sb_off;
        const int r = local / cols, c = local % cols;
        const uint8_t *p = d.y + (size_t)(r * 64 + (t >> 3) * 8) * d.pitch + c * 64 + (t & 7) * 8;
        int s = 0;
        unsigned q = 0;
#pragma unroll
        for (int k = 0; k < 8; k++) {
            const uint2 w = *reinterpret_cast<const uint2 *>(p + (size_t)k * d.pitch);
#pragma unroll
            for (int b = 0; b < 4; b++) {
                const int x0 = (w.x >> (8 * b)) & 255, x1 = (w.y >> (8 * b)) & 255;
                s += x0 + x1;
                q += x0 * x0 + x1 * x1;
            }
        }
        v[g][t] = av1f_var8x8_from_sums(s, q);
        base = d.base_q_idx;
    }
    __syncthreads();
    if (live) {
        const unsigned x = v[g][t];
        int rank = 0;
#pragma unroll 8
        for (int j = 0; j < 64; j++) {
            const unsigned y = v[g][j];
            rank += (y < x) | ((y == x) & (j < t));
        }
        if (rank == 31) sel[g][0] = x;
        if (rank == 39) sel[g][1] = x;
        if (rank == 47) sel[g][2] = x;
    }
    __syncthreads();
    if (live && t == 0) {
        const unsigned var = (sel[g][0] + 2 * sel[g][1] + sel[g][2] + 2) / 4;
        if (sb_var) sb_var[sb] = var;
        sb_target[sb] = var < 1024 ? luts[base * 1024 + var] : (uint8_t)base;
    }
}

// one work item = (image, tile) [TILE] or (image, tile, SB row in tile) [ROW] or SB [NEAREST]
__global__ void k_sb_round(const av1f_vb_desc *desc, int n, const uint8_t *target, uint8_t *out, int order) {
    const int img = blockIdx.x;
    const av1f_vb_desc &d = desc[img];
    const int cols = ((((d.width + 7) >> 3) * 8) + 63) >> 6, rows = ((((d.height + 7) >> 3) * 8) + 63) >> 6;
    const int base = d.base_q_idx, res = av1f_delta_q_res(base);
    const uint8_t *tg = target + d.sb_off;
    uint8_t *o = out + d.sb_off;
    if (order == AV1F_VB_NEAREST) {
        for (int i = threadIdx.x; i < rows * cols; i += blockDim.x) {
            const int d = tg[i] - base + res / 2;                       // nearest, ties toward higher qindex
            const int k = d >= 0 ? d / res : -((-d + res - 1) / res);   // floor(d / res)
            int q = base + k * res;
            while (q < 1) q += res;
            while (q > 255) q -= res;
            o[i] = (uint8_t)q;
        }
        return;
    }
    const int tw = (cols + (1 << d.tile_cols_log2) - 1) >> d.tile_cols_log2;  // uniform tile size in SBs
    const int th = (rows + (1 << d.tile_rows_log2) - 1) >> d.tile_rows_log2;
    const int tcols = (cols + tw - 1) / tw, trows = (rows + th - 1) / th;
    const int items = order == AV1F_VB_ROW ? rows * tcols : tcols * trows;
    for (int it = threadIdx.x; it < items; it += blockDim.x) {
        int c0, c1, r0, r1;
        if (order == AV1F_VB_ROW) {
            const int r = it / tcols, tc = it % tcols;
            r0 = r; r1 = r + 1;
            c0 = tc * tw; c1 = min(cols, c0 + tw);
        } else {
            const int tr = it / tcols, tc = it % tcols;
            r0 = tr * th; r1 = min(rows, r0 + th);
            c0 = tc * tw; c1 = min(cols, c0 + tw);
        }
        int prev = base;
        for (int r = r0; r < r1; r++)
            for (int c = c0; c < c1; c++) {
                prev = av1f_adjust_q(res, prev, tg[r * cols + c]);
                o[r * cols + c] = (uint8_t)prev;
            }
    }
}

// The LUT is computed on the HOST: libaom evaluates log2() with glibc, CUDA's log2 differs by up to
// 1 ulp and flipped 2 of 783k (base, variance, strength) entries at a target-q tie in testing.
extern "C" int av1f_vb_build_luts(uint8_t *d_luts, int strength, cudaStream_t s) {
    static thread_local std::vector<uint8_t> h;
    h.resize(256 * 1024);
    for (int b = 0; b < 256; b++)
        for (int v = 0; v < 1024; v++) h[b * 1024 + v] = (uint8_t)(b == 0 ? 0 : av1f_varboost_qindex(b, (unsigned)v, strength));
    cudaError_t e = cudaMemcpyAsync(d_luts, h.data(), h.size(), cudaMemcpyHostToDevice, s);
    if (!e) e = cudaStreamSynchronize(s);  // h is reused by the next call
    return (int)e;
}

// Device-side LUT (for reference / tests only - see above).
extern "C" int av1f_vb_build_luts_device(uint8_t *d_luts, int strength, cudaStream_t s) {
    k_vb_luts<<<256 * 1024 / 256, 256, 0, s>>>(d_luts, strength);
    return (int)cudaGetLastError();
}

extern "C" int av1f_varboost(const av1f_vb_desc *d_desc, int n, uint32_t total, const uint8_t *d_luts, int order,
                             uint32_t *d_sb_var, uint8_t *d_sb_target, uint8_t *d_sb_qindex, cudaStream_t s) {
    uint8_t *tgt = d_sb_target;
    if (!tgt) {
        cudaError_t e = cudaMallocAsync(&tgt, total, s);
        if (e) return (int)e;
    }
    k_sb_stats<<<(total + 3) / 4, 256, 0, s>>>(d_desc, n, total, d_luts, d_sb_var, tgt);
    k_sb_round<<<n, 128, 0, s>>>(d_desc, n, tgt, d_sb_qindex, order);
    if (!d_sb_target) cudaFreeAsync(tgt, s);
    return (int)cudaGetLastError();
}

extern "C" int av1f_varboost_host(const uint8_t *y, int pitch, int width, int height, int base, int tcl2, int trl2, int order,
                                  int strength, uint32_t *sb_var, uint8_t *sb_target, uint8_t *sb_qindex) {
    const uint32_t nsb = av1f_vb_num_sbs(width, height);
    const int rows = (int)(nsb / (uint32_t)((((width + 7) >> 3) * 8 + 63) >> 6));
    uint8_t *dy, *dl, *dt, *dq;
    uint32_t *dv;
    av1f_vb_desc *dd;
    const size_t bytes = (size_t)pitch * rows * 64;
    cudaMalloc(&dy, bytes);
    cudaMalloc(&dl, 256 * 1024);
    cudaMalloc(&dt, nsb);
    cudaMalloc(&dq, nsb);
    cudaMalloc(&dv, 4 * nsb);
    cudaMalloc(&dd, sizeof(av1f_vb_desc));
    cudaMemcpy(dy, y, bytes, cudaMemcpyHostToDevice);
    av1f_vb_desc d = {dy, pitch, width, height, base, tcl2, trl2, 0};
    cudaMemcpy(dd, &d, sizeof d, cudaMemcpyHostToDevice);
    av1f_vb_build_luts(dl, strength, 0);
    int rc = av1f_varboost(dd, 1, nsb, dl, order, dv, dt, dq, 0);
    cudaDeviceSynchronize();
    if (!rc) rc = (int)cudaGetLastError();
    if (sb_var) cudaMemcpy(sb_var, dv, 4 * nsb, cudaMemcpyDeviceToHost);
    if (sb_target) cudaMemcpy(sb_target, dt, nsb, cudaMemcpyDeviceToHost);
    if (sb_qindex) cudaMemcpy(sb_qindex, dq, nsb, cudaMemcpyDeviceToHost);
    cudaFree(dy); cudaFree(dl); cudaFree(dt); cudaFree(dq); cudaFree(dv); cudaFree(dd);
    return rc;
}
