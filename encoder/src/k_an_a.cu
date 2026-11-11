// open-loop analysis kernels (4x4, 4x8, 8x4, 8x8, 4x16, 16x4), the live 16x16 cells they take their blocks from, and the
// block lists of the staged analysis (k_an_list, an_stage.h).
#include "kernels.h"
#include "dev_common.cuh"
#include "k_analysis.cuh"
#include "k_analysis_small.cuh"
#include "k_an_list.cuh"

GP_DEFINE_TABLE_UPLOAD(gp_upload_tables_an_a)

// Live 16x16 cells, one thread per cell of the batch.  prune_flat decides the same for every block below 16x16 of a
// cell (one SB, so one qindex; one var16 entry), so a cell is live unless it prunes the cell's blocks.  The blocks of
// a pruned cell (inside the frame by prune_flat's test) get J = GP_INF, ncand = 0 here, as the analysis kernels wrote
// them; the live cells are appended to L.cell, a warp's in one atomic.  Their order is that of the atomics, which
// only changes which unit analyses which block.
template <int BD>
__global__ void k_live16(GpBatch B, GpParams P, GpLive L) {
    const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    bool live = false;
    uint32_t img = 0, cell = 0;
    if (g < L.nCells) {
        int lo = 0, hi = B.nimg - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (L.cellStart[mid] <= g) lo = mid; else hi = mid - 1;
        }
        const GpImage &I = B.img[lo];
        const uint32_t c = g - L.cellStart[lo];
        const uint32_t cols16 = (uint32_t)(I.MiCols + 3) >> 2;
        const int cx = (int)(c % cols16), cy = (int)(c / cols16);
        live = !prune_flat<BD>(I, P, cx * 16, cy * 16, 4, 4, sb_qindex(I, cx * 16, cy * 16));
        if (!live) {
            const int shapes[] = {AN_LIVE_SHAPES};   // every an_cellwise shape (k_analysis.cuh)
#pragma unroll
            for (int k = 0; k < (int)(sizeof shapes / sizeof shapes[0]); k++) {
                const int s = shapes[k], nbx = I.shNbx[s];
                if (!nbx) continue;
                const int w = bw_px(s), h = bh_px(s);
                for (int by = cy * 16 / h; by < (cy + 1) * 16 / h; by++)
                    for (int bx = cx * 16 / w; bx < (cx + 1) * 16 / w; bx++) {
                        GpBlockRes &o = B.blk[I.shOff[s] + (uint32_t)by * nbx + bx];
                        o.J = GP_INF;
                        o.ncand = 0;
                    }
            }
        }
        img = (uint32_t)lo;
        cell = ((uint32_t)cy << 16) | (uint32_t)cx;
    }
    const unsigned m = __ballot_sync(0xffffffffu, live);
    if (!m) return;
    const int lane = threadIdx.x & 31;
    uint32_t base = 0;
    if (lane == 0) base = atomicAdd(L.count, (uint32_t)__popc(m));
    base = __shfl_sync(0xffffffffu, base, 0);
    if (live) ((uint2 *)L.cell)[base + __popc(m & ((1u << lane) - 1))] = make_uint2(img, cell);
}

void gp_launch_live16(const GpBatch &B, const GpParams &P, const GpLive &L, cudaStream_t st, int bd) {
    if (!L.cell) return;
    cudaMemsetAsync(L.count, 0, sizeof(uint32_t), st);
    if (!L.nCells) return;
    if (bd == 10) k_live16<10><<<(L.nCells + 127) / 128, 128, 0, st>>>(B, P, L);
    else k_live16<8><<<(L.nCells + 127) / 128, 128, 0, st>>>(B, P, L);
}

// The blocks of one stage's shapes (one per blockIdx.y) that the staged analysis analyses (an_list_alive): appended to
// the shape's list as shape-global indices, a warp's in one atomic (the list order only changes which unit analyses
// which block); the others inside the grid get J = GP_INF, ncand = 0, as the analysis kernels write pruned blocks.
template <int BD>
__global__ void k_an_list(GpBatch B, GpParams P, GpAnListStage L) {
    const GpAnListShape &sh = L.s[blockIdx.y];
    const uint32_t g = blockIdx.x * blockDim.x + threadIdx.x;
    bool live = false;
    if (g < sh.total) {
        const uint32_t *starts = B.shImgStart + (size_t)sh.bs * (B.nimg + 1);
        int lo = 0, hi = B.nimg - 1;
        while (lo < hi) {
            const int mid = (lo + hi + 1) >> 1;
            if (starts[mid] <= g) lo = mid; else hi = mid - 1;
        }
        const GpImage &I = B.img[lo];
        const uint32_t li = g - starts[lo];
        const int nbx = I.shNbx[sh.bs];
        const int x = (int)(li % nbx) * bw_px(sh.bs), y = (int)(li / nbx) * bh_px(sh.bs);
        live = x < I.MiCols * 4 && y < I.MiRows * 4 && an_list_alive<BD>(P, L, sh, I, x, y);
        if (!live) {
            GpBlockRes &o = B.blk[I.shOff[sh.bs] + li];
            o.J = GP_INF;
            o.ncand = 0;
        }
    }
    const unsigned m = __ballot_sync(0xffffffffu, live);
    if (!m) return;
    const int lane = threadIdx.x & 31;
    uint32_t base = 0;
    if (lane == 0) base = atomicAdd(sh.count, (uint32_t)__popc(m));
    base = __shfl_sync(0xffffffffu, base, 0);
    if (live) sh.list[base + __popc(m & ((1u << lane) - 1))] = g;
}

void gp_launch_an_list(const GpBatch &B, const GpParams &P, const GpAnListStage &L, cudaStream_t st, int bd) {
    uint32_t mx = 0;
    for (int k = 0; k < L.n; k++) mx = L.s[k].total > mx ? L.s[k].total : mx;
    if (!mx) return;
    const dim3 grid((mx + 127) / 128, (unsigned)L.n);
    if (bd == 10) k_an_list<10><<<grid, 128, 0, st>>>(B, P, L);
    else k_an_list<8><<<grid, 128, 0, st>>>(B, P, L);
}

// x (may be null): the shape's options of the staged analysis; the MODE 1 kernel when it sets any (an_x_on)
template <int W, int H, int BS, int NW>
static void launch_an(const GpBatch &B, const GpParams &P, const GpLive &L, uint32_t n, cudaStream_t st, int bd, const GpAnX *x) {
    n = an_units<W, H>(L, n);
    if (!n) return;
    uint32_t grid = (n + NW - 1) / NW;
    // non-persistent grid: short CTAs let the hardware scheduler interleave other streams (recon wavefronts)
    if (grid > (1u << 30)) grid = 1u << 30;
    if constexpr (an_staged_shape(W, H)) {   // (no MODE 1 instance for the others)
        if (an_x_on(x)) {
            if (bd == 10) k_analysis<W, H, BS, NW, 10, 1><<<grid, NW * 32, 0, st>>>(B, P, n, L, *x);
            else k_analysis<W, H, BS, NW, 8, 1><<<grid, NW * 32, 0, st>>>(B, P, n, L, *x);
            return;
        }
    }
    if (bd == 10) k_analysis<W, H, BS, NW, 10><<<grid, NW * 32, 0, st>>>(B, P, n, L, GpAnX{});
    else k_analysis<W, H, BS, NW, 8><<<grid, NW * 32, 0, st>>>(B, P, n, L, GpAnX{});
}

template <int W, int H, int BS>
static void launch_small(const GpBatch &B, const GpParams &P, const GpLive &L, uint32_t n, cudaStream_t st, int bd, const GpAnX *x) {
    n = an_units<W, H>(L, n);
    if (!n) return;
    const uint32_t grid = (n + 127) / 128;
    if constexpr (W == 4 && H <= 8) {   // 4x4, 4x8: 6 CTAs per SM (k_analysis_small.cuh)
        if constexpr (an_staged_shape(W, H)) {   // (no MODE 1 instance for the others)
            if (an_x_on(x)) {
                if (bd == 10) k_analysis_small6<W, H, BS, 10, 1><<<grid, 128, 0, st>>>(B, P, n, L, *x);
                else k_analysis_small6<W, H, BS, 8, 1><<<grid, 128, 0, st>>>(B, P, n, L, *x);
                return;
            }
        }
        if (bd == 10) k_analysis_small6<W, H, BS, 10><<<grid, 128, 0, st>>>(B, P, n, L, GpAnX{});
        else k_analysis_small6<W, H, BS, 8><<<grid, 128, 0, st>>>(B, P, n, L, GpAnX{});
    } else {
        if constexpr (an_staged_shape(W, H)) {   // (no MODE 1 instance for the others)
            if (an_x_on(x)) {
                if (bd == 10) k_analysis_small<W, H, BS, 10, 1><<<grid, 128, 0, st>>>(B, P, n, L, *x);
                else k_analysis_small<W, H, BS, 8, 1><<<grid, 128, 0, st>>>(B, P, n, L, *x);
                return;
            }
        }
        if (bd == 10) k_analysis_small<W, H, BS, 10><<<grid, 128, 0, st>>>(B, P, n, L, GpAnX{});
        else k_analysis_small<W, H, BS, 8><<<grid, 128, 0, st>>>(B, P, n, L, GpAnX{});
    }
}

// xs (null: none): per shape, the options of the staged analysis; mask: the shapes to launch (AV1G_BLOCK_* bits)
void gp_launch_an_a(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st, int bd, const GpLive &L,
                    const GpAnX *xs, uint32_t mask) {
    auto n = [&](int s) -> uint32_t { return (mask >> s) & 1 ? shTotal[s] : 0; };
    auto x = [&](int s) -> const GpAnX * { return xs ? xs + s : nullptr; };
    if (P.smallKernel) {
        launch_small<4, 4, AV1G_BLOCK_4X4>(B, P, L, n(AV1G_BLOCK_4X4), st, bd, x(AV1G_BLOCK_4X4));
        launch_small<4, 8, AV1G_BLOCK_4X8>(B, P, L, n(AV1G_BLOCK_4X8), st, bd, x(AV1G_BLOCK_4X8));
        launch_small<8, 4, AV1G_BLOCK_8X4>(B, P, L, n(AV1G_BLOCK_8X4), st, bd, x(AV1G_BLOCK_8X4));
        launch_small<8, 8, AV1G_BLOCK_8X8>(B, P, L, n(AV1G_BLOCK_8X8), st, bd, x(AV1G_BLOCK_8X8));
        launch_small<4, 16, AV1G_BLOCK_4X16>(B, P, L, n(AV1G_BLOCK_4X16), st, bd, x(AV1G_BLOCK_4X16));
        launch_small<16, 4, AV1G_BLOCK_16X4>(B, P, L, n(AV1G_BLOCK_16X4), st, bd, x(AV1G_BLOCK_16X4));
        return;
    }
    launch_an<4, 4, AV1G_BLOCK_4X4, 4>(B, P, L, n(AV1G_BLOCK_4X4), st, bd, x(AV1G_BLOCK_4X4));
    launch_an<4, 8, AV1G_BLOCK_4X8, 4>(B, P, L, n(AV1G_BLOCK_4X8), st, bd, x(AV1G_BLOCK_4X8));
    launch_an<8, 4, AV1G_BLOCK_8X4, 4>(B, P, L, n(AV1G_BLOCK_8X4), st, bd, x(AV1G_BLOCK_8X4));
    launch_an<8, 8, AV1G_BLOCK_8X8, 4>(B, P, L, n(AV1G_BLOCK_8X8), st, bd, x(AV1G_BLOCK_8X8));
    launch_an<4, 16, AV1G_BLOCK_4X16, 4>(B, P, L, n(AV1G_BLOCK_4X16), st, bd, x(AV1G_BLOCK_4X16));
    launch_an<16, 4, AV1G_BLOCK_16X4, 4>(B, P, L, n(AV1G_BLOCK_16X4), st, bd, x(AV1G_BLOCK_16X4));
}
