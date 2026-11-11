// The staged analysis (opt-in: --topdown / --prune-41, an_stage.h): its MODE 1 analysis kernels (the listed shapes'),
// the block lists (k_an_list), the DP stages (k_dp<BD, true>) and the host sequence of stages.  A module of its own:
// its tables are uploaded by gp_staged_prepare (k_misc.cu) only for an encoder with a switch on, so a process without
// the switches never loads this code (lazy module loading, the default outside batch mode).
#include "kernels.h"
#include "dev_common.cuh"
#include "k_analysis.cuh"
#include "k_analysis_small.cuh"
#include "k_an_list.cuh"
#include "k_dp.cuh"
#include <algorithm>
#include <cstring>

GP_DEFINE_TABLE_UPLOAD(gp_upload_tables_an_x)

// the default launchers (k_an_a.cu .. k_an_d.cu, k_an_uv.cu): a shape whose GpAnX is on goes to gp_launch_an_x
void gp_launch_an_uv(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st, int bd);
void gp_launch_an_a(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st, int bd, const GpLive &L,
                    const GpAnX *xs, uint32_t mask);
void gp_launch_an_b(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st, int bd, const GpLive &L,
                    const GpAnX *xs, uint32_t mask);
void gp_launch_an_c(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st, int bd, const GpAnX *xs,
                    uint32_t mask);
void gp_launch_an_d(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st, int bd, const GpAnX *xs,
                    uint32_t mask);

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

static void gp_launch_an_list(const GpBatch &B, const GpParams &P, const GpAnListStage &L, cudaStream_t st, int bd) {
    uint32_t mx = 0;
    for (int k = 0; k < L.n; k++) mx = L.s[k].total > mx ? L.s[k].total : mx;
    if (!mx) return;
    const dim3 grid((mx + 127) / 128, (unsigned)L.n);
    if (bd == 10) k_an_list<10><<<grid, 128, 0, st>>>(B, P, L);
    else k_an_list<8><<<grid, 128, 0, st>>>(B, P, L);
}

static void gp_launch_dp_stage(const GpBatch &B, const GpParams &P, const GpDpStage &S, cudaStream_t st, int bd) {
    if (!B.nSb) return;
    if (bd == 10) k_dp<10, true><<<(B.nSb + 63) / 64, 64, 0, st>>>(B, P, S);
    else k_dp<8, true><<<(B.nSb + 63) / 64, 64, 0, st>>>(B, P, S);
}

// The MODE 1 kernels, with the default launchers' kernel and launch shape for each listed shape (keep them in step:
// k_an_a.cu launch_small, k_an_b.cu / k_an_d.cu launch_an; n is the caller's, after an_units)
template <int W, int H, int BS, int NW>
static void launch_x_warp(const GpBatch &B, const GpParams &P, uint32_t n, const GpLive &L, cudaStream_t st, int bd, const GpAnX &x) {
    uint32_t grid = (n + NW - 1) / NW;
    if (grid > (1u << 30)) grid = 1u << 30;
    if (bd == 10) k_analysis<W, H, BS, NW, 10, 1><<<grid, NW * 32, 0, st>>>(B, P, n, L, x);
    else k_analysis<W, H, BS, NW, 8, 1><<<grid, NW * 32, 0, st>>>(B, P, n, L, x);
}
template <int W, int H, int BS>
static void launch_x_small(const GpBatch &B, const GpParams &P, uint32_t n, const GpLive &L, cudaStream_t st, int bd, const GpAnX &x) {
    const uint32_t grid = (n + 127) / 128;
    if constexpr (W == 4 && H <= 8) {   // 4x4, 4x8: 6 CTAs per SM (k_analysis_small.cuh)
        if (bd == 10) k_analysis_small6<W, H, BS, 10, 1><<<grid, 128, 0, st>>>(B, P, n, L, x);
        else k_analysis_small6<W, H, BS, 8, 1><<<grid, 128, 0, st>>>(B, P, n, L, x);
    } else {
        if (bd == 10) k_analysis_small<W, H, BS, 10, 1><<<grid, 128, 0, st>>>(B, P, n, L, x);
        else k_analysis_small<W, H, BS, 8, 1><<<grid, 128, 0, st>>>(B, P, n, L, x);
    }
}

// the MODE 1 analysis of shape bs (an_staged_shape; the shapes up to 64 pixels in their small-kernel version, as
// an_stage_of keeps them in stage 1 with --small-kernel 0)
void gp_launch_an_x(const GpBatch &B, const GpParams &P, int bs, uint32_t n, const GpLive &L, cudaStream_t st, int bd, const GpAnX &x) {
    switch (bs) {
    case AV1G_BLOCK_4X4: launch_x_small<4, 4, AV1G_BLOCK_4X4>(B, P, n, L, st, bd, x); break;
    case AV1G_BLOCK_4X8: launch_x_small<4, 8, AV1G_BLOCK_4X8>(B, P, n, L, st, bd, x); break;
    case AV1G_BLOCK_8X4: launch_x_small<8, 4, AV1G_BLOCK_8X4>(B, P, n, L, st, bd, x); break;
    case AV1G_BLOCK_8X8: launch_x_small<8, 8, AV1G_BLOCK_8X8>(B, P, n, L, st, bd, x); break;
    case AV1G_BLOCK_4X16: launch_x_small<4, 16, AV1G_BLOCK_4X16>(B, P, n, L, st, bd, x); break;
    case AV1G_BLOCK_16X4: launch_x_small<16, 4, AV1G_BLOCK_16X4>(B, P, n, L, st, bd, x); break;
    case AV1G_BLOCK_8X16: launch_x_warp<8, 16, AV1G_BLOCK_8X16, 4>(B, P, n, L, st, bd, x); break;
    case AV1G_BLOCK_16X8: launch_x_warp<16, 8, AV1G_BLOCK_16X8, 4>(B, P, n, L, st, bd, x); break;
    case AV1G_BLOCK_8X32: launch_x_warp<8, 32, AV1G_BLOCK_8X32, 4>(B, P, n, L, st, bd, x); break;
    case AV1G_BLOCK_32X8: launch_x_warp<32, 8, AV1G_BLOCK_32X8, 4>(B, P, n, L, st, bd, x); break;
    case AV1G_BLOCK_16X64: launch_x_warp<16, 64, AV1G_BLOCK_16X64, 2>(B, P, n, L, st, bd, x); break;
    case AV1G_BLOCK_64X16: launch_x_warp<64, 16, AV1G_BLOCK_64X16, 2>(B, P, n, L, st, bd, x); break;
    default: break;   // (not listed: an_staged_shape)
    }
}

void gp_launch_analysis_staged(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st, int bd,
                               const GpLive *live, const GpAnRun &R) {
    const bool staged = R.topdown || R.gateMul > 0.f;
    // the k_live16 cells serve the cell-wise shapes only without the top-down stages (which list those shapes' blocks)
    const GpLive L0 = live && !R.topdown ? *live : GpLive{};
    GpAnX xs[GP_NSHAPES];
    memset(xs, 0, sizeof xs);
    uint32_t stage[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int s = 0; s < GP_NSHAPES; s++) {
        if (!shTotal[s]) continue;
        const int w = 4 * AV1G_T(Num_4x4_Blocks_Wide)[s], h = 4 * AV1G_T(Num_4x4_Blocks_High)[s];
        const int sg = an_stage_of(R.topdown, R.gateMul, P.smallKernel, w, h);
        stage[sg] |= 1u << s;
        if (sg >= 2) { xs[s].list = R.list[s]; xs[s].count = R.count + s; }
    }
    if (staged) cudaMemsetAsync(R.count, 0, GP_NSHAPES * sizeof(uint32_t), st);
    uint32_t done = 0;
    // a stage's shapes (large to small, as gp_launch_analysis), then their chroma term and palette candidates, which the
    // next DP stage sees
    auto launch = [&](uint32_t mask, const GpLive &L) {
        if (!mask) return;
        gp_launch_an_d(B, P, shTotal, st, bd, xs, mask);
        gp_launch_an_c(B, P, shTotal, st, bd, xs, mask);
        gp_launch_an_b(B, P, shTotal, st, bd, L, xs, mask);
        gp_launch_an_a(B, P, shTotal, st, bd, L, xs, mask);
        uint32_t sub[GP_NSHAPES];
        for (int s = 0; s < GP_NSHAPES; s++) sub[s] = (mask >> s) & 1 ? shTotal[s] : 0;
        if (P.uvInDp) gp_launch_an_uv(B, P, sub, st, bd);
        gp_launch_pal_analysis(B, P, sub, st, bd);
        done |= mask;
    };
    // a DP stage over the shapes done, then the block lists of the next stage's shapes.  The map marks the cells under
    // leaves whose smaller side is the smallest of the shapes done (16 for the 16x16 map, 8 for the 8x8 one in every
    // preset; larger when --shapes leaves out a group, so that the next group still refines where the DP went finest)
    auto next = [&](uint32_t mask, uint8_t *core, int cell, uint8_t *gate) {
        int side = 64;
        for (int s = 0; s < GP_NSHAPES; s++)
            if ((done >> s) & 1)
                side = std::min(side, std::min(4 * (int)AV1G_T(Num_4x4_Blocks_Wide)[s], 4 * (int)AV1G_T(Num_4x4_Blocks_High)[s]));
        GpDpStage S;
        memset(&S, 0, sizeof S);
        S.shapeMask = done; S.core = core; S.coreCell = cell; S.coreSide = std::max(cell, side); S.gate = gate; S.gateMul = R.gateMul;
        gp_launch_dp_stage(B, P, S, st, bd);
        GpAnListStage Ls;
        memset(&Ls, 0, sizeof Ls);
        Ls.core16 = R.core16; Ls.core8 = R.core8; Ls.gate = R.gate; Ls.neigh = 1;
        for (int s = 0; s < GP_NSHAPES; s++) {
            if (!((mask >> s) & 1)) continue;
            const int w = 4 * AV1G_T(Num_4x4_Blocks_Wide)[s], h = 4 * AV1G_T(Num_4x4_Blocks_High)[s], m = std::min(w, h);
            GpAnListShape &sh = Ls.s[Ls.n++];
            sh.bs = s; sh.total = shTotal[s]; sh.list = R.list[s]; sh.count = R.count + s; sh.flat = 1;
            sh.cell = R.topdown ? (m == 8 ? 16 : m == 4 ? 8 : 0) : 0;
            sh.gate = gate != nullptr;
        }
        gp_launch_an_list(B, P, Ls, st, bd);
    };
    // stages in the order of an_stage_of: 1, 5, 2, 6, 3, 7 with the top-down stages, else 1, 4
    launch(stage[1], L0);
    // (a gated group without the group before it in the shape set maps the cells its blocks need in the gating DP)
    const int order[] = {5, 2, 6, 3, 7, 4};
    bool have16 = false, have8 = false;
    for (int sg : order) {
        if (!stage[sg]) continue;
        uint8_t *core = nullptr;
        int cell = 16;
        if (sg == 2 || (sg == 6 && !have16)) { core = R.core16; have16 = true; }
        if (sg == 3 || (sg == 7 && !have8)) { core = R.core8; cell = 8; have8 = true; }
        next(stage[sg], core, cell, sg >= 4 ? R.gate : nullptr);
        launch(stage[sg], GpLive{});
    }
}
