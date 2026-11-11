// open-loop analysis kernels (32x64, 64x32, 64x64, 16x64, 64x16).
#include "kernels.h"
#include "dev_common.cuh"
#include "k_analysis.cuh"

GP_DEFINE_TABLE_UPLOAD(gp_upload_tables_an_d)

// the staged analysis' MODE 1 kernels (k_an_x.cu), for a shape whose GpAnX is on
void gp_launch_an_x(const GpBatch &B, const GpParams &P, int bs, uint32_t n, const GpLive &L, cudaStream_t st, int bd, const GpAnX &x);

// x (may be null): the shape's options of the staged analysis; its MODE 1 kernel (k_an_x.cu) when it sets any
template <int W, int H, int BS, int NW>
static void launch_an(const GpBatch &B, const GpParams &P, uint32_t n, cudaStream_t st, int bd, const GpAnX *x) {
    if (!n) return;
    if constexpr (an_staged_shape(W, H))   // (16x64, 64x16 only)
        if (an_x_on(x)) { gp_launch_an_x(B, P, BS, n, GpLive{}, st, bd, *x); return; }
    uint32_t grid = (n + NW - 1) / NW;
    // non-persistent grid: short CTAs let the hardware scheduler interleave other streams (recon wavefronts)
    if (grid > (1u << 30)) grid = 1u << 30;
    if (bd == 10) k_analysis<W, H, BS, NW, 10><<<grid, NW * 32, 0, st>>>(B, P, n, GpLive{}, GpAnX{});
    else k_analysis<W, H, BS, NW, 8><<<grid, NW * 32, 0, st>>>(B, P, n, GpLive{}, GpAnX{});
}

// xs (null: none): per shape, the options of the staged analysis; mask: the shapes to launch (AV1G_BLOCK_* bits)
void gp_launch_an_d(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st, int bd, const GpAnX *xs,
                    uint32_t mask) {
    auto n = [&](int s) -> uint32_t { return (mask >> s) & 1 ? shTotal[s] : 0; };
    auto x = [&](int s) -> const GpAnX * { return xs ? xs + s : nullptr; };
    // 64x64 / 64x32: one warp per CTA.  Their shared memory per warp limits the CTAs per SM: two warps per CTA held fewer
    // warps per SM than one (on an L4, at the former 17.4 / 13.3 KB: 4 / 6 warps against 5 / 7).  64x64 and 32x64 transform
    // in place (ASmem::cm): 64x64 needs 9.4 KB per warp, 9 CTAs per SM on 100 KB of shared memory; 32x64 (two warps)
    // 14.4 KB per CTA, 6 CTAs; 64x32 still 13.3 KB, 7 CTAs.  Each warp analyses one block.
    launch_an<32, 64, AV1G_BLOCK_32X64, 2>(B, P, n(AV1G_BLOCK_32X64), st, bd, x(AV1G_BLOCK_32X64));
    launch_an<64, 32, AV1G_BLOCK_64X32, 1>(B, P, n(AV1G_BLOCK_64X32), st, bd, x(AV1G_BLOCK_64X32));
    launch_an<64, 64, AV1G_BLOCK_64X64, 1>(B, P, n(AV1G_BLOCK_64X64), st, bd, x(AV1G_BLOCK_64X64));
    launch_an<16, 64, AV1G_BLOCK_16X64, 2>(B, P, n(AV1G_BLOCK_16X64), st, bd, x(AV1G_BLOCK_16X64));
    launch_an<64, 16, AV1G_BLOCK_64X16, 2>(B, P, n(AV1G_BLOCK_64X16), st, bd, x(AV1G_BLOCK_64X16));
}
