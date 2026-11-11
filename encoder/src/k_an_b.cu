// open-loop analysis kernels (8x16, 16x8, 16x16, 8x32, 32x8).
#include "kernels.h"
#include "dev_common.cuh"
#include "k_analysis.cuh"

GP_DEFINE_TABLE_UPLOAD(gp_upload_tables_an_b)

// x (may be null): the shape's options of the staged analysis; the MODE 1 kernel when it sets any (an_x_on)
template <int W, int H, int BS, int NW>
static void launch_an(const GpBatch &B, const GpParams &P, const GpLive &L, uint32_t n, cudaStream_t st, int bd, const GpAnX *x) {
    n = an_units<W, H>(L, n);   // 8x16, 16x8: a warp per block of the live cells
    if (!n) return;
    uint32_t grid = (n + NW - 1) / NW;
    // non-persistent grid: short CTAs let the hardware scheduler interleave other streams (recon wavefronts)
    if (grid > (1u << 30)) grid = 1u << 30;
    if (an_x_on(x)) {
        if (bd == 10) k_analysis<W, H, BS, NW, 10, 1><<<grid, NW * 32, 0, st>>>(B, P, n, L, *x);
        else k_analysis<W, H, BS, NW, 8, 1><<<grid, NW * 32, 0, st>>>(B, P, n, L, *x);
        return;
    }
    if (bd == 10) k_analysis<W, H, BS, NW, 10><<<grid, NW * 32, 0, st>>>(B, P, n, L, GpAnX{});
    else k_analysis<W, H, BS, NW, 8><<<grid, NW * 32, 0, st>>>(B, P, n, L, GpAnX{});
}

// xs (null: none): per shape, the options of the staged analysis; mask: the shapes to launch (AV1G_BLOCK_* bits)
void gp_launch_an_b(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st, int bd, const GpLive &L,
                    const GpAnX *xs, uint32_t mask) {
    auto n = [&](int s) -> uint32_t { return (mask >> s) & 1 ? shTotal[s] : 0; };
    auto x = [&](int s) -> const GpAnX * { return xs ? xs + s : nullptr; };
    launch_an<8, 16, AV1G_BLOCK_8X16, 4>(B, P, L, n(AV1G_BLOCK_8X16), st, bd, x(AV1G_BLOCK_8X16));
    launch_an<16, 8, AV1G_BLOCK_16X8, 4>(B, P, L, n(AV1G_BLOCK_16X8), st, bd, x(AV1G_BLOCK_16X8));
    launch_an<16, 16, AV1G_BLOCK_16X16, 4>(B, P, L, n(AV1G_BLOCK_16X16), st, bd, x(AV1G_BLOCK_16X16));
    launch_an<8, 32, AV1G_BLOCK_8X32, 4>(B, P, L, n(AV1G_BLOCK_8X32), st, bd, x(AV1G_BLOCK_8X32));
    launch_an<32, 8, AV1G_BLOCK_32X8, 4>(B, P, L, n(AV1G_BLOCK_32X8), st, bd, x(AV1G_BLOCK_32X8));
}
