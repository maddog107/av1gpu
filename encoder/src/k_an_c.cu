// open-loop analysis kernels (16x32, 32x16, 32x32).
#include "kernels.h"
#include "dev_common.cuh"
#include "k_analysis.cuh"

GP_DEFINE_TABLE_UPLOAD(gp_upload_tables_an_c)

// x: unused (none of these shapes is listed by the staged analysis: an_staged_shape)
template <int W, int H, int BS, int NW>
static void launch_an(const GpBatch &B, const GpParams &P, uint32_t n, cudaStream_t st, int bd, const GpAnX *x) {
    static_assert(!an_staged_shape(W, H), "a listed shape needs its MODE 1 launch (k_an_x.cu)");
    (void)x;
    if (!n) return;
    uint32_t grid = (n + NW - 1) / NW;
    // non-persistent grid: short CTAs let the hardware scheduler interleave other streams (recon wavefronts)
    if (grid > (1u << 30)) grid = 1u << 30;
    if (bd == 10) k_analysis<W, H, BS, NW, 10><<<grid, NW * 32, 0, st>>>(B, P, n, GpLive{}, GpAnX{});
    else k_analysis<W, H, BS, NW, 8><<<grid, NW * 32, 0, st>>>(B, P, n, GpLive{}, GpAnX{});
}

// xs (null: none): per shape, the options of the staged analysis; mask: the shapes to launch (AV1G_BLOCK_* bits)
void gp_launch_an_c(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st, int bd, const GpAnX *xs,
                    uint32_t mask) {
    auto n = [&](int s) -> uint32_t { return (mask >> s) & 1 ? shTotal[s] : 0; };
    auto x = [&](int s) -> const GpAnX * { return xs ? xs + s : nullptr; };
    launch_an<16, 32, AV1G_BLOCK_16X32, 4>(B, P, n(AV1G_BLOCK_16X32), st, bd, x(AV1G_BLOCK_16X32));
    launch_an<32, 16, AV1G_BLOCK_32X16, 4>(B, P, n(AV1G_BLOCK_32X16), st, bd, x(AV1G_BLOCK_32X16));
    launch_an<32, 32, AV1G_BLOCK_32X32, 4>(B, P, n(AV1G_BLOCK_32X32), st, bd, x(AV1G_BLOCK_32X32));
}
