// open-loop analysis kernels (32x64, 64x32, 64x64, 16x64, 64x16).
#include "kernels.h"
#include "dev_common.cuh"
#include "k_analysis.cuh"

GP_DEFINE_TABLE_UPLOAD(gp_upload_tables_an_d)

template <int W, int H, int BS, int NW>
static void launch_an(const GpBatch &B, const GpParams &P, uint32_t n, cudaStream_t st, int bd) {
    if (!n) return;
    uint32_t grid = (n + NW - 1) / NW;
    // non-persistent grid: short CTAs let the hardware scheduler interleave other streams (recon wavefronts)
    if (grid > (1u << 30)) grid = 1u << 30;
    if (bd == 10) k_analysis<W, H, BS, NW, 10><<<grid, NW * 32, 0, st>>>(B, P, n);
    else k_analysis<W, H, BS, NW, 8><<<grid, NW * 32, 0, st>>>(B, P, n);
}

void gp_launch_an_d(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st, int bd) {
    // 64x64 / 64x32: one warp per CTA.  Their shared memory per warp (17.4 / 13.3 KB) limits the CTAs per SM, so with
    // two warps per CTA an L4 SM held 2 / 3 CTAs (4 / 6 warps); with one, 5 / 7 warps.  Each warp still analyses one
    // block, the same way.
    launch_an<32, 64, AV1G_BLOCK_32X64, 2>(B, P, shTotal[AV1G_BLOCK_32X64], st, bd);
    launch_an<64, 32, AV1G_BLOCK_64X32, 1>(B, P, shTotal[AV1G_BLOCK_64X32], st, bd);
    launch_an<64, 64, AV1G_BLOCK_64X64, 1>(B, P, shTotal[AV1G_BLOCK_64X64], st, bd);
    launch_an<16, 64, AV1G_BLOCK_16X64, 2>(B, P, shTotal[AV1G_BLOCK_16X64], st, bd);
    launch_an<64, 16, AV1G_BLOCK_64X16, 2>(B, P, shTotal[AV1G_BLOCK_64X16], st, bd);
}
