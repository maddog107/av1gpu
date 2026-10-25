// open-loop analysis kernels (8x16, 16x8, 16x16, 8x32, 32x8).
#include "kernels.h"
#include "dev_common.cuh"
#include "k_analysis.cuh"

GP_DEFINE_TABLE_UPLOAD(gp_upload_tables_an_b)

template <int W, int H, int BS, int NW>
static void launch_an(const GpBatch &B, const GpParams &P, uint32_t n, cudaStream_t st) {
    if (!n) return;
    uint32_t grid = (n + NW - 1) / NW;
    // non-persistent grid: short CTAs let the hardware scheduler interleave other streams (recon wavefronts)
    if (grid > (1u << 30)) grid = 1u << 30;
    k_analysis<W, H, BS, NW><<<grid, NW * 32, 0, st>>>(B, P, n);
}

void gp_launch_an_b(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st) {
    launch_an<8, 16, AV1G_BLOCK_8X16, 4>(B, P, shTotal[AV1G_BLOCK_8X16], st);
    launch_an<16, 8, AV1G_BLOCK_16X8, 4>(B, P, shTotal[AV1G_BLOCK_16X8], st);
    launch_an<16, 16, AV1G_BLOCK_16X16, 4>(B, P, shTotal[AV1G_BLOCK_16X16], st);
    launch_an<8, 32, AV1G_BLOCK_8X32, 4>(B, P, shTotal[AV1G_BLOCK_8X32], st);
    launch_an<32, 8, AV1G_BLOCK_32X8, 4>(B, P, shTotal[AV1G_BLOCK_32X8], st);
}
