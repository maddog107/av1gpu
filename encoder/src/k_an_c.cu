// open-loop analysis kernels (16x32, 32x16, 32x32).
#include "kernels.h"
#include "dev_common.cuh"
#include "k_analysis.cuh"

GP_DEFINE_TABLE_UPLOAD(gp_upload_tables_an_c)

template <int W, int H, int BS, int NW>
static void launch_an(const GpBatch &B, const GpParams &P, uint32_t n, cudaStream_t st) {
    if (!n) return;
    uint32_t grid = (n + NW - 1) / NW;
    // non-persistent grid: short CTAs let the hardware scheduler interleave other streams (recon wavefronts)
    if (grid > (1u << 30)) grid = 1u << 30;
    k_analysis<W, H, BS, NW><<<grid, NW * 32, 0, st>>>(B, P, n);
}

void gp_launch_an_c(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st) {
    launch_an<16, 32, AV1G_BLOCK_16X32, 4>(B, P, shTotal[AV1G_BLOCK_16X32], st);
    launch_an<32, 16, AV1G_BLOCK_32X16, 4>(B, P, shTotal[AV1G_BLOCK_32X16], st);
    launch_an<32, 32, AV1G_BLOCK_32X32, 4>(B, P, shTotal[AV1G_BLOCK_32X32], st);
}
