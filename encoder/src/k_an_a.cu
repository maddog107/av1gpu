// open-loop analysis kernels (4x4, 4x8, 8x4, 8x8, 4x16, 16x4).
#include "kernels.h"
#include "dev_common.cuh"
#include "k_analysis.cuh"
#include "k_analysis_small.cuh"

GP_DEFINE_TABLE_UPLOAD(gp_upload_tables_an_a)

template <int W, int H, int BS, int NW>
static void launch_an(const GpBatch &B, const GpParams &P, uint32_t n, cudaStream_t st, int bd) {
    if (!n) return;
    uint32_t grid = (n + NW - 1) / NW;
    // non-persistent grid: short CTAs let the hardware scheduler interleave other streams (recon wavefronts)
    if (grid > (1u << 30)) grid = 1u << 30;
    if (bd == 10) k_analysis<W, H, BS, NW, 10><<<grid, NW * 32, 0, st>>>(B, P, n);
    else k_analysis<W, H, BS, NW, 8><<<grid, NW * 32, 0, st>>>(B, P, n);
}

template <int W, int H, int BS>
static void launch_small(const GpBatch &B, const GpParams &P, uint32_t n, cudaStream_t st, int bd) {
    if (!n) return;
    if (bd == 10) k_analysis_small<W, H, BS, 10><<<(n + 127) / 128, 128, 0, st>>>(B, P, n);
    else k_analysis_small<W, H, BS, 8><<<(n + 127) / 128, 128, 0, st>>>(B, P, n);
}

void gp_launch_an_a(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st, int bd) {
    if (P.smallKernel) {
        launch_small<4, 4, AV1G_BLOCK_4X4>(B, P, shTotal[AV1G_BLOCK_4X4], st, bd);
        launch_small<4, 8, AV1G_BLOCK_4X8>(B, P, shTotal[AV1G_BLOCK_4X8], st, bd);
        launch_small<8, 4, AV1G_BLOCK_8X4>(B, P, shTotal[AV1G_BLOCK_8X4], st, bd);
        launch_small<8, 8, AV1G_BLOCK_8X8>(B, P, shTotal[AV1G_BLOCK_8X8], st, bd);
        launch_small<4, 16, AV1G_BLOCK_4X16>(B, P, shTotal[AV1G_BLOCK_4X16], st, bd);
        launch_small<16, 4, AV1G_BLOCK_16X4>(B, P, shTotal[AV1G_BLOCK_16X4], st, bd);
        return;
    }
    launch_an<4, 4, AV1G_BLOCK_4X4, 4>(B, P, shTotal[AV1G_BLOCK_4X4], st, bd);
    launch_an<4, 8, AV1G_BLOCK_4X8, 4>(B, P, shTotal[AV1G_BLOCK_4X8], st, bd);
    launch_an<8, 4, AV1G_BLOCK_8X4, 4>(B, P, shTotal[AV1G_BLOCK_8X4], st, bd);
    launch_an<8, 8, AV1G_BLOCK_8X8, 4>(B, P, shTotal[AV1G_BLOCK_8X8], st, bd);
    launch_an<4, 16, AV1G_BLOCK_4X16, 4>(B, P, shTotal[AV1G_BLOCK_4X16], st, bd);
    launch_an<16, 4, AV1G_BLOCK_16X4, 4>(B, P, shTotal[AV1G_BLOCK_16X4], st, bd);
}
