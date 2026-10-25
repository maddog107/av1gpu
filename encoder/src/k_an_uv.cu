// chroma term of the open-loop analysis (GpParams::uvInDp), all shapes.
#include "kernels.h"
#include "dev_common.cuh"
#include "k_analysis_uv.cuh"

GP_DEFINE_TABLE_UPLOAD(gp_upload_tables_an_uv)

template <int W, int H, int BS>
static void launch_uv(const GpBatch &B, const GpParams &P, uint32_t n, cudaStream_t st) {
    if (!n) return;
    uint32_t grid = (n + 3) / 4;
    // non-persistent grid: short CTAs let the hardware scheduler interleave other streams (recon wavefronts)
    if (grid > (1u << 30)) grid = 1u << 30;
    k_analysis_uv<W, H, BS, 4><<<grid, 128, 0, st>>>(B, P, n);
}

void gp_launch_an_uv(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st) {
    launch_uv<4, 4, AV1G_BLOCK_4X4>(B, P, shTotal[AV1G_BLOCK_4X4], st);
    launch_uv<4, 8, AV1G_BLOCK_4X8>(B, P, shTotal[AV1G_BLOCK_4X8], st);
    launch_uv<8, 4, AV1G_BLOCK_8X4>(B, P, shTotal[AV1G_BLOCK_8X4], st);
    launch_uv<8, 8, AV1G_BLOCK_8X8>(B, P, shTotal[AV1G_BLOCK_8X8], st);
    launch_uv<8, 16, AV1G_BLOCK_8X16>(B, P, shTotal[AV1G_BLOCK_8X16], st);
    launch_uv<16, 8, AV1G_BLOCK_16X8>(B, P, shTotal[AV1G_BLOCK_16X8], st);
    launch_uv<16, 16, AV1G_BLOCK_16X16>(B, P, shTotal[AV1G_BLOCK_16X16], st);
    launch_uv<16, 32, AV1G_BLOCK_16X32>(B, P, shTotal[AV1G_BLOCK_16X32], st);
    launch_uv<32, 16, AV1G_BLOCK_32X16>(B, P, shTotal[AV1G_BLOCK_32X16], st);
    launch_uv<32, 32, AV1G_BLOCK_32X32>(B, P, shTotal[AV1G_BLOCK_32X32], st);
    launch_uv<32, 64, AV1G_BLOCK_32X64>(B, P, shTotal[AV1G_BLOCK_32X64], st);
    launch_uv<64, 32, AV1G_BLOCK_64X32>(B, P, shTotal[AV1G_BLOCK_64X32], st);
    launch_uv<64, 64, AV1G_BLOCK_64X64>(B, P, shTotal[AV1G_BLOCK_64X64], st);
    launch_uv<4, 16, AV1G_BLOCK_4X16>(B, P, shTotal[AV1G_BLOCK_4X16], st);
    launch_uv<16, 4, AV1G_BLOCK_16X4>(B, P, shTotal[AV1G_BLOCK_16X4], st);
    launch_uv<8, 32, AV1G_BLOCK_8X32>(B, P, shTotal[AV1G_BLOCK_8X32], st);
    launch_uv<32, 8, AV1G_BLOCK_32X8>(B, P, shTotal[AV1G_BLOCK_32X8], st);
    launch_uv<16, 64, AV1G_BLOCK_16X64>(B, P, shTotal[AV1G_BLOCK_16X64], st);
    launch_uv<64, 16, AV1G_BLOCK_64X16>(B, P, shTotal[AV1G_BLOCK_64X16], st);
}
