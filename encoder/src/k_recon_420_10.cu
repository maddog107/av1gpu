// closed-loop reconstruction kernel: the 10-bit 4:2:0 instantiation (one per translation unit, see k_recon.cu).
#include "kernels.h"
#include "dev_common.cuh"
#include "k_recon.cuh"

GP_DEFINE_TABLE_UPLOAD(gp_upload_tables_recon_420_10)

void gp_launch_recon_420_10(const GpBatch &B, const GpParams &P, const uint32_t *dSbList, uint32_t n, cudaStream_t st) {
    k_recon<1, 10><<<n, RC_NT, 0, st>>>(B, P, dSbList);
}
