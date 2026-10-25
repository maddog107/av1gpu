// closed-loop reconstruction kernel.
#include "kernels.h"
#include "dev_common.cuh"
#include "k_recon.cuh"

GP_DEFINE_TABLE_UPLOAD(gp_upload_tables_recon)

// W5/W3: bytes of k_recon global scratch per CTA (B.bigScratch = max CTAs per launch x this)
size_t gp_recon_scratch_bytes() { return RC_CTA_BIG_BYTES; }

void gp_launch_recon(const GpBatch &B, const GpParams &P, const uint32_t *dSbList, uint32_t n, cudaStream_t st) {
    if (!n) return;
    k_recon<<<n, RC_NT, 0, st>>>(B, P, dSbList);
}
