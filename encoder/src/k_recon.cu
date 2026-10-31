// closed-loop reconstruction kernel.
#include "kernels.h"
#include "dev_common.cuh"
#include "k_recon.cuh"

GP_DEFINE_TABLE_UPLOAD(gp_upload_tables_recon)

// W5/W3: bytes of k_recon global scratch per CTA (B.bigScratch = max CTAs per launch x this)
size_t gp_recon_scratch_bytes() { return RC_CTA_BIG_BYTES; }

// one wavefront step: dSbList[0, n420) with the 4:2:0 instantiation, then dSbList[n420, n420 + n444) with the 4:4:4 one
// (the same stream: the second launch reuses the per-CTA scratch slots after the first)
void gp_launch_recon(const GpBatch &B, const GpParams &P, const uint32_t *dSbList, uint32_t n420, uint32_t n444, cudaStream_t st) {
    if (n420) k_recon<1><<<n420, RC_NT, 0, st>>>(B, P, dSbList);
    if (n444) k_recon<0><<<n444, RC_NT, 0, st>>>(B, P, dSbList + n420);
}
