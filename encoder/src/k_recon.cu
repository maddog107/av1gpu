// closed-loop reconstruction kernel: the 8-bit 4:2:0 instantiation (the default) and the launcher.
// One k_recon instantiation per translation unit (k_recon_444_8.cu, k_recon_420_10.cu, k_recon_444_10.cu): in
// one module the kernels share their non-inlined helpers (rc_luma<BD>, cta_inverse_add<BD>, called from both
// layouts), which ptxas then compiles with more spills (rc_luma<8>: 15.4 KB of spill loads instead of 12.9 KB);
// that made the 4:2:0 closed loop ~2 % slower on an L4.
#include "kernels.h"
#include "dev_common.cuh"
#include "k_recon.cuh"

GP_DEFINE_TABLE_UPLOAD(gp_upload_tables_recon)
void gp_launch_recon_444_8(const GpBatch &B, const GpParams &P, const uint32_t *dSbList, uint32_t n, cudaStream_t st);
void gp_launch_recon_420_10(const GpBatch &B, const GpParams &P, const uint32_t *dSbList, uint32_t n, cudaStream_t st);
void gp_launch_recon_444_10(const GpBatch &B, const GpParams &P, const uint32_t *dSbList, uint32_t n, cudaStream_t st);

// W5/W3: bytes of k_recon global scratch per CTA (B.bigScratch = max CTAs per launch x this)
size_t gp_recon_scratch_bytes() { return RC_CTA_BIG_BYTES; }

// one wavefront step: dSbList[0, n420) with the 4:2:0 instantiation, then dSbList[n420, n420 + n444) with the 4:4:4 one
// (the same stream: the second launch reuses the per-CTA scratch slots after the first)
void gp_launch_recon(const GpBatch &B, const GpParams &P, const uint32_t *dSbList, uint32_t n420, uint32_t n444, cudaStream_t st,
                     int bd) {
    if (bd == 10) {   // 10-bit batches
        if (n420) gp_launch_recon_420_10(B, P, dSbList, n420, st);
        if (n444) gp_launch_recon_444_10(B, P, dSbList + n420, n444, st);
        return;
    }
    if (n420) k_recon<1, 8><<<n420, RC_NT, 0, st>>>(B, P, dSbList);
    if (n444) gp_launch_recon_444_8(B, P, dSbList + n420, n444, st);
}
