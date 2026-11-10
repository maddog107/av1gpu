// Longest-first CTA order inside each wavefront step of k_recon (perf-wavefront idea 3).  A step lasts as long as its
// slowest superblock; when it has more superblocks than the GPU has k_recon slots, the ones dispatched last can end
// after it.  k_recon_order sorts each step's 4:2:0 and 4:4:4 groups (seg: their boundaries in the step lists) by
// descending leaf count (k_dp's B.leafCountW; a superblock takes about 207 us + 56 us per block), ties in list order,
// so the long superblocks start first.  It runs on the encoder stream after the partition DP: no host round trip.
// CTA order changes no output: each k_recon CTA writes only its superblock's slots and its own B.bigScratch slot (by
// blockIdx); pipeline.cpp's GP_RC_ORDER=0 (list order) checks that.  Its own translation unit, so the k_recon
// instantiations do not depend on it.
#include <cuda_runtime.h>
#include "gp.h"

#define RC_ORDER_MAX 6144   // ids + keys of one group in 48 KB of shared memory; a longer group keeps the list order

__global__ void k_recon_order(const uint16_t *leafCount, uint32_t *steps, const uint32_t *seg) {
    extern __shared__ uint32_t sm[];   // ids[n], keys[n]
    const uint32_t b = seg[blockIdx.x], n = seg[blockIdx.x + 1] - b;
    if (n < 2 || n > RC_ORDER_MAX) return;
    uint32_t *ids = sm, *keys = sm + n;
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {
        ids[i] = steps[b + i];
        keys[i] = leafCount[ids[i]];
    }
    __syncthreads();
    for (uint32_t i = threadIdx.x; i < n; i += blockDim.x) {   // rank = position in (key descending, list order)
        const uint32_t k = keys[i];
        uint32_t r = 0;
        for (uint32_t j = 0; j < n; j++) r += keys[j] > k || (keys[j] == k && j < i);
        steps[b + r] = ids[i];
    }
}

void gp_recon_order(const GpBatch &B, uint32_t *dSteps, const uint32_t *dSeg, int nSeg, uint32_t maxSeg, cudaStream_t st) {
    if (nSeg <= 0 || maxSeg < 2) return;
    const uint32_t n = maxSeg < RC_ORDER_MAX ? maxSeg : RC_ORDER_MAX;   // the longest group that is sorted
    k_recon_order<<<nSeg, 256, 2 * n * sizeof(uint32_t), st>>>(B.leafCountW, dSteps, dSeg);
}
