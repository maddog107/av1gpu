// device-resident DecisionRecords of a batch (output of k_pack) and their GPU entropy coding
// through av1core's av1g_gpu::run_batch_device (entropy_dev.cu).
#ifndef GP_ENTROPY_DEV_H
#define GP_ENTROPY_DEV_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <string>
#include <vector>
#include "av1g/decisions.h"

struct GpDevRecords {
    std::vector<av1g_frame_params> fp;   // per frame
    std::vector<uint32_t> blk0, txb0;    // [nFrames + 1] frame bases in blocks / txb
    std::vector<uint64_t> coef0;         // [nFrames + 1] frame bases in coeffs
    // device arrays (decode order, frames back to back), all inside `pool`
    av1g_block *blocks = nullptr;
    uint32_t *blockTxbRel = nullptr;     // frame-relative first txb of each block (no trailing entry)
    av1g_txb *txb = nullptr;             // coeff_off frame-relative
    int16_t *coeffs = nullptr;
    void *pool = nullptr;                // cudaMallocAsync'ed; free with gp_dev_records_free
};

// per-thread / per-stream scratch for run_batch_device (grows on demand, reused across calls)
struct GpEntScratch {
    void *p = nullptr;
    size_t bytes = 0;
    uint64_t symbolHint = 0;             // 0 = av1core default capacity
};

// Entropy code all frames: tileBytes[frame][tile] = tile payload.  Synchronizes `st` (only).
int gp_entropy_dev(const GpDevRecords &R, cudaStream_t st, GpEntScratch &scr,
                   std::vector<std::vector<std::vector<uint8_t>>> &tileBytes, double *gpuMs, std::string &err);
void gp_dev_records_free(GpDevRecords &R, cudaStream_t st);
void gp_ent_scratch_free(GpEntScratch &s, cudaStream_t st);

#endif
