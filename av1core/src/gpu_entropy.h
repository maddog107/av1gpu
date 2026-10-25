// av1core GPU entropy: GPU symbolization + entropy coding of a batch of DecisionRecords.
// Device entry point (records already on the GPU): include/av1g/gpu_entropy.h.
// This header adds the host-convenience API used by tests/benchmarks (uploads records,
// calls run_batch_device, downloads).
#ifndef AV1G_GPU_ENTROPY_H
#define AV1G_GPU_ENTROPY_H

#include <cstdint>
#include <string>
#include <vector>
#include "av1g/decisions.h"
#include "av1g/gpu_entropy.h"

namespace av1g_gpu {

struct BatchResult {
    // [frame][tile] -> entropy-coded tile payload
    std::vector<std::vector<std::vector<uint8_t>>> tileBytes;
    // [frame][tile] -> packed symbol stream (only if keepSyms)
    std::vector<std::vector<std::vector<uint32_t>>> tileSyms;
    uint64_t totalSyms = 0;
    uint32_t nTiles = 0, nSb = 0;
    uint64_t nBlocks = 0, nTxb = 0;
    // GPU timings (ms, cudaEvent): upload, prep, maps, count, scan, emit, entropy, pack, download
    double msUpload = 0, msPrep = 0, msMaps = 0, msCount = 0, msScan = 0, msEmit = 0, msEntropy = 0, msPack = 0,
           msDownload = 0;
    size_t deviceBytes = 0;
};

// Frames must be valid records (av1g_validate) with layout.  Pure symbol coding:
// no reconstruction.  Returns 0 on success.  (Uploads, then run_batch_device.)
int run_batch(const std::vector<const av1g_decision_record *> &frames, bool keepSyms, BatchResult &res, std::string &err);

// Entropy-code given per-tile symbol streams on the GPU only (kernel per g_entropy_mode).
int code_streams(const std::vector<const std::vector<uint32_t> *> &streams, const std::vector<int> &qidx,
                 std::vector<std::vector<uint8_t>> &out, double &ms, std::string &err);

// mode 2 only: true (default) per-tile CDF state in shared memory; false: in global memory
extern bool g_entropy_smem;

}  // namespace av1g_gpu
#endif
