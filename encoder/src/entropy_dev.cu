// GPU entropy coding of DecisionRecords that are already on the GPU (output of k_pack),
// through av1core's device entry point av1g_gpu::run_batch_device (stream-ordered, no allocations or
// syncs inside, byte-identical to the CPU writer).  This file only manages the scratch buffer, the
// symbol-capacity retry and the download of the packed tile payloads.
#include "entropy_dev.h"
#include "pinned.h"
#include "av1g/gpu_entropy.h"

#include <cuda_runtime.h>
#include <algorithm>
#include <cstring>

#define ECK(x)                                                                                     \
    do {                                                                                           \
        cudaError_t e_ = (x);                                                                      \
        if (e_ != cudaSuccess) { err = std::string("CUDA: ") + cudaGetErrorString(e_) + " at " #x; return -1; } \
    } while (0)

int gp_entropy_dev(const GpDevRecords &R, cudaStream_t st, GpEntScratch &scr,
                   std::vector<std::vector<std::vector<uint8_t>>> &tileBytes, double *gpuMs, std::string &err) {
    const uint32_t nF = (uint32_t)R.fp.size();
    std::vector<av1g_gpu::DevFrame> frames(nF);
    for (uint32_t f = 0; f < nF; f++) {
        frames[f].fp = R.fp[f];
        frames[f].blockBase = R.blk0[f];
        frames[f].nBlocks = R.blk0[f + 1] - R.blk0[f];
        frames[f].txbBase = R.txb0[f];
        frames[f].nTxb = R.txb0[f + 1] - R.txb0[f];
        frames[f].coeffBase = R.coef0[f];
    }
    av1g_gpu::DevBatch b;
    b.frames = frames.data();
    b.nFrames = nF;
    b.blocks = R.blocks;
    b.blockTxb = R.blockTxbRel;
    b.txb = R.txb;
    b.coeffs = R.coeffs;
    b.symbolCapacity = scr.symbolHint;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    av1g_gpu::DevResult r;
    std::vector<uint8_t> res;
    for (int attempt = 0; attempt < 2; attempt++) {
        const size_t need = av1g_gpu::run_batch_device_scratch_bytes(b);
        if (need > scr.bytes) {
            if (scr.p) cudaFreeAsync(scr.p, st);
            scr.p = nullptr;
            scr.bytes = 0;
            ECK(cudaMallocAsync(&scr.p, need, st));
            scr.bytes = need;
        }
        cudaEventRecord(e0, st);
        if (av1g_gpu::run_batch_device(b, st, scr.p, need, r, err)) return -1;
        cudaEventRecord(e1, st);
        res.resize(r.resultBytes);
        {
            GpPinnedStage &PS = gp_pinned();
            if (PS.stage(res.data(), r.result, r.resultBytes, st) || PS.flush(st)) { err = "pinned download failed"; return -1; }
        }
        av1g_gpu::DevStatus status;
        memcpy(&status, res.data(), sizeof status);
        if (status.error == av1g_gpu::AV1G_GPU_ERR_SYMBOLS && attempt == 0) {
            b.symbolCapacity = status.symbols + status.symbols / 8;   // rerun with the exact need (+12 %)
            scr.symbolHint = std::max(scr.symbolHint, b.symbolCapacity);
            continue;
        }
        if (status.error) { err = "run_batch_device error " + std::to_string(status.error); return -1; }
        break;
    }
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    if (gpuMs) *gpuMs = ms;
    const uint64_t *off = (const uint64_t *)(res.data() + sizeof(av1g_gpu::DevStatus));
    std::vector<uint8_t> bytes(off[r.nTiles]);
    {
        GpPinnedStage &PS = gp_pinned();
        if (PS.stage(bytes.data(), r.bytes, bytes.size(), st) || PS.flush(st)) { err = "pinned download failed"; return -1; }
    }
    tileBytes.assign(nF, {});
    for (uint32_t f = 0; f < nF; f++)
        for (uint32_t t = r.frameTile0[f]; t < r.frameTile0[f + 1]; t++)
            tileBytes[f].emplace_back(bytes.begin() + off[t], bytes.begin() + off[t + 1]);
    return 0;
}

void gp_dev_records_free(GpDevRecords &R, cudaStream_t st) {
    if (R.pool) cudaFreeAsync(R.pool, st);
    R.pool = nullptr;
}

void gp_ent_scratch_free(GpEntScratch &s, cudaStream_t st) {
    if (s.p) cudaFreeAsync(s.p, st);
    s.p = nullptr;
    s.bytes = 0;
}
