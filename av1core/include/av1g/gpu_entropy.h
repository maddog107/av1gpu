// av1g/gpu_entropy.h - GPU symbolization + entropy coding of a batch of DecisionRecords whose
// arrays are already on the device (M4, device entry point).  Byte-identical to the CPU writer.
//
//   av1g_gpu::DevBatch b;                       // host struct
//   b.frames = table; b.nFrames = n;            // host table: fp + where each frame's arrays start
//   b.blocks/blockTxb/txb/coeffs = device arrays (frames concatenated, see DevFrame)
//   b.symbolCapacity = 0;                       // 0 = default bound (see below)
//   size_t need = av1g_gpu::run_batch_device_scratch_bytes(b);   // host only, no CUDA calls
//   ... caller allocates `need` bytes of device scratch once (reuse across calls) ...
//   av1g_gpu::DevResult r;
//   av1g_gpu::run_batch_device(b, stream, scratch, need, r, err);   // enqueue only: no sync, no
//                                                                  // allocation, all work on `stream`
//   // after `stream` reaches this point (event/sync of the caller's choice):
//   //   download r.resultBytes bytes from r.result  -> DevStatus + uint64 tileOffset[nTiles + 1]
//   //   status.error == 0  ->  tile i of the batch = r.bytes[tileOffset[i] .. tileOffset[i + 1])
//   //   frame f owns tiles r.frameTile0[f] .. r.frameTile0[f + 1] - 1 (tile raster order)
//   // assemble OBUs: sequence header + frame header (+ tile sizes) + payloads (Encoder::write_headers)
//
// Inputs (device, frames concatenated):
//   blocks[]    av1g_block, frame f at blocks[frames[f].blockBase .. + nBlocks), decode order
//   blockTxb[]  av1g_decision_record::block_txb WITHOUT the trailing entry: frame-relative index of
//               each block's first tx block; frame f at blockTxb[frames[f].blockBase ..]
//   txb[]       av1g_txb, frame f at txb[frames[f].txbBase .. + nTxb); coeff_off frame-relative
//   coeffs[]    int16 levels, frame f's coefficient i at coeffs[frames[f].coeffBase + i]
//   txb[].tx_type of chroma tx blocks and txb[].eob are ignored (derived), as in av1g_encode.
//   Frames must be stored in frame order and back to back in blocks/blockTxb and in txb
//   (frames[f+1].blockBase == frames[f].blockBase + frames[f].nBlocks, same for txb); the first
//   base may be non-zero; coeffBase is free.  Records are not modified.
//
// Concurrency: the call only enqueues work on `stream` (~20 kernels, 3 memsets; host side is
// O(frames + tiles)).  The scratch and the DevResult pointers belong to that call until the
// stream has passed it: to overlap batches use one scratch per batch in flight.  Needs ~12 KB of
// shared memory per tile and ~2 warps per tile while entropy coding, so it can share the GPU with
// the caller's other streams.
//
// Errors are reported on the device (DevStatus.error bit set; payloads must not be used):
//   AV1G_GPU_ERR_SYMBOLS   more symbols than symbolCapacity (DevStatus.symbols = the count needed;
//                          rerun with a larger capacity / scratch)
//   AV1G_GPU_ERR_OUTPUT    a tile's payload exceeded its capacity (symbols / 2 + 256 bytes)
//   AV1G_GPU_ERR_ORDER     blocks not in decode order / do not tile the superblocks
// run_batch_device itself fails (returns -1, err set) only for bad host arguments or launch errors.
#ifndef AV1G_GPU_ENTROPY_PUBLIC_H
#define AV1G_GPU_ENTROPY_PUBLIC_H

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>
#include <cuda_runtime_api.h>
#include "av1g/decisions.h"

namespace av1g_gpu {

enum { AV1G_GPU_ERR_SYMBOLS = 1, AV1G_GPU_ERR_OUTPUT = 2, AV1G_GPU_ERR_ORDER = 4 };

struct DevFrame {
    av1g_frame_params fp;  // frame parameters (host copy)
    uint32_t blockBase;    // first block in DevBatch::blocks / blockTxb
    uint32_t nBlocks;
    uint32_t txbBase;      // first tx block in DevBatch::txb
    uint32_t nTxb;
    uint64_t coeffBase;    // base of the frame's coefficients in DevBatch::coeffs
};

struct DevBatch {
    const DevFrame *frames = nullptr;  // HOST array
    uint32_t nFrames = 0;
    const av1g_block *blocks = nullptr;   // device
    const uint32_t *blockTxb = nullptr;   // device
    const av1g_txb *txb = nullptr;        // device
    const int16_t *coeffs = nullptr;      // device
    // Symbol buffer capacity: scratch needs ~5 bytes per symbol of capacity (4 symbol + 1 payload
    // staging/packing).  0 = default 0.75 * luma pixels (x 2 for 4:4:4 frames) + 16 * blocks + 8 * tx blocks.  Measured:
    // 0.41 symbols/pixel for the encoder's dev batch (q120 class), 0.54 for libaom s6 q60 decisions;
    // lower qindex means more.  Too small -> AV1G_GPU_ERR_SYMBOLS with the exact count; rerun.
    uint64_t symbolCapacity = 0;
};

struct DevStatus {
    int32_t error;            // 0 = ok, else AV1G_GPU_ERR_* bits
    int32_t reserved;
    uint64_t symbols;         // symbols in the batch (also when it exceeded the capacity)
    uint64_t symbolCapacity;
    uint64_t bytes;           // total payload bytes = tileOffset[nTiles]
};

struct DevResult {
    // device pointers into the scratch buffer, valid once the stream has executed the call and
    // until the scratch is reused
    const void *result = nullptr;       // DevStatus followed by uint64 tileOffset[nTiles + 1]
    size_t resultBytes = 0;             // = sizeof(DevStatus) + 8 * (nTiles + 1): one download
    const DevStatus *status = nullptr;  // = result
    const uint64_t *tileOffset = nullptr;
    const uint8_t *bytes = nullptr;     // packed payloads
    size_t bytesCapacity = 0;
    // host
    uint32_t nTiles = 0;
    std::vector<uint32_t> frameTile0;   // [nFrames + 1]
    // diagnostics / tests (device): the symbol streams, tile i = syms[symBegin[i] .. symEnd[i])
    const uint32_t *syms = nullptr;
    const uint64_t *symBegin = nullptr, *symEnd = nullptr;
};

// Optional phase timing: if non-null, events e[0..7] are recorded on the stream at: start, prep
// done (tables, block/SB/tile ranges, delta-q chain), maps + contexts, count, scan, emit, entropy,
// packed.  The caller creates the events.
struct DevEvents {
    cudaEvent_t e[8];
};

size_t run_batch_device_scratch_bytes(const DevBatch &b);
int run_batch_device(const DevBatch &b, cudaStream_t s, void *scratch, size_t scratchBytes, DevResult &out, std::string &err,
                     const DevEvents *ev = nullptr);

// Entropy kernel selection (benchmarks): 3 = two-warp windowed coder with libaom's 64-bit od_ec
// window (default), 4 = same with the 32-bit precarry od_ec; code_streams() also has 2 = M4.
extern int g_entropy_mode;

}  // namespace av1g_gpu
#endif
