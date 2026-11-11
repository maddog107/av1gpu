// structures shared by host code and CUDA kernels.
//
// One "batch" = many images encoded together.  All per-image state lives on the GPU; kernels
// index it through GpImage descriptors.  Stage overview (see docs/ARCHITECTURE.md):
//   analysis  per block shape: open-loop (source-edge) SATD screening of luma modes -> top-K,
//             exact-rate RD of the top-K (DCT_DCT), tx-type search for the winner     (k_analysis)
//   dp        per 64x64 SB: partition DP over the per-shape J grids -> leaf blocks   (k_dp)
//   recon     closed loop in SB wavefront order (2-SB lag): re-check top-K with the real
//             reconstructed edges + exact contexts, chroma modes / CfL, quantize, exact
//             inverse transform + reconstruction (bit-exact with the decoder)          (k_recon)
//   pack      per-SB outputs -> DecisionRecord arrays in decode order                 (k_pack)
//   entropy   av1core GPU symbolize + od_ec (or the CPU writer), OBUs, AVIF
#ifndef GP_H
#define GP_H

#include <stdint.h>
#include "av1g/decisions.h"

#define GP_MAXK 6            // max top-K luma candidates kept per block by the analysis
#define GP_NSHAPES 22        // indexed by AV1 bsize (13..15 = 128-sizes, unused)
#define GP_MAX_LEAVES 256    // per 64x64 SB
#define GP_MAX_TXB 384       // per SB: 256 luma + 2 x 64 chroma (4:2:0)
#define GP_SB_COEFFS 6144    // per SB: 64*64 + 2*32*32 (64-point tx store 32x32 only)
#define GP_MAX_TXB_444 768   // per SB in batches with 4:4:4 frames: 3 x 256
#define GP_SB_COEFFS_444 12288  // 3 x 64*64
#define GP_RC_BIG_BYTES 18560  // per-CTA global scratch of k_recon for 32/64-point evaluations
#define GP_NCDF (sizeof(av1g_cdfs) / 2)

// Tunables (host -> kernels by value).
struct GpParams {
    float lambdaMul;     // multiplier on the libaom-style lambda (calibrate by BD-rate sweep)
    float satdLambda;    // screening: J_s = SATD + satdLambda * sqrt(lambda) * bits
    float rnd, zbin;     // dead-zone quantizer: q = |c|/step < zbin ? 0 : floor(|c|/step + rnd)
    float rndC, zbinC;   // same for chroma
    int K;               // luma candidates kept by the analysis (1..GP_MAXK)
    int Krecheck;        // candidates re-evaluated in the closed loop with real edges (1..K)
    int txSearch;        // 1: tx-type search (luma); 0: DCT_DCT only
    int txSearchCL;      // tx-type search again in the closed loop (real residual)
    int deltas;          // 0: no angle deltas, 1: +-1..3 around the best 2 directional modes
    int uvDeltas;        // 1: chroma angle deltas in the closed loop (blocks of 8x8 and up)
    int cfl;             // 1: try CfL
    int uvK;             // chroma modes fully evaluated after SATD screening (>= 1)
    int shapeMask;       // bit per bsize (AV1G_BLOCK_*): shapes the analysis evaluates
    int partMask8, partMaskN;  // allowed partition types (bit per AV1G_PARTITION_*) at 8x8 / larger nodes
    int maxPart;         // largest node size allowing NONE/rect (64 or 32)
    float uvOverheadBits;  // open-loop estimate of chroma signalling per chroma-carrying block
    float splitBias;     // multiplies the J of PARTITION_NONE (1 = neutral)
    int rdoq;            // 1: greedy RDOQ in the closed loop
    float lambdaCMul;    // chroma lambda relative to luma
    int qmDist;          // 1: coefficient errors weighted by (32 / QM weight)^2 (libaom dist-metric=qm-psnr)
    int qmRnd;           // 1: rounding offset scaled by 32 / QM weight (libaom FP quantizer with QM)
    int smallKernel;     // 1: thread-per-block analysis for shapes <= 64 pixels
    float pruneVar;      // > 0: skip blocks of >= pruneMinArea pixels whose source variance > pruneVar * step_ac^2
    int pruneMinArea;
    int screenSub;       // 2: checkerboard-subsampled SATD screening for blocks >= 256 pixels
    int filterIntra;     // 1: filter-intra candidates in the closed loop (enable_filter_intra)
    int kLarge;          // > 0: K for shapes of >= 1024 pixels
    int deltasMaxArea;   // angle-delta screening only for shapes up to this area (0 = all)
    int profile;         // 1: per-phase clock64 accounting in k_recon (B.prof)
    int uvInDp;          // 1: add an open-loop chroma RD term to the analysis J (chroma-aware partitioning)
    int txSizeSearch;    // 1: closed-loop tx-size search (largest vs one split level), frame uses TX_MODE_SELECT
    int txAllCL;         // 1: closed loop evaluates every tx type for every re-checked candidate (no separate pass 2)
    int poisonSmem;      // debug: fill k_recon shared memory with 0x7F at CTA start (GP_POISON_SMEM)
    float pruneFlat;     // > 0: skip shapes < 16x16 inside 16x16 regions whose 8x8 variances are all < pruneFlat * step_ac^2
};

// Per-image descriptor (device copy in an array; host fills it).
struct GpImage {
    int W, H;                // visible size
    int MiCols, MiRows;      // coded area in 4x4 units (picture padded to a multiple of 8)
    int sbCols, sbRows;
    int mono;
    int baseQ, qctx;         // base_q_idx, coefficient-CDF quantizer context (0..3)
    int dqRes;               // delta_q_res
    const uint8_t *sbQ;      // per-SB qindex (raster sbRows x sbCols); NULL = baseQ everywhere
    int usingQm, qm[3];      // quantizer matrices (level 15 = flat)
    int dqUDc, dqUAc, dqVDc, dqVAc;  // frame chroma delta-q (DeltaQUDc ...)
    int edgeFilter;          // enable_intra_edge_filter
    const float *ssimLog;    // per 16x16 cell: log of the normalised tune=iq SSIM rdmult scale (NULL = off)
    int cols16, rows16;
    const float *var16;      // per 16x16 cell: max per-pixel variance of its 8x8 blocks (flat pruning; NULL = off)
    const uint8_t *src[3];   // source planes, padded (replicated) to aw x ah (chroma (aw >> ssx) x (ah >> ssy));
                             // 10-bit batches: uint16_t samples, pitches in samples (gp_src<BD>() / gp_rec<BD>(); the
                             // bit depth is a kernel template parameter, not a field: GpImage keeps its 8-bit layout)
    int spitch[3];
    uint8_t *rec[3];         // reconstruction planes, same geometry and sample type
    int rpitch[3];
    int aw, ah;
    // tiles (mi units; [n] = MiCols / MiRows)
    int tileCols, tileRows;
    int tileColStart[65], tileRowStart[65];
    // analysis grids per shape: blocks shOff[s] .. shOff[s] + shNbx[s]*shNby[s] in the shape arrays
    uint32_t shOff[GP_NSHAPES];
    int shNbx[GP_NSHAPES], shNby[GP_NSHAPES];
    uint32_t sbBase;         // first SB of this image in the global SB arrays (raster per image)
    // 4x4 maps (MiRows x MiCols, raster) for closed-loop contexts
    uint8_t *mYMode, *mUVMode, *mSkip, *mMiSize, *mTx;
    // per-plane 4x4 coefficient-context maps (culLevel, dcCategory); chroma (MiRows >> ssy) x (MiCols >> ssx)
    uint8_t *cLvl[3], *cDc[3];
    // (appended, so the fields above keep their offsets and the 4:2:0 kernels their code)
    int ssx, ssy;            // chroma subsampling: 1, 1 (4:2:0, monochrome) or 0, 0 (4:4:4)
    int sct;                 // 1: screen-content tools (palette candidates; set by the detection, gp_launch_sc_detect,
                             // in the device copy)
    uint32_t scBlk0;         // the detection's first 16x16 block of this image ((W / 16) * (H / 16) blocks per image)
};

// Analysis result per (shape, position).
struct GpBlockRes {
    float J;                   // best RD cost (luma, open loop) incl. mode bits
    uint8_t cand[GP_MAXK];     // candidates (mode << 3 | (delta + 3)), best first
    uint8_t ncand;
    uint8_t tx;                // best tx type for cand[0]
};

// DP output: leaf blocks of one SB in decode order.
struct GpLeaf {
    uint8_t bsize;
    uint8_t x4, y4;            // position inside the SB in 4x4 units
    uint8_t pad;
};

// Closed-loop outputs per SB (fixed-capacity slots, compacted by k_pack).
struct GpSbOut {
    uint32_t nBlocks, nTxb, nCoeffs;
    uint32_t nPal;             // palette blocks (GpBatch::oPal slots; their block fields SB-local)
    uint32_t nMap;             // colour index map bytes (GpBatch::oMap slots)
    uint32_t pad[3];
};
#define GP_SB_PAL 64           // palette blocks per SB at most (blocks of >= 32 pixels)
#define GP_SB_MAP 8192         // colour index map bytes per SB at most (64 x 64 luma + the U / V map, 64 x 64 in 4:4:4)

// External candidate (optional external screener hook, 8 bytes; unused by default).
struct GpExtCand {
    float J;
    uint8_t mode;
    int8_t delta;
    uint8_t tx;
    uint8_t pad;
};

// Per-batch device pointers.
struct GpBatch {
    int nimg;
    GpImage *img;              // [nimg]
    GpBlockRes *blk;           // analysis results (all shapes, all images)
    uint32_t *shImgStart;      // [GP_NSHAPES][nimg + 1] first block of each image in shape s
    uint32_t nSb;              // total SBs
    uint32_t *sbImg;           // [nSb] image of each SB
    uint8_t *leafCount;        // [nSb] (0..255 means 1..256: stored as count-1)
    uint16_t *leafCountW;      // [nSb]
    GpLeaf *leaves;            // [nSb * GP_MAX_LEAVES]
    float *sbJ;                // [nSb] DP cost (stats; not finite: k_dp's safety net coded blocks of the SB)
    // closed-loop outputs
    GpSbOut *sbOut;            // [nSb]
    av1g_block *oBlocks;       // [nSb * GP_MAX_LEAVES]
    uint16_t *oBlockTxb;       // [nSb * GP_MAX_LEAVES] first txb (SB-local) of each block
    av1g_txb *oTxb;            // [nSb * sbTxbCap]  (coeff_off SB-local)
    int16_t *oCoeffs;          // [nSb * sbCoefCap]
    uint32_t *stats;           // [16] counters
    // optional external screener: per shape [nblocks][extTopK] candidates, index extOff[s * nimg + img] + local.
    // (Nothing sets it today.)  Contract: at least one candidate with mode < 13 for every in-frame block of an analysed
    // shape.  With none, k_analysis_small writes J = GP_INF, ncand 0 (k_dp's safety net covers its 4x4 leaves), but
    // k_analysis (shapes above 64 pixels, or all with smallKernel 0) runs its tx search on top[0] unset and can write a
    // finite record with an undefined candidate.
    const void *extCand[GP_NSHAPES];
    const int64_t *extOff;
    int extTopK;
    unsigned long long *prof;  // [16] k_recon phase cycles (GpParams::profile)
    uint32_t *sbCyc;           // [nSb] k_recon cycles per SB (profile only, may be null)
    void *bigScratch;          // [max CTAs per wavefront step] x GP_RC_BIG_BYTES (k_recon)
    // (appended, so the fields above keep their offsets)
    uint32_t sbTxbCap, sbCoefCap;  // per-SB output slots: GP_MAX_TXB / GP_SB_COEFFS, or the _444 sizes when the
                                   // batch has a 4:4:4 frame
    int any420, any444;        // the batch has 4:2:0 / 4:4:4 colour frames (chroma kernels per layout)
    // palettes (frames with GpImage::sct): per-SB slots, null when no frame of the batch has screen-content tools
    av1g_palette *oPal;        // [nSb * GP_SB_PAL] (block = SB-local leaf index, map_off SB-local)
    uint8_t *oMap;             // [nSb * GP_SB_MAP]
    const uint32_t *palAny;    // device flag: some image of the batch has GpImage::sct (the lossy palette search runs)
};

// Live 16x16 cells of a batch with the flat-area pruning (gp_launch_live16), an argument of the analysis kernels of their
// own (GpBatch keeps its layout): the cells that the pruning keeps, as (image, cy << 16 | cx) pairs in no particular
// order.  The shapes below 16x16 take their blocks cell by cell from it, so pruned cells cost no threads.  All zero: no
// list (the kernels use the shapes' raster grids).
struct GpLive {
    uint32_t nCells;           // 16x16 cells of the batch (raster per image)
    const uint32_t *cellStart; // [nimg + 1] first cell of each image
    uint32_t *cell;            // [2 * nCells]
    uint32_t *count;           // [1]
};

// Lossless analysis (k_lossless.cu): per-plane 4x4 cost tables (uint16, 1/16 bit, position-major:
// cost[p][costOff[img][p] + pos * NE_p + entry], pos = y4 * (MiCols >> ssx) + x4).
#define LL_NCAND_Y 66    // 61 (mode, angle delta) candidates + 5 filter-intra modes
#define LL_NCAND_UV 61
#define LL_NE_Y 93       // cost entries: candidates x the above-right / below-left availabilities they depend on
#define LL_NE_UV 88
struct GpLossless {
    uint16_t *cost[3];
    const uint64_t (*costOff)[3];   // [nimg][3] (device)
    const uint64_t *segStart;       // [nSeg + 1] first k_ll_cost thread of each (image, plane) segment, 3 per image
    int nSeg;
    uint64_t hostTotal;             // segStart[nSeg]
    float lambdaScale;              // J = bits * gp_lambda(0, 1) * lambdaScale (k_dp's lambda units)
    int cfl;                        // try CfL (4x4 chroma blocks)
};

#endif
