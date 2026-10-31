// host API of the GPU AV1 still-image encoder (batch of frames in flight).
#ifndef GP_PIPELINE_H
#define GP_PIPELINE_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <string>
#include <vector>
#include "av1g/decisions.h"
#include "gp.h"
#include "entropy_dev.h"

struct GpConfig {
    int qindex = 100;
    GpParams P;
    // size-adaptive effort: images with W*H < smallMp * 1e6 use Psmall (--small-preset); 0 = off
    double smallMp = 0;
    GpParams Psmall;
    // tool hooks (av1f / av1core)
    int qm = -1;              // -1 off, -2 libaom all-intra curve (av1g_qm_level_allintra), else level 0..15
    int qmUV = -1;            // chroma level (-1: same as luma)
    int deltaq = 0;           // 1: per-SB qindex from GpFrameIn::sbQ or the built-in variance boost
    int deltaqRes = 2;
    float deltaqStrength = 0; // built-in variance boost strength (qindex per log2 activity unit); 0 = off
    int dqUDc = 0, dqUAc = 0, dqVDc = 0, dqVAc = 0;  // frame chroma delta-q
    int edgeFilter = 0;       // enable_intra_edge_filter
    // post-filters (av1core); strength search = av1f (hooks: these fields / per-SB cdef_idx)
    int lfMode = 0;           // 0 off, 1 libaom LPF_PICK_FROM_Q level (times lfMul), 2 fixed lfLevel
    float lfMul = 1.0f;
    int lfLevel = 0;
    int cdef = 0;             // 1: one CDEF strength preset for the frame (cdef_bits 0)
    int cdefDampingMinus3 = 0, cdefYPri = 0, cdefYSec = 0, cdefUVPri = 0, cdefUVSec = 0;
    bool verbose = false;
    bool wantRecon = true;
    // av1f tools: tune=iq frame decisions (chroma delta-q, QM levels, delta_q_res), variance boost,
    // loop-filter / CDEF search (av1f_*).  --tune iq turns all of them on (libaom s6 tune=iq emulation).
    int tuneIq = 0;           // frame-level tune=iq decisions from av1f_tuneiq_frame_init
    int vb = 0;               // variance-boost per-SB qindex (av1f_varboost)
    int vbOrder = 0;          // AV1F_VB_TILE (libaom -j1) / ROW / NEAREST
    int vbStrength = 100;     // deltaq-strength %
    int fLf = -1;             // -1: own simple path (lfMode/cdef fields); else av1f lf_mode (0 off, 1 from-Q, 2 search)
    int fCdef = 0;            // av1f cdef_mode (0 off, 1 libaom LVL4 emulation, 2 search)
    int fDist = 0;            // av1f distortion (0 SSE, 1 variance-weighted)
    float tileMinMp = 0;      // > 0: images of >= tileMinMp megapixels get tileColsLog2 / tileRowsLog2 uniform tiles
    int tileColsLog2 = 1, tileRowsLog2 = 0;
    int ssimLambda = 0;       // tune=iq per-16x16 SSIM rdmult scaling of lambda
    int yuv444 = 0;           // front ends (CLI, batch mode): produce full-resolution chroma input planes and
                              // code colour frames 4:4:4 (the encoder itself follows GpFrameIn::yuv444)
    int depth = 8;            // front ends: 10 = produce 10-bit input planes and code 10-bit (high_bitdepth) streams
                              // (the encoder itself follows GpFrameIn::bitDepth)
    GpConfig();
};

// One input frame: device planes, padded (edge replicated) to aw x ah (multiples of 64); chroma
// (aw/2 x ah/2, or aw x ah with yuv444).
struct GpFrameIn {
    int W = 0, H = 0, mono = 0;
    int yuv444 = 0;            // 1: chroma planes are full size and the frame is coded 4:4:4 (High profile)
    int bitDepth = 8;          // 10: dsrc are uint16_t planes (10-bit samples, pitches in samples), coded 10-bit.
                               // All frames of one encode() call must have the same bit depth.
    const uint8_t *dsrc[3] = {nullptr, nullptr, nullptr};
    int spitch[3] = {0, 0, 0};
    int aw = 0, ah = 0;
    std::vector<uint8_t> sbQ;  // optional per-SB qindex target map (raster, sbRows x sbCols); legalized here
};

struct GpFrameOut {
    av1g_frame_params fp;
    std::vector<av1g_block> blocks;
    std::vector<uint32_t> block_txb;
    std::vector<av1g_txb> txb;
    std::vector<int16_t> coeffs;
    std::vector<uint8_t> recon[3];   // cropped: Y W x H, U/V ((W+1)/2) x ((H+1)/2) (4:4:4: W x H); 10-bit frames:
                                     // uint16_t samples (2 bytes each, native byte order)
    int orientation = 1;             // EXIF orientation of the source (set by the caller; AVIF irot/imir)
    av1g_decision_record record();   // points into the vectors above
};

struct GpTimes {  // ms, CUDA events
    float setup = 0, analysis = 0, dp = 0, recon = 0, pack = 0, postfilter = 0, download = 0, total = 0;
    uint32_t steps = 0, sbs = 0, blocks = 0, skipBlocks = 0, cflBlocks = 0, rangeFixes = 0;
    uint32_t txSplit = 0;     // blocks coded with tx_depth 1 (tx-size search)
    size_t deviceBytes = 0;
    unsigned long long prof[16] = {0};  // k_recon phase cycles (--prof)
};

class GpEncoder {
public:
    GpEncoder();
    ~GpEncoder();
    int init(const GpConfig &cfg, std::string &err);
    // Encode a batch of frames (all on the GPU at once).  Decisions + reconstruction; entropy coding is
    // done separately (gp_entropy_*).
    // dev != NULL: the DecisionRecords stay on the GPU (ownership passes to dev; consume with gp_entropy_dev,
    // free with gp_dev_records_free); out[i] then only has fp (+ recon if wanted).
    int encode(const std::vector<GpFrameIn> &in, std::vector<GpFrameOut> &out, GpTimes &t, std::string &err,
               cudaStream_t st = 0, GpDevRecords *dev = nullptr);
    const GpConfig &config() const { return cfg_; }

private:
    GpConfig cfg_;
    cudaEvent_t ev_[9];
    cudaEvent_t evx_[2];
    cudaStream_t hiSt_ = nullptr;   // high-priority stream for the wavefront
    uint8_t *vbLuts_ = nullptr;
    uint8_t *vbLuts10_ = nullptr;   // variance-boost LUTs with libaom's 10-bit q tables
};

// Front ends (--depth 10): convert the 8-bit frames of `frames` (bitDepth 8) to 10-bit device planes in place
// (Y10 = round(Y8 * 1023 / 255), C10 = round((C8 - 128) * 1023 / 255) + 512, libavif's full-range mapping).  One
// device allocation on st for all of them is returned in *buf (NULL if nothing was converted); free it with
// cudaFreeAsync(*buf, st) after the encode.  The source planes must stay valid until the work on st is done.
int gp_frames_to_10bit(std::vector<GpFrameIn> &frames, cudaStream_t st, uint8_t **buf, std::string &err);

// Entropy coding.  CPU: av1core's writer (av1g_encode, bitstream only).  GPU: av1core GPU entropy
// (symbolize + od_ec on the GPU) + OBU assembly.  Both produce identical bytes.
// obu gets [TD] + sequence header + frame; avif the AVIF file (icc may be empty).
int gp_entropy_cpu(GpFrameOut &f, const std::vector<uint8_t> &icc, std::vector<uint8_t> &obuWithTd,
                   std::vector<uint8_t> &avif, std::string &err);
int gp_assemble_obus(const av1g_frame_params *fp, const std::vector<std::vector<uint8_t>> &tileData, bool td,
                     std::vector<uint8_t> &obu, std::string &err);
int gp_entropy_gpu(std::vector<GpFrameOut *> &frames, const std::vector<const std::vector<uint8_t> *> &icc,
                   std::vector<std::vector<uint8_t>> &obuWithTd, std::vector<std::vector<uint8_t>> &avif, double *gpuMs,
                   std::string &err);

#endif
