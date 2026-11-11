// host API of the GPU AV1 still-image encoder (batch of frames in flight).
#ifndef GP_PIPELINE_H
#define GP_PIPELINE_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <string>
#include <utility>
#include <vector>
#include "av1g/decisions.h"
#include "gp.h"
#include "entropy_dev.h"
#include "imageops.h"

// Front ends (--crop / --max-size, per-line batch options): a crop rectangle and a box to fit in, applied to the
// decoded planes on the GPU before the encode.
struct GpXform {
    int cx = 0, cy = 0, cw = 0, ch = 0;   // crop rectangle in source pixels (stored orientation); cw = 0: no crop
    int maxW = 0, maxH = 0;               // fit inside maxW x maxH (0: unbounded), aspect kept, never enlarged
    // --resize: ImageMagick's -resize geometry (displayed orientation, like the max-size box); rsFlags 0: none
    enum { RS_ON = 1, RS_W = 2, RS_H = 4, RS_PCT = 8, RS_AREA = 16, RS_EXACT = 32, RS_SHRINK = 64, RS_ENLARGE = 128, RS_FILL = 256 };
    int rsFlags = 0;
    double rsW = 0, rsH = 0;              // the geometry's numbers (percent with RS_PCT, pixels with RS_AREA in rsW)
    bool active() const { return cw > 0 || maxW > 0 || maxH > 0 || rsFlags; }
    // the box and the geometry for an image displayed turned by 90 degrees (EXIF orientation 5..8)
    void turn() {
        std::swap(maxW, maxH);
        if (rsFlags & RS_AREA) return;
        std::swap(rsW, rsH);
        const int w = rsFlags & RS_W, h = rsFlags & RS_H;
        rsFlags = (rsFlags & ~(RS_W | RS_H)) | (w ? RS_H : 0) | (h ? RS_W : 0);
    }
};

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
    int lossless = 0;         // 1: lossless frames (base_q_idx 0, k_lossless.cu); qindex, the lossy search knobs (P),
                              // QM / delta-q / tune=iq and the post-filters are not used.  edgeFilter is.
    int losslessCfl = 1;      // lossless: try CfL on 4x4 chroma blocks
    int llTileArea = 262144;  // lossless: uniform tiles of at most this many pixels (0: the tileMinMp rule as lossy)
    int ycgcoRe = 0;          // front ends, lossless RGB input: YCgCo-Re (CICP matrix 16, 8-bit RGB -> 10-bit 4:4:4)
                              // instead of identity / GBR (matrix 0)
    int alpha = 1;            // front ends: PNG / WebP transparency is coded as an AVIF alpha item (monochrome frames
                              // encoded with gp_alpha_config(); 0 = --no-alpha: not coded, as before)
    int alphaQindex = -1;     // alpha frames: 0 = lossless, 1..255 = qindex; -1 = the colour frames' qindex
    int palette = 1;          // palette mode (allow_screen_content_tools): 0 off, 1 frames that pass the screen-content
                              // detection (gp_launch_sc_detect), 2 every frame.  Frames without palettes code 0.
    GpXform xform;           // front ends: crop / fit-inside for every image (batch lines may override it)
    GpTrim trim;              // front ends: --trim (after the crop, before the resize) and its conditions
    int gray = 0;             // front ends: --grayscale / --colorspace Gray (GpGray): monochrome frames, after the resize
    GpTone tone;              // front ends: --auto-level / --level / --gamma / --negate, after the trim, before the resize
    double maxMp = 0;         // front ends: largest picture encoded, after crop / resize (--max-mp; 0: the front end's
                              // default, gp_enc_max_mp).  Pictures above 35.65 MP (AV1 level 6.3, the most AVIF's
                              // Advanced profile allows) are written with seq_level_idx 31 (no level limits)
    double maxInputMp = 0;    // front ends: largest source decoded (--max-input-mp; 0 or less than the maxMp limit: that
                              // limit); above it only helps sources that a crop / max-size brings under maxMp
    // Staged analysis (an_stage.h): decision-changing speed knobs, all off by default, for every preset (not in GpParams,
    // so the default analysis and the closed loop are unchanged).  Lossless frames ignore them.
    int anTopdown = 0;        // --topdown 1: shapes with a side of 8 / 4 only where a DP over the larger shapes asks
    GpConfig();
};

// palette search exists for this configuration (GpConfig::palette applies): lossless (luma + chroma palettes) and lossy
// (luma palettes)
inline bool gp_palette_supported(const GpConfig &) { return true; }

// The front ends' limits in megapixels (0: none): the largest picture encoded (--max-mp, else dflt: 35.6 in batch mode
// and for JPEGs, none for single PNG / WebP images, as before the options) and the largest source decoded.
inline double gp_enc_max_mp(const GpConfig &c, double dflt) { return c.maxMp > 0 ? c.maxMp : dflt; }
inline double gp_dec_max_mp(const GpConfig &c, double dflt) {
    const double e = gp_enc_max_mp(c, dflt);
    return e > 0 && c.maxInputMp < e ? e : c.maxInputMp;
}
// The largest side a stream can have: its sequence header gives each 16 bits (frame_width_minus_1)
constexpr int GP_MAX_SIDE = 65536;
// The tallest output of gp_frames_transform: k_rz_h has one block per row of the padded plane, and a grid dimension
// holds at most 65535
constexpr int GP_MAX_RESIZED_H = 65472;
// Why a picture of w x h is not encoded ("": it is): a side above GP_MAX_SIDE, a crop / resize taller than
// GP_MAX_RESIZED_H, or more than maxMp megapixels (0: no limit).  after: what the size comes from, as " after crop /
// resize", when gp_frames_transform will run (gp_xform_runs); empty when the picture is the source as decoded
std::string gp_size_refusal(int w, int h, double maxMp, const char *after = "");
// Whether gp_frames_transform changes a W x H frame with x, whose output is ow x oh
bool gp_xform_runs(const GpXform &x, int W, int H, int ow, int oh);
// The message for a source of w x h above the largest source decoded, decMp (w = 0: size not known).  x (may be NULL):
// its crop / resize, for the value to suggest; inputLimit: decMp is --max-input-mp, above the --max-mp limit
std::string gp_source_refusal(int w, int h, double decMp, const GpXform *x, bool inputLimit);
// The ingest's pinned staging arena for sources up to decMp megapixels: JPEGs the CPU route decodes (sides above 6000
// px) stage their coefficients there, up to 6 bytes per pixel (4:4:4); 256 MiB for the default 35.6 MP
inline int64_t gp_ingest_pinned_bytes(double decMp) {
    const int64_t b = (int64_t)(decMp * 6e6) + (16ll << 20);
    return b > (256ll << 20) ? b : 256ll << 20;
}

// Encoder configuration for the alpha frames of images encoded with `colour` (monochrome frames at the colour's bit
// depth): lossless when colour.lossless or alphaQindex == 0 (the lossless path, as libavif's alpha quality 100);
// otherwise lossy at alphaQindex (or colour.qindex) with libavif's alpha tuning, tune=psnr: no tune=iq frame decisions,
// QM, variance boost / delta-q or SSIM lambda (libavif: libaom's SSIM / IQ tunings ring on alpha); the speed preset (P)
// and the deblocking / CDEF choice stay the colour's.  The front ends use it with the main preset for every image (also
// below --small-mp: --small-preset applies to the colour frames only).
GpConfig gp_alpha_config(const GpConfig &colour);

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
    int jpegChroma = 0;        // 1: 4:2:0 chroma as a 4:2:0 JPEG coded it, which its decoders upsample (and so --trim,
                               // the tone operators and --grayscale do); 0 after a crop / resize
    int colorMatrix = -1;      // CICP matrix_coefficients of the planes (0 identity / GBR, 6 BT.601, 16 YCgCo-Re; written
                               // with BT.709 primaries and sRGB transfer, full range); -1: av1core's default (1/13/6);
                               // -2: unspecified 2/2/2 (no colour description in the sequence header: AVIF alpha frames)
};

struct GpFrameOut {
    av1g_frame_params fp;
    std::vector<av1g_block> blocks;
    std::vector<uint32_t> block_txb;
    std::vector<av1g_txb> txb;
    std::vector<int16_t> coeffs;
    std::vector<av1g_palette> palettes;  // v5 palettes and their colour index maps
    std::vector<uint8_t> color_map;
    std::vector<uint8_t> recon[3];   // cropped: Y W x H, U/V ((W+1)/2) x ((H+1)/2) (4:4:4: W x H); 10-bit frames:
                                     // uint16_t samples (2 bytes each, native byte order)
    int orientation = 1;             // EXIF orientation of the source (set by the caller; AVIF irot/imir)
    bool dpNet = false;              // k_dp's safety net coded blocks of this frame (GpTimes::dpRepaired counts them)
    av1g_decision_record record();   // points into the vectors above
};

struct GpTimes {  // ms, CUDA events
    float setup = 0, analysis = 0, dp = 0, recon = 0, pack = 0, postfilter = 0, download = 0, total = 0;
    uint32_t steps = 0, sbs = 0, blocks = 0, skipBlocks = 0, cflBlocks = 0, rangeFixes = 0;
    uint32_t txSplit = 0;     // blocks coded with tx_depth 1 (tx-size search)
    uint32_t dpRepaired = 0;  // blocks k_dp's safety net coded (8x8 areas where the DP had no partition of finite cost;
                              // 0 unless GpEncoder::init's partition check missed a case; the apps warn and name the
                              // inputs, GpFrameOut::dpNet; tests/decoder_gate.py fails)
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

// The partition check of GpEncoder::init without the pruning (lossy options): -1 and err when --shapes / --part8 /
// --partN / --maxpart leave some superblock of some frame size no partition the analysis evaluates.  av1gpu checks
// the small-image preset's options with it at start (that encoder is only set up for a small image).
int gp_check_partitions(const GpParams &P, std::string &err);
// The warning (stderr) for k_dp's safety net: an input's name (or NULL) and / or a block count (0: none given);
// tests/decoder_gate.py fails a run that prints it.
void gp_warn_dp_net(const char *what, unsigned long long n);

// Front ends (--depth 10): convert the 8-bit frames of `frames` (bitDepth 8) to 10-bit device planes in place
// (Y10 = round(Y8 * 1023 / 255), C10 = round((C8 - 128) * 1023 / 255) + 512, libavif's full-range mapping).  One
// device allocation on st for all of them is returned in *buf (NULL if nothing was converted); free it with
// cudaFreeAsync(*buf, st) after the encode.  The source planes must stay valid until the work on st is done.
int gp_frames_to_10bit(std::vector<GpFrameIn> &frames, cudaStream_t st, uint8_t **buf, std::string &err);

// Output size of a W x H source: the crop (clipped to the image, as ImageMagick's -crop), then ImageMagick's
// -resize 'WxH>' geometry (scale = min(maxW / w, maxH / h) if it is < 1, each side max(1, floor(side * scale + 0.5))).
// false: the crop lies outside the image.
bool gp_xform_size(const GpXform &x, int W, int H, int *ow, int *oh);
// "WxH+X+Y" of x's crop (for messages)
std::string gp_crop_str(const GpXform &x);
// Option values: crop "WxH+X+Y" (ImageMagick geometry, e.g. from -format %@) or "X,Y,W,H" (W, H >= 1); max-size "WxH",
// "xH" or "Wx" (sides >= 1).  false: *why says what is wrong.
bool gp_parse_crop(const char *v, GpXform &x, std::string *why = nullptr);
bool gp_parse_max_size(const char *v, GpXform &x, std::string *why = nullptr);
// --resize: ImageMagick's -resize geometry: W, xH, WxH (fit inside), WxH^ (fill), WxH! (exactly), any of them with > (only
// shrink) or < (only enlarge); N% or N%xM% (scale); @A (A pixels).  No offsets.  false: *why says what is wrong.
bool gp_parse_resize(const char *v, GpXform &x, std::string *why = nullptr);
// Apply xf[i] to frames[i] (frames whose transform is not active, or is an identity, are left alone): crop, then
// shrink with ImageMagick's default -resize filter for shrinking (Lanczos, 3 lobes, support stretched by the shrink
// factor, taps clipped to the crop and renormalised, in the planes' own (YCbCr, gamma-encoded) samples), written at
// outDepth (10: the k_to10 mapping applied to the filtered 8-bit values, i.e. without an 8-bit rounding in between).
// 4:2:0 chroma is resampled at its own sample positions with the luma scale.  alpha (optional): per frame its alpha
// frame before the transform (W = 0: none); the colour of a frame with alpha is filtered with alpha-weighted taps
// (associated alpha, as ImageMagick resizes transparent images, so hidden colour does not bleed into edges), and keeps
// the plain filter's value where the weighted alpha is below half a step (nothing visible).  The new planes share one
// device allocation on st returned in *buf (NULL if none); free it like gp_frames_to_10bit's.  A crop outside a frame
// is an error.
int gp_frames_transform(std::vector<GpFrameIn> &frames, const std::vector<GpXform> &xf, int outDepth, cudaStream_t st,
                        uint8_t **buf, std::string &err, const std::vector<GpFrameIn> *alpha = nullptr);

// --trim: measure frames[i] inside its crop xf[i] (frames with W = 0 are left alone; alpha: optional, per frame its alpha
// frame, W = 0: none) on st, wait for it, and decide (gp_trim_decide): verdict[i] is a GpTrimVerdict, and for GP_TRIM_CUT
// xf[i]'s crop becomes the kept rectangle (source pixels), so gp_frames_transform cuts it.
int gp_frames_trim(const std::vector<GpFrameIn> &frames, std::vector<GpXform> &xf, const GpTrim &t, cudaStream_t st,
                   std::vector<int> &verdict, std::string &err, const std::vector<GpFrameIn> *alpha = nullptr);

// --auto-level / --level / --gamma / --negate: frames[i] changed in place on st (--auto-level measures inside xf[i]'s
// crop, then waits for st).  -1: a CUDA error (err).
int gp_frames_tone(std::vector<GpFrameIn> &frames, const std::vector<GpXform> &xf, const GpTone &t, cudaStream_t st,
                   std::string &err);

// --grayscale: colour frames of `frames` made monochrome in place on st (the Y plane rewritten with the gray of each
// pixel, GpGray method; Rec601Luma on BT.601 YCbCr planes keeps the Y plane as it is).  Monochrome frames are left alone.
void gp_frames_gray(std::vector<GpFrameIn> &frames, int method, cudaStream_t st);

// Entropy coding.  CPU: av1core's writer (av1g_encode, bitstream only).  GPU: av1core GPU entropy
// (symbolize + od_ec on the GPU) + OBU assembly.  Both produce identical bytes.
// obu gets [TD] + sequence header + frame; avif the AVIF file (icc may be empty).  alpha (optional, per frame, may be
// NULL): the frame's alpha frame, coded into alphaObuWithTd and muxed as the AVIF's alpha item (av1g_mux_avif_alpha).
int gp_entropy_cpu(GpFrameOut &f, const std::vector<uint8_t> &icc, std::vector<uint8_t> &obuWithTd,
                   std::vector<uint8_t> &avif, std::string &err, GpFrameOut *alpha = nullptr,
                   std::vector<uint8_t> *alphaObuWithTd = nullptr);
int gp_assemble_obus(const av1g_frame_params *fp, const std::vector<std::vector<uint8_t>> &tileData, bool td,
                     std::vector<uint8_t> &obu, std::string &err);
int gp_entropy_gpu(std::vector<GpFrameOut *> &frames, const std::vector<const std::vector<uint8_t> *> &icc,
                   std::vector<std::vector<uint8_t>> &obuWithTd, std::vector<std::vector<uint8_t>> &avif, double *gpuMs,
                   std::string &err, const std::vector<GpFrameOut *> *alpha = nullptr,
                   std::vector<std::vector<uint8_t>> *alphaObuWithTd = nullptr);
// AVIF file of one image from its colour OBUs and optional alpha OBUs (no temporal delimiter; alpha may be NULL), the
// ICC profile (may be empty) and the EXIF orientation (irot / imir, on both items).
int gp_mux_avif(const av1g_frame_params &fp, const uint8_t *obu, size_t obuSize, const uint8_t *alphaObu, size_t alphaSize,
                const std::vector<uint8_t> &icc, int orientation, std::vector<uint8_t> &avif, std::string &err);

#endif
