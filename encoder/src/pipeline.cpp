// host orchestration of one batch (all frames in flight on the GPU at once).
#include "pipeline.h"
#include <mutex>
#include "pinned.h"
#include "kernels.h"
#include "pack.h"
size_t gp_recon_scratch_bytes();   // k_recon.cu (W5/W3)
void gp_recon_order(const GpBatch &B, uint32_t *dSteps, const uint32_t *dSeg, int nSeg, uint32_t maxSeg, cudaStream_t st);   // k_recon_order.cu
#include "av1g/av1g.h"
#include "av1g/blockinfo.h"
#include "av1g/dsp.h"
#include "av1g/cdfctx.h"
#include "av1g/gpu_postfilter.h"
#include "av1f/tuneiq.h"
#include "av1f/varboost_gpu.h"
#include "av1f/lfsearch_gpu.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <set>

#define GPCK(x)                                                                                   \
    do {                                                                                          \
        cudaError_t e_ = (x);                                                                     \
        if (e_ != cudaSuccess) {                                                                  \
            err = std::string("CUDA: ") + cudaGetErrorString(e_) + " at " #x " (" + __FILE__ + ":" + \
                  std::to_string(__LINE__) + ")";                                                 \
            return -1;                                                                            \
        }                                                                                         \
    } while (0)

GpConfig::GpConfig() {
    memset(&P, 0, sizeof P);
    P.lambdaMul = 1.0f;
    P.satdLambda = 1.0f;
    P.rnd = 0.375f; P.zbin = 0.656f;
    P.rndC = 0.375f; P.zbinC = 0.656f;
    P.K = 4;
    P.Krecheck = 2;
    P.txSearch = 1;
    P.txSearchCL = 1;
    P.deltas = 1;
    P.cfl = 1;
    P.uvK = 2;
    P.shapeMask = 0x3F1FFF;  // all 19 sizes (bits 0..12, 16..21)
    P.partMask8 = 0xF;
    P.partMaskN = 0x3FF;
    P.maxPart = 64;
    P.uvOverheadBits = 1.5f;
    P.splitBias = 1.0f;
    P.rdoq = 0;
    P.lambdaCMul = 1.0f;
    P.qmDist = 0;
    P.qmRnd = 0;
    P.smallKernel = 1;
    P.pruneVar = 0.f;
    P.pruneMinArea = 1024;
    P.screenSub = 1;
    P.filterIntra = 0;
    P.uvDeltas = 1;
    P.pruneFlat = 0.f;
    P.uvInDp = 0;
    P.profile = 0;
    P.poisonSmem = getenv("GP_POISON_SMEM") ? 1 : 0;
    P.kLarge = 0;
    P.deltasMaxArea = 0;
}

av1g_decision_record GpFrameOut::record() {
    av1g_decision_record r;
    memset(&r, 0, sizeof r);
    r.fp = fp;
    r.n_blocks = (uint32_t)blocks.size();
    r.n_txb = (uint32_t)txb.size();
    r.n_coeffs = coeffs.size();
    r.blocks = blocks.data();
    r.block_txb = block_txb.data();
    r.txb = txb.data();
    r.coeffs = coeffs.data();
    r.n_palettes = (uint32_t)palettes.size();
    r.palettes = palettes.empty() ? nullptr : palettes.data();
    r.n_color_map = color_map.size();
    r.color_map = color_map.empty() ? nullptr : color_map.data();
    return r;
}

// The variance-boost LUT of av1f_vb_build_luts_bd (lut[base][var], 256 x 1024) holds only the row of the encoder's
// qindex: k_sb_stats reads lut[base_q_idx][var], and every frame of an encoder has base_q_idx = its qindex.  The other
// rows stay zero.  The row is the same av1f_varboost_qindex() on the host; all 256 rows took 24 ms per encoder.
static cudaError_t gp_vb_lut_row(uint8_t *d_luts, int base, int strength, int bd, cudaStream_t st) {
    uint8_t row[1024];
    for (int v = 0; v < 1024; v++) row[v] = (uint8_t)(base == 0 ? 0 : av1f_varboost_qindex(base, (unsigned)v, strength, bd == 10 ? 10 : 8));
    cudaError_t e = cudaMemsetAsync(d_luts, 0, 256 * 1024, st);
    if (e == cudaSuccess) e = cudaMemcpyAsync(d_luts + (size_t)base * 1024, row, sizeof row, cudaMemcpyHostToDevice, st);
    if (e == cudaSuccess) e = cudaStreamSynchronize(st);   // row is on the stack
    return e;
}

GpEncoder::GpEncoder() { memset(ev_, 0, sizeof ev_); memset(evx_, 0, sizeof evx_); }
GpEncoder::~GpEncoder() {
    for (auto &e : ev_)
        if (e) cudaEventDestroy(e);
    for (auto &e : evx_)
        if (e) cudaEventDestroy(e);
    if (hiSt_) cudaStreamDestroy(hiSt_);
    if (vbLuts_) cudaFree(vbLuts_);
    if (vbLuts10_) cudaFree(vbLuts10_);
}

// k_dp (k_dp.cuh) takes the cheapest partition of finite J at every node; at a node with none it keeps
// PARTITION_SPLIT, down to 4x4 leaves.  Such leaves have no analysis record to code from: J = GP_INF and ncand 0
// with cand / tx never written (blocks the flat or variance pruning skipped), or a shape the analysis does not
// evaluate (I.shNbx 0: the record index lands in another shape's).  k_recon then codes them from whatever the device
// memory holds (stale data of earlier batches): out-of-range modes and tx types, illegal addresses, streams that
// differ from the reconstruction.  dp_complete runs k_dp's choice on finiteness alone, for a superblock at every
// distance from the frame's right and bottom edges (the coded size is a multiple of 8), with every block that the
// pruning may skip skipped (a flat frame prunes every complete 16x16 cell; any block may exceed the variance limit):
// true if every superblock of every frame keeps a partition of finite J.  (k_dp's safety net, dp_net8, catches what
// this model misses: it gives the 4x4 leaves of such a node a DC_PRED record and counts them, GpTimes::dpRepaired, or
// fails the frame where the 4x4 shape is not analysed.)
// Rule for any future pruning or skipping of analysis blocks:
// - pruning that depends only on the options and the source (as prune_flat and the variance pruning) must be mirrored
//   in blk() below, as "the block may have no record", and switched off in init where it makes dp_complete fail;
// - skipping driven by a DP (analyse only what a coarser stage's partition needs) is allowed only under a partition
//   that a stage DP with the same GpParams found finite, so that the final DP keeps a finite partition there.
static bool dp_complete(const GpParams &P, bool flat, bool var) {
    for (int FH = 8; FH <= 64; FH += 8)
        for (int FW = 8; FW <= 64; FW += 8) {
            // the block's record has candidates (finite J): its shape analysed (I.shNbx), inside, not pruned
            auto blk = [&](int bs, int x, int y) {
                if (bs >= AV1G_BLOCK_INVALID || (bs >= 13 && bs <= 15) || !((P.shapeMask >> bs) & 1) || x >= FW || y >= FH)
                    return false;
                const int a = 16 * AV1G_T(Num_4x4_Blocks_Wide)[bs] * AV1G_T(Num_4x4_Blocks_High)[bs];
                if (flat && a < 256 && ((x >> 4) + 1) * 16 <= FW && ((y >> 4) + 1) * 16 <= FH) return false;   // prune_flat
                if (var && a >= P.pruneMinArea && !(P.smallKernel && a <= 64)) return false;   // k_analysis' pruneVar
                return true;
            };
            bool fin[85];
            const int lvBase[4] = {0, 1, 5, 21};
            for (int L = 3; L >= 0; L--) {
                const int n = 64 >> L, half = n >> 1, q4 = n >> 2, per = 1 << L;
                const int sq = n == 8 ? AV1G_BLOCK_8X8 : n == 16 ? AV1G_BLOCK_16X16 : n == 32 ? AV1G_BLOCK_32X32 : AV1G_BLOCK_64X64;
                auto sub = [&](int p) { return (int)AV1G_T(Partition_Subsize)[p][sq]; };
                const int pmask = n == 8 ? P.partMask8 : P.partMaskN;
                auto on = [&](int p) { return ((pmask >> p) & 1) != 0; };
                for (int ny = 0; ny < per; ny++)
                    for (int nx = 0; nx < per; nx++) {
                        bool &f = fin[lvBase[L] + ny * per + nx];
                        const int x = nx * n, y = ny * n;
                        if (x >= FW || y >= FH) { f = true; continue; }   // outside: J 0, no leaves
                        const bool hasRows = (y + half) < FH, hasCols = (x + half) < FW, big = n > P.maxPart;
                        bool split = true;   // SPLIT, always a choice
                        if (n == 8) {
                            for (int k = 0; k < 4; k++) {
                                const int xx = x + (k & 1) * 4, yy = y + (k >> 1) * 4;
                                if (xx < FW && yy < FH) split = split && blk(AV1G_BLOCK_4X4, xx, yy);
                            }
                        } else {
                            const int cb = lvBase[L + 1], cper = per * 2;
                            for (int k = 0; k < 4; k++) split = split && fin[cb + (ny * 2 + (k >> 1)) * cper + nx * 2 + (k & 1)];
                        }
                        f = split;
                        if (big) continue;
                        const int hz = sub(AV1G_PARTITION_HORZ), vt = sub(AV1G_PARTITION_VERT);
                        if (hasRows && hasCols) {
                            const int sp = sub(AV1G_PARTITION_SPLIT);
                            f = f || (on(AV1G_PARTITION_NONE) && blk(sq, x, y));
                            f = f || (on(AV1G_PARTITION_HORZ) && blk(hz, x, y) && blk(hz, x, y + half));
                            f = f || (on(AV1G_PARTITION_VERT) && blk(vt, x, y) && blk(vt, x + half, y));
                            if (n > 8) {
                                const bool s00 = blk(sp, x, y), s01 = blk(sp, x + half, y), s10 = blk(sp, x, y + half),
                                           s11 = blk(sp, x + half, y + half);
                                f = f || (on(AV1G_PARTITION_HORZ_A) && s00 && s01 && blk(hz, x, y + half));
                                f = f || (on(AV1G_PARTITION_HORZ_B) && blk(hz, x, y) && s10 && s11);
                                f = f || (on(AV1G_PARTITION_VERT_A) && s00 && s10 && blk(vt, x + half, y));
                                f = f || (on(AV1G_PARTITION_VERT_B) && blk(vt, x, y) && s01 && s11);
                                const int h4 = sub(AV1G_PARTITION_HORZ_4), v4 = sub(AV1G_PARTITION_VERT_4);
                                bool fh = on(AV1G_PARTITION_HORZ_4), fv = on(AV1G_PARTITION_VERT_4);
                                for (int i = 0; i < 4; i++) {
                                    if (i < 3 || y + 3 * q4 < FH) fh = fh && blk(h4, x, y + i * q4);
                                    if (i < 3 || x + 3 * q4 < FW) fv = fv && blk(v4, x + i * q4, y);
                                }
                                f = f || fh || fv;
                            }
                        } else if (hasCols) {   // split_or_horz (no partition mask, as k_dp)
                            f = f || blk(hz, x, y);
                        } else if (hasRows) {   // split_or_vert
                            f = f || blk(vt, x, y);
                        }
                    }
            }
            if (!fin[0]) return false;
        }
    return true;
}

void gp_warn_dp_repaired(unsigned long long n) {
    fprintf(stderr, "[av1gpu] warning: dp safety net: %llu 4x4 blocks had no analysis record and were coded from DC_PRED "
                    "(a gap in the encoder's partition check; please report the options)\n", n);
}

int gp_check_partitions(const GpParams &P, std::string &err) {
    if (dp_complete(P, false, false)) return 0;
    err = "--shapes / --part8 / --partN / --maxpart leave no partition the analysis evaluates for some blocks (at some "
          "frame sizes at least; such options are refused for every frame)";
    return -1;
}

int GpEncoder::init(const GpConfig &cfg, std::string &err) {
    cfg_ = cfg;
    // candidate counts clamped as av1gpu's parser does (K <= 0 from a library caller left every record without a
    // candidate, but with a finite J where the tx search ran)
    cfg_.P.K = std::max(1, std::min(cfg_.P.K, GP_MAXK));
    cfg_.P.Krecheck = std::max(1, std::min(cfg_.P.Krecheck, cfg_.P.K));
    // the analysis must leave k_dp a partition of finite J in every superblock (dp_complete).  The flat / variance
    // pruning count on larger / smaller blocks to take over the ones they skip; where the partition options leave none
    // (--maxpart below 16 with the flat pruning of turbo / faster, --partN / --shapes without them, ...) the analysis
    // evaluates every block instead.  Options that leave no partition even then are refused.
    if (!cfg_.lossless) {
        GpParams &P = cfg_.P;
        if (gp_check_partitions(P, err)) return -1;
        const bool fl = P.pruneFlat > 0.f, vr = P.pruneVar > 0.f;
        // debug: GP_DP_UNCHECKED=1 keeps the pruning anyway, for k_dp's safety net to act (tests/maxpart_test.py)
        static const bool unchecked = getenv("GP_DP_UNCHECKED") && atoi(getenv("GP_DP_UNCHECKED")) != 0;
        if ((fl || vr) && !unchecked && !dp_complete(P, fl, vr)) {
            const char *what;
            if (fl && dp_complete(P, false, vr)) {
                P.pruneFlat = 0.f;
                what = "the flat pruning (--prune-flat, on in turbo, turbo2, turbo3 and faster) is off: the partition options "
                       "(--maxpart below 16, --partN without NONE, --shapes without 16x16, ...) give some flat 16x16 cells no "
                       "block it keeps";
            } else if (vr && dp_complete(P, fl, false)) {
                P.pruneVar = 0.f;
                what = "--prune-var is off: with the partition options, blocks of --prune-area pixels and up are needed "
                       "where they exceed the variance limit";
            } else {
                P.pruneFlat = 0.f;
                P.pruneVar = 0.f;
                what = "the flat pruning (--prune-flat) and --prune-var are off: together they leave some blocks no "
                       "partition the analysis evaluates";
            }
            if (cfg_.verbose) {   // once per message (one per encoder otherwise)
                static std::mutex mu;
                static std::set<std::string> said;
                std::lock_guard<std::mutex> lk(mu);
                if (said.insert(what).second) fprintf(stderr, "[av1gpu] %s; every block is analysed instead\n", what);
            }
        }
    }
    char buf[256] = {0};
    if (gp_init_tables(buf, sizeof buf)) { err = buf; return -1; }
    // the closed loop for the frames this configuration makes (colour frames, and monochrome ones without 4:4:4
    // frames beside them, as 4:2:0); encode prepares any other one on first use
    if (!cfg_.lossless && gp_recon_prepare(cfg_.depth == 10 ? 10 : 8, !cfg_.yuv444, cfg_.yuv444 != 0)) {
        err = "k_recon table upload failed";
        return -1;
    }
    for (auto &e : ev_) GPCK(cudaEventCreate(&e));
    for (auto &e : evx_) GPCK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    {
        int lo = 0, hi = 0;
        cudaDeviceGetStreamPriorityRange(&lo, &hi);
        GPCK(cudaStreamCreateWithPriority(&hiSt_, cudaStreamNonBlocking, hi));
    }
    if (cfg_.vb) {
        GPCK(cudaMalloc(&vbLuts_, 256 * 1024));
        GPCK(gp_vb_lut_row(vbLuts_, cfg_.qindex, cfg_.vbStrength, 8, 0));
        GPCK(cudaDeviceSynchronize());
    }
    // keep freed stream-ordered allocations cached between batches
    cudaMemPool_t pool;
    if (cudaDeviceGetDefaultMemPool(&pool, 0) == cudaSuccess) {
        uint64_t thr = UINT64_MAX;
        cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &thr);
    }
    return 0;
}

namespace {
struct Alloc {
    std::vector<void *> ptrs;
    cudaStream_t st;
    size_t bytes = 0;
    template <class T>
    cudaError_t get(T *&p, size_t n) {
        void *v = nullptr;
        cudaError_t e = cudaMallocAsync(&v, std::max<size_t>(n, 1) * sizeof(T), st);
        if (e == cudaSuccess) { ptrs.push_back(v); bytes += n * sizeof(T); }
        // debug: GP_POISON=<byte> fills every allocation (uninitialised-read detector)
        static const char *poison = getenv("GP_POISON");
        if (e == cudaSuccess && poison) e = cudaMemsetAsync(v, atoi(poison), std::max<size_t>(n, 1) * sizeof(T), st);
        p = (T *)v;
        return e;
    }
    void release() {
        for (void *p : ptrs) cudaFreeAsync(p, st);
        ptrs.clear();
    }
};
}  // namespace

int GpEncoder::encode(const std::vector<GpFrameIn> &in, std::vector<GpFrameOut> &out, GpTimes &t, std::string &err,
                      cudaStream_t st, GpDevRecords *dev) {
    t = GpTimes();
    const int n = (int)in.size();
    out.assign(n, GpFrameOut());
    if (!n) return 0;
    // lossless: its own search parameters (every shape and partition; J in bits * the q-0 lambda, see k_lossless.cu)
    const bool ll = cfg_.lossless != 0;
    GpParams Pll = cfg_.P;
    if (ll) {
        Pll.shapeMask = 0x3F1FFF; Pll.partMask8 = 0xF; Pll.partMaskN = 0x3FF; Pll.maxPart = 64;
        Pll.splitBias = 1.f; Pll.lambdaMul = 1.f; Pll.pruneFlat = 0.f; Pll.pruneVar = 0.f; Pll.filterIntra = 1;
    }
    const GpParams &P = ll ? Pll : cfg_.P;
    cudaEventRecord(ev_[0], st);
    std::vector<GpImage> imgs(n);
    std::vector<av1f_tuneiq_frame> tiq(n);
    uint32_t shTotal[GP_NSHAPES] = {0};
    std::vector<uint32_t> shStart((size_t)GP_NSHAPES * (n + 1), 0);
    uint32_t nSb = 0;
    std::vector<uint32_t> sbImg;
    std::vector<av1g_tiles> tiles(n);
    std::vector<std::array<int, 2>> tileReq(n);
    // bit depth of the batch: every frame the same (the kernels are instantiated per bit depth)
    const int bd = in[0].bitDepth == 10 ? 10 : 8, bps = bd == 8 ? 1 : 2;
    for (int i = 0; i < n; i++)
        if ((in[i].bitDepth == 10 ? 10 : 8) != bd) { err = "frames of different bit depths in one batch"; return -1; }
    for (int i = 0; i < n; i++) {
        const GpFrameIn &f = in[i];
        GpImage &I = imgs[i];
        memset(&I, 0, sizeof I);
        I.W = f.W; I.H = f.H;
        I.MiCols = av1g_mi_cols(f.W); I.MiRows = av1g_mi_rows(f.H);
        I.sbCols = (I.MiCols + 15) >> 4; I.sbRows = (I.MiRows + 15) >> 4;
        if (f.aw < I.sbCols * 64 || f.ah < I.sbRows * 64) { err = "input planes not padded to 64"; return -1; }
        I.mono = f.mono;
        I.ssx = I.ssy = (f.yuv444 && !f.mono) ? 0 : 1;
        I.baseQ = ll ? 0 : cfg_.qindex;
        I.qctx = av1g_coeff_cdf_q_ctx(I.baseQ);
        I.dqRes = cfg_.deltaqRes;
        I.edgeFilter = cfg_.edgeFilter;
        I.dqUDc = cfg_.dqUDc; I.dqUAc = cfg_.dqUAc; I.dqVDc = cfg_.dqVDc; I.dqVAc = cfg_.dqVAc;
        if (ll) I.dqUDc = I.dqUAc = I.dqVDc = I.dqVAc = 0;
        if (cfg_.tuneIq && !ll) {
            av1f_tuneiq_frame_init(cfg_.qindex, I.ssx, I.ssy, f.mono, &tiq[i], bd);   // 4:4:4: libaom's 4:4:4 chroma delta-q / QM
            I.dqUDc = tiq[i].dq_u_dc; I.dqUAc = tiq[i].dq_u_ac; I.dqVDc = tiq[i].dq_v_dc; I.dqVAc = tiq[i].dq_v_ac;
        }
        if (cfg_.qm != -1 && !ll) {
            I.usingQm = 1;
            const int ly = cfg_.qm == -2 ? av1g_qm_level_allintra(cfg_.qindex) : cfg_.qm;
            const int lc = cfg_.qmUV >= 0 ? cfg_.qmUV : (cfg_.qm == -2 ? av1g_qm_level_allintra(cfg_.qindex + cfg_.dqUAc) : ly);
            I.qm[0] = ly; I.qm[1] = lc; I.qm[2] = lc;
        } else {
            I.qm[0] = I.qm[1] = I.qm[2] = 15;
        }
        if (cfg_.tuneIq && !ll) {
            I.usingQm = 1;
            I.qm[0] = tiq[i].qm_y; I.qm[1] = tiq[i].qm_u; I.qm[2] = tiq[i].qm_v;
        }
        for (int p = 0; p < 3; p++) { I.src[p] = f.dsrc[p]; I.spitch[p] = f.spitch[p]; }
        I.aw = f.aw; I.ah = f.ah;
        // tiles: minimum legal, or (cfg.tileMinMp) extra uniform tiles for large images (more entropy tiles in flight)
        {
            int tc = -1, tr = -1;
            if (ll && cfg_.llTileArea > 0) {
                // lossless: ~ every sample is coded, so a frame has many more symbols; tiles of at most llTileArea
                // pixels keep the per-tile entropy coders short (each tile costs ~0.1-0.4 % in size)
                av1g_tiles t0, tt;
                av1g_tile_layout(f.W, f.H, -1, -1, &t0);
                tc = t0.cols_log2; tr = t0.rows_log2;
                for (;;) {
                    if (av1g_tile_layout(f.W, f.H, tc, tr, &tt) != AV1G_OK) break;
                    const int tw = (int)tt.mi_col_starts[1] * 4, th = (int)tt.mi_row_starts[1] * 4;
                    if ((double)std::min(tw, (int)f.W) * std::min(th, (int)f.H) <= cfg_.llTileArea) break;
                    av1g_tiles tn;
                    if (tw >= th && av1g_tile_layout(f.W, f.H, tc + 1, tr, &tn) == AV1G_OK && tn.cols > tt.cols) tc++;
                    else if (av1g_tile_layout(f.W, f.H, tc, tr + 1, &tn) == AV1G_OK && tn.rows > tt.rows) tr++;
                    else if (av1g_tile_layout(f.W, f.H, tc + 1, tr, &tn) == AV1G_OK && tn.cols > tt.cols) tc++;
                    else break;
                }
                if (av1g_tile_layout(f.W, f.H, tc, tr, &tiles[i]) != AV1G_OK) { tc = -1; tr = -1; }
            } else if (cfg_.tileMinMp > 0 && (double)f.W * f.H >= cfg_.tileMinMp * 1e6) {
                av1g_tiles t0;
                av1g_tile_layout(f.W, f.H, -1, -1, &t0);
                tc = std::max(t0.cols_log2, cfg_.tileColsLog2);
                tr = std::max(t0.rows_log2, cfg_.tileRowsLog2);
                if (av1g_tile_layout(f.W, f.H, tc, tr, &tiles[i]) != AV1G_OK) { tc = -1; tr = -1; }
            }
            if (tc < 0 && av1g_tile_layout(f.W, f.H, -1, -1, &tiles[i]) != AV1G_OK) { err = "tile layout"; return -1; }
            tileReq[i][0] = tc; tileReq[i][1] = tr;
        }
        I.tileCols = (int)tiles[i].cols; I.tileRows = (int)tiles[i].rows;
        for (uint32_t k = 0; k <= tiles[i].cols; k++) I.tileColStart[k] = (int)tiles[i].mi_col_starts[k];
        for (uint32_t k = 0; k <= tiles[i].rows; k++) I.tileRowStart[k] = (int)tiles[i].mi_row_starts[k];
        I.sbBase = nSb;
        nSb += (uint32_t)(I.sbCols * I.sbRows);
        for (int k = 0; k < I.sbCols * I.sbRows; k++) sbImg.push_back((uint32_t)i);
    }
    // ---- screen-content tools (palette mode): per frame, the detection (on the GPU, after the image upload) or --palette
    // on; frames without them encode as before (no palette syntax at all) ----
    const bool palOn = cfg_.palette && gp_palette_supported(cfg_);
    bool anySct = palOn && cfg_.palette == 2;   // known on the host: --palette on, or (lossless) after the detection
    if (palOn) {
        uint32_t b0 = 0;
        for (int i = 0; i < n; i++) {
            imgs[i].scBlk0 = b0;
            b0 += (uint32_t)((imgs[i].W / 16) * (imgs[i].H / 16));
            imgs[i].sct = cfg_.palette == 2;
        }
    }
    uint32_t shapeBase[GP_NSHAPES];
    uint64_t blkTotal = 0;
    for (int s = 0; s < GP_NSHAPES; s++) {
        shapeBase[s] = (uint32_t)blkTotal;
        const bool en = (P.shapeMask >> s) & 1;
        for (int i = 0; i < n; i++) {
            GpImage &I = imgs[i];
            shStart[(size_t)s * (n + 1) + i] = shTotal[s];
            if (en && s < AV1G_BLOCK_INVALID && !(s >= 13 && s <= 15)) {
                const int w = 4 * AV1G_T(Num_4x4_Blocks_Wide)[s], h = 4 * AV1G_T(Num_4x4_Blocks_High)[s];
                I.shNbx[s] = (I.MiCols * 4 + w - 1) / w;
                I.shNby[s] = (I.MiRows * 4 + h - 1) / h;
            }
            I.shOff[s] = shapeBase[s] + shTotal[s];
            shTotal[s] += (uint32_t)(I.shNbx[s] * I.shNby[s]);
        }
        shStart[(size_t)s * (n + 1) + n] = shTotal[s];
        blkTotal += shTotal[s];
    }
    // ---- device allocations ----
    Alloc A;
    A.st = st;
    size_t planeBytes = 0, miTotal = 0, ctxTotal = 0, sbqTotal = 0;
    bool any444 = false, any420 = false, anyMono = false;
    for (int i = 0; i < n; i++) {
        const GpImage &I = imgs[i];
        planeBytes += ((size_t)in[i].aw * in[i].ah + 2 * (size_t)(in[i].aw >> I.ssx) * (in[i].ah >> I.ssy)) * bps;
        miTotal += (size_t)I.MiCols * I.MiRows;
        ctxTotal += (size_t)I.MiCols * I.MiRows + 2 * (size_t)(I.MiCols >> I.ssx) * (I.MiRows >> I.ssy);
        sbqTotal += (size_t)I.sbCols * I.sbRows;
        if (!I.mono) (I.ssx ? any420 : any444) = true;
        anyMono |= I.mono != 0;
    }
    // the closed-loop instantiations of the wavefront's two groups (monochrome frames join the 4:4:4 one when there is one)
    if (!ll && gp_recon_prepare(bd, any420 || (anyMono && !any444), any444)) {
        err = "k_recon table upload failed";
        return -1;
    }
    uint8_t *dRec, *dMaps, *dCtx, *dSbq = nullptr;
    GPCK(A.get(dRec, planeBytes));
    // the closed loop's 4x4 maps: with device records of a lossy batch they outlive the encode (trusted records for the
    // entropy stage: DevBatch::miSize .. ctxDc, its own allocation handed to dev), else in the batch's arena
    // (the four the entropy stage reads first, the UV mode map, which it does not, after them or in the arena)
    const bool keepMaps = dev && !ll;
    uint8_t *mapsMem = nullptr, *dUV = nullptr;
    struct MapsGuard {   // frees the kept maps on an error return (set to null once dev owns them)
        uint8_t *&p;
        cudaStream_t s;
        ~MapsGuard() { if (p) cudaFreeAsync(p, s); }
    } mapsGuard{mapsMem, st};
    const size_t mapsAt = (miTotal * 4 + 255) & ~(size_t)255;
    if (keepMaps) {
        GPCK(cudaMallocAsync((void **)&mapsMem, mapsAt + ctxTotal * 2, st));
        dMaps = mapsMem;
        dCtx = mapsMem + mapsAt;
        GPCK(A.get(dUV, miTotal));
    } else {
        GPCK(A.get(dMaps, miTotal * 5));
        GPCK(A.get(dCtx, ctxTotal * 2));
        dUV = dMaps + 4 * miTotal;
    }
    GPCK(cudaMemsetAsync(dMaps, 0, miTotal * 4, st));
    GPCK(cudaMemsetAsync(dUV, 0, miTotal, st));
    GPCK(cudaMemsetAsync(dCtx, 0, ctxTotal * 2, st));
    std::vector<uint8_t> hSbq;
    const bool anySbq = !ll && cfg_.deltaq && std::any_of(in.begin(), in.end(), [](const GpFrameIn &f) { return !f.sbQ.empty(); });
    if (anySbq) {
        GPCK(A.get(dSbq, sbqTotal));
        hSbq.assign(sbqTotal, (uint8_t)cfg_.qindex);
    }
    {
        size_t pr = 0, pm = 0, pc = 0, pq = 0;
        for (int i = 0; i < n; i++) {
            GpImage &I = imgs[i];
            // bytes (10-bit: 2 per sample); the pitches are in samples
            const size_t ysz = (size_t)in[i].aw * in[i].ah * bps, csz = (size_t)(in[i].aw >> I.ssx) * (in[i].ah >> I.ssy) * bps;
            I.rec[0] = dRec + pr; I.rpitch[0] = in[i].aw;
            I.rec[1] = dRec + pr + ysz; I.rpitch[1] = in[i].aw >> I.ssx;
            I.rec[2] = dRec + pr + ysz + csz; I.rpitch[2] = in[i].aw >> I.ssx;
            pr += ysz + 2 * csz;
            const size_t mi = (size_t)I.MiCols * I.MiRows;
            I.mYMode = dMaps + pm; I.mSkip = dMaps + miTotal + pm; I.mMiSize = dMaps + 2 * miTotal + pm;
            I.mTx = dMaps + 3 * miTotal + pm; I.mUVMode = dUV + pm;
            pm += mi;
            const size_t c1 = (size_t)(I.MiCols >> I.ssx) * (I.MiRows >> I.ssy);
            I.cLvl[0] = dCtx + pc; I.cLvl[1] = dCtx + pc + mi; I.cLvl[2] = dCtx + pc + mi + c1;
            I.cDc[0] = dCtx + ctxTotal + pc; I.cDc[1] = dCtx + ctxTotal + pc + mi; I.cDc[2] = dCtx + ctxTotal + pc + mi + c1;
            pc += mi + 2 * c1;
            const size_t nsb = (size_t)I.sbCols * I.sbRows;
            if (anySbq) {
                I.sbQ = dSbq + pq;
                if (!in[i].sbQ.empty()) {
                    if (in[i].sbQ.size() != nsb) { err = "sbQ map size mismatch"; A.release(); return -1; }
                    av1g_legalize_sb_qindex(cfg_.qindex, cfg_.deltaqRes, in[i].sbQ.data(), nsb, hSbq.data() + pq);
                }
                pq += nsb;
            }
        }
    }
    if (anySbq) GPCK(cudaMemcpyAsync(dSbq, hSbq.data(), sbqTotal, cudaMemcpyHostToDevice, st));
    if (P.pruneFlat > 0.f) {
        size_t cells = 0;
        for (int i = 0; i < n; i++) cells += (size_t)((imgs[i].MiCols + 3) / 4) * ((imgs[i].MiRows + 3) / 4);
        float *dV;
        GPCK(A.get(dV, cells));
        size_t at = 0;
        for (int i = 0; i < n; i++) {
            GpImage &I = imgs[i];
            I.cols16 = (I.MiCols + 3) / 4; I.rows16 = (I.MiRows + 3) / 4;
            I.var16 = dV + at;
            gp_launch_var16(in[i].dsrc[0], in[i].spitch[0], I.MiCols, I.MiRows, dV + at, st, bd);
            at += (size_t)I.cols16 * I.rows16;
        }
    }
    if (cfg_.ssimLambda && !ll) {
        size_t cells = 0;
        for (int i = 0; i < n; i++) cells += (size_t)((imgs[i].MiCols + 3) / 4) * ((imgs[i].MiRows + 3) / 4);
        float *dS;
        double *dSum;
        GPCK(A.get(dS, cells));
        GPCK(A.get(dSum, n));
        size_t at = 0;
        for (int i = 0; i < n; i++) {
            GpImage &I = imgs[i];
            I.cols16 = (I.MiCols + 3) / 4; I.rows16 = (I.MiRows + 3) / 4;
            I.ssimLog = dS + at;
            gp_launch_ssim_map(in[i].dsrc[0], in[i].spitch[0], I.MiCols, I.MiRows, dS + at, dSum + i, st, bd);
            at += (size_t)I.cols16 * I.rows16;
        }
        if (getenv("GP_DEBUG_SSIM")) {
            std::vector<float> h(cells);
            cudaMemcpyAsync(h.data(), dS, cells * 4, cudaMemcpyDeviceToHost, st);
            cudaStreamSynchronize(st);
            float mn = 1e9f, mx = -1e9f;
            double s = 0;
            for (float v : h) { mn = std::min(mn, v); mx = std::max(mx, v); s += v; }
            fprintf(stderr, "ssim map: %zu cells, log min %.3f max %.3f mean %.4f -> scale %.3f .. %.3f\n", cells, mn, mx, s / cells,
                    expf(mn), expf(mx));
        }
    }
    std::vector<int> dqResLog2(n, 0);
    if (cfg_.vb && !anySbq && !ll) {
        // av1f GPU variance boost: per-SB qindex, legal for delta_q_res (libaom tune=iq deltaq-mode 6)
        std::vector<av1f_vb_desc> vd(n);
        for (int i = 0; i < n; i++) {
            const GpImage &I = imgs[i];
            vd[i].y = in[i].dsrc[0]; vd[i].pitch = in[i].spitch[0];
            vd[i].width = I.W; vd[i].height = I.H; vd[i].base_q_idx = cfg_.qindex;
            vd[i].tile_cols_log2 = tiles[i].cols_log2; vd[i].tile_rows_log2 = tiles[i].rows_log2;
            vd[i].sb_off = I.sbBase;
            vd[i].bit_depth = bd;
        }
        av1f_vb_desc *dVd;
        GPCK(A.get(dVd, n));
        GPCK(A.get(dSbq, sbqTotal));
        GPCK(cudaMemcpyAsync(dVd, vd.data(), n * sizeof(av1f_vb_desc), cudaMemcpyHostToDevice, st));
        if (bd == 10 && !vbLuts10_) {   // 10-bit LUTs on first use (libaom's 10-bit q tables)
            GPCK(cudaMalloc(&vbLuts10_, 256 * 1024));
            if (gp_vb_lut_row(vbLuts10_, cfg_.qindex, cfg_.vbStrength, 10, st)) { err = "variance-boost LUT failed"; A.release(); return -1; }
        }
        if (bd == 10)
            GPCK((cudaError_t)av1f_varboost_bd(dVd, n, (uint32_t)sbqTotal, vbLuts10_, cfg_.vbOrder, 10, nullptr, nullptr, dSbq, st));
        else
            GPCK((cudaError_t)av1f_varboost(dVd, n, (uint32_t)sbqTotal, vbLuts_, cfg_.vbOrder, nullptr, nullptr, dSbq, st));
        const int res = av1f_delta_q_res(cfg_.qindex);
        for (int i = 0; i < n; i++) {
            imgs[i].sbQ = dSbq + imgs[i].sbBase;
            imgs[i].dqRes = res;
            dqResLog2[i] = av1f_log2i(res);
        }
    } else if (anySbq) {
        for (int i = 0; i < n; i++) dqResLog2[i] = cfg_.deltaqRes;
    }
    GpBatch B;
    memset(&B, 0, sizeof B);
    B.nimg = n;
    B.nSb = nSb;
    B.sbTxbCap = any444 ? GP_MAX_TXB_444 : GP_MAX_TXB;
    B.sbCoefCap = any444 ? GP_SB_COEFFS_444 : GP_SB_COEFFS;
    B.any420 = any420; B.any444 = any444;
    GPCK(A.get(B.img, n));
    GPCK(A.get(B.blk, blkTotal));
    GPCK(A.get(B.shImgStart, shStart.size()));
    GPCK(A.get(B.sbImg, nSb));
    GPCK(A.get(B.leafCountW, nSb));
    GPCK(A.get(B.leaves, (size_t)nSb * GP_MAX_LEAVES));
    GPCK(A.get(B.sbJ, nSb));
    GPCK(A.get(B.sbOut, nSb));
    GPCK(A.get(B.oBlocks, (size_t)nSb * GP_MAX_LEAVES));
    GPCK(A.get(B.oBlockTxb, (size_t)nSb * GP_MAX_LEAVES));
    GPCK(A.get(B.oTxb, (size_t)nSb * B.sbTxbCap));
    GPCK(A.get(B.oCoeffs, (size_t)nSb * B.sbCoefCap));
    // live 16x16 cells for the analysis of the shapes below 16x16, with the flat-area pruning (gp_launch_live16 fills the
    // list on the GPU; without pruning the shapes' raster order measured faster)
    GpLive live;
    memset(&live, 0, sizeof live);
    std::vector<uint32_t> cellStart;
    if (!ll && P.pruneFlat > 0.f) {
        cellStart.assign(n + 1, 0);
        for (int i = 0; i < n; i++)
            cellStart[i + 1] = cellStart[i] + (uint32_t)(((imgs[i].MiCols + 3) / 4) * ((imgs[i].MiRows + 3) / 4));
        live.nCells = cellStart[n];
        uint32_t *dCs;
        GPCK(A.get(dCs, n + 1));
        GPCK(A.get(live.cell, 2 * (size_t)live.nCells));
        GPCK(A.get(live.count, 1));
        GPCK(cudaMemcpyAsync(dCs, cellStart.data(), (n + 1) * sizeof(uint32_t), cudaMemcpyHostToDevice, st));
        live.cellStart = dCs;
    }
    unsigned long long *dScCnt = nullptr;
    uint32_t *dPalAny = nullptr;
    if (palOn) {
        GPCK(A.get(dScCnt, (size_t)3 * n));
        GPCK(A.get(dPalAny, 1));
        B.palAny = dPalAny;
    }
    // palette slots: lossy batches get them whenever palettes are possible, so the detection can decide on the GPU
    // without a host sync (k_pal and the closed loop read the images' sct, photos pay nothing else); lossless batches
    // only with a screen-content frame (below), because their kernels are instantiated with or without palettes
    auto palSlots = [&]() -> cudaError_t {
        cudaError_t e = A.get(B.oPal, (size_t)nSb * GP_SB_PAL);
        return e == cudaSuccess ? A.get(B.oMap, (size_t)nSb * GP_SB_MAP) : e;
    };
    if (palOn && (!ll || anySct)) GPCK(palSlots());
    GPCK(A.get(B.stats, 16));
    GPCK(A.get(B.prof, 16));
    GPCK(cudaMemsetAsync(B.prof, 0, 16 * sizeof(unsigned long long), st));
    B.sbCyc = nullptr;
    if (P.profile && getenv("GP_SBCYC_DUMP")) GPCK(A.get(B.sbCyc, nSb));
    GPCK(cudaMemsetAsync(B.stats, 0, 16 * sizeof(uint32_t), st));
    GPCK(cudaMemcpyAsync(B.img, imgs.data(), n * sizeof(GpImage), cudaMemcpyHostToDevice, st));
    if (palOn && cfg_.palette == 1) {
        gp_launch_sc_detect(B.img, imgs.data(), n, bd, dScCnt, dPalAny, st);
        if (ll) {
            uint32_t any = 0;
            GPCK(cudaMemcpyAsync(&any, dPalAny, 4, cudaMemcpyDeviceToHost, st));
            GPCK(cudaStreamSynchronize(st));
            if (any) {
                anySct = true;
                GPCK(palSlots());
            }
        }
    } else if (palOn) GPCK(cudaMemsetAsync(dPalAny, 1, 4, st));   // --palette on: every frame
    GPCK(cudaMemcpyAsync(B.shImgStart, shStart.data(), shStart.size() * 4, cudaMemcpyHostToDevice, st));
    GPCK(cudaMemcpyAsync(B.sbImg, sbImg.data(), nSb * 4, cudaMemcpyHostToDevice, st));
    // wavefront step lists
    std::vector<std::vector<uint32_t>> steps, steps444;
    // wavefront per tile (tiles are independent for prediction and contexts): step = col + 2 * row
    // relative to the tile origin, so every tile of every image starts at step 0.  k_recon is instantiated
    // per chroma layout: each step lists the SBs of 4:2:0 frames first, then those of 4:4:4 frames
    // (monochrome frames join the 4:4:4 group when the batch has 4:4:4 frames: one launch per step)
    for (int i = 0; i < n && !ll; i++) {   // (lossless: no wavefront)
        const GpImage &I = imgs[i];
        const av1g_tiles &T = tiles[i];
        const bool k444 = I.mono ? any444 : I.ssx == 0;
        for (uint32_t tr = 0; tr < T.rows; tr++)
            for (uint32_t tc = 0; tc < T.cols; tc++) {
                const int r0 = (int)T.mi_row_starts[tr] >> 4, r1 = ((int)T.mi_row_starts[tr + 1] + 15) >> 4;
                const int c0 = (int)T.mi_col_starts[tc] >> 4, c1 = ((int)T.mi_col_starts[tc + 1] + 15) >> 4;
                for (int r = r0; r < r1; r++)
                    for (int c = c0; c < c1; c++) {
                        const size_t s = (size_t)(c - c0) + 2 * (size_t)(r - r0);
                        if (steps.size() <= s) { steps.resize(s + 1); steps444.resize(s + 1); }
                        (k444 ? steps444 : steps)[s].push_back(I.sbBase + (uint32_t)(r * I.sbCols + c));
                    }
            }
    }
    std::vector<uint32_t> stepFlat, stepOff, step420;   // step420[s]: SBs of the step in the 4:2:0 group
    for (size_t s = 0; s < steps.size(); s++) {
        step420.push_back((uint32_t)steps[s].size());
        steps[s].insert(steps[s].end(), steps444[s].begin(), steps444[s].end());
        stepOff.push_back((uint32_t)stepFlat.size());
        stepFlat.insert(stepFlat.end(), steps[s].begin(), steps[s].end());
    }
    stepOff.push_back((uint32_t)stepFlat.size());
    // boundaries of every step's 4:2:0 and 4:4:4 groups (one k_recon launch each), for k_recon_order after the DP
    std::vector<uint32_t> stepSeg;
    uint32_t maxSeg = 0;
    for (size_t s = 0; s + 1 < stepOff.size(); s++) {
        stepSeg.push_back(stepOff[s]);
        stepSeg.push_back(stepOff[s] + step420[s]);
        maxSeg = std::max({maxSeg, step420[s], stepOff[s + 1] - stepOff[s] - step420[s]});
    }
    stepSeg.push_back(stepOff.back());
    uint32_t *dSteps = nullptr, *dStepSeg = nullptr;
    if (!ll) {
        GPCK(A.get(dSteps, stepFlat.size()));
        GPCK(A.get(dStepSeg, stepSeg.size()));
        size_t maxStep = 1;
        for (auto &v : steps) maxStep = std::max(maxStep, v.size());
        uint8_t *bs;
        GPCK(A.get(bs, maxStep * gp_recon_scratch_bytes()));
        B.bigScratch = bs;
        GPCK(cudaMemcpyAsync(dSteps, stepFlat.data(), stepFlat.size() * 4, cudaMemcpyHostToDevice, st));
        GPCK(cudaMemcpyAsync(dStepSeg, stepSeg.data(), stepSeg.size() * 4, cudaMemcpyHostToDevice, st));
    }
    // lossless: per-plane 4x4 cost tables and the k_ll_cost segments (3 per image: Y, U, V; empty for monochrome chroma)
    GpLossless L;
    memset(&L, 0, sizeof L);
    if (ll) {
        std::vector<uint64_t> segStart(3 * (size_t)n + 1, 0), costOff(3 * (size_t)n, 0);
        uint64_t tot[3] = {0, 0, 0}, thr = 0;
        for (int i = 0; i < n; i++)
            for (int p = 0; p < 3; p++) {
                const GpImage &I = imgs[i];
                const uint64_t n4 = (p && I.mono) ? 0 : (uint64_t)(I.MiCols >> (p ? I.ssx : 0)) * (I.MiRows >> (p ? I.ssy : 0));
                const int ne = p ? LL_NE_UV : LL_NE_Y;
                segStart[(size_t)i * 3 + p] = thr;
                costOff[(size_t)i * 3 + p] = tot[p];
                thr += n4 * ne;
                tot[p] += n4 * ne;
            }
        segStart[3 * (size_t)n] = thr;
        for (int p = 0; p < 3; p++) GPCK(A.get(L.cost[p], tot[p]));
        uint64_t *dSeg, *dOff;
        GPCK(A.get(dSeg, segStart.size()));
        GPCK(A.get(dOff, costOff.size()));
        GPCK(cudaMemcpyAsync(dSeg, segStart.data(), segStart.size() * 8, cudaMemcpyHostToDevice, st));
        GPCK(cudaMemcpyAsync(dOff, costOff.data(), costOff.size() * 8, cudaMemcpyHostToDevice, st));
        L.segStart = dSeg;
        L.costOff = (const uint64_t(*)[3])dOff;
        L.nSeg = 3 * n;
        L.hostTotal = thr;
        L.lambdaScale = 1.f;
        L.cfl = cfg_.losslessCfl;
    }
    cudaEventRecord(ev_[1], st);
    // ---- analysis + DP ----
    if (ll) gp_launch_ll_analysis(B, L, shTotal, cfg_.edgeFilter, P.filterIntra, st, bd);
    else {
        gp_launch_live16(B, P, live, st, bd);
        gp_launch_analysis(B, P, shTotal, st, bd, &live);
        gp_launch_pal_analysis(B, P, shTotal, st, bd);   // palette candidates (batches with screen-content frames)
    }
    GPCK(cudaGetLastError());
    cudaEventRecord(ev_[2], st);
    gp_launch_dp(B, P, st, bd);
    // longest superblocks first in each step; debug: GP_RC_ORDER=0 keeps the list order (the CTA order changes no output)
    static const bool rcOrder = !getenv("GP_RC_ORDER") || atoi(getenv("GP_RC_ORDER")) != 0;
    if (!ll && rcOrder) gp_recon_order(B, dSteps, dStepSeg, (int)stepSeg.size() - 1, maxSeg, st);
    GPCK(cudaGetLastError());
    cudaEventRecord(ev_[3], st);
    // ---- closed loop, SB wavefront ----
    // wavefront on a high-priority stream: when several encoders share the GPU, the latency-bound recon
    // CTAs (critical path) are scheduled before other encoders' analysis CTAs
    if (ll) {   // lossless: decisions + levels of every SB at once (the reconstruction is the source)
        gp_launch_ll_recon(B, L, cfg_.edgeFilter, P.filterIntra, st, bd);
    } else {
        GPCK(cudaEventRecord(evx_[0], st));
        GPCK(cudaStreamWaitEvent(hiSt_, evx_[0], 0));
        for (size_t s = 0; s + 1 < stepOff.size(); s++)
            gp_launch_recon(B, P, dSteps + stepOff[s], step420[s], stepOff[s + 1] - stepOff[s] - step420[s], hiSt_, bd);
        GPCK(cudaEventRecord(evx_[1], hiSt_));
        GPCK(cudaStreamWaitEvent(st, evx_[1], 0));
    }
    GPCK(cudaGetLastError());
    cudaEventRecord(ev_[4], st);
    t.steps = (uint32_t)steps.size();
    t.sbs = nSb;
    // ---- pack ----
    std::vector<GpSbOut> so(nSb);
    GpPinnedStage &PS = gp_pinned();
    if (PS.stage(so.data(), B.sbOut, nSb * sizeof(GpSbOut), st)) GPCK(cudaErrorMemoryAllocation);
    uint32_t stats[16];
    if (PS.stage(stats, B.stats, sizeof stats, st)) GPCK(cudaErrorMemoryAllocation);
    if (PS.stage(t.prof, B.prof, sizeof t.prof, st)) GPCK(cudaErrorMemoryAllocation);
    std::vector<uint32_t> sbCyc;
    if (B.sbCyc) {
        sbCyc.resize(nSb);
        if (PS.stage(sbCyc.data(), B.sbCyc, nSb * 4, st)) GPCK(cudaErrorMemoryAllocation);
    }
    if (PS.flush(st)) GPCK(cudaGetLastError() != cudaSuccess ? cudaGetLastError() : cudaErrorUnknown);
    // k_dp's safety net (dp_net8, k_dp.cuh) flagged superblocks it could not give records: their frames fail (the whole
    // call, as any encode error).  stats[6] stays 0 unless dp_complete missed a case; only then are the costs copied.
    if (stats[6]) {
        std::vector<float> sbJ(nSb);
        GPCK(cudaMemcpyAsync(sbJ.data(), B.sbJ, nSb * sizeof(float), cudaMemcpyDeviceToHost, st));
        GPCK(cudaStreamSynchronize(st));
        std::string fr;
        for (int i = 0; i < n; i++)
            for (uint32_t k = 0; k < (uint32_t)(imgs[i].sbCols * imgs[i].sbRows); k++)
                if (sbJ[imgs[i].sbBase + k] <= GP_SB_FLAGGED) {
                    fr += (fr.empty() ? "" : ", ") + std::to_string(i) + " (" + std::to_string(imgs[i].W) + "x" + std::to_string(imgs[i].H) + ")";
                    break;
                }
        err = "k_dp: " + std::to_string(stats[6]) + " superblock(s) of frame(s) " + fr + " left 8x8 areas without an analysis "
              "record and without a 4x4 shape to give them one (a gap in GpEncoder::init's partition check; please report the "
              "options): not encoded";
        A.release();
        return -1;
    }
    if (B.sbCyc) {
        // per-SB recon latency dump for schedule studies: batch img tile sbr sbc step cycles
        static std::mutex dumpMu;
        static int dumpBatch = 0;
        std::lock_guard<std::mutex> lk(dumpMu);
        if (FILE *f = fopen(getenv("GP_SBCYC_DUMP"), "a")) {
            for (size_t s = 0; s < steps.size(); s++)
                for (uint32_t sbg : steps[s]) {
                    int im = 0;
                    while (im + 1 < n && imgs[im + 1].sbBase <= sbg) im++;
                    const uint32_t loc = sbg - imgs[im].sbBase;
                    const int c = (int)(loc % imgs[im].sbCols), r = (int)(loc / imgs[im].sbCols);
                    const av1g_tiles &T = tiles[im];
                    int tr = 0, tc = 0;
                    while (tr + 1 < (int)T.rows && (int)(T.mi_row_starts[tr + 1] >> 4) <= r) tr++;
                    while (tc + 1 < (int)T.cols && (int)(T.mi_col_starts[tc + 1] >> 4) <= c) tc++;
                    fprintf(f, "%d %d %d %d %d %zu %u %u %u\n", dumpBatch, im, tr * (int)T.cols + tc, r, c, s, sbCyc[sbg],
                            so[sbg].nBlocks, so[sbg].nTxb);
                }
            fclose(f);
        }
        dumpBatch++;
    }
    std::vector<GpPackOff> po(nSb);
    std::vector<uint32_t> imgBlk0(n + 1), imgTxb0(n + 1), imgPal0(n + 1);
    std::vector<uint64_t> imgCoef0(n + 1), imgMap0(n + 1);
    uint32_t gBlk = 0, gTxb = 0, gPal = 0, tileCounter = 0;
    uint64_t gCoef = 0, gMap = 0;
    for (int i = 0; i < n; i++) {
        const GpImage &I = imgs[i];
        imgBlk0[i] = gBlk; imgTxb0[i] = gTxb; imgCoef0[i] = gCoef; imgPal0[i] = gPal; imgMap0[i] = gMap;
        uint32_t rTxb = 0, rCoef = 0;
        const av1g_tiles &T = tiles[i];
        for (uint32_t tr = 0; tr < T.rows; tr++)
            for (uint32_t tc = 0; tc < T.cols; tc++) {
                const uint32_t tileIdx = tileCounter++;
                for (uint32_t r = T.mi_row_starts[tr]; r < T.mi_row_starts[tr + 1]; r += 16)
                    for (uint32_t c = T.mi_col_starts[tc]; c < T.mi_col_starts[tc + 1]; c += 16) {
                        const uint32_t sb = I.sbBase + (r >> 4) * I.sbCols + (c >> 4);
                        GpPackOff &o = po[sb];
                        o.gBlk = gBlk; o.gTxb = gTxb; o.gCoef = gCoef; o.rTxb = rTxb; o.rCoef = rCoef;
                        o.img = (uint32_t)i; o.tile = tileIdx;
                        o.gPal = gPal; o.gMap = gMap; o.rBlk = gBlk - imgBlk0[i]; o.rMap = gMap - imgMap0[i];
                        gBlk += so[sb].nBlocks; gTxb += so[sb].nTxb; gCoef += so[sb].nCoeffs;
                        rTxb += so[sb].nTxb; rCoef += so[sb].nCoeffs;
                        gPal += so[sb].nPal; gMap += so[sb].nMap;
                    }
            }
    }
    imgBlk0[n] = gBlk; imgTxb0[n] = gTxb; imgCoef0[n] = gCoef; imgPal0[n] = gPal; imgMap0[n] = gMap;
    // record arrays: one allocation; handed to the caller (dev) or freed at the end
    auto al = [](size_t x) { return (x + 255) & ~(size_t)255; };
    // (k_pack's other auxiliary arrays - block tile, global first tx block, tx block -> block, -> coefficient - had no
    // reader and are gone; the entropy stage derives its indices itself)
    const size_t szB = al((size_t)gBlk * sizeof(av1g_block)), szBT = al((size_t)gBlk * 4), szAux = al(((size_t)gBlk + 1) * 4);
    const size_t szT = al((size_t)gTxb * sizeof(av1g_txb));
    const size_t szC = al((size_t)gCoef * 2 + 2);
    const size_t szP = gPal ? al((size_t)gPal * sizeof(av1g_palette)) : 0, szM = gMap ? al(gMap) : 0;
    // trusted records: the tx blocks' symbol counts in the symbolizer's item order (blocks + tx blocks + 1)
    const size_t szCnt = keepMaps ? al(((size_t)gBlk + gTxb + 1) * 4) : 0;
    const size_t poolBytes = szB + szBT + szAux + szT + szC + szP + szM + szCnt;
    uint8_t *pool = nullptr;
    GPCK(cudaMallocAsync((void **)&pool, poolBytes, st));
    size_t pp = 0;
    auto take = [&](size_t s) { uint8_t *q = pool + pp; pp += s; return q; };
    av1g_block *dB = (av1g_block *)take(szB);
    uint32_t *dBT = (uint32_t *)take(szBT);
    GpPackAux aux;
    memset(&aux, 0, sizeof aux);
    aux.blockFrame = (uint32_t *)take(szAux);
    av1g_txb *dT = (av1g_txb *)take(szT);
    int16_t *dC = (int16_t *)take(szC);
    av1g_palette *dPal = gPal ? (av1g_palette *)take(szP) : nullptr;
    uint8_t *dMap = gMap ? take(szM) : nullptr;
    uint32_t *dCnt = keepMaps ? (uint32_t *)take(szCnt) : nullptr;
    GpPackOff *dPo;
    GPCK(A.get(dPo, nSb));
    GPCK(cudaMemcpyAsync(dPo, po.data(), nSb * sizeof(GpPackOff), cudaMemcpyHostToDevice, st));
    cudaEventRecord(ev_[5], st);
    gp_launch_pack_counts(B, dPo, dB, dBT, dT, dC, aux, st, dPal, dMap, dCnt, 0 /* fp.reduced_tx_set below */);
    GPCK(cudaGetLastError());
    cudaEventRecord(ev_[8], st);
    // ---- frame parameters ----
    for (int i = 0; i < n; i++) {
        const GpImage &I = imgs[i];
        av1g_frame_params &fp = out[i].fp;
        memset(&fp, 0, sizeof fp);
        fp.version = AV1G_DECISIONS_VERSION;
        fp.width = (uint32_t)I.W; fp.height = (uint32_t)I.H;
        fp.monochrome = (uint8_t)I.mono;
        fp.chroma_format = (uint8_t)(!I.mono && !I.ssx ? AV1G_CHROMA_444 : AV1G_CHROMA_420);
        fp.bit_depth = (uint8_t)bd;
        fp.base_q_idx = (uint8_t)cfg_.qindex;
        fp.tx_mode_select = (uint8_t)(P.txSizeSearch ? 1 : 0);
        fp.reduced_tx_set = 0;
        fp.tile_cols_log2 = (int8_t)tileReq[i][0]; fp.tile_rows_log2 = (int8_t)tileReq[i][1];
        fp.delta_q_u_dc = (int8_t)(I.mono ? 0 : I.dqUDc); fp.delta_q_u_ac = (int8_t)(I.mono ? 0 : I.dqUAc);
        fp.delta_q_v_dc = (int8_t)(I.mono ? 0 : I.dqVDc); fp.delta_q_v_ac = (int8_t)(I.mono ? 0 : I.dqVAc);
        fp.using_qmatrix = (uint8_t)I.usingQm;
        fp.qm_y = (uint8_t)I.qm[0]; fp.qm_u = (uint8_t)I.qm[1]; fp.qm_v = (uint8_t)I.qm[2];
        fp.enable_intra_edge_filter = (uint8_t)cfg_.edgeFilter;
        fp.enable_filter_intra = (uint8_t)cfg_.P.filterIntra;
        fp.delta_q_present = (uint8_t)(I.sbQ != nullptr);
        fp.delta_q_res = (uint8_t)(I.sbQ ? dqResLog2[i] : 0);
        fp.allow_screen_content_tools = (uint8_t)(imgPal0[i + 1] > imgPal0[i]);   // only frames that use a palette
        if (cfg_.lfMode) {
            // libaom LPF_PICK_FROM_Q (key frame, 8-bit): ROUND_POWER_OF_TWO(acq * 17563 - 421574, 18), times lfMul
            int lv = cfg_.lfLevel;
            if (cfg_.lfMode == 1 && bd == 10) {
                // 10 bits: libaom's high-bitdepth fit ROUND_POWER_OF_TWO(acq * 20723 + 4060632, 20) - 4 (key frame)
                const int acq = av1g_ac_q<10>(cfg_.qindex);
                const long v = (((long)acq * 20723 + 4060632 + (1 << 19)) >> 20) - 4;
                lv = (int)lrintf((float)v * cfg_.lfMul);
            } else if (cfg_.lfMode == 1) {
                const int acq = av1g_ac_q(cfg_.qindex);
                const long v = ((long)acq * 17563 - 421574 + (1 << 17)) >> 18;
                lv = (int)lrintf((float)v * cfg_.lfMul);
            }
            lv = std::max(0, std::min(63, lv));
            fp.loop_filter_level[0] = fp.loop_filter_level[1] = (uint8_t)lv;
            fp.loop_filter_level[2] = fp.loop_filter_level[3] = (uint8_t)(I.mono ? 0 : lv);
            fp.loop_filter_sharpness = 0;
        }
        if (cfg_.cdef) {
            fp.enable_cdef = 1;
            fp.cdef_damping_minus_3 = (uint8_t)cfg_.cdefDampingMinus3;
            fp.cdef_bits = 0;
            fp.cdef_y_pri[0] = (uint8_t)cfg_.cdefYPri; fp.cdef_y_sec[0] = (uint8_t)cfg_.cdefYSec;
            fp.cdef_uv_pri[0] = (uint8_t)cfg_.cdefUVPri; fp.cdef_uv_sec[0] = (uint8_t)cfg_.cdefUVSec;
        }
        if (ll) {   // CodedLossless: qindex 0, no delta-q / QM / tx select / post-filters (av1core rejects them)
            fp.base_q_idx = 0;
            fp.tx_mode_select = 0;
            fp.delta_q_u_dc = fp.delta_q_u_ac = fp.delta_q_v_dc = fp.delta_q_v_ac = 0;
            fp.using_qmatrix = 0; fp.qm_y = fp.qm_u = fp.qm_v = 0;
            fp.delta_q_present = 0; fp.delta_q_res = 0;
            memset(fp.loop_filter_level, 0, sizeof fp.loop_filter_level);
            fp.loop_filter_sharpness = 0;
            fp.enable_cdef = 0; fp.cdef_damping_minus_3 = 0; fp.cdef_bits = 0;
            memset(fp.cdef_y_pri, 0, sizeof fp.cdef_y_pri); memset(fp.cdef_y_sec, 0, sizeof fp.cdef_y_sec);
            memset(fp.cdef_uv_pri, 0, sizeof fp.cdef_uv_pri); memset(fp.cdef_uv_sec, 0, sizeof fp.cdef_uv_sec);
            fp.enable_filter_intra = (uint8_t)P.filterIntra;
        }
        if (in[i].colorMatrix >= 0) {   // BT.709 primaries, sRGB transfer, the frame's matrix, full range
            fp.color_description = 1;
            fp.color_primaries = 1;
            fp.transfer_characteristics = 13;
            fp.matrix_coefficients = (uint8_t)in[i].colorMatrix;
        } else if (in[i].colorMatrix == -2) {   // unspecified (an alpha plane: CICP does not apply), not coded
            fp.color_description = 1;
            fp.color_primaries = fp.transfer_characteristics = fp.matrix_coefficients = 2;
        }
    }
    // ---- post-filters (decoder-exact, av1core GPU kernels) on the pre-filter reconstruction ----
    std::vector<std::array<uint8_t *, 3>> dl(n);   // planes to download
    std::vector<std::array<int, 3>> dlp(n);
    for (int i = 0; i < n; i++)
        for (int p = 0; p < 3; p++) { dl[i][p] = imgs[i].rec[p]; dlp[i][p] = imgs[i].rpitch[p]; }
    if (ll) {   // lossless: no post-filters; the reconstruction is the source
        for (int i = 0; i < n; i++)
            for (int p = 0; p < 3; p++) { dl[i][p] = (uint8_t *)in[i].dsrc[p]; dlp[i][p] = in[i].spitch[p]; }
    } else
    if (cfg_.fLf >= 0) {
        // av1f: deblocking levels (+ apply) and CDEF presets / per-SB cdef_idx search, then CDEF apply
        av1f_lf_opts lo = av1f_lf_opts_default();
        lo.lf_mode = cfg_.fLf;
        lo.cdef_mode = cfg_.fCdef;
        lo.dist = cfg_.fDist;
        size_t scrMax = 0;
        for (int i = 0; i < n; i++) scrMax = std::max(scrMax, av1f_lf_scratch_size(&out[i].fp, &lo));
        uint8_t *scr;
        av1f_lf_result *dRes;
        int8_t *dSbc;
        GPCK(A.get(scr, scrMax));
        GPCK(A.get(dRes, n));
        GPCK(A.get(dSbc, nSb));
        std::vector<uint8_t *> pscr(n);   // per frame: the av1core maps + CDEF directions of the search, reused by the apply
        for (int i = 0; i < n; i++) {
            const GpImage &I = imgs[i];
            uint8_t *const rec[3] = {I.rec[0], I.rec[1], I.rec[2]};
            const uint8_t *const src[3] = {in[i].dsrc[0], in[i].dsrc[1], in[i].dsrc[2]};
            GPCK(A.get(pscr[i], av1g_gpu_pf_scratch_size(&out[i].fp)));
            GPCK((cudaError_t)av1f_lf_search_async_pf(&out[i].fp, dB + imgBlk0[i], imgBlk0[i + 1] - imgBlk0[i], rec, I.rpitch, src,
                                                      in[i].spitch, &lo, dRes + i, dSbc + I.sbBase, scr, pscr[i], st));
        }
        std::vector<av1f_lf_result> hr(n);
        std::vector<int8_t> hsbc(nSb);
        std::vector<int> cdefOn(n, 0), sbCols(n);
        std::vector<uint32_t> sbBase(n);
        for (int i = 0; i < n; i++) { sbCols[i] = imgs[i].sbCols; sbBase[i] = imgs[i].sbBase; }
        {
            GpPinnedStage &PS2 = gp_pinned();
            if (PS2.stage(hr.data(), dRes, n * sizeof(av1f_lf_result), st) || PS2.stage(hsbc.data(), dSbc, nSb, st) || PS2.flush(st))
                GPCK(cudaErrorUnknown);
        }
        for (int i = 0; i < n; i++) {
            const GpImage &I = imgs[i];
            av1g_frame_params &fp = out[i].fp;
            const av1f_lf_result &r = hr[i];
            for (int k = 0; k < 4; k++) fp.loop_filter_level[k] = (uint8_t)r.lf_level[k];
            if (I.mono) fp.loop_filter_level[2] = fp.loop_filter_level[3] = 0;
            fp.loop_filter_sharpness = (uint8_t)r.sharpness;
            fp.enable_cdef = (uint8_t)r.cdef_enable;
            if (r.cdef_enable) {
                fp.cdef_damping_minus_3 = (uint8_t)(r.cdef_damping - 3);
                fp.cdef_bits = (uint8_t)r.cdef_bits;
                for (int k = 0; k < 8; k++) {
                    fp.cdef_y_pri[k] = (uint8_t)r.y_pri[k]; fp.cdef_y_sec[k] = (uint8_t)r.y_sec[k];
                    fp.cdef_uv_pri[k] = (uint8_t)r.uv_pri[k]; fp.cdef_uv_sec[k] = (uint8_t)r.uv_sec[k];
                }
                cdefOn[i] = 1;
                // CDEF on the deblocked frame (the search left d_recon deblocked, and its maps and directions in pscr[i]).
                // Only the reconstruction download reads it (the search set cdef_idx), so without wantRecon it is skipped
                if (!cfg_.wantRecon) continue;
                av1g_pf_frame pf;
                av1g_gpu_pf_attach(&fp, pscr[i], &pf);
                pf.cdefIdx = dSbc + I.sbBase;
                for (int p = 0; p < 3; p++) { pf.plane[p] = I.rec[p]; pf.stride[p] = I.rpitch[p]; }
                uint8_t *co;
                const size_t ysz = (size_t)I.aw * I.ah * bps, csz = (size_t)(I.aw >> I.ssx) * (I.ah >> I.ssy) * bps;
                GPCK(A.get(co, ysz + 2 * csz));
                uint8_t *op[3] = {co, co + ysz, co + ysz + csz};
                const int os[3] = {I.aw, I.aw >> I.ssx, I.aw >> I.ssx};
                GPCK((cudaError_t)av1g_gpu_cdef(&pf, op, os, st));
                for (int p = 0; p < 3; p++) { dl[i][p] = op[p]; dlp[i][p] = os[p]; }
            }
        }
        int *dOn, *dCols;
        uint32_t *dBase;
        GPCK(A.get(dOn, n)); GPCK(A.get(dCols, n)); GPCK(A.get(dBase, n));
        GPCK(cudaMemcpyAsync(dOn, cdefOn.data(), n * 4, cudaMemcpyHostToDevice, st));
        GPCK(cudaMemcpyAsync(dCols, sbCols.data(), n * 4, cudaMemcpyHostToDevice, st));
        GPCK(cudaMemcpyAsync(dBase, sbBase.data(), n * 4, cudaMemcpyHostToDevice, st));
        gp_launch_set_cdef(dB, gBlk, aux.blockFrame, dBase, dCols, dOn, dSbc, st);
    } else
    for (int i = 0; i < n; i++) {
        const GpImage &I = imgs[i];
        const av1g_frame_params &fp = out[i].fp;
        if (!fp.loop_filter_level[0] && !fp.loop_filter_level[1] && !fp.enable_cdef) continue;
        uint8_t *scr;
        GPCK(A.get(scr, av1g_gpu_pf_scratch_size(&fp)));
        av1g_pf_frame f;
        GPCK((cudaError_t)av1g_gpu_pf_prepare(&fp, dB + imgBlk0[i], imgBlk0[i + 1] - imgBlk0[i], scr, &f, st));
        for (int p = 0; p < 3; p++) { f.plane[p] = I.rec[p]; f.stride[p] = I.rpitch[p]; }
        GPCK((cudaError_t)av1g_gpu_deblock(&f, st));
        if (fp.enable_cdef) {
            uint8_t *co;
            const size_t ysz = (size_t)I.aw * I.ah * bps, csz = (size_t)(I.aw >> I.ssx) * (I.ah >> I.ssy) * bps;
            GPCK(A.get(co, ysz + 2 * csz));
            uint8_t *op[3] = {co, co + ysz, co + ysz + csz};
            const int os[3] = {I.aw, I.aw >> I.ssx, I.aw >> I.ssx};
            GPCK((cudaError_t)av1g_gpu_cdef_dir(&f, st));
            GPCK((cudaError_t)av1g_gpu_cdef(&f, op, os, st));
            for (int p = 0; p < 3; p++) { dl[i][p] = op[p]; dlp[i][p] = os[p]; }
        }
    }
    cudaEventRecord(ev_[6], st);
    // ---- download ----
    for (int i = 0; i < n; i++) {
        GpFrameOut &o = out[i];
        const GpImage &I = imgs[i];
        const uint32_t nb = imgBlk0[i + 1] - imgBlk0[i], nt = imgTxb0[i + 1] - imgTxb0[i];
        const uint64_t nc = imgCoef0[i + 1] - imgCoef0[i];
        if (!dev) {
            o.blocks.resize(nb); o.block_txb.resize(nb + 1); o.txb.resize(nt); o.coeffs.resize(nc + 1);
            GpPinnedStage &PS3 = gp_pinned();
            if (PS3.stage(o.blocks.data(), dB + imgBlk0[i], nb * sizeof(av1g_block), st) ||
                PS3.stage(o.block_txb.data(), dBT + imgBlk0[i], nb * 4, st) ||
                PS3.stage(o.txb.data(), dT + imgTxb0[i], nt * sizeof(av1g_txb), st) ||
                PS3.stage(o.coeffs.data(), dC + imgCoef0[i], nc * 2, st))
                GPCK(cudaErrorUnknown);
            o.block_txb[nb] = nt;
            const uint32_t np = imgPal0[i + 1] - imgPal0[i];
            const uint64_t nm = imgMap0[i + 1] - imgMap0[i];
            o.palettes.resize(np);
            o.color_map.resize(nm);
            if ((np && PS3.stage(o.palettes.data(), dPal + imgPal0[i], np * sizeof(av1g_palette), st)) ||
                (nm && PS3.stage(o.color_map.data(), dMap + imgMap0[i], nm, st)))
                GPCK(cudaErrorUnknown);
        }
        const int np = !cfg_.wantRecon ? 0 : I.mono ? 1 : 3;
        for (int p = 0; p < np; p++) {
            const int w = p ? (I.W + I.ssx) >> I.ssx : I.W, h = p ? (I.H + I.ssy) >> I.ssy : I.H;
            o.recon[p].resize((size_t)w * h * bps);
            if (gp_pinned().stage2d(o.recon[p].data(), (size_t)w * bps, dl[i][p], (size_t)dlp[i][p] * bps, (size_t)w * bps, h, st))
                GPCK(cudaErrorUnknown);
        }
    }
    cudaEventRecord(ev_[7], st);
    if (gp_pinned().flush(st)) GPCK(cudaErrorUnknown);
    GPCK(cudaStreamSynchronize(st));
    t.deviceBytes = A.bytes + poolBytes + (keepMaps ? mapsAt + ctxTotal * 2 : 0);
    A.release();
    if (dev) {
        dev->fp.clear();
        for (int i = 0; i < n; i++) dev->fp.push_back(out[i].fp);
        dev->blk0 = imgBlk0; dev->txb0 = imgTxb0; dev->coef0 = imgCoef0;
        dev->blocks = dB; dev->blockTxbRel = dBT; dev->txb = dT; dev->coeffs = dC;
        dev->pal0 = imgPal0; dev->map0 = imgMap0;
        dev->palettes = dPal; dev->colorMap = dMap;
        dev->pool = pool;
        dev->maps = mapsMem;
        mapsMem = nullptr;   // (now dev's)
        dev->ctx0.clear();
        if (keepMaps) {   // the closed loop's maps (layout above: I.mYMode .. I.mTx, I.cLvl / I.cDc)
            dev->yMode = dMaps; dev->skip = dMaps + miTotal; dev->miSize = dMaps + 2 * miTotal; dev->txSize = dMaps + 3 * miTotal;
            dev->ctxLvl = dCtx; dev->ctxDc = dCtx + ctxTotal;
            for (int i = 0; i < n; i++)
                for (int p = 0; p < 3; p++) dev->ctx0.push_back((uint64_t)(imgs[i].cLvl[p] - dCtx));
            dev->itemCount = dCnt;
        } else {
            dev->yMode = dev->skip = dev->miSize = dev->txSize = dev->ctxLvl = dev->ctxDc = nullptr;
            dev->itemCount = nullptr;
        }
    } else {
        cudaFreeAsync(pool, st);
    }
    float ms[7], msPack = 0, msPf = 0;
    for (int k = 0; k < 7; k++) cudaEventElapsedTime(&ms[k], ev_[k], ev_[k + 1]);
    cudaEventElapsedTime(&msPack, ev_[5], ev_[8]);
    cudaEventElapsedTime(&msPf, ev_[8], ev_[6]);
    t.setup = ms[0]; t.analysis = ms[1]; t.dp = ms[2]; t.recon = ms[3]; t.pack = msPack; t.postfilter = msPf; t.download = ms[6];
    cudaEventElapsedTime(&t.total, ev_[0], ev_[7]);
    t.blocks = stats[0]; t.skipBlocks = stats[1]; t.cflBlocks = stats[2]; t.rangeFixes = stats[3]; t.txSplit = stats[4];
    t.dpRepaired = stats[5];
    return 0;
}

// ---------------------------------------------------------------------------
int gp_frames_to_10bit(std::vector<GpFrameIn> &frames, cudaStream_t st, uint8_t **buf, std::string &err) {
    *buf = nullptr;
    size_t total = 0;
    auto planeBytes = [](const GpFrameIn &f, int p) -> size_t {
        const int s = (p && !f.yuv444) ? 1 : 0;
        return ((size_t)(f.aw >> s) * (f.ah >> s) * 2 + 255) & ~(size_t)255;
    };
    for (const GpFrameIn &f : frames)
        if (f.bitDepth != 10)
            for (int p = 0; p < (f.mono ? 1 : 3); p++) total += planeBytes(f, p);
    if (!total) return 0;
    uint8_t *d = nullptr;
    GPCK(cudaMallocAsync((void **)&d, total, st));
    size_t at = 0;
    for (GpFrameIn &f : frames) {
        if (f.bitDepth == 10) continue;
        for (int p = 0; p < (f.mono ? 1 : 3); p++) {
            const int s = (p && !f.yuv444) ? 1 : 0, w = f.aw >> s, h = f.ah >> s;
            uint16_t *o = (uint16_t *)(d + at);
            gp_launch_to10(f.dsrc[p], f.spitch[p], o, w, w, h, p > 0, st);
            f.dsrc[p] = (const uint8_t *)o;
            f.spitch[p] = w;
            at += planeBytes(f, p);
        }
        f.bitDepth = 10;
    }
    GPCK(cudaGetLastError());
    *buf = d;
    return 0;
}

namespace {
// the crop of a W x H source clipped to the image (the whole image without a crop); false: none of it is inside
bool rz_crop(const GpXform &x, int W, int H, int *cx, int *cy, int *cw, int *ch) {
    if (x.cw <= 0) {
        *cx = *cy = 0;
        *cw = W;
        *ch = H;
        return true;
    }
    const int64_t x0 = std::max<int64_t>(x.cx, 0), y0 = std::max<int64_t>(x.cy, 0);
    const int64_t x1 = std::min<int64_t>((int64_t)x.cx + x.cw, W), y1 = std::min<int64_t>((int64_t)x.cy + x.ch, H);
    if (x1 <= x0 || y1 <= y0) return false;
    *cx = (int)x0; *cy = (int)y0; *cw = (int)(x1 - x0); *ch = (int)(y1 - y0);
    return true;
}
// option values: a whole number 0 .. 2^20 at s (no sign or spaces), or the character c; s moves past it
bool rz_num(const char *&s, int &v) {
    if (*s < '0' || *s > '9') return false;
    long n = 0;
    for (; *s >= '0' && *s <= '9'; s++)
        if ((n = n * 10 + (*s - '0')) > (1l << 20)) return false;
    v = (int)n;
    return true;
}
bool rz_ch(const char *&s, char c) {
    if (*s != c) return false;
    s++;
    return true;
}
}  // namespace

bool gp_xform_size(const GpXform &x, int W, int H, int *ow, int *oh) {
    int cx, cy, w, h;
    if (!rz_crop(x, W, H, &cx, &cy, &w, &h)) return false;
    if (x.rsFlags) {   // --resize: ImageMagick's ParseMetaGeometry (each side floor(v + 0.5), here at least 1)
        const int f = x.rsFlags;
        auto rnd = [](double v) { return (int)std::min(std::max(1.0, std::floor(v + 0.5)), 1073741824.0); };
        int tw = w, th = h;
        if (f & GpXform::RS_PCT) {   // (the parser sets both numbers)
            tw = rnd(w * x.rsW / 100);
            th = rnd(h * x.rsH / 100);
        } else if (f & GpXform::RS_AREA) {
            const double k = std::sqrt(x.rsW / ((double)w * h));
            tw = rnd(w * k);
            th = rnd(h * k);
        } else if (f & GpXform::RS_EXACT) {   // a side not given stays
            if (f & GpXform::RS_W) tw = (int)x.rsW;
            if (f & GpXform::RS_H) th = (int)x.rsH;
        } else if ((f & GpXform::RS_W) && (f & GpXform::RS_H)) {
            const double sw = x.rsW / w, sh = x.rsH / h, k = (f & GpXform::RS_FILL) ? std::max(sw, sh) : std::min(sw, sh);
            tw = rnd(w * k);
            th = rnd(h * k);
        } else {   // one side; ^: the number is the smaller side's
            const bool gw = f & GpXform::RS_W;
            const double n = gw ? x.rsW : x.rsH, s = n / (gw ? w : h);
            const double k = (f & GpXform::RS_FILL) ? std::max(s, n / (gw ? h : w)) : s;
            tw = gw && !(f & GpXform::RS_FILL) ? (int)n : rnd(w * k);
            th = !gw && !(f & GpXform::RS_FILL) ? (int)n : rnd(h * k);
        }
        if (!(f & GpXform::RS_PCT)) {   // (ImageMagick: a percentage is already the size that > and < compare with)
            if (f & GpXform::RS_SHRINK) { tw = std::min(tw, w); th = std::min(th, h); }
            if ((f & GpXform::RS_ENLARGE) && !(f & GpXform::RS_AREA)) { tw = std::max(tw, w); th = std::max(th, h); }
        }
        *ow = tw;
        *oh = th;
        return true;
    }
    double s = 1.0;
    if (x.maxW > 0) s = std::min(s, (double)x.maxW / w);
    if (x.maxH > 0) s = std::min(s, (double)x.maxH / h);
    *ow = w;
    *oh = h;
    if (s < 1.0) {
        *ow = std::max(1, (int)std::floor(w * s + 0.5));
        *oh = std::max(1, (int)std::floor(h * s + 0.5));
    }
    return true;
}

std::string gp_size_refusal(int w, int h, double maxMp, const char *after) {
    char b[200];
    if (w > GP_MAX_SIDE || h > GP_MAX_SIDE) {
        snprintf(b, sizeof b, "%dx%d%s is larger than AV1 can code (%d pixels per side)", w, h, after, GP_MAX_SIDE);
        return b;
    }
    if (*after && h > GP_MAX_RESIZED_H) {
        snprintf(b, sizeof b, "%dx%d%s is taller than a resized picture can be (%d rows)", w, h, after, GP_MAX_RESIZED_H);
        return b;
    }
    const double mp = (double)w * h / 1e6;
    if (maxMp <= 0 || mp <= maxMp) return "";
    const int need = (int)std::ceil(mp);
    if (need <= 1000) snprintf(b, sizeof b, "%dx%d%s is larger than %.1f megapixels (--max-mp); add --max-mp %d to encode it anyway", w, h, after, maxMp, need);
    else snprintf(b, sizeof b, "%dx%d%s is larger than %.1f megapixels (--max-mp, 1000 at most)", w, h, after, maxMp);
    return b;
}

bool gp_xform_runs(const GpXform &x, int W, int H, int ow, int oh) {
    int cx, cy, cw, ch;
    if (!x.active() || !rz_crop(x, W, H, &cx, &cy, &cw, &ch)) return false;
    return !(cx == 0 && cy == 0 && cw == W && ch == H && ow == W && oh == H);   // (gp_frames_transform's identity)
}

std::string gp_source_refusal(int w, int h, double decMp, const GpXform *x, bool inputLimit) {
    char b[240];
    if (w <= 0 || h <= 0) {
        snprintf(b, sizeof b, "larger than %.1f megapixels; %s raises the limit", decMp, inputLimit ? "--max-input-mp" : "--max-mp");
        return b;
    }
    // what --max-mp must allow: the source, and what its crop / resize makes of it (either way up: EXIF may turn it)
    double mp = (double)w * h / 1e6;
    if (x && x->active() && !inputLimit) {
        GpXform t = *x;
        for (int k = 0; k < 2; k++, t.turn()) {
            int ow, oh;
            if (!gp_xform_size(t, w, h, &ow, &oh)) continue;
            if (ow > GP_MAX_SIDE || oh > GP_MAX_SIDE) {   // (no --max-mp would help)
                snprintf(b, sizeof b, "larger than %.1f megapixels, and its crop / resize, %dx%d, is larger than AV1 can code (%d pixels "
                         "per side)", decMp, ow, oh, GP_MAX_SIDE);
                return b;
            }
            mp = std::max(mp, (double)ow * oh / 1e6);
        }
    }
    const int need = (int)std::ceil(mp);
    const char *opt = inputLimit ? "--max-input-mp" : "--max-mp";
    if (need <= 1000)
        snprintf(b, sizeof b, "larger than %.1f megapixels; add %s %d to %s", decMp, opt, need, inputLimit ? "decode it" : "encode it anyway");
    else
        snprintf(b, sizeof b, "larger than %.1f megapixels, and above the 1000 that %s allows", decMp, opt);
    return b;
}

std::string gp_crop_str(const GpXform &x) {
    char b[64];
    snprintf(b, sizeof b, "%dx%d+%d+%d", x.cw, x.ch, x.cx, x.cy);
    return b;
}

bool gp_parse_crop(const char *v, GpXform &x, std::string *why) {
    auto four = [&](const char *sep, int *n) {   // 4 numbers between the separators sep[0..2]
        const char *s = v;
        for (int k = 0; k < 4; k++)
            if ((k && !rz_ch(s, sep[k - 1])) || !rz_num(s, n[k])) return false;
        return !*s;
    };
    int n[4], w, h, cx, cy;
    if (four("x++", n)) {
        w = n[0]; h = n[1]; cx = n[2]; cy = n[3];
    } else if (four(",,,", n)) {
        cx = n[0]; cy = n[1]; w = n[2]; h = n[3];
    } else {
        if (why) *why = "expected WxH+X+Y or X,Y,W,H (whole pixels)";
        return false;
    }
    if (w < 1 || h < 1) {
        if (why) *why = "the crop is empty (its width and height must be at least 1)";
        return false;
    }
    x.cw = w; x.ch = h; x.cx = cx; x.cy = cy;
    return true;
}

bool gp_parse_resize(const char *v, GpXform &x, std::string *why) {
    // W[%][x[H[%]]] or @A or A@, followed by any of ! > < ^ (and %, also after the numbers)
    const char *s = v;
    int f = GpXform::RS_ON;
    double w = 0, h = 0;
    auto num = [&](double &d, double most) {
        const char *b = s;
        while ((*s >= '0' && *s <= '9') || *s == '.') s++;
        if (s == b) return false;
        char *e = nullptr;
        d = strtod(std::string(b, s).c_str(), &e);
        return e && !*e && d > 0 && d <= most;
    };
    auto flags = [&]() {
        for (;; s++) {
            if (*s == '%') f |= GpXform::RS_PCT;
            else if (*s == '!') f |= GpXform::RS_EXACT;
            else if (*s == '>') f |= GpXform::RS_SHRINK;
            else if (*s == '<') f |= GpXform::RS_ENLARGE;
            else if (*s == '^') f |= GpXform::RS_FILL;
            else return;
        }
    };
    constexpr double kSide = 1 << 20, kArea = 1e12;
    bool ok = true;
    if (strchr(s, '@')) {
        const bool before = *s == '@';
        if (before) s++;
        f |= GpXform::RS_AREA;
        ok = num(w, kArea);
        if (ok && !before && *s == '@') s++;
        flags();
    } else {
        if (*s != 'x') { ok = num(w, kSide); f |= GpXform::RS_W; }
        flags();
        if (ok && *s == 'x') {
            s++;
            if (*s && !strchr("%!<>^", *s)) { ok = num(h, kSide); f |= GpXform::RS_H; }
            flags();
        }
        ok = ok && (f & (GpXform::RS_W | GpXform::RS_H));
    }
    if (!ok || *s) {
        if (why) *why = *s == '+' || *s == '-' ? "--resize takes no offset" : "expected ImageMagick's -resize geometry: W, xH, WxH, WxH^, WxH!, "
                                                                            "with > or <; N% or N%xM%; @AREA or AREA@";
        return false;
    }
    if ((f & GpXform::RS_PCT) && (f & GpXform::RS_AREA)) {
        if (why) *why = "@AREA cannot be a percentage";
        return false;
    }
    if (f & GpXform::RS_PCT) {   // ParseMetaGeometry: no height: the width's percentage; no width: 100 %
        if (!(f & GpXform::RS_W)) w = 100;
        else if (!(f & GpXform::RS_H)) h = w;
        f |= GpXform::RS_W | GpXform::RS_H;
    } else if (!(f & GpXform::RS_AREA)) {   // GetGeometry rounds the numbers; a side must be at least 1
        w = std::floor(w + 0.5);
        h = std::floor(h + 0.5);
        if (((f & GpXform::RS_W) && w < 1) || ((f & GpXform::RS_H) && h < 1)) {
            if (why) *why = "a side of the geometry rounds to 0";
            return false;
        }
    }
    x.rsFlags = f;
    x.rsW = w;
    x.rsH = h;
    x.maxW = x.maxH = 0;   // (the last of --resize / --max-size counts)
    return true;
}

bool gp_parse_max_size(const char *v, GpXform &x, std::string *why) {
    const char *s = v;
    int w = 0, h = 0;
    const bool hasW = *s != 'x';
    bool ok = (!hasW || rz_num(s, w)) && rz_ch(s, 'x');
    const bool hasH = ok && *s;
    ok = ok && (!hasH || (rz_num(s, h) && !*s)) && (hasW || hasH);
    if (!ok) {
        if (why) *why = "expected WxH, Wx or xH (whole pixels)";
        return false;
    }
    if ((hasW && w < 1) || (hasH && h < 1)) {
        if (why) *why = "the box sides must be at least 1";
        return false;
    }
    x.maxW = w;
    x.maxH = h;
    x.rsFlags = 0;   // (the last of --resize / --max-size counts)
    return true;
}

namespace {
// ImageMagick's Lanczos (resize.c: Sinc windowed by Sinc, support 3)
double rz_lanczos(double x) {
    x = std::fabs(x);
    if (x >= 3.0) return 0.0;
    if (x < 1e-9) return 1.0;
    const double a = M_PI * x, b = a / 3.0;
    return std::sin(a) / a * std::sin(b) / b;
}
// ImageMagick's default filter for enlarging (resize.c: Mitchell-Netravali cubic, B = C = 1/3, support 2)
double rz_mitchell(double x) {
    x = std::fabs(x);
    const double B = 1.0 / 3, C = 1.0 / 3;
    if (x < 1.0) return ((12 - 9 * B - 6 * C) * x * x * x + (-18 + 12 * B + 6 * C) * x * x + (6 - 2 * B)) / 6;
    if (x < 2.0) return ((-B - 6 * C) * x * x * x + (6 * B + 30 * C) * x * x + (-12 * B - 48 * C) * x + (8 * B + 24 * C)) / 6;
    return 0.0;
}
// Filter taps of one axis, as resize.c's HorizontalFilter / VerticalFilter: n output samples at scale f (output samples
// per source sample: the luma scale, also for subsampled chroma) from the source window starting at lo (plane samples;
// fractional for 4:2:0 chroma at an odd crop offset), clipped to whole samples [wlo, whi) and renormalised.  Taps of
// zero weight at the ends are dropped (a crop without a resize copies).  Each row is shifted so that start + taps <= whi
// (the kernels read `taps` samples unconditionally).
void rz_taps(double lo, double f, int n, int wlo, int whi, std::vector<int> &start, std::vector<float> &w, int &taps,
             bool mitchell = false) {
    double (*const filt)(double) = mitchell ? rz_mitchell : rz_lanczos;
    const double support = (mitchell ? 2.0 : 3.0) * std::max(1.0 / f, 1.0), sc = std::min(f, 1.0);
    std::vector<int> s0(n), cnt(n);
    std::vector<size_t> at(n);
    std::vector<double> wd;   // the weights of output o: wd[at[o] ..], cnt[o] of them from source sample s0[o]
    taps = 1;
    for (int o = 0; o < n; o++) {
        const double c = lo + (o + 0.5) / f;
        int a = std::max((int)std::floor(c - support + 0.5), wlo), b = std::min((int)std::floor(c + support + 0.5), whi);
        if (b <= a) { a = std::min(std::max((int)std::floor(c), wlo), whi - 1); b = a + 1; }
        const size_t k0 = wd.size();
        double sum = 0;
        for (int k = a; k < b; k++) {
            wd.push_back(filt(sc * (k + 0.5 - c)));
            sum += wd.back();
        }
        for (size_t k = k0; k < wd.size(); k++) wd[k] = sum != 0 ? wd[k] / sum : k == k0;
        size_t e = wd.size(), k1 = k0;
        while (e - k1 > 1 && std::fabs(wd[e - 1]) < 1e-9) e--;
        while (e - k1 > 1 && std::fabs(wd[k1]) < 1e-9) k1++;
        wd.erase(wd.begin() + e, wd.end());
        wd.erase(wd.begin() + k0, wd.begin() + k1);
        s0[o] = a + (int)(k1 - k0);
        cnt[o] = (int)(wd.size() - k0);
        at[o] = k0;
        taps = std::max(taps, cnt[o]);
    }
    taps = std::min(taps, whi - wlo);
    start.resize(n);
    w.assign((size_t)n * taps, 0.f);
    for (int o = 0; o < n; o++) {
        const int st = std::max(wlo, std::min(s0[o], whi - taps)), k0 = s0[o] - st;
        start[o] = st;
        for (int k = 0; k < cnt[o]; k++) w[(size_t)o * taps + k0 + k] = (float)wd[at[o] + k];
    }
}
}  // namespace

int gp_frames_transform(std::vector<GpFrameIn> &frames, const std::vector<GpXform> &xf, int outDepth, cudaStream_t st,
                        uint8_t **buf, std::string &err, const std::vector<GpFrameIn> *alpha) {
    *buf = nullptr;
    struct Pl { int fi, p, s, in10, wlo, tw, ow, oh, aw, ah, yt, xt; size_t dst, ys, yw, xs, xw; const GpFrameIn *a; };
    std::vector<Pl> pls;
    std::vector<uint8_t> host;   // taps of every plane, uploaded at once
    auto put = [&](const void *d, size_t n) {
        const size_t at = (host.size() + 255) & ~(size_t)255;
        host.resize(at + n);
        memcpy(host.data() + at, d, n);
        return at;
    };
    size_t outBytes = 0, tmpMax = 0;
    std::vector<int> newW(frames.size()), newH(frames.size());
    std::vector<char> keepJpeg(frames.size(), 0);   // a crop alone at even offsets keeps a JPEG's 4:2:0 chroma as it was
    for (size_t i = 0; i < frames.size() && i < xf.size(); i++) {
        const GpFrameIn &f = frames[i];
        if (f.W <= 0 || !xf[i].active()) continue;
        int cx, cy, cw, ch, ow, oh;
        if (!rz_crop(xf[i], f.W, f.H, &cx, &cy, &cw, &ch) || !gp_xform_size(xf[i], f.W, f.H, &ow, &oh)) {
            err = "the crop " + gp_crop_str(xf[i]) + " is outside the " + std::to_string(f.W) + "x" + std::to_string(f.H) + " image";
            return -1;
        }
        if (cx == 0 && cy == 0 && cw == f.W && ch == f.H && ow == f.W && oh == f.H) continue;   // identity
        const GpFrameIn *a = alpha && i < alpha->size() && (*alpha)[i].W == f.W && (*alpha)[i].H == f.H ? &(*alpha)[i] : nullptr;
        newW[i] = ow;
        newH[i] = oh;
        keepJpeg[i] = f.jpegChroma && ow == cw && oh == ch && !(cx & 1) && !(cy & 1);
        const int aw = (ow + 63) & ~63, ah = (oh + 63) & ~63, out10 = outDepth == 10 || f.bitDepth == 10;
        const double fx = (double)ow / cw, fy = (double)oh / ch;
        // --resize, as ImageMagick: Mitchell to enlarge and for anything with alpha, Lanczos to shrink (--max-size: Lanczos)
        const bool mitchell = xf[i].rsFlags && (a || f.colorMatrix == -2 || (double)ow * oh > (double)cw * ch);
        for (int p = 0; p < (f.mono ? 1 : 3); p++) {
            Pl q{};
            q.fi = (int)i;
            q.p = p;
            q.s = (p && !f.yuv444) ? 1 : 0;
            q.in10 = f.bitDepth == 10;
            q.a = a;
            const double d = 1 << q.s;
            const int pw = (f.W + q.s) >> q.s, ph = (f.H + q.s) >> q.s;   // visible plane size
            q.ow = (ow + q.s) >> q.s;
            q.oh = (oh + q.s) >> q.s;
            q.aw = aw >> q.s;
            q.ah = ah >> q.s;
            const int xlo = (int)std::floor(cx / d), xhi = std::min(pw, (int)std::ceil((cx + cw) / d));
            const int ylo = (int)std::floor(cy / d), yhi = std::min(ph, (int)std::ceil((cy + ch) / d));
            std::vector<int> s;
            std::vector<float> w;
            rz_taps(cy / d, fy, q.oh, ylo, yhi, s, w, q.yt, mitchell);
            q.ys = put(s.data(), s.size() * 4);
            q.yw = put(w.data(), w.size() * 4);
            rz_taps(cx / d, fx, q.ow, xlo, xhi, s, w, q.xt, mitchell);
            q.xs = put(s.data(), s.size() * 4);
            q.xw = put(w.data(), w.size() * 4);
            q.wlo = xlo;
            q.tw = xhi - xlo;
            q.dst = outBytes;
            outBytes += ((size_t)q.aw * q.ah * (out10 ? 2 : 1) + 255) & ~(size_t)255;
            tmpMax = std::max(tmpMax, (size_t)q.oh * q.tw * 4 * (a ? 3 : 1));
            pls.push_back(q);
        }
    }
    if (pls.empty()) return 0;
    uint8_t *d = nullptr, *scr = nullptr;
    const size_t tabAt = (tmpMax + 255) & ~(size_t)255;
    cudaError_t e = cudaMallocAsync((void **)&d, outBytes, st);
    if (e == cudaSuccess) e = cudaMallocAsync((void **)&scr, tabAt + host.size(), st);
    if (e != cudaSuccess) {
        cudaGetLastError();
        if (d) cudaFreeAsync(d, st);
        err = std::string("CUDA: ") + cudaGetErrorString(e) + " (crop / resize planes)";
        return -1;
    }
    GPCK(cudaMemcpyAsync(scr + tabAt, host.data(), host.size(), cudaMemcpyHostToDevice, st));
    float *tmp = (float *)scr;
    const uint8_t *tab = scr + tabAt;
    for (const Pl &q : pls) {
        const GpFrameIn &f = frames[q.fi];
        const int out10 = outDepth == 10 || f.bitDepth == 10;
        gp_launch_resize(f.dsrc[q.p], f.spitch[q.p], q.in10, q.wlo, q.tw, (const int *)(tab + q.ys), (const float *)(tab + q.yw), q.yt,
                         (const int *)(tab + q.xs), (const float *)(tab + q.xw), q.xt, tmp, d + q.dst, q.aw, q.ow, q.oh, q.aw, q.ah,
                         out10, q.p > 0, q.a ? q.a->dsrc[0] : nullptr, q.a ? q.a->spitch[0] : 0, q.s, st);
    }
    GPCK(cudaGetLastError());
    GPCK(cudaFreeAsync(scr, st));
    for (const Pl &q : pls) {   // after every launch: the launches read the old frame fields
        GpFrameIn &f = frames[q.fi];
        f.dsrc[q.p] = d + q.dst;
        f.spitch[q.p] = q.aw;
        if (q.p == (f.mono ? 0 : 2)) {
            f.bitDepth = outDepth == 10 || f.bitDepth == 10 ? 10 : 8;
            f.W = newW[q.fi];
            f.H = newH[q.fi];
            f.jpegChroma = keepJpeg[q.fi];
            f.aw = (f.W + 63) & ~63;
            f.ah = (f.H + 63) & ~63;
        }
    }
    *buf = d;
    return 0;
}

namespace {
// the measurement / kernel view of a frame: its planes, how to read RGB from them, and the window (luma pixels)
GpTrimFrame rgb_view(const GpFrameIn &f, int x0 = 0, int y0 = 0, int w = 0, int h = 0) {
    GpTrimFrame g{};
    for (int p = 0; p < 3; p++) {
        g.p[p] = f.dsrc[p];
        g.pitch[p] = f.spitch[p];
    }
    g.mono = f.mono;
    g.yuv444 = f.yuv444;
    g.in10 = f.bitDepth == 10;
    g.cw = (f.W + 1) >> 1;
    g.ch = (f.H + 1) >> 1;
    g.up = f.jpegChroma && !f.mono && !f.yuv444 && g.cw > 2;   // (libjpeg-turbo upsamples chroma only wider than 2)
    g.matrix = f.colorMatrix == 0 ? 1 : f.colorMatrix == 16 ? 2 : 0;
    g.scale = g.matrix == 2 ? 255.f : g.in10 ? 1023.f : 255.f;
    g.x0 = x0; g.y0 = y0; g.w = w; g.h = h;
    return g;
}
}  // namespace

int gp_frames_tone(std::vector<GpFrameIn> &frames, const std::vector<GpXform> &xf, const GpTone &t, cudaStream_t st,
                   std::string &err) {
    std::vector<GpToneMap> maps(frames.size(), gp_tone_map(t));
    if (t.autoLevel) {
        std::vector<int> init(2 * frames.size());
        for (size_t i = 0; i < frames.size(); i++) { init[2 * i] = 0x7f7fffff; init[2 * i + 1] = 0; }
        int *d = nullptr;
        cudaError_t e = cudaMallocAsync((void **)&d, init.size() * sizeof(int), st);
        if (e == cudaSuccess) e = cudaMemcpyAsync(d, init.data(), init.size() * sizeof(int), cudaMemcpyHostToDevice, st);
        for (size_t i = 0; e == cudaSuccess && i < frames.size(); i++) {
            const GpFrameIn &f = frames[i];
            int cx = 0, cy = 0, cw = f.W, ch = f.H;
            if (f.W <= 0 || (i < xf.size() && !rz_crop(xf[i], f.W, f.H, &cx, &cy, &cw, &ch))) continue;
            gp_launch_tone_range(rgb_view(f, cx, cy, cw, ch), d + 2 * i, st);
        }
        if (e == cudaSuccess) e = cudaGetLastError();
        if (e == cudaSuccess) e = cudaMemcpyAsync(init.data(), d, init.size() * sizeof(int), cudaMemcpyDeviceToHost, st);
        if (d) cudaFreeAsync(d, st);
        if (e == cudaSuccess) e = cudaStreamSynchronize(st);
        if (e != cudaSuccess) {
            cudaGetLastError();
            err = std::string("CUDA: ") + cudaGetErrorString(e) + " (--auto-level)";
            return -1;
        }
        for (size_t i = 0; i < frames.size(); i++) {
            float lo, hi;
            memcpy(&lo, &init[2 * i], 4);
            memcpy(&hi, &init[2 * i + 1], 4);
            if (hi > lo) maps[i] = gp_tone_map(t, lo, hi);
        }
    }
    // the frames to change (an identity leaves the planes as they are), and for 4:2:0 a copy of the chroma to read
    auto runs = [&](size_t i) {
        const GpToneMap &m = maps[i];
        return frames[i].W > 0 && (m.s != 1.f || m.o != 0.f || m.e != 1.f || m.negate);
    };
    auto chromaBytes = [&](const GpFrameIn &f) {   // one plane, padding included
        return (size_t)f.spitch[1] * (f.ah >> 1) * (f.bitDepth == 10 ? 2 : 1);
    };
    size_t copyBytes = 0;   // (only upsampled chroma reads neighbouring samples; the nearest one is the block's own)
    for (size_t i = 0; i < frames.size(); i++)
        if (runs(i) && rgb_view(frames[i]).up) copyBytes += 2 * chromaBytes(frames[i]);
    uint8_t *copy = nullptr;
    cudaError_t e = copyBytes ? cudaMallocAsync((void **)&copy, copyBytes, st) : cudaSuccess;
    size_t at = 0;
    for (size_t i = 0; e == cudaSuccess && i < frames.size(); i++) {
        if (!runs(i)) continue;
        const GpFrameIn &f = frames[i];
        GpTrimFrame g = rgb_view(f);
        void *du = (void *)f.dsrc[1], *dv = (void *)f.dsrc[2];
        if (g.up) {
            const size_t n = chromaBytes(f);
            e = cudaMemcpyAsync(copy + at, f.dsrc[1], n, cudaMemcpyDeviceToDevice, st);
            if (e == cudaSuccess) e = cudaMemcpyAsync(copy + at + n, f.dsrc[2], n, cudaMemcpyDeviceToDevice, st);
            if (e != cudaSuccess) break;
            g.p[1] = copy + at;
            g.p[2] = copy + at + n;
            at += 2 * n;
        }
        gp_launch_tone(g, f.aw, f.ah, maps[i], du, dv, st);
    }
    if (e == cudaSuccess) e = cudaGetLastError();
    if (copy) cudaFreeAsync(copy, st);
    if (e != cudaSuccess) {
        cudaGetLastError();
        err = std::string("CUDA: ") + cudaGetErrorString(e) + " (tone)";
        return -1;
    }
    return 0;
}

void gp_frames_gray(std::vector<GpFrameIn> &frames, int method, cudaStream_t st) {
    for (GpFrameIn &f : frames) {
        if (f.W <= 0 || f.mono) continue;
        const int matrix = f.colorMatrix == 0 ? 1 : f.colorMatrix == 16 ? 2 : 0;
        if (!(method == GP_GRAY_REC601 && matrix == 0)) gp_launch_gray(rgb_view(f), f.aw, f.ah, method, st);
        f.mono = 1;
        f.yuv444 = 0;
        f.dsrc[1] = f.dsrc[2] = nullptr;
        f.colorMatrix = -1;   // (identity / YCgCo are RGB matrices; AV1 allows identity only without subsampling)
    }
}

int gp_frames_trim(const std::vector<GpFrameIn> &frames, std::vector<GpXform> &xf, const GpTrim &t, cudaStream_t st,
                   std::vector<int> &verdict, std::string &err, const std::vector<GpFrameIn> *alpha) {
    verdict.assign(frames.size(), GP_TRIM_WHOLE);
    std::vector<GpTrimFrame> tf;
    std::vector<size_t> at;   // frame of each tf entry
    for (size_t i = 0; i < frames.size() && i < xf.size(); i++) {
        const GpFrameIn &f = frames[i];
        if (f.W <= 0) continue;
        int cx, cy, cw, ch;
        if (!rz_crop(xf[i], f.W, f.H, &cx, &cy, &cw, &ch)) continue;   // (gp_frames_transform reports it)
        const GpFrameIn *a = alpha && i < alpha->size() && (*alpha)[i].W == f.W && (*alpha)[i].H == f.H ? &(*alpha)[i] : nullptr;
        GpTrimFrame g = rgb_view(f, cx, cy, cw, ch);
        g.alpha = a ? a->dsrc[0] : nullptr;
        g.apitch = a ? a->spitch[0] : 0;
        tf.push_back(g);
        at.push_back(i);
    }
    std::vector<GpTrimBox> boxes;
    if (gp_trim_measure(tf, t, st, boxes, err)) return -1;
    for (size_t k = 0; k < tf.size(); k++) {
        int kx, ky, kw, kh;
        const size_t i = at[k];
        verdict[i] = gp_trim_decide(t, tf[k].w, tf[k].h, boxes[k], &kx, &ky, &kw, &kh);
        if (verdict[i] != GP_TRIM_CUT) continue;
        xf[i].cx = tf[k].x0 + kx;
        xf[i].cy = tf[k].y0 + ky;
        xf[i].cw = kw;
        xf[i].ch = kh;
    }
    return 0;
}

GpConfig gp_alpha_config(const GpConfig &c) {
    GpConfig a = c;
    const int q = c.lossless ? 0 : c.alphaQindex >= 0 ? c.alphaQindex : c.qindex;
    a.lossless = q == 0;   // lossless alpha: the lossless (monochrome) path
    a.qindex = q == 0 ? c.qindex : q;
    a.tuneIq = 0; a.vb = 0; a.ssimLambda = 0; a.deltaq = 0; a.qm = -1; a.qmUV = -1;
    a.dqUDc = a.dqUAc = a.dqVDc = a.dqVAc = 0;
    a.P.qmRnd = a.P.qmDist = 0;
    a.Psmall.qmRnd = a.Psmall.qmDist = 0;
    return a;
}

int gp_mux_avif(const av1g_frame_params &fp, const uint8_t *obu, size_t obuSize, const uint8_t *alphaObu, size_t alphaSize,
                const std::vector<uint8_t> &icc, int orientation, std::vector<uint8_t> &avif, std::string &err) {
    av1g_avif_meta meta;
    memset(&meta, 0, sizeof meta);
    meta.icc = icc.empty() ? nullptr : icc.data();
    meta.icc_size = icc.size();
    av1g_exif_orientation_to_irot_imir(orientation, &meta.irot, &meta.imir);
    uint8_t *a = nullptr;
    size_t an = 0;
    if (av1g_mux_avif_alpha(obu, obuSize, alphaObu, alphaSize, fp.width, fp.height, &meta, &a, &an) != AV1G_OK) {
        err = alphaObu ? "AVIF mux failed (alpha)" : "AVIF mux failed";
        return -1;
    }
    avif.assign(a, a + an);
    free(a);
    return 0;
}

static int cpu_obus(GpFrameOut &f, std::vector<uint8_t> &obu, std::string &err) {
    av1g_decision_record r = f.record();
    av1g_encode_opts eo;
    eo.flags = AV1G_ENC_BITSTREAM_ONLY | AV1G_ENC_TD;
    eo.seq_level_idx = -1;
    av1g_output o;
    memset(&o, 0, sizeof o);
    int rc = av1g_encode(&r, &eo, &o);
    if (rc != AV1G_OK) { err = std::string("av1g_encode: ") + o.error; av1g_output_free(&o); return rc; }
    obu.assign(o.obu, o.obu + o.obu_size);
    av1g_output_free(&o);
    return 0;
}

int gp_entropy_cpu(GpFrameOut &f, const std::vector<uint8_t> &icc, std::vector<uint8_t> &obu, std::vector<uint8_t> &avif,
                   std::string &err, GpFrameOut *alpha, std::vector<uint8_t> *alphaObu) {
    if (int rc = cpu_obus(f, obu, err)) return rc;
    std::vector<uint8_t> ao;
    if (alpha) {
        if (int rc = cpu_obus(*alpha, ao, err)) return rc;
        if (alphaObu) *alphaObu = ao;
    }
    return gp_mux_avif(f.fp, obu.data() + 2, obu.size() - 2, alpha ? ao.data() + 2 : nullptr, alpha ? ao.size() - 2 : 0, icc,
                       f.orientation, avif, err);
}
