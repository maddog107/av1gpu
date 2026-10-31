// host orchestration of one batch (all frames in flight on the GPU at once).
#include "pipeline.h"
#include <mutex>
#include "pinned.h"
#include "kernels.h"
size_t gp_recon_scratch_bytes();   // k_recon.cu (W5/W3)
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
    return r;
}

GpEncoder::GpEncoder() { memset(ev_, 0, sizeof ev_); memset(evx_, 0, sizeof evx_); }
GpEncoder::~GpEncoder() {
    for (auto &e : ev_)
        if (e) cudaEventDestroy(e);
    for (auto &e : evx_)
        if (e) cudaEventDestroy(e);
    if (hiSt_) cudaStreamDestroy(hiSt_);
    if (vbLuts_) cudaFree(vbLuts_);
}

int GpEncoder::init(const GpConfig &cfg, std::string &err) {
    cfg_ = cfg;
    char buf[256] = {0};
    if (gp_init_tables(buf, sizeof buf)) { err = buf; return -1; }
    for (auto &e : ev_) GPCK(cudaEventCreate(&e));
    for (auto &e : evx_) GPCK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
    {
        int lo = 0, hi = 0;
        cudaDeviceGetStreamPriorityRange(&lo, &hi);
        GPCK(cudaStreamCreateWithPriority(&hiSt_, cudaStreamNonBlocking, hi));
    }
    if (cfg_.vb) {
        GPCK(cudaMalloc(&vbLuts_, 256 * 1024));
        if (av1f_vb_build_luts(vbLuts_, cfg_.vbStrength, 0)) { err = "av1f_vb_build_luts failed"; return -1; }
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
    const GpParams &P = cfg_.P;
    cudaEventRecord(ev_[0], st);
    std::vector<GpImage> imgs(n);
    std::vector<av1f_tuneiq_frame> tiq(n);
    uint32_t shTotal[GP_NSHAPES] = {0};
    std::vector<uint32_t> shStart((size_t)GP_NSHAPES * (n + 1), 0);
    uint32_t nSb = 0;
    std::vector<uint32_t> sbImg;
    std::vector<av1g_tiles> tiles(n);
    std::vector<std::array<int, 2>> tileReq(n);
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
        I.baseQ = cfg_.qindex;
        I.qctx = av1g_coeff_cdf_q_ctx(cfg_.qindex);
        I.dqRes = cfg_.deltaqRes;
        I.edgeFilter = cfg_.edgeFilter;
        I.dqUDc = cfg_.dqUDc; I.dqUAc = cfg_.dqUAc; I.dqVDc = cfg_.dqVDc; I.dqVAc = cfg_.dqVAc;
        if (cfg_.tuneIq) {
            av1f_tuneiq_frame_init(cfg_.qindex, I.ssx, I.ssy, f.mono, &tiq[i]);   // 4:4:4: libaom's 4:4:4 chroma delta-q / QM
            I.dqUDc = tiq[i].dq_u_dc; I.dqUAc = tiq[i].dq_u_ac; I.dqVDc = tiq[i].dq_v_dc; I.dqVAc = tiq[i].dq_v_ac;
        }
        if (cfg_.qm != -1) {
            I.usingQm = 1;
            const int ly = cfg_.qm == -2 ? av1g_qm_level_allintra(cfg_.qindex) : cfg_.qm;
            const int lc = cfg_.qmUV >= 0 ? cfg_.qmUV : (cfg_.qm == -2 ? av1g_qm_level_allintra(cfg_.qindex + cfg_.dqUAc) : ly);
            I.qm[0] = ly; I.qm[1] = lc; I.qm[2] = lc;
        } else {
            I.qm[0] = I.qm[1] = I.qm[2] = 15;
        }
        if (cfg_.tuneIq) {
            I.usingQm = 1;
            I.qm[0] = tiq[i].qm_y; I.qm[1] = tiq[i].qm_u; I.qm[2] = tiq[i].qm_v;
        }
        for (int p = 0; p < 3; p++) { I.src[p] = f.dsrc[p]; I.spitch[p] = f.spitch[p]; }
        I.aw = f.aw; I.ah = f.ah;
        // tiles: minimum legal, or (cfg.tileMinMp) extra uniform tiles for large images (more entropy tiles in flight)
        {
            int tc = -1, tr = -1;
            if (cfg_.tileMinMp > 0 && (double)f.W * f.H >= cfg_.tileMinMp * 1e6) {
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
    bool any444 = false, any420 = false;
    for (int i = 0; i < n; i++) {
        const GpImage &I = imgs[i];
        planeBytes += (size_t)in[i].aw * in[i].ah + 2 * (size_t)(in[i].aw >> I.ssx) * (in[i].ah >> I.ssy);
        miTotal += (size_t)I.MiCols * I.MiRows;
        ctxTotal += (size_t)I.MiCols * I.MiRows + 2 * (size_t)(I.MiCols >> I.ssx) * (I.MiRows >> I.ssy);
        sbqTotal += (size_t)I.sbCols * I.sbRows;
        if (!I.mono) (I.ssx ? any420 : any444) = true;
    }
    uint8_t *dRec, *dMaps, *dCtx, *dSbq = nullptr;
    GPCK(A.get(dRec, planeBytes));
    GPCK(A.get(dMaps, miTotal * 5));
    GPCK(A.get(dCtx, ctxTotal * 2));
    GPCK(cudaMemsetAsync(dMaps, 0, miTotal * 5, st));
    GPCK(cudaMemsetAsync(dCtx, 0, ctxTotal * 2, st));
    std::vector<uint8_t> hSbq;
    const bool anySbq = cfg_.deltaq && std::any_of(in.begin(), in.end(), [](const GpFrameIn &f) { return !f.sbQ.empty(); });
    if (anySbq) {
        GPCK(A.get(dSbq, sbqTotal));
        hSbq.assign(sbqTotal, (uint8_t)cfg_.qindex);
    }
    {
        size_t pr = 0, pm = 0, pc = 0, pq = 0;
        for (int i = 0; i < n; i++) {
            GpImage &I = imgs[i];
            const size_t ysz = (size_t)in[i].aw * in[i].ah, csz = (size_t)(in[i].aw >> I.ssx) * (in[i].ah >> I.ssy);
            I.rec[0] = dRec + pr; I.rpitch[0] = in[i].aw;
            I.rec[1] = dRec + pr + ysz; I.rpitch[1] = in[i].aw >> I.ssx;
            I.rec[2] = dRec + pr + ysz + csz; I.rpitch[2] = in[i].aw >> I.ssx;
            pr += ysz + 2 * csz;
            const size_t mi = (size_t)I.MiCols * I.MiRows;
            I.mYMode = dMaps + pm; I.mUVMode = dMaps + miTotal + pm; I.mSkip = dMaps + 2 * miTotal + pm;
            I.mMiSize = dMaps + 3 * miTotal + pm; I.mTx = dMaps + 4 * miTotal + pm;
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
            gp_launch_var16(in[i].dsrc[0], in[i].spitch[0], I.MiCols, I.MiRows, dV + at, st);
            at += (size_t)I.cols16 * I.rows16;
        }
    }
    if (cfg_.ssimLambda) {
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
            gp_launch_ssim_map(in[i].dsrc[0], in[i].spitch[0], I.MiCols, I.MiRows, dS + at, dSum + i, st);
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
    if (cfg_.vb && !anySbq) {
        // av1f GPU variance boost: per-SB qindex, legal for delta_q_res (libaom tune=iq deltaq-mode 6)
        std::vector<av1f_vb_desc> vd(n);
        for (int i = 0; i < n; i++) {
            const GpImage &I = imgs[i];
            vd[i].y = in[i].dsrc[0]; vd[i].pitch = in[i].spitch[0];
            vd[i].width = I.W; vd[i].height = I.H; vd[i].base_q_idx = cfg_.qindex;
            vd[i].tile_cols_log2 = tiles[i].cols_log2; vd[i].tile_rows_log2 = tiles[i].rows_log2;
            vd[i].sb_off = I.sbBase;
        }
        av1f_vb_desc *dVd;
        GPCK(A.get(dVd, n));
        GPCK(A.get(dSbq, sbqTotal));
        GPCK(cudaMemcpyAsync(dVd, vd.data(), n * sizeof(av1f_vb_desc), cudaMemcpyHostToDevice, st));
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
    GPCK(A.get(B.stats, 16));
    GPCK(A.get(B.prof, 16));
    GPCK(cudaMemsetAsync(B.prof, 0, 16 * sizeof(unsigned long long), st));
    B.sbCyc = nullptr;
    if (P.profile && getenv("GP_SBCYC_DUMP")) GPCK(A.get(B.sbCyc, nSb));
    GPCK(cudaMemsetAsync(B.stats, 0, 16 * sizeof(uint32_t), st));
    GPCK(cudaMemcpyAsync(B.img, imgs.data(), n * sizeof(GpImage), cudaMemcpyHostToDevice, st));
    GPCK(cudaMemcpyAsync(B.shImgStart, shStart.data(), shStart.size() * 4, cudaMemcpyHostToDevice, st));
    GPCK(cudaMemcpyAsync(B.sbImg, sbImg.data(), nSb * 4, cudaMemcpyHostToDevice, st));
    // wavefront step lists
    std::vector<std::vector<uint32_t>> steps;
    // wavefront per tile (tiles are independent for prediction and contexts): step = col + 2 * row
    // relative to the tile origin, so every tile of every image starts at step 0
    for (int i = 0; i < n; i++) {
        const GpImage &I = imgs[i];
        const av1g_tiles &T = tiles[i];
        for (uint32_t tr = 0; tr < T.rows; tr++)
            for (uint32_t tc = 0; tc < T.cols; tc++) {
                const int r0 = (int)T.mi_row_starts[tr] >> 4, r1 = ((int)T.mi_row_starts[tr + 1] + 15) >> 4;
                const int c0 = (int)T.mi_col_starts[tc] >> 4, c1 = ((int)T.mi_col_starts[tc + 1] + 15) >> 4;
                for (int r = r0; r < r1; r++)
                    for (int c = c0; c < c1; c++) {
                        const size_t s = (size_t)(c - c0) + 2 * (size_t)(r - r0);
                        if (steps.size() <= s) steps.resize(s + 1);
                        steps[s].push_back(I.sbBase + (uint32_t)(r * I.sbCols + c));
                    }
            }
    }
    std::vector<uint32_t> stepFlat, stepOff;
    for (auto &v : steps) { stepOff.push_back((uint32_t)stepFlat.size()); stepFlat.insert(stepFlat.end(), v.begin(), v.end()); }
    stepOff.push_back((uint32_t)stepFlat.size());
    uint32_t *dSteps;
    GPCK(A.get(dSteps, stepFlat.size()));
    {
        size_t maxStep = 1;
        for (auto &v : steps) maxStep = std::max(maxStep, v.size());
        uint8_t *bs;
        GPCK(A.get(bs, maxStep * gp_recon_scratch_bytes()));
        B.bigScratch = bs;
    }
    GPCK(cudaMemcpyAsync(dSteps, stepFlat.data(), stepFlat.size() * 4, cudaMemcpyHostToDevice, st));
    cudaEventRecord(ev_[1], st);
    // ---- analysis + DP ----
    gp_launch_analysis(B, P, shTotal, st);
    GPCK(cudaGetLastError());
    cudaEventRecord(ev_[2], st);
    gp_launch_dp(B, P, st);
    GPCK(cudaGetLastError());
    cudaEventRecord(ev_[3], st);
    // ---- closed loop, SB wavefront ----
    // wavefront on a high-priority stream: when several encoders share the GPU, the latency-bound recon
    // CTAs (critical path) are scheduled before other encoders' analysis CTAs
    GPCK(cudaEventRecord(evx_[0], st));
    GPCK(cudaStreamWaitEvent(hiSt_, evx_[0], 0));
    for (size_t s = 0; s + 1 < stepOff.size(); s++)
        gp_launch_recon(B, P, dSteps + stepOff[s], stepOff[s + 1] - stepOff[s], hiSt_);
    GPCK(cudaEventRecord(evx_[1], hiSt_));
    GPCK(cudaStreamWaitEvent(st, evx_[1], 0));
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
    std::vector<uint32_t> imgBlk0(n + 1), imgTxb0(n + 1);
    std::vector<uint64_t> imgCoef0(n + 1);
    uint32_t gBlk = 0, gTxb = 0, tileCounter = 0;
    uint64_t gCoef = 0;
    for (int i = 0; i < n; i++) {
        const GpImage &I = imgs[i];
        imgBlk0[i] = gBlk; imgTxb0[i] = gTxb; imgCoef0[i] = gCoef;
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
                        gBlk += so[sb].nBlocks; gTxb += so[sb].nTxb; gCoef += so[sb].nCoeffs;
                        rTxb += so[sb].nTxb; rCoef += so[sb].nCoeffs;
                    }
            }
    }
    imgBlk0[n] = gBlk; imgTxb0[n] = gTxb; imgCoef0[n] = gCoef;
    // record arrays: one allocation; handed to the caller (dev) or freed at the end
    auto al = [](size_t x) { return (x + 255) & ~(size_t)255; };
    const size_t szB = al((size_t)gBlk * sizeof(av1g_block)), szBT = al((size_t)gBlk * 4), szAux = al(((size_t)gBlk + 1) * 4);
    const size_t szT = al((size_t)gTxb * sizeof(av1g_txb)), szTB = al((size_t)gTxb * 4), szTC = al((size_t)gTxb * 8);
    const size_t szC = al((size_t)gCoef * 2 + 2);
    const size_t poolBytes = szB + szBT + 3 * szAux + szT + szTB + szTC + szC;
    uint8_t *pool = nullptr;
    GPCK(cudaMallocAsync((void **)&pool, poolBytes, st));
    size_t pp = 0;
    auto take = [&](size_t s) { uint8_t *q = pool + pp; pp += s; return q; };
    av1g_block *dB = (av1g_block *)take(szB);
    uint32_t *dBT = (uint32_t *)take(szBT);
    GpPackAux aux;
    aux.blockFrame = (uint32_t *)take(szAux);
    aux.blockTile = (uint32_t *)take(szAux);
    aux.blockTxbG = (uint32_t *)take(szAux);
    av1g_txb *dT = (av1g_txb *)take(szT);
    aux.txbBlock = (uint32_t *)take(szTB);
    aux.txbCoeff = (uint64_t *)take(szTC);
    int16_t *dC = (int16_t *)take(szC);
    GPCK(cudaMemcpyAsync(aux.blockTxbG + gBlk, &gTxb, 4, cudaMemcpyHostToDevice, st));
    GpPackOff *dPo;
    GPCK(A.get(dPo, nSb));
    GPCK(cudaMemcpyAsync(dPo, po.data(), nSb * sizeof(GpPackOff), cudaMemcpyHostToDevice, st));
    cudaEventRecord(ev_[5], st);
    gp_launch_pack(B, dPo, dB, dBT, dT, dC, aux, st);
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
        if (cfg_.lfMode) {
            // libaom LPF_PICK_FROM_Q (key frame, 8-bit): ROUND_POWER_OF_TWO(acq * 17563 - 421574, 18), times lfMul
            int lv = cfg_.lfLevel;
            if (cfg_.lfMode == 1) {
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
    }
    // ---- post-filters (decoder-exact, av1core GPU kernels) on the pre-filter reconstruction ----
    std::vector<std::array<uint8_t *, 3>> dl(n);   // planes to download
    std::vector<std::array<int, 3>> dlp(n);
    for (int i = 0; i < n; i++)
        for (int p = 0; p < 3; p++) { dl[i][p] = imgs[i].rec[p]; dlp[i][p] = imgs[i].rpitch[p]; }
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
        for (int i = 0; i < n; i++) {
            const GpImage &I = imgs[i];
            uint8_t *const rec[3] = {I.rec[0], I.rec[1], I.rec[2]};
            const uint8_t *const src[3] = {in[i].dsrc[0], in[i].dsrc[1], in[i].dsrc[2]};
            GPCK((cudaError_t)av1f_lf_search_async(&out[i].fp, dB + imgBlk0[i], imgBlk0[i + 1] - imgBlk0[i], rec, I.rpitch, src,
                                                   in[i].spitch, &lo, dRes + i, dSbc + I.sbBase, scr, st));
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
                // CDEF on the deblocked frame (the search left d_recon deblocked)
                uint8_t *pscr;
                GPCK(A.get(pscr, av1g_gpu_pf_scratch_size(&fp)));
                av1g_pf_frame pf;
                GPCK((cudaError_t)av1g_gpu_pf_prepare(&fp, dB + imgBlk0[i], imgBlk0[i + 1] - imgBlk0[i], pscr, &pf, st));
                GPCK(cudaMemcpyAsync(pf.cdefIdx, dSbc + I.sbBase, (size_t)I.sbCols * I.sbRows, cudaMemcpyDeviceToDevice, st));
                for (int p = 0; p < 3; p++) { pf.plane[p] = I.rec[p]; pf.stride[p] = I.rpitch[p]; }
                uint8_t *co;
                const size_t ysz = (size_t)I.aw * I.ah, csz = (size_t)(I.aw >> I.ssx) * (I.ah >> I.ssy);
                GPCK(A.get(co, ysz + 2 * csz));
                uint8_t *op[3] = {co, co + ysz, co + ysz + csz};
                const int os[3] = {I.aw, I.aw >> I.ssx, I.aw >> I.ssx};
                GPCK((cudaError_t)av1g_gpu_cdef_dir(&pf, st));
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
            const size_t ysz = (size_t)I.aw * I.ah, csz = (size_t)(I.aw >> I.ssx) * (I.ah >> I.ssy);
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
        }
        const int np = !cfg_.wantRecon ? 0 : I.mono ? 1 : 3;
        for (int p = 0; p < np; p++) {
            const int w = p ? (I.W + I.ssx) >> I.ssx : I.W, h = p ? (I.H + I.ssy) >> I.ssy : I.H;
            o.recon[p].resize((size_t)w * h);
            if (gp_pinned().stage2d(o.recon[p].data(), w, dl[i][p], dlp[i][p], w, h, st)) GPCK(cudaErrorUnknown);
        }
    }
    cudaEventRecord(ev_[7], st);
    if (gp_pinned().flush(st)) GPCK(cudaErrorUnknown);
    GPCK(cudaStreamSynchronize(st));
    t.deviceBytes = A.bytes + poolBytes;
    A.release();
    if (dev) {
        dev->fp.clear();
        for (int i = 0; i < n; i++) dev->fp.push_back(out[i].fp);
        dev->blk0 = imgBlk0; dev->txb0 = imgTxb0; dev->coef0 = imgCoef0;
        dev->blocks = dB; dev->blockTxbRel = dBT; dev->txb = dT; dev->coeffs = dC;
        dev->pool = pool;
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
    return 0;
}

// ---------------------------------------------------------------------------
int gp_entropy_cpu(GpFrameOut &f, const std::vector<uint8_t> &icc, std::vector<uint8_t> &obu, std::vector<uint8_t> &avif,
                   std::string &err) {
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
    av1g_avif_meta meta;
    memset(&meta, 0, sizeof meta);
    meta.icc = icc.empty() ? nullptr : icc.data();
    meta.icc_size = icc.size();
    av1g_exif_orientation_to_irot_imir(f.orientation, &meta.irot, &meta.imir);
    uint8_t *a = nullptr;
    size_t an = 0;
    if (av1g_mux_avif_ex(obu.data() + 2, obu.size() - 2, f.fp.width, f.fp.height, &meta, &a, &an) != AV1G_OK) {
        err = "AVIF mux failed";
        return -1;
    }
    avif.assign(a, a + an);
    free(a);
    return 0;
}
