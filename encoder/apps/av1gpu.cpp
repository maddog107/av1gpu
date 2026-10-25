// av1gpu CLI: GPU AV1/AVIF still-image encoder.
//
//   av1gpu IN OUT.avif [options]             single image
//   av1gpu --batch LIST|DIR --outdir DIR     many images in flight (throughput); see --help
//
// --debug-outputs additionally writes OUT.avif.obu (with a temporal delimiter, for dav1d --demuxer
// section5) and OUT.avif.recon.yuv (the encoder's reconstruction) for decoder identity checks.
#include "pipeline.h"
#include "imageio.h"
#include "ingest_api.h"
#include "av1g/av1g.h"
#include "av1f/tuneiq.h"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <vector>

// Defaults = the configuration validated in production (avifenc -q 60 equivalent): applied before the
// command-line options, so any option overrides them; --no-defaults starts from the bare encoder.
static const char *const kDefaultArgs[] = {"--quality", "60", "--tune", "iq", "--edge-filter", "--preset", "turbo",
                                           "--small-preset", "base", "--gpu-entropy"};

static void usage() {
    fprintf(stderr,
            "usage: av1gpu IN OUT.avif [options]\n"
            "       av1gpu --batch LIST|DIR --outdir DIR [options]\n"
            "\n"
            "Input: JPEG (GPU decode), PNG or WebP (CPU decode).  Output: 8-bit 4:2:0 (or monochrome) AVIF.\n"
            "Defaults: --quality 60 --tune iq --edge-filter --preset turbo --small-preset base --gpu-entropy\n"
            "\n"
            "common options:\n"
            "  --quality Q             0..100, same scale as avifenc -q with tune=iq (default 60 = qindex 120)\n"
            "  --qindex Q[,Q2,...]     AV1 base_q_idx 1..255 instead of --quality (batch mode: one pass per value)\n"
            "  --preset P              slow|base|fast|faster|turbo|turbo2|turbo3 (default turbo)\n"
            "  --small-preset P|none   preset for images below --small-mp megapixels (default base, 1 MP)\n"
            "  --tune iq|none          libaom tune=iq tool set: chroma delta-q, QM, variance boost, deblock, CDEF\n"
            "  --cpu-entropy           entropy-code on the CPU instead of the GPU (identical bytes)\n"
            "  --batch LIST|DIR        encode every file in LIST (one path per line) or DIR; outputs <name>.avif\n"
            "  --outdir DIR            batch output directory (outputs are named <input file name>.avif)\n"
            "  --stats                 print one JSON line with timings / counts\n"
            "  --debug-outputs         also write <out>.obu and <out>.recon.yuv (decoder identity tests)\n"
            "  --no-defaults           do not apply the defaults above (expert / research use)\n"
            "\n"
            "expert options:\n"
            "  --q Q                   alias of --qindex\n"
            "  --small-mp X            --small-preset threshold in megapixels (default 1)\n"
            "  --lambda M              lambda multiplier (default 1.0)\n"
            "  --K k --Krecheck k      analysis top-K, closed-loop re-check count\n"
            "  --no-txsearch --no-txsearch-cl --no-deltas --no-cfl --uvK k\n"
            "  --shapes HEX --part8 HEX --partN HEX --maxpart N\n"
            "  --rnd R --zbin Z --rndc R --zbinc Z --satd-lambda S --uv-overhead B --split-bias S\n"
            "  --qm L|auto --qm-uv L --edge-filter --lf q|LEVEL --lf-mul M --cdef ypri,ysec,uvpri,uvsec,damp-3\n"
            "  --vb 0|1 --vb-order tile|row|nearest --vb-strength PCT --f-lf off|q|search --f-cdef off|aom|search\n"
            "  --check                 compare the reconstruction with av1core's CPU decode of the record\n"
            "  --gpu-entropy           GPU symbolization + entropy coding (default)\n"
            "\n"
            "environment (batch mode): GP_ENCODERS (concurrent encoders, default 3), GP_ENCODE_MP (MP per GPU batch,\n"
            "  default 48), GP_INGEST_WORKERS (JPEG parse/decode threads, default 4), GP_INGEST_DECODER=1 (no nvJPEG)\n");
}

struct Opts {
    std::vector<int> qlist;   // --qindex a,b,c (batch mode: one pass per q, outputs <name>.q<Q>.avif)
    bool nameQ = false;       // --name-q: add .q<Q> to batch output names even for a single q
    std::string in, out, batch, outdir;
    bool check = false, gpuEntropy = false, stats = false, devEntropy = false, checkEntropy = false, debugOutputs = false;
};

// speed presets: the speed knobs only (base = the GpConfig defaults)
static bool apply_preset(GpConfig &cfg, const char *v) {
    const GpConfig d;   // defaults
    // reset to base first, so a preset fully defines the speed knobs
    cfg.P.shapeMask = d.P.shapeMask; cfg.P.K = d.P.K; cfg.P.Krecheck = d.P.Krecheck; cfg.P.txSearch = d.P.txSearch;
    cfg.P.txSearchCL = d.P.txSearchCL; cfg.P.pruneFlat = d.P.pruneFlat; cfg.P.screenSub = d.P.screenSub; cfg.P.uvK = d.P.uvK;
    cfg.P.deltas = d.P.deltas; cfg.P.uvDeltas = d.P.uvDeltas;
    cfg.P.filterIntra = d.P.filterIntra; cfg.P.txAllCL = d.P.txAllCL; cfg.P.txSizeSearch = d.P.txSizeSearch;
    cfg.tileMinMp = d.tileMinMp; cfg.tileColsLog2 = d.tileColsLog2; cfg.tileRowsLog2 = d.tileRowsLog2;
    if (!strcmp(v, "base")) return true;
    if (!strcmp(v, "slow")) {
        // max quality (round 4): closed-loop tx-size search (largest vs one split), every tx type for
        // every re-checked candidate, K 6 / Krecheck 6, filter intra, deblocking-level + CDEF search
        cfg.P.txSizeSearch = 1; cfg.P.txAllCL = 1; cfg.P.K = 6; cfg.P.Krecheck = 6; cfg.P.filterIntra = 1;
        cfg.fLf = 2; cfg.fCdef = 2;
        return true;
    }
    if (!strcmp(v, "fast")) {          // no 4:1 shapes, K 2, no tx search in the analysis
        cfg.P.shapeMask = 0x1FFF; cfg.P.K = 2; cfg.P.Krecheck = 2; cfg.P.txSearch = 0;
        return true;
    }
    if (!strcmp(v, "faster")) {        // + flat pruning, single closed-loop candidate
        cfg.P.shapeMask = 0x1FFF; cfg.P.K = 2; cfg.P.Krecheck = 1; cfg.P.txSearch = 0; cfg.P.pruneFlat = 1.0f;
        return true;
    }
    if (!strcmp(v, "turbo") || !strcmp(v, "turbo2") || !strcmp(v, "turbo3")) {
        // faster + no 32x64/64x32 shapes, 2x-subsampled SATD screening for >= 16x16, 2x2 tiles for
        // images >= 2 MP (shorter wavefront / entropy critical paths), K 1
        cfg.P.shapeMask = 0x13FF; cfg.P.K = 1; cfg.P.Krecheck = 1; cfg.P.txSearch = 0; cfg.P.pruneFlat = 1.0f;
        cfg.P.screenSub = 2;
        cfg.tileMinMp = 2.f; cfg.tileColsLog2 = 1; cfg.tileRowsLog2 = 1;
        if (strcmp(v, "turbo")) {     // turbo2: + no 4x4/4x8/8x4, no closed-loop tx search, 1 chroma RD candidate
            cfg.P.shapeMask = 0x13F8; cfg.P.txSearchCL = 0; cfg.P.uvK = 1;
        }
        if (!strcmp(v, "turbo3")) {   // + no luma / UV angle deltas, 4x2 tiles for >= 2 MP
            cfg.P.deltas = 0; cfg.P.uvDeltas = 0; cfg.tileColsLog2 = 2;
        }
        return true;
    }
    return false;
}

static bool parse(int argc, char **argv, GpConfig &cfg, Opts &o) {
    std::string smallPreset;
    std::vector<std::string> pos, args;
    bool defaults = true;
    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--no-defaults")) defaults = false;
    if (defaults) args.assign(std::begin(kDefaultArgs), std::end(kDefaultArgs));
    for (int i = 1; i < argc; i++) args.push_back(argv[i]);
    const int nargs = (int)args.size();
    for (int i = 0; i < nargs; i++) {
        const std::string a = args[i];
        auto nxt = [&]() -> const char * {
            if (i + 1 >= nargs) { fprintf(stderr, "missing value for %s\n", a.c_str()); exit(2); }
            return args[++i].c_str();
        };
        if (a == "--no-defaults") continue;
        if (a == "--quality") {
            const int qv = atoi(nxt());
            if (qv < 0 || qv > 100) { fprintf(stderr, "--quality must be 0..100\n"); return false; }
            o.qlist.assign(1, std::max(1, av1f_qindex_for_quality(qv)));
            cfg.qindex = o.qlist[0];
        }
        else if (a == "--qindex" || a == "--q") {
            o.qlist.clear();
            for (const char *s = nxt(); *s;) {
                o.qlist.push_back(atoi(s));
                while (*s && *s != ',') s++;
                if (*s == ',') s++;
            }
            cfg.qindex = o.qlist.empty() ? 0 : o.qlist[0];
        }
        else if (a == "--name-q") o.nameQ = true;
        else if (a == "--lambda") cfg.P.lambdaMul = (float)atof(nxt());
        else if (a == "--lambda-c") cfg.P.lambdaCMul = (float)atof(nxt());
        else if (a == "--K") cfg.P.K = atoi(nxt());
        else if (a == "--Krecheck") cfg.P.Krecheck = atoi(nxt());
        else if (a == "--no-txsearch") cfg.P.txSearch = 0;
        else if (a == "--no-txsearch-cl") cfg.P.txSearchCL = 0;
        else if (a == "--no-deltas") cfg.P.deltas = 0;
        else if (a == "--no-cfl") cfg.P.cfl = 0;
        else if (a == "--uvK") cfg.P.uvK = atoi(nxt());
        else if (a == "--shapes") cfg.P.shapeMask = (int)strtol(nxt(), nullptr, 16);
        else if (a == "--part8") cfg.P.partMask8 = (int)strtol(nxt(), nullptr, 16);
        else if (a == "--partN") cfg.P.partMaskN = (int)strtol(nxt(), nullptr, 16);
        else if (a == "--maxpart") cfg.P.maxPart = atoi(nxt());
        else if (a == "--rnd") cfg.P.rnd = (float)atof(nxt());
        else if (a == "--zbin") cfg.P.zbin = (float)atof(nxt());
        else if (a == "--rndc") cfg.P.rndC = (float)atof(nxt());
        else if (a == "--zbinc") cfg.P.zbinC = (float)atof(nxt());
        else if (a == "--satd-lambda") cfg.P.satdLambda = (float)atof(nxt());
        else if (a == "--uv-overhead") cfg.P.uvOverheadBits = (float)atof(nxt());
        else if (a == "--split-bias") cfg.P.splitBias = (float)atof(nxt());
        else if (a == "--qm") { const char *v = nxt(); cfg.qm = !strcmp(v, "auto") ? -2 : atoi(v); }
        else if (a == "--qm-uv") cfg.qmUV = atoi(nxt());
        else if (a == "--edge-filter") cfg.edgeFilter = 1;
        else if (a == "--no-edge-filter") cfg.edgeFilter = 0;
        else if (a == "--lf") { const char *v = nxt(); if (!strcmp(v, "q")) cfg.lfMode = 1; else { cfg.lfMode = 2; cfg.lfLevel = atoi(v); } }
        else if (a == "--lf-mul") { cfg.lfMode = 1; cfg.lfMul = (float)atof(nxt()); }
        else if (a == "--cdef") {  // ypri,ysec,uvpri,uvsec,damping-3
            cfg.cdef = 1;
            sscanf(nxt(), "%d,%d,%d,%d,%d", &cfg.cdefYPri, &cfg.cdefYSec, &cfg.cdefUVPri, &cfg.cdefUVSec, &cfg.cdefDampingMinus3);
        }
        else if (a == "--preset") {
            // speed presets (applied in order with the other options; later options override)
            const char *v = nxt();
            if (!apply_preset(cfg, v)) { fprintf(stderr, "unknown preset %s (slow, base, fast, faster, turbo, turbo2, turbo3)\n", v); return false; }
        }
        else if (a == "--small-preset") {
            smallPreset = nxt();
            if (smallPreset == "none") smallPreset.clear();
            else if (cfg.smallMp <= 0) cfg.smallMp = 1.0;
        }
        else if (a == "--small-mp") cfg.smallMp = atof(nxt());
        else if (a == "--tune") {
            const char *v = nxt();
            // libaom SSIM rdmult scaling: --ssim-lambda 1 (measured +1.5 % worse here).  QM-scaled rounding +
            // QM-weighted RD distortion (libaom's weighted-FP rounding form; dev turbo −1.6 %): on, --qm-rnd 0 --qm-dist 0 = off
            if (!strcmp(v, "iq")) { cfg.tuneIq = 1; cfg.vb = 1; cfg.fLf = 1; cfg.fCdef = 1; cfg.qm = -2; cfg.P.qmRnd = 1; cfg.P.qmDist = 1; }
            else if (!strcmp(v, "none")) {
                const GpConfig d;
                cfg.tuneIq = d.tuneIq; cfg.vb = d.vb; cfg.fLf = d.fLf; cfg.fCdef = d.fCdef; cfg.qm = d.qm;
                cfg.P.qmRnd = d.P.qmRnd; cfg.P.qmDist = d.P.qmDist;
            }
            else { fprintf(stderr, "unknown --tune %s (iq, none)\n", v); return false; }
        }
        else if (a == "--tuneiq-frame") cfg.tuneIq = atoi(nxt());
        else if (a == "--vb") cfg.vb = atoi(nxt());
        else if (a == "--ssim-lambda") cfg.ssimLambda = atoi(nxt());
        else if (a == "--qm-dist") cfg.P.qmDist = atoi(nxt());
        else if (a == "--qm-rnd") cfg.P.qmRnd = atoi(nxt());
        else if (a == "--small-kernel") cfg.P.smallKernel = atoi(nxt());
        else if (a == "--prune-var") cfg.P.pruneVar = (float)atof(nxt());
        else if (a == "--prune-area") cfg.P.pruneMinArea = atoi(nxt());
        else if (a == "--screen-sub") cfg.P.screenSub = atoi(nxt());
        else if (a == "--filter-intra") cfg.P.filterIntra = 1;
        else if (a == "--tx-all") cfg.P.txAllCL = 1;
        else if (a == "--tx-size") cfg.P.txSizeSearch = 1;
        else if (a == "--uv-deltas") cfg.P.uvDeltas = atoi(nxt());
        else if (a == "--prune-flat") cfg.P.pruneFlat = (float)atof(nxt());
        else if (a == "--k-large") cfg.P.kLarge = atoi(nxt());
        else if (a == "--tiles") { sscanf(nxt(), "%f,%d,%d", &cfg.tileMinMp, &cfg.tileColsLog2, &cfg.tileRowsLog2); }
        else if (a == "--deltas-max-area") cfg.P.deltasMaxArea = atoi(nxt());
        else if (a == "--prof") cfg.P.profile = 1;
        else if (a == "--uv-in-dp") { cfg.P.uvInDp = 1; cfg.P.uvOverheadBits = 0.f; }
        else if (a == "--vb-order") { const char *v = nxt(); cfg.vbOrder = !strcmp(v, "row") ? 1 : !strcmp(v, "nearest") ? 2 : 0; }
        else if (a == "--vb-strength") cfg.vbStrength = atoi(nxt());
        else if (a == "--f-lf") { const char *v = nxt(); cfg.fLf = !strcmp(v, "off") ? 0 : !strcmp(v, "q") ? 1 : !strcmp(v, "search") ? 2 : -1; }
        else if (a == "--f-cdef") { const char *v = nxt(); cfg.fCdef = !strcmp(v, "aom") ? 1 : !strcmp(v, "search") ? 2 : 0; if (cfg.fLf < 0) cfg.fLf = 0; }
        else if (a == "--f-dist") cfg.fDist = atoi(nxt());
        else if (a == "--check") o.check = true;
        else if (a == "--gpu-entropy") o.gpuEntropy = true;
        else if (a == "--cpu-entropy") o.gpuEntropy = false;
        else if (a == "--debug-outputs") o.debugOutputs = true;
        else if (a == "--dev-entropy") o.devEntropy = true;
        else if (a == "--check-entropy") o.checkEntropy = true;
        else if (a == "--stats") o.stats = true;
        else if (a == "--verbose") cfg.verbose = true;
        else if (a == "--batch") o.batch = nxt();
        else if (a == "--outdir") o.outdir = nxt();
        else if (a == "-h" || a == "--help") { usage(); exit(0); }
        else if (a.size() > 1 && a[0] == '-' && a[1] == '-') { fprintf(stderr, "unknown option %s\n", a.c_str()); return false; }
        else pos.push_back(a);
    }
    if (o.batch.empty()) {
        if (pos.size() != 2) return false;
        o.in = pos[0];
        o.out = pos[1];
    }
    if (!smallPreset.empty()) {
        GpConfig cs = cfg;
        if (!apply_preset(cs, smallPreset.c_str())) { fprintf(stderr, "unknown small preset %s\n", smallPreset.c_str()); return false; }
        cfg.Psmall = cs.P;
    } else {
        cfg.smallMp = 0;
    }
    if (o.qlist.empty()) o.qlist.push_back(cfg.qindex);
    for (int q : o.qlist)
        if (q < 1 || q > 255) { fprintf(stderr, "qindex must be 1..255\n"); return false; }
    if (o.batch.empty() && o.qlist.size() > 1) { fprintf(stderr, "several qindex values need --batch\n"); return false; }
    cfg.P.K = std::max(1, std::min(cfg.P.K, GP_MAXK));
    cfg.P.Krecheck = std::max(1, std::min(cfg.P.Krecheck, cfg.P.K));
    return true;
}

// ---------------------------------------------------------------------------
// single image
const char *gp_fallback_reason(int r);   // batch.cpp
static int run_single(const GpConfig &cfg, const Opts &o) {
    std::string err;
    GpEncoder enc;
    if (enc.init(cfg, err)) { fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    const GpFileKind kind = gp_sniff(o.in);
    GpFrameIn fin;
    std::vector<uint8_t> icc;
    int orientation = 1;
    ingest_ctx *ictx = nullptr;
    ingest_batch ib;
    memset(&ib, 0, sizeof ib);
    uint8_t *dplanes = nullptr;
    auto t0 = std::chrono::steady_clock::now();
    if (kind == GP_FILE_JPEG) {
        ingest_config ic;
        ingest_default_config(&ic);
        ic.n_workers = 1;
        ic.max_wait_ms = 0;
        ic.activity = 0;
        ic.pinned_bytes = 256ll << 20;   // D round 2: > 6000 px sides go to the CPU route (coefficients staged in this arena)
        char eb[256] = {0};
        ictx = ingest_create(&ic, eb, sizeof eb);
        if (!ictx) { fprintf(stderr, "ingest_create: %s\n", eb); return 1; }
        const char *paths[1] = {o.in.c_str()};
        ingest_submit(ictx, paths, 1);
        ingest_finish_submit(ictx);
        int r = ingest_next_batch(ictx, &ib, -1);
        if (r != 1) {
            ingest_fallback fb;
            int nf = ingest_get_fallbacks(ictx, &fb, 1);
            fprintf(stderr, "%s: %s\n", o.in.c_str(), nf ? gp_fallback_reason(fb.reason) : "decode failed");
            ingest_destroy(ictx);
            return 3;
        }
        const ingest_image &im = ib.images[0];
        fin.W = im.width; fin.H = im.height; fin.mono = im.mono;
        fin.aw = im.aw; fin.ah = im.ah;
        fin.dsrc[0] = ib.pool + im.y_off; fin.spitch[0] = im.y_pitch;
        if (!im.mono) {
            fin.dsrc[1] = ib.pool + im.cb_off; fin.dsrc[2] = ib.pool + im.cr_off;
            fin.spitch[1] = fin.spitch[2] = im.c_pitch;
        }
        if (im.icc_size > 0) icc.assign(im.icc, im.icc + im.icc_size);
        if (!im.oriented) orientation = im.orientation;
    } else {
        GpHostPlanes hp;
        if (!gp_decode_cpu(o.in, hp, err)) { fprintf(stderr, "decode %s: %s\n", o.in.c_str(), err.c_str()); return 3; }
        const size_t ys = (size_t)hp.aw * hp.ah, cs = (size_t)(hp.aw / 2) * (hp.ah / 2);
        if (cudaMalloc(&dplanes, ys + 2 * cs) != cudaSuccess) { fprintf(stderr, "cudaMalloc failed\n"); return 1; }
        cudaMemcpy(dplanes, hp.p[0].data(), ys, cudaMemcpyHostToDevice);
        cudaMemcpy(dplanes + ys, hp.p[1].data(), cs, cudaMemcpyHostToDevice);
        cudaMemcpy(dplanes + ys + cs, hp.p[2].data(), cs, cudaMemcpyHostToDevice);
        fin.W = hp.W; fin.H = hp.H; fin.mono = 0; fin.aw = hp.aw; fin.ah = hp.ah;
        fin.dsrc[0] = dplanes; fin.spitch[0] = hp.aw;
        fin.dsrc[1] = dplanes + ys; fin.dsrc[2] = dplanes + ys + cs;
        fin.spitch[1] = fin.spitch[2] = hp.aw / 2;
    }
    auto t1 = std::chrono::steady_clock::now();
    GpEncoder encS;
    GpEncoder *E = &enc;
    if (cfg.smallMp > 0 && (double)fin.W * fin.H < cfg.smallMp * 1e6) {
        GpConfig cs = cfg;
        cs.P = cfg.Psmall;
        if (encS.init(cs, err)) { fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
        E = &encS;
    }
    std::vector<GpFrameOut> outs;
    GpTimes tm;
    GpDevRecords devRec;
    if (o.devEntropy || o.checkEntropy) {
        // device-resident records -> stream-ordered GPU entropy (batch-mode path)
        std::vector<GpFrameOut> o2;
        GpTimes t2;
        if (E->encode({fin}, o2, t2, err, 0, &devRec)) { fprintf(stderr, "encode(dev): %s\n", err.c_str()); return 1; }
    }
    if (E->encode({fin}, outs, tm, err)) { fprintf(stderr, "encode: %s\n", err.c_str()); return 1; }
    if (ictx) { ingest_release_batch(ictx, ib.batch_id, nullptr); ingest_destroy(ictx); }
    if (dplanes) cudaFree(dplanes);
    auto t2 = std::chrono::steady_clock::now();
    GpFrameOut &f = outs[0];
    f.orientation = orientation;
    std::vector<uint8_t> obu, avif;
    double entMs = 0;
    if (o.gpuEntropy) {
        std::vector<GpFrameOut *> fr = {&f};
        std::vector<const std::vector<uint8_t> *> ic = {&icc};
        std::vector<std::vector<uint8_t>> ob, av;
        if (gp_entropy_gpu(fr, ic, ob, av, &entMs, err)) { fprintf(stderr, "entropy: %s\n", err.c_str()); return 1; }
        obu = ob[0];
        avif = av[0];
    } else if (o.devEntropy) {
        std::vector<std::vector<std::vector<uint8_t>>> tb;
        GpEntScratch es;
        const int erc = gp_entropy_dev(devRec, 0, es, tb, &entMs, err);
        gp_ent_scratch_free(es, 0);
        if (erc || gp_assemble_obus(&devRec.fp[0], tb[0], true, obu, err)) {
            fprintf(stderr, "entropy(dev): %s\n", err.c_str());
            return 1;
        }
        std::vector<uint8_t> o2, a2;
        if (gp_entropy_cpu(f, icc, o2, a2, err)) { fprintf(stderr, "entropy: %s\n", err.c_str()); return 1; }
        avif = a2;  // same OBUs (checked with --check-entropy); mux from the CPU path for brevity
    } else if (gp_entropy_cpu(f, icc, obu, avif, err)) {
        fprintf(stderr, "entropy: %s\n", err.c_str());
        return 1;
    }
    if (o.checkEntropy) {
        std::vector<std::vector<std::vector<uint8_t>>> tb;
        std::vector<uint8_t> dobu, cobu, cavif;
        double ms2 = 0;
        GpEntScratch es;
        const int erc = gp_entropy_dev(devRec, 0, es, tb, &ms2, err);
        gp_ent_scratch_free(es, 0);
        if (erc || gp_assemble_obus(&devRec.fp[0], tb[0], true, dobu, err) ||
            gp_entropy_cpu(f, icc, cobu, cavif, err)) {
            fprintf(stderr, "check-entropy: %s\n", err.c_str());
            return 6;
        }
        if (dobu != cobu) { fprintf(stderr, "check-entropy: MISMATCH dev %zu bytes vs cpu %zu bytes\n", dobu.size(), cobu.size()); return 6; }
        fprintf(stderr, "check-entropy: device entropy == CPU writer (%zu bytes, %.1f ms GPU)\n", dobu.size(), ms2);
    }
    if (devRec.pool) gp_dev_records_free(devRec, 0);
    auto t3 = std::chrono::steady_clock::now();
    if (!gp_write_file(o.out, avif.data(), avif.size()) ||
        (o.debugOutputs && !gp_write_file(o.out + ".obu", obu.data(), obu.size()))) {
        fprintf(stderr, "write failed\n");
        return 1;
    }
    if (o.debugOutputs) {
        std::vector<uint8_t> yuv;
        for (int p = 0; p < (f.fp.monochrome ? 1 : 3); p++) yuv.insert(yuv.end(), f.recon[p].begin(), f.recon[p].end());
        if (!gp_write_file(o.out + ".recon.yuv", yuv.data(), yuv.size())) { fprintf(stderr, "write failed\n"); return 1; }
    }
    int rc = 0;
    if (o.check) {
        av1g_decision_record r = f.record();
        av1g_encode_opts eo;
        eo.flags = 0;
        eo.seq_level_idx = -1;
        av1g_output ao;
        memset(&ao, 0, sizeof ao);
        if (av1g_encode(&r, &eo, &ao) != AV1G_OK) {
            fprintf(stderr, "check: av1g_encode failed: %s\n", ao.error);
            rc = 4;
        } else {
            for (int p = 0; p < (f.fp.monochrome ? 1 : 3); p++) {
                const int w = ao.recon_w[p], h = ao.recon_h[p];
                for (int y = 0; y < h && !rc; y++)
                    for (int x = 0; x < w; x++)
                        if (ao.recon[p][(size_t)y * w + x] != f.recon[p][(size_t)y * w + x]) {
                            fprintf(stderr, "check: MISMATCH plane %d at (%d,%d): av1core %d, gpu %d\n", p, x, y,
                                    ao.recon[p][(size_t)y * w + x], f.recon[p][(size_t)y * w + x]);
                            rc = 5;
                            break;
                        }
                if (rc) break;
            }
            if (!rc) fprintf(stderr, "check: recon identical to av1core (%zu bytes obu)\n", obu.size());
        }
        av1g_output_free(&ao);
    }
    if (cfg.P.profile) {
        const char *nm[9] = {"luma setup", "luma pass1", "luma pass2", "luma final", "chroma edges+CfL", "chroma SATD",
                             "chroma RD", "chroma final", "maps+record"};
        double tot = 0;
        for (int k = 0; k < 9; k++) tot += (double)tm.prof[k];
        for (int k = 0; k < 9; k++) fprintf(stderr, "prof %-18s %5.1f %%\n", nm[k], 100.0 * tm.prof[k] / std::max(1.0, tot));
    }
    if (o.stats) {
        auto ms = [](std::chrono::steady_clock::time_point a, std::chrono::steady_clock::time_point b) {
            return std::chrono::duration<double, std::milli>(b - a).count();
        };
        printf("{\"file\":\"%s\",\"w\":%d,\"h\":%d,\"q\":%d,\"bytes\":%zu,\"load_ms\":%.2f,\"gpu_total_ms\":%.2f,"
               "\"analysis_ms\":%.2f,\"dp_ms\":%.2f,\"recon_ms\":%.2f,\"pack_ms\":%.2f,\"postfilter_ms\":%.2f,\"download_ms\":%.2f,"
               "\"entropy_ms\":%.2f,\"wall_encode_ms\":%.2f,\"steps\":%u,\"blocks\":%u,\"skip\":%u,\"cfl\":%u,\"fixes\":%u,\"txsplit\":%u,"
               "\"dev_mb\":%.1f}\n",
               o.in.c_str(), f.fp.width, f.fp.height, cfg.qindex, avif.size(), ms(t0, t1), tm.total, tm.analysis, tm.dp,
               tm.recon, tm.pack, tm.postfilter, tm.download, o.gpuEntropy ? entMs : ms(t2, t3), ms(t1, t2), tm.steps, tm.blocks,
               tm.skipBlocks, tm.cflBlocks, tm.rangeFixes, tm.txSplit, tm.deviceBytes / 1048576.0);
    }
    return rc;
}

int run_batch_mode(const GpConfig &cfg, const std::string &list, const std::string &outdir, bool gpuEntropy, bool stats,
                   const std::string &suffix);

int main(int argc, char **argv) {
    GpConfig cfg;
    Opts o;
    if (!parse(argc, argv, cfg, o)) { usage(); return 2; }
    // blocking sync: host threads sleep while waiting for the GPU (CPU budget <= 4 cores)
    // more hardware work queues than the default 8: ingest (nvJPEG + per-worker streams), encoders
    // (main + high-priority wavefront stream) and entropy threads use > 20 streams; aliased queues
    // serialise unrelated work (nsys 2026-10-02: next batch's analysis queued behind entropy)
    setenv("CUDA_DEVICE_MAX_CONNECTIONS", getenv("GP_MAX_CONNECTIONS") ? getenv("GP_MAX_CONNECTIONS") : "32", 0);
    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
    if (cudaSetDevice(0) != cudaSuccess) { fprintf(stderr, "no CUDA device\n"); return 1; }
    if (!o.batch.empty()) {
        if (o.debugOutputs) setenv("GP_WANT_RECON", "1", 1);
        // one process for every q point (CUDA context, tables and the process start are paid once)
        int rc = 0;
        for (int q : o.qlist) {
            GpConfig c = cfg;
            c.qindex = q;
            const std::string suffix = (o.qlist.size() > 1 || o.nameQ) ? ".q" + std::to_string(q) : "";
            rc |= run_batch_mode(c, o.batch, o.outdir, o.gpuEntropy, o.stats, suffix);
        }
        return rc;
    }
    return run_single(cfg, o);
}
