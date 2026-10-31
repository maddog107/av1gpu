// Batch mode = many images in flight.  The ingest module delivers GPU-resident 4:2:0 (--yuv 444: 4:4:4) planes in
// size-classed batches (NVJPG hardware decode, CPU workers only parse/read); each ingest batch is
// encoded as one GPU batch (analysis -> DP -> wavefront closed loop -> pack), then entropy coded
// (GPU: av1core GPU entropy, or CPU writer threads) and written.  Reports img/s, MP/s, per-stage times, CPU use.
#include "pipeline.h"
#include "imageio.h"
#include "ingest_api.h"
#include "av1g/av1g.h"

#include <cuda_runtime.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

namespace {
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
double cpu_ms() {
    rusage r;
    getrusage(RUSAGE_SELF, &r);
    return (r.ru_utime.tv_sec + r.ru_stime.tv_sec) * 1e3 + (r.ru_utime.tv_usec + r.ru_stime.tv_usec) / 1e3;
}
std::string base_name(const std::string &p) {
    const size_t s = p.find_last_of('/');
    return s == std::string::npos ? p : p.substr(s + 1);
}
bool has_image_ext(const std::string &n) {
    const size_t d = n.find_last_of('.');
    if (d == std::string::npos) return false;
    std::string e = n.substr(d + 1);
    for (auto &c : e) c = (char)tolower((unsigned char)c);
    return e == "jpg" || e == "jpeg" || e == "jfif" || e == "png" || e == "webp";
}
// LIST is a text file with one path per line, or a directory (its JPEG/PNG/WebP files, sorted)
bool list_inputs(const std::string &list, std::vector<std::string> &paths) {
    struct stat st;
    if (stat(list.c_str(), &st) != 0) return false;
    if (S_ISDIR(st.st_mode)) {
        DIR *dp = opendir(list.c_str());
        if (!dp) return false;
        while (dirent *e = readdir(dp))
            if (e->d_name[0] != '.' && has_image_ext(e->d_name)) paths.push_back(list + "/" + e->d_name);
        closedir(dp);
        std::sort(paths.begin(), paths.end());
        return true;
    }
    std::ifstream f(list);
    std::string l;
    while (std::getline(f, l))
        if (!l.empty()) paths.push_back(l);
    return true;
}
struct Agg {
    double ingestWait = 0, gpuSetup = 0, analysis = 0, dp = 0, recon = 0, pack = 0, postfilter = 0, download = 0, gpuTotal = 0;
    double entropyGpu = 0, entropyWall = 0, writeWall = 0, encodeWall = 0;
    uint64_t images = 0, pixels = 0, bytes = 0, blocks = 0, batches = 0;
    uint64_t alphaImages = 0, alphaPixels = 0;   // images with an alpha item (their alpha frames)
    unsigned long long prof[16] = {0};
    size_t peakDev = 0;
};
}  // namespace


// debug / batch gate (GP_WANT_RECON): <out>.obu (temporal delimiter + sequence header + frame) and the
// encoder's reconstruction <out>.recon.yuv (visible planes, Y then U then V) for decoder comparison; an image with an
// alpha item also gets <out>.alpha.obu and <out>.alpha.recon.yuv (the monochrome alpha frame)
static bool gp_write_debug1(const std::string &base, const GpFrameOut &o, const std::vector<uint8_t> &obu) {
    std::vector<uint8_t> yuv;
    for (int p = 0; p < 3; p++) yuv.insert(yuv.end(), o.recon[p].begin(), o.recon[p].end());
    std::vector<uint8_t> td;
    const bool hasTd = obu.size() >= 2 && obu[0] == 0x12 && obu[1] == 0x00;
    if (!hasTd) { td.push_back(0x12); td.push_back(0x00); }
    td.insert(td.end(), obu.begin(), obu.end());
    return gp_write_file(base + ".recon.yuv", yuv.data(), yuv.size()) && gp_write_file(base + ".obu", td.data(), td.size());
}
static bool gp_write_debug(const std::string &avifPath, const GpFrameOut &o, const std::vector<uint8_t> &obu,
                           const GpFrameOut *alpha = nullptr, const std::vector<uint8_t> *alphaObu = nullptr) {
    return gp_write_debug1(avifPath, o, obu) && (!alpha || !alphaObu || gp_write_debug1(avifPath + ".alpha", *alpha, *alphaObu));
}

const char *gp_jpeg_sub_name(int s) {
    switch (s) {
        case INGEST_SUB_420: return "4:2:0";
        case INGEST_SUB_444: return "4:4:4";
        case INGEST_SUB_422: return "4:2:2";
        case INGEST_SUB_440: return "4:4:0";
        case INGEST_SUB_411: return "4:1:1";
        case INGEST_SUB_GRAY: return "gray";
        default: return "non-standard sampling";
    }
}

const char *gp_fallback_reason(int r) {
    switch (r) {
        case INGEST_FB_READ_ERROR: return "read error";
        case INGEST_FB_DECODE_ERROR: return "not a decodable JPEG";
        case INGEST_FB_GIANT: return "larger than 35.6 megapixels";
        case INGEST_FB_CMYK: return "CMYK/YCCK JPEG";
        case INGEST_FB_COLORSPACE: return "unsupported JPEG colour space";
        case INGEST_FB_PRECISION: return "not an 8-bit JPEG";
        case INGEST_FB_TOO_BIG: return "too large for the staging buffers";
        default: return "unsupported";
    }
}

int run_batch_mode(const GpConfig &cfg, const std::string &list, const std::string &outdir, bool gpuEntropy, bool stats,
                   const std::string &suffix) {
    std::vector<std::string> paths;
    if (!list_inputs(list, paths)) { fprintf(stderr, "cannot read %s\n", list.c_str()); return 2; }
    if (paths.empty()) { fprintf(stderr, "no input files in %s\n", list.c_str()); return 2; }
    const bool write = !outdir.empty();
    if (write) mkdir(outdir.c_str(), 0755);
    std::string err;
    GpEncoder enc;
    GpConfig bcfg = cfg;
    bcfg.wantRecon = getenv("GP_WANT_RECON") != nullptr;
    if (enc.init(bcfg, err)) { fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    const char *bmp = getenv("GP_BATCH_MP");
    const char *nw = getenv("GP_INGEST_WORKERS");
    const int nEntThreads = getenv("GP_ENTROPY_THREADS") ? atoi(getenv("GP_ENTROPY_THREADS")) : 3;  // GPU: concurrent entropy batches; CPU: writer threads
    ingest_config ic;
    ingest_default_config(&ic);
    ic.n_workers = nw ? atoi(nw) : 4;
    ic.activity = 0;
    ic.max_inflight = getenv("GP_INGEST_INFLIGHT") ? atoi(getenv("GP_INGEST_INFLIGHT")) : 24;
    ic.batch_pixels = (int64_t)((bmp ? atof(bmp) : 16.0) * 1e6);
    ic.pinned_bytes = 256ll << 20;
    if (getenv("GP_INGEST_DECODER")) ic.decoder = atoi(getenv("GP_INGEST_DECODER"));   // 1 = CPU entropy decode + GPU IDCT (no nvJPEG/NVJPG)
    if (getenv("GP_INGEST_HWBATCH")) ic.hw_batch = atoi(getenv("GP_INGEST_HWBATCH"));
    ic.out_444 = cfg.lossless ? 2 : cfg.yuv444;   // --yuv 444: full-resolution chroma planes; --lossless: the JPEG's own format
    char eb[256] = {0};
    const double t0 = now_ms(), c0 = cpu_ms();
    ingest_ctx *ictx = ingest_create(&ic, eb, sizeof eb);
    if (!ictx) { fprintf(stderr, "ingest_create: %s\n", eb); return 1; }
    std::vector<const char *> cp;
    for (auto &p : paths) cp.push_back(p.c_str());
    ingest_submit(ictx, cp.data(), (int)cp.size());
    ingest_finish_submit(ictx);
    Agg A;
    int failures = 0;
    struct Job {
        std::vector<GpFrameOut> outs;
        std::vector<std::vector<uint8_t>> iccs;
        std::vector<int> orients;   // EXIF orientation per image (AVIF irot/imir)
        std::vector<std::string> names;
        GpDevRecords dev;   // device-resident records (gpuEntropy)
        // alpha: per image its monochrome alpha frame (W = 0: none), encoded after the colour frames as one more
        // batch of the job; alphaOf[i] = the image's index in aouts / adev, -1 without alpha
        std::vector<GpFrameIn> afin;
        std::vector<int> alphaOf;
        std::vector<GpFrameOut> aouts;
        GpDevRecords adev;
    };
    auto emit = [&](Job &j) -> int {
        std::vector<GpFrameOut> &outs = j.outs;
        const std::vector<std::vector<uint8_t>> &iccs = j.iccs;
        const std::vector<int> &orients = j.orients;
        const std::vector<std::string> &names = j.names;
        const size_t n = outs.size();
        for (size_t i = 0; i < n && i < orients.size(); i++) outs[i].orientation = orients[i];
        std::string err;  // shadows the main thread's (emit runs on the entropy thread)
        std::vector<std::vector<uint8_t>> obus(n), avifs(n), aobus(n);
        auto alphaOut = [&](size_t i) -> GpFrameOut * {
            return i < j.alphaOf.size() && j.alphaOf[i] >= 0 ? &j.aouts[(size_t)j.alphaOf[i]] : nullptr;
        };
        const double te = now_ms();
        if (gpuEntropy) {
            std::vector<GpFrameOut *> fr, al;
            std::vector<const std::vector<uint8_t> *> ic2;
            for (size_t i = 0; i < n; i++) { fr.push_back(&outs[i]); ic2.push_back(&iccs[i]); al.push_back(alphaOut(i)); }
            double gms = 0;
            if (gp_entropy_gpu(fr, ic2, obus, avifs, &gms, err, &al, &aobus)) { fprintf(stderr, "entropy: %s\n", err.c_str()); return -1; }
            A.entropyGpu += gms;
        } else {
            std::atomic<size_t> next{0};
            std::atomic<int> bad{0};
            auto work = [&]() {
                std::string e2;
                for (;;) {
                    const size_t i = next.fetch_add(1);
                    if (i >= n) break;
                    if (gp_entropy_cpu(outs[i], iccs[i], obus[i], avifs[i], e2, alphaOut(i), &aobus[i])) bad++;
                    if (const char *rd = getenv("GP_DUMP_RECORDS")) {
                        // debug: raw DecisionRecord arrays (fp, blocks, block_txb, txb, coeffs) per image
                        const GpFrameOut &o = outs[i];
                        if (FILE *f = fopen((std::string(rd) + "/" + names[i] + ".rec").c_str(), "wb")) {
                            const uint64_t c[4] = {o.blocks.size(), o.block_txb.size(), o.txb.size(), o.coeffs.size()};
                            fwrite(c, 8, 4, f);
                            fwrite(&o.fp, sizeof o.fp, 1, f);
                            fwrite(o.blocks.data(), sizeof(av1g_block), c[0], f);
                            fwrite(o.block_txb.data(), 4, c[1], f);
                            fwrite(o.txb.data(), sizeof(av1g_txb), c[2], f);
                            fwrite(o.coeffs.data(), 2, c[3], f);
                            fclose(f);
                        }
                    }
                }
            };
            std::vector<std::thread> th;
            for (int k = 0; k < nEntThreads; k++) th.emplace_back(work);
            for (auto &t : th) t.join();
            if (bad) { fprintf(stderr, "entropy failed for %d images\n", bad.load()); return -1; }
        }
        const double tw = now_ms();
        A.entropyWall += tw - te;
        for (size_t i = 0; i < n; i++) {
            A.bytes += avifs[i].size();
            if (write) {
                const std::string o = outdir + "/" + names[i] + suffix + ".avif";
                if (!gp_write_file(o, avifs[i].data(), avifs[i].size())) failures++;
                if (getenv("GP_WANT_RECON") && !gp_write_debug(o, outs[i], obus[i], alphaOut(i), &aobus[i])) failures++;
            }
        }
        A.writeWall += now_ms() - tw;
        return 0;
    };
    // ---- pipeline: main thread = ingest + GPU decisions/recon (stream es); entropy thread = GPU entropy
    // (av1core GPU entropy, per-thread default stream) + mux + write of the previous batch ----
    const double targetMp = getenv("GP_ENCODE_MP") ? atof(getenv("GP_ENCODE_MP")) : 48.0;
    const int maxImgs = getenv("GP_ENCODE_IMGS") ? atoi(getenv("GP_ENCODE_IMGS")) : 192;
    cudaStream_t es;
    cudaStreamCreateWithFlags(&es, cudaStreamNonBlocking);
    std::mutex mu, aggMu;
    std::condition_variable cv;
    std::deque<Job *> q;
    bool done = false;
    std::atomic<int> entFail{0};
    // entropy threads: device-resident records -> stream-ordered GPU entropy (several batches in flight,
    // overlapping the encoders), or the CPU writer (merged jobs, worker threads)
    std::vector<std::thread> entThreads;
    for (int et = 0; et < (gpuEntropy ? nEntThreads : 1); et++)
        entThreads.emplace_back([&]() {
            cudaStream_t ts;
            cudaStreamCreateWithFlags(&ts, cudaStreamNonBlocking);
            std::string terr;
            GpEntScratch escr;
            for (;;) {
                Job *j;
                {
                    std::unique_lock<std::mutex> lk(mu);
                    cv.wait(lk, [&] { return !q.empty() || done; });
                    if (q.empty()) break;
                    j = q.front();
                    q.pop_front();
                }
                cv.notify_all();
                if (gpuEntropy) {
                    const double te = now_ms();
                    std::vector<std::vector<std::vector<uint8_t>>> tb;
                    double gms = 0;
                    if (gp_entropy_dev(j->dev, ts, escr, tb, &gms, terr)) { fprintf(stderr, "entropy: %s\n", terr.c_str()); entFail++; }
                    gp_dev_records_free(j->dev, ts);
                    std::vector<std::vector<std::vector<uint8_t>>> atb;   // alpha frames of the job (one more entropy batch)
                    if (j->adev.pool) {
                        double gms2 = 0;
                        if (gp_entropy_dev(j->adev, ts, escr, atb, &gms2, terr)) { fprintf(stderr, "entropy (alpha): %s\n", terr.c_str()); entFail++; }
                        gp_dev_records_free(j->adev, ts);
                        gms += gms2;
                    }
                    const double tm = now_ms();
                    uint64_t bytes = 0;
                    int wf = 0;
                    for (size_t i = 0; i < tb.size(); i++) {
                        std::vector<uint8_t> obu, aobu, av;
                        if (gp_assemble_obus(&j->dev.fp[i], tb[i], false, obu, terr)) { entFail++; continue; }
                        const int ai = i < j->alphaOf.size() ? j->alphaOf[i] : -1;
                        if (ai >= 0 && ((size_t)ai >= atb.size() || gp_assemble_obus(&j->adev.fp[ai], atb[ai], false, aobu, terr))) { entFail++; continue; }
                        if (gp_mux_avif(j->dev.fp[i], obu.data(), obu.size(), ai >= 0 ? aobu.data() : nullptr, aobu.size(), j->iccs[i],
                                        j->orients[i], av, terr)) { entFail++; continue; }
                        bytes += av.size();
                        if (write && !gp_write_file(outdir + "/" + j->names[i] + suffix + ".avif", av.data(), av.size())) wf++;
                        if (write && bcfg.wantRecon && i < j->outs.size() &&
                            !gp_write_debug(outdir + "/" + j->names[i] + suffix + ".avif", j->outs[i], obu,
                                            ai >= 0 && (size_t)ai < j->aouts.size() ? &j->aouts[ai] : nullptr, &aobu))
                            wf++;
                    }
                    std::lock_guard<std::mutex> lk2(aggMu);
                    A.entropyGpu += gms;
                    A.entropyWall += tm - te;
                    A.writeWall += now_ms() - tm;
                    A.bytes += bytes;
                    failures += wf;
                } else if (emit(*j)) {
                    entFail++;
                }
                delete j;
            }
            gp_ent_scratch_free(escr, ts);
            cudaStreamSynchronize(ts);
            cudaStreamDestroy(ts);
        });
    auto push = [&](Job *j) {
        std::unique_lock<std::mutex> lk(mu);
        cv.wait(lk, [&] { return q.size() < 3; });  // bounded queue towards the entropy thread
        q.push_back(j);
        lk.unlock();
        cv.notify_all();
    };
    // N encoder threads (separate streams/encoders) so one batch's latency-bound wavefront overlaps the
    // next batch's throughput-bound analysis.  Ingest batches are gathered under a lock.
    const int nEnc = getenv("GP_ENCODERS") ? std::max(1, atoi(getenv("GP_ENCODERS"))) : 3;
    std::mutex inMu;
    bool more = true;
    std::atomic<int> encFail{0};
    // WebP/PNG named .jpg (ingest fallbacks): a side thread decodes them on the CPU as soon as ingest reports
    // them, uploads the planes and queues them; the encoder threads take them into their next batch like
    // ingest images (GPU entropy, overlapped).  The rest (giant, CMYK, ...) -> CPU avifenc (not done here).
    // (Before: serial CPU decode + single-image encode + host-path entropy after the main loop, ~0.45 s per
    // image with the GPU mostly idle.)
    struct FbFrame { GpFrameIn f, a; uint8_t *d = nullptr; std::string name; };   // a: alpha frame (W = 0: none)
    std::mutex fbMu;
    std::deque<FbFrame> fbReady;
    std::atomic<bool> ingestDrained{false}, fbDone{false};
    std::atomic<int> cpuFallbackA{0}, gpuFallbackA{0};
    std::thread fbThread([&]() {
        // GP_FALLBACK_WORKERS (default 4) threads decode + upload the fallback images of each poll in parallel
        const int nfw = getenv("GP_FALLBACK_WORKERS") ? std::max(1, atoi(getenv("GP_FALLBACK_WORKERS"))) : 4;
        std::vector<cudaStream_t> fss(nfw);
        for (auto &x : fss) cudaStreamCreateWithFlags(&x, cudaStreamNonBlocking);
        std::vector<ingest_fallback> fb(4096);
        int handled = 0;
        auto one = [&](const ingest_fallback &fe, cudaStream_t fs) {
            const std::string &p = paths[fe.index];
            const GpFileKind kind = gp_sniff(p);
            if (kind != GP_FILE_WEBP && kind != GP_FILE_PNG) {
                fprintf(stderr, "[av1gpu] skipped %s: %s\n", p.c_str(), gp_fallback_reason(fe.reason));
                cpuFallbackA++;
                return;
            }
            GpHostPlanes hp;
            std::string e2, note;
            int matrix = -1;
            const bool keepAlpha = cfg.alpha != 0;
            const bool dec = cfg.lossless ? gp_decode_lossless(p, hp, cfg.ycgcoRe, &matrix, note, e2, keepAlpha)
                                          : gp_decode_cpu(p, hp, e2, cfg.yuv444, cfg.depth, keepAlpha, &note);
            if (dec && !note.empty()) fprintf(stderr, "[av1gpu] %s: %s\n", p.c_str(), note.c_str());
            if (!dec || (double)hp.W * hp.H > 35.6e6) {
                fprintf(stderr, "[av1gpu] skipped %s: %s\n", p.c_str(), e2.empty() ? "larger than 35.6 megapixels" : e2.c_str());
                cpuFallbackA++;
                return;
            }
            const size_t ys = hp.bytes(0), cs = hp.bytes(1), as = hp.abytes();   // bytes (10-bit: uint16_t samples)
            uint8_t *d = nullptr;
            if (cudaMallocAsync((void **)&d, ys + 2 * cs + as, fs) != cudaSuccess) { cpuFallbackA++; return; }
            cudaMemcpyAsync(d, hp.data(0), ys, cudaMemcpyHostToDevice, fs);
            cudaMemcpyAsync(d + ys, hp.data(1), cs, cudaMemcpyHostToDevice, fs);
            cudaMemcpyAsync(d + ys + cs, hp.data(2), cs, cudaMemcpyHostToDevice, fs);
            if (as) cudaMemcpyAsync(d + ys + 2 * cs, hp.adata(), as, cudaMemcpyHostToDevice, fs);
            if (cudaStreamSynchronize(fs) != cudaSuccess) { cpuFallbackA++; return; }
            FbFrame F;
            F.d = d;
            F.name = base_name(p);
            F.f.W = hp.W; F.f.H = hp.H; F.f.aw = hp.aw; F.f.ah = hp.ah; F.f.yuv444 = hp.yuv444; F.f.bitDepth = hp.bitDepth;
            F.f.mono = hp.mono; F.f.colorMatrix = matrix;
            F.f.dsrc[0] = d; F.f.dsrc[1] = d + ys; F.f.dsrc[2] = d + ys + cs;
            F.f.spitch[0] = hp.aw; F.f.spitch[1] = F.f.spitch[2] = hp.cw();
            if (as) {   // the alpha plane: a monochrome frame of the same size and bit depth, unspecified colour description
                F.a.W = hp.W; F.a.H = hp.H; F.a.aw = hp.aw; F.a.ah = hp.ah; F.a.mono = 1; F.a.bitDepth = hp.bitDepth;
                F.a.colorMatrix = -2;
                F.a.dsrc[0] = d + ys + 2 * cs;
                F.a.spitch[0] = hp.aw;
            }
            std::lock_guard<std::mutex> lk(fbMu);
            fbReady.push_back(F);
        };
        for (;;) {
            const bool last = ingestDrained.load();   // all fallbacks are recorded once ingest is drained
            int nfb = ingest_get_fallbacks(ictx, fb.data(), (int)fb.size());
            if (nfb > (int)fb.size()) {   // long lists: grow instead of dropping fallbacks beyond 4096 (old code did)
                fb.resize((size_t)nfb * 2);
                nfb = std::min(ingest_get_fallbacks(ictx, fb.data(), (int)fb.size()), (int)fb.size());
            }
            if (handled < nfb) {
                std::atomic<int> next{handled};
                std::vector<std::thread> th;
                for (int w = 0; w < std::min(nfw, nfb - handled); w++)
                    th.emplace_back([&, w]() {
                        for (int k; (k = next.fetch_add(1)) < nfb;) one(fb[k], fss[w]);
                    });
                for (auto &t : th) t.join();
                handled = nfb;
            }
            if (last && handled >= nfb) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        for (auto &x : fss) cudaStreamDestroy(x);
        fbDone = true;
    });
    // alpha frames of a job: one more GPU batch (monochrome frames, gp_alpha_config: lossless or lossy at the alpha
    // quality), on the job's stream after its colour frames; each encoder thread creates its alpha encoders on first use
    GpConfig acfg[2] = {gp_alpha_config(bcfg), bcfg};
    acfg[1].P = bcfg.Psmall;
    acfg[1] = gp_alpha_config(acfg[1]);
    struct AlphaEncs {
        GpEncoder *e[2] = {nullptr, nullptr};   // [0]: main preset, [1]: --small-preset
        ~AlphaEncs() { delete e[0]; delete e[1]; }
    };
    auto encodeAlpha = [&](Job *j, int small, AlphaEncs &ea, cudaStream_t s, std::string &eerr) -> bool {
        std::vector<GpFrameIn> al;
        j->alphaOf.assign(j->afin.size(), -1);
        for (size_t i = 0; i < j->afin.size(); i++)
            if (j->afin[i].W) { j->alphaOf[i] = (int)al.size(); al.push_back(j->afin[i]); }
        if (al.empty()) return true;
        if (!ea.e[small]) {
            ea.e[small] = new GpEncoder;
            if (ea.e[small]->init(acfg[small], eerr)) return false;
        }
        GpTimes ta;
        if (ea.e[small]->encode(al, j->aouts, ta, eerr, s, gpuEntropy ? &j->adev : nullptr)) return false;
        std::lock_guard<std::mutex> lk2(aggMu);
        A.analysis += ta.analysis; A.recon += ta.recon; A.gpuTotal += ta.total; A.blocks += ta.blocks;
        A.peakDev = std::max(A.peakDev, ta.deviceBytes);
        A.alphaImages += al.size();
        for (auto &f : al) A.alphaPixels += (uint64_t)f.W * f.H;
        return true;
    };
    auto encLoop = [&](GpEncoder *E, GpEncoder *Es, cudaStream_t s) {
        std::string eerr;
        AlphaEncs ea;
        for (;;) {
            std::vector<ingest_batch> held;
            std::vector<GpFrameIn> fin;
            Job *job = new Job;
            double mpAcc = 0;
            std::vector<uint8_t *> fbHeld;
            int fbConv = 0;   // 1 when fbHeld also holds the 10-bit conversion buffer (not a fallback image)
            {
                std::lock_guard<std::mutex> lk(fbMu);
                while (!fbReady.empty() && mpAcc < targetMp && (int)fin.size() < maxImgs) {
                    const FbFrame F = fbReady.front();
                    fbReady.pop_front();
                    fin.push_back(F.f);
                    job->iccs.emplace_back();
                    job->orients.push_back(1);
                    job->names.push_back(F.name);
                    job->afin.push_back(F.a);
                    fbHeld.push_back(F.d);
                    mpAcc += F.f.W * (double)F.f.H / 1e6;
                }
            }
            {
                std::lock_guard<std::mutex> lk(inMu);
                while (more && mpAcc < targetMp && (int)fin.size() < maxImgs && (int)held.size() < ic.max_inflight / (nEnc + 1)) {
                    const double tw = now_ms();
                    ingest_batch b;
                    const int r = ingest_next_batch(ictx, &b, -1);
                    {
                        std::lock_guard<std::mutex> lk2(aggMu);
                        A.ingestWait += now_ms() - tw;
                    }
                    if (r == 0) { more = false; ingestDrained = true; break; }
                    if (r < 0) { char e2[256]; ingest_last_error(ictx, e2, sizeof e2); fprintf(stderr, "ingest: %s\n", e2); more = false; ingestDrained = true; encFail++; break; }
                    held.push_back(b);
                    for (int i = 0; i < b.n_images; i++) {
                        const ingest_image &im = b.images[i];
                        if (cfg.lossless && im.src_sub != INGEST_SUB_420 && im.src_sub != INGEST_SUB_444 && im.src_sub != INGEST_SUB_GRAY) {
                            fprintf(stderr, "[av1gpu] skipped %s: --lossless keeps a JPEG's YCbCr planes, AV1 can only do that for 4:2:0, "
                                            "4:4:4 and gray JPEGs (this one is %s)\n", paths[im.index].c_str(), gp_jpeg_sub_name(im.src_sub));
                            cpuFallbackA++;
                            continue;
                        }
                        GpFrameIn f;
                        f.W = im.width; f.H = im.height; f.mono = im.mono; f.aw = im.aw; f.ah = im.ah; f.yuv444 = im.chroma_full;
                        f.dsrc[0] = b.pool + im.y_off; f.spitch[0] = im.y_pitch;
                        if (!im.mono) { f.dsrc[1] = b.pool + im.cb_off; f.dsrc[2] = b.pool + im.cr_off; f.spitch[1] = f.spitch[2] = im.c_pitch; }
                        fin.push_back(f);
                        job->iccs.emplace_back(im.icc_size > 0 ? std::vector<uint8_t>(im.icc, im.icc + im.icc_size) : std::vector<uint8_t>());
                        job->orients.push_back(im.oriented ? 1 : im.orientation);
                        job->names.push_back(base_name(paths[im.index]));
                        job->afin.emplace_back();   // (JPEG: no alpha)
                        mpAcc += im.width * (double)im.height / 1e6;
                    }
                }
            }
            if (!fin.empty() && cfg.depth == 10 && !cfg.lossless) {
                // --depth 10: ingest frames (8-bit JPEG samples) -> 10-bit planes; freed with the fallback planes
                uint8_t *cb = nullptr;
                if (gp_frames_to_10bit(fin, s, &cb, eerr)) { fprintf(stderr, "10-bit conversion: %s\n", eerr.c_str()); encFail++; delete job; return; }
                if (cb) { fbHeld.push_back(cb); fbConv = 1; }
            }
            if (fin.empty()) {
                delete job;
                bool idle;
                {
                    std::lock_guard<std::mutex> lk(inMu);
                    idle = !more;
                }
                if (idle && fbDone.load()) {
                    std::lock_guard<std::mutex> lk(fbMu);
                    if (fbReady.empty()) return;
                }
                if (encFail) return;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));   // fallback frames still being decoded
                continue;
            }
            GpTimes tm;
            const double ts = now_ms();
            // lossless batches may mix 8- and 10-bit frames (16-bit PNGs, --ycgco-re): the 10-bit ones get their own encode
            if (cfg.lossless && std::any_of(fin.begin(), fin.end(), [](const GpFrameIn &f) { return f.bitDepth == 10; }) &&
                std::any_of(fin.begin(), fin.end(), [](const GpFrameIn &f) { return f.bitDepth != 10; })) {
                Job *jd = new Job;
                std::vector<GpFrameIn> f10, f8, a8;
                std::vector<std::vector<uint8_t>> i8;
                std::vector<int> o8;
                std::vector<std::string> n8;
                for (size_t i = 0; i < fin.size(); i++) {
                    if (fin[i].bitDepth == 10) {
                        f10.push_back(fin[i]); jd->iccs.push_back(job->iccs[i]); jd->orients.push_back(job->orients[i]); jd->names.push_back(job->names[i]);
                        jd->afin.push_back(job->afin[i]);
                    } else {
                        f8.push_back(fin[i]); i8.push_back(job->iccs[i]); o8.push_back(job->orients[i]); n8.push_back(job->names[i]);
                        a8.push_back(job->afin[i]);
                    }
                }
                GpTimes t10;
                if (E->encode(f10, jd->outs, t10, eerr, s, gpuEntropy ? &jd->dev : nullptr) || !encodeAlpha(jd, 0, ea, s, eerr)) {
                    fprintf(stderr, "encode: %s\n", eerr.c_str()); encFail++; delete jd; delete job; return;
                }
                {
                    std::lock_guard<std::mutex> lk2(aggMu);
                    A.analysis += t10.analysis; A.recon += t10.recon; A.gpuTotal += t10.total; A.blocks += t10.blocks;
                    A.peakDev = std::max(A.peakDev, t10.deviceBytes);
                    A.images += f10.size();
                    for (auto &f : f10) A.pixels += (uint64_t)f.W * f.H;
                }
                push(jd);
                fin.swap(f8); job->iccs.swap(i8); job->orients.swap(o8); job->names.swap(n8); job->afin.swap(a8);
            }
            // size-adaptive effort: small images go to a second encoder with cfg.Psmall (own job)
            if (Es && !cfg.lossless) {
                std::vector<int> sm;
                for (size_t i = 0; i < fin.size(); i++)
                    if ((double)fin[i].W * fin[i].H < bcfg.smallMp * 1e6) sm.push_back((int)i);
                if (!sm.empty()) {
                    Job *js = new Job;
                    std::vector<GpFrameIn> fs;
                    std::vector<char> isSmall(fin.size(), 0);
                    for (int i : sm) {
                        isSmall[i] = 1;
                        fs.push_back(fin[i]);
                        js->iccs.push_back(job->iccs[i]);
                        js->orients.push_back(job->orients[i]);
                        js->names.push_back(job->names[i]);
                        js->afin.push_back(job->afin[i]);
                    }
                    GpTimes ts2;
                    if (Es->encode(fs, js->outs, ts2, eerr, s, gpuEntropy ? &js->dev : nullptr) || !encodeAlpha(js, 1, ea, s, eerr)) {
                        fprintf(stderr, "encode: %s\n", eerr.c_str()); encFail++; delete js; delete job; return;
                    }
                    {
                        std::lock_guard<std::mutex> lk2(aggMu);
                        A.analysis += ts2.analysis; A.recon += ts2.recon; A.gpuTotal += ts2.total; A.blocks += ts2.blocks;
                        A.peakDev = std::max(A.peakDev, ts2.deviceBytes);
                        A.images += fs.size();
                        for (auto &f : fs) A.pixels += (uint64_t)f.W * f.H;
                    }
                    push(js);
                    std::vector<GpFrameIn> fl, al;
                    std::vector<std::vector<uint8_t>> il;
                    std::vector<std::string> nl;
                    std::vector<int> ol;
                    for (size_t i = 0; i < fin.size(); i++)
                        if (!isSmall[i]) {
                            fl.push_back(fin[i]); il.push_back(job->iccs[i]); ol.push_back(job->orients[i]); nl.push_back(job->names[i]);
                            al.push_back(job->afin[i]);
                        }
                    fin.swap(fl);
                    job->iccs.swap(il);
                    job->orients.swap(ol);
                    job->names.swap(nl);
                    job->afin.swap(al);
                }
            }
            if (fin.empty()) {   // every image of this batch was small: nothing left for the main encoder
                for (auto &b : held) ingest_release_batch(ictx, b.batch_id, s);
                for (uint8_t *d : fbHeld) cudaFreeAsync(d, s);
                gpuFallbackA += (int)fbHeld.size() - fbConv;
                delete job;
                continue;
            }
            if (E->encode(fin, job->outs, tm, eerr, s, gpuEntropy ? &job->dev : nullptr) || !encodeAlpha(job, 0, ea, s, eerr)) {
                fprintf(stderr, "encode: %s\n", eerr.c_str()); encFail++; delete job; return;
            }
            const double te = now_ms() - ts;
            for (auto &b : held) ingest_release_batch(ictx, b.batch_id, s);
            for (uint8_t *d : fbHeld) cudaFreeAsync(d, s);
            gpuFallbackA += (int)fbHeld.size() - fbConv;
            {
                std::lock_guard<std::mutex> lk2(aggMu);
                A.encodeWall += te;
                A.gpuSetup += tm.setup; A.analysis += tm.analysis; A.dp += tm.dp; A.recon += tm.recon; A.pack += tm.pack; A.postfilter += tm.postfilter;
                A.download += tm.download; A.gpuTotal += tm.total; A.blocks += tm.blocks;
                A.peakDev = std::max(A.peakDev, tm.deviceBytes);
                for (int k = 0; k < 16; k++) A.prof[k] += tm.prof[k];
                A.images += fin.size();
                A.batches++;
                for (auto &f : fin) A.pixels += (uint64_t)f.W * f.H;
            }
            push(job);
        }
    };
    std::vector<GpEncoder *> encs;
    std::vector<cudaStream_t> streams;
    std::vector<std::thread> encThreads;
    for (int k = 0; k < nEnc; k++) {
        GpEncoder *E = k == 0 ? &enc : new GpEncoder;
        if (k > 0 && E->init(bcfg, err)) { fprintf(stderr, "init: %s\n", err.c_str()); ingestDrained = true; fbThread.join(); return 1; }
        cudaStream_t s = es;
        if (k > 0) cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        encs.push_back(E);
        streams.push_back(s);
    }
    std::vector<GpEncoder *> encsSmall(nEnc, nullptr);
    if (bcfg.smallMp > 0) {
        GpConfig cs = bcfg;
        cs.P = bcfg.Psmall;
        for (int k = 0; k < nEnc; k++) {
            encsSmall[k] = new GpEncoder;
            if (encsSmall[k]->init(cs, err)) { fprintf(stderr, "init: %s\n", err.c_str()); ingestDrained = true; fbThread.join(); return 1; }
        }
    }
    for (int k = 0; k < nEnc; k++) encThreads.emplace_back(encLoop, encs[k], encsSmall[k], streams[k]);
    for (auto &t : encThreads) t.join();
    ingestDrained = true;
    fbThread.join();
    for (auto &F : fbReady) cudaFree(F.d);   // only left over after an encoder failure
    if (encFail) return 1;
    {
        std::unique_lock<std::mutex> lk(mu);
        done = true;
    }
    cv.notify_all();
    for (auto &t : entThreads) t.join();
    if (entFail) return 1;
    const int cpuFallback = cpuFallbackA.load(), gpuFallback = gpuFallbackA.load();
    const double wall = now_ms() - t0, cpu = cpu_ms() - c0;
    ingest_stats is;
    ingest_get_stats(ictx, &is);
    ingest_destroy(ictx);
    const double mp = A.pixels / 1e6;
    fprintf(stderr,
            "[av1gpu] %llu images (%.1f MP) in %.1f ms: %.2f img/s, %.2f MP/s; %llu batches; out %.2f MB (%.3f bpp); "
            "CPU %.0f ms = %.2f cores; skipped (unsupported): %d; WebP/PNG via CPU decode: %d; write failures %d\n",
            (unsigned long long)A.images, mp, wall, A.images * 1e3 / wall, mp * 1e3 / wall, (unsigned long long)A.batches,
            A.bytes / 1e6, A.bytes * 8.0 / A.pixels, cpu, cpu / wall, cpuFallback, gpuFallback, failures);
    fprintf(stderr,
            "[av1gpu] stage ms: ingest wait %.1f | gpu setup %.1f analysis %.1f dp %.1f recon %.1f pack %.1f postfilter %.1f download %.1f "
            "(gpu total %.1f, encode wall %.1f) | entropy wall %.1f (gpu %.1f) | write %.1f | peak dev %.0f MB\n",
            A.ingestWait, A.gpuSetup, A.analysis, A.dp, A.recon, A.pack, A.postfilter, A.download, A.gpuTotal, A.encodeWall, A.entropyWall,
            A.entropyGpu, A.writeWall, A.peakDev / 1048576.0);
    if (A.alphaImages)
        fprintf(stderr, "[av1gpu] alpha: %llu images with an alpha item (%.1f MP of alpha frames, %s)\n", (unsigned long long)A.alphaImages,
                A.alphaPixels / 1e6, acfg[0].lossless ? "lossless" : ("qindex " + std::to_string(acfg[0].qindex)).c_str());
    if (cfg.P.profile) {
        // k_recon CTA-cycles per phase vs the capacity of the recon wall time (SM clock: GP_CLOCK_GHZ)
        unsigned long long tot = 0;
        for (int k = 0; k < 9; k++) tot += A.prof[k];
        const double clk = getenv("GP_CLOCK_GHZ") ? atof(getenv("GP_CLOCK_GHZ")) : 2.7;
        const int rcCtas = getenv("GP_RC_CTAS") ? atoi(getenv("GP_RC_CTAS")) : 4;
        static const char *ph[9] = {"luma setup", "luma pass1", "luma pass2", "luma final", "chroma edges+CfL", "chroma SATD",
                                    "chroma RD", "chroma final", "maps+record"};
        fprintf(stderr, "[av1gpu] recon phases:");
        for (int k = 0; k < 9; k++) fprintf(stderr, " %s %.1f%%", ph[k], tot ? 100.0 * A.prof[k] / tot : 0.0);
        fprintf(stderr, "\n");
        int sms = 0;
        cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
        fprintf(stderr, "[av1gpu] recon CTA-cycles %.3g = %.1f%% of %d SMs x %d CTAs x %.1f GHz x recon %.0f ms\n", (double)tot,
                100.0 * tot / ((double)sms * rcCtas * clk * 1e6 * A.recon), sms, rcCtas, clk, A.recon);
    }
    if (stats)
        printf("{\"images\":%llu,\"mp\":%.3f,\"wall_ms\":%.1f,\"img_s\":%.3f,\"mp_s\":%.3f,\"cpu_cores\":%.3f,\"bytes\":%llu,"
               "\"ingest_wait_ms\":%.1f,\"analysis_ms\":%.1f,\"dp_ms\":%.1f,\"recon_ms\":%.1f,\"pack_ms\":%.1f,\"postfilter_ms\":%.1f,\"download_ms\":%.1f,"
               "\"gpu_total_ms\":%.1f,\"encode_wall_ms\":%.1f,\"entropy_wall_ms\":%.1f,\"entropy_gpu_ms\":%.1f,\"write_ms\":%.1f,"
               "\"peak_dev_mb\":%.1f,\"fallback_cpu\":%d}\n",
               (unsigned long long)A.images, mp, wall, A.images * 1e3 / wall, mp * 1e3 / wall, cpu / wall,
               (unsigned long long)A.bytes, A.ingestWait, A.analysis, A.dp, A.recon, A.pack, A.postfilter, A.download, A.gpuTotal,
               A.encodeWall, A.entropyWall, A.entropyGpu, A.writeWall, A.peakDev / 1048576.0, cpuFallback);
    return failures ? 1 : 0;
}
