// Batch mode = many images in flight.  The ingest module delivers GPU-resident 4:2:0 (--yuv 444: 4:4:4) planes in
// size-classed batches (NVJPG hardware decode, CPU workers only parse/read); each ingest batch is
// encoded as one GPU batch (analysis -> DP -> wavefront closed loop -> pack), then entropy coded
// (GPU: av1core GPU entropy, or CPU writer threads) and written.  Reports img/s, MP/s, per-stage times, CPU use.
#include "pipeline.h"
#include "pinned.h"
#include "imageio.h"
#include "ingest_api.h"
#include "av1g/av1g.h"
size_t gp_recon_local_bytes();   // k_recon.cu

#include <cuda_runtime.h>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <filesystem>
#else
#include <sched.h>
#include <sys/resource.h>
#include <sys/stat.h>
#include <dirent.h>
#include <unistd.h>
#endif

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <functional>
#include <string>
#include <thread>
#include <vector>

namespace {
double now_ms() {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
double cpu_ms() {
#ifdef _WIN32
    FILETIME c, e, k, u;   // 100 ns units
    if (!GetProcessTimes(GetCurrentProcess(), &c, &e, &k, &u)) return 0;
    auto ms = [](const FILETIME &t) { return (double)(((unsigned long long)t.dwHighDateTime << 32) | t.dwLowDateTime) / 1e4; };
    return ms(k) + ms(u);
#else
    rusage r;
    getrusage(RUSAGE_SELF, &r);
    return (r.ru_utime.tv_sec + r.ru_stime.tv_sec) * 1e3 + (r.ru_utime.tv_usec + r.ru_stime.tv_usec) / 1e3;
#endif
}
// cgroup CPU limit in CPUs (docker --cpus, Kubernetes limits), 0 = none.  v2: cpu.max ("max 100000" or
// "200000 100000") of this process's cgroup and its parents; v1: the cpu controller's cfs quota / period.
double cgroup_cpu_limit() {
    double lim = 0;
#ifndef _WIN32
    auto take = [&](double quota, double period) {
        if (quota > 0 && period > 0) lim = lim > 0 ? std::min(lim, quota / period) : quota / period;
    };
    std::string v2;
    if (std::ifstream cg{"/proc/self/cgroup"}) {
        for (std::string l; std::getline(cg, l);)
            if (l.compare(0, 3, "0::") == 0) v2 = l.substr(3);
    }
    if (!v2.empty()) {
        for (std::string d = v2;;) {
            if (std::ifstream f{"/sys/fs/cgroup" + (d == "/" ? std::string() : d) + "/cpu.max"}) {
                std::string q;
                double period = 0;
                if (f >> q >> period && q != "max") take(atof(q.c_str()), period);
            }
            if (d.empty() || d == "/") break;
            const size_t s = d.find_last_of('/');
            d = s == 0 || s == std::string::npos ? "/" : d.substr(0, s);
        }
    }
    for (const char *dir : {"/sys/fs/cgroup/cpu", "/sys/fs/cgroup/cpu,cpuacct"}) {
        std::ifstream fq{std::string(dir) + "/cpu.cfs_quota_us"}, fp{std::string(dir) + "/cpu.cfs_period_us"};
        double quota = 0, period = 0;
        if (fq >> quota && fp >> period) { take(quota, period); break; }
    }
#endif
    return lim;
}
// CPUs this process may run on: the affinity mask (taskset, cpusets), lowered to a cgroup CPU limit (rounded up)
int usable_cpus() {
    int n = 0;
#ifdef __linux__
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof set, &set) == 0) n = CPU_COUNT(&set);
#elif defined(_WIN32)
    // a restricted process affinity mask (start /affinity) within the process's processor group; else the CPUs
    // of all groups (both masks are 0 once the process has threads in several groups)
    DWORD_PTR proc = 0, sys = 0;
    if (GetProcessAffinityMask(GetCurrentProcess(), &proc, &sys) && proc && proc != sys)
        for (; proc; proc &= proc - 1) n++;
    else
        n = (int)GetActiveProcessorCount(ALL_PROCESSOR_GROUPS);
#endif
    if (n <= 0) n = (int)std::thread::hardware_concurrency();
    if (n <= 0) n = 4;
    const double lim = cgroup_cpu_limit();
    if (lim > 0) n = std::min(n, std::max(1, (int)std::ceil(lim - 1e-9)));
    return n;
}
int env_int(const char *name, int dflt) {
    const char *v = getenv(name);
    return v ? atoi(v) : dflt;
}
std::string base_name(const std::string &p) {
#ifdef _WIN32
    const size_t s = p.find_last_of("/\\:");
#else
    const size_t s = p.find_last_of('/');
#endif
    return s == std::string::npos ? p : p.substr(s + 1);
}
bool has_image_ext(const std::string &n) {
    const size_t d = n.find_last_of('.');
    if (d == std::string::npos) return false;
    std::string e = n.substr(d + 1);
    for (auto &c : e) c = (char)tolower((unsigned char)c);
    return e == "jpg" || e == "jpeg" || e == "jfif" || e == "png" || e == "webp";
}
// mkdir -p: DIR and any missing parents; true if DIR is a directory afterwards
bool make_dirs(const std::string &dir) {
#ifdef _WIN32
    std::error_code ec;
    const std::filesystem::path d = std::filesystem::u8path(dir);
    std::filesystem::create_directories(d, ec);
    if (ec) errno = ec.default_error_condition().value();   // Win32 error -> errno value (strerror)
    return std::filesystem::is_directory(d, ec);
#else
    for (size_t i = 1; i <= dir.size(); i++)
        if (i == dir.size() || dir[i] == '/')
            if (mkdir(dir.substr(0, i).c_str(), 0755) != 0 && errno != EEXIST) return false;
    struct stat st;
    return stat(dir.c_str(), &st) == 0 && S_ISDIR(st.st_mode);
#endif
}
// LIST is a text file with one path per line, or a directory (its JPEG/PNG/WebP files, sorted).  A line may add
// TAB-separated options for its image: crop=WxH+X+Y (or X,Y,W,H), max-size=WxH (xH, Wx) and resize=GEOMETRY; they
// replace the command line's --crop / --max-size / --resize for that image.  A line that names an existing file as a
// whole is a path.  A line with a bad option is not encoded: bad gets its message (the rest of the list is, as with
// unreadable files).
bool list_inputs(const std::string &list, std::vector<std::string> &paths, std::vector<GpXform> &xf, const GpXform &def,
                 std::vector<std::string> &bad) {
    auto exists = [](const std::string &p) {
#ifdef _WIN32
        std::error_code ec;
        return std::filesystem::exists(std::filesystem::u8path(p), ec);
#else
        struct stat st;
        return stat(p.c_str(), &st) == 0;
#endif
    };
    auto add = [&](std::string l, int line) {
        GpXform x = def;
        const size_t tab = l.find('\t');
        if (tab != std::string::npos && !exists(l)) {
            for (size_t at = tab; at != std::string::npos;) {
                const size_t nx = l.find('\t', at + 1);
                std::string o = l.substr(at + 1, nx == std::string::npos ? std::string::npos : nx - at - 1), why;
                if (!o.empty() && o.back() == '\r') o.pop_back();   // CRLF lists
                GpXform t;
                if (o.empty()) {
                } else if (!o.compare(0, 5, "crop=") && gp_parse_crop(o.c_str() + 5, t, &why)) {
                    x.cx = t.cx; x.cy = t.cy; x.cw = t.cw; x.ch = t.ch;
                } else if (!o.compare(0, 9, "max-size=") && gp_parse_max_size(o.c_str() + 9, t, &why)) {
                    x.maxW = t.maxW; x.maxH = t.maxH; x.rsFlags = 0;
                } else if (!o.compare(0, 7, "resize=") && gp_parse_resize(o.c_str() + 7, t, &why)) {
                    x.rsFlags = t.rsFlags; x.rsW = t.rsW; x.rsH = t.rsH; x.maxW = x.maxH = 0;
                } else {
                    bad.push_back(l.substr(0, tab) + " (" + list + " line " + std::to_string(line) + "): bad option '" + o + "'" +
                                  (why.empty() ? " (crop=WxH+X+Y or X,Y,W,H, max-size=WxH, Wx or xH, resize=GEOMETRY)" : ": " + why));
                    return;
                }
                at = nx;
            }
            l.resize(tab);
        }
        paths.push_back(l);
        xf.push_back(x);
    };
#ifdef _WIN32
    namespace fs = std::filesystem;
    std::error_code ec;
    const fs::path lp = fs::u8path(list);
    if (!fs::exists(lp, ec)) return false;
    if (fs::is_directory(lp, ec)) {
        for (fs::directory_iterator it(lp, ec), end; !ec && it != end; it.increment(ec)) {
            const std::string n = it->path().filename().u8string();
            if (!n.empty() && n[0] != '.' && has_image_ext(n)) paths.push_back(list + "/" + n);
        }
        if (ec) return false;
        std::sort(paths.begin(), paths.end());
        xf.assign(paths.size(), def);
        return true;
    }
    std::ifstream f(lp);
    std::string l;
    int line = 0;
    for (bool first = true; std::getline(f, l); first = false) {
        line++;
        if (first && l.compare(0, 3, "\xEF\xBB\xBF") == 0) l.erase(0, 3);   // UTF-8 byte order mark (Notepad)
        if (!l.empty() && l.back() == '\r') l.pop_back();
        if (!l.empty()) add(l, line);
    }
    return true;
#else
    struct stat st;
    if (stat(list.c_str(), &st) != 0) return false;
    if (S_ISDIR(st.st_mode)) {
        DIR *dp = opendir(list.c_str());
        if (!dp) return false;
        while (dirent *e = readdir(dp))
            if (e->d_name[0] != '.' && has_image_ext(e->d_name)) paths.push_back(list + "/" + e->d_name);
        closedir(dp);
        std::sort(paths.begin(), paths.end());
        xf.assign(paths.size(), def);
        return true;
    }
    std::ifstream f(list);
    std::string l;
    int line = 0;
    while (std::getline(f, l)) {
        line++;
        if (!l.empty()) add(l, line);
    }
    return true;
#endif
}
struct Agg {
    double ingestWait = 0, gpuSetup = 0, analysis = 0, dp = 0, recon = 0, pack = 0, postfilter = 0, download = 0, gpuTotal = 0;
    double entropyGpu = 0, entropyWall = 0, writeWall = 0, encodeWall = 0;
    uint64_t images = 0, pixels = 0, bytes = 0, blocks = 0, batches = 0;
    uint64_t dpRepaired = 0;   // GpTimes::dpRepaired of every encode (colour and alpha)
    uint64_t alphaImages = 0, alphaPixels = 0;   // images with an alpha item (their alpha frames)
    unsigned long long prof[16] = {0};
    size_t peakDev = 0;
};
}  // namespace

// the number of inputs a --batch LIST|DIR names (-1: unreadable); av1gpu.cpp counts them before CUDA starts
long gp_batch_input_count(const std::string &list) {
    std::vector<std::string> paths, bad;
    std::vector<GpXform> xf;
    return list_inputs(list, paths, xf, GpXform(), bad) ? (long)paths.size() : -1;
}

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

// --lossless refuses JPEGs (skipped in batch mode, status 3 for one image): their lossless AVIF (the decoded YCbCr planes)
// was a median 2.2x the JPEG's size on the size-range photos, up to 6.2x
extern const char *const gp_lossless_jpeg_refusal =
    "--lossless on a JPEG makes a file about twice the size of the JPEG, or more; drop --lossless to encode it lossy";

// The hint printed with a GPU out-of-memory error under --lossless (empty otherwise).  Lossless frames code about every
// sample, so they need several times the GPU memory of lossy ones: on an L4 (23,034 MiB), batch mode on 5-24 MP PNGs
// peaked at 14,553 MiB with GP_ENCODE_MP=12, 17,305 MiB with 16 and 21,945 MiB with 24, and ran out with the default 48.
// batchMp: the batch mode's GP_ENCODE_MP (0: one image).
std::string gp_lossless_oom_hint(const GpConfig &cfg, const std::string &err, double batchMp) {
    if (!cfg.lossless || err.find("out of memory") == std::string::npos) return "";
    size_t fr = 0, tot = 0;
    cudaMemGetInfo(&fr, &tot);
    char b[400];
    if (batchMp > 0)
        snprintf(b, sizeof b, "[av1gpu] out of GPU memory: lossless batches need about 1 GB of GPU memory per megapixel of "
                 "GP_ENCODE_MP (now %.0f; this GPU has %.1f GB); try a smaller GP_ENCODE_MP, e.g. %d", batchMp, tot / 1e9,
                 std::max(4, (int)(0.7 * tot / 1e9)));
    else
        snprintf(b, sizeof b, "out of GPU memory: lossless frames need several times the GPU memory of lossy ones (this GPU has "
                 "%.1f GB)", tot / 1e9);
    return b;
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
    std::vector<std::string> paths, bad;
    std::vector<GpXform> xforms;   // per input: crop / fit-inside (command line, or the list line's options)
    if (!list_inputs(list, paths, xforms, cfg.xform, bad)) { fprintf(stderr, "cannot read %s\n", list.c_str()); return 2; }
    for (auto &b : bad) fprintf(stderr, "[av1gpu] skipped %s\n", b.c_str());
    const bool anyXform = cfg.trim.on || std::any_of(xforms.begin(), xforms.end(), [](const GpXform &x) { return x.active(); });
    // the largest picture encoded (--max-mp) and source decoded (--max-input-mp: sources that a crop / max-size brings
    // under --max-mp)
    const double encMaxMp = gp_enc_max_mp(cfg, 35.6), decMaxMp = gp_dec_max_mp(cfg, 35.6);
    const bool inputLimit = decMaxMp > encMaxMp;   // (--max-input-mp refuses sources, not --max-mp)
    if (paths.empty()) { fprintf(stderr, "no input files in %s\n", list.c_str()); return 2; }
    const bool write = !outdir.empty();
    if (write && !make_dirs(outdir)) { fprintf(stderr, "cannot create %s: %s\n", outdir.c_str(), strerror(errno)); return 2; }
    std::string err;
    GpEncoder enc;
    GpConfig bcfg = cfg;
    bcfg.wantRecon = getenv("GP_WANT_RECON") != nullptr;
    if (enc.init(bcfg, err)) { fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
    if (!cfg.lossless) {
        // local memory (stack) for the closed loop up front.  The driver grows it at the first launch that needs more
        // (k_analysis, k_dp, k_recon: 1.2-3.3 KB per thread) and waits for the GPU to drain meanwhile, with every
        // thread's CUDA calls blocked: 0.17 s at the first k_dp launch on an L4 (nsys)
        size_t cur = 0;
        const size_t need = gp_recon_local_bytes();
        if (cudaDeviceGetLimit(&cur, cudaLimitStackSize) == cudaSuccess && need > cur && cudaDeviceSetLimit(cudaLimitStackSize, need) != cudaSuccess)
            cudaGetLastError();   // (only slower)
    }
    const char *bmp = getenv("GP_BATCH_MP");
    // CPU threads scale with the CPUs this process may use (affinity mask, cgroup limit): JPEG parse / CPU entropy
    // decode, PNG / WebP decode and --cpu-entropy writers each get one thread per CPU.  The GPU-side counts (encoders,
    // concurrent GPU entropy batches) stay fixed.
    const int nCpu = usable_cpus();
    const int nEntThreads = std::max(1, env_int("GP_ENTROPY_THREADS", gpuEntropy ? 3 : nCpu));  // GPU: concurrent entropy batches; CPU: writer threads
    ingest_config ic;
    ingest_default_config(&ic);
    ic.n_workers = std::max(1, env_int("GP_INGEST_WORKERS", nCpu));
    ic.activity = 0;
    ic.max_inflight = getenv("GP_INGEST_INFLIGHT") ? atoi(getenv("GP_INGEST_INFLIGHT")) : 24;
    ic.batch_pixels = (int64_t)((bmp ? atof(bmp) : 16.0) * 1e6);
    ic.pinned_bytes = gp_ingest_pinned_bytes(decMaxMp);   // 256 MiB up to 35.6 MP sources
    if (getenv("GP_INGEST_DECODER")) ic.decoder = atoi(getenv("GP_INGEST_DECODER"));   // 1 = CPU entropy decode + GPU IDCT (no nvJPEG/NVJPG)
    if (getenv("GP_INGEST_HWBATCH")) ic.hw_batch = atoi(getenv("GP_INGEST_HWBATCH"));
    ic.out_444 = cfg.lossless ? 2 : cfg.yuv444;   // --yuv 444: full-resolution chroma planes; --lossless: the JPEG's own format
    ic.max_mp = decMaxMp;
    // image i of W x H with EXIF orientation orient: its crop / box (an orientation that swaps the sides turns the box:
    // the box is what is displayed) and output size; "" or why it is skipped
    auto prepare = [&](int i, int W, int H, int orient, GpXform &x, int *ow, int *oh) -> std::string {
        x = xforms[i];
        if (orient >= 5) x.turn();
        *ow = W;
        *oh = H;
        if (x.active() && !gp_xform_size(x, W, H, ow, oh))
            return "the crop " + gp_crop_str(x) + " is outside the " + std::to_string(W) + "x" + std::to_string(H) + " image";
        if (cfg.trim.on) return "";   // (the size that counts is known after the trim: the encoder threads check it)
        return gp_size_refusal(*ow, *oh, encMaxMp, gp_xform_runs(x, W, H, *ow, *oh) ? " after crop / resize" : "");
    };
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
    // --lossless: the first out-of-GPU-memory error also prints how to avoid it (once; failures are handled as before)
    std::atomic<bool> oomHinted{false};
    auto oomHint = [&](const std::string &e) {
        const std::string h = gp_lossless_oom_hint(cfg, e, targetMp);
        if (!h.empty() && !oomHinted.exchange(true)) fprintf(stderr, "%s\n", h.c_str());
    };
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
                    if (gp_entropy_dev(j->dev, ts, escr, tb, &gms, terr)) { fprintf(stderr, "entropy: %s\n", terr.c_str()); oomHint(terr); entFail++; }
                    gp_dev_records_free(j->dev, ts);
                    std::vector<std::vector<std::vector<uint8_t>>> atb;   // alpha frames of the job (one more entropy batch)
                    if (j->adev.pool) {
                        double gms2 = 0;
                        if (gp_entropy_dev(j->adev, ts, escr, atb, &gms2, terr)) {
                            fprintf(stderr, "entropy (alpha): %s\n", terr.c_str());
                            oomHint(terr);
                            entFail++;
                        }
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
    const int nEnc = std::max(1, env_int("GP_ENCODERS", 3));
    // ingest batches one encoder may hold; at least 1, or with GP_ENCODERS >= GP_INGEST_INFLIGHT no encoder took any
    // and the run never finished
    const int heldMax = std::max(1, ic.max_inflight / (nEnc + 1));
    std::mutex inMu;
    bool more = true;
    std::atomic<int> encFail{0};
    // WebP/PNG named .jpg (ingest fallbacks): worker threads decode them on the CPU as soon as ingest reports
    // them, upload the planes and queue them; the encoder threads take them into their next batch like
    // ingest images (GPU entropy, overlapped).  The rest (giant, CMYK, ...) -> CPU avifenc (not done here).
    // (Before: serial CPU decode + single-image encode + host-path entropy after the main loop, ~0.45 s per
    // image with the GPU mostly idle.)
    struct FbFrame { GpFrameIn f, a; uint8_t *d = nullptr; std::string name, path; GpXform x; };   // a: alpha frame (W = 0: none)
    std::mutex fbMu;
    std::condition_variable fbCv;   // an encoder took frames out of fbReady, or an encoder thread ended
    std::deque<FbFrame> fbReady;
    int64_t fbQueuedPx = 0;         // luma pixels of the frames in fbReady (their planes are on the GPU)
    std::atomic<bool> ingestDrained{false}, fbDone{false};
    std::atomic<int> cpuFallbackA{(int)bad.size()}, gpuFallbackA{0};   // skipped (list lines with a bad option too), via CPU decode
    std::atomic<int> trimCutA{0}, trimBlankA{0};   // --trim: images encoded trimmed, images skipped as blank
    std::atomic<int> encLive{0};    // encoder threads running; 0 with ingestDrained = no encoder will take frames any more
    // GP_FALLBACK_WORKERS (default: one per CPU) threads decode + upload fallback images as soon as ingest reports them.
    // Decoded frames wait on the GPU for an encoder: at most GP_FALLBACK_QUEUE_MP megapixels of them (default
    // 2 x GP_ENCODE_MP, plus one image), so CPUs that decode faster than the GPU encodes don't fill its memory.
    const int nfw = std::max(1, env_int("GP_FALLBACK_WORKERS", nCpu));
    std::thread fbThread([&]() {
        const int64_t queuePx = (int64_t)((getenv("GP_FALLBACK_QUEUE_MP") ? atof(getenv("GP_FALLBACK_QUEUE_MP")) : 2 * targetMp) * 1e6);
        std::mutex lmu;
        std::condition_variable lcv;
        std::vector<ingest_fallback> list;   // fallbacks reported so far; list[next..] not yet taken by a worker
        size_t next = 0;
        bool listDone = false;
        auto one = [&](const ingest_fallback &fe, cudaStream_t &fs) {
            if (ingestDrained && encLive == 0) return;   // the encoders stopped (error)
            const std::string &p = paths[fe.index];
            const GpFileKind kind = gp_sniff(p);
            if (kind != GP_FILE_WEBP && kind != GP_FILE_PNG) {
                fprintf(stderr, "[av1gpu] skipped %s: %s\n", p.c_str(),
                        fe.reason == INGEST_FB_GIANT ? gp_source_refusal(fe.width, fe.height, decMaxMp, &xforms[fe.index], inputLimit).c_str()
                                                     : gp_fallback_reason(fe.reason));
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
            GpXform fx;
            int ow, oh;
            if (dec && (double)hp.W * hp.H > decMaxMp * 1e6) e2 = gp_source_refusal(hp.W, hp.H, decMaxMp, &xforms[fe.index], inputLimit);
            else if (dec) e2 = prepare(fe.index, hp.W, hp.H, 1, fx, &ow, &oh);
            if (!dec || !e2.empty()) {
                fprintf(stderr, "[av1gpu] skipped %s: %s\n", p.c_str(), e2.c_str());
                cpuFallbackA++;
                return;
            }
            const int64_t px = (int64_t)hp.W * hp.H;
            {
                std::unique_lock<std::mutex> lk(fbMu);
                while (fbQueuedPx > 0 && fbQueuedPx + px > queuePx && !(ingestDrained && encLive == 0))
                    fbCv.wait_for(lk, std::chrono::milliseconds(20));
                if (ingestDrained && encLive == 0) return;   // the encoders stopped (error): nobody would take the frame
                fbQueuedPx += px;   // reserved before the upload, so the workers together stay within the limit
            }
            auto unreserve = [&]() {
                std::lock_guard<std::mutex> lk(fbMu);
                fbQueuedPx -= px;
            };
            const size_t ys = hp.bytes(0), cs = hp.bytes(1), as = hp.abytes();   // bytes (10-bit: uint16_t samples)
            if (!fs) cudaStreamCreateWithFlags(&fs, cudaStreamNonBlocking);
            uint8_t *d = nullptr;
            if (cudaMallocAsync((void **)&d, ys + 2 * cs + as, fs) != cudaSuccess) {
                cudaGetLastError();
                fprintf(stderr, "[av1gpu] skipped %s: out of GPU memory for its planes\n", p.c_str());
                oomHint("out of memory");
                unreserve();
                cpuFallbackA++;
                return;
            }
            cudaMemcpyAsync(d, hp.data(0), ys, cudaMemcpyHostToDevice, fs);
            cudaMemcpyAsync(d + ys, hp.data(1), cs, cudaMemcpyHostToDevice, fs);
            cudaMemcpyAsync(d + ys + cs, hp.data(2), cs, cudaMemcpyHostToDevice, fs);
            if (as) cudaMemcpyAsync(d + ys + 2 * cs, hp.adata(), as, cudaMemcpyHostToDevice, fs);
            if (const cudaError_t se = cudaStreamSynchronize(fs)) {
                cudaGetLastError();
                fprintf(stderr, "[av1gpu] skipped %s: upload failed (%s)\n", p.c_str(), cudaGetErrorString(se));
                unreserve();
                cpuFallbackA++;
                return;
            }
            FbFrame F;
            F.d = d;
            F.name = base_name(p);
            F.path = p;
            F.x = fx;
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
        // persistent workers (each with its own stream): one image at a time, the next as soon as one is done.  The
        // stream is created with the worker's first image: JPEG-only lists then create and destroy none (16 workers'
        // cudaStreamDestroy calls at the end of ingest held other threads' launches for 57 ms, nsys on an L4)
        std::vector<std::thread> th;
        for (int w = 0; w < nfw; w++)
            th.emplace_back([&]() {
                cudaStream_t fs = nullptr;
                for (;;) {
                    ingest_fallback fe;
                    {
                        std::unique_lock<std::mutex> lk(lmu);
                        lcv.wait(lk, [&] { return next < list.size() || listDone; });
                        if (next >= list.size()) break;
                        fe = list[next++];
                    }
                    one(fe, fs);
                }
                if (fs) cudaStreamDestroy(fs);
            });
        std::vector<ingest_fallback> fb(4096);
        for (;;) {
            const bool last = ingestDrained.load();   // all fallbacks are recorded once ingest is drained
            int nfb = ingest_get_fallbacks(ictx, fb.data(), (int)fb.size());
            if (nfb > (int)fb.size()) {   // long lists: grow instead of dropping fallbacks beyond 4096 (old code did)
                fb.resize((size_t)nfb * 2);
                nfb = std::min(ingest_get_fallbacks(ictx, fb.data(), (int)fb.size()), (int)fb.size());
            }
            {
                std::lock_guard<std::mutex> lk(lmu);
                for (size_t k = list.size(); k < (size_t)nfb; k++) list.push_back(fb[k]);
                if (last) listDone = true;
            }
            lcv.notify_all();
            if (last) break;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
        }
        for (auto &t : th) t.join();
        fbDone = true;
    });
    // alpha frames of a job: one more GPU batch (monochrome frames, gp_alpha_config: lossless or lossy at the alpha
    // quality), on the job's stream after its colour frames; each encoder thread creates its alpha encoder on first use.
    // Alpha frames always use the main preset, also for small images (--small-preset is for their colour): on the
    // RGBA test set the base preset gave 3-4 % smaller lossy alpha payloads (0.4 % of the files) at 0.1 dB higher alpha
    // PSNR, but 35 % lower throughput on a set of mostly small images
    const GpConfig acfg = gp_alpha_config(bcfg);
    struct AlphaEnc {
        GpEncoder *e = nullptr;
        ~AlphaEnc() { delete e; }
    };
    // k_dp's safety net acted (GpTimes::dpRepaired; 0 unless the partition check missed a case): the inputs it coded blocks of
    auto warnNet = [](const Job *j, bool alpha) {
        const std::vector<GpFrameOut> &o = alpha ? j->aouts : j->outs;
        for (size_t i = 0; i < j->names.size(); i++) {
            const int k = !alpha ? (int)i : i < j->alphaOf.size() ? j->alphaOf[i] : -1;
            if (k >= 0 && (size_t)k < o.size() && o[k].dpNet) gp_warn_dp_net((j->names[i] + (alpha ? " (alpha)" : "")).c_str(), 0);
        }
    };
    auto encodeAlpha = [&](Job *j, AlphaEnc &ea, cudaStream_t s, std::string &eerr) -> bool {
        std::vector<GpFrameIn> al;
        j->alphaOf.assign(j->afin.size(), -1);
        for (size_t i = 0; i < j->afin.size(); i++)
            if (j->afin[i].W) { j->alphaOf[i] = (int)al.size(); al.push_back(j->afin[i]); }
        if (al.empty()) return true;
        if (!ea.e) {
            ea.e = new GpEncoder;
            if (ea.e->init(acfg, eerr)) return false;
        }
        GpTimes ta;
        if (ea.e->encode(al, j->aouts, ta, eerr, s, gpuEntropy ? &j->adev : nullptr)) return false;
        if (ta.dpRepaired) warnNet(j, true);
        std::lock_guard<std::mutex> lk2(aggMu);
        A.analysis += ta.analysis; A.recon += ta.recon; A.gpuTotal += ta.total; A.blocks += ta.blocks; A.dpRepaired += ta.dpRepaired;
        A.peakDev = std::max(A.peakDev, ta.deviceBytes);
        A.alphaImages += al.size();
        for (auto &f : al) A.alphaPixels += (uint64_t)f.W * f.H;
        return true;
    };
    auto encLoop = [&](GpEncoder *E, GpEncoder *Es, cudaStream_t s) {
        std::string eerr;
        AlphaEnc ea;
        for (;;) {
            std::vector<ingest_batch> held;
            std::vector<GpFrameIn> fin;
            Job *job = new Job;
            double mpAcc = 0;   // megapixels encoded (after crop / resize)
            std::vector<uint8_t *> fbHeld;
            std::vector<GpXform> xfin;   // per fin entry
            std::vector<std::string> srcs;   // per fin entry: its input path (messages)
            int fbConv = 0;   // how many of fbHeld are conversion buffers (10-bit, crop / resize), not fallback images
            // ingest batches up to timeoutMs (-1: wait for one), until the batch has GP_ENCODE_MP
            auto takeIngest = [&](int timeoutMs) {
                std::lock_guard<std::mutex> lk(inMu);
                while (more && mpAcc < targetMp && (int)fin.size() < maxImgs && (int)held.size() < heldMax) {
                    const double tw = now_ms();
                    ingest_batch b;
                    const int r = ingest_next_batch(ictx, &b, timeoutMs);
                    {
                        std::lock_guard<std::mutex> lk2(aggMu);
                        A.ingestWait += now_ms() - tw;
                    }
                    if (r == -1 && timeoutMs >= 0) break;   // none ready yet
                    if (r == 0) { more = false; ingestDrained = true; break; }
                    if (r < 0) { char e2[256]; ingest_last_error(ictx, e2, sizeof e2); fprintf(stderr, "ingest: %s\n", e2); more = false; ingestDrained = true; encFail++; break; }
                    held.push_back(b);
                    for (int i = 0; i < b.n_images; i++) {
                        const ingest_image &im = b.images[i];
                        if (cfg.lossless) {   // (ingest delivers JPEGs only)
                            fprintf(stderr, "[av1gpu] skipped %s: %s\n", paths[im.index].c_str(), gp_lossless_jpeg_refusal);
                            cpuFallbackA++;
                            continue;
                        }
                        GpXform x;
                        int ow, oh;
                        const std::string why = prepare(im.index, im.width, im.height, im.oriented ? 1 : im.orientation, x, &ow, &oh);
                        if (!why.empty()) {
                            fprintf(stderr, "[av1gpu] skipped %s: %s\n", paths[im.index].c_str(), why.c_str());
                            cpuFallbackA++;
                            continue;
                        }
                        xfin.push_back(x);
                        GpFrameIn f;
                        f.W = im.width; f.H = im.height; f.mono = im.mono; f.aw = im.aw; f.ah = im.ah; f.yuv444 = im.chroma_full;
                        f.jpegChroma = !im.mono && !im.chroma_full && im.src_sub == INGEST_SUB_420;
                        f.dsrc[0] = b.pool + im.y_off; f.spitch[0] = im.y_pitch;
                        if (!im.mono) { f.dsrc[1] = b.pool + im.cb_off; f.dsrc[2] = b.pool + im.cr_off; f.spitch[1] = f.spitch[2] = im.c_pitch; }
                        fin.push_back(f);
                        job->iccs.emplace_back(im.icc_size > 0 ? std::vector<uint8_t>(im.icc, im.icc + im.icc_size) : std::vector<uint8_t>());
                        job->orients.push_back(im.oriented ? 1 : im.orientation);
                        job->names.push_back(base_name(paths[im.index]));
                        srcs.push_back(paths[im.index]);
                        job->afin.emplace_back();   // (JPEG: no alpha)
                        mpAcc += ow * (double)oh / 1e6;
                    }
                }
            };
            // JPEG batches that are ready first, then decoded PNG / WebP frames, then wait for JPEG batches.  With fast PNG
            // decoding, taking the PNG frames first ran a mixed list's PNGs early and left its slow small-JPEG batches (base
            // preset) for the end, with fewer encoders busy (L4: 2-5 % slower).  The fallback queue is bounded, so letting
            // it wait costs no extra GPU memory.
            takeIngest(0);
            {
                std::lock_guard<std::mutex> lk(fbMu);
                while (!fbReady.empty() && mpAcc < targetMp && (int)fin.size() < maxImgs) {
                    const FbFrame F = fbReady.front();
                    fbReady.pop_front();
                    fbQueuedPx -= (int64_t)F.f.W * F.f.H;
                    fin.push_back(F.f);
                    job->iccs.emplace_back();
                    job->orients.push_back(1);
                    job->names.push_back(F.name);
                    srcs.push_back(F.path);
                    job->afin.push_back(F.a);
                    fbHeld.push_back(F.d);
                    xfin.push_back(F.x);
                    int ow = F.f.W, oh = F.f.H;
                    if (F.x.active()) gp_xform_size(F.x, F.f.W, F.f.H, &ow, &oh);
                    mpAcc += ow * (double)oh / 1e6;
                }
            }
            if (!fbHeld.empty()) fbCv.notify_all();   // room in the fallback queue
            takeIngest(-1);
            if (!fin.empty() && cfg.trim.on) {
                // --trim: the box of each frame (one launch, then a wait for this stream), the kept crop in xfin; blank
                // frames (--skip-blank) leave the batch here
                std::vector<int> verdict;
                if (gp_frames_trim(fin, xfin, cfg.trim, s, verdict, eerr, &job->afin)) { fprintf(stderr, "trim: %s\n", eerr.c_str()); encFail++; delete job; return; }
                size_t k = 0;
                for (size_t i = 0; i < fin.size(); i++) {
                    if (verdict[i] == GP_TRIM_BLANK) {
                        fprintf(stderr, "[av1gpu] skipped %s: blank (--trim would remove more than %g%% of its width or height, "
                                "--skip-blank)\n", srcs[i].c_str(), cfg.trim.blankPct);
                        trimBlankA++;
                        continue;
                    }
                    int ow = fin[i].W, oh = fin[i].H;
                    gp_xform_size(xfin[i], fin[i].W, fin[i].H, &ow, &oh);
                    const std::string why = gp_size_refusal(ow, oh, encMaxMp, gp_xform_runs(xfin[i], fin[i].W, fin[i].H, ow, oh) ? " after trim / resize" : "");
                    if (!why.empty()) {
                        fprintf(stderr, "[av1gpu] skipped %s: %s\n", srcs[i].c_str(), why.c_str());
                        cpuFallbackA++;
                        continue;
                    }
                    if (verdict[i] == GP_TRIM_CUT) trimCutA++;
                    if (k != i) {
                        fin[k] = fin[i]; xfin[k] = xfin[i]; srcs[k] = srcs[i]; job->iccs[k].swap(job->iccs[i]);
                        job->orients[k] = job->orients[i]; job->names[k].swap(job->names[i]); job->afin[k] = job->afin[i];
                    }
                    k++;
                }
                fin.resize(k); xfin.resize(k); srcs.resize(k); job->iccs.resize(k); job->orients.resize(k); job->names.resize(k);
                job->afin.resize(k);
            }
            if (!fin.empty() && cfg.tone.on() && gp_frames_tone(fin, xfin, cfg.tone, s, eerr)) {   // in place, before the resize
                fprintf(stderr, "tone: %s\n", eerr.c_str()); encFail++; delete job; return;
            }
            if (!fin.empty() && anyXform) {
                // crop / shrink (straight to 10 bits with --depth 10; colour with alpha-weighted taps where it has alpha),
                // then the alpha frames (PNG / WebP transparency) the same way: monochrome, already at the colour's depth.
                // Freed with the fallback planes
                uint8_t *cb = nullptr;
                if (gp_frames_transform(fin, xfin, cfg.depth, s, &cb, eerr, &job->afin)) { fprintf(stderr, "crop / resize: %s\n", eerr.c_str()); oomHint(eerr); encFail++; delete job; return; }
                if (cb) { fbHeld.push_back(cb); fbConv++; }
                if (gp_frames_transform(job->afin, xfin, cfg.depth, s, &cb, eerr)) { fprintf(stderr, "crop / resize: %s\n", eerr.c_str()); oomHint(eerr); encFail++; delete job; return; }
                if (cb) { fbHeld.push_back(cb); fbConv++; }
            }
            if (!fin.empty() && cfg.gray) gp_frames_gray(fin, cfg.gray, s);   // --grayscale: in place, after the resize
            if (!fin.empty() && cfg.depth == 10 && !cfg.lossless) {
                // --depth 10: ingest frames (8-bit JPEG samples) -> 10-bit planes; freed with the fallback planes
                uint8_t *cb = nullptr;
                if (gp_frames_to_10bit(fin, s, &cb, eerr)) { fprintf(stderr, "10-bit conversion: %s\n", eerr.c_str()); encFail++; delete job; return; }
                if (cb) { fbHeld.push_back(cb); fbConv++; }
            }
            if (fin.empty()) {
                // --lossless: ingest batches whose JPEGs were all skipped; holding them filled ingest's in-flight limit
                // and stalled the list (100 JPEGs: hung after about 75 skips)
                for (auto &b : held) ingest_release_batch(ictx, b.batch_id, s);
                for (uint8_t *d : fbHeld) cudaFreeAsync(d, s);   // (--trim: fallback frames that were all blank)
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
                if (E->encode(f10, jd->outs, t10, eerr, s, gpuEntropy ? &jd->dev : nullptr) || !encodeAlpha(jd, ea, s, eerr)) {
                    fprintf(stderr, "encode: %s\n", eerr.c_str()); oomHint(eerr); encFail++; delete jd; delete job; return;
                }
                if (t10.dpRepaired) warnNet(jd, false);
                {
                    std::lock_guard<std::mutex> lk2(aggMu);
                    A.analysis += t10.analysis; A.recon += t10.recon; A.gpuTotal += t10.total; A.blocks += t10.blocks; A.dpRepaired += t10.dpRepaired;
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
                    if (Es->encode(fs, js->outs, ts2, eerr, s, gpuEntropy ? &js->dev : nullptr) || !encodeAlpha(js, ea, s, eerr)) {
                        fprintf(stderr, "encode: %s\n", eerr.c_str()); oomHint(eerr); encFail++; delete js; delete job; return;
                    }
                    if (ts2.dpRepaired) warnNet(js, false);
                    {
                        std::lock_guard<std::mutex> lk2(aggMu);
                        A.analysis += ts2.analysis; A.recon += ts2.recon; A.gpuTotal += ts2.total; A.blocks += ts2.blocks; A.dpRepaired += ts2.dpRepaired;
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
            if (E->encode(fin, job->outs, tm, eerr, s, gpuEntropy ? &job->dev : nullptr) || !encodeAlpha(job, ea, s, eerr)) {
                fprintf(stderr, "encode: %s\n", eerr.c_str()); oomHint(eerr); encFail++; delete job; return;
            }
            if (tm.dpRepaired) warnNet(job, false);
            const double te = now_ms() - ts;
            for (auto &b : held) ingest_release_batch(ictx, b.batch_id, s);
            for (uint8_t *d : fbHeld) cudaFreeAsync(d, s);
            gpuFallbackA += (int)fbHeld.size() - fbConv;
            {
                std::lock_guard<std::mutex> lk2(aggMu);
                A.encodeWall += te;
                A.gpuSetup += tm.setup; A.analysis += tm.analysis; A.dp += tm.dp; A.recon += tm.recon; A.pack += tm.pack; A.postfilter += tm.postfilter;
                A.download += tm.download; A.gpuTotal += tm.total; A.blocks += tm.blocks; A.dpRepaired += tm.dpRepaired;
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
    // every return from here on (error returns too) joins the threads first: a joinable std::thread's destructor calls
    // std::terminate.  Encoders first (they may still push to the entropy queue), then the fallback decoder, then the
    // entropy threads once the queue is closed (they finish the jobs already queued).  No-op after the normal joins.
    struct JoinAll {
        std::function<void()> f;
        ~JoinAll() { f(); }
    } joinAll{[&]() {
        for (auto &t : encThreads)
            if (t.joinable()) t.join();
        ingestDrained = true;
        if (fbThread.joinable()) fbThread.join();
        {
            std::lock_guard<std::mutex> lk(mu);
            done = true;
        }
        cv.notify_all();
        for (auto &t : entThreads)
            if (t.joinable()) t.join();
    }};
    for (int k = 0; k < nEnc; k++) {
        GpEncoder *E = k == 0 ? &enc : new GpEncoder;
        if (k > 0 && E->init(bcfg, err)) { fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
        cudaStream_t s = es;
        if (k > 0) cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        encs.push_back(E);
        streams.push_back(s);
    }
    std::vector<GpEncoder *> encsSmall(nEnc, nullptr);
    if (bcfg.smallMp > 0) {
        GpConfig cs = bcfg;
        cs.P = bcfg.Psmall;
        cs.an = bcfg.anSmall;
        for (int k = 0; k < nEnc; k++) {
            encsSmall[k] = new GpEncoder;
            if (encsSmall[k]->init(cs, err)) { fprintf(stderr, "init: %s\n", err.c_str()); return 1; }
        }
    }
    encLive = nEnc;
    for (int k = 0; k < nEnc; k++)
        encThreads.emplace_back([&, k]() {
            encLoop(encs[k], encsSmall[k], streams[k]);
            encLive--;
            fbCv.notify_all();
        });
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
    gp_pinned_free_exited();
    const double mp = A.pixels / 1e6;
    fprintf(stderr,
            "[av1gpu] %llu images (%.1f MP) in %.1f ms: %.2f img/s, %.2f MP/s; %llu batches; out %.2f MB (%.3f bpp); "
            "CPU %.0f ms = %.2f cores; skipped (unsupported): %d; WebP/PNG via CPU decode: %d; write failures %d\n",
            (unsigned long long)A.images, mp, wall, A.images * 1e3 / wall, mp * 1e3 / wall, (unsigned long long)A.batches,
            A.bytes / 1e6, A.pixels ? A.bytes * 8.0 / A.pixels : 0.0, cpu, cpu / wall, cpuFallback, gpuFallback, failures);
    if (cfg.trim.on) fprintf(stderr, "[av1gpu] trim: %d images trimmed, %d skipped as blank\n", trimCutA.load(), trimBlankA.load());
    if (A.dpRepaired) gp_warn_dp_net(nullptr, A.dpRepaired);   // (the inputs were named as they were encoded)
    fprintf(stderr,
            "[av1gpu] stage ms: ingest wait %.1f | gpu setup %.1f analysis %.1f dp %.1f recon %.1f pack %.1f postfilter %.1f download %.1f "
            "(gpu total %.1f, encode wall %.1f) | entropy wall %.1f (gpu %.1f) | write %.1f | peak dev %.0f MB\n",
            A.ingestWait, A.gpuSetup, A.analysis, A.dp, A.recon, A.pack, A.postfilter, A.download, A.gpuTotal, A.encodeWall, A.entropyWall,
            A.entropyGpu, A.writeWall, A.peakDev / 1048576.0);
    if (gp_trusted_checked)
        fprintf(stderr, "[av1gpu] GP_CHECK_TRUSTED: %llu entropy batches checked, %llu differences between trusted and derived records\n",
                (unsigned long long)gp_trusted_checked.load(), (unsigned long long)gp_trusted_diffs.load());
    if (A.alphaImages)
        fprintf(stderr, "[av1gpu] alpha: %llu images with an alpha item (%.1f MP of alpha frames, %s)\n", (unsigned long long)A.alphaImages,
                A.alphaPixels / 1e6, acfg.lossless ? "lossless" : ("qindex " + std::to_string(acfg.qindex)).c_str());
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
               "\"peak_dev_mb\":%.1f,\"fallback_cpu\":%d,\"cpus\":%d,\"ingest_workers\":%d,\"fallback_workers\":%d,\"dp_repaired\":%llu%s}\n",
               (unsigned long long)A.images, mp, wall, A.images * 1e3 / wall, mp * 1e3 / wall, cpu / wall,
               (unsigned long long)A.bytes, A.ingestWait, A.analysis, A.dp, A.recon, A.pack, A.postfilter, A.download, A.gpuTotal,
               A.encodeWall, A.entropyWall, A.entropyGpu, A.writeWall, A.peakDev / 1048576.0, cpuFallback, nCpu, ic.n_workers, nfw,
               (unsigned long long)A.dpRepaired,
               cfg.trim.on ? (",\"trimmed\":" + std::to_string(trimCutA.load()) + ",\"blank\":" + std::to_string(trimBlankA.load())).c_str() : "");
    return failures ? 1 : 0;
}
