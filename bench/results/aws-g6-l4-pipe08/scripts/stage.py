#!/usr/bin/env python3
"""stage.py save|N: 'save' copies the worktree's final files to FINAL; N (1..4) writes the worktree files as of commit N
(the final files with the edits of the later groups reverted)."""
import os, shutil, sys
W = "/Users/mei/git/av1gpu/.claude/worktrees/agent-a3ea690e9d945437e"
FINAL = "/private/tmp/claude-501/-Users-mei-git-av1gpu/3ff16727-2f42-41f2-b891-0efc6c7767ab/scratchpad/pipe08/final"
FILES = ["encoder/apps/av1gpu.cpp", "encoder/src/batch.cpp", "encoder/src/pipeline.cpp", "ingest/src/ingest_api.h",
         "ingest/src/pipeline.cu"]
# (group, file, main text, final text)
E = []
# 1: hardware queues
E.append((1, "encoder/apps/av1gpu.cpp", """    // more hardware work queues than the default 8: ingest (nvJPEG + per-worker streams), encoders
    // (main + high-priority wavefront stream) and entropy threads use > 20 streams; aliased queues
    // serialise unrelated work (nsys 2026-10-02: next batch's analysis queued behind entropy)
    setenv("CUDA_DEVICE_MAX_CONNECTIONS", getenv("GP_MAX_CONNECTIONS") ? getenv("GP_MAX_CONNECTIONS") : "32", 0);""",
"""    // hardware work queues: long batches use 32, more than the default 8: ingest (nvJPEG + per-worker streams), encoders
    // (main + high-priority wavefront stream) and entropy threads use > 20 streams; aliased queues serialise unrelated
    // work (nsys 2026-10-02: next batch's analysis queued behind entropy).  But every queue is set up with the context
    // and torn down at exit: on an L4, 32 queues made each process 0.5 s slower than 1 (a 320 x 213 JPEG: 1.10 s against
    // 0.59 s).  A single image or a short list has little work to overlap, so it gets one queue.
    const char *conns = getenv("GP_MAX_CONNECTIONS");
    if (!conns) conns = gp_short_run(o) ? "1" : "32";
    setenv("CUDA_DEVICE_MAX_CONNECTIONS", conns, 0);"""))
E.append((1, "encoder/apps/av1gpu.cpp", """                   const std::string &suffix);

int main(int argc, char **argv) {""", """                   const std::string &suffix);
long gp_batch_input_count(const std::string &list);   // batch.cpp
// A run with little GPU work to overlap: single-file mode, or a --batch of at most kShortBatch encodes (inputs times
// --qindex values), such as one upload and its thumbnail.  Counted before CUDA starts (run_batch_mode reads the list
// again).
static bool gp_short_run(const Opts &o) {
    const long kShortBatch = 8;
    if (o.batch.empty()) return true;
    const long n = gp_batch_input_count(o.batch);   // -1 (unreadable): run_batch_mode reports it
    return n * std::max<long>(1, (long)o.qlist.size()) <= kShortBatch;
}

int main(int argc, char **argv) {"""))
E.append((1, "encoder/src/batch.cpp", """}  // namespace


// debug / batch gate (GP_WANT_RECON)""", """}  // namespace

// the number of inputs a --batch LIST|DIR names (-1: unreadable); av1gpu.cpp counts them before CUDA starts
long gp_batch_input_count(const std::string &list) {
    std::vector<std::string> paths, bad;
    std::vector<GpXform> xf;
    return list_inputs(list, paths, xf, GpXform(), bad) ? (long)paths.size() : -1;
}

// debug / batch gate (GP_WANT_RECON)"""))
# 2: no reconstruction unless an output uses it
E.append((2, "encoder/apps/av1gpu.cpp", """static int run_single(const GpConfig &cfg, const Opts &o) {
    std::string err;""", """static int run_single(const GpConfig &cfg0, const Opts &o) {
    std::string err;
    // the reconstruction (CDEF applied, downloaded) only for the outputs that use it, as in batch mode
    GpConfig cfg = cfg0;
    cfg.wantRecon = o.debugOutputs || o.check;"""))
E.append((2, "encoder/src/pipeline.cpp", """                cdefOn[i] = 1;
                // CDEF on the deblocked frame (the search left d_recon deblocked, and its maps and directions in pscr[i])
                av1g_pf_frame pf;""", """                cdefOn[i] = 1;
                // CDEF on the deblocked frame (the search left d_recon deblocked, and its maps and directions in pscr[i]).
                // Only the reconstruction download reads it (the search set cdef_idx), so without wantRecon it is skipped
                if (!cfg_.wantRecon) continue;
                av1g_pf_frame pf;"""))
# 3: variance-boost LUT row
E.append((3, "encoder/src/pipeline.cpp", """GpEncoder::GpEncoder() { memset(ev_, 0, sizeof ev_); memset(evx_, 0, sizeof evx_); }""",
"""// The variance-boost LUT of av1f_vb_build_luts_bd (lut[base][var], 256 x 1024) holds only the row of the encoder's
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

GpEncoder::GpEncoder() { memset(ev_, 0, sizeof ev_); memset(evx_, 0, sizeof evx_); }"""))
E.append((3, "encoder/src/pipeline.cpp", """        if (av1f_vb_build_luts(vbLuts_, cfg_.vbStrength, 0)) { err = "av1f_vb_build_luts failed"; return -1; }""",
          """        GPCK(gp_vb_lut_row(vbLuts_, cfg_.qindex, cfg_.vbStrength, 8, 0));"""))
E.append((3, "encoder/src/pipeline.cpp", """            if (av1f_vb_build_luts_bd(vbLuts10_, cfg_.vbStrength, 10, st)) { err = "av1f_vb_build_luts_bd failed"; A.release(); return -1; }""",
          """            if (gp_vb_lut_row(vbLuts10_, cfg_.qindex, cfg_.vbStrength, 10, st)) { err = "variance-boost LUT failed"; A.release(); return -1; }"""))
# 4: lazy arena
E.append((4, "ingest/src/ingest_api.h", """  double small_mp;        /* images below this go to the SMALL class (default 1.0) */
} ingest_config;""", """  double small_mp;        /* images below this go to the SMALL class (default 1.0) */
  int lazy_arena;         /* 1: pin the arena when the first image needs it (CPU-route JPEGs), not in ingest_create
                             (one image: baseline JPEGs go to the hardware decoder and never use it); default 0 */
} ingest_config;"""))
E.append((4, "ingest/src/pipeline.cu", """class Arena {
 public:
  bool init(size_t cap) {
    cap_ = align_up(cap, 4096);
    free_[0] = cap_;
#ifndef _WIN32""", """class Arena {
 public:
  // lazy: only the size now; the memory is pinned by the first alloc (map() under the lock)
  bool init(size_t cap, bool lazy = false, int device = 0) {
    cap_ = align_up(cap, 4096);
    free_[0] = cap_;
    device_ = device;
    return lazy || map();
  }
  bool map() {
#ifndef _WIN32"""))
E.append((4, "ingest/src/pipeline.cu", """    if (n > cap_) return nullptr;
    std::unique_lock<std::mutex> lk(m_);
    uint64_t ticket = next_ticket_++;""", """    if (n > cap_) return nullptr;
    std::unique_lock<std::mutex> lk(m_);
    if (!base_) {   // lazy arena: its first use (a worker thread)
      if (map_failed_) return nullptr;
      cudaSetDevice(device_);
      if (!map()) { map_failed_ = true; return nullptr; }
    }
    uint64_t ticket = next_ticket_++;"""))
E.append((4, "ingest/src/pipeline.cu", """  bool mapped_ = false;   // mmap + cudaHostRegister (else cudaHostAlloc)""", """  bool mapped_ = false;   // mmap + cudaHostRegister (else cudaHostAlloc)
  bool map_failed_ = false;
  int device_ = 0;"""))
E.append((4, "ingest/src/pipeline.cu", """  if (!c->arena.init((size_t)k.pinned_bytes)) return fail("cudaHostAlloc of the pinned arena failed");""",
          """  if (!c->arena.init((size_t)k.pinned_bytes, k.lazy_arena != 0, k.device)) return fail("cudaHostAlloc of the pinned arena failed");"""))
E.append((4, "encoder/apps/av1gpu.cpp", """        ic.max_mp = decMaxMp;
        if (getenv("GP_INGEST_DECODER")) ic.decoder = atoi(getenv("GP_INGEST_DECODER"));   // 1: CPU entropy decode + GPU IDCT""",
          """        ic.max_mp = decMaxMp;
        ic.lazy_arena = 1;   // baseline JPEGs (the hardware decoder) never touch it: 24 ms of pinning saved
        if (getenv("GP_INGEST_DECODER")) ic.decoder = atoi(getenv("GP_INGEST_DECODER"));   // 1: CPU entropy decode + GPU IDCT"""))

# 5: JPEG decode overlapped with the encoder's init
E.append((5, "encoder/apps/av1gpu.cpp", """#include <string>
#include <vector>""", """#include <string>
#include <thread>
#include <vector>"""))
E.append((5, "encoder/apps/av1gpu.cpp", """    GpEncoder enc;
    if (enc.init(cfg, err)) { fprintf(stderr, "init: %s\\n", err.c_str()); return 1; }
    GpFrameIn fin, afin;   // afin: the alpha frame (W = 0: none)
    std::vector<uint8_t> icc;
    int orientation = 1;
    ingest_ctx *ictx = nullptr;
    ingest_batch ib;
    memset(&ib, 0, sizeof ib);
    uint8_t *dplanes = nullptr, *dconv = nullptr, *dxf = nullptr;
    auto t0 = std::chrono::steady_clock::now();
    if (kind == GP_FILE_JPEG) {
        ingest_config ic;""", """    GpEncoder enc;
    GpFrameIn fin, afin;   // afin: the alpha frame (W = 0: none)
    std::vector<uint8_t> icc;
    int orientation = 1;
    ingest_ctx *ictx = nullptr;
    ingest_batch ib;
    memset(&ib, 0, sizeof ib);
    uint8_t *dplanes = nullptr, *dconv = nullptr, *dxf = nullptr;
    auto t0 = std::chrono::steady_clock::now();
    // a JPEG is decoded in a thread while the encoder initialises: setting up nvJPEG's hardware decoder (~70 ms on an
    // L4) overlaps the table uploads and kernel-module loads of GpEncoder::init
    std::thread ingestT;
    int ir = 0;
    char eb[256] = {0};
    if (kind == GP_FILE_JPEG) {
        ingest_config ic;"""))
E.append((5, "encoder/apps/av1gpu.cpp", """        if (getenv("GP_INGEST_DECODER")) ic.decoder = atoi(getenv("GP_INGEST_DECODER"));   // 1: CPU entropy decode + GPU IDCT
        char eb[256] = {0};
        ictx = ingest_create(&ic, eb, sizeof eb);
        if (!ictx) { fprintf(stderr, "ingest_create: %s\\n", eb); return 1; }
        const char *paths[1] = {o.in.c_str()};
        ingest_submit(ictx, paths, 1);
        ingest_finish_submit(ictx);
        int r = ingest_next_batch(ictx, &ib, -1);
        if (r != 1) {""", """        if (getenv("GP_INGEST_DECODER")) ic.decoder = atoi(getenv("GP_INGEST_DECODER"));   // 1: CPU entropy decode + GPU IDCT
        ingestT = std::thread([&, ic]() {
            ictx = ingest_create(&ic, eb, sizeof eb);
            if (!ictx) return;
            const char *paths[1] = {o.in.c_str()};
            ingest_submit(ictx, paths, 1);
            ingest_finish_submit(ictx);
            ir = ingest_next_batch(ictx, &ib, -1);
        });
    }
    const int initRc = enc.init(cfg, err);
    if (ingestT.joinable()) ingestT.join();
    if (initRc) {
        fprintf(stderr, "init: %s\\n", err.c_str());
        if (ictx) ingest_destroy(ictx);
        return 1;
    }
    if (kind == GP_FILE_JPEG) {
        if (!ictx) { fprintf(stderr, "ingest_create: %s\\n", eb); return 1; }
        if (ir != 1) {"""))
# 6: no orderly teardown at the end of a single image (Linux)
E.append((6, "encoder/apps/av1gpu.cpp", """#include <cuda_runtime.h>
""", """#include <cuda_runtime.h>
#ifndef _WIN32
#include <unistd.h>
#endif
"""))
E.append((6, "encoder/apps/av1gpu.cpp", """    if (ictx) { ingest_release_batch(ictx, ib.batch_id, nullptr); ingest_destroy(ictx); }
""", """#ifdef _WIN32
    if (ictx) { ingest_release_batch(ictx, ib.batch_id, nullptr); ingest_destroy(ictx); }
#else
    // Linux: the ingest is not torn down; the process ends after this image (main: _exit), and the kernel frees its
    // memory faster than nvJPEG's teardown (the destroy took 77 ms on an L4, 35 ms of it freeing the preallocated decoder)
#endif
"""))
E.append((6, "encoder/apps/av1gpu.cpp", """    return run_single(cfg, o);
}""", """    const int rc = run_single(cfg, o);
#ifndef _WIN32
    // single image, Linux: end without the orderly CUDA / nvJPEG teardown (run_single closed every output file; the kernel
    // releases the GPU context and memory as the process exits): 40 ms less per process on an L4
    fflush(stdout);
    fflush(stderr);
    _exit(rc);
#endif
    return rc;
}"""))

if sys.argv[1] == "save":
    os.makedirs(FINAL, exist_ok=True)
    for f in FILES:
        shutil.copy(f"{W}/{f}", f"{FINAL}/{f.replace('/', '_')}")
    print("saved")
    sys.exit()
N = int(sys.argv[1])
for f in FILES:
    s = open(f"{FINAL}/{f.replace('/', '_')}").read()
    for g, ff, old, new in reversed(E):
        if ff != f or g <= N:
            continue
        if s.count(new) != 1:
            sys.exit(f"group {g} {f}: final text found {s.count(new)} times")
        s = s.replace(new, old)
    open(f"{W}/{f}", "w").write(s)
print("state", N)
