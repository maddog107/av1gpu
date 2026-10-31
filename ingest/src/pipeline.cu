// Ingest pipeline: CPU entropy-decode workers -> pinned arena -> batcher (H2D + GPU IDCT/finalize)
// -> consumer (ingest_next_batch / ingest_release_batch).  See ingest_api.h for the contract.
#include <sys/resource.h>
#include <sys/syscall.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>
#include <nvjpeg.h>

#include "ingest_api.h"
#include "jpeg_reader.h"
#include "kernels.cuh"

void ingest_fill_meta(const ingest::JpegInfo &in, ingest_image *o);

namespace ingest {

static inline double thread_cpu_ms() {
  timespec ts;
  clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts);
  return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}
static inline size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

// ------------------------------------------------------------------ pinned arena (FIFO first-fit)
class Arena {
 public:
  bool init(size_t cap) {
    cap_ = align_up(cap, 4096);
    if (cudaHostAlloc((void **)&base_, cap_, cudaHostAllocPortable) != cudaSuccess) return false;
    free_[0] = cap_;
    return true;
  }
  ~Arena() {
    if (base_) cudaFreeHost(base_);
  }
  size_t cap() const { return cap_; }
  // Blocks until it fits (FIFO among waiters). Returns nullptr if closing or n > cap.
  uint8_t *alloc(size_t n) {
    n = align_up(std::max<size_t>(n, 1), 256);
    if (n > cap_) return nullptr;
    std::unique_lock<std::mutex> lk(m_);
    uint64_t ticket = next_ticket_++;
    for (;;) {
      if (closing_) return nullptr;
      if (ticket == serving_) {
        for (auto it = free_.begin(); it != free_.end(); ++it) {
          if (it->second >= n) {
            size_t off = it->first, sz = it->second;
            free_.erase(it);
            if (sz > n) free_[off + n] = sz - n;
            used_ += n;
            peak_ = std::max(peak_, used_);
            ++serving_;
            cv_.notify_all();
            return base_ + off;
          }
        }
      }
      cv_.wait(lk);
    }
  }
  void release(uint8_t *p, size_t n) {
    n = align_up(std::max<size_t>(n, 1), 256);
    std::lock_guard<std::mutex> lk(m_);
    size_t off = (size_t)(p - base_);
    auto it = free_.emplace(off, n).first;
    auto nx = std::next(it);
    if (nx != free_.end() && it->first + it->second == nx->first) {
      it->second += nx->second;
      free_.erase(nx);
    }
    if (it != free_.begin()) {
      auto pv = std::prev(it);
      if (pv->first + pv->second == it->first) {
        pv->second += it->second;
        free_.erase(it);
      }
    }
    used_ -= n;
    cv_.notify_all();
  }
  void close() {
    std::lock_guard<std::mutex> lk(m_);
    closing_ = true;
    cv_.notify_all();
  }
  size_t peak() const { return peak_; }

 private:
  uint8_t *base_ = nullptr;
  size_t cap_ = 0, used_ = 0, peak_ = 0;
  std::map<size_t, size_t> free_;
  std::mutex m_;
  std::condition_variable cv_;
  uint64_t next_ticket_ = 0, serving_ = 0;
  bool closing_ = false;
};

struct Item {
  int index = 0;
  JpegInfo info;
  ingest_image meta;
  uint8_t *chunk = nullptr;
  size_t chunk_bytes = 0;
  int cls = 0;
  long long pixels = 0;
  bool hw = false;              // route: nvJPEG hardware decode of `bits`
  std::vector<uint8_t> bits;    // the JPEG file (hardware route only)
};

struct Batch {
  int64_t id = 0;
  int cls = 0;
  std::vector<std::unique_ptr<Item>> items;
  std::vector<ingest_image> metas;
  long long pixels = 0;
  std::chrono::steady_clock::time_point t_first;
  cudaStream_t stream = nullptr;
  cudaEvent_t ev0 = nullptr, ev1 = nullptr, ev2 = nullptr, evh0 = nullptr, evh1 = nullptr;
  int n_hw = 0;
  uint8_t *d_pool = nullptr, *d_scratch = nullptr, *d_native = nullptr;
  size_t pool_bytes = 0, scratch_bytes = 0, native_bytes = 0, coef_bytes = 0;
  bool scratch_freed = false;
  std::vector<uint8_t> hdesc;
};

struct HostFree {
  Arena *arena;
  std::vector<std::pair<uint8_t *, size_t>> chunks;
};
static void CUDART_CB host_free_cb(void *p) {
  auto *h = static_cast<HostFree *>(p);
  for (auto &c : h->chunks) h->arena->release(c.first, c.second);
  delete h;
}

}  // namespace ingest

using namespace ingest;

struct ingest_ctx {
  ingest_config cfg;
  Arena arena;
  std::vector<std::thread> workers;
  std::thread batcher;
  std::vector<cudaStream_t> streams;
  int stream_prio = 0;

  std::mutex m;
  std::condition_variable cv_jobs, cv_ready, cv_done, cv_inflight;
  std::vector<std::string> paths;
  size_t next_job = 0;
  bool submit_finished = false;
  bool stopping = false;
  int active_workers = 0;
  std::deque<std::unique_ptr<Item>> ready;
  std::deque<std::unique_ptr<Batch>> done;
  std::unordered_map<int64_t, std::unique_ptr<Batch>> held;
  bool batcher_finished = false;
  int inflight = 0;
  int64_t next_batch_id = 0;
  std::vector<ingest_fallback> fallbacks;
  std::string error;

  // nvJPEG hardware decoder (used by the batcher thread only)
  bool hw = false;
  unsigned hw_engines = 0;
  nvjpegHandle_t nvj = nullptr;
  nvjpegJpegState_t nvj_state = nullptr;
  int nvj_bs = -1;
  cudaEvent_t hw_done = nullptr;   // recorded on the stream of every nvJPEG call
  size_t hw_slot_bytes = 0;
  // Pre-allocated decoder (default): nvjpegDecodeBatchedPreAllocate for 1 image of up to
  // hw_max_side x hw_max_side (4:4:4) and every call decodes exactly 1 image, so the state is never
  // re-initialised and libnvcuvid never reallocates.  Otherwise it calls a synchronous cuMemFree
  // (device-wide drain under the driver lock) about 1.5x per image whenever the image size or the
  // batch size changes.  Images with a larger side take the CPU route.
  bool hw_prealloc = false;
  int hw_max_side = 0;
  // Hardware-route output: nvJPEG writes from its own internal stream, so its destination must not
  // be stream-ordered (cudaMallocAsync) memory of a batch stream.  Persistent cudaMalloc slots (1 by
  // default, INGEST_HW_SLOTS=2 double-buffers), sized at create for the largest possible batch, used
  // round-robin; a slot is reused only after the kernels that read it ran (event `free_ev`).
  struct HwSlot {
    uint8_t *buf = nullptr;
    size_t cap = 0;
    cudaEvent_t free_ev = nullptr;
    bool pending = false;
  } hw_slots[2];
  int hw_nslots = 1, hw_next = 0;
  cudaStream_t hw_stream = nullptr;  // every nvJPEG call is issued on this stream
  std::vector<uint8_t *> hw_retired;  // outgrown slot buffers, freed at destroy
  bool hw_legacy_out = false;        // experiment knob INGEST_HW_OUT=legacy: decode into batch scratch on s

  // stats
  int64_t n_hw = 0, n_hw_failed = 0;
  double hw_ms = 0, batcher_cpu = 0;
  int64_t n_decoded = 0, n_batches = 0, n_released = 0, pixels = 0, coef_bytes = 0;
  double cpu_read = 0, cpu_decode = 0, cpu_copy = 0, gpu_ms = 0, h2d_ms = 0;
  size_t device_bytes = 0, device_peak = 0;
};

static void set_error(ingest_ctx *c, const std::string &e) {
  std::lock_guard<std::mutex> lk(c->m);
  if (c->error.empty()) c->error = e;
  c->cv_done.notify_all();
}

#define CK(call)                                                                              \
  do {                                                                                        \
    cudaError_t _e = (call);                                                                  \
    if (_e != cudaSuccess) {                                                                  \
      set_error(c, std::string(#call) + ": " + cudaGetErrorString(_e));                       \
      return false;                                                                           \
    }                                                                                         \
  } while (0)

static bool read_file(const std::string &p, std::vector<uint8_t> &buf) {
  FILE *f = fopen(p.c_str(), "rb");
  if (!f) return false;
  if (fseek(f, 0, SEEK_END) != 0) { fclose(f); return false; }
  long n = ftell(f);
  if (n <= 0) { fclose(f); return false; }
  fseek(f, 0, SEEK_SET);
  buf.resize((size_t)n);
  size_t got = fread(buf.data(), 1, (size_t)n, f);
  fclose(f);
  return got == (size_t)n;
}

// Structural check for the hardware route: exactly one scan whose entropy-coded data contains
// only stuffed bytes (FF00), fill bytes and RSTn, terminated by EOI at the end of the file (a short
// 00/FF/whitespace tail is allowed).  Anything else -- truncation (no EOI), a stray marker from
// corruption, a second scan -- goes to the CPU route, where libjpeg-turbo's recovery behaviour
// (e.g. mid-gray fill of a truncated tail) is reproduced exactly; NVJPG would leave such regions
// unwritten / decode them differently.  Cost: one memchr pass over the file (~0.1 ms/MB).
static bool hw_stream_clean(const std::vector<uint8_t> &b) {
  const uint8_t *p = b.data(), *end = p + b.size();
  // trim padding tail
  size_t k = 0;
  while (end - p > 2 && k < 4096 &&
         (end[-1] == 0x00 || end[-1] == 0xFF || end[-1] == 0x0A || end[-1] == 0x0D || end[-1] == 0x20)) {
    --end;
    ++k;
  }
  if (end - p < 4 || p[0] != 0xFF || p[1] != 0xD8) return false;
  const uint8_t *q = p + 2;
  // marker segments up to SOS
  for (;;) {
    while (q < end && *q == 0xFF && q + 1 < end && q[1] == 0xFF) ++q;  // fill bytes
    if (end - q < 4 || q[0] != 0xFF) return false;
    const uint8_t m = q[1];
    const size_t len = ((size_t)q[2] << 8) | q[3];
    if (len < 2 || q + 2 + len > end) return false;
    q += 2 + len;
    if (m == 0xDA) break;
  }
  // entropy-coded data
  while (q < end) {
    const uint8_t *f = static_cast<const uint8_t *>(memchr(q, 0xFF, (size_t)(end - q)));
    if (!f || f + 1 >= end) return false;  // no EOI
    uint8_t m = f[1];
    while (m == 0xFF && f + 2 < end) {
      ++f;
      m = f[1];
    }
    if (m == 0x00 || (m >= 0xD0 && m <= 0xD7)) {
      q = f + 2;
      continue;
    }
    return m == 0xD9 && f + 2 == end;  // EOI must end the (trimmed) file
  }
  return false;
}

static void add_fallback(ingest_ctx *c, int index, int reason, int w, int h) {
  std::lock_guard<std::mutex> lk(c->m);
  c->fallbacks.push_back({index, reason, w, h});
}

// ------------------------------------------------------------------ worker
static void worker_main(ingest_ctx *c) {
  if (c->cfg.nice_workers) setpriority(PRIO_PROCESS, (id_t)syscall(SYS_gettid), c->cfg.nice_workers);
  CoefReader reader;
  std::vector<uint8_t> buf;
  for (;;) {
    int index;
    std::string path;
    {
      std::unique_lock<std::mutex> lk(c->m);
      c->cv_jobs.wait(lk, [&] { return c->stopping || c->next_job < c->paths.size() || c->submit_finished; });
      if (c->stopping || c->next_job >= c->paths.size()) {
        if (c->stopping || c->submit_finished) break;
        continue;
      }
      index = (int)c->next_job++;
      path = c->paths[index];
      ++c->active_workers;
    }
    double t0 = thread_cpu_ms();
    bool ok = read_file(path, buf);
    double t1 = thread_cpu_ms();
    auto item = std::make_unique<Item>();
    item->index = index;
    int fb = ok ? INGEST_FB_NONE : INGEST_FB_READ_ERROR;
    double t2 = t1, t2b = t1, t3 = t1;
    bool routed = false;
    if (ok && c->hw) {
      // header only; baseline Huffman files in a subsampling the engine supports go to NVJPG
      fb = reader.read(buf.data(), buf.size(), c->cfg.max_mp, false, true, item->info);
      reader.reset();
      const JpegInfo &in = item->info;
      const bool fits = !c->hw_prealloc || (in.width <= c->hw_max_side && in.height <= c->hw_max_side);
      if (!fb && fits && !in.progressive && !in.arithmetic && hw_stream_clean(buf) &&
          (in.sub == INGEST_SUB_420 || in.sub == INGEST_SUB_444 || in.sub == INGEST_SUB_422 ||
           in.sub == INGEST_SUB_440 || in.sub == INGEST_SUB_GRAY)) {
        item->hw = true;
        item->bits.swap(buf);
        routed = true;
      }
      if (fb) routed = true;  // fallback decided from the header
      t2 = t2b = t3 = thread_cpu_ms();
    }
    if (routed) {
    } else if (ok && !c->cfg.legacy_coef) {
      // entropy decode straight into the pinned arena (IDCT hook); alloc may block, which is
      // not CPU time, so the measured CPU time stays clean
      fb = reader.read_fast(buf.data(), buf.size(), c->cfg.max_mp, item->info, [&](size_t bytes) -> int16_t * {
        if (bytes > c->arena.cap()) return nullptr;
        item->chunk = c->arena.alloc(bytes);
        item->chunk_bytes = bytes;
        return reinterpret_cast<int16_t *>(item->chunk);
      });
      t2 = t2b = t3 = thread_cpu_ms();
      if (fb && item->chunk) {
        c->arena.release(item->chunk, item->chunk_bytes);
        item->chunk = nullptr;
      }
    } else if (ok) {
      fb = reader.read(buf.data(), buf.size(), c->cfg.max_mp, true, true, item->info);
      t2 = thread_cpu_ms();
      if (!fb) {
        item->chunk_bytes = item->info.total_blocks() * 128;
        if (item->chunk_bytes > c->arena.cap()) fb = INGEST_FB_TOO_BIG;
      }
      if (!fb) {
        item->chunk = c->arena.alloc(item->chunk_bytes);  // may block (not CPU time)
        if (!item->chunk) fb = INGEST_FB_TOO_BIG;          // arena closing
      }
      t2b = thread_cpu_ms();
      if (!fb) reader.copy_coefs(reinterpret_cast<int16_t *>(item->chunk));
      reader.reset();
      t3 = thread_cpu_ms();
    }
    if (fb) {
      reader.reset();
      add_fallback(c, index, fb, item->info.width, item->info.height);
    } else {
      ingest_fill_meta(item->info, &item->meta);
      item->meta.index = index;
      item->meta.cpu_ms_read = (float)(t1 - t0);
      item->meta.cpu_ms_decode = (float)(t2 - t1);
      item->meta.cpu_ms_copy = (float)(t3 - t2b);
      item->pixels = (long long)item->info.width * item->info.height;
      item->cls = item->meta.tiled ? INGEST_CLASS_TILED
                  : (double)item->pixels < c->cfg.small_mp * 1e6 ? INGEST_CLASS_SMALL
                                                                 : INGEST_CLASS_REGULAR;
    }
    {
      std::lock_guard<std::mutex> lk(c->m);
      c->cpu_read += t1 - t0;
      if (!fb) {
        c->cpu_decode += item->meta.cpu_ms_decode;
        c->cpu_copy += item->meta.cpu_ms_copy;
        ++c->n_decoded;
        c->ready.push_back(std::move(item));
      } else {
        c->cpu_decode += t2 - t1;
      }
      --c->active_workers;
      c->cv_ready.notify_all();
    }
  }
  std::lock_guard<std::mutex> lk(c->m);
  c->cv_ready.notify_all();
}

// ------------------------------------------------------------------ batch dispatch (batcher thread)
static bool hw_decode(ingest_ctx *c, Batch *b, const std::vector<int> &hw_items, const std::vector<int> &first_plane,
                      const std::vector<PlaneDesc> &planes, const std::vector<uint8_t *> &dst, cudaStream_t hs);

static bool dispatch(ingest_ctx *c, std::unique_ptr<Batch> b) {
  {
    std::unique_lock<std::mutex> lk(c->m);
    c->cv_inflight.wait(lk, [&] { return c->stopping || c->inflight < c->cfg.max_inflight; });
    if (c->stopping) {
      for (auto &it : b->items)
        if (it->chunk) c->arena.release(it->chunk, it->chunk_bytes);
      return false;
    }
    ++c->inflight;
    b->id = c->next_batch_id++;
  }
  const bool rotate = c->cfg.apply_orientation != 0;
  const bool act = c->cfg.activity != 0;
  const size_t A = (size_t)c->cfg.align;
  const size_t n = b->items.size();

  std::vector<PlaneDesc> planes;
  std::vector<OutDesc> outs;
  std::vector<ActDesc> acts;
  std::vector<ActPixDesc> apix;
  std::vector<uint16_t> qts;
  std::vector<long long> act_native;  // per plane: activity map offset in the native buffer, -1 = pool/none
  std::vector<int> first_plane(n, -1), hw_items;
  std::vector<long long> slot_off;     // per plane: offset in the hardware output slot, -1 = not there
  std::vector<int> apix_plane;         // per apix entry: its luma plane
  long long blk = 0, tiles = 0, act_elems = 0, apix_blocks = 0;
  size_t nat = 0, pool = 0, hw_nat = 0;
  const bool in_slot = c->hw && !c->hw_legacy_out;   // hw planes live in a slot ...
  const bool copy_back = in_slot && c->cfg.keep_native;  // ... and are copied to scratch for keep_native

  // native planes: CPU-path (coefficient) images first so the IDCT kernel sees a contiguous,
  // blk0-sorted prefix of `planes`; hardware-decoded images after them (no coefficients).
  int n_coef_planes = 0;
  for (int pass = 0; pass < 2; ++pass) {
    for (size_t i = 0; i < n; ++i) {
      Item &it = *b->items[i];
      if ((pass == 0) == it.hw) continue;
      if (it.hw) hw_items.push_back((int)i);
      const JpegInfo &in = it.info;
      first_plane[i] = (int)planes.size();
      for (int k = 0; k < in.ncomp; ++k) {
        PlaneDesc p{};
        p.blk0 = blk;
        p.wb = in.wb[k];
        p.hb = in.hb[k];
        p.qt = (int)(qts.size() / 64);
        qts.insert(qts.end(), in.qt[k], in.qt[k] + 64);
        p.nat_pitch = in.wb[k] * 8;
        p.nat_off = (long long)nat;
        const size_t pbytes = (size_t)p.nat_pitch * in.hb[k] * 8;
        if (it.hw && in_slot) {
          slot_off.push_back((long long)hw_nat);
          hw_nat = align_up(hw_nat + pbytes, 256);
          if (copy_back) nat = align_up(nat + pbytes, 256);  // reported native copy in scratch
          else p.nat_off = -1;                              // rebased to the slot below
        } else {
          slot_off.push_back(-1);
          nat = align_up(nat + pbytes, 256);
        }
        if (!it.hw) blk += (long long)in.wb[k] * in.hb[k];
        it.meta.nat_off[k] = c->cfg.keep_native ? p.nat_off : -1;
        it.meta.nat_pitch[k] = p.nat_pitch;
        planes.push_back(p);
        act_native.push_back(-1);
      }
    }
    if (pass == 0) n_coef_planes = (int)planes.size();
  }

  b->metas.resize(n);
  for (size_t i = 0; i < n; ++i) {
    Item &it = *b->items[i];
    ingest_image &m = it.meta;
    const JpegInfo &in = it.info;
    const int W = in.width, H = in.height;
    const bool orient = rotate && in.orientation > 1;
    const bool swap = orient && in.orientation >= 5;
    const int Wo = swap ? H : W, Ho = swap ? W : H;
    const int fp = first_plane[i];
    m.width = Wo;
    m.height = Ho;
    m.oriented = orient ? 1 : 0;
    m.decoder_used = it.hw ? 1 : 0;
    m.w8 = (int)align_up(Wo, 8);
    m.h8 = (int)align_up(Ho, 8);
    m.aw = (int)align_up(Wo, A);
    m.ah = (int)align_up(Ho, A);
    // 4:4:4 output (cfg.out_444 = 1): full-size chroma planes; out_444 = 2 (source format): full-size for 4:4:4
    // JPEGs, 4:2:0 (an exact copy of the native chroma for 4:2:0 JPEGs) otherwise
    const int cfull = !m.mono && (c->cfg.out_444 == 1 || (c->cfg.out_444 == 2 && in.sub == INGEST_SUB_444)) ? 1 : 0;
    m.chroma_full = cfull;
    m.y_pitch = m.aw;
    m.c_pitch = cfull ? m.aw : m.aw / 2;
    m.icc = in.icc.empty() ? nullptr : in.icc.data();
    m.exif = in.exif.empty() ? nullptr : in.exif.data();
    // output planes
    m.y_off = (long long)pool;
    pool = align_up(pool + (size_t)m.y_pitch * m.ah, 256);
    if (!m.mono) {
      const int ch = cfull ? m.ah : m.ah / 2;
      m.cb_off = (long long)pool;
      pool = align_up(pool + (size_t)m.c_pitch * ch, 256);
      m.cr_off = (long long)pool;
      pool = align_up(pool + (size_t)m.c_pitch * ch, 256);
    } else {
      m.cb_off = m.cr_off = -1;
    }
    for (int k = 0; k < (m.mono ? 1 : 3); ++k) {
      OutDesc o{};
      o.out_off = k == 0 ? m.y_off : (k == 1 ? m.cb_off : m.cr_off);
      o.out_pitch = k == 0 ? m.y_pitch : m.c_pitch;
      o.out_w = (k == 0 || cfull) ? m.aw : m.aw / 2;
      o.out_h = (k == 0 || cfull) ? m.ah : m.ah / 2;
      o.tiles_x = (o.out_w + 63) / 64;
      o.tile0 = tiles;
      tiles += (long long)o.tiles_x * ((o.out_h + 15) / 16);
      o.chroma = k == 0 ? 0 : cfull ? 2 : 1;
      o.src = fp + k;
      o.W = W;
      o.H = H;
      o.Wo = Wo;
      o.Ho = Ho;
      o.hs = in.h[k];
      o.vs = in.v[k];
      o.hmax = in.hmax;
      o.vmax = in.vmax;
      o.orient = orient ? in.orientation : 1;
      outs.push_back(o);
    }
    // activity
    m.act_off = -1;
    m.act_w = m.w8 / 8;
    m.act_h = m.h8 / 8;
    if (act) {
      size_t bytes = (size_t)m.act_w * m.act_h * 16;
      m.act_off = (long long)pool;
      pool = align_up(pool + bytes, 256);
      PlaneDesc &py = planes[fp];
      long long target;          // pool-relative, or native-relative when act_native >= 0
      int target_pitch;
      // luma native block grid == output grid when Y is full-res and not rotated
      bool direct = !orient && in.h[0] == in.hmax && in.v[0] == in.vmax && py.wb == m.act_w && py.hb == m.act_h;
      if (direct) {
        target = m.act_off;
        target_pitch = m.act_w;
      } else {
        ActDesc d{};
        d.src_off = (long long)nat;
        act_native[fp] = (long long)nat;
        target = -1;
        target_pitch = py.wb;
        nat = align_up(nat + (size_t)py.wb * py.hb * 16, 256);
        d.dst_off = m.act_off;
        d.src_w = py.wb;
        d.src_h = py.hb;
        d.dst_w = m.act_w;
        d.dst_h = m.act_h;
        d.W = (in.h[0] == in.hmax) ? W : (int)((long)W * in.h[0] / in.hmax);  // luma-plane pixel space
        d.H = (in.v[0] == in.vmax) ? H : (int)((long)H * in.v[0] / in.vmax);
        d.Wo = Wo;
        d.Ho = Ho;
        d.orient = orient ? in.orientation : 1;
        d.t0 = act_elems;
        act_elems += (long long)m.act_w * m.act_h;
        acts.push_back(d);
      }
      if (!it.hw) {
        py.has_act = 1;
        py.act_off = target;  // fixed up below when native-resident
        py.act_pitch = target_pitch;
      } else {
        ActPixDesc a{};
        a.nat_off = py.nat_off;
        a.act_off = target;
        a.blk0 = apix_blocks;
        a.wb = py.wb;
        a.hb = py.hb;
        a.nat_pitch = py.nat_pitch;
        a.act_pitch = target_pitch;
        a.vis_w = (int)(((long)W * in.h[0] + in.hmax - 1) / in.hmax);
        a.vis_h = (int)(((long)H * in.v[0] + in.vmax - 1) / in.vmax);
        a.qt = py.qt;
        apix_blocks += (long long)py.wb * py.hb;
        apix.push_back(a);
        apix_plane.push_back(fp);
      }
    }
    m.quality = in.quality;
    b->metas[i] = m;
  }
  const size_t coef_bytes = (size_t)blk * 128;
  const size_t sz_planes = align_up(planes.size() * sizeof(PlaneDesc), 256);
  const size_t sz_outs = align_up(outs.size() * sizeof(OutDesc), 256);
  const size_t sz_acts = align_up(acts.size() * sizeof(ActDesc), 256);
  const size_t sz_apix = align_up(apix.size() * sizeof(ActPixDesc), 256);
  const size_t desc_bytes = sz_planes + sz_outs + sz_acts + sz_apix + align_up(qts.size() * 2, 256);
  const size_t o_out = sz_planes, o_act = o_out + sz_outs, o_apix = o_act + sz_acts, o_qt = o_apix + sz_apix;
  // scratch layout: [coef][native][desc]
  const size_t nat_base = align_up(coef_bytes, 256);
  const size_t desc_base = align_up(nat_base + nat, 256);
  b->scratch_bytes = desc_base + desc_bytes;
  b->pool_bytes = std::max<size_t>(pool, 256);
  b->native_bytes = nat;
  b->coef_bytes = coef_bytes;

  cudaStream_t s = c->streams[b->id % c->streams.size()];
  b->stream = s;
  CK(cudaEventCreate(&b->ev0));
  CK(cudaEventCreate(&b->ev1));
  CK(cudaEventCreate(&b->evh0));
  CK(cudaEventCreate(&b->evh1));
  CK(cudaEventCreateWithFlags(&b->ev2, cudaEventBlockingSync));  // consumer sleeps instead of spinning
  CK(cudaMallocAsync((void **)&b->d_scratch, b->scratch_bytes, s));
  CK(cudaMallocAsync((void **)&b->d_pool, b->pool_bytes, s));
  b->d_native = b->d_scratch + nat_base;
  {
    std::lock_guard<std::mutex> lk(c->m);
    c->device_bytes += b->scratch_bytes + b->pool_bytes;
    c->device_peak = std::max(c->device_peak, c->device_bytes);
  }
  // hardware route: acquire an output slot (waits until the kernels that read it last are done)
  ingest_ctx::HwSlot *slot = nullptr;
  if (!hw_items.empty() && in_slot) {
    slot = &c->hw_slots[c->hw_next];
    c->hw_next = (c->hw_next + 1) % c->hw_nslots;
    if (slot->pending) CK(cudaEventSynchronize(slot->free_ev));
    slot->pending = false;
    if (slot->cap < hw_nat) {
      // Only if batch_pixels / INGEST_HW_MAX_SIDE were exceeded (slots are sized for both at
      // create).  The old buffer is retired, not freed: cudaFree would synchronise the device.
      const size_t old_cap = slot->cap;
      if (slot->buf) c->hw_retired.push_back(slot->buf);
      slot->cap = align_up(std::max<size_t>(hw_nat + hw_nat / 4, 64u << 20), 1 << 20);
      CK(cudaMalloc((void **)&slot->buf, slot->cap));
      std::lock_guard<std::mutex> lk(c->m);
      c->hw_slot_bytes = 0;
      for (int k2 = 0; k2 < c->hw_nslots; ++k2) c->hw_slot_bytes += c->hw_slots[k2].cap;
      c->device_bytes += slot->cap - old_cap;
      c->device_peak = std::max(c->device_peak, c->device_bytes);
    }
    if (!copy_back) {
      const long long base = (long long)(slot->buf - b->d_native);
      for (size_t k = 0; k < planes.size(); ++k)
        if (slot_off[k] >= 0) planes[k].nat_off = base + slot_off[k];
    }
  }
  // activity maps living in the native buffer are addressed relative to the pool base
  const long long rebase = (long long)(b->d_native - b->d_pool);
  for (size_t k = 0; k < planes.size(); ++k)
    if (act_native[k] >= 0 && planes[k].has_act) planes[k].act_off = rebase + act_native[k];
  for (size_t j = 0; j < apix.size(); ++j) {
    const int pl = apix_plane[j];
    apix[j].nat_off = planes[pl].nat_off;  // after the slot rebase
    if (apix[j].act_off < 0) apix[j].act_off = rebase + act_native[pl];
  }
  // Descriptors are staged from pageable memory owned by the batch (not the pinned arena: the
  // batcher must never wait on the arena, workers waiting for arena space depend on it).
  b->hdesc.assign(desc_bytes, 0);
  uint8_t *hdesc = b->hdesc.data();
  std::memcpy(hdesc, planes.data(), planes.size() * sizeof(PlaneDesc));
  std::memcpy(hdesc + o_out, outs.data(), outs.size() * sizeof(OutDesc));
  std::memcpy(hdesc + o_act, acts.data(), acts.size() * sizeof(ActDesc));
  std::memcpy(hdesc + o_apix, apix.data(), apix.size() * sizeof(ActPixDesc));
  std::memcpy(hdesc + o_qt, qts.data(), qts.size() * 2);

  CK(cudaEventRecord(b->ev0, s));
  uint8_t *d_desc = b->d_scratch + desc_base;
  CK(cudaMemcpyAsync(d_desc, hdesc, desc_bytes, cudaMemcpyHostToDevice, s));
  size_t coff = 0;
  auto *hf = new HostFree{&c->arena, {}};
  for (auto &it : b->items) {
    if (it->hw) continue;
    CK(cudaMemcpyAsync(b->d_scratch + coff, it->chunk, it->chunk_bytes, cudaMemcpyHostToDevice, s));
    coff += it->chunk_bytes;
    hf->chunks.emplace_back(it->chunk, it->chunk_bytes);
    it->chunk = nullptr;
  }
  CK(cudaEventRecord(b->ev1, s));
  const PlaneDesc *d_planes = reinterpret_cast<const PlaneDesc *>(d_desc);
  const OutDesc *d_outs = reinterpret_cast<const OutDesc *>(d_desc + o_out);
  const ActDesc *d_acts = reinterpret_cast<const ActDesc *>(d_desc + o_act);
  const ActPixDesc *d_apix = reinterpret_cast<const ActPixDesc *>(d_desc + o_apix);
  const uint16_t *d_qt = reinterpret_cast<const uint16_t *>(d_desc + o_qt);
  launch_idct(reinterpret_cast<const short *>(b->d_scratch), blk, d_planes, n_coef_planes, d_qt, b->d_native,
              b->d_pool, s);
  CK(cudaEventRecord(b->evh0, s));
  if (!hw_items.empty()) {
    std::vector<uint8_t *> dst(planes.size(), nullptr);  // where nvJPEG writes each hw plane
    for (size_t k = 0; k < planes.size(); ++k)
      dst[k] = slot ? (slot_off[k] >= 0 ? slot->buf + slot_off[k] : nullptr) : b->d_native + planes[k].nat_off;
    if (!hw_decode(c, b.get(), hw_items, first_plane, planes, dst, slot ? c->hw_stream : s)) return false;
    if (slot) {
      // join: the batch stream waits for everything issued on the hardware stream
      CK(cudaEventRecord(c->hw_done, c->hw_stream));
      CK(cudaStreamWaitEvent(s, c->hw_done, 0));
      if (copy_back)
        for (size_t k = 0; k < planes.size(); ++k)
          if (slot_off[k] >= 0)
            CK(cudaMemcpyAsync(b->d_native + planes[k].nat_off, slot->buf + slot_off[k],
                               (size_t)planes[k].nat_pitch * planes[k].hb * 8, cudaMemcpyDeviceToDevice, s));
    }
  }
  CK(cudaEventRecord(b->evh1, s));
  launch_act_pixels(d_apix, (int)apix.size(), apix_blocks, d_qt, b->d_native, b->d_pool, s);
  launch_finalize(d_outs, (int)outs.size(), tiles, d_planes, b->d_native, b->d_pool, s);
  launch_act_remap(d_acts, (int)acts.size(), act_elems, b->d_native, b->d_pool, s);
  CK(cudaGetLastError());
  if (slot) {  // the slot may be overwritten once the kernels above have read it
    CK(cudaEventRecord(slot->free_ev, s));
    slot->pending = true;
  }
  CK(cudaEventRecord(b->ev2, s));
  CK(cudaLaunchHostFunc(s, host_free_cb, hf));
  for (auto &it : b->items) {  // bitstreams are no longer needed once the stream got this far
    if (it->hw) std::vector<uint8_t>().swap(it->bits);
  }
  if (!c->cfg.keep_native) {
    CK(cudaFreeAsync(b->d_scratch, s));
    b->scratch_freed = true;
    b->d_native = nullptr;
    std::lock_guard<std::mutex> lk(c->m);
    c->device_bytes -= b->scratch_bytes;
  }
  b->n_hw = (int)hw_items.size();
  {
    std::lock_guard<std::mutex> lk(c->m);
    ++c->n_batches;
    c->coef_bytes += (int64_t)coef_bytes;
    c->pixels += b->pixels;
    c->n_hw += b->n_hw;
    c->done.push_back(std::move(b));
    c->cv_done.notify_all();
  }
  return true;
}

// nvJPEG hardware decode of the batch's baseline images into their native planes (dst[plane]),
// issued on stream `hs` (the dedicated hardware stream).  On a failed call the images are retried
// one by one; an image the engine rejects is decoded on the CPU by libjpeg-turbo (islow, identical
// output) and copied into place on the same stream.
static bool hw_decode(ingest_ctx *c, Batch *b, const std::vector<int> &hw_items, const std::vector<int> &first_plane,
                      const std::vector<PlaneDesc> &planes, const std::vector<uint8_t *> &dst, cudaStream_t hs) {
  auto dest_of = [&](int i) {
    nvjpegImage_t d;
    std::memset(&d, 0, sizeof(d));
    const Item &it = *b->items[i];
    for (int k = 0; k < it.info.ncomp; ++k) {
      const PlaneDesc &p = planes[first_plane[i] + k];
      d.channel[k] = dst[first_plane[i] + k];
      d.pitch[k] = (unsigned)p.nat_pitch;
    }
    return d;
  };
  auto run = [&](const std::vector<int> &ids) -> nvjpegStatus_t {
    const int bs = (int)ids.size();
    // Wait for the previous nvJPEG call to finish.  Without pre-allocation its buffers are
    // re-initialised / reallocated when sizes change (cuMemFree inside libnvcuvid), and doing that
    // while an earlier hardware decode was in flight deadlocked (NOTES).  Kept with the
    // pre-allocated decoder as a safety margin; the engine is serial anyway.
    cudaEventSynchronize(c->hw_done);
    if (c->nvj_bs != bs) {  // pre-allocated decoder: only after a failed re-initialisation
      nvjpegStatus_t st = nvjpegDecodeBatchedInitialize(c->nvj, c->nvj_state, bs, 1, NVJPEG_OUTPUT_YUV);
      if (st != NVJPEG_STATUS_SUCCESS) return st;
      c->nvj_bs = bs;
    }
    std::vector<const unsigned char *> data(bs);
    std::vector<size_t> len(bs);
    std::vector<nvjpegImage_t> out(bs);
    for (int k = 0; k < bs; ++k) {
      data[k] = b->items[ids[k]]->bits.data();
      len[k] = b->items[ids[k]]->bits.size();
      out[k] = dest_of(ids[k]);
    }
    nvjpegStatus_t st = nvjpegDecodeBatched(c->nvj, c->nvj_state, data.data(), len.data(), out.data(), hs);
    cudaEventRecord(c->hw_done, hs);
    if (st != NVJPEG_STATUS_SUCCESS) {
      // the state must be re-initialised after a failure (rare: the stream was checked by
      // hw_stream_clean); with the pre-allocated decoder that means pre-allocating again
      cudaEventSynchronize(c->hw_done);
      c->nvj_bs = -1;
      if (c->hw_prealloc &&
          nvjpegDecodeBatchedInitialize(c->nvj, c->nvj_state, 1, 1, NVJPEG_OUTPUT_YUV) == NVJPEG_STATUS_SUCCESS &&
          nvjpegDecodeBatchedPreAllocate(c->nvj, c->nvj_state, 1, c->hw_max_side, c->hw_max_side, NVJPEG_CSS_444,
                                         NVJPEG_OUTPUT_YUV) == NVJPEG_STATUS_SUCCESS)
        c->nvj_bs = 1;
    }
    return st;
  };
  const int B = c->hw_prealloc ? 1 : std::max(1, c->cfg.hw_batch);
  for (size_t k0 = 0; k0 < hw_items.size(); k0 += B) {
    std::vector<int> ids(hw_items.begin() + k0, hw_items.begin() + std::min(hw_items.size(), k0 + B));
    if (run(ids) == NVJPEG_STATUS_SUCCESS) continue;
    for (int i : ids) {
      if (run({i}) == NVJPEG_STATUS_SUCCESS) continue;
      // CPU rescue: libjpeg-turbo planar decode (same islow output) into the native planes
      Item &it = *b->items[i];
      std::vector<uint8_t> tmp;
      int32_t info[32];
      int64_t need = ingest_ref_decode_raw(it.bits.data(), (int64_t)it.bits.size(), nullptr, 0, info);
      if (need < -1) {
        tmp.resize((size_t)(-need));
        need = ingest_ref_decode_raw(it.bits.data(), (int64_t)it.bits.size(), tmp.data(), (int64_t)tmp.size(), info);
      }
      if (need < 0) {
        set_error(c, "hardware decode and CPU rescue both failed for image " + std::to_string(it.index));
        return false;
      }
      size_t off = 0;
      for (int k = 0; k < it.info.ncomp; ++k) {
        const PlaneDesc &p = planes[first_plane[i] + k];
        size_t bytes = (size_t)p.nat_pitch * p.hb * 8;
        CK(cudaMemcpyAsync(dst[first_plane[i] + k], tmp.data() + off, bytes, cudaMemcpyHostToDevice, hs));
        off += bytes;
      }
      CK(cudaStreamSynchronize(hs));  // tmp is pageable and local
      b->metas[i].decoder_used = 2;
      std::lock_guard<std::mutex> lk(c->m);
      ++c->n_hw_failed;
    }
  }
  return true;
}

static void batcher_main(ingest_ctx *c) {
  cudaSetDevice(c->cfg.device);
  const double cpu0 = thread_cpu_ms();
  std::unique_ptr<Batch> open[3];
  auto limit_hit = [&](const Batch &b, const Item &it) {
    if (b.cls == INGEST_CLASS_TILED) return true;
    return b.pixels + it.pixels > c->cfg.batch_pixels || (int)b.items.size() >= c->cfg.max_batch_images;
  };
  for (;;) {
    std::vector<std::unique_ptr<Batch>> to_send;
    bool finished = false;
    {
      std::unique_lock<std::mutex> lk(c->m);
      c->cv_ready.wait_for(lk, std::chrono::milliseconds(5), [&] {
        return c->stopping || !c->ready.empty() ||
               (c->submit_finished && c->next_job >= c->paths.size() && c->active_workers == 0);
      });
      if (c->stopping) break;
      while (!c->ready.empty()) {
        auto it = std::move(c->ready.front());
        c->ready.pop_front();
        auto &ob = open[it->cls];
        if (ob && limit_hit(*ob, *it)) to_send.push_back(std::move(ob));
        if (!ob) {
          ob = std::make_unique<Batch>();
          ob->cls = it->cls;
          ob->t_first = std::chrono::steady_clock::now();
        }
        ob->pixels += it->pixels;
        ob->items.push_back(std::move(it));
        if (ob->cls == INGEST_CLASS_TILED) to_send.push_back(std::move(ob));
      }
      finished = c->submit_finished && c->next_job >= c->paths.size() && c->active_workers == 0 &&
                 c->ready.empty();
    }
    auto now = std::chrono::steady_clock::now();
    for (auto &ob : open) {
      if (!ob) continue;
      bool full = ob->pixels >= c->cfg.batch_pixels || (int)ob->items.size() >= c->cfg.max_batch_images;
      bool aged = std::chrono::duration<double, std::milli>(now - ob->t_first).count() >= c->cfg.max_wait_ms;
      if (full || aged || finished) to_send.push_back(std::move(ob));
    }
    for (auto &b : to_send)
      if (!dispatch(c, std::move(b))) {
        std::lock_guard<std::mutex> lk(c->m);
        if (c->error.empty() && !c->stopping) c->error = "dispatch failed";
        c->cv_done.notify_all();
      }
    if (finished && !open[0] && !open[1] && !open[2]) break;
  }
  std::lock_guard<std::mutex> lk(c->m);
  c->batcher_cpu = thread_cpu_ms() - cpu0;
  c->batcher_finished = true;
  c->cv_done.notify_all();
}

// ------------------------------------------------------------------ C API
extern "C" {

int ingest_abi_version(void) { return INGEST_ABI_VERSION; }

int ingest_sizeof(int which) {
  switch (which) {
    case 0: return (int)sizeof(ingest_config);
    case 1: return (int)sizeof(ingest_image);
    case 2: return (int)sizeof(ingest_batch);
    case 3: return (int)sizeof(ingest_stats);
    case 4: return (int)sizeof(ingest_fallback);
  }
  return -1;
}

void ingest_default_config(ingest_config *cfg) {
  std::memset(cfg, 0, sizeof(*cfg));
  cfg->n_workers = 2;
  cfg->device = 0;
  cfg->max_inflight = 3;
  cfg->max_wait_ms = 30;
  cfg->max_batch_images = 256;
  cfg->apply_orientation = 0;
  cfg->activity = 1;
  cfg->keep_native = 0;
  cfg->align = 64;
  cfg->nice_workers = 19;
  cfg->legacy_coef = 0;
  cfg->decoder = 0;
  cfg->hw_batch = 32;
  cfg->pinned_bytes = 512ll << 20;
  cfg->batch_pixels = 16000000;
  cfg->max_mp = 35.6;
  cfg->small_mp = 1.0;
}

ingest_ctx *ingest_create(const ingest_config *cfg, char *err, int errlen) {
  auto c = std::make_unique<ingest_ctx>();
  if (cfg) c->cfg = *cfg;
  else ingest_default_config(&c->cfg);
  auto fail = [&](const char *msg) -> ingest_ctx * {
    if (err && errlen > 0) snprintf(err, errlen, "%s", msg);
    return nullptr;
  };
  ingest_config &k = c->cfg;
  if (k.align < 8 || (k.align & (k.align - 1))) return fail("align must be a power of 2 >= 8");
  k.n_workers = std::max(1, k.n_workers);
  k.max_inflight = std::max(1, k.max_inflight);
  k.max_batch_images = std::max(1, k.max_batch_images);
  // Blocking sync so waits (ours and nvJPEG's) sleep instead of spinning a core; only possible
  // before the primary context is active, otherwise ignored.
  cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);
  cudaGetLastError();
  if (cudaSetDevice(k.device) != cudaSuccess) return fail("cudaSetDevice failed");
  if (cudaFree(nullptr) != cudaSuccess) return fail("CUDA context creation failed (GPU full?)");
  if (kernels_warmup() != cudaSuccess) return fail("loading the ingest kernels failed");
  // Highest stream priority for all ingest work: its kernels are tiny (~0.03 ms/MP) but gate the
  // pipeline (hardware output slots are reused only after finalize ran), so they should not queue
  // behind a co-running encoder's long kernels.  INGEST_STREAM_PRIORITY=0 disables (diagnostic).
  {
    int least = 0, greatest = 0;
    cudaDeviceGetStreamPriorityRange(&least, &greatest);
    const char *sp = getenv("INGEST_STREAM_PRIORITY");
    c->stream_prio = (sp && atoi(sp) == 0) ? least : greatest;
  }
  if (k.decoder == 0 && !k.legacy_coef) {
    unsigned cores = 0;
    // diagnostic knob (not ABI): INGEST_NVJPEG_FLAGS = nvjpegCreateEx flags
    const unsigned nflags = getenv("INGEST_NVJPEG_FLAGS") ? (unsigned)atoi(getenv("INGEST_NVJPEG_FLAGS")) : 0u;
    if (nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, nullptr, nullptr, nflags, &c->nvj) == NVJPEG_STATUS_SUCCESS &&
        nvjpegGetHardwareDecoderInfo(c->nvj, &c->hw_engines, &cores) == NVJPEG_STATUS_SUCCESS && c->hw_engines > 0 &&
        nvjpegJpegStateCreate(c->nvj, &c->nvj_state) == NVJPEG_STATUS_SUCCESS &&
        cudaEventCreateWithFlags(&c->hw_done, cudaEventBlockingSync | cudaEventDisableTiming) == cudaSuccess &&
        cudaEventCreateWithFlags(&c->hw_slots[0].free_ev, cudaEventBlockingSync | cudaEventDisableTiming) == cudaSuccess &&
        cudaEventCreateWithFlags(&c->hw_slots[1].free_ev, cudaEventBlockingSync | cudaEventDisableTiming) == cudaSuccess &&
        cudaStreamCreateWithPriority(&c->hw_stream, cudaStreamNonBlocking, c->stream_prio) == cudaSuccess) {
      c->hw = true;
      c->hw_legacy_out = getenv("INGEST_HW_OUT") && !strcmp(getenv("INGEST_HW_OUT"), "legacy");
      // INGEST_HW_MAX_SIDE (not ABI; default 6000, 0 = off): pre-allocated single-image decoder
      const char *ms = getenv("INGEST_HW_MAX_SIDE");
      const int side = ms ? atoi(ms) : 6000;
      if (side > 0) {
        nvjpegStatus_t st = nvjpegDecodeBatchedInitialize(c->nvj, c->nvj_state, 1, 1, NVJPEG_OUTPUT_YUV);
        if (st == NVJPEG_STATUS_SUCCESS)
          st = nvjpegDecodeBatchedPreAllocate(c->nvj, c->nvj_state, 1, side, side, NVJPEG_CSS_444, NVJPEG_OUTPUT_YUV);
        if (st == NVJPEG_STATUS_SUCCESS) {
          c->hw_prealloc = true;
          c->hw_max_side = side;
          c->nvj_bs = 1;
          // output slots up front (a cudaMalloc mid-run held the driver lock ~30-100 ms): native
          // planes of a full 4:4:4 batch or of the largest hardware-route image, whichever is
          // larger, so a slot never has to grow
          const size_t biggest = std::max<size_t>((size_t)k.batch_pixels, (size_t)side * side);
          const size_t cap = align_up(biggest * 3 + biggest * 3 / 8 + (8u << 20), 1 << 20);  // + block padding
          const char *ns = getenv("INGEST_HW_SLOTS");
          c->hw_nslots = (ns && atoi(ns) == 2) ? 2 : 1;
          for (int k2 = 0; k2 < c->hw_nslots; ++k2) {
            auto &sl = c->hw_slots[k2];
            if (cudaMalloc((void **)&sl.buf, cap) != cudaSuccess) return fail("cudaMalloc of the NVJPG output slot failed");
            sl.cap = cap;
          }
          c->hw_slot_bytes = c->hw_nslots * cap;
          c->device_bytes += c->hw_slot_bytes;
          c->device_peak = c->device_bytes;
        } else {
          fprintf(stderr, "[ingest] nvJPEG pre-allocation for %dx%d failed (status %d); batched calls with "
                          "per-call reallocation\n", side, side, (int)st);
          c->nvj_bs = -1;
        }
      }
    } else if (c->nvj) {
      nvjpegDestroy(c->nvj);
      c->nvj = nullptr;
    }
  }
  cudaMemPool_t mp;
  if (cudaDeviceGetDefaultMemPool(&mp, k.device) == cudaSuccess) {
    uint64_t thr = 1ull << 30;
    cudaMemPoolSetAttribute(mp, cudaMemPoolAttrReleaseThreshold, &thr);
  }
  if (!c->arena.init((size_t)k.pinned_bytes)) return fail("cudaHostAlloc of the pinned arena failed");
  for (int i = 0; i < k.max_inflight + 1; ++i) {
    cudaStream_t s;
    if (cudaStreamCreateWithPriority(&s, cudaStreamNonBlocking, c->stream_prio) != cudaSuccess)
      return fail("stream create failed");
    c->streams.push_back(s);
  }
  ingest_ctx *raw = c.release();
  for (int i = 0; i < k.n_workers; ++i) raw->workers.emplace_back(worker_main, raw);
  raw->batcher = std::thread(batcher_main, raw);
  return raw;
}

int ingest_submit(ingest_ctx *c, const char *const *paths, int n) {
  std::lock_guard<std::mutex> lk(c->m);
  if (c->submit_finished) return -1;
  for (int i = 0; i < n; ++i) c->paths.emplace_back(paths[i]);
  c->cv_jobs.notify_all();
  return (int)c->paths.size();
}

void ingest_finish_submit(ingest_ctx *c) {
  std::lock_guard<std::mutex> lk(c->m);
  c->submit_finished = true;
  c->cv_jobs.notify_all();
  c->cv_ready.notify_all();
}

int ingest_next_batch(ingest_ctx *c, ingest_batch *out, int timeout_ms) {
  cudaSetDevice(c->cfg.device);
  std::unique_ptr<Batch> b;
  {
    std::unique_lock<std::mutex> lk(c->m);
    auto pred = [&] { return !c->error.empty() || !c->done.empty() || c->batcher_finished; };
    if (timeout_ms < 0) c->cv_done.wait(lk, pred);
    else if (!c->cv_done.wait_for(lk, std::chrono::milliseconds(timeout_ms), pred)) return -1;
    if (!c->error.empty()) return -2;
    if (c->done.empty()) return 0;
    b = std::move(c->done.front());
    c->done.pop_front();
  }
  if (cudaEventSynchronize(b->ev2) != cudaSuccess) {
    set_error(c, "kernel failure");
    return -2;
  }
  float h2d = 0, gpu = 0, hwt = 0;
  cudaEventElapsedTime(&h2d, b->ev0, b->ev1);
  cudaEventElapsedTime(&gpu, b->ev1, b->ev2);
  cudaEventElapsedTime(&hwt, b->evh0, b->evh1);
  std::memset(out, 0, sizeof(*out));
  out->batch_id = b->id;
  out->n_images = (int32_t)b->metas.size();
  out->cls = b->cls;
  out->pool = b->d_pool;
  out->pool_bytes = (int64_t)b->pool_bytes;
  out->native = b->d_native;
  out->native_bytes = b->d_native ? (int64_t)b->native_bytes : 0;
  out->images = b->metas.data();
  out->stream = b->stream;
  out->h2d_ms = h2d;
  out->gpu_ms = gpu;
  out->hw_ms = hwt;
  out->n_hw = b->n_hw;
  out->coef_bytes = (int64_t)b->coef_bytes;
  out->total_pixels = b->pixels;
  std::lock_guard<std::mutex> lk(c->m);
  c->gpu_ms += gpu;
  c->h2d_ms += h2d;
  c->hw_ms += hwt;
  c->held[b->id] = std::move(b);
  return 1;
}

void ingest_release_batch(ingest_ctx *c, int64_t batch_id, void *stream) {
  cudaSetDevice(c->cfg.device);
  std::unique_ptr<Batch> b;
  {
    std::lock_guard<std::mutex> lk(c->m);
    auto it = c->held.find(batch_id);
    if (it == c->held.end()) return;
    b = std::move(it->second);
    c->held.erase(it);
  }
  cudaStream_t s = stream ? (cudaStream_t)stream : b->stream;
  size_t freed = b->pool_bytes;
  cudaFreeAsync(b->d_pool, s);
  if (!b->scratch_freed) {
    cudaFreeAsync(b->d_scratch, s);
    freed += b->scratch_bytes;
  }
  cudaEventDestroy(b->ev0);
  cudaEventDestroy(b->ev1);
  cudaEventDestroy(b->ev2);
  cudaEventDestroy(b->evh0);
  cudaEventDestroy(b->evh1);
  std::lock_guard<std::mutex> lk(c->m);
  c->device_bytes -= freed;
  --c->inflight;
  ++c->n_released;
  c->cv_inflight.notify_all();
}

int ingest_get_fallbacks(ingest_ctx *c, ingest_fallback *out, int cap) {
  std::lock_guard<std::mutex> lk(c->m);
  int n = (int)c->fallbacks.size();
  for (int i = 0; i < std::min(n, cap); ++i) out[i] = c->fallbacks[i];
  return n;
}

void ingest_get_stats(ingest_ctx *c, ingest_stats *o) {
  std::lock_guard<std::mutex> lk(c->m);
  std::memset(o, 0, sizeof(*o));
  o->submitted = (int64_t)c->paths.size();
  o->decoded = c->n_decoded;
  o->fallbacks = (int64_t)c->fallbacks.size();
  o->batches = c->n_batches;
  o->released = c->n_released;
  o->worker_cpu_ms_read = c->cpu_read;
  o->worker_cpu_ms_decode = c->cpu_decode;
  o->worker_cpu_ms_copy = c->cpu_copy;
  o->gpu_ms = c->gpu_ms;
  o->h2d_ms = c->h2d_ms;
  o->pixels = c->pixels;
  o->coef_bytes = c->coef_bytes;
  o->pinned_peak = (int64_t)c->arena.peak();
  o->device_peak = (int64_t)c->device_peak;
  o->n_hw = c->n_hw;
  o->n_hw_failed = c->n_hw_failed;
  o->hw_ms = c->hw_ms;
  o->batcher_cpu_ms = c->batcher_cpu;
  o->hw_available = c->hw ? 1 : 0;
  o->hw_engines = (int32_t)c->hw_engines;
}

int ingest_last_error(ingest_ctx *c, char *buf, int len) {
  std::lock_guard<std::mutex> lk(c->m);
  if (buf && len > 0) snprintf(buf, len, "%s", c->error.c_str());
  return (int)c->error.size();
}

void ingest_destroy(ingest_ctx *c) {
  if (!c) return;
  {
    std::lock_guard<std::mutex> lk(c->m);
    c->stopping = true;
    c->cv_jobs.notify_all();
    c->cv_ready.notify_all();
    c->cv_inflight.notify_all();
    c->cv_done.notify_all();
  }
  c->arena.close();
  for (auto &t : c->workers) t.join();
  if (c->batcher.joinable()) c->batcher.join();
  cudaSetDevice(c->cfg.device);
  cudaDeviceSynchronize();
  for (auto &kv : c->held) {
    cudaFree(kv.second->d_pool);
    if (!kv.second->scratch_freed) cudaFree(kv.second->d_scratch);
  }
  for (auto &b : c->done) {
    cudaFree(b->d_pool);
    if (!b->scratch_freed) cudaFree(b->d_scratch);
  }
  for (auto s : c->streams) cudaStreamDestroy(s);
  if (c->hw_done) cudaEventDestroy(c->hw_done);
  for (auto &sl : c->hw_slots) {
    if (sl.free_ev) cudaEventDestroy(sl.free_ev);
    if (sl.buf) cudaFree(sl.buf);
  }
  for (auto *p : c->hw_retired) cudaFree(p);
  if (c->hw_stream) cudaStreamDestroy(c->hw_stream);
  if (c->nvj_state) nvjpegJpegStateDestroy(c->nvj_state);
  if (c->nvj) nvjpegDestroy(c->nvj);
  delete c;
}

}  // extern "C"
