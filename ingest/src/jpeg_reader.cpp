// CPU side of the ingest: libjpeg-turbo entropy decode to quantized DCT coefficients,
// plus cheap header metadata (EXIF orientation, ICC, IJG quality estimate, tiling).
#include "jpeg_reader.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "ingest_api.h"
#include <jpegint.h>

namespace ingest {

// ---------------------------------------------------------------- libjpeg error plumbing
void reader_error_exit(j_common_ptr cinfo) {
  auto *e = reinterpret_cast<CoefReader::ErrMgr *>(cinfo->err);
  char buf[JMSG_LENGTH_MAX];
  (*cinfo->err->format_message)(cinfo, buf);
  e->owner->err_msg_ = buf;
  longjmp(e->jb, 1);
}
static void quiet_output_message(j_common_ptr) {}

CoefReader::CoefReader() {
  std::memset(&cinfo_, 0, sizeof(cinfo_));
  cinfo_.err = jpeg_std_error(&err_.pub);
  err_.pub.error_exit = reader_error_exit;
  err_.pub.output_message = quiet_output_message;
  err_.owner = this;
  jpeg_create_decompress(&cinfo_);
}

CoefReader::~CoefReader() { jpeg_destroy_decompress(&cinfo_); }

void CoefReader::reset() {
  if (active_) jpeg_abort_decompress(&cinfo_);
  active_ = false;
  coefs_ = nullptr;
}

// ---------------------------------------------------------------- helpers
static const uint16_t kStdLuma[64] = {
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56, 14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99};
static const uint16_t kStdChroma[64] = {
    17, 18, 24, 47, 99, 99, 99, 99, 18, 21, 26, 66, 99, 99, 99, 99,
    24, 26, 56, 99, 99, 99, 99, 99, 47, 66, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99,
    99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99, 99};

static inline int ijg_entry(int basic, int q) {
  int scale = q < 50 ? 5000 / q : 200 - 2 * q;
  long t = ((long)basic * scale + 50) / 100;
  if (t <= 0) t = 1;
  if (t > 255) t = 255;  // force_baseline (what cjpeg/libjpeg/most encoders do)
  return (int)t;
}

// Nearest IJG quality (1..100) by L1 distance over luma (+chroma) tables.
int estimate_ijg_quality(const uint16_t *luma, const uint16_t *chroma, int *exact) {
  long best = -1;
  int bestq = 0;
  for (int q = 1; q <= 100; ++q) {
    long err = 0;
    for (int i = 0; i < 64; ++i) err += std::labs((long)luma[i] - ijg_entry(kStdLuma[i], q));
    if (chroma)
      for (int i = 0; i < 64; ++i) err += std::labs((long)chroma[i] - ijg_entry(kStdChroma[i], q));
    if (best < 0 || err < best) {  // strict: ties keep the lower q (conservative)
      best = err;
      bestq = q;
    }
  }
  if (exact) *exact = (best == 0);
  return bestq;
}

static inline uint16_t rd16(const uint8_t *p, bool le) {
  return le ? (uint16_t)(p[0] | (p[1] << 8)) : (uint16_t)((p[0] << 8) | p[1]);
}
static inline uint32_t rd32(const uint8_t *p, bool le) {
  return le ? (uint32_t)p[0] | ((uint32_t)p[1] << 8) | ((uint32_t)p[2] << 16) | ((uint32_t)p[3] << 24)
            : ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) | ((uint32_t)p[2] << 8) | (uint32_t)p[3];
}

// tiff = APP1 payload after "Exif\0\0".  Returns orientation 1..8 (1 if absent/invalid).
int parse_exif_orientation(const uint8_t *tiff, size_t n) {
  if (n < 8) return 1;
  bool le;
  if (tiff[0] == 'I' && tiff[1] == 'I') le = true;
  else if (tiff[0] == 'M' && tiff[1] == 'M') le = false;
  else return 1;
  if (rd16(tiff + 2, le) != 42) return 1;
  uint32_t ifd = rd32(tiff + 4, le);
  if (ifd + 2 > n) return 1;
  unsigned cnt = rd16(tiff + ifd, le);
  for (unsigned i = 0; i < cnt; ++i) {
    size_t e = ifd + 2 + 12 * (size_t)i;
    if (e + 12 > n) break;
    if (rd16(tiff + e, le) == 0x0112) {
      unsigned type = rd16(tiff + e + 2, le);
      unsigned v = type == 3 ? rd16(tiff + e + 8, le) : (type == 4 ? rd32(tiff + e + 8, le) : 1);
      return (v >= 1 && v <= 8) ? (int)v : 1;
    }
  }
  return 1;
}

int classify_sub(const JpegInfo &in) {
  if (in.ncomp == 1) return INGEST_SUB_GRAY;
  if (in.ncomp != 3) return INGEST_SUB_OTHER;
  if (in.h[1] != in.h[2] || in.v[1] != in.v[2] || in.h[0] != in.hmax || in.v[0] != in.vmax)
    return INGEST_SUB_OTHER;
  int rh = in.h[0] / in.h[1], rv = in.v[0] / in.v[1];
  if (in.h[0] % in.h[1] || in.v[0] % in.v[1]) return INGEST_SUB_OTHER;
  if (rh == 2 && rv == 2) return INGEST_SUB_420;
  if (rh == 1 && rv == 1) return INGEST_SUB_444;
  if (rh == 2 && rv == 1) return INGEST_SUB_422;
  if (rh == 1 && rv == 2) return INGEST_SUB_440;
  if (rh == 4 && rv == 1) return INGEST_SUB_411;
  return INGEST_SUB_OTHER;
}

// Minimal uniform AV1 tiling for 64x64 superblocks (spec 5.9.15 / 7.x limits).
void tile_layout(int w, int h, int *cols_log2, int *rows_log2) {
  auto tile_log2 = [](int blk, int target) { int k = 0; while ((blk << k) < target) ++k; return k; };
  const int sbCols = (w + 63) >> 6, sbRows = (h + 63) >> 6;
  const int maxTileWidthSb = 4096 >> 6, maxTileAreaSb = (4096 * 2304) >> 12;
  int minLog2TileCols = tile_log2(maxTileWidthSb, sbCols);
  int minLog2Tiles = std::max(minLog2TileCols, tile_log2(maxTileAreaSb, sbRows * sbCols));
  int c = minLog2TileCols;
  int r = std::max(minLog2Tiles - c, 0);
  for (;;) {  // uniform tiles may still exceed the area limit when sizes round up
    int tw = (sbCols + (1 << c) - 1) >> c, th = (sbRows + (1 << r) - 1) >> r;
    if ((long)tw * th <= maxTileAreaSb || (1 << r) >= sbRows) break;
    ++r;
  }
  *cols_log2 = c;
  *rows_log2 = r;
}

// ---------------------------------------------------------------- the reader
void CoefReader::grab_qt(JpegInfo &info) {
  for (int c = 0; c < std::min(info.ncomp, 4); ++c) {
    const JQUANT_TBL *t = cinfo_.comp_info[c].quant_table;
    if (!t) t = cinfo_.quant_tbl_ptrs[info.qt_slot[c]];
    if (t) for (int i = 0; i < 64; ++i) info.qt[c][i] = t->quantval[i];
  }
  const uint16_t *chroma = info.ncomp >= 3 ? info.qt[1] : nullptr;
  if (info.qt[0][0]) info.quality = estimate_ijg_quality(info.qt[0], chroma, &info.quality_exact);
}

// Must be called inside the caller's setjmp scope.
int CoefReader::header(const uint8_t *data, size_t size, double max_mp, bool want_markers, JpegInfo &info) {
  active_ = true;
  err_.pub.num_warnings = 0;
  jpeg_mem_src(&cinfo_, data, (unsigned long)size);
  jpeg_save_markers(&cinfo_, JPEG_APP0 + 1, want_markers ? 0xFFFF : 0);
  jpeg_save_markers(&cinfo_, JPEG_APP0 + 2, want_markers ? 0xFFFF : 0);
  jpeg_read_header(&cinfo_, TRUE);

  info.width = (int)cinfo_.image_width;
  info.height = (int)cinfo_.image_height;
  info.ncomp = cinfo_.num_components;
  info.color = cinfo_.jpeg_color_space;
  info.precision = cinfo_.data_precision;
  info.progressive = cinfo_.progressive_mode ? 1 : 0;
  info.arithmetic = cinfo_.arith_code ? 1 : 0;
  info.restart_interval = (int)cinfo_.restart_interval;
  info.hmax = cinfo_.max_h_samp_factor;
  info.vmax = cinfo_.max_v_samp_factor;
  for (int c = 0; c < std::min(info.ncomp, 4); ++c) {
    info.h[c] = cinfo_.comp_info[c].h_samp_factor;
    info.v[c] = cinfo_.comp_info[c].v_samp_factor;
    info.qt_slot[c] = cinfo_.comp_info[c].quant_tbl_no;
    // width_in_blocks is only set by jpeg_start_decompress/read_coefficients; compute it here.
    long dw = ((long)info.width * info.h[c] + info.hmax - 1) / info.hmax;
    long dh = ((long)info.height * info.v[c] + info.vmax - 1) / info.vmax;
    info.wb[c] = (int)((dw + 7) / 8);
    info.hb[c] = (int)((dh + 7) / 8);
  }
  info.sub = classify_sub(info);

  // markers: EXIF orientation, ICC (APP2 "ICC_PROFILE\0" seq cnt data)
  if (want_markers) {
    std::vector<std::pair<int, const jpeg_marker_struct *>> icc_parts;
    for (jpeg_saved_marker_ptr m = cinfo_.marker_list; m; m = m->next) {
      if (m->marker == JPEG_APP0 + 1 && m->data_length > 6 && !std::memcmp(m->data, "Exif\0\0", 6) &&
          info.exif.empty()) {
        info.exif.assign(m->data + 6, m->data + m->data_length);
        info.orientation = parse_exif_orientation(info.exif.data(), info.exif.size());
      } else if (m->marker == JPEG_APP0 + 2 && m->data_length > 14 &&
                 !std::memcmp(m->data, "ICC_PROFILE\0", 12)) {
        icc_parts.emplace_back(m->data[12], m);
      }
    }
    if (!icc_parts.empty()) {
      std::stable_sort(icc_parts.begin(), icc_parts.end(),
                       [](auto &a, auto &b) { return a.first < b.first; });
      for (auto &p : icc_parts) info.icc.insert(info.icc.end(), p.second->data + 14,
                                                p.second->data + p.second->data_length);
    }
  }
  grab_qt(info);

  int fb = INGEST_FB_NONE;
  if (info.precision != 8) fb = INGEST_FB_PRECISION;
  else if (info.color == JCS_CMYK || info.color == JCS_YCCK) fb = INGEST_FB_CMYK;
  else if (!((info.ncomp == 3 && info.color == JCS_YCbCr) || (info.ncomp == 1 && info.color == JCS_GRAYSCALE)))
    fb = INGEST_FB_COLORSPACE;
  else if ((double)info.width * info.height > max_mp * 1e6) fb = INGEST_FB_GIANT;
  info.warnings = (int)err_.pub.num_warnings;
  return fb;
}

int CoefReader::read(const uint8_t *data, size_t size, double max_mp, bool decode, bool want_markers,
                     JpegInfo &info) {
  reset();
  err_msg_.clear();
  info = JpegInfo();
  if (setjmp(err_.jb)) {
    reset();
    return INGEST_FB_DECODE_ERROR;
  }
  int fb = header(data, size, max_mp, want_markers, info);
  if (fb || !decode) {
    if (fb) reset();
    return fb;
  }
  coefs_ = jpeg_read_coefficients(&cinfo_);
  if (!coefs_) {
    reset();
    return INGEST_FB_DECODE_ERROR;
  }
  for (int c = 0; c < info.ncomp; ++c) {  // authoritative values from the decoder
    info.wb[c] = (int)cinfo_.comp_info[c].width_in_blocks;
    info.hb[c] = (int)cinfo_.comp_info[c].height_in_blocks;
  }
  grab_qt(info);
  info.warnings = (int)err_.pub.num_warnings;
  return INGEST_FB_NONE;
}

namespace {
struct HookCtx {
  int16_t *dst[4];
  int wb[4];
  JSAMPROW *rowbase[4];
};
// Replaces jpeg_idct_islow: store the block instead of transforming it.
void coef_hook(j_decompress_ptr cinfo, jpeg_component_info *comp, JCOEFPTR coef, JSAMPARRAY out,
               JDIMENSION col) {
  const HookCtx *h = static_cast<const HookCtx *>(cinfo->client_data);
  const int ci = comp->component_index;
  const size_t brow = (size_t)cinfo->output_iMCU_row * comp->v_samp_factor + (size_t)((out - h->rowbase[ci]) >> 3);
  std::memcpy(h->dst[ci] + (brow * h->wb[ci] + (col >> 3)) * 64, coef, 64 * sizeof(JCOEF));
}
}  // namespace

int CoefReader::read_fast(const uint8_t *data, size_t size, double max_mp, JpegInfo &info,
                          const std::function<int16_t *(size_t)> &alloc) {
  reset();
  err_msg_.clear();
  info = JpegInfo();
  HookCtx hook{};
  std::vector<JSAMPROW> rows;
  std::vector<uint8_t> dummy;
  if (setjmp(err_.jb)) {
    cinfo_.client_data = nullptr;
    reset();
    return INGEST_FB_DECODE_ERROR;
  }
  int fb = header(data, size, max_mp, true, info);
  if (fb) {
    reset();
    return fb;
  }
  cinfo_.raw_data_out = TRUE;
  cinfo_.dct_method = JDCT_ISLOW;
  cinfo_.do_fancy_upsampling = FALSE;
  jpeg_start_decompress(&cinfo_);  // progressive / multi-scan: absorbs the whole file here
  int maxwb = 0;
  for (int c = 0; c < info.ncomp; ++c) {
    info.wb[c] = (int)cinfo_.comp_info[c].width_in_blocks;
    info.hb[c] = (int)cinfo_.comp_info[c].height_in_blocks;
    maxwb = std::max(maxwb, info.wb[c]);
  }
  grab_qt(info);  // final tables (multi-scan files may define them late)
  int16_t *dst = alloc(info.total_blocks() * 64 * sizeof(int16_t));
  if (!dst) {
    reset();
    return INGEST_FB_TOO_BIG;
  }
  // one shared dummy row; the hook never writes samples
  dummy.assign((size_t)maxwb * 8 + 64, 0);
  int nrows_total = 0;
  for (int c = 0; c < info.ncomp; ++c) nrows_total += cinfo_.comp_info[c].v_samp_factor * 8;
  rows.assign(nrows_total, dummy.data());
  JSAMPARRAY comps[4];
  int16_t *p = dst;
  for (int c = 0, r = 0; c < info.ncomp; ++c) {
    hook.dst[c] = p;
    hook.wb[c] = info.wb[c];
    hook.rowbase[c] = rows.data() + r;
    comps[c] = rows.data() + r;
    r += cinfo_.comp_info[c].v_samp_factor * 8;
    p += (size_t)info.wb[c] * info.hb[c] * 64;
    cinfo_.idct->inverse_DCT[c] = coef_hook;
  }
  cinfo_.client_data = &hook;
  const JDIMENSION lines = (JDIMENSION)cinfo_.max_v_samp_factor * 8;
  while (cinfo_.output_scanline < cinfo_.output_height)
    if (!jpeg_read_raw_data(&cinfo_, comps, lines)) break;
  cinfo_.client_data = nullptr;
  info.warnings = (int)err_.pub.num_warnings;
  reset();
  return INGEST_FB_NONE;
}

void CoefReader::copy_coefs(int16_t *dst) const {
  auto *ci = const_cast<jpeg_decompress_struct *>(&cinfo_);
  for (int c = 0; c < ci->num_components; ++c) {
    const jpeg_component_info *comp = &ci->comp_info[c];
    const size_t row_bytes = (size_t)comp->width_in_blocks * sizeof(JBLOCK);
    for (JDIMENSION r = 0; r < comp->height_in_blocks; r += comp->v_samp_factor) {
      JDIMENSION nr = std::min<JDIMENSION>(comp->v_samp_factor, comp->height_in_blocks - r);
      JBLOCKARRAY rows = (*ci->mem->access_virt_barray)((j_common_ptr)ci, coefs_[c], r, nr, FALSE);
      for (JDIMENSION k = 0; k < nr; ++k) {
        std::memcpy(dst, rows[k], row_bytes);
        dst += (size_t)comp->width_in_blocks * 64;
      }
    }
  }
}

}  // namespace ingest

// ---------------------------------------------------------------- C entry points
using namespace ingest;

extern "C" int ingest_estimate_quality(const uint16_t *luma, const uint16_t *chroma, int *exact) {
  return estimate_ijg_quality(luma, chroma, exact);
}

static void fill_meta(const JpegInfo &in, ingest_image *o) {
  std::memset(o, 0, sizeof(*o));
  o->width = in.width;
  o->height = in.height;
  o->mono = in.ncomp == 1;
  o->src_sub = in.sub;
  o->orientation = in.orientation;
  o->quality = in.quality;
  o->quality_exact = in.quality_exact;
  o->progressive = in.progressive;
  o->arithmetic = in.arithmetic;
  o->restart_interval = in.restart_interval;
  o->warnings = in.warnings;
  o->icc_size = (int32_t)in.icc.size();
  o->exif_size = (int32_t)in.exif.size();
  o->ncomp = in.ncomp;
  for (int c = 0; c < std::min(in.ncomp, 3); ++c) {
    o->h_samp[c] = in.h[c];
    o->v_samp[c] = in.v[c];
    o->nat_w[c] = in.wb[c] * 8;
    o->nat_h[c] = in.hb[c] * 8;
    std::memcpy(o->qt[c], in.qt[c], sizeof(o->qt[c]));
    o->nat_off[c] = -1;
  }
  tile_layout(in.width, in.height, &o->tile_cols_log2, &o->tile_rows_log2);
  o->tiled = (o->tile_cols_log2 + o->tile_rows_log2) > 0;
  double mp = (double)in.width * in.height / 1e6;
  o->bucket = o->tiled ? INGEST_BUCKET_HUGE
              : mp < 0.3 ? INGEST_BUCKET_TINY
              : mp < 1.0 ? INGEST_BUCKET_SMALL
              : mp < 4.0 ? INGEST_BUCKET_TYPICAL
                         : INGEST_BUCKET_LARGE;
}

// exported for the pipeline
void ingest_fill_meta(const JpegInfo &in, ingest_image *o) { fill_meta(in, o); }

extern "C" int ingest_probe(const uint8_t *jpg, int64_t n, double max_mp, ingest_image *out) {
  CoefReader r;
  JpegInfo info;
  int fb = r.read(jpg, (size_t)n, max_mp, false, true, info);
  fill_meta(info, out);
  r.reset();
  return fb;
}

extern "C" int64_t ingest_ref_decode_raw(const uint8_t *jpg, int64_t n, uint8_t *out, int64_t cap,
                                         int32_t *info) {
  jpeg_decompress_struct cinfo;
  CoefReader::ErrMgr err;
  std::memset(&cinfo, 0, sizeof(cinfo));
  cinfo.err = jpeg_std_error(&err.pub);
  err.pub.output_message = quiet_output_message;
  struct Local {
    static void exit_fn(j_common_ptr c) {
      if (getenv("INGEST_DEBUG")) {
        char buf[JMSG_LENGTH_MAX];
        (*c->err->format_message)(c, buf);
        fprintf(stderr, "ingest_ref_decode_raw: %s\n", buf);
      }
      longjmp(reinterpret_cast<CoefReader::ErrMgr *>(c->err)->jb, 1);
    }
  };
  err.pub.error_exit = Local::exit_fn;
  std::vector<std::vector<uint8_t>> planes;
  std::vector<int> pw, ph, alloc_h;
  std::vector<std::vector<JSAMPROW>> rowptrs;
  std::vector<JSAMPARRAY> comps;
  if (setjmp(err.jb)) {
    jpeg_destroy_decompress(&cinfo);
    return -1;
  }
  jpeg_create_decompress(&cinfo);
  jpeg_mem_src(&cinfo, jpg, (unsigned long)n);
  jpeg_read_header(&cinfo, TRUE);
  cinfo.raw_data_out = TRUE;
  cinfo.dct_method = JDCT_ISLOW;
  cinfo.do_fancy_upsampling = FALSE;
  jpeg_start_decompress(&cinfo);
  const int nc = cinfo.num_components;
  const int vmax = cinfo.max_v_samp_factor;
  int64_t need = 0;
  pw.assign(nc, 0);
  ph.assign(nc, 0);
  alloc_h.assign(nc, 0);
  for (int c = 0; c < nc; ++c) {
    auto *cp = &cinfo.comp_info[c];
    pw[c] = (int)cp->width_in_blocks * 8;
    ph[c] = (int)cp->height_in_blocks * 8;
    int imcu_rows = (int)((cinfo.output_height + vmax * 8 - 1) / (vmax * 8));
    alloc_h[c] = imcu_rows * cp->v_samp_factor * 8;
    // buffers padded to whole iMCUs (dummy blocks); libjpeg may write up to MCU width
    planes.emplace_back((size_t)(pw[c] + 64 * 8) * alloc_h[c]);
    need += (int64_t)pw[c] * ph[c];
  }
  if (need > cap) {
    jpeg_destroy_decompress(&cinfo);
    return -need;
  }
  rowptrs.assign(nc, {});
  comps.assign(nc, nullptr);
  int row = 0;
  while (cinfo.output_scanline < cinfo.output_height) {
    for (int c = 0; c < nc; ++c) {
      int rows = cinfo.comp_info[c].v_samp_factor * 8;
      rowptrs[c].resize(rows);
      for (int k = 0; k < rows; ++k) {
        int y = std::min(row / vmax * cinfo.comp_info[c].v_samp_factor + k, alloc_h[c] - 1);
        rowptrs[c][k] = planes[c].data() + (size_t)y * (pw[c] + 64 * 8);
      }
      comps[c] = rowptrs[c].data();
    }
    JDIMENSION got = jpeg_read_raw_data(&cinfo, comps.data(), vmax * 8);
    if (!got) break;
    row += vmax * 8;
  }
  // No jpeg_finish_decompress: trailing garbage / bogus markers after the last scan would make it
  // fail although every sample has been produced (same policy as the ingest's own reader).
  jpeg_abort_decompress(&cinfo);
  info[0] = nc;
  info[1] = (int32_t)cinfo.image_width;
  info[2] = (int32_t)cinfo.image_height;
  int64_t off = 0;
  for (int c = 0; c < nc; ++c) {
    info[3 + 4 * c] = pw[c];
    info[4 + 4 * c] = ph[c];
    info[5 + 4 * c] = cinfo.comp_info[c].h_samp_factor;
    info[6 + 4 * c] = cinfo.comp_info[c].v_samp_factor;
    for (int y = 0; y < ph[c]; ++y)
      std::memcpy(out + off + (int64_t)y * pw[c], planes[c].data() + (size_t)y * (pw[c] + 64 * 8), pw[c]);
    off += (int64_t)pw[c] * ph[c];
  }
  jpeg_destroy_decompress(&cinfo);
  return off;
}
