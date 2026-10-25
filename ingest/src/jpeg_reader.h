// CPU side of the ingest: libjpeg-turbo entropy decode to quantized DCT coefficients.
#pragma once
#include <cstddef>
#include <cstdint>
#include <csetjmp>
#include <functional>
#include <string>
#include <vector>

#include <cstdio>
#include <jpeglib.h>

namespace ingest {

struct JpegInfo {
  int width = 0, height = 0, ncomp = 0;
  int h[4] = {0}, v[4] = {0}, hmax = 1, vmax = 1;
  int wb[4] = {0}, hb[4] = {0};          // width_in_blocks / height_in_blocks per component
  uint16_t qt[4][64] = {{0}};            // dequant table per component, natural order
  int qt_slot[4] = {0};
  int color = 0;                         // J_COLOR_SPACE of the file
  int precision = 8;
  int progressive = 0, arithmetic = 0, restart_interval = 0;
  int orientation = 1;
  int quality = 0, quality_exact = 0;
  int sub = 0;                           // INGEST_SUB_*
  int warnings = 0;
  std::vector<uint8_t> icc, exif;
  size_t total_blocks() const {
    size_t n = 0;
    for (int c = 0; c < ncomp; ++c) n += (size_t)wb[c] * hb[c];
    return n;
  }
};

// Reusable decompressor (one per worker thread).
class CoefReader {
 public:
  CoefReader();
  ~CoefReader();
  CoefReader(const CoefReader &) = delete;
  CoefReader &operator=(const CoefReader &) = delete;

  // Parse header (+ markers) and, if decode=true and the image is GPU-eligible, run the entropy
  // decoder (jpeg_read_coefficients).  Returns an INGEST_FB_* code (0 = ok).
  int read(const uint8_t *data, size_t size, double max_mp, bool decode, bool want_markers,
           JpegInfo &info);
  // Copy coefficients (natural order int16, comp after comp, row-major blocks) to dst.
  void copy_coefs(int16_t *dst) const;

  // Fast path (default in the pipeline): runs libjpeg-turbo's normal raw-data decompression with
  // the IDCT method pointers replaced by a hook that stores each block's (quantized) coefficients
  // straight into dst = alloc(bytes) -- comp after comp, row-major blocks, natural order int16.
  // No whole-image coefficient array for baseline files (MCU buffer stays in L1), and progressive
  // files go through libjpeg's own output pass (incl. block smoothing for incomplete scans), so the
  // GPU sees exactly the coefficients libjpeg-turbo would have fed to its IDCT.
  // alloc may block; returning nullptr makes read_fast return INGEST_FB_TOO_BIG.
  int read_fast(const uint8_t *data, size_t size, double max_mp, JpegInfo &info,
                const std::function<int16_t *(size_t)> &alloc);
  // Release decoder memory for the current image (keeps the object reusable).
  void reset();
  const std::string &last_error() const { return err_msg_; }

  struct ErrMgr {
    jpeg_error_mgr pub;
    jmp_buf jb;
    CoefReader *owner;
  };

 private:
  jpeg_decompress_struct cinfo_;
  ErrMgr err_;
  jvirt_barray_ptr *coefs_ = nullptr;
  int header(const uint8_t *data, size_t size, double max_mp, bool want_markers, JpegInfo &info);
  void grab_qt(JpegInfo &info);
  bool active_ = false;
  std::string err_msg_;
  friend void reader_error_exit(j_common_ptr);
};

int estimate_ijg_quality(const uint16_t *luma, const uint16_t *chroma, int *exact);
int parse_exif_orientation(const uint8_t *tiff, size_t n);
int classify_sub(const JpegInfo &info);
void tile_layout(int w, int h, int *cols_log2, int *rows_log2);

}  // namespace ingest
