// input decoding helpers (JPEG via the ingest module's GPU ingest; WebP/PNG-in-disguise via a CPU path).
#ifndef GP_IMAGEIO_H
#define GP_IMAGEIO_H

#include <stdint.h>
#include <string>
#include <vector>

enum GpFileKind { GP_FILE_UNKNOWN = 0, GP_FILE_JPEG, GP_FILE_WEBP, GP_FILE_PNG };
GpFileKind gp_sniff(const std::string &path);

// CPU fallback: decode WebP/PNG to RGB, convert to full-range BT.601 YCbCr (JPEG equations), 2x2 box
// to 4:2:0 (yuv444: no chroma downsampling), pad to multiples of 64 by edge replication.  Planes:
// Y aw x ah, U/V aw/2 x ah/2 (yuv444: aw x ah).
// bitDepth 10: 10-bit planes (p16) from the RGB samples at full precision (8-bit PNG / WebP, or 16-bit PNG
// samples): YCbCr in double from the normalised RGB (v / 255 or v / 65535), 4:2:0 chroma averaged before the
// rounding, Y = round(y * 1023), Cb/Cr = round(c * 1023 + 512) (libavif's full-range unorm mapping).  16-bit PNGs
// are read with their 16-bit samples (alpha is dropped).
struct GpHostPlanes {
    int W = 0, H = 0, aw = 0, ah = 0, mono = 0;
    int yuv444 = 0;              // 1: U/V are full size
    int bitDepth = 8;            // 10: the planes are p16 (uint16_t samples), p[] is empty
    std::vector<uint8_t> p[3];
    std::vector<uint16_t> p16[3];
    std::vector<uint8_t> icc;
    int cw() const { return yuv444 ? aw : aw / 2; }   // chroma plane width / height
    int ch() const { return yuv444 ? ah : ah / 2; }
    const void *data(int i) const { return bitDepth == 10 ? (const void *)p16[i].data() : (const void *)p[i].data(); }
    size_t bytes(int i) const { return bitDepth == 10 ? p16[i].size() * 2 : p[i].size(); }
};
bool gp_decode_cpu(const std::string &path, GpHostPlanes &out, std::string &err, bool yuv444 = false, int bitDepth = 8);

bool gp_read_file(const std::string &path, std::vector<uint8_t> &data);
bool gp_write_file(const std::string &path, const uint8_t *d, size_t n);

#endif
