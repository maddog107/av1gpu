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
// are read with their 16-bit samples.
//
// Alpha (keepAlpha; av1gpu's default, --no-alpha turns it off).  An input with an alpha channel (PNG colour types 4 / 6,
// a tRNS chunk, WebP with alpha) is read with its alpha samples and straight (not premultiplied) colour: the colour of
// every pixel is what the PNG / WebP stores, also under fully transparent pixels (no premultiplication, no zeroing,
// as libavif does).  Where the alpha is opaque this is the same colour as without keepAlpha (libpng's RGB and RGBA
// outputs agree there), so an image whose alpha is opaque everywhere gives the same planes, and no alpha plane.
// Otherwise (some alpha below the maximum) out.alpha = 1 and a / a16 hold the alpha plane, aw x ah, edge-replicated
// like Y, at the planes' bit depth (full range): 8-bit sources as stored (8-bit output) or rescaled to 10 bits, 16-bit
// PNG alpha rounded to 8 / 10 bits, both with libavif's (int)(0.5f + v / srcMax * dstMax) (avifenc -d 8 / -d 10 does
// the same; libpng's 16 -> 8-bit alpha is identical).  *note gets a remark when 16-bit alpha loses precision.
// Without keepAlpha the alpha channel is not coded, as before: PNG colour is composited onto black by libpng, WebP
// colour is its stored RGB; *note says so.
struct GpHostPlanes {
    int W = 0, H = 0, aw = 0, ah = 0, mono = 0;
    int yuv444 = 0;              // 1: U/V are full size
    int bitDepth = 8;            // 10: the planes are p16 (uint16_t samples), p[] is empty
    std::vector<uint8_t> p[3];
    std::vector<uint16_t> p16[3];
    std::vector<uint8_t> icc;
    int alpha = 0;               // 1: a (8-bit) / a16 (10-bit) hold the alpha plane (aw x ah, bitDepth samples, full range)
    std::vector<uint8_t> a;
    std::vector<uint16_t> a16;
    int cw() const { return yuv444 ? aw : aw / 2; }   // chroma plane width / height
    int ch() const { return yuv444 ? ah : ah / 2; }
    const void *data(int i) const { return bitDepth == 10 ? (const void *)p16[i].data() : (const void *)p[i].data(); }
    size_t bytes(int i) const { return bitDepth == 10 ? p16[i].size() * 2 : p[i].size(); }
    const void *adata() const { return bitDepth == 10 ? (const void *)a16.data() : (const void *)a.data(); }
    size_t abytes() const { return !alpha ? 0 : bitDepth == 10 ? a16.size() * 2 : a.size(); }
};
bool gp_decode_cpu(const std::string &path, GpHostPlanes &out, std::string &err, bool yuv444 = false, int bitDepth = 8,
                   bool keepAlpha = false, std::string *note = nullptr);

// Lossless input (av1gpu --lossless): the PNG / WebP samples themselves, no colour conversion, gamma or chroma
// downsampling, padded to multiples of 64 by edge replication.
//   RGB (PNG RGB / palette, WebP): 4:4:4 planes, identity / GBR (Y = G, U = B, V = R; CICP matrix 0) at the source
//     depth, or with ycgco YCgCo-Re (matrix 16, libavif 1.4 / H.273 formulas: Co = R - B, t = B + (Co >> 1),
//     Cg = G - t, Y = t + (Cg >> 1); U = Cg + 512, V = Co + 512) in 10 bits from 8-bit RGB.
//   gray (PNG gray): monochrome Y = gray (matrix left at the default).
//   16-bit PNG samples are accepted when they hold 8 bits (every sample = 257 v: coded 8-bit) or 10 bits (every
//     sample = libavif's 10 -> 16-bit expansion (uint16)(0.5f + v / 1023.0f * 65535.0f), as avifdec writes it:
//     coded 10-bit); anything else cannot be represented losslessly (AV1 here: 8 or 10 bits) and is rejected.
// Alpha (keepAlpha): as gp_decode_cpu, the stored alpha samples with straight colour; the alpha plane has the colour's
// bit depth (AVIF: an alpha image has its master image's bit depth): 16-bit PNG alpha is checked together with the
// colour samples (all 8 or all 10 significant bits, else rejected), and with --ycgco-re (10-bit colour from 8-bit RGB)
// 8-bit alpha is rescaled to 10 bits with libavif's (int)(0.5f + v / 255 * 1023), which avifdec maps back exactly.  An
// opaque alpha channel is not coded.  Without keepAlpha the alpha channel is ignored (a note says so).
// *matrix receives the CICP matrix to signal (0, 16 or -1 for the default), *note a human-readable remark (may be empty).
bool gp_decode_lossless(const std::string &path, GpHostPlanes &out, bool ycgco, int *matrix, std::string &note, std::string &err,
                        bool keepAlpha = false);

bool gp_read_file(const std::string &path, std::vector<uint8_t> &data);
bool gp_write_file(const std::string &path, const uint8_t *d, size_t n);

#endif
