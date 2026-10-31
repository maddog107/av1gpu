// input helpers.
#include "imageio.h"

#include <png.h>
#include <webp/decode.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>

bool gp_read_file(const std::string &path, std::vector<uint8_t> &data) {
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    data.resize(n > 0 ? (size_t)n : 0);
    bool ok = n >= 0 && fread(data.data(), 1, data.size(), f) == data.size();
    fclose(f);
    return ok;
}

bool gp_write_file(const std::string &path, const uint8_t *d, size_t n) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return false;
    bool ok = fwrite(d, 1, n, f) == n;
    ok &= fclose(f) == 0;
    return ok;
}

GpFileKind gp_sniff(const std::string &path) {
    uint8_t h[16] = {0};
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return GP_FILE_UNKNOWN;
    size_t n = fread(h, 1, sizeof h, f);
    fclose(f);
    if (n >= 3 && h[0] == 0xFF && h[1] == 0xD8 && h[2] == 0xFF) return GP_FILE_JPEG;
    if (n >= 12 && !memcmp(h, "RIFF", 4) && !memcmp(h + 8, "WEBP", 4)) return GP_FILE_WEBP;
    if (n >= 8 && !memcmp(h, "\x89PNG\r\n\x1a\n", 8)) return GP_FILE_PNG;
    return GP_FILE_UNKNOWN;
}

static bool decode_png(const std::vector<uint8_t> &d, int &W, int &H, std::vector<uint8_t> &rgb, std::string &err) {
    png_image img;
    memset(&img, 0, sizeof img);
    img.version = PNG_IMAGE_VERSION;
    if (!png_image_begin_read_from_memory(&img, d.data(), d.size())) { err = "png header"; return false; }
    img.format = PNG_FORMAT_RGB;
    rgb.resize(PNG_IMAGE_SIZE(img));
    if (!png_image_finish_read(&img, nullptr, rgb.data(), 0, nullptr)) { err = "png decode"; return false; }
    W = (int)img.width;
    H = (int)img.height;
    return true;
}

// 16-bit PNG samples (libpng classic API): RGB, 16 bits per sample (alpha stripped, palette / gray expanded)
namespace {
struct PngMem { const uint8_t *p; size_t n, at; };
void png_mem_read(png_structp ps, png_bytep out, png_size_t len) {
    PngMem *m = (PngMem *)png_get_io_ptr(ps);
    if (m->at + len > m->n) png_error(ps, "truncated");
    memcpy(out, m->p + m->at, len);
    m->at += len;
}
}  // namespace
static bool decode_png16(const std::vector<uint8_t> &d, int &W, int &H, std::vector<uint16_t> &rgb, std::string &err) {
    png_structp ps = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!ps) { err = "png init"; return false; }
    png_infop pi = png_create_info_struct(ps);
    if (!pi) { png_destroy_read_struct(&ps, nullptr, nullptr); err = "png init"; return false; }
    if (setjmp(png_jmpbuf(ps))) { png_destroy_read_struct(&ps, &pi, nullptr); err = "png decode"; return false; }
    PngMem m = {d.data(), d.size(), 0};
    png_set_read_fn(ps, &m, png_mem_read);
    png_read_info(ps, pi);
    const int ct = png_get_color_type(ps, pi), bdp = png_get_bit_depth(ps, pi);
    if (ct == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(ps);
    if (ct == PNG_COLOR_TYPE_GRAY && bdp < 8) png_set_expand_gray_1_2_4_to_8(ps);
    if (ct == PNG_COLOR_TYPE_GRAY || ct == PNG_COLOR_TYPE_GRAY_ALPHA) png_set_gray_to_rgb(ps);
    if (ct & PNG_COLOR_MASK_ALPHA) png_set_strip_alpha(ps);
    if (bdp < 16) png_set_expand_16(ps);
    png_read_update_info(ps, pi);
    W = (int)png_get_image_width(ps, pi);
    H = (int)png_get_image_height(ps, pi);
    const size_t rb = png_get_rowbytes(ps, pi);
    if (rb != (size_t)W * 6) { png_destroy_read_struct(&ps, &pi, nullptr); err = "png layout"; return false; }
    std::vector<uint8_t> raw(rb * H);
    std::vector<png_bytep> rows(H);
    for (int y = 0; y < H; y++) rows[y] = raw.data() + rb * y;
    png_read_image(ps, rows.data());
    png_destroy_read_struct(&ps, &pi, nullptr);
    rgb.resize((size_t)W * H * 3);
    for (size_t i = 0; i < rgb.size(); i++) rgb[i] = (uint16_t)(raw[2 * i] << 8 | raw[2 * i + 1]);   // PNG: big endian
    return true;
}

// 10-bit planes from normalised RGB (see imageio.h)
static void rgb_to_planes10(const std::vector<double> &rgbn, int W, int H, bool yuv444, GpHostPlanes &out) {
    std::vector<uint16_t> Y((size_t)W * H);
    std::vector<double> Cb((size_t)W * H), Cr((size_t)W * H);
    auto q10 = [](double v) { const double r = std::floor(v + 0.5); return (uint16_t)(r < 0 ? 0 : r > 1023 ? 1023 : (int)r); };
    for (size_t i = 0; i < (size_t)W * H; i++) {
        const double r = rgbn[3 * i], g = rgbn[3 * i + 1], b = rgbn[3 * i + 2];
        Y[i] = q10((0.299 * r + 0.587 * g + 0.114 * b) * 1023.0);
        Cb[i] = -0.168735892 * r - 0.331264108 * g + 0.5 * b;
        Cr[i] = 0.5 * r - 0.418687589 * g - 0.081312411 * b;
    }
    out.p16[0].assign((size_t)out.aw * out.ah, 0);
    out.p16[1].assign((size_t)out.cw() * out.ch(), 0);
    out.p16[2].assign((size_t)out.cw() * out.ch(), 0);
    for (int y = 0; y < out.ah; y++)
        for (int x = 0; x < out.aw; x++) out.p16[0][(size_t)y * out.aw + x] = Y[(size_t)std::min(y, H - 1) * W + std::min(x, W - 1)];
    if (yuv444) {
        for (int y = 0; y < out.ah; y++)
            for (int x = 0; x < out.aw; x++) {
                const size_t o = (size_t)std::min(y, H - 1) * W + std::min(x, W - 1);
                out.p16[1][(size_t)y * out.aw + x] = q10(Cb[o] * 1023.0 + 512.0);
                out.p16[2][(size_t)y * out.aw + x] = q10(Cr[o] * 1023.0 + 512.0);
            }
        return;
    }
    const int cw = (W + 1) / 2, ch = (H + 1) / 2, caw = out.aw / 2, cah = out.ah / 2;
    std::vector<uint16_t> cbs((size_t)cw * ch), crs((size_t)cw * ch);
    for (int y = 0; y < ch; y++)
        for (int x = 0; x < cw; x++) {
            double sb = 0, sr = 0;
            for (int dy = 0; dy < 2; dy++)
                for (int dx = 0; dx < 2; dx++) {
                    const size_t o = (size_t)std::min(2 * y + dy, H - 1) * W + std::min(2 * x + dx, W - 1);
                    sb += Cb[o]; sr += Cr[o];
                }
            cbs[(size_t)y * cw + x] = q10(sb * 0.25 * 1023.0 + 512.0);
            crs[(size_t)y * cw + x] = q10(sr * 0.25 * 1023.0 + 512.0);
        }
    for (int y = 0; y < cah; y++)
        for (int x = 0; x < caw; x++) {
            const size_t o = (size_t)std::min(y, ch - 1) * cw + std::min(x, cw - 1);
            out.p16[1][(size_t)y * caw + x] = cbs[o];
            out.p16[2][(size_t)y * caw + x] = crs[o];
        }
}

bool gp_decode_cpu(const std::string &path, GpHostPlanes &out, std::string &err, bool yuv444, int bitDepth) {
    std::vector<uint8_t> d;
    if (!gp_read_file(path, d)) { err = "read failed"; return false; }
    int W = 0, H = 0;
    std::vector<uint8_t> rgb;
    const GpFileKind k = gp_sniff(path);
    if (bitDepth == 10) {
        std::vector<double> rgbn;
        bool png16 = false;
        if (k == GP_FILE_PNG) {   // 16-bit PNG? (the simplified API reports it as a linear format)
            png_image img;
            memset(&img, 0, sizeof img);
            img.version = PNG_IMAGE_VERSION;
            if (!png_image_begin_read_from_memory(&img, d.data(), d.size())) { err = "png header"; return false; }
            png16 = (img.format & PNG_FORMAT_FLAG_LINEAR) != 0;
            png_image_free(&img);
        }
        if (png16) {
            std::vector<uint16_t> rgb16;
            if (!decode_png16(d, W, H, rgb16, err)) return false;
            rgbn.resize(rgb16.size());
            for (size_t i = 0; i < rgb16.size(); i++) rgbn[i] = rgb16[i] / 65535.0;
        } else {
            if (k == GP_FILE_WEBP) {
                uint8_t *p = WebPDecodeRGB(d.data(), d.size(), &W, &H);
                if (!p) { err = "webp decode failed"; return false; }
                rgb.assign(p, p + (size_t)W * H * 3);
                WebPFree(p);
            } else if (k == GP_FILE_PNG) {
                if (!decode_png(d, W, H, rgb, err)) return false;
            } else {
                err = "unsupported input format";
                return false;
            }
            rgbn.resize(rgb.size());
            for (size_t i = 0; i < rgb.size(); i++) rgbn[i] = rgb[i] / 255.0;
        }
        out.W = W; out.H = H; out.mono = 0; out.yuv444 = yuv444 ? 1 : 0; out.bitDepth = 10;
        const int MiW = 2 * ((W + 7) >> 3) * 4, MiH = 2 * ((H + 7) >> 3) * 4;
        out.aw = (MiW + 63) & ~63;
        out.ah = (MiH + 63) & ~63;
        rgb_to_planes10(rgbn, W, H, yuv444, out);
        return true;
    }
    if (k == GP_FILE_WEBP) {
        uint8_t *p = WebPDecodeRGB(d.data(), d.size(), &W, &H);
        if (!p) { err = "webp decode failed"; return false; }
        rgb.assign(p, p + (size_t)W * H * 3);
        WebPFree(p);
    } else if (k == GP_FILE_PNG) {
        if (!decode_png(d, W, H, rgb, err)) return false;
    } else {
        err = "unsupported input format";
        return false;
    }
    out.W = W; out.H = H; out.mono = 0; out.yuv444 = yuv444 ? 1 : 0;
    const int MiW = 2 * ((W + 7) >> 3) * 4, MiH = 2 * ((H + 7) >> 3) * 4;
    out.aw = (MiW + 63) & ~63;
    out.ah = (MiH + 63) & ~63;
    const int cw = (W + 1) / 2, ch = (H + 1) / 2;
    std::vector<uint8_t> Y((size_t)W * H), Cb((size_t)W * H), Cr((size_t)W * H);
    for (size_t i = 0; i < (size_t)W * H; i++) {
        const double r = rgb[3 * i], g = rgb[3 * i + 1], b = rgb[3 * i + 2];
        auto cl = [](double v) { return (uint8_t)(v < 0 ? 0 : v > 255 ? 255 : (int)lrint(v)); };
        Y[i] = cl(0.299 * r + 0.587 * g + 0.114 * b);
        Cb[i] = cl(-0.168735892 * r - 0.331264108 * g + 0.5 * b + 128);
        Cr[i] = cl(0.5 * r - 0.418687589 * g - 0.081312411 * b + 128);
    }
    out.p[0].assign((size_t)out.aw * out.ah, 0);
    out.p[1].assign((size_t)out.cw() * out.ch(), 0);
    out.p[2].assign((size_t)out.cw() * out.ch(), 0);
    for (int y = 0; y < out.ah; y++)
        for (int x = 0; x < out.aw; x++) out.p[0][(size_t)y * out.aw + x] = Y[(size_t)std::min(y, H - 1) * W + std::min(x, W - 1)];
    if (yuv444) {   // full-resolution chroma, edge-replicated like luma
        for (int y = 0; y < out.ah; y++)
            for (int x = 0; x < out.aw; x++) {
                const size_t o = (size_t)std::min(y, H - 1) * W + std::min(x, W - 1);
                out.p[1][(size_t)y * out.aw + x] = Cb[o];
                out.p[2][(size_t)y * out.aw + x] = Cr[o];
            }
        return true;
    }
    const int caw = out.aw / 2, cah = out.ah / 2;
    std::vector<uint8_t> cbs((size_t)cw * ch), crs((size_t)cw * ch);
    for (int y = 0; y < ch; y++)
        for (int x = 0; x < cw; x++) {
            int sb = 0, sr = 0;
            for (int dy = 0; dy < 2; dy++)
                for (int dx = 0; dx < 2; dx++) {
                    const size_t o = (size_t)std::min(2 * y + dy, H - 1) * W + std::min(2 * x + dx, W - 1);
                    sb += Cb[o]; sr += Cr[o];
                }
            cbs[(size_t)y * cw + x] = (uint8_t)((sb + 2) >> 2);
            crs[(size_t)y * cw + x] = (uint8_t)((sr + 2) >> 2);
        }
    for (int y = 0; y < cah; y++)
        for (int x = 0; x < caw; x++) {
            const size_t o = (size_t)std::min(y, ch - 1) * cw + std::min(x, cw - 1);
            out.p[1][(size_t)y * caw + x] = cbs[o];
            out.p[2][(size_t)y * caw + x] = crs[o];
        }
    return true;
}

// ---------------------------------------------------------------------------
// lossless input (see imageio.h)
// ---------------------------------------------------------------------------
// PNG samples as stored (libpng classic API without gamma / colour transforms): palette -> RGB, gray < 8 bits ->
// 8 bits, alpha dropped (*alpha set); 16-bit samples kept.  ch = 1 (gray) or 3 (RGB).
static bool decode_png_raw(const std::vector<uint8_t> &d, int &W, int &H, int &ch, int &bits, bool &alpha,
                           std::vector<uint16_t> &px, std::string &err) {
    png_structp ps = png_create_read_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
    if (!ps) { err = "png init"; return false; }
    png_infop pi = png_create_info_struct(ps);
    if (!pi) { png_destroy_read_struct(&ps, nullptr, nullptr); err = "png init"; return false; }
    if (setjmp(png_jmpbuf(ps))) { png_destroy_read_struct(&ps, &pi, nullptr); err = "png decode"; return false; }
    PngMem m = {d.data(), d.size(), 0};
    png_set_read_fn(ps, &m, png_mem_read);
    png_read_info(ps, pi);
    const int ct = png_get_color_type(ps, pi), bdp = png_get_bit_depth(ps, pi);
    alpha = (ct & PNG_COLOR_MASK_ALPHA) != 0 || png_get_valid(ps, pi, PNG_INFO_tRNS);
    if (ct == PNG_COLOR_TYPE_PALETTE) png_set_palette_to_rgb(ps);   // (tRNS is not expanded to alpha)
    if ((ct == PNG_COLOR_TYPE_GRAY || ct == PNG_COLOR_TYPE_GRAY_ALPHA) && bdp < 8) png_set_expand_gray_1_2_4_to_8(ps);
    if (ct & PNG_COLOR_MASK_ALPHA) png_set_strip_alpha(ps);
    png_set_interlace_handling(ps);
    png_read_update_info(ps, pi);
    W = (int)png_get_image_width(ps, pi);
    H = (int)png_get_image_height(ps, pi);
    ch = png_get_channels(ps, pi);
    bits = png_get_bit_depth(ps, pi);
    if ((ch != 1 && ch != 3) || (bits != 8 && bits != 16)) {
        png_destroy_read_struct(&ps, &pi, nullptr);
        err = "png layout";
        return false;
    }
    const size_t rb = png_get_rowbytes(ps, pi);
    if (rb != (size_t)W * ch * (bits / 8)) { png_destroy_read_struct(&ps, &pi, nullptr); err = "png layout"; return false; }
    std::vector<uint8_t> raw(rb * H);
    std::vector<png_bytep> rows(H);
    for (int y = 0; y < H; y++) rows[y] = raw.data() + rb * y;
    png_read_image(ps, rows.data());
    png_destroy_read_struct(&ps, &pi, nullptr);
    px.resize((size_t)W * H * ch);
    if (bits == 16)
        for (size_t i = 0; i < px.size(); i++) px[i] = (uint16_t)(raw[2 * i] << 8 | raw[2 * i + 1]);   // PNG: big endian
    else
        for (size_t i = 0; i < px.size(); i++) px[i] = raw[i];
    return true;
}

// libavif 1.4's 10 -> 16-bit full-range expansion (reformat.c: table (float)v / 1023, then (uint16_t)(0.5f + x * 65535))
static uint16_t ll_expand10(int v) {
    const float f = ((float)v - 0.0f) / 1023.0f;
    return (uint16_t)(0.5f + f * 65535.0f);
}

bool gp_decode_lossless(const std::string &path, GpHostPlanes &out, bool ycgco, int *matrix, std::string &note, std::string &err) {
    std::vector<uint8_t> d;
    if (!gp_read_file(path, d)) { err = "read failed"; return false; }
    note.clear();
    int W = 0, H = 0, ch = 3, bits = 8;
    bool alpha = false;
    std::vector<uint16_t> px;   // W x H x ch samples
    const GpFileKind k = gp_sniff(path);
    if (k == GP_FILE_PNG) {
        if (!decode_png_raw(d, W, H, ch, bits, alpha, px, err)) return false;
    } else if (k == GP_FILE_WEBP) {
        WebPBitstreamFeatures ft;
        if (WebPGetFeatures(d.data(), d.size(), &ft) != VP8_STATUS_OK) { err = "webp header"; return false; }
        alpha = ft.has_alpha != 0;
        uint8_t *p = WebPDecodeRGB(d.data(), d.size(), &W, &H);
        if (!p) { err = "webp decode failed"; return false; }
        px.assign(p, p + (size_t)W * H * 3);
        WebPFree(p);
        if (ft.format == 1) note = "lossy WebP: lossless to its decoded RGB";
    } else {
        err = "unsupported input format for --lossless (PNG, WebP, JPEG)";
        return false;
    }
    if (alpha) note += std::string(note.empty() ? "" : "; ") + "alpha channel ignored (not coded)";
    // 16-bit samples: 8 or 10 significant bits, as defined by libavif's expansions
    int depth = bits;
    if (bits == 16) {
        bool all257 = true;
        for (uint16_t v : px) if (v % 257) { all257 = false; break; }
        if (all257) {
            for (auto &v : px) v = (uint16_t)(v / 257);
            depth = 8;
            note += std::string(note.empty() ? "" : "; ") + "16-bit PNG with 8 significant bits (v * 257): coded 8-bit";
        } else {
            size_t bad = (size_t)-1;
            for (size_t i = 0; i < px.size() && bad == (size_t)-1; i++) {
                const int v = (int)(((uint32_t)px[i] * 1023u + 32767u) / 65535u);
                if (ll_expand10(v) != px[i]) bad = i;
                else px[i] = (uint16_t)v;
            }
            if (bad != (size_t)-1) {
                char b[400];
                const size_t pix = bad / ch;
                snprintf(b, sizeof b,
                         "16-bit PNG holds more than 10 significant bits (sample %u at (%zu,%zu) is not v * 65535 / 1023 for a 10-bit v): "
                         "AV1 lossless here is 8- or 10-bit, so it cannot be coded losslessly (convert to 8 bits, or encode lossy with --depth 10)",
                         (unsigned)px[bad], pix % W, pix / W);
                err = b;
                return false;
            }
            depth = 10;
            note += std::string(note.empty() ? "" : "; ") + "16-bit PNG with 10 significant bits: coded 10-bit";
        }
    }
    const bool mono = ch == 1;
    if (ycgco && mono) { err = "--ycgco-re needs an RGB input (this PNG is grayscale)"; return false; }
    if (ycgco && depth != 8) {
        err = "--ycgco-re: YCgCo-Re adds two bits, so it needs 8-bit RGB (this input has 10 significant bits: use identity / GBR)";
        return false;
    }
    const int outDepth = ycgco ? 10 : depth;
    out.W = W; out.H = H; out.mono = mono ? 1 : 0; out.yuv444 = mono ? 0 : 1; out.bitDepth = outDepth;
    const int MiW = 2 * ((W + 7) >> 3) * 4, MiH = 2 * ((H + 7) >> 3) * 4;
    out.aw = (MiW + 63) & ~63;
    out.ah = (MiH + 63) & ~63;
    *matrix = mono ? -1 : ycgco ? 16 : 0;
    std::vector<uint16_t> pl[3];
    const int np = mono ? 1 : 3;
    for (int p = 0; p < np; p++) pl[p].assign((size_t)out.aw * out.ah, 0);
    for (int y = 0; y < out.ah; y++)
        for (int x = 0; x < out.aw; x++) {
            const size_t s = ((size_t)std::min(y, H - 1) * W + std::min(x, W - 1)) * ch, o = (size_t)y * out.aw + x;
            if (mono) { pl[0][o] = px[s]; continue; }
            const int R = px[s], G = px[s + 1], B = px[s + 2];
            if (ycgco) {   // H.273 (2024) formulas 58..61, as libavif 1.4 avifImageRGBToYUV (full range, 2 extra bits)
                const int Co = R - B, t = B + (Co >> 1), Cg = G - t;
                pl[0][o] = (uint16_t)(t + (Cg >> 1));
                pl[1][o] = (uint16_t)(Cg + 512);
                pl[2][o] = (uint16_t)(Co + 512);
            } else {       // identity (GBR): Y = G, U = B, V = R
                pl[0][o] = (uint16_t)G;
                pl[1][o] = (uint16_t)B;
                pl[2][o] = (uint16_t)R;
            }
        }
    for (int p = 0; p < 3; p++) {
        out.p[p].clear();
        out.p16[p].clear();
        if (p >= np) continue;
        if (outDepth == 10) out.p16[p].swap(pl[p]);
        else out.p[p].assign(pl[p].begin(), pl[p].end());
    }
    return true;
}
