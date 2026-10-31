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
