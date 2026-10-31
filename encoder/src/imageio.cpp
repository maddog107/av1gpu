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

bool gp_decode_cpu(const std::string &path, GpHostPlanes &out, std::string &err, bool yuv444) {
    std::vector<uint8_t> d;
    if (!gp_read_file(path, d)) { err = "read failed"; return false; }
    int W = 0, H = 0;
    std::vector<uint8_t> rgb;
    const GpFileKind k = gp_sniff(path);
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
