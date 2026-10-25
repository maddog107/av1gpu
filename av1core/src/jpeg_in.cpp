// JPEG -> 4:2:0 (or monochrome) planes with libjpeg-turbo, no RGB round trip.
#include "tools.h"

#include <csetjmp>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <jpeglib.h>

namespace av1g {

namespace {
struct ErrMgr {
    jpeg_error_mgr pub;
    jmp_buf jb;
    char msg[JMSG_LENGTH_MAX];
};
void on_error(j_common_ptr c) {
    ErrMgr *e = (ErrMgr *)c->err;
    (*c->err->format_message)(c, e->msg);
    longjmp(e->jb, 1);
}

// resample one chroma plane (sw x sh) to (tw x th) when the ratio per axis is 1, 2
// (box average) or 1/2 (replication); returns false for anything else.
bool resample(const std::vector<uint8_t> &s, int sw, int sh, int sstride, std::vector<uint8_t> &d, int tw, int th) {
    auto fac = [](int sdim, int tdim) -> int {
        if (sdim == tdim) return 1;
        if ((sdim + 1) / 2 == tdim) return 2;     // downsample by 2
        if ((tdim + 1) / 2 == sdim) return -2;    // upsample by 2
        return 0;
    };
    int fx = fac(sw, tw), fy = fac(sh, th);
    if (!fx || !fy) return false;
    d.resize((size_t)tw * th);
    for (int y = 0; y < th; y++)
        for (int x = 0; x < tw; x++) {
            int xs0, xs1, ys0, ys1;
            if (fx == 2) { xs0 = 2 * x; xs1 = 2 * x + 1 < sw ? 2 * x + 1 : sw - 1; }
            else if (fx == 1) xs0 = xs1 = x;
            else xs0 = xs1 = x / 2;
            if (fy == 2) { ys0 = 2 * y; ys1 = 2 * y + 1 < sh ? 2 * y + 1 : sh - 1; }
            else if (fy == 1) ys0 = ys1 = y;
            else ys0 = ys1 = y / 2;
            int a = s[(size_t)ys0 * sstride + xs0], b = s[(size_t)ys0 * sstride + xs1];
            int c = s[(size_t)ys1 * sstride + xs0], e = s[(size_t)ys1 * sstride + xs1];
            int v;
            if (xs0 != xs1 && ys0 != ys1) v = (a + b + c + e + 2) >> 2;
            else if (xs0 != xs1) v = (a + b + 1) >> 1;
            else if (ys0 != ys1) v = (a + c + 1) >> 1;
            else v = a;
            d[(size_t)y * tw + x] = (uint8_t)v;
        }
    return true;
}
// EXIF orientation from an APP1 payload ("Exif\0\0" + TIFF), IFD0 tag 0x0112 SHORT count 1
// (same acceptance rules as libavif's avifGetExifOrientationOffset); 1 if absent/invalid.
int exif_orientation(const uint8_t *d, size_t n) {
    if (n < 14 || memcmp(d, "Exif\0\0", 6)) return 1;
    d += 6;
    n -= 6;
    const bool le = d[0] == 'I' && d[1] == 'I';
    if (!le && !(d[0] == 'M' && d[1] == 'M')) return 1;
    auto u16 = [&](size_t o) -> uint32_t { return o + 2 > n ? 0 : le ? (d[o] | d[o + 1] << 8) : (d[o] << 8 | d[o + 1]); };
    auto u32 = [&](size_t o) -> uint32_t {
        if (o + 4 > n) return 0;
        return le ? (d[o] | d[o + 1] << 8 | d[o + 2] << 16 | (uint32_t)d[o + 3] << 24)
                  : ((uint32_t)d[o] << 24 | d[o + 1] << 16 | d[o + 2] << 8 | d[o + 3]);
    };
    const size_t ifd = u32(4);
    const uint32_t count = u16(ifd);
    for (uint32_t i = 0; i < count; i++) {
        const size_t e = ifd + 2 + 12 * (size_t)i;
        if (e + 12 > n) break;
        if (u16(e) == 0x0112 && u16(e + 2) == 3 && u32(e + 4) == 1) {
            const uint32_t v = u16(e + 8);
            return (v >= 1 && v <= 8) ? (int)v : 1;
        }
    }
    return 1;
}
}  // namespace

bool load_jpeg(const char *path, Image &img, std::string &info, std::string &err) {
    FILE *f = fopen(path, "rb");
    if (!f) { err = std::string("cannot open ") + path; return false; }
    jpeg_decompress_struct ci;
    ErrMgr em;
    ci.err = jpeg_std_error(&em.pub);
    em.pub.error_exit = on_error;
    if (setjmp(em.jb)) {
        err = em.msg;
        jpeg_destroy_decompress(&ci);
        fclose(f);
        return false;
    }
    jpeg_create_decompress(&ci);
    jpeg_stdio_src(&ci, f);
    jpeg_save_markers(&ci, JPEG_APP0 + 1, 0xFFFF);
    jpeg_save_markers(&ci, JPEG_APP0 + 2, 0xFFFF);
    jpeg_read_header(&ci, TRUE);
    {
        JOCTET *icc = nullptr;
        unsigned int iccLen = 0;
        if (jpeg_read_icc_profile(&ci, &icc, &iccLen) && icc) img.icc.assign(icc, icc + iccLen);
        free(icc);
        for (jpeg_saved_marker_ptr m = ci.marker_list; m; m = m->next)
            if (m->marker == JPEG_APP0 + 1) {
                int o = exif_orientation(m->data, m->data_length);
                if (o != 1 || img.orientation == 1) img.orientation = o;
                if (m->data_length >= 6 && !memcmp(m->data, "Exif\0\0", 6)) break;
            }
    }
    const int nc = ci.num_components;
    img.W = ci.image_width;
    img.H = ci.image_height;
    if (nc == 4 || ci.jpeg_color_space == JCS_CMYK || ci.jpeg_color_space == JCS_YCCK) {
        err = "CMYK/YCCK JPEG not supported (CPU fallback)";
        jpeg_destroy_decompress(&ci);
        fclose(f);
        return false;
    }
    img.mono = nc == 1;
    const int W = img.W, H = img.H, cw = (W + 1) >> 1, ch = (H + 1) >> 1;
    bool raw = nc == 1 || (nc == 3 && ci.jpeg_color_space == JCS_YCbCr);
    if (raw) {
        // raw planes at native sampling
        ci.raw_data_out = TRUE;
        ci.do_fancy_upsampling = FALSE;
        jpeg_start_decompress(&ci);
        const int maxv = ci.max_v_samp_factor;
        std::vector<std::vector<uint8_t>> pl(nc);
        std::vector<int> pw(nc), ph(nc), ps(nc);
        for (int c = 0; c < nc; c++) {
            jpeg_component_info &k = ci.comp_info[c];
            ps[c] = k.width_in_blocks * DCTSIZE;
            int rows = (int)((ci.output_height + ci.max_v_samp_factor * DCTSIZE - 1) / (ci.max_v_samp_factor * DCTSIZE)) *
                       k.v_samp_factor * DCTSIZE;
            pl[c].assign((size_t)ps[c] * rows, 0);
            pw[c] = k.downsampled_width;
            ph[c] = k.downsampled_height;
        }
        std::vector<JSAMPROW> rowp[3];
        JSAMPARRAY arr[3];
        int y = 0;
        while (ci.output_scanline < ci.output_height) {
            for (int c = 0; c < nc; c++) {
                jpeg_component_info &k = ci.comp_info[c];
                int n = k.v_samp_factor * DCTSIZE;
                rowp[c].resize(n);
                int r0 = y / maxv * k.v_samp_factor;
                for (int i = 0; i < n; i++) rowp[c][i] = pl[c].data() + (size_t)(r0 + i) * ps[c];
                arr[c] = rowp[c].data();
            }
            int got = jpeg_read_raw_data(&ci, arr, maxv * DCTSIZE);
            y += got ? got : maxv * DCTSIZE;
        }
        img.p[0].resize((size_t)W * H);
        for (int r = 0; r < H; r++) memcpy(img.p[0].data() + (size_t)r * W, pl[0].data() + (size_t)r * ps[0], W);
        if (nc == 3) {
            const jpeg_component_info *k = ci.comp_info;
            char sub[64];
            snprintf(sub, sizeof sub, "%dx%d,%dx%d,%dx%d", k[0].h_samp_factor, k[0].v_samp_factor, k[1].h_samp_factor,
                     k[1].v_samp_factor, k[2].h_samp_factor, k[2].v_samp_factor);
            bool ok = true;
            for (int c = 1; c < 3 && ok; c++) ok = resample(pl[c], pw[c], ph[c], ps[c], img.p[c], cw, ch);
            if (ok) {
                info = (pw[1] == cw && ph[1] == ch) ? "420" : (pw[1] == W && ph[1] == H) ? "444->420"
                     : (pw[1] == cw && ph[1] == H) ? "422->420" : std::string("sub ") + sub + "->420";
                jpeg_finish_decompress(&ci);
                jpeg_destroy_decompress(&ci);
                fclose(f);
                return true;
            }
            info = std::string("sub ") + sub + " (full decode)";
            raw = false;  // fall through to full-resolution decode
            jpeg_abort_decompress(&ci);
            fseek(f, 0, SEEK_SET);
            jpeg_stdio_src(&ci, f);
            jpeg_read_header(&ci, TRUE);
        } else {
            info = "gray";
            jpeg_finish_decompress(&ci);
            jpeg_destroy_decompress(&ci);
            fclose(f);
            return true;
        }
    }
    // full-resolution YCbCr decode (libjpeg converts RGB JPEGs with BT.601 full range)
    ci.raw_data_out = FALSE;
    ci.out_color_space = JCS_YCbCr;
    jpeg_start_decompress(&ci);
    std::vector<uint8_t> full((size_t)W * H * 3);
    while (ci.output_scanline < ci.output_height) {
        JSAMPROW row = full.data() + (size_t)ci.output_scanline * W * 3;
        jpeg_read_scanlines(&ci, &row, 1);
    }
    std::vector<uint8_t> pc[3];
    for (int c = 0; c < 3; c++) {
        pc[c].resize((size_t)W * H);
        for (size_t i = 0; i < (size_t)W * H; i++) pc[c][i] = full[i * 3 + c];
    }
    img.p[0] = pc[0];
    resample(pc[1], W, H, W, img.p[1], cw, ch);
    resample(pc[2], W, H, W, img.p[2], cw, ch);
    if (info.empty()) info = nc == 3 && ci.jpeg_color_space == JCS_RGB ? "rgb->ycbcr->420" : "full decode->420";
    jpeg_finish_decompress(&ci);
    jpeg_destroy_decompress(&ci);
    fclose(f);
    return true;
}

}  // namespace av1g
