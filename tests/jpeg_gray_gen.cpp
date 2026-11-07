// Test JPEGs written by libjpeg-turbo, with libjpeg-turbo's decode of each.
//
// jpeg_gray_gen OUTDIR: gray JPEGs for jpeg_gray_sampling_test.py (#44).  One-component JPEGs of a synthetic picture
// (gradient, 16-row bands, a sine and a little texture): baseline at 37x29, 640x480, 1001x667 and 2000x1500, with a
// restart interval of 5 MCUs and progressive at 640x480 and 1001x667, all at quality 90.  Each is written as
// <name>_11.jpg and again with the component's sampling factors in the SOF set to 2x2, 2x1 and 1x2
// (<name>_22/_21/_12.jpg).  That is legal: a one-component scan is non-interleaved, so its MCU is one block whatever
// the factors say.  <name>.gray holds libjpeg-turbo's decode (islow, W x H bytes); the decode of every tagged file
// must equal it, or the generator fails.
//
// jpeg_gray_gen --ycbcr OUTDIR: 3-component JPEGs for jpeg_inverted_sampling_test.py, quality 90, with the sampling
// factors listed in ycbcr_main(): luma below a chroma component's factors (Y 1x1 with Cb / Cr 2x2, ...: the
// horizontal and vertical libjpeg-turbo upsamplers, replication for 3:1 and 4:1, a chroma component below luma too, a
// 4x3 picture whose luma is too narrow for fancy upsampling) and 4:2:0 / 4:4:4 controls, at 37x29 and 333x257.
// <name>.jpg; <name>.444: libjpeg-turbo's YCbCr decode (islow, fancy upsampling, no colour conversion; Y, Cb, Cr
// planes of W x H bytes); <name>.420: its Y plane, then Cb and Cr of (W+1)/2 x (H+1)/2 as av1gpu's 4:2:0 conversion
// makes them from the coded chroma (each chroma component read at full resolution by replicating its samples, 2x2
// average with the rounding bias alternating 1, 2 by column, edges replicated; for a component at the maximum factors
// that is libjpeg-turbo's decode, averaged).
//
// jpeg_gray_gen --pages OUTDIR: 4:2:0 JPEGs for imageops_jpeg_test.py, quality 90: page.jpg, 640x480 white with a
// 320x240 picture (a coloured checker pattern) at +64+48, aligned to the 16x16 MCUs so that no ringing crosses into the
// paper, with a white field inside it crossed by 1-pixel red lines (chroma shared across a colour edge), and
// blank.jpg, 640x480 white.
//
// One line per name on stdout: "<name> <W> <H>".
//
//   jpeg_gray_gen [--ycbcr|--pages] OUTDIR
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include <jpeglib.h>

typedef std::vector<uint8_t> Bytes;

// px: W x H gray, or W x H x 3 RGB coded with the sampling factors hv[c] = {h, v}
static Bytes compress(const Bytes &px, int w, int h, int restart, bool progressive, int nc = 1,
                      const int (*hv)[2] = nullptr) {
    jpeg_compress_struct ci;
    jpeg_error_mgr err;
    ci.err = jpeg_std_error(&err);
    jpeg_create_compress(&ci);
    unsigned char *buf = nullptr;
    unsigned long n = 0;
    jpeg_mem_dest(&ci, &buf, &n);
    ci.image_width = w;
    ci.image_height = h;
    ci.input_components = nc;
    ci.in_color_space = nc == 1 ? JCS_GRAYSCALE : JCS_RGB;
    jpeg_set_defaults(&ci);
    jpeg_set_quality(&ci, 90, TRUE);
    for (int c = 0; hv && c < nc; c++) {
        ci.comp_info[c].h_samp_factor = hv[c][0];
        ci.comp_info[c].v_samp_factor = hv[c][1];
    }
    ci.restart_interval = restart;
    if (progressive) jpeg_simple_progression(&ci);
    jpeg_start_compress(&ci, TRUE);
    while (ci.next_scanline < ci.image_height) {
        JSAMPROW row = (JSAMPROW)&px[(size_t)ci.next_scanline * w * nc];
        jpeg_write_scanlines(&ci, &row, 1);
    }
    jpeg_finish_compress(&ci);
    jpeg_destroy_compress(&ci);
    Bytes out(buf, buf + n);
    free(buf);
    return out;
}

// libjpeg-turbo's decode: gray, or YCbCr without colour conversion as planes (Y, Cb, Cr); empty if not W x H x nc
static Bytes decode(const Bytes &jpg, int w, int h, int nc = 1) {
    jpeg_decompress_struct ci;
    jpeg_error_mgr err;
    ci.err = jpeg_std_error(&err);
    jpeg_create_decompress(&ci);
    jpeg_mem_src(&ci, jpg.data(), (unsigned long)jpg.size());
    jpeg_read_header(&ci, TRUE);
    ci.dct_method = JDCT_ISLOW;
    ci.out_color_space = nc == 1 ? JCS_GRAYSCALE : JCS_YCbCr;
    jpeg_start_decompress(&ci);
    Bytes px;
    if ((int)ci.output_width == w && (int)ci.output_height == h && ci.output_components == nc) {
        Bytes row((size_t)w * nc);
        px.resize((size_t)w * h * nc);
        while (ci.output_scanline < ci.output_height) {
            const size_t y = ci.output_scanline;
            JSAMPROW r = row.data();
            jpeg_read_scanlines(&ci, &r, 1);
            for (int x = 0; x < w; x++)
                for (int c = 0; c < nc; c++) px[((size_t)c * h + y) * w + x] = row[(size_t)x * nc + c];
        }
        jpeg_finish_decompress(&ci);
    }
    jpeg_destroy_decompress(&ci);
    return px;
}

// the coded (native) planes, libjpeg-turbo raw data output; plane c has pitch[c] = width_in_blocks * 8 + 64
static void decode_raw(const Bytes &jpg, std::vector<Bytes> &nat, std::vector<int> &pitch) {
    jpeg_decompress_struct ci;
    jpeg_error_mgr err;
    ci.err = jpeg_std_error(&err);
    jpeg_create_decompress(&ci);
    jpeg_mem_src(&ci, jpg.data(), (unsigned long)jpg.size());
    jpeg_read_header(&ci, TRUE);
    ci.raw_data_out = TRUE;
    ci.dct_method = JDCT_ISLOW;
    jpeg_start_decompress(&ci);
    const int nc = ci.num_components, rows = ci.max_v_samp_factor * DCTSIZE;
    const int imcus = (int)((ci.output_height + rows - 1) / rows);
    nat.assign(nc, Bytes());
    pitch.assign(nc, 0);
    std::vector<std::vector<JSAMPROW>> rp(nc);
    std::vector<JSAMPARRAY> comps(nc);
    for (int c = 0; c < nc; c++) {
        const jpeg_component_info &k = ci.comp_info[c];
        pitch[c] = (int)k.width_in_blocks * DCTSIZE + 64;   // libjpeg may write up to the MCU width
        nat[c].assign((size_t)pitch[c] * imcus * k.v_samp_factor * DCTSIZE, 0);
        rp[c].resize(k.v_samp_factor * DCTSIZE);
    }
    for (int m = 0; ci.output_scanline < ci.output_height; m++) {
        for (int c = 0; c < nc; c++) {
            for (size_t r = 0; r < rp[c].size(); r++) rp[c][r] = &nat[c][((size_t)m * rp[c].size() + r) * pitch[c]];
            comps[c] = rp[c].data();
        }
        if (!jpeg_read_raw_data(&ci, comps.data(), rows)) break;
    }
    jpeg_finish_decompress(&ci);
    jpeg_destroy_decompress(&ci);
}

// av1gpu's 4:2:0 conversion of one chroma component (ingest finalize, OutDesc::chroma 1)
static Bytes to420(const Bytes &nat, int pitch, int hs, int vs, int hm, int vm, int w, int h) {
    const int cw = (w + 1) / 2, ch = (h + 1) / 2;
    auto at = [&](int x, int y) {
        x = std::min(x, w - 1);
        y = std::min(y, h - 1);
        return (int)nat[(size_t)(y * vs / vm) * pitch + x * hs / hm];
    };
    Bytes out((size_t)cw * ch);
    for (int y = 0; y < ch; y++)
        for (int x = 0; x < cw; x++) {
            const int s = at(2 * x, 2 * y) + at(2 * x + 1, 2 * y) + at(2 * x, 2 * y + 1) + at(2 * x + 1, 2 * y + 1);
            out[(size_t)y * cw + x] = (uint8_t)((s + ((x & 1) ? 2 : 1)) >> 2);
        }
    return out;
}

// offset of the one component's H/V byte in the SOF (marker segments walked from SOI), -1 if not found
static long sof_hv(const Bytes &b) {
    size_t q = 2;
    while (q + 4 <= b.size() && b[q] == 0xFF) {
        const int m = b[q + 1];
        const size_t len = ((size_t)b[q + 2] << 8) | b[q + 3];
        if (m >= 0xC0 && m <= 0xCF && m != 0xC4 && m != 0xC8 && m != 0xCC)
            return len == 11 && q + 12 < b.size() && b[q + 9] == 1 ? (long)(q + 11) : -1;
        if (m == 0xDA) break;
        q += 2 + len;
    }
    return -1;
}

static bool write_file(const std::string &p, const Bytes &b) {
    FILE *f = fopen(p.c_str(), "wb");
    if (!f) return false;
    const bool ok = fwrite(b.data(), 1, b.size(), f) == b.size();
    return fclose(f) == 0 && ok;
}

static int ycbcr_main(const std::string &dir) {
    static const int sets[][3][2] = {
        {{1, 1}, {2, 2}, {2, 2}}, {{1, 1}, {2, 1}, {2, 1}}, {{1, 1}, {1, 2}, {1, 2}}, {{1, 2}, {2, 2}, {2, 2}},
        {{2, 1}, {2, 2}, {2, 2}}, {{2, 1}, {4, 1}, {4, 1}}, {{1, 1}, {3, 1}, {3, 1}}, {{1, 1}, {4, 1}, {4, 1}},
        {{1, 1}, {2, 2}, {1, 1}}, {{2, 1}, {1, 2}, {1, 2}}, {{2, 2}, {1, 1}, {1, 1}}, {{1, 1}, {1, 1}, {1, 1}}};
    struct Size { int w, h; };
    for (const auto &hv : sets) {
        for (Size sz : {Size{37, 29}, Size{333, 257}, Size{4, 3}}) {
            const int w = sz.w, h = sz.h;
            if (w == 4 && !(hv[0][0] == 1 && hv[1][0] == 2 && hv[2][0] == 2)) continue;   // 4x3: Y 1xN, chroma 2xN
            Bytes px((size_t)w * h * 3);
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) {
                    const unsigned t = ((unsigned)x * 2654435761u ^ (unsigned)y * 40503u) >> 26;   // 0..63
                    const double r = x * 180 / std::max(w - 1, 1) + std::sin(y / 5.0) * 30 + (t & 15);
                    const double g = y * 160 / std::max(h - 1, 1) + (x / 6 % 2) * 40 + (t >> 2 & 15);
                    const double b = 128 + std::sin((x + y) / 7.0) * 90 + (t >> 4);
                    uint8_t *p = &px[((size_t)y * w + x) * 3];
                    p[0] = (uint8_t)std::min(std::max(r, 0.0), 255.0);
                    p[1] = (uint8_t)std::min(std::max(g, 0.0), 255.0);
                    p[2] = (uint8_t)std::min(std::max(b, 0.0), 255.0);
                }
            char name[64];
            snprintf(name, sizeof name, "ycc_%dx%d_%dx%d_%dx%d_%dx%d", hv[0][0], hv[0][1], hv[1][0], hv[1][1], hv[2][0],
                     hv[2][1], w, h);
            const Bytes jpg = compress(px, w, h, 0, false, 3, hv);
            const Bytes full = decode(jpg, w, h, 3);
            std::vector<Bytes> nat;
            std::vector<int> pitch;
            decode_raw(jpg, nat, pitch);
            if (full.empty() || nat.size() != 3) {
                fprintf(stderr, "%s: decode failed\n", name);
                return 1;
            }
            const int hm = std::max({hv[0][0], hv[1][0], hv[2][0]}), vm = std::max({hv[0][1], hv[1][1], hv[2][1]});
            for (int c = 0; c < 3; c++)   // a component at the maximum factors is decoded as coded
                if (hv[c][0] == hm && hv[c][1] == vm)
                    for (int y = 0; y < h; y++)
                        if (memcmp(&nat[c][(size_t)y * pitch[c]], &full[((size_t)c * h + y) * w], w)) {
                            fprintf(stderr, "%s: component %d: raw and full decode differ\n", name, c);
                            return 1;
                        }
            Bytes p420(full.begin(), full.begin() + (size_t)w * h);
            for (int c = 1; c < 3; c++) {
                const Bytes q = to420(nat[c], pitch[c], hv[c][0], hv[c][1], hm, vm, w, h);
                p420.insert(p420.end(), q.begin(), q.end());
            }
            const std::string base = dir + "/" + name;
            if (!write_file(base + ".jpg", jpg) || !write_file(base + ".444", full) ||
                !write_file(base + ".420", p420)) {
                fprintf(stderr, "%s: write failed\n", name);
                return 1;
            }
            printf("%s %d %d\n", name, w, h);
        }
    }
    return 0;
}

static int pages_main(const std::string &dir) {
    const int w = 640, h = 480, hv[3][2] = {{2, 2}, {1, 1}, {1, 1}};
    Bytes page((size_t)w * h * 3, 255), blank = page;
    for (int y = 48; y < 288; y++)
        for (int x = 64; x < 384; x++) {
            static const uint8_t colours[4][3] = {{200, 30, 30}, {30, 160, 60}, {40, 60, 200}, {90, 90, 90}};
            memcpy(&page[((size_t)y * w + x) * 3], colours[((x - 64) / 16 + 2 * ((y - 48) / 16)) & 3], 3);
            if (x >= 128 && x < 256 && y >= 112 && y < 224) {
                static const uint8_t white[3] = {255, 255, 255}, red[3] = {230, 20, 20};
                memcpy(&page[((size_t)y * w + x) * 3], x % 4 == 1 ? red : white, 3);
            }
        }
    if (!write_file(dir + "/page.jpg", compress(page, w, h, 0, false, 3, hv)) ||
        !write_file(dir + "/blank.jpg", compress(blank, w, h, 0, false, 3, hv)))
        return 1;
    printf("page %d %d\nblank %d %d\n", w, h, w, h);
    return 0;
}

int main(int argc, char **argv) {
    if (argc == 3 && !strcmp(argv[1], "--ycbcr")) return ycbcr_main(argv[2]);
    if (argc == 3 && !strcmp(argv[1], "--pages")) return pages_main(argv[2]);
    if (argc != 2) {
        fprintf(stderr, "usage: jpeg_gray_gen [--ycbcr|--pages] OUTDIR\n");
        return 2;
    }
    const std::string dir = argv[1];
    struct Kind { const char *name; int w, h, restart; bool progressive; };
    const Kind kinds[] = {{"base", 37, 29, 0, false},  {"base", 640, 480, 0, false}, {"base", 1001, 667, 0, false},
                          {"base", 2000, 1500, 0, false}, {"rst", 640, 480, 5, false}, {"rst", 1001, 667, 5, false},
                          {"prog", 640, 480, 0, true},  {"prog", 1001, 667, 0, true}};
    for (const Kind &k : kinds) {
        const int w = k.w, h = k.h;
        Bytes px((size_t)w * h);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                const unsigned t = ((unsigned)x * 2654435761u ^ (unsigned)y * 40503u) >> 28;   // 0..15
                const double v = x * 200 / std::max(w - 1, 1) + (y / 16 % 2) * 40 + std::sin(x / 9.0) * 15 + t;
                px[(size_t)y * w + x] = (uint8_t)std::min(std::max(v, 0.0), 255.0);
            }
        const std::string name = std::string(k.name) + "_" + std::to_string(w) + "x" + std::to_string(h);
        Bytes jpg = compress(px, w, h, k.restart, k.progressive);
        const long hv = sof_hv(jpg);
        const Bytes ref = decode(jpg, w, h);
        if (hv < 0 || jpg[hv] != 0x11 || ref.empty() || !write_file(dir + "/" + name + ".gray", ref)) {
            fprintf(stderr, "%s: no 1x1 one-component SOF, or decode / write failed\n", name.c_str());
            return 1;
        }
        for (int tag : {0x11, 0x22, 0x21, 0x12}) {
            jpg[hv] = (uint8_t)tag;
            char suffix[8];
            snprintf(suffix, sizeof suffix, "_%x.jpg", tag);
            if (decode(jpg, w, h) != ref || !write_file(dir + "/" + name + suffix, jpg)) {
                fprintf(stderr, "%s%s: libjpeg-turbo's decode differs from the 1x1 file's, or write failed\n",
                        name.c_str(), suffix);
                return 1;
            }
        }
        printf("%s %d %d\n", name.c_str(), w, h);
    }
    return 0;
}
