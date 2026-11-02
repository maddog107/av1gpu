// Gray JPEGs for jpeg_gray_sampling_test.py (#44).
//
// libjpeg-turbo writes one-component JPEGs of a synthetic picture (gradient, 16-row bands, a sine and a little
// texture): baseline at 37x29, 640x480, 1001x667 and 2000x1500, with a restart interval of 5 MCUs and progressive at
// 640x480 and 1001x667, all at quality 90.  Each is written as <name>_11.jpg and again with the component's sampling
// factors in the SOF set to 2x2, 2x1 and 1x2 (<name>_22/_21/_12.jpg).  That is legal: a one-component scan is
// non-interleaved, so its MCU is one block whatever the factors say.  <name>.gray holds libjpeg-turbo's decode (islow,
// W x H bytes); the decode of every tagged file must equal it, or the generator fails.  One line per name on stdout:
// "<name> <W> <H>".
//
//   jpeg_gray_gen OUTDIR
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <jpeglib.h>

typedef std::vector<uint8_t> Bytes;

static Bytes compress(const Bytes &px, int w, int h, int restart, bool progressive) {
    jpeg_compress_struct ci;
    jpeg_error_mgr err;
    ci.err = jpeg_std_error(&err);
    jpeg_create_compress(&ci);
    unsigned char *buf = nullptr;
    unsigned long n = 0;
    jpeg_mem_dest(&ci, &buf, &n);
    ci.image_width = w;
    ci.image_height = h;
    ci.input_components = 1;
    ci.in_color_space = JCS_GRAYSCALE;
    jpeg_set_defaults(&ci);
    jpeg_set_quality(&ci, 90, TRUE);
    ci.restart_interval = restart;
    if (progressive) jpeg_simple_progression(&ci);
    jpeg_start_compress(&ci, TRUE);
    while (ci.next_scanline < ci.image_height) {
        JSAMPROW row = (JSAMPROW)&px[(size_t)ci.next_scanline * w];
        jpeg_write_scanlines(&ci, &row, 1);
    }
    jpeg_finish_compress(&ci);
    jpeg_destroy_compress(&ci);
    Bytes out(buf, buf + n);
    free(buf);
    return out;
}

static Bytes decode(const Bytes &jpg, int w, int h) {
    jpeg_decompress_struct ci;
    jpeg_error_mgr err;
    ci.err = jpeg_std_error(&err);
    jpeg_create_decompress(&ci);
    jpeg_mem_src(&ci, jpg.data(), (unsigned long)jpg.size());
    jpeg_read_header(&ci, TRUE);
    ci.dct_method = JDCT_ISLOW;
    ci.out_color_space = JCS_GRAYSCALE;
    jpeg_start_decompress(&ci);
    Bytes px;
    if ((int)ci.output_width == w && (int)ci.output_height == h && ci.output_components == 1) {
        px.resize((size_t)w * h);
        while (ci.output_scanline < ci.output_height) {
            JSAMPROW row = &px[(size_t)ci.output_scanline * w];
            jpeg_read_scanlines(&ci, &row, 1);
        }
        jpeg_finish_decompress(&ci);
    }
    jpeg_destroy_decompress(&ci);
    return px;
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

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: jpeg_gray_gen OUTDIR\n");
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
