// ImageMagick-style operations on the decoded planes, on the GPU, before the encode (front ends: --trim and its
// options).  Host side in imageops.cpp, kernels in k_imageops.cu.
#ifndef GP_IMAGEOPS_H
#define GP_IMAGEOPS_H

#include <cuda_runtime.h>
#include <stdint.h>
#include <string>
#include <vector>

// --trim: ImageMagick's -trim (with -fuzz), and the conditions under which a trim is kept.
struct GpTrim {
    bool on = false;
    double fuzz = 0;            // --fuzz as a fraction of the largest colour distance (ImageMagick: 10% = 0.10)
    int marginPx = 0;           // --trim-margin N: pixels of the border kept around what the trim keeps ...
    double marginPct = 0;       // ... or N%: percent of the shorter side (floor)
    double limitMin = -1;       // --trim-limits MIN%,MAX%: a trim is kept only when it removes MIN..MAX percent of the
    double limitMax = -1;       //   width and MIN..MAX percent of the height (floor of each); -1: no limits
    double lightness = -1;      // --trim-lightness L: kept only when 3 of the 4 corners have HSL lightness >= L percent
    double blankPct = -1;       // --skip-blank P: an image whose trim removes more than P percent of its width or height
                                //   is not encoded (-1: such an image is encoded whole)
    bool guarded() const { return limitMin >= 0 || lightness >= 0; }
};

// What the trim measured on one frame (inside its crop window, window coordinates): the bounding box of the pixels
// that differ from the corner colours as ImageMagick's GetImageBoundingBox finds it (left / top: first pixel differing
// from the top-left corner, right: last differing from the top-right, bottom: last differing from the bottom-left, or
// from the bottom-right left of the right edge; left = INT_MAX: none differ), compared as the decoder's whole RGB
// samples, and the four corners' HSL lightness in percent (top-left, top-right, bottom-left, bottom-right).
struct GpTrimBox {
    int left, top, right, bottom;
    float light[4];
};

// What the front ends do with a frame after --trim: encode it whole, encode the trimmed crop, or skip it (blank).
enum GpTrimVerdict { GP_TRIM_WHOLE = 0, GP_TRIM_CUT = 1, GP_TRIM_BLANK = 2 };

// Option values.  fuzz: "N%" (percent of the largest distance) or "N" (ImageMagick Q16 units, 0..65535); margin: "N"
// pixels or "N%"; limits: "MIN%,MAX%" (the % signs optional); lightness and blank: a percentage 0..100.  false: *why.
bool gp_parse_fuzz(const char *v, GpTrim &t, std::string *why = nullptr);
bool gp_parse_trim_margin(const char *v, GpTrim &t, std::string *why = nullptr);
bool gp_parse_trim_limits(const char *v, GpTrim &t, std::string *why = nullptr);
bool gp_parse_percent(const char *v, double *pct, std::string *why = nullptr);

// One frame for the measurement: its planes as the encoder takes them (GpFrameIn's layout), the window to measure in
// (luma pixels; the whole frame without a crop), how to read RGB from the planes, and its alpha plane (may be NULL).
struct GpTrimFrame {
    const void *p[3];
    int pitch[3];
    int mono, yuv444, in10;
    int cw, ch;                 // 4:2:0: size of the chroma planes without their padding
    int up;                     // 4:2:0: read chroma upsampled as libjpeg-turbo does (a 4:2:0 JPEG's own chroma), else
                                //   the sample covering the pixel (PNG / WebP planes: av1gpu's 2x2 mean)
    int matrix;                 // 0: BT.601 full-range YCbCr (lossy planes), 1: identity / GBR, 2: YCgCo-Re (lossless)
    float scale;                // RGB = samples / scale (255 or 1023; YCgCo-Re: 255)
    int x0, y0, w, h;           // window
    const void *alpha;
    int apitch;
};

// Measure frames[i] (one launch for all of them, on st, then a wait for st): boxes[i].
int gp_trim_measure(const std::vector<GpTrimFrame> &frames, const GpTrim &t, cudaStream_t st, std::vector<GpTrimBox> &boxes,
                    std::string &err);
// The verdict for a w x h window whose measurement is b, and with GP_TRIM_CUT the kept rectangle (window coordinates,
// the margin added and clipped to the window).
int gp_trim_decide(const GpTrim &t, int w, int h, const GpTrimBox &b, int *kx, int *ky, int *kw, int *kh);

// --grayscale METHOD / --colorspace Gray (= Rec709Luma): ImageMagick's -grayscale methods, on the gamma-encoded RGB.
enum GpGray { GP_GRAY_NONE = 0, GP_GRAY_REC601, GP_GRAY_REC709, GP_GRAY_AVERAGE, GP_GRAY_BRIGHTNESS, GP_GRAY_LIGHTNESS, GP_GRAY_RMS };
// "Rec601Luma", "Rec709Luma", "Average", "Brightness", "Lightness", "RMS" (any case) -> GpGray; 0: unknown
int gp_parse_gray(const char *v);

// --auto-level, --level, --gamma, --negate: ImageMagick's tone operators, per gamma-encoded RGB channel, in this order.
struct GpTone {
    bool autoLevel = false;     // stretch the image's smallest..largest R, G or B sample to 0..1 (ImageMagick: channels synced)
    bool level = false;         // --level B,W[,G]: ((v - B) / (W - B)) ^ (1 / G), fractions of the largest sample
    double black = 0, white = 1, levelGamma = 1;
    double gamma = 1;           // --gamma G: v ^ (1 / G)
    bool negate = false;        // --negate: 1 - v
    bool on() const { return autoLevel || level || gamma != 1 || negate; }
};
// what the kernel applies to each channel: v = clamp(v * s + o, 0, 1) ^ e, then 1 - v with negate
struct GpToneMap {
    float s, o, e;
    int negate;
};
// "B[%][,W[%]][,G]" (also x as the separator; plain numbers in Q16 units, 0..65535; W defaults to 100% - B)
bool gp_parse_level(const char *v, GpTone &t, std::string *why = nullptr);
// the map for t, with --auto-level's measured range lo..hi (0..1)
GpToneMap gp_tone_map(const GpTone &t, float lo = 0.f, float hi = 1.f);

// kernels (k_imageops.cu).  in10: the frames' samples are uint16_t (all n frames of one launch alike); dBoxes must
// hold left = top = INT_MAX, right = bottom = -1 before the launch
void gp_launch_trim_measure(const GpTrimFrame *dFrames, int n, int maxH, float thr, GpTrimBox *dBoxes, int in10, cudaStream_t st);
// --auto-level: atomicMin / atomicMax of the float bits of f's R, G, B samples in its window into dRange[0] / [1]
void gp_launch_tone_range(const GpTrimFrame &f, int *dRange, cudaStream_t st);
// m applied to f's planes (aw x ah luma samples), luma in place; the chroma read from f.p[1], f.p[2] and written to du, dv
// (with f.up they can't be the same planes: k_tone reads neighbouring samples)
void gp_launch_tone(const GpTrimFrame &f, int aw, int ah, const GpToneMap &m, void *du, void *dv, cudaStream_t st);
// f's Y plane (aw x ah samples) replaced by the gray of each pixel (window fields of f unused)
void gp_launch_gray(const GpTrimFrame &f, int aw, int ah, int method, cudaStream_t st);

#endif
