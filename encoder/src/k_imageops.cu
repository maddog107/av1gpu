// ImageMagick-style operations on the decoded planes (imageops.h): the --trim measurement.
#include "imageops.h"

#include <algorithm>
#include <climits>

namespace {

struct Rgba {
    float r, g, b, a;
};

// 4:2:0: chroma at luma (x, y) of a w x h plane, as libjpeg-turbo's fancy upsampling gives it, and so ImageMagick's
// decode of a JPEG: a triangle filter, with its rounding.  (x, y) may be in the padding, which replicates the edges.
template <typename T>
__device__ int chroma_up(const T *p, int pitch, int w, int h, int x, int y) {
    const int cx = min(x >> 1, w - 1), cy = min(y >> 1, h - 1);
    const int nx = max(0, min(cx + ((x & 1) ? 1 : -1), w - 1)), ny = max(0, min(cy + ((y & 1) ? 1 : -1), h - 1));
    const T *r0 = p + (size_t)cy * pitch, *r1 = p + (size_t)ny * pitch;
    const int here = 3 * r0[cx] + r1[cx], side = 3 * r0[nx] + r1[nx];
    return (3 * here + side + ((x & 1) ? 7 : 8)) >> 4;
}

// The pixel at (x, y) of f (luma coordinates) as normalised RGB, alpha 1 without an alpha plane.  4:2:0 chroma:
// upsampled with f.up, else (or with nearest) the sample covering the pixel.  clip: YCbCr can leave RGB's range where chroma is shared at a
// colour edge; false keeps that, so that a linear operation stays exact.
template <typename T>
__device__ Rgba trim_px(const GpTrimFrame &f, int x, int y, bool clip = true, bool nearest = false) {
    const T *P0 = (const T *)f.p[0];
    const float Y = (float)P0[(size_t)y * f.pitch[0] + x];
    Rgba o;
    o.a = f.alpha ? (float)((const T *)f.alpha)[(size_t)y * f.apitch + x] / (sizeof(T) == 2 ? 1023.f : 255.f) : 1.f;
    if (f.mono) {
        o.r = o.g = o.b = Y / f.scale;
        return o;
    }
    const int s = f.yuv444 ? 0 : 1;
    const T *P1 = (const T *)f.p[1], *P2 = (const T *)f.p[2];
    const bool up = s && f.up && !nearest;
    const float U = up ? (float)chroma_up(P1, f.pitch[1], f.cw, f.ch, x, y) : (float)P1[(size_t)(y >> s) * f.pitch[1] + (x >> s)];
    const float V = up ? (float)chroma_up(P2, f.pitch[2], f.cw, f.ch, x, y) : (float)P2[(size_t)(y >> s) * f.pitch[2] + (x >> s)];
    if (f.matrix == 1) {   // identity: Y = G, U = B, V = R
        o.g = Y / f.scale; o.b = U / f.scale; o.r = V / f.scale;
        return o;
    }
    if (f.matrix == 2) {   // YCgCo-Re (H.273 formulas 58..61 inverted; Cg / Co stored + 512)
        const int cg = (int)U - 512, co = (int)V - 512, t = (int)Y - (cg >> 1), G = cg + t, B = t - (co >> 1), R = B + co;
        o.r = R / f.scale; o.g = G / f.scale; o.b = B / f.scale;
        return o;
    }
    // BT.601 full range, as the decoders write it (8-bit: chroma centred on 128; 10-bit: c * 1023 + 512)
    const float mid = sizeof(T) == 2 ? 512.f : 128.f;
    const float yy = Y / f.scale, cb = (U - mid) / f.scale, cr = (V - mid) / f.scale;
    o.r = yy + 1.402f * cr;
    o.g = yy - 0.344136286f * cb - 0.714136286f * cr;
    o.b = yy + 1.772f * cb;
    if (clip) {
        o.r = fminf(fmaxf(o.r, 0.f), 1.f);
        o.g = fminf(fmaxf(o.g, 0.f), 1.f);
        o.b = fminf(fmaxf(o.b, 0.f), 1.f);
    }
    return o;
}

// ImageMagick's IsFuzzyEquivalencePixelInfo as -trim uses it, measured on ImageMagick 7.1.2: the colours differ when
// the sum of the squared channel differences (0..1 each) exceeds 6 fuzz^2 (thr); with alpha, alphas that differ at
// all differ, and the colour distance is weighted by the product of the alphas (two transparent pixels never differ).
__device__ bool trim_differs(const Rgba &p, const Rgba &q, float thr) {
    float s = 1.f;
    if (p.a != 1.f || q.a != 1.f) {
        if (p.a != q.a) return true;
        s = p.a * q.a;
        if (s <= 1e-6f) return false;
    }
    const float dr = p.r - q.r, dg = p.g - q.g, db = p.b - q.b;
    return s * (dr * dr + dg * dg + db * db) > thr;
}

constexpr int kTrimRows = 4;   // rows per block

// The pixel as ImageMagick compares it: RGB from YCbCr planes rounded to whole samples of their depth, as a decoder
// writes them (near the --fuzz threshold the unrounded value decides differently, and ImageMagick has only the integers)
template <typename T>
__device__ Rgba trim_rgb(const GpTrimFrame &f, int x, int y) {
    Rgba p = trim_px<T>(f, x, y);
    if (f.matrix == 0) {
        p.r = rintf(p.r * f.scale) / f.scale;
        p.g = rintf(p.g * f.scale) / f.scale;
        p.b = rintf(p.b * f.scale) / f.scale;
    }
    return p;
}

// pass 0: left / top (pixels differing from the top-left corner), right (from the top-right) and bottom (from the
// bottom-left).  pass 1, after pass 0: ImageMagick's fourth rule, a pixel left of the right edge that differs from the
// bottom-right corner extends the bottom too (GetImageBoundingBox; it counts where the bottom-left corner's colour runs
// along the bottom edge and the bottom-right's does not)
template <typename T>
__global__ void k_trim_measure(const GpTrimFrame *frames, float thr, GpTrimBox *boxes, int pass) {
    const GpTrimFrame f = frames[blockIdx.y];
    const int y0 = blockIdx.x * kTrimRows;
    if (y0 >= f.h) return;
    __shared__ Rgba corner[4];
    __shared__ int sL, sT, sR, sB;
    if (threadIdx.x < 4) {
        const int cx = f.x0 + ((threadIdx.x & 1) ? f.w - 1 : 0), cy = f.y0 + ((threadIdx.x & 2) ? f.h - 1 : 0);
        corner[threadIdx.x] = trim_rgb<T>(f, cx, cy);
    }
    if (threadIdx.x == 0) { sL = INT_MAX; sT = INT_MAX; sR = -1; sB = -1; }
    __syncthreads();
    int l = INT_MAX, t = INT_MAX, r = -1, b = -1;
    const int right = pass ? boxes[blockIdx.y].right : 0;
    for (int y = y0; y < min(y0 + kTrimRows, f.h); y++)
        for (int x = threadIdx.x; x < f.w; x += blockDim.x) {
            const Rgba p = trim_rgb<T>(f, f.x0 + x, f.y0 + y);
            if (pass) {
                if (x < right && trim_differs(p, corner[3], thr)) b = max(b, y);
                continue;
            }
            if (trim_differs(p, corner[0], thr)) { l = min(l, x); t = min(t, y); }
            if (trim_differs(p, corner[1], thr)) r = max(r, x);
            if (trim_differs(p, corner[2], thr)) b = max(b, y);
        }
    if (l != INT_MAX) { atomicMin(&sL, l); atomicMin(&sT, t); }
    if (r >= 0) atomicMax(&sR, r);
    if (b >= 0) atomicMax(&sB, b);
    __syncthreads();
    GpTrimBox *o = &boxes[blockIdx.y];
    if (threadIdx.x == 0) {
        if (sL != INT_MAX) { atomicMin(&o->left, sL); atomicMin(&o->top, sT); }
        if (sR >= 0) atomicMax(&o->right, sR);
        if (sB >= 0) atomicMax(&o->bottom, sB);
    }
    if (!pass && blockIdx.x == 0 && threadIdx.x < 4) {
        const Rgba &c = corner[threadIdx.x];
        o->light[threadIdx.x] = 50.f * (fmaxf(c.r, fmaxf(c.g, c.b)) + fminf(c.r, fminf(c.g, c.b)));
    }
}

// --grayscale / --colorspace Gray: the gray value of each pixel written over its Y sample (in place: each thread reads
// the samples of its own pixel and writes only its Y), over the padded aw x ah plane, whose padding replicates the
// edges in every plane alike.  method: GpGray.
template <typename T>
__global__ void k_gray(GpTrimFrame f, int aw, int ah, int method) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y + blockIdx.z * gridDim.y;   // (tone_grid)
    if (x >= aw || y >= ah) return;
    const Rgba p = trim_px<T>(f, x, y);
    float g;
    switch (method) {
        case GP_GRAY_REC601: g = 0.298839f * p.r + 0.586811f * p.g + 0.114350f * p.b; break;
        case GP_GRAY_REC709: g = 0.212656f * p.r + 0.715158f * p.g + 0.072186f * p.b; break;
        case GP_GRAY_AVERAGE: g = (p.r + p.g + p.b) / 3.f; break;
        case GP_GRAY_BRIGHTNESS: g = fmaxf(p.r, fmaxf(p.g, p.b)); break;
        case GP_GRAY_LIGHTNESS: g = 0.5f * (fmaxf(p.r, fmaxf(p.g, p.b)) + fminf(p.r, fminf(p.g, p.b))); break;
        default: g = sqrtf((p.r * p.r + p.g * p.g + p.b * p.b) / 3.f); break;   // RMS
    }
    const float m = sizeof(T) == 2 ? 1023.f : 255.f;
    ((T *)f.p[0])[(size_t)y * f.pitch[0] + x] = (T)fminf(fmaxf(g * m + 0.5f, 0.f), m);
}

// --auto-level: the smallest and largest R, G, B sample in f's window (float bits: non-negative floats order as ints)
template <typename T>
__global__ void k_tone_range(GpTrimFrame f, int *range) {
    const int y = blockIdx.y + blockIdx.z * gridDim.y;
    if (y >= f.h) return;
    int lo = 0x7f7fffff, hi = 0;
    for (int x = blockIdx.x * blockDim.x + threadIdx.x; x < f.w; x += gridDim.x * blockDim.x) {
        const Rgba p = trim_px<T>(f, f.x0 + x, f.y0 + y);
        lo = min(lo, __float_as_int(fminf(p.r, fminf(p.g, p.b))));
        hi = max(hi, __float_as_int(fmaxf(p.r, fmaxf(p.g, p.b))));
    }
    atomicMin(&range[0], lo);
    atomicMax(&range[1], hi);
}

// --level / --gamma / --negate (and --auto-level's stretch), per RGB channel: v = clamp(v * s + o, 0, 1) ^ e, then 1 - v.
// --negate alone does not clamp: it is linear, so it is exact on Y and chroma even where v left 0..1 (trim_px)
__device__ float tone_v(float v, const GpToneMap &m) {
    if (m.s != 1.f || m.o != 0.f || m.e != 1.f) {
        v = fminf(fmaxf(v * m.s + m.o, 0.f), 1.f);
        if (m.e != 1.f) v = powf(v, m.e);
    }
    return m.negate ? 1.f - v : v;
}
// RGB (0..1) -> the frame's samples: YCbCr (BT.601 full range as the decoders write it; chroma unrounded, the caller
// averages 4:2:0 chroma), identity / GBR, or YCgCo-Re
template <typename T>
__device__ void tone_store(const GpTrimFrame &f, int x, int y, const Rgba &p, float *cb, float *cr) {
    const float m = f.scale;
    auto q = [&](float v, float hi) { return (T)fminf(fmaxf(v + 0.5f, 0.f), hi); };
    const float top = sizeof(T) == 2 ? 1023.f : 255.f;
    T *Y = (T *)f.p[0] + (size_t)y * f.pitch[0] + x;
    if (f.mono) { *Y = q(p.r * m, top); return; }
    if (f.matrix == 1) {
        *Y = q(p.g * m, top); *cb = p.b * m; *cr = p.r * m;
        return;
    }
    if (f.matrix == 2) {
        const int R = (int)(p.r * 255.f + 0.5f), G = (int)(p.g * 255.f + 0.5f), B = (int)(p.b * 255.f + 0.5f);
        const int co = R - B, t = B + (co >> 1), cg = G - t;
        *Y = (T)(t + (cg >> 1)); *cb = (float)(cg + 512); *cr = (float)(co + 512);
        return;
    }
    const float mid = sizeof(T) == 2 ? 512.f : 128.f;
    *Y = q((0.299f * p.r + 0.587f * p.g + 0.114f * p.b) * m, top);
    *cb = (-0.168735892f * p.r - 0.331264108f * p.g + 0.5f * p.b) * m + mid;
    *cr = (0.5f * p.r - 0.418687589f * p.g - 0.081312411f * p.b) * m + mid;
}
// in place over the padded aw x ah planes: one thread per pixel (4:4:4, monochrome) or per 2x2 block (4:2:0: its four
// pixels share one chroma sample).  f.p[1], f.p[2] are what to read: with f.up a copy, as a block reads its neighbours'
// samples while they are written; du, dv what to write.  With f.up each luma comes from the pixel's upsampled RGB, as a
// decoder sees it; the block's chroma gets the mean of its pixels' new chroma from the block's own sample, so that an
// operation that leaves a colour alone leaves its sample alone
template <typename T>
__global__ void k_tone(GpTrimFrame f, int aw, int ah, GpToneMap m, T *du, T *dv) {
    const int s = (f.mono || f.yuv444) ? 0 : 1;
    const int cx = blockIdx.x * blockDim.x + threadIdx.x, cy = blockIdx.y + blockIdx.z * gridDim.y;
    if (cx >= (aw >> s) || cy >= (ah >> s)) return;
    float cb = 0.f, cr = 0.f;
    for (int k = 0; k < (1 << (2 * s)); k++) {
        const int x = (cx << s) + (k & 1), y = (cy << s) + (k >> 1);
        Rgba p = trim_px<T>(f, x, y, false, true);
        const bool up = s && f.up;
        Rgba u = p;
        if (up) u = trim_px<T>(f, x, y, false);   // (before Y is written)
        p.r = tone_v(p.r, m); p.g = tone_v(p.g, m); p.b = tone_v(p.b, m);
        float b1 = 0.f, r1 = 0.f;
        tone_store<T>(f, x, y, p, &b1, &r1);
        cb += b1;
        cr += r1;
        if (up) {
            u.r = tone_v(u.r, m); u.g = tone_v(u.g, m); u.b = tone_v(u.b, m);
            tone_store<T>(f, x, y, u, &b1, &r1);
        }
    }
    if (f.mono) return;
    const float n = (float)(1 << (2 * s)), top = sizeof(T) == 2 ? 1023.f : 255.f;
    du[(size_t)cy * f.pitch[1] + cx] = (T)fminf(fmaxf(cb / n + 0.5f, 0.f), top);
    dv[(size_t)cy * f.pitch[2] + cx] = (T)fminf(fmaxf(cr / n + 0.5f, 0.f), top);
}

// One block per row: a grid dimension holds at most 65535, and a padded plane can have 65536 rows
dim3 tone_grid(int cols, int rows) { return dim3(cols, std::min(rows, 32768), (rows + 32767) / 32768); }

}  // namespace

void gp_launch_tone_range(const GpTrimFrame &f, int *dRange, cudaStream_t st) {
    if (f.w <= 0 || f.h <= 0) return;
    const dim3 grid = tone_grid(std::min((f.w + 255) / 256, 8), f.h);
    if (f.in10) k_tone_range<uint16_t><<<grid, 256, 0, st>>>(f, dRange);
    else k_tone_range<uint8_t><<<grid, 256, 0, st>>>(f, dRange);
}

void gp_launch_tone(const GpTrimFrame &f, int aw, int ah, const GpToneMap &m, void *du, void *dv, cudaStream_t st) {
    if (aw <= 0 || ah <= 0) return;
    const int s = (f.mono || f.yuv444) ? 0 : 1;
    const dim3 grid = tone_grid(((aw >> s) + 127) / 128, ah >> s);
    if (f.in10) k_tone<uint16_t><<<grid, 128, 0, st>>>(f, aw, ah, m, (uint16_t *)du, (uint16_t *)dv);
    else k_tone<uint8_t><<<grid, 128, 0, st>>>(f, aw, ah, m, (uint8_t *)du, (uint8_t *)dv);
}

void gp_launch_gray(const GpTrimFrame &f, int aw, int ah, int method, cudaStream_t st) {
    if (aw <= 0 || ah <= 0) return;
    const dim3 grid = tone_grid((aw + 127) / 128, ah);
    if (f.in10) k_gray<uint16_t><<<grid, 128, 0, st>>>(f, aw, ah, method);
    else k_gray<uint8_t><<<grid, 128, 0, st>>>(f, aw, ah, method);
}

void gp_launch_trim_measure(const GpTrimFrame *dFrames, int n, int maxH, float thr, GpTrimBox *dBoxes, int in10, cudaStream_t st) {
    if (n <= 0 || maxH <= 0) return;
    const dim3 grid((maxH + kTrimRows - 1) / kTrimRows, n);
    for (int pass = 0; pass < 2; pass++) {
        if (in10) k_trim_measure<uint16_t><<<grid, 256, 0, st>>>(dFrames, thr, dBoxes, pass);
        else k_trim_measure<uint8_t><<<grid, 256, 0, st>>>(dFrames, thr, dBoxes, pass);
    }
}
