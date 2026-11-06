// ImageMagick-style operations on the decoded planes (imageops.h): option parsing, the --trim measurement's launch
// and the decision taken on it.
#include "imageops.h"

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdlib>
#include <cstring>

namespace {
// a number at s ("12", "12.5"; no sign, no exponent), s moved past it
bool io_num(const char *&s, double &v) {
    const char *b = s;
    while ((*s >= '0' && *s <= '9') || *s == '.') s++;
    if (s == b) return false;
    char *e = nullptr;
    v = strtod(std::string(b, s).c_str(), &e);
    return e && !*e;
}
}  // namespace

bool gp_parse_percent(const char *v, double *pct, std::string *why) {
    const char *s = v;
    double p = 0;
    if (!io_num(s, p) || (*s == '%' && *++s) || *s || p > 100) {
        if (why) *why = "expected a percentage 0..100";
        return false;
    }
    *pct = p;
    return true;
}

bool gp_parse_fuzz(const char *v, GpTrim &t, std::string *why) {
    const char *s = v;
    double d = 0;
    if (!io_num(s, d)) {
        if (why) *why = "expected N% (percent of the largest colour distance) or N (ImageMagick Q16 units, 0..65535)";
        return false;
    }
    const bool pct = *s == '%';
    if (pct) s++;
    if (*s || (pct ? d > 100 : d > 65535)) {
        if (why) *why = pct ? "the percentage is above 100" : "expected N% or N (Q16 units, 0..65535)";
        return false;
    }
    t.fuzz = pct ? d / 100 : d / 65535;
    return true;
}

bool gp_parse_trim_margin(const char *v, GpTrim &t, std::string *why) {
    const char *s = v;
    double d = 0;
    if (!io_num(s, d) || (*s == '%' ? *(s + 1) != 0 : *s != 0) || (*s == '%' ? d > 100 : d != std::floor(d) || d > (1 << 20))) {
        if (why) *why = "expected whole pixels N or a percentage N% of the shorter side";
        return false;
    }
    if (*s == '%') { t.marginPct = d; t.marginPx = 0; }
    else { t.marginPx = (int)d; t.marginPct = 0; }
    return true;
}

bool gp_parse_trim_limits(const char *v, GpTrim &t, std::string *why) {
    const char *s = v;
    double lo = 0, hi = 0;
    bool ok = io_num(s, lo);
    if (ok && *s == '%') s++;
    ok = ok && *s == ',';
    if (ok) s++;
    ok = ok && io_num(s, hi);
    if (ok && *s == '%') s++;
    if (!ok || *s || lo > hi || hi > 100) {
        if (why) *why = "expected MIN%,MAX% with 0 <= MIN <= MAX <= 100";
        return false;
    }
    t.limitMin = lo;
    t.limitMax = hi;
    return true;
}


bool gp_parse_level(const char *v, GpTone &t, std::string *why) {
    const char *s = v;
    double n[3] = {0, 0, 1};
    bool pct[2] = {false, false};
    int k = 0;
    bool ok = true;
    for (; ok && k < 3; k++) {
        ok = io_num(s, n[k]);
        if (ok && k < 2 && *s == '%') { pct[k] = true; s++; }
        if (!*s) { k++; break; }
        ok = ok && (*s == ',' || *s == 'x');
        s++;
    }
    if (!ok || *s) {
        if (why) *why = "expected BLACK[%][,WHITE[%][,GAMMA]] (ImageMagick's -level)";
        return false;
    }
    auto frac = [&](int i) { return pct[i] ? n[i] / 100 : n[i] / 65535; };
    t.black = frac(0);
    t.white = k >= 2 ? frac(1) : 1 - t.black;
    t.levelGamma = k >= 3 ? n[2] : 1;
    if (!(t.white > t.black) || t.white > 1 || !(t.levelGamma > 0)) {
        if (why) *why = "the white point must be above the black point (both 0..100%), and the gamma above 0";
        return false;
    }
    t.level = true;
    return true;
}

GpToneMap gp_tone_map(const GpTone &t, float lo, float hi) {
    // auto-level: v -> (v - lo) / (hi - lo); level: v -> (v - B) / (W - B); composed into one scale and offset
    double s = 1, o = 0;
    if (t.autoLevel && hi > lo) { s = 1.0 / (hi - lo); o = -lo * s; }
    if (t.level) {
        const double r = 1.0 / (t.white - t.black);
        s *= r;
        o = (o - t.black) * r;
    }
    GpToneMap m;
    m.s = (float)s;
    m.o = (float)o;
    m.e = (float)((t.level ? 1.0 / t.levelGamma : 1.0) / t.gamma);
    m.negate = t.negate;
    return m;
}

int gp_parse_gray(const char *v) {
    static const char *const names[] = {"rec601luma", "rec709luma", "average", "brightness", "lightness", "rms"};
    std::string l(v);
    for (char &c : l) c = (char)tolower((unsigned char)c);
    for (int i = 0; i < 6; i++)
        if (l == names[i]) return GP_GRAY_REC601 + i;
    return GP_GRAY_NONE;
}

int gp_trim_measure(const std::vector<GpTrimFrame> &frames, const GpTrim &t, cudaStream_t st, std::vector<GpTrimBox> &boxes,
                    std::string &err) {
    const size_t n = frames.size();
    boxes.assign(n, GpTrimBox{INT_MAX, INT_MAX, -1, -1, {0, 0, 0, 0}});
    if (!n) return 0;
    // frames grouped by sample size (one kernel instantiation each): 8-bit first, then 10-bit
    std::vector<size_t> order;
    for (int g = 0; g < 2; g++)
        for (size_t i = 0; i < n; i++)
            if (frames[i].in10 == g) order.push_back(i);
    std::vector<GpTrimFrame> hf(n);
    std::vector<GpTrimBox> hb(boxes);
    for (size_t k = 0; k < n; k++) hf[k] = frames[order[k]];
    GpTrimFrame *df = nullptr;
    GpTrimBox *db = nullptr;
    cudaError_t e = cudaMallocAsync((void **)&df, n * sizeof(GpTrimFrame), st);
    if (e == cudaSuccess) e = cudaMallocAsync((void **)&db, n * sizeof(GpTrimBox), st);
    if (e == cudaSuccess) e = cudaMemcpyAsync(df, hf.data(), n * sizeof(GpTrimFrame), cudaMemcpyHostToDevice, st);
    if (e == cudaSuccess) e = cudaMemcpyAsync(db, hb.data(), n * sizeof(GpTrimBox), cudaMemcpyHostToDevice, st);
    if (e == cudaSuccess) {
        // 6 fuzz^2: ImageMagick's threshold on the summed squared RGB differences (k_imageops.cu)
        const float thr = (float)(6.0 * t.fuzz * t.fuzz);
        for (size_t k0 = 0; k0 < n;) {
            size_t k1 = k0;
            int maxH = 0;
            while (k1 < n && hf[k1].in10 == hf[k0].in10) maxH = std::max(maxH, hf[k1++].h);
            gp_launch_trim_measure(df + k0, (int)(k1 - k0), maxH, thr, db + k0, hf[k0].in10, st);
            k0 = k1;
        }
        e = cudaGetLastError();
    }
    if (e == cudaSuccess) e = cudaMemcpyAsync(hb.data(), db, n * sizeof(GpTrimBox), cudaMemcpyDeviceToHost, st);
    if (df) cudaFreeAsync(df, st);
    if (db) cudaFreeAsync(db, st);
    if (e == cudaSuccess) e = cudaStreamSynchronize(st);
    if (e != cudaSuccess) {
        cudaGetLastError();
        err = std::string("CUDA: ") + cudaGetErrorString(e) + " (--trim)";
        return -1;
    }
    for (size_t k = 0; k < n; k++) boxes[order[k]] = hb[k];
    return 0;
}

int gp_trim_decide(const GpTrim &t, int w, int h, const GpTrimBox &b, int *kx, int *ky, int *kw, int *kh) {
    *kx = 0; *ky = 0; *kw = w; *kh = h;
    // ImageMagick's box: no pixel differing from a corner leaves that side at the far edge, so an image that is all
    // border colour has none (0x0)
    const int left = b.left == INT_MAX ? w : b.left, top = b.top == INT_MAX ? h : b.top;
    const int right = std::max(b.right, 0), bottom = std::max(b.bottom, 0);
    int bw = right - left + 1, bh = bottom - top + 1;
    if ((b.right < 0 && b.bottom < 0 && b.left == INT_MAX) || bw <= 0 || bh <= 0) bw = bh = 0;
    const int cutW = w - bw, cutH = h - bh;
    auto share = [](int side, double pct) { return (int)std::floor(side * pct / 100.0 + 1e-9); };
    if (t.blankPct >= 0 && (cutW > share(w, t.blankPct) || cutH > share(h, t.blankPct))) return GP_TRIM_BLANK;
    if (bw == 0 || (cutW == 0 && cutH == 0)) return GP_TRIM_WHOLE;
    if (t.limitMin >= 0 && !(share(w, t.limitMin) <= cutW && cutW <= share(w, t.limitMax) && share(h, t.limitMin) <= cutH &&
                             cutH <= share(h, t.limitMax)))
        return GP_TRIM_WHOLE;
    if (t.lightness >= 0) {
        float l[4];
        std::copy(b.light, b.light + 4, l);
        std::sort(l, l + 4);
        if (!(l[1] >= t.lightness)) return GP_TRIM_WHOLE;   // the second-darkest corner: 3 of 4 are at least that light
    }
    const int m = t.marginPx > 0 ? t.marginPx : share(std::min(w, h), t.marginPct);
    *kx = std::max(0, left - m);
    *ky = std::max(0, top - m);
    *kw = std::min(w, left + bw + m) - *kx;
    *kh = std::min(h, top + bh + m) - *ky;
    return *kw == w && *kh == h ? GP_TRIM_WHOLE : GP_TRIM_CUT;
}
