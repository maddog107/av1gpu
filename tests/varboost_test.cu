// varboost_test - GPU variance boost vs the host reference (same av1f/tuneiq.h functions).
//  1. LUT: device double math == host for every base 1..255 and variance 0..1023
//  2. random planes (smooth / noisy / flat mixes), random sizes and tilings, all three rounding
//     orders: GPU sb_var / sb_target / sb_qindex == host loops
// (libaom equivalence of the model itself is shown by py/validate_tuneiq.py on real encodes, and
//  py/check_varboost_gpu.py runs this kernel on the dev images against the Python model.)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "av1f/tuneiq.h"
#include "av1f/varboost_gpu.h"
extern "C" int av1f_vb_build_luts_device(uint8_t *d_luts, int strength, cudaStream_t s);

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint32_t rnd() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)rng; }

int main() {
    // ---- 1. LUT
    uint8_t *dl;
    cudaMalloc(&dl, 256 * 1024);
    long lutBad = 0;
    long lutDevBad = 0;
    for (int strength : {100, 50, 200}) {
        av1f_vb_build_luts_device(dl, strength, 0);
        std::vector<uint8_t> gd(256 * 1024);
        cudaMemcpy(gd.data(), dl, gd.size(), cudaMemcpyDeviceToHost);
        for (int b = 1; b < 256; b++)
            for (int v = 0; v < 1024; v++)
                if (gd[b * 1024 + v] != av1f_varboost_qindex(b, v, strength)) {
                    if (lutDevBad < 4) printf("  device-math LUT differs: strength %d base %d var %u: %d vs host %d\n", strength, b, v,
                                              gd[b * 1024 + v], av1f_varboost_qindex(b, v, strength));
                    lutDevBad++;
                }
        av1f_vb_build_luts(dl, strength, 0);
        std::vector<uint8_t> g(256 * 1024);
        cudaMemcpy(g.data(), dl, g.size(), cudaMemcpyDeviceToHost);
        for (int b = 1; b < 256; b++)
            for (int v = 0; v < 1024; v++) lutBad += g[b * 1024 + v] != av1f_varboost_qindex(b, v, strength);
    }
    cudaFree(dl);
    printf("LUT (3 strengths x 255 bases x 1024 variances): device-math %ld differ (informational), uploaded %ld differ\n",
           lutDevBad, lutBad);

    // ---- 2. random planes
    long bad = 0, nsbTot = 0, boosted = 0;
    for (int it = 0; it < 60; it++) {
        const int W = 1 + rnd() % 1500, H = 1 + rnd() % 1200;
        const int cols = (((W + 7) >> 3) * 8 + 63) >> 6, rows = (((H + 7) >> 3) * 8 + 63) >> 6;
        const int pitch = cols * 64 + 64 * (rnd() % 2);
        std::vector<uint8_t> y((size_t)pitch * rows * 64);
        const int kind = rnd() % 3;
        for (int r = 0; r < rows * 64; r++)
            for (int c = 0; c < pitch; c++) {
                int v;
                if (kind == 0) v = 128 + (int)(40 * __builtin_sin(r * 0.05) * __builtin_cos(c * 0.03)) + (int)(rnd() % 7) - 3;
                else if (kind == 1) v = (r / 64 + c / 64) % 3 == 0 ? 100 + rnd() % (1 + (r * 7 + c) % 60) : 90 + (rnd() % 3);
                else v = rnd() % 256;
                y[(size_t)r * pitch + c] = (uint8_t)av1f_clampi(v, 0, 255);
            }
        const int base = 1 + rnd() % 255, tcl2 = rnd() % 3, trl2 = rnd() % 3;
        const int strength = rnd() % 4 == 0 ? 150 : 100;
        const uint32_t n = cols * rows;
        // host reference
        std::vector<uint32_t> hv(n);
        std::vector<int> ht(n);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) {
                unsigned v[64];
                for (int k = 0; k < 64; k++) {
                    int s = 0;
                    unsigned q = 0;
                    for (int i = 0; i < 8; i++)
                        for (int j = 0; j < 8; j++) {
                            const int x = y[(size_t)(r * 64 + (k >> 3) * 8 + i) * pitch + c * 64 + (k & 7) * 8 + j];
                            s += x;
                            q += x * x;
                        }
                    v[k] = av1f_var8x8_from_sums(s, q);
                }
                hv[r * cols + c] = av1f_sb_variance(v);
                ht[r * cols + c] = av1f_varboost_qindex(base, hv[r * cols + c], strength);
            }
        for (int order = 0; order < 3; order++) {
            std::vector<uint32_t> gv(n);
            std::vector<uint8_t> gt(n), gq(n);
            if (av1f_varboost_host(y.data(), pitch, W, H, base, tcl2, trl2, order, strength, gv.data(), gt.data(), gq.data())) {
                printf("kernel error\n");
                return 1;
            }
            // host rounding
            std::vector<int> hq(n);
            const int res = av1f_delta_q_res(base);
            if (order == AV1F_VB_NEAREST) {
                for (uint32_t i = 0; i < n; i++) {
                    int best = -1;
                    for (int q = 1; q <= 255; q++)
                        if ((q - base) % res == 0 && (best < 0 || abs(q - ht[i]) < abs(best - ht[i]) ||
                                                      (abs(q - ht[i]) == abs(best - ht[i]) && q > best)))
                            best = q;
                    hq[i] = best;
                }
            } else {
                const int tw = (cols + (1 << tcl2) - 1) >> tcl2, th = (rows + (1 << trl2) - 1) >> trl2;
                for (int tr = 0; tr * th < rows; tr++)
                    for (int tc = 0; tc * tw < cols; tc++) {
                        int prev = base;
                        for (int r = tr * th; r < std::min(rows, tr * th + th); r++) {
                            if (order == AV1F_VB_ROW) prev = base;
                            for (int c = tc * tw; c < std::min(cols, tc * tw + tw); c++) {
                                prev = av1f_adjust_q(res, prev, ht[r * cols + c]);
                                hq[r * cols + c] = prev;
                            }
                        }
                    }
            }
            for (uint32_t i = 0; i < n; i++) {
                const bool b = gv[i] != hv[i] || gt[i] != ht[i] || gq[i] != hq[i];
                if (b && bad < 5)
                    printf("  mismatch it %d order %d sb %u: var %u/%u target %d/%d q %d/%d\n", it, order, i, gv[i], hv[i], gt[i],
                           ht[i], gq[i], hq[i]);
                bad += b;
                boosted += order == 0 && hq[i] != base;
            }
            nsbTot += n;
        }
    }
    printf("random planes: %ld SB results (3 orders), %ld boosted (TILE order), %ld differ\n", nsbTot, boosted, bad);
    const bool ok = !lutBad && !bad;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
