// varboost_test - GPU variance boost vs the host reference (same av1f/tuneiq.h functions).
//  1. LUT: device double math == host for every base 1..255 and variance 0..1023
//  2. random planes (smooth / noisy / flat mixes), random sizes and tilings, all three rounding
//     orders: GPU sb_var / sb_target / sb_qindex == host loops
//  3. tune=iq frame decisions (av1f_tuneiq_frame_init) for 4:2:0, 4:4:4 and monochrome, every base qindex:
//     device == host, and the chroma delta-q / QM levels == libaom v3.13.1 av1_set_quantizer() as
//     transcribed below (4:4:4: chroma AC delta-q +clamp(q/2, 0, 24), DC 0, aom_get_qmlevel_444_chroma)
//  4. 10 bits: the 10-bit LUT (uploaded == host), the host model == libaom v3.13.1's high-bitdepth path transcribed
//     independently below (av1_convert_qindex_to_q / av1_convert_q_to_qindex at AOM_BITS_10: ac_q / 16 over the
//     10-bit table; aom_highbd_10_variance8x8 vs zeros: ROUND_POWER_OF_TWO(sse, 4), ROUND_POWER_OF_TWO(sum, 2));
//     random 10-bit planes GPU == host; tune=iq frame decisions at 10 bits: deblocking level == libaom
//     LPF_PICK_FROM_Q (ROUND_POWER_OF_TWO(q * 20723 + 4060632, 20) - 4, key frame), rdmult ==
//     av1_compute_rd_mult_based_on_qindex (ROUND_POWER_OF_TWO(rdmult, 4)), chroma delta-q / QM / delta_q_res /
//     sharpness / CDEF damping and mode == the 8-bit values (qindex rules)
// (libaom equivalence of the model itself is shown by py/validate_tuneiq.py on real encodes, and
//  py/check_varboost_gpu.py runs this kernel on the dev images against the Python model.)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "av1f/tuneiq.h"
#include "av1f/varboost_gpu.h"
#include "av1g/spec_tables.h"
extern "C" int av1f_vb_build_luts_device(uint8_t *d_luts, int strength, cudaStream_t s);

static uint64_t rng = 0x9E3779B97F4A7C15ull;

__global__ void k_tuneiq(int ss, int mono, av1f_tuneiq_frame *out) {
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b < 256) av1f_tuneiq_frame_init(b, ss, ss, mono, &out[b]);
}
// libaom v3.13.1 (av1/encoder/av1_quantize.c av1_set_quantizer, all-intra, tune=iq, enable_chroma_deltaq = 1;
// av1/common/quant_common.h aom_get_qmlevel_allintra / aom_get_qmlevel_444_chroma), written out independently
static int ref_clamp(int v, int lo, int hi) { return v < lo ? lo : v > hi ? hi : v; }
static int ref_qm_allintra(int q) {
    const int l = q <= 40 ? 10 : q <= 100 ? 9 : q <= 160 ? 8 : q <= 200 ? 7 : q <= 220 ? 6 : q <= 240 ? 5 : 4;
    return ref_clamp(l, 2, 10);
}
static int ref_qm_444_chroma(int q) {
    int l;
    if (q <= 12) l = 10; else if (q <= 24) l = 9; else if (q <= 32) l = 8; else if (q <= 36) l = 7; else if (q <= 44) l = 6;
    else if (q <= 48) l = 5; else if (q <= 56) l = 4; else if (q <= 88) l = 3; else l = 2;
    return ref_clamp(l, 2, 10);
}
static void ref_set_quantizer(int base, bool is444, int *udc, int *uac, int *qmy, int *qmu) {
    int dc = 0, ac = 0;
    if (base) {
        if (!is444) dc = ac = -ref_clamp(base / 2 - 14, 0, 16);   // 4:2:0
        else ac = ref_clamp(base / 2, 0, 24);                     // 4:4:4 (DC stays 0)
    }
    *udc = dc; *uac = ac;
    *qmy = ref_qm_allintra(base);
    *qmu = is444 ? ref_qm_444_chroma(base + ac) : ref_qm_allintra(base + ac);
}
static uint32_t rnd() { rng ^= rng << 13; rng ^= rng >> 7; rng ^= rng << 17; return (uint32_t)rng; }

// ---- libaom v3.13.1 high-bitdepth (AOM_BITS_10) rules, transcribed independently ----
// av1/common/quant_common.c av1_ac/dc_quant_QTX(q, 0, AOM_BITS_10) = ac/dc_qlookup_10_QTX[clamp(q, 0, 255)]
// (== spec Dc/Ac_Qlookup[1], checked against libaom's quant_common.c by the generator scripts)
static int ref_ac10(int q) { return Ac_Qlookup_host[1][ref_clamp(q, 0, 255)]; }
static int ref_dc10(int q) { return Dc_Qlookup_host[1][ref_clamp(q, 0, 255)]; }
// av1/encoder/ratectrl.c av1_convert_qindex_to_q (AOM_BITS_10: / 16.0), av1_convert_q_to_qindex
static double ref_q10(int qindex) { return ref_ac10(qindex) / 16.0; }
static int ref_qindex10(double q) {
    int qindex = 0;
    while (qindex < 255 && ref_q10(qindex) < q) qindex++;
    return qindex;
}
// av1/encoder/allintra_vis.c av1_get_sbq_variance_boost, bit_depth = AOM_BITS_10
static int ref_varboost10(int base, unsigned variance, int pct) {
    double strength = (pct / 100.0) * 3.0;
    strength = strength < 0.0 ? 0.0 : strength > 6.0 ? 6.0 : strength;
    if (variance == 0) variance = 1;
    double qstep_ratio = 0.15 * strength * (-log2((double)variance) + 10.0) + 1.0;
    qstep_ratio = qstep_ratio < 1.0 ? 1.0 : qstep_ratio > 8.0 ? 8.0 : qstep_ratio;
    const double base_q = ref_q10(base);
    const double target_q = base_q / qstep_ratio;
    const int target_qindex = ref_qindex10(target_q);
    int boost = (int)round((base + 544.0) * (base - target_qindex) / 1279.0);
    boost = boost < 80 ? boost : 80;
    const int q = base - boost;
    return q > 1 ? q : 1;
}
// aom_dsp/variance.c aom_highbd_10_variance8x8_c(src, all zeros): highbd_10_variance + max(0, .) ; / 64 (varboost)
static unsigned ref_hbd10_var8x8_div64(const uint16_t *p, int stride) {
    uint64_t sse_long = 0;
    int64_t sum_long = 0;
    for (int i = 0; i < 8; i++)
        for (int j = 0; j < 8; j++) {
            const int d = p[i * stride + j];
            sum_long += d;
            sse_long += (uint32_t)(d * d);
        }
    const uint32_t sse = (uint32_t)((sse_long + 8) >> 4);
    const int sum = (int)((sum_long + 2) >> 2);
    const int64_t var = (int64_t)sse - (((int64_t)sum * sum) / 64);
    return (var >= 0 ? (uint32_t)var : 0) / 64;
}
// av1/encoder/picklpf.c av1_pick_filter_level LPF_PICK_FROM_Q, AOM_BITS_10, KEY_FRAME (before the deltas)
static int ref_lf10(int base) {
    const int q = ref_ac10(base);
    int filt_guess = (q * 20723 + 4060632 + (1 << 19)) >> 20;   // ROUND_POWER_OF_TWO(.., 20)
    filt_guess -= 4;                                             // bit_depth != 8 && KEY_FRAME
    return ref_clamp(filt_guess, 0, 63);
}
// av1/encoder/rd.c av1_compute_rd_mult_based_on_qindex(AOM_BITS_10, KF_UPDATE, q, AOM_TUNE_IQ)
static int ref_rdmult10(int qindex) {
    const int q = ref_dc10(qindex);
    int64_t rdmult = (int64_t)q * q;
    rdmult = (int64_t)((double)rdmult * (3.3 + (0.0015 * (double)q)));
    const int weight = ref_clamp(((255 - qindex) * 3) / 4, 0, 72) + 128;
    rdmult = (int64_t)((double)rdmult * weight / 128.0);
    rdmult = (rdmult + 8) >> 4;                                  // ROUND_POWER_OF_TWO(rdmult, 4)
    return rdmult > 0 ? (int)rdmult : 1;
}

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

    // ---- 3. tune=iq frame decisions per chroma layout
    long tiqBad = 0, tiqDev = 0, tiqDevRd = 0;
    av1f_tuneiq_frame *dt;
    cudaMalloc(&dt, 256 * sizeof(av1f_tuneiq_frame));
    for (int layout = 0; layout < 3; layout++) {   // 0: 4:2:0, 1: 4:4:4, 2: monochrome
        const int ss = layout == 1 ? 0 : 1, mono = layout == 2;
        k_tuneiq<<<1, 256>>>(ss, mono, dt);
        std::vector<av1f_tuneiq_frame> g(256);
        cudaMemcpy(g.data(), dt, 256 * sizeof(av1f_tuneiq_frame), cudaMemcpyDeviceToHost);
        for (int b = 1; b < 256; b++) {
            av1f_tuneiq_frame h;
            av1f_tuneiq_frame_init(b, ss, ss, mono, &h);
            const av1f_tuneiq_frame &d = g[b];
            // the decisions (integer rules) must match exactly; rdmult is double arithmetic that nvcc may contract into
            // FMA (as in the LUT's device math above), so its device value is informational - the encoder computes
            // the frame decisions on the host
            tiqDev += d.base_q_idx != h.base_q_idx || d.dq_u_dc != h.dq_u_dc || d.dq_u_ac != h.dq_u_ac || d.dq_v_dc != h.dq_v_dc ||
                      d.dq_v_ac != h.dq_v_ac || d.qm_y != h.qm_y || d.qm_u != h.qm_u || d.qm_v != h.qm_v ||
                      d.delta_q_res != h.delta_q_res || d.lf_level != h.lf_level || d.lf_sharpness != h.lf_sharpness ||
                      d.cdef_damping != h.cdef_damping || d.cdef_mode != h.cdef_mode;
            tiqDevRd += d.rdmult != h.rdmult;
            int udc, uac, qmy, qmu;
            ref_set_quantizer(b, layout == 1, &udc, &uac, &qmy, &qmu);
            if (mono) udc = uac = 0;   // chroma delta-q is not coded for monochrome
            const bool okq = h.dq_u_dc == udc && h.dq_v_dc == udc && h.dq_u_ac == uac && h.dq_v_ac == uac && h.qm_y == qmy &&
                             (mono || (h.qm_u == qmu && h.qm_v == qmu));
            if (!okq && tiqBad < 4)
                printf("  tune=iq layout %d base %d: dq %d/%d qm %d/%d vs libaom dq %d/%d qm %d/%d\n", layout, b, h.dq_u_dc, h.dq_u_ac,
                       h.qm_y, h.qm_u, udc, uac, qmy, qmu);
            tiqBad += !okq;
        }
    }
    cudaFree(dt);
    printf("tune=iq frame decisions (4:2:0, 4:4:4, monochrome x 255 bases): %ld differ from libaom av1_set_quantizer, %ld device/host "
           "differ (rdmult device-math: %ld differ, informational)\n", tiqBad, tiqDev, tiqDevRd);
    // ---- 4. 10 bits
    long lut10Bad = 0, lut10Ref = 0;
    cudaMalloc(&dl, 256 * 1024);
    for (int strength : {100, 50, 200}) {
        av1f_vb_build_luts_bd(dl, strength, 10, 0);
        std::vector<uint8_t> g(256 * 1024);
        cudaMemcpy(g.data(), dl, g.size(), cudaMemcpyDeviceToHost);
        for (int b = 1; b < 256; b++)
            for (int v = 0; v < 1024; v++) {
                const int h = av1f_varboost_qindex(b, (unsigned)v, strength, 10);
                lut10Bad += g[b * 1024 + v] != h;
                if (h != ref_varboost10(b, (unsigned)v, strength)) {
                    if (lut10Ref < 4) printf("  10-bit varboost: base %d var %d strength %d: %d vs libaom %d\n", b, v, strength, h,
                                             ref_varboost10(b, (unsigned)v, strength));
                    lut10Ref++;
                }
            }
    }
    cudaFree(dl);
    printf("10-bit LUT (3 strengths x 255 x 1024): uploaded %ld differ from the host, host %ld differ from libaom's 10-bit path\n",
           lut10Bad, lut10Ref);
    long bad10 = 0, nsb10 = 0, boost10 = 0, var10Ref = 0;
    for (int it = 0; it < 30; it++) {
        const int W = 1 + rnd() % 1500, H = 1 + rnd() % 1200;
        const int cols = (((W + 7) >> 3) * 8 + 63) >> 6, rows = (((H + 7) >> 3) * 8 + 63) >> 6;
        const int pitch = cols * 64 + 64 * (rnd() % 2);
        std::vector<uint16_t> y((size_t)pitch * rows * 64);
        const int kind = rnd() % 3;
        for (int r = 0; r < rows * 64; r++)
            for (int c = 0; c < pitch; c++) {
                int v;
                if (kind == 0) v = 512 + (int)(160 * __builtin_sin(r * 0.05) * __builtin_cos(c * 0.03)) + (int)(rnd() % 13) - 6;
                else if (kind == 1) v = (r / 64 + c / 64) % 3 == 0 ? 400 + rnd() % (1 + (r * 7 + c) % 240) : 360 + (rnd() % 9);
                else v = rnd() % 1024;
                y[(size_t)r * pitch + c] = (uint16_t)av1f_clampi(v, 0, 1023);
            }
        const int base = 1 + rnd() % 255, tcl2 = rnd() % 3, trl2 = rnd() % 3, strength = 100;
        const uint32_t n = cols * rows;
        std::vector<uint32_t> hv(n);
        std::vector<int> ht(n);
        for (int r = 0; r < rows; r++)
            for (int c = 0; c < cols; c++) {
                unsigned v[64];
                for (int k = 0; k < 64; k++) {
                    const uint16_t *p = &y[(size_t)(r * 64 + (k >> 3) * 8) * pitch + c * 64 + (k & 7) * 8];
                    int64_t s = 0;
                    uint64_t q = 0;
                    for (int i = 0; i < 8; i++)
                        for (int j = 0; j < 8; j++) { s += p[i * pitch + j]; q += (uint32_t)(p[i * pitch + j] * p[i * pitch + j]); }
                    v[k] = av1f_var8x8_from_sums_10(s, q);
                    var10Ref += v[k] != ref_hbd10_var8x8_div64(p, pitch);
                }
                hv[r * cols + c] = av1f_sb_variance(v);
                ht[r * cols + c] = av1f_varboost_qindex(base, hv[r * cols + c], strength, 10);
            }
        std::vector<uint32_t> gv(n);
        std::vector<uint8_t> gt(n), gq(n);
        if (av1f_varboost_host_bd(y.data(), pitch, W, H, base, tcl2, trl2, AV1F_VB_TILE, strength, 10, gv.data(), gt.data(), gq.data())) {
            printf("kernel error (10-bit)\n");
            return 1;
        }
        const int res = av1f_delta_q_res(base);
        const int tw = (cols + (1 << tcl2) - 1) >> tcl2, th = (rows + (1 << trl2) - 1) >> trl2;
        std::vector<int> hq(n);
        for (int tr = 0; tr * th < rows; tr++)
            for (int tc = 0; tc * tw < cols; tc++) {
                int prev = base;
                for (int r = tr * th; r < std::min(rows, tr * th + th); r++)
                    for (int c = tc * tw; c < std::min(cols, tc * tw + tw); c++) {
                        prev = av1f_adjust_q(res, prev, ht[r * cols + c]);
                        hq[r * cols + c] = prev;
                    }
            }
        for (uint32_t i = 0; i < n; i++) {
            const bool b = gv[i] != hv[i] || gt[i] != ht[i] || gq[i] != hq[i];
            if (b && bad10 < 5)
                printf("  10-bit mismatch it %d sb %u: var %u/%u target %d/%d q %d/%d\n", it, i, gv[i], hv[i], gt[i], ht[i], gq[i], hq[i]);
            bad10 += b;
            boost10 += hq[i] != base;
        }
        nsb10 += n;
    }
    printf("random 10-bit planes: %ld SB results (TILE order), %ld boosted, %ld differ; 8x8 variances vs libaom highbd_10: %ld differ\n",
           nsb10, boost10, bad10, var10Ref);
    long tiq10Bad = 0;
    for (int layout = 0; layout < 3; layout++) {
        const int ss = layout == 1 ? 0 : 1, mono = layout == 2;
        for (int b = 1; b < 256; b++) {
            av1f_tuneiq_frame h8, h10;
            av1f_tuneiq_frame_init(b, ss, ss, mono, &h8);
            av1f_tuneiq_frame_init(b, ss, ss, mono, &h10, 10);
            const int lv = ref_lf10(b);
            const int lfEff = lv == 0 ? 0 : ref_clamp(lv + (1 << (lv >> 5)), 0, 63);   // + INTRA ref delta, as at 8 bits
            const bool okb = h10.lf_level == lfEff && h10.rdmult == ref_rdmult10(b) && h10.dq_u_dc == h8.dq_u_dc &&
                             h10.dq_u_ac == h8.dq_u_ac && h10.dq_v_dc == h8.dq_v_dc && h10.dq_v_ac == h8.dq_v_ac && h10.qm_y == h8.qm_y &&
                             h10.qm_u == h8.qm_u && h10.qm_v == h8.qm_v && h10.delta_q_res == h8.delta_q_res &&
                             h10.lf_sharpness == h8.lf_sharpness && h10.cdef_damping == h8.cdef_damping && h10.cdef_mode == h8.cdef_mode;
            if (!okb && tiq10Bad < 4)
                printf("  10-bit tune=iq base %d: lf %d (libaom %d) rdmult %d (libaom %d)\n", b, h10.lf_level, lfEff, h10.rdmult, ref_rdmult10(b));
            tiq10Bad += !okb;
        }
    }
    printf("10-bit tune=iq frame decisions (3 layouts x 255 bases): %ld differ from libaom's high-bitdepth rules\n", tiq10Bad);
    const bool ok = !lutBad && !bad && !tiqBad && !tiqDev && !lut10Bad && !lut10Ref && !bad10 && !var10Ref && !tiq10Bad;
    printf("%s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
