/* av1f/qm_quant.h - quantizer-matrix (QM) weighted quantization for the GPU engines (tracks
 * B/C/E), header-only, host + device.  Include after (or instead of) av1g/dsp.h.
 *
 * Conventions (= av1core's DecisionRecord): coefficients are the spec's Quant[] / Dequant[]
 * layout, row-major tw x th with tw = min(32, w), th = min(32, h); positions index the weight
 * tables the same way.  Coefficient scale = libaom's "QTX" domain: dequantized value =
 * level * step >> log_scale, step = Round2(q * iwt, 5), log_scale = log2(dqDenom).
 *
 * Weight tables (all 19 tx sizes, QM levels 0..14 + flat 15, luma / chroma), generated from libaom 3.13.1:
 *   av1f_qm_iwt()  decoder weights (bitstream contract; == spec Quantizer_Matrix == av1g_qm())
 *   av1f_qm_wt()   libaom's encoder-side weights (~1024 / iwt, rounded) used by its quantizer
 * Both return NULL (flat, weight 32) for level 15 or tx types >= IDTX (1-D / identity).
 *
 * Steps:      av1f_coeff_step(q, iwt, pos)          per-coefficient decoder step
 *             av1f_step_table(...)                    a whole tx block's steps
 * Quantizers: av1f_qparam_init(qindex, dc_delta, ac_delta, sharpness)  per (SB qindex, plane)
 *             av1f_quantize_fp_coef()   one coefficient, bit-exact libaom quantize_fp_helper_c /
 *                                       av1_quantize_fp_no_qmatrix (the quantizer libaom's
 *                                       all-intra encoder uses before trellis)
 *             av1f_quantize_txb_fp()    a tx block in natural order -> levels (+ dequantized), eob
 *             av1f_get_iscan()          inverse scans (eob for thread-per-coefficient quantizers)
 * Tested bit-exact against libaom 3.13.1's own av1_quantize_fp_facade (tests/qm_quant_test.cu).
 * With tune=iq libaom builds its tables with sharpness 7 -> rounding factor 64/128 for FP.
 */
#ifndef AV1F_QM_QUANT_H
#define AV1F_QM_QUANT_H

#include <stdint.h>
#include "av1g/hd.h"
#include "av1g/dsp.h"
#include "av1f/aom_tables_gen.h"

#define AV1F_QM_BITS 5
#define AV1F_QM_FLAT 15

/* libaom av1_get_tx_scale(): log2(dqDenom) */
AV1G_HD int av1f_log_scale(int txSz) {
    const int pels = av1g_tx_w(txSz) * av1g_tx_h(txSz);
    return (pels > 256) + (pels > 1024);
}
AV1G_HD int av1f_qm_offset(int txSz) { return AV1G_T(Qm_Offset)[txSz]; }

AV1G_HD const uint8_t *av1f_qm_iwt(int level, int plane, int txSz, int txType) {
    if (level >= AV1F_QM_FLAT || txType >= AV1G_IDTX) return 0;
    return &AV1G_T(Aom_Qm_Iwt)[level][plane > 0][av1f_qm_offset(txSz)];
}
AV1G_HD const uint8_t *av1f_qm_wt(int level, int plane, int txSz, int txType) {
    if (level >= AV1F_QM_FLAT || txType >= AV1G_IDTX) return 0;
    return &AV1G_T(Aom_Qm_Wt)[level][plane > 0][av1f_qm_offset(txSz)];
}

/* decoder step of coefficient pos (before dqDenom): Round2(q * iwt[pos], 5); q = dc for pos 0 */
AV1G_HD int av1f_coeff_step(int q, const uint8_t *iwt, int pos) {
    return iwt ? (q * iwt[pos] + (1 << (AV1F_QM_BITS - 1))) >> AV1F_QM_BITS : q;
}
AV1G_HD void av1f_step_table(int qdc, int qac, int txSz, const uint8_t *iwt, int32_t *step) {
    const int tw = av1g_tx_w(txSz) < 32 ? av1g_tx_w(txSz) : 32, th = av1g_tx_h(txSz) < 32 ? av1g_tx_h(txSz) : 32;
    for (int i = 0; i < tw * th; i++) step[i] = av1f_coeff_step(i ? qac : qdc, iwt, i);
}

/* Inverse of av1g_get_scan(): natural position -> scan index (same table choice as the spec's
 * get_scan()).  For thread-per-coefficient quantizers: eob = 1 + max iscan[pos] over nonzero levels. */
AV1G_HD const uint16_t *av1f_get_iscan(int txSz, int txType) {
    if (txSz == AV1G_TX_16X64) return AV1G_T(IDefault_Scan_16x32);
    if (txSz == AV1G_TX_64X16) return AV1G_T(IDefault_Scan_32x16);
    if (AV1G_T(Tx_Size_Sqr_Up)[txSz] == AV1G_TX_64X64) return AV1G_T(IDefault_Scan_32x32);
    const int cls = (txType == AV1G_IDTX) ? AV1G_TX_CLASS_2D : av1g_tx_class(txType);
#define AV1F_SC_(P)                                                   \
    switch (txSz) {                                                   \
    case AV1G_TX_4X4: return AV1G_T(P##_Scan_4x4);                    \
    case AV1G_TX_4X8: return AV1G_T(P##_Scan_4x8);                    \
    case AV1G_TX_8X4: return AV1G_T(P##_Scan_8x4);                    \
    case AV1G_TX_8X8: return AV1G_T(P##_Scan_8x8);                    \
    case AV1G_TX_8X16: return AV1G_T(P##_Scan_8x16);                  \
    case AV1G_TX_16X8: return AV1G_T(P##_Scan_16x8);                  \
    case AV1G_TX_16X16: return AV1G_T(P##_Scan_16x16);                \
    case AV1G_TX_4X16: return AV1G_T(P##_Scan_4x16);                  \
    default: return AV1G_T(P##_Scan_16x4);                            \
    }
    if (cls == AV1G_TX_CLASS_VERT) { AV1F_SC_(IMrow) }
    if (cls == AV1G_TX_CLASS_HORIZ) { AV1F_SC_(IMcol) }
#undef AV1F_SC_
    switch (txSz) {
    case AV1G_TX_4X4: return AV1G_T(IDefault_Scan_4x4);
    case AV1G_TX_4X8: return AV1G_T(IDefault_Scan_4x8);
    case AV1G_TX_8X4: return AV1G_T(IDefault_Scan_8x4);
    case AV1G_TX_8X8: return AV1G_T(IDefault_Scan_8x8);
    case AV1G_TX_8X16: return AV1G_T(IDefault_Scan_8x16);
    case AV1G_TX_16X8: return AV1G_T(IDefault_Scan_16x8);
    case AV1G_TX_16X16: return AV1G_T(IDefault_Scan_16x16);
    case AV1G_TX_16X32: return AV1G_T(IDefault_Scan_16x32);
    case AV1G_TX_32X16: return AV1G_T(IDefault_Scan_32x16);
    case AV1G_TX_4X16: return AV1G_T(IDefault_Scan_4x16);
    case AV1G_TX_16X4: return AV1G_T(IDefault_Scan_16x4);
    case AV1G_TX_8X32: return AV1G_T(IDefault_Scan_8x32);
    case AV1G_TX_32X8: return AV1G_T(IDefault_Scan_32x8);
    default: return AV1G_T(IDefault_Scan_32x32);
    }
}

/* libaom per-qindex quantizer parameters for one plane (av1_build_quantizer) */
typedef struct av1f_qparam {
    int16_t dequant[2];   /* dc, ac quantizer (QTX), incl. the plane's delta-q */
    int16_t quant_fp[2];  /* (1 << 16) / dequant */
    int16_t round_fp[2];  /* (qrounding_factor_fp * dequant) >> 7 */
} av1f_qparam;

AV1G_HD void av1f_qparam_init(int qindex, int dc_delta, int ac_delta, int sharpness, av1f_qparam *p) {
    const int adj = 16 * (7 - sharpness) / 7;
    const int rf = (sharpness != 0 && qindex != 0) ? 64 - adj : 64;
    for (int i = 0; i < 2; i++) {
        const int d = i == 0 ? AV1G_T(Aom_Dc_Qlookup)[av1g_clip3(0, 255, qindex + dc_delta)]
                             : AV1G_T(Aom_Ac_Qlookup)[av1g_clip3(0, 255, qindex + ac_delta)];
        p->dequant[i] = (int16_t)d;
        p->quant_fp[i] = (int16_t)((1 << 16) / d);
        p->round_fp[i] = (int16_t)((rf * d) >> 7);
    }
}

/* One coefficient through libaom's FP quantizer (quantize_fp_helper_c; with wt = iwt = NULL it is
 * av1_quantize_fp_no_qmatrix).  Returns the signed level; *dq = libaom's dequantized value. */
AV1G_HD int32_t av1f_quantize_fp_coef(int32_t coeff, int pos, const av1f_qparam *p, const uint8_t *wt, const uint8_t *iwt,
                                      int log_scale, int32_t *dq) {
    const int ac = pos != 0;
    const int w = wt ? wt[pos] : (1 << AV1F_QM_BITS);
    const int iw = iwt ? iwt[pos] : (1 << AV1F_QM_BITS);
    const int dequant = (p->dequant[ac] * iw + (1 << (AV1F_QM_BITS - 1))) >> AV1F_QM_BITS;
    const int neg = coeff < 0;
    int64_t a = neg ? -(int64_t)coeff : coeff;
    int32_t lvl = 0, d = 0;
    if (a * w >= ((int64_t)p->dequant[ac] << (AV1F_QM_BITS - (1 + log_scale)))) {
        a += (p->round_fp[ac] + ((1 << log_scale) >> 1)) >> log_scale;     /* ROUND_POWER_OF_TWO */
        a = a > 32767 ? 32767 : a;
        lvl = (int32_t)((a * w * p->quant_fp[ac]) >> (16 - log_scale + AV1F_QM_BITS));
        d = (lvl * dequant) >> log_scale;
    }
    if (dq) *dq = neg ? -d : d;
    return neg ? -lvl : lvl;
}

/* A whole tx block (natural row-major order, tw x th).  qm_level 15 = flat.  Writes levels
 * (int16, the DecisionRecord's coefficient format, clamped to +-32767) and optionally libaom's dequantized values;
 * returns eob = 1 + last nonzero position in the tx type's scan (0 if all zero). */
AV1G_HD int av1f_quantize_txb_fp(const int32_t *coeff, int16_t *level, int32_t *dqcoeff, int txSz, int txType, int plane,
                                 int qm_level, const av1f_qparam *p) {
    const int tw = av1g_tx_w(txSz) < 32 ? av1g_tx_w(txSz) : 32, th = av1g_tx_h(txSz) < 32 ? av1g_tx_h(txSz) : 32;
    const uint8_t *wt = av1f_qm_wt(qm_level, plane, txSz, txType);
    const uint8_t *iwt = av1f_qm_iwt(qm_level, plane, txSz, txType);
    const int ls = av1f_log_scale(txSz);
    const uint16_t *scan = av1g_get_scan(txSz, txType);
    int eob = 0;
    for (int i = 0; i < tw * th; i++) {
        const int pos = scan[i];
        int32_t d;
        const int32_t l = av1f_quantize_fp_coef(coeff[pos], pos, p, wt, iwt, ls, &d);
        level[pos] = (int16_t)(l > 32767 ? 32767 : l < -32767 ? -32767 : l);  /* record limit; libaom's int32 level only differs
                                                                                    when dequantization saturates anyway */
        if (dqcoeff) dqcoeff[pos] = d;
        if (l) eob = i + 1;
    }
    return eob;
}

#endif
