// table init (host), partition DP, pack, launchers.
#include "kernels.h"
#include "dev_common.cuh"
#include "k_dp.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <type_traits>
#include <vector>

GP_DEFINE_TABLE_UPLOAD(gp_upload_tables_misc)
int gp_upload_tables_an_a(const GpHostTables *);
int gp_upload_tables_an_b(const GpHostTables *);
int gp_upload_tables_an_c(const GpHostTables *);
int gp_upload_tables_an_d(const GpHostTables *);
int gp_upload_tables_recon(const GpHostTables *);
int gp_upload_tables_recon_444_8(const GpHostTables *);
int gp_upload_tables_recon_420_10(const GpHostTables *);
int gp_upload_tables_recon_444_10(const GpHostTables *);
int gp_upload_tables_an_uv(const GpHostTables *);
int gp_upload_tables_ll(const GpHostTables *);
void gp_launch_an_uv(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st, int bd);
void gp_launch_an_a(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st, int bd);
void gp_launch_an_b(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st, int bd);
void gp_launch_an_c(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st, int bd);
void gp_launch_an_d(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st, int bd);

// ---------------------------------------------------------------------------
// pack: per-SB output slots -> DecisionRecord arrays (decode order, one CTA per SB)
__global__ void k_pack(GpBatch B, const GpPackOff *off, av1g_block *oB, uint32_t *oBT, av1g_txb *oT, int16_t *oC, GpPackAux aux) {
    const uint32_t sb = blockIdx.x;
    const GpSbOut so = B.sbOut[sb];
    const GpPackOff o = off[sb];
    for (uint32_t k = threadIdx.x; k < so.nBlocks; k += blockDim.x) {
        oB[o.gBlk + k] = B.oBlocks[(size_t)sb * GP_MAX_LEAVES + k];
        const uint32_t t0 = B.oBlockTxb[(size_t)sb * GP_MAX_LEAVES + k];
        oBT[o.gBlk + k] = o.rTxb + t0;
        if (aux.blockFrame) {
            const uint32_t t1 = k + 1 < so.nBlocks ? B.oBlockTxb[(size_t)sb * GP_MAX_LEAVES + k + 1] : so.nTxb;
            aux.blockFrame[o.gBlk + k] = o.img;
            aux.blockTile[o.gBlk + k] = o.tile;
            aux.blockTxbG[o.gBlk + k] = o.gTxb + t0;
            for (uint32_t t = t0; t < t1; t++) {
                aux.txbBlock[o.gTxb + t] = o.gBlk + k;
                aux.txbCoeff[o.gTxb + t] = o.gCoef + B.oTxb[(size_t)sb * B.sbTxbCap + t].coeff_off;
            }
        }
    }
    for (uint32_t k = threadIdx.x; k < so.nTxb; k += blockDim.x) {
        av1g_txb t = B.oTxb[(size_t)sb * B.sbTxbCap + k];
        t.coeff_off += o.rCoef;
        oT[o.gTxb + k] = t;
    }
    const int16_t *src = B.oCoeffs + (size_t)sb * B.sbCoefCap;
    for (uint32_t k = threadIdx.x; k < so.nCoeffs; k += blockDim.x) oC[o.gCoef + k] = src[k];
}

__global__ void k_set_cdef(av1g_block *blocks, uint32_t nB, const uint32_t *blockFrame, const uint32_t *frameSbBase,
                           const int *frameSbCols, const int *cdefOn, const int8_t *sbCdef) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nB) return;
    const uint32_t f = blockFrame[b];
    if (!cdefOn[f]) return;
    av1g_block &k = blocks[b];
    const int v = sbCdef[frameSbBase[f] + (uint32_t)(k.mi_row >> 4) * frameSbCols[f] + (k.mi_col >> 4)];
    k.cdef_idx = (int8_t)(v < 0 ? 0 : v);
}
void gp_launch_set_cdef(av1g_block *blocks, uint32_t nB, const uint32_t *blockFrame, const uint32_t *frameSbBase,
                        const int *frameSbCols, const int *cdefOn, const int8_t *sbCdef, cudaStream_t st) {
    if (nB) k_set_cdef<<<(nB + 255) / 256, 256, 0, st>>>(blocks, nB, blockFrame, frameSbBase, frameSbCols, cdefOn, sbCdef);
}

// tune=iq SSIM rdmult factors per 16x16 (libaom av1_set_mb_ssim_rdmult_scaling): log of the factor;
// k_ssim_norm subtracts the frame mean of the logs (= divide by the geometric mean).
// Per-pixel 8x8 variance as av1_get_perpixel_variance: 8 bits aom_variance8x8 (against the flat 128 reference:
// the same as against 0); 10 bits aom_highbd_10_variance8x8 against AV1_HIGH_VAR_OFFS_10 (flat 512) with its
// rounding to 8-bit scale: sse = ROUND_POWER_OF_TWO(sum d^2, 4), sum = ROUND_POWER_OF_TWO(sum d, 2), var >= 0.
template <int BD>
__global__ void k_ssim_cells(const av1g_pixel_t(BD) *y, int pitch, int MiCols, int MiRows, int cols16, int rows16, float *out,
                             double *sum) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= cols16 * rows16) return;
    const int row = cell / cols16, col = cell % cols16;
    double var = 0.0, nv = 0.0;
    for (int mr = row * 4; mr < MiRows && mr < (row + 1) * 4; mr += 2)
        for (int mc = col * 4; mc < MiCols && mc < (col + 1) * 4; mc += 2) {
            unsigned v;
            if (BD == 8) {
                int s = 0;
                unsigned sse = 0;
                for (int i = 0; i < 8; i++)
                    for (int j = 0; j < 8; j++) {
                        const int p = y[(size_t)(mr * 4 + i) * pitch + mc * 4 + j];
                        s += p;
                        sse += (unsigned)(p * p);
                    }
                v = sse - (unsigned)(((int64_t)s * s) / 64);
            } else {
                int64_t s = 0;
                uint64_t q = 0;
                for (int i = 0; i < 8; i++)
                    for (int j = 0; j < 8; j++) {
                        const int d = (int)y[(size_t)(mr * 4 + i) * pitch + mc * 4 + j] - (128 << (BD - 8));
                        s += d;
                        q += (uint32_t)(d * d);
                    }
                const uint32_t sse = (uint32_t)((q + 8) >> 4);
                const int sm = (int)((s + 2) >> 2);
                const int64_t vv = (int64_t)sse - ((int64_t)sm * sm) / 64;
                v = vv >= 0 ? (unsigned)vv : 0u;
            }
            var += (double)((v + 32) >> 6);
            nv += 1.0;
        }
    var = var / nv;
    const double f = 67.035434 * (1.0 - exp(-0.0021489 * var)) + 17.492222;
    const double l = log(f);
    out[cell] = (float)l;
    atomicAdd(sum, l);
}
__global__ void k_ssim_norm(float *out, int n, const double *sum) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] -= (float)(*sum / n);
}
// per 16x16 cell: max over its 8x8 blocks of the per-pixel source variance (flat pruning; native units)
template <typename Px>
__global__ void k_var16(const Px *y, int pitch, int MiCols, int MiRows, int cols16, int rows16, float *out) {
    const int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell >= cols16 * rows16) return;
    const int row = cell / cols16, col = cell % cols16;
    float mx = 0.f;
    for (int by = 0; by < 2; by++)
        for (int bx = 0; bx < 2; bx++) {
            const int py = row * 16 + by * 8, px = col * 16 + bx * 8;
            if (py >= MiRows * 4 || px >= MiCols * 4) continue;
            int s = 0, s2 = 0;   // 10 bits: 64 * 1023^2 < 2^31
            for (int i = 0; i < 8; i++)
                for (int j = 0; j < 8; j++) {
                    const int p = y[(size_t)(py + i) * pitch + px + j];
                    s += p;
                    s2 += p * p;
                }
            mx = fmaxf(mx, (float)s2 / 64.f - ((float)s / 64.f) * ((float)s / 64.f));
        }
    out[cell] = mx;
}
void gp_launch_var16(const uint8_t *y, int pitch, int MiCols, int MiRows, float *out, cudaStream_t st, int bd) {
    const int cols16 = (MiCols + 3) / 4, rows16 = (MiRows + 3) / 4, n = cols16 * rows16;
    if (bd == 10) k_var16<<<(n + 127) / 128, 128, 0, st>>>((const uint16_t *)y, pitch, MiCols, MiRows, cols16, rows16, out);
    else k_var16<<<(n + 127) / 128, 128, 0, st>>>(y, pitch, MiCols, MiRows, cols16, rows16, out);
}
void gp_launch_ssim_map(const uint8_t *y, int pitch, int MiCols, int MiRows, float *out, double *sum, cudaStream_t st, int bd) {
    const int cols16 = (MiCols + 3) / 4, rows16 = (MiRows + 3) / 4, n = cols16 * rows16;
    cudaMemsetAsync(sum, 0, sizeof(double), st);
    if (bd == 10)
        k_ssim_cells<10><<<(n + 127) / 128, 128, 0, st>>>((const uint16_t *)y, pitch, MiCols, MiRows, cols16, rows16, out, sum);
    else k_ssim_cells<8><<<(n + 127) / 128, 128, 0, st>>>(y, pitch, MiCols, MiRows, cols16, rows16, out, sum);
    k_ssim_norm<<<(n + 255) / 256, 256, 0, st>>>(out, n, sum);
}

// 8-bit YCbCr planes -> 10-bit (front ends with --depth 10, sources with 8 bits per sample): the full-range
// unorm mapping libavif applies when it writes 10-bit YUV from 8-bit RGB (avifImageRGBToYUV: v / 255 normalised,
// Y = round(y * 1023), Cb/Cr = round(c * 1023 + 512)), here on the 8-bit Y / Cb / Cr samples:
// Y10 = round(Y8 * 1023 / 255), C10 = clamp(round((C8 - 128) * 1023 / 255) + 512).  No ties exist (3 v mod 255 is
// never 127.5), so the integer rounding below is exact.
__global__ void k_to10(const uint8_t *src, int sp, uint16_t *dst, int dp, int w, int h, int chroma) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
    if (x >= w || y >= h) return;
    const int v = src[(size_t)y * sp + x];
    int o;
    if (!chroma) {
        o = (2 * 1023 * v + 255) / 510;
    } else {
        const int t = (v - 128) * 1023;
        o = (t >= 0 ? (2 * t + 255) / 510 : -((-2 * t + 255) / 510)) + 512;
        o = o < 0 ? 0 : o > 1023 ? 1023 : o;
    }
    dst[(size_t)y * dp + x] = (uint16_t)o;
}
void gp_launch_to10(const uint8_t *src, int spitch, uint16_t *dst, int dpitch, int w, int h, int chroma, cudaStream_t st) {
    if (w <= 0 || h <= 0) return;
    k_to10<<<dim3((w + 255) / 256, h), 256, 0, st>>>(src, spitch, dst, dpitch, w, h, chroma);
}

void gp_launch_analysis(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st, int bd) {
    gp_launch_an_d(B, P, shTotal, st, bd);
    gp_launch_an_c(B, P, shTotal, st, bd);
    gp_launch_an_b(B, P, shTotal, st, bd);
    gp_launch_an_a(B, P, shTotal, st, bd);
    if (P.uvInDp) gp_launch_an_uv(B, P, shTotal, st, bd);
}

void gp_launch_dp(const GpBatch &B, const GpParams &P, cudaStream_t st, int bd) {
    if (!B.nSb) return;
    if (bd == 10) k_dp<10><<<(B.nSb + 63) / 64, 64, 0, st>>>(B, P);
    else k_dp<8><<<(B.nSb + 63) / 64, 64, 0, st>>>(B, P);
}

void gp_launch_pack(const GpBatch &B, const GpPackOff *dOff, av1g_block *oB, uint32_t *oBT, av1g_txb *oT, int16_t *oC,
                    const GpPackAux &aux, cudaStream_t st) {
    if (!B.nSb) return;
    k_pack<<<B.nSb, 256, 0, st>>>(B, dOff, oB, oBT, oT, oC, aux);
}

// ---------------------------------------------------------------------------
// table init (host)
namespace {

std::vector<double> invert(std::vector<double> a, int n) {
    std::vector<double> inv((size_t)n * n, 0.0);
    for (int i = 0; i < n; i++) inv[(size_t)i * n + i] = 1.0;
    for (int c = 0; c < n; c++) {
        int p = c;
        for (int r = c + 1; r < n; r++)
            if (fabs(a[(size_t)r * n + c]) > fabs(a[(size_t)p * n + c])) p = r;
        if (p != c)
            for (int k = 0; k < n; k++) {
                std::swap(a[(size_t)c * n + k], a[(size_t)p * n + k]);
                std::swap(inv[(size_t)c * n + k], inv[(size_t)p * n + k]);
            }
        const double d = a[(size_t)c * n + c];
        for (int k = 0; k < n; k++) { a[(size_t)c * n + k] /= d; inv[(size_t)c * n + k] /= d; }
        for (int r = 0; r < n; r++)
            if (r != c) {
                const double f = a[(size_t)r * n + c];
                if (f != 0.0)
                    for (int k = 0; k < n; k++) {
                        a[(size_t)r * n + k] -= f * a[(size_t)c * n + k];
                        inv[(size_t)r * n + k] -= f * inv[(size_t)c * n + k];
                    }
            }
    }
    return inv;
}

template <typename Arr>
void cost_field(const av1g_cdfs &c, const Arr &field, float *out) {
    typedef typename std::remove_reference<Arr>::type T;
    constexpr size_t L = std::extent<T, std::rank<T>::value - 1>::value;
    const uint16_t *p = (const uint16_t *)&field;
    const size_t off = (size_t)(p - (const uint16_t *)&c);
    const size_t total = sizeof(T) / 2;
    for (size_t b = 0; b < total; b += L)
        for (size_t s = 0; s + 1 < L; s++) {
            const double lo = s ? p[b + s - 1] : 0.0, hi = p[b + s];
            const double pr = std::max((hi - lo) / 32768.0, 1.0 / 32768.0);
            out[off + b + s] = (float)-std::log2(pr);
        }
}

}  // namespace

int gp_init_tables(char *err, int errlen) {
    static GpHostTables H;
    // forward matrices
    std::vector<float> fwd(GP_FWD_FLOATS, 0.f);
    int fwdOff[3][7];
    double norm2[3][7];
    memset(fwdOff, 0, sizeof fwdOff);
    memset(norm2, 0, sizeof norm2);
    size_t at = 0;
    for (int k = 0; k < 3; k++)
        for (int ln = 2; ln <= 6; ln++) {
            if (k == AV1G_K_ADST && ln > 4) continue;
            if (k == AV1G_K_IDT && ln > 5) continue;
            const int n = 1 << ln;
            std::vector<double> M((size_t)n * n);
            const int A = 1 << 14;
            double nn = 0;
            for (int j = 0; j < n; j++) {
                int32_t T[64] = {0};
                T[j] = A;
                uint32_t e = 0;
                av1g_inv_1d(T, k, ln, 31, &e);
                for (int i = 0; i < n; i++) {
                    M[(size_t)i * n + j] = (double)T[i] / A;
                    nn += M[(size_t)i * n + j] * M[(size_t)i * n + j];
                }
            }
            norm2[k][ln] = nn / n;
            std::vector<double> F = invert(M, n);
            fwdOff[k][ln] = (int)at;
            for (size_t i = 0; i < (size_t)n * n; i++) fwd[at + i] = (float)F[i];
            at += (size_t)n * n;
        }
    if (at != GP_FWD_FLOATS) { snprintf(err, errlen, "fwd table size %zu != %d", at, (int)GP_FWD_FLOATS); return -1; }
    float txg[19][16], txScale[19];
    for (int t = 0; t < 19; t++) {
        const int lw = AV1G_T(Tx_Width_Log2)[t], lh = AV1G_T(Tx_Height_Log2)[t];
        const double rect = (lw - lh == 1 || lh - lw == 1) ? 2896.0 / 4096.0 : 1.0;
        const double s = (double)(1 << (AV1G_T(Transform_Row_Shift)[t] + 4));
        txScale[t] = (float)(s / rect);
        for (int ty = 0; ty < 16; ty++) {
            const int rk = av1g_row_kind(ty), ck = av1g_col_kind(ty);
            double nr = norm2[rk][lw], nc = norm2[ck][lh];
            if (nr == 0 || nc == 0) { txg[t][ty] = 0; continue; }  // illegal combination (e.g. ADST64)
            txg[t][ty] = (float)(rect * rect * nr * nc / (s * s));
        }
    }
    // symbol costs per coefficient q-context
    std::vector<float> costs((size_t)4 * GP_NCDF, 0.f);
    const int qrep[4] = {20, 60, 120, 255};
    float avgY[13], avgPart[5][10], avgSkip0 = 0;
    for (int qc = 0; qc < 4; qc++) {
        av1g_cdfs c;
        av1g_cdfs_init(&c, qrep[qc]);
        float *out = costs.data() + (size_t)qc * GP_NCDF;
#define CF(f) cost_field(c, c.f, out)
        CF(y_mode); CF(uv_mode_cfl_not_allowed); CF(uv_mode_cfl_allowed); CF(angle_delta);
        CF(partition_w8); CF(partition_w16); CF(partition_w32); CF(partition_w64);
        CF(tx_8x8); CF(tx_16x16); CF(tx_32x32); CF(tx_64x64); CF(skip); CF(filter_intra_mode); CF(filter_intra);
        CF(delta_q); CF(delta_lf); CF(delta_lf_multi); CF(intra_tx_set1); CF(intra_tx_set2); CF(cfl_sign); CF(cfl_alpha);
        CF(palette_y_mode); CF(palette_uv_mode); CF(txb_skip); CF(eob_pt_16); CF(eob_pt_32); CF(eob_pt_64); CF(eob_pt_128);
        CF(eob_pt_256); CF(eob_pt_512); CF(eob_pt_1024); CF(eob_extra); CF(dc_sign); CF(coeff_base_eob); CF(coeff_base);
        CF(coeff_br);
#undef CF
        if (qc == 0) {
            for (int m = 0; m < 13; m++) {
                double p = 0;
                for (int a = 0; a < 5; a++)
                    for (int l = 0; l < 5; l++) {
                        const uint16_t *cd = c.y_mode[a][l];
                        p += (cd[m] - (m ? cd[m - 1] : 0)) / 32768.0;
                    }
                avgY[m] = (float)-std::log2(std::max(p / 25.0, 1e-6));
            }
            for (int bsl = 0; bsl < 5; bsl++)
                for (int p = 0; p < 10; p++) avgPart[bsl][p] = 20.f;
            for (int bsl = 1; bsl <= 4; bsl++) {
                const int N = bsl == 1 ? 4 : 10;
                for (int p = 0; p < N; p++) {
                    double pr = 0;
                    for (int ctx = 0; ctx < 4; ctx++) {
                        const uint16_t *cd = bsl == 1 ? c.partition_w8[ctx] : bsl == 2 ? c.partition_w16[ctx]
                                           : bsl == 3 ? c.partition_w32[ctx] : c.partition_w64[ctx];
                        pr += (cd[p] - (p ? cd[p - 1] : 0)) / 32768.0;
                    }
                    avgPart[bsl][p] = (float)-std::log2(std::max(pr / 4.0, 1e-6));
                }
            }
            double ps = 0;
            for (int ctx = 0; ctx < 3; ctx++) ps += c.skip[ctx][0] / 32768.0;
            avgSkip0 = (float)-std::log2(ps / 3.0);
        }
    }
    memcpy(H.fwd, fwd.data(), sizeof H.fwd);
    for (int k = 0; k < 3; k++)
        for (int ln = 2; ln <= 6; ln++) {
            if (k == AV1G_K_ADST && ln > 4) continue;
            if (k == AV1G_K_IDT && ln > 5) continue;
            const int nn = 1 << ln, o = fwdOff[k][ln];
            for (int i = 0; i < nn; i++)
                for (int j = 0; j < nn; j++) H.fwdT[o + j * nn + i] = fwd[o + i * nn + j];
        }
    memcpy(H.fwdOff, fwdOff, sizeof H.fwdOff);
    memcpy(H.txg, txg, sizeof H.txg);
    memcpy(H.txScale, txScale, sizeof H.txScale);
    memcpy(H.cost, costs.data(), sizeof H.cost);
    memcpy(H.avgYMode, avgY, sizeof H.avgYMode);
    memcpy(H.avgPart, avgPart, sizeof H.avgPart);
    H.avgSkip0 = avgSkip0;
    int (*ups[])(const GpHostTables *) = {gp_upload_tables_misc, gp_upload_tables_an_a, gp_upload_tables_an_b,
                                          gp_upload_tables_an_c, gp_upload_tables_an_d, gp_upload_tables_recon,
                                          gp_upload_tables_recon_444_8, gp_upload_tables_recon_420_10,
                                          gp_upload_tables_recon_444_10, gp_upload_tables_an_uv, gp_upload_tables_ll};
    for (auto f : ups)
        if (f(&H)) { snprintf(err, errlen, "table upload failed"); return -1; }
    return 0;
}
