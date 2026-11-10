// rc_eval_drift_test: k_recon's evaluation of a tx block, rc_eval<BD, G> (encoder/src/rc_eval.cuh: the whole warp,
// G = 32, or one half-warp, G = 16), against dev_tx.cuh's warp_eval_rt, the evaluation it replaced (the analysis
// kernels still use it): the levels, eob and the bits of D and R must be equal.
//   - all 19 tx sizes x every tx type the size can have (64-point: DCT_DCT; 32-point: DCT_DCT, IDTX; the rest: all 16)
//   - random residuals: full range, small, sparse, smooth, extreme, zero
//   - QM off and on (random level and plane, qm-dist / qm-rnd 0 / 1), random q index, rounding offset, contexts
//   - BitDepth 8 and 10
//   - G = 16 for the sizes k_recon runs on half-warps (up to 8x8): two different cases of one size on the two halves
//     of a warp, as k_recon pairs its tasks
// The encoder's own tables (gp_init_tables).  Exit status 1 on any difference.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "kernels.h"
#include "rc_eval.cuh"

const GpHostTables *gp_host_tables();   // k_misc.cu: the tables gp_init_tables built
GP_DEFINE_TABLE_UPLOAD(drift_upload_tables)

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static uint64_t rs = 0x9e3779b97f4a7c15ull;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)rs; }
static int rint_in(int lo, int hi) { return lo + (int)(rnd() % (uint32_t)(hi - lo + 1)); }

struct Case {
    int txSz, txType, plane, qindex, qmLevel, qmDist, qmRnd, skipCtx, dcCtx, intraDir, reduced, costCtx, resOff;
    float rnd, zbin, resEnergy;
};
struct Res { float D, R; int eob; };
#define TMP_FLOATS (32 * 65)   // k_recon's RcBigScr.tmp: th x (W + 1) for every size
#define Q_MAX 1024

template <int BD>
__device__ EvalIn case_in(const Case &c) {
    EvalIn e;
    e.txSz = c.txSz; e.txType = c.txType; e.plane = c.plane;
    e.dcQ = av1g_dc_q<BD>(c.qindex); e.acQ = av1g_ac_q<BD>(c.qindex);
    e.qm = av1g_qm(c.qmLevel >= 0, c.qmLevel < 0 ? 0 : c.qmLevel, c.plane, c.txSz, c.txType);
    e.rnd = c.rnd; e.zbin = c.zbin;
    e.skipCtx = c.skipCtx; e.dcCtx = c.dcCtx; e.intraDir = c.intraDir; e.reduced = c.reduced;
    e.C = g_cost[c.costCtx];
    e.resEnergy = c.resEnergy;
    e.qmDist = c.qmDist; e.qmRnd = c.qmRnd;
    return e;
}

// one warp per case: MODE 0 warp_eval_rt, 1 rc_eval<BD, 32>
template <int BD, int MODE>
__global__ void k_eval(const Case *cs, const int16_t *res, float *tmp, int16_t *q, Res *out, int n) {
    const int w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    if (w >= n) return;
    const Case c = cs[w];
    const int W = AV1G_T(Tx_Width)[c.txSz], H = AV1G_T(Tx_Height)[c.txSz];
    const EvalIn e = case_in<BD>(c);
    EvalOut o;
    if (MODE == 0) warp_eval_rt<BD>(res + c.resOff, tmp + (size_t)w * TMP_FLOATS, q + (size_t)w * Q_MAX, e, o, W, H);
    else o = rc_eval<BD, 32>(res + c.resOff, tmp + (size_t)w * TMP_FLOATS, q + (size_t)w * Q_MAX, e, W, H, rc_warp_lanes());
    if ((threadIdx.x & 31) == 0) out[w] = Res{o.D, o.R, o.eob};
}

// rc_eval<BD, 16>: cases ids[2 w] and ids[2 w + 1] (same size) on the two half-warps of warp w
template <int BD>
__global__ void k_eval_half(const Case *cs, const int *ids, int m, const int16_t *res, float *tmp, int16_t *q, Res *out) {
    const int w = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int k = 2 * w + ((threadIdx.x >> 4) & 1);
    if (k >= m) return;   // (a half-warp at a time: rc_eval<BD, 16> syncs its own half only)
    const int i = ids[k];
    const Case c = cs[i];
    const int W = AV1G_T(Tx_Width)[c.txSz], H = AV1G_T(Tx_Height)[c.txSz];
    const RcLanes<16> L = rc_half_lanes();
    const EvalOut o = rc_eval<BD, 16>(res + c.resOff, tmp + (size_t)i * TMP_FLOATS, q + (size_t)i * Q_MAX, case_in<BD>(c), W, H, L);
    if (L.l == 0) out[i] = Res{o.D, o.R, o.eob};
}

// the tx types a size can have (the intra and inter sets together)
static bool type_ok(int txSz, int t) {
    const int sqrUp = AV1G_T(Tx_Size_Sqr_Up)[txSz];
    if (sqrUp == AV1G_TX_64X64) return t == AV1G_DCT_DCT;
    if (sqrUp == AV1G_TX_32X32) return t == AV1G_DCT_DCT || t == AV1G_IDTX;
    return true;
}

static void make_residual(int16_t *r, int W, int H, int BD, int style) {
    const int mx = (1 << BD) - 1;
    const int a = rint_in(-8, 8), b = rint_in(-8, 8), c0 = rint_in(-mx / 2, mx / 2);
    for (int i = 0; i < H; i++)
        for (int j = 0; j < W; j++) {
            int v;
            switch (style) {
            case 0: v = rint_in(-mx, mx); break;                                   // full range
            case 1: v = rint_in(-3, 3); break;                                      // small
            case 2: v = rnd() % 20 == 0 ? rint_in(-mx, mx) : 0; break;              // sparse
            case 3: v = c0 + a * i + b * j + rint_in(-2, 2); break;                 // smooth
            case 4: v = (rnd() & 1) ? mx : -mx; break;                              // extreme
            default: v = 0; break;                                                  // zero
            }
            r[i * W + j] = (int16_t)(v < -mx ? -mx : v > mx ? mx : v);
        }
}

template <int BD>
static int run(int reps) {
    std::vector<Case> cs;
    std::vector<int16_t> res;
    for (int txSz = 0; txSz < 19; txSz++) {   // size-major: the G = 16 pairs below have one size
        const int W = AV1G_T(Tx_Width)[txSz], H = AV1G_T(Tx_Height)[txSz];
        for (int t = 0; t < 16; t++) {
            if (!type_ok(txSz, t)) continue;
            for (int qm = 0; qm < 2; qm++)
                for (int r = 0; r < reps; r++) {
                    Case c;
                    c.txSz = txSz; c.txType = t;
                    c.plane = rint_in(0, 2);
                    const int style = r % 7 == 6 ? 5 : r % 6;
                    c.qindex = style == 1 ? rint_in(1, 40) : rint_in(1, 255);
                    c.qmLevel = qm ? rint_in(0, 15) : -1;
                    c.qmDist = (int)(rnd() & 1); c.qmRnd = (int)(rnd() & 1);
                    c.skipCtx = rint_in(0, 12); c.dcCtx = rint_in(0, 2); c.intraDir = rint_in(0, 12);
                    c.reduced = (int)(rnd() & 1); c.costCtx = rint_in(0, 3);
                    const float rnds[4] = {0.375f, 0.5f, 0.25f, 0.2f + 0.3f * (float)(rnd() % 1000) / 1000.f};
                    const float zbins[4] = {0.656f, 0.5f, 0.75f, 0.4f + 0.4f * (float)(rnd() % 1000) / 1000.f};
                    c.rnd = rnds[rnd() % 4]; c.zbin = zbins[rnd() % 4];
                    c.resOff = (int)res.size();
                    res.resize(res.size() + (size_t)W * H);
                    make_residual(&res[c.resOff], W, H, BD, style);
                    float en = 0.f;
                    for (int p = 0; p < W * H; p++) en += (float)((int)res[c.resOff + p] * (int)res[c.resOff + p]);
                    c.resEnergy = en;
                    cs.push_back(c);
                }
        }
    }
    const int n = (int)cs.size();
    std::vector<int> ids;   // the G = 16 cases (up to 8x8), an even number per size
    for (int i = 0; i < n; i++)
        if (AV1G_T(Tx_Width)[cs[i].txSz] <= 8 && AV1G_T(Tx_Height)[cs[i].txSz] <= 8) ids.push_back(i);
    const int m = (int)ids.size();

    Case *dcs; int16_t *dres, *dq; float *dtmp; Res *dout; int *dids;
    CK(cudaMalloc(&dcs, n * sizeof(Case)));
    CK(cudaMalloc(&dres, res.size() * 2));
    CK(cudaMalloc(&dq, (size_t)n * Q_MAX * 2));
    CK(cudaMalloc(&dtmp, (size_t)n * TMP_FLOATS * 4));
    CK(cudaMalloc(&dout, n * sizeof(Res)));
    CK(cudaMalloc(&dids, m * sizeof(int)));
    CK(cudaMemcpy(dcs, cs.data(), n * sizeof(Case), cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dres, res.data(), res.size() * 2, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dids, ids.data(), m * sizeof(int), cudaMemcpyHostToDevice));
    std::vector<int16_t> q[3];
    std::vector<Res> out[3];
    for (int v = 0; v < 3; v++) {   // 0: warp_eval_rt, 1: rc_eval G = 32, 2: rc_eval G = 16 (ids only)
        CK(cudaMemset(dq, 0, (size_t)n * Q_MAX * 2));
        CK(cudaMemset(dtmp, 0, (size_t)n * TMP_FLOATS * 4));
        CK(cudaMemset(dout, 0xff, n * sizeof(Res)));
        if (v == 0) k_eval<BD, 0><<<(n + 3) / 4, 128>>>(dcs, dres, dtmp, dq, dout, n);
        else if (v == 1) k_eval<BD, 1><<<(n + 3) / 4, 128>>>(dcs, dres, dtmp, dq, dout, n);
        else k_eval_half<BD><<<(m / 2 + 3) / 4, 128>>>(dcs, dids, m, dres, dtmp, dq, dout);
        CK(cudaGetLastError());
        CK(cudaDeviceSynchronize());
        q[v].resize((size_t)n * Q_MAX);
        out[v].resize(n);
        CK(cudaMemcpy(q[v].data(), dq, q[v].size() * 2, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(out[v].data(), dout, n * sizeof(Res), cudaMemcpyDeviceToHost));
    }
    int bad[3] = {0, 0, 0}, shown = 0, nz = 0, en64 = 0;
    for (int v = 1; v < 3; v++)
        for (int k = 0; k < (v == 1 ? n : m); k++) {
            const int i = v == 1 ? k : ids[k];
            const Case &c = cs[i];
            const int nq = std::min(32, (int)AV1G_T(Tx_Width)[c.txSz]) * std::min(32, (int)AV1G_T(Tx_Height)[c.txSz]);
            const Res &a = out[0][i], &b = out[v][i];
            const bool same = !memcmp(&a.D, &b.D, 4) && !memcmp(&a.R, &b.R, 4) && a.eob == b.eob &&
                              !memcmp(&q[0][(size_t)i * Q_MAX], &q[v][(size_t)i * Q_MAX], (size_t)nq * 2);
            if (!same) {
                bad[v]++;
                if (shown++ < 10)
                    printf("  MISMATCH BD %d G %d case %d: txSz %d type %d plane %d q %d qm %d/%d/%d: D %.9g / %.9g, R %.9g / %.9g, eob %d / %d\n",
                           BD, v == 1 ? 32 : 16, i, c.txSz, c.txType, c.plane, c.qindex, c.qmLevel, c.qmDist, c.qmRnd, a.D, b.D,
                           a.R, b.R, a.eob, b.eob);
            }
            if (v == 1) {
                nz += a.eob > 0;
                en64 += (AV1G_T(Tx_Width)[c.txSz] == 64 || AV1G_T(Tx_Height)[c.txSz] == 64) && a.eob > 0;
            }
        }
    printf("BitDepth %d: %d cases vs rc_eval<%d, 32>: %d different; %d cases (up to 8x8) vs rc_eval<%d, 16>: %d different "
           "(%d with eob > 0, %d of them 64-point)\n", BD, n, BD, bad[1], m, BD, bad[2], nz, en64);
    cudaFree(dcs); cudaFree(dres); cudaFree(dq); cudaFree(dtmp); cudaFree(dout); cudaFree(dids);
    return bad[1] + bad[2];
}

int main(int argc, char **argv) {
    const int reps = argc > 1 ? atoi(argv[1]) : 20;
    char err[256];
    if (gp_init_tables(err, sizeof err)) { printf("gp_init_tables: %s\n", err); return 1; }
    if (drift_upload_tables(gp_host_tables())) return 1;
    int bad = run<8>(reps);
    bad += run<10>(reps);
    printf(bad ? "FAILED\n" : "PASSED\n");
    return bad ? 1 : 0;
}
