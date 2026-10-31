// gpu_dsp_test: the av1core bit-exact math is header-only __host__ __device__ code.
// This test compiles it for sm_120 and checks that device results are identical to
// host results for random inputs:
//   1. dequant + 2-D inverse transform (all 19 sizes x 16 types) + conformance flags
//   2. intra prediction (13 modes x angle deltas, all tx sizes, random edges/availability)
//   3. CFL (4:2:0 and 4:4:4 luma sampling, incl. clamping at MaxLumaW / MaxLumaH)
//   4. od_ec range coder + CDF adaptation (one thread per stream; M4 building block)
// Small memory footprint (< 64 MB).  Not a benchmark.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cuda_runtime.h>

#include "av1g/dsp.h"
#include "av1g/entenc.h"
#include "av1g/cdfctx.h"
#include "av1g/blockinfo.h"

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { printf("CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); exit(1); } } while (0)

static uint64_t rs = 0x123456789abcdefull;
static uint32_t rnd() { rs ^= rs << 13; rs ^= rs >> 7; rs ^= rs << 17; return (uint32_t)rs; }

// ---------------- 1. inverse transforms ----------------
struct ItxCase { int txSz, txType, dcQ, acQ, qmLevel, plane; };

__host__ __device__ void itx_case(const ItxCase &c, const int16_t *q, int32_t *res, uint32_t *err, int32_t *dq) {
    av1g_dequant(q, dq, c.txSz, c.dcQ, c.acQ, av1g_qm(c.qmLevel >= 0, c.qmLevel, c.plane, c.txSz, c.txType));
    *err = 0;
    av1g_inv_txfm2d(dq, c.txSz, c.txType, res, err);
}
__global__ void k_itx(const ItxCase *cs, const int16_t *q, int32_t *res, uint32_t *err, int32_t *scratch, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) itx_case(cs[i], q + (size_t)i * 1024, res + (size_t)i * 4096, err + i, scratch + (size_t)i * 1024);
}

// ---------------- 2./3. prediction + CFL ----------------
struct PredCase { int mode, delta, log2W, log2H, haveLeft, haveAbove, haveAR, haveBL, x, y, maxX, maxY, cfl, alpha, edgeFilter, filterType, fiMode;
                  int ss, lumaCutW, lumaCutH; };   // CfL: subsampling (1: 4:2:0, 0: 4:4:4), MaxLumaW/H reduction in 4-sample units
#define PW 160
__host__ __device__ void pred_case(const PredCase &c, const uint8_t *plane, const uint8_t *luma, uint8_t *out) {
    uint8_t ea[AV1G_EDGE_LEN], el[AV1G_EDGE_LEN];
    uint8_t *A = ea + AV1G_EDGE_OFF, *L = el + AV1G_EDGE_OFF;
    // copy the plane so prediction writes stay private
    for (int i = 0; i < PW * PW; i++) out[i] = plane[i];
    av1g_build_edges(out, PW, c.x, c.y, c.log2W, c.log2H, c.haveLeft, c.haveAbove, c.haveAR, c.haveBL, c.maxX, c.maxY, A, L);
    av1g_predict_intra_ex(c.mode, c.delta, c.fiMode, c.edgeFilter, c.filterType, c.haveLeft, c.haveAbove, c.log2W, c.log2H, c.x,
                          c.y, c.maxX, c.maxY, A, L, out + c.y * PW + c.x, PW);
    if (c.cfl) {
        int txSz = -1;
        for (int t = 0; t < 19; t++)
            if (AV1G_T(Tx_Width_Log2)[t] == c.log2W && AV1G_T(Tx_Height_Log2)[t] == c.log2H) txSz = t;
        const int maxLumaW = ((c.x + (1 << c.log2W)) << c.ss) - 4 * c.lumaCutW, maxLumaH = ((c.y + (1 << c.log2H)) << c.ss) - 4 * c.lumaCutH;
        av1g_cfl_predict(out, PW, luma, 2 * PW, c.x, c.y, txSz, c.ss, c.ss, maxLumaW, maxLumaH, c.alpha);
    }
}
__global__ void k_pred(const PredCase *cs, const uint8_t *plane, const uint8_t *luma, uint8_t *out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) pred_case(cs[i], plane + (size_t)i * PW * PW, luma, out + (size_t)i * PW * PW);
}

// ---------------- 4. entropy coder ----------------
struct Sym { uint16_t cdfOff, s, N, isBool; };
__host__ __device__ int ec_stream(const Sym *syms, int nsym, av1g_cdfs *cdf, uint16_t *pbuf, uint32_t cap, uint8_t *out) {
    av1g_ec e;
    av1g_ec_init(&e, pbuf, cap);
    uint16_t *base = (uint16_t *)cdf;
    for (int i = 0; i < nsym; i++) {
        if (syms[i].isBool) av1g_ec_bool(&e, syms[i].s);
        else av1g_ec_symbol(&e, syms[i].s, base + syms[i].cdfOff, syms[i].N);
    }
    return av1g_ec_done(&e, out);
}
__global__ void k_ec(const Sym *syms, const int *off, const int *cnt, av1g_cdfs *cdfs, uint16_t *pbuf, uint8_t *out, int *nbytes,
                     uint32_t cap, int n, int qidx) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    av1g_cdfs_init(&cdfs[i], qidx);
    nbytes[i] = ec_stream(syms + off[i], cnt[i], &cdfs[i], pbuf + (size_t)i * cap, cap, out + (size_t)i * cap);
}

int main() {
    int dev = 0;
    CK(cudaSetDevice(dev));
    cudaDeviceProp prop;
    CK(cudaGetDeviceProperties(&prop, dev));
    printf("device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
    int failures = 0;

    // ---- 1 ----
    {
        const int n = 2048;
        std::vector<ItxCase> cs(n);
        std::vector<int32_t> hscratch(1024);
        std::vector<int16_t> q((size_t)n * 1024, 0);
        for (int i = 0; i < n; i++) {
            ItxCase &c = cs[i];
            c.txSz = rnd() % 19;
            c.txType = (AV1G_T(Tx_Size_Sqr_Up)[c.txSz] == AV1G_TX_64X64) ? AV1G_DCT_DCT : (int)(rnd() % 16);
            if (AV1G_T(Tx_Size_Sqr_Up)[c.txSz] == AV1G_TX_32X32 && c.txType != AV1G_IDTX) c.txType = AV1G_DCT_DCT;
            int qi = 1 + rnd() % 255;
            c.dcQ = av1g_dc_q(qi);
            c.acQ = av1g_ac_q(qi);
            c.qmLevel = (rnd() % 2) ? (int)(rnd() % 16) : -1;  // half the cases with quantizer matrices
            c.plane = rnd() % 3;
            int nc = av1g_tx_coeffs(c.txSz), k = 1 + rnd() % 12;
            for (int j = 0; j < k; j++) {
                int pos = (rnd() % 3 == 0) ? (int)(rnd() % nc) : (int)(rnd() % (nc < 16 ? nc : 16));
                int mag = (rnd() % 10 == 0) ? (int)(rnd() % 3000) : (int)(rnd() % 40);
                q[(size_t)i * 1024 + pos] = (int16_t)((rnd() & 1) ? -mag : mag);
            }
        }
        std::vector<int32_t> hres((size_t)n * 4096), dres((size_t)n * 4096);
        std::vector<uint32_t> herr(n), derr(n);
        for (int i = 0; i < n; i++) itx_case(cs[i], &q[(size_t)i * 1024], &hres[(size_t)i * 4096], &herr[i], hscratch.data());
        ItxCase *dcs; int16_t *dq; int32_t *dr, *dsc; uint32_t *de;
        CK(cudaMalloc(&dsc, (size_t)n * 1024 * 4));
        CK(cudaMalloc(&dcs, n * sizeof(ItxCase))); CK(cudaMalloc(&dq, q.size() * 2));
        CK(cudaMalloc(&dr, dres.size() * 4)); CK(cudaMalloc(&de, n * 4));
        CK(cudaMemcpy(dcs, cs.data(), n * sizeof(ItxCase), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dq, q.data(), q.size() * 2, cudaMemcpyHostToDevice));
        CK(cudaDeviceSetLimit(cudaLimitStackSize, 2048));
        k_itx<<<(n + 63) / 64, 64>>>(dcs, dq, dr, de, dsc, n);
        CK(cudaGetLastError());
        CK(cudaMemcpy(dres.data(), dr, dres.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(derr.data(), de, n * 4, cudaMemcpyDeviceToHost));
        int bad = 0, flagged = 0;
        for (int i = 0; i < n; i++) {
            int w = AV1G_T(Tx_Width)[cs[i].txSz], h = AV1G_T(Tx_Height)[cs[i].txSz];
            if (herr[i] != derr[i] || memcmp(&hres[(size_t)i * 4096], &dres[(size_t)i * 4096], (size_t)w * h * 4)) bad++;
            flagged += herr[i] != 0;
        }
        printf("1. dequant (incl. quantizer matrices) + inverse transforms: %d cases, %d host/device mismatches (%d flagged non-conformant on both)\n", n, bad, flagged);
        failures += bad;
        cudaFree(dcs); cudaFree(dq); cudaFree(dr); cudaFree(de); cudaFree(dsc);
    }
    // ---- 2/3 ----
    {
        const int n = 1024;
        std::vector<PredCase> cs(n);
        std::vector<uint8_t> planes((size_t)n * PW * PW), luma((size_t)4 * PW * PW);
        for (auto &v : planes) v = (uint8_t)rnd();
        for (auto &v : luma) v = (uint8_t)rnd();
        int ncfl = 0, ncfl444 = 0;
        for (int i = 0; i < n; i++) {
            PredCase &c = cs[i];
            int t = rnd() % 19;
            c.log2W = AV1G_T(Tx_Width_Log2)[t];
            c.log2H = AV1G_T(Tx_Height_Log2)[t];
            c.mode = rnd() % 13;
            c.delta = av1g_is_directional(c.mode) ? (int)(rnd() % 7) - 3 : 0;
            c.haveLeft = rnd() & 1; c.haveAbove = rnd() & 1; c.haveAR = rnd() & 1; c.haveBL = rnd() & 1;
            c.x = c.haveLeft ? 1 + rnd() % 8 : 0;
            c.y = c.haveAbove ? 1 + rnd() % 8 : 0;
            c.maxX = c.x + (1 << c.log2W) + (int)(rnd() % 64) - 1;
            c.maxY = c.y + (1 << c.log2H) + (int)(rnd() % 64) - 1;
            if (c.maxX > PW - 1) c.maxX = PW - 1;
            if (c.maxY > PW - 1) c.maxY = PW - 1;
            c.edgeFilter = rnd() & 1;
            c.filterType = rnd() & 1;
            c.fiMode = (c.mode == AV1G_DC_PRED && c.log2W <= 5 && c.log2H <= 5 && (rnd() & 1)) ? (int)(rnd() % 5) : -1;
            c.cfl = (c.log2W <= 5 && c.log2H <= 5 && (rnd() % 3 == 0)) ? 1 : 0;
            c.ss = 1; c.lumaCutW = c.lumaCutH = 0;
            if (c.cfl) {
                c.mode = AV1G_DC_PRED; c.delta = 0; c.alpha = (int)(rnd() % 33) - 16; c.fiMode = -1; ncfl++;
                c.ss = (int)(rnd() & 1);
                ncfl444 += !c.ss;
                // MaxLumaW/H can end before the chroma block (the last luma tx block lies outside the frame)
                if (rnd() % 3 == 0) c.lumaCutW = (int)(rnd() % (((1 << c.log2W) << c.ss) / 4));
                if (rnd() % 3 == 0) c.lumaCutH = (int)(rnd() % (((1 << c.log2H) << c.ss) / 4));
            }
        }
        std::vector<uint8_t> hout(planes.size()), dout(planes.size());
        for (int i = 0; i < n; i++) pred_case(cs[i], &planes[(size_t)i * PW * PW], luma.data(), &hout[(size_t)i * PW * PW]);
        PredCase *dcs; uint8_t *dp, *dl, *dout_;
        CK(cudaMalloc(&dcs, n * sizeof(PredCase))); CK(cudaMalloc(&dp, planes.size()));
        CK(cudaMalloc(&dl, luma.size())); CK(cudaMalloc(&dout_, planes.size()));
        CK(cudaMemcpy(dcs, cs.data(), n * sizeof(PredCase), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dp, planes.data(), planes.size(), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dl, luma.data(), luma.size(), cudaMemcpyHostToDevice));
        k_pred<<<(n + 63) / 64, 64>>>(dcs, dp, dl, dout_, n);
        CK(cudaGetLastError());
        CK(cudaMemcpy(dout.data(), dout_, planes.size(), cudaMemcpyDeviceToHost));
        int bad = 0;
        for (int i = 0; i < n; i++) bad += memcmp(&hout[(size_t)i * PW * PW], &dout[(size_t)i * PW * PW], PW * PW) != 0;
        printf("2/3. intra prediction (incl. edge filter/upsampling, filter intra) + CFL: %d cases (%d CFL, %d of them 4:4:4), %d host/device mismatches\n",
               n, ncfl, ncfl444, bad);
        failures += bad;
        cudaFree(dcs); cudaFree(dp); cudaFree(dl); cudaFree(dout_);
    }
    // ---- 4 ----
    {
        const int n = 128, qidx = 100;
        const uint32_t cap = 1 << 15;
        av1g_cdfs proto;
        av1g_cdfs_init(&proto, qidx);
        // pick symbols from the real CDF layout: (offset, N) of a few arrays
        struct Arr { uint16_t *p; int N, count; };
        Arr arrs[] = {{&proto.y_mode[0][0][0], 13, 25}, {&proto.skip[0][0], 2, 3}, {&proto.coeff_base[0][0][0][0], 4, 5 * 2 * 42},
                      {&proto.coeff_br[0][0][0][0], 4, 5 * 2 * 21}, {&proto.txb_skip[0][0][0], 2, 5 * 13},
                      {&proto.partition_w64[0][0], 10, 4}, {&proto.cfl_alpha[0][0], 16, 6}, {&proto.eob_pt_1024[0][0], 11, 2}};
        std::vector<Sym> syms;
        std::vector<int> off(n), cnt(n);
        for (int i = 0; i < n; i++) {
            off[i] = (int)syms.size();
            cnt[i] = 1 + rnd() % 8000;
            for (int k = 0; k < cnt[i]; k++) {
                Sym s;
                if (rnd() % 6 == 0) { s.isBool = 1; s.s = rnd() & 1; s.cdfOff = 0; s.N = 2; }
                else {
                    Arr &a = arrs[rnd() % 8];
                    int ctx = rnd() % a.count;
                    s.isBool = 0;
                    s.N = (uint16_t)a.N;
                    s.cdfOff = (uint16_t)((a.p - (uint16_t *)&proto) + ctx * (a.N + 1));
                    s.s = (uint16_t)((rnd() % 4) ? rnd() % 2 : rnd() % a.N);
                }
                syms.push_back(s);
            }
        }
        std::vector<std::vector<uint8_t>> hout(n);
        std::vector<uint16_t> pbuf(cap);
        for (int i = 0; i < n; i++) {
            av1g_cdfs c;
            av1g_cdfs_init(&c, qidx);
            hout[i].resize(cap);
            int nb = ec_stream(&syms[off[i]], cnt[i], &c, pbuf.data(), cap, hout[i].data());
            hout[i].resize(nb);
        }
        Sym *ds; int *doff, *dcnt, *dnb; av1g_cdfs *dc; uint16_t *dpb; uint8_t *dout;
        CK(cudaMalloc(&ds, syms.size() * sizeof(Sym))); CK(cudaMalloc(&doff, n * 4)); CK(cudaMalloc(&dcnt, n * 4));
        CK(cudaMalloc(&dnb, n * 4)); CK(cudaMalloc(&dc, n * sizeof(av1g_cdfs))); CK(cudaMalloc(&dpb, (size_t)n * cap * 2));
        CK(cudaMalloc(&dout, (size_t)n * cap));
        CK(cudaMemcpy(ds, syms.data(), syms.size() * sizeof(Sym), cudaMemcpyHostToDevice));
        CK(cudaMemcpy(doff, off.data(), n * 4, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(dcnt, cnt.data(), n * 4, cudaMemcpyHostToDevice));
        k_ec<<<(n + 31) / 32, 32>>>(ds, doff, dcnt, dc, dpb, dout, dnb, cap, n, qidx);
        CK(cudaGetLastError());
        std::vector<int> nb(n);
        std::vector<uint8_t> dbytes((size_t)n * cap);
        CK(cudaMemcpy(nb.data(), dnb, n * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(dbytes.data(), dout, dbytes.size(), cudaMemcpyDeviceToHost));
        int bad = 0;
        long long tot = 0;
        for (int i = 0; i < n; i++) {
            tot += cnt[i];
            if (nb[i] != (int)hout[i].size() || memcmp(hout[i].data(), &dbytes[(size_t)i * cap], nb[i])) bad++;
        }
        printf("4. od_ec + CDF adaptation: %d streams, %lld symbols, %d host/device mismatches\n", n, tot, bad);
        failures += bad;
        cudaFree(ds); cudaFree(doff); cudaFree(dcnt); cudaFree(dnb); cudaFree(dc); cudaFree(dpb); cudaFree(dout);
    }
    printf("%s\n", failures ? "GPU_DSP_TEST FAILED" : "GPU_DSP_TEST PASSED");
    return failures ? 1 : 0;
}
