// av1core GPU entropy: GPU symbolization + entropy coding.
//
// Symbolization is parallel because every context of the AV1 intra syntax is a
// function of the DecisionRecord only (never of CDF values):
//   prep       k_blocks (1 thread / block): frame, tile, global tx block index, SB starts,
//              decode-order checks; k_sbs (1 / SB): SB block ranges; k_tiles (1 warp / tile):
//              tile block ranges + the delta-q CurrentQIndex chain (warp scan)
//   k_maps     1 thread / block:   4x4 mode-info maps (MiSize, YMode, Skip, TxSize)
//   k_txctx    1 thread / tx block: eob, culLevel, dcCategory -> per-plane 4x4
//                                   context maps (= Above/Left Level/Dc contexts)
//   k_txsym    1 thread / tx block: coefficient syntax (all_zero, tx type, eob,
//                                   coeff_base(_eob), coeff_br, signs, Golomb)
//   k_sbsym    1 thread / 64x64 SB: partition walk + mode info syntax
//   (count pass -> CUB exclusive scan -> emit pass, so streams are in bitstream order)
//   k_entropy_win  1 warp / tile: windows of 32 symbols - CDF adaptation lane-parallel
//                  (serialised only between symbols of the same CDF), then lane 0 range-codes
//                  the precomputed (fl, fh) words with in-place carry propagation
//   k_pack     tile payloads -> one packed buffer
// Output is byte-identical to the CPU writer (src/encoder.cpp), which uses the
// same headers (dsp.h, blockinfo.h, cdfctx.h, symstream.h, entenc.h).
#include "gpu_entropy.h"
#include "internal.h"
#include "av1g/av1g.h"
#include "av1g/blockinfo.h"
#include "av1g/cdfctx.h"
#include "av1g/dsp.h"
#include "av1g/entenc.h"
#include "av1g/symstream.h"

#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdio>
#include <cstring>

namespace av1g_gpu {

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { err = std::string("CUDA: ") + cudaGetErrorString(e_) + " at " #x; return -1; } } while (0)

struct FrameG {
    int W, H, MiCols, MiRows, mono, qidx, reduced, txsel, sct, dq, dqRes, fi, cdefBits;  // cdefBits -1: CDEF off
    int ssx, ssy;                                // chroma subsampling (4:2:0 / monochrome 1, 4:4:4 0)
    uint64_t miBase;
    uint64_t ctxBase[3];
    int ctxStride[3], ctxRows[3];
    uint32_t blockBase, nBlocks, txbBase, nTxb;  // batch-global (relative to frame 0)
    uint64_t coeffBase;                          // absolute index into the caller's coeffs[]
    uint32_t sbBase, nSb, tileBase, nTiles;
    int tileCols, tileRows, tileWsb, tileHsb;    // uniform tile grid
};
// lossless frame (base_q_idx 0, every DeltaQ 0: av1g_encode's rule): 4x4 WHT, no tx types, the lossless CfL rule.
// The symbolization kernels take a template flag LL: batches without lossless frames run the LL = false
// instantiations, which compile exactly as before lossless support.
__device__ __forceinline__ int frame_lossless(const FrameG &F) { return F.qidx == 0; }
struct TileG {
    uint32_t frame;
    int miRowStart, miRowEnd, miColStart, miColEnd;
    uint32_t firstBlock, endBlock;               // device-computed
    uint32_t sbBase, nSb;                        // global decode-order SB range
    int sbRow0, sbCol0, sbCols;
};
struct SbG {
    uint32_t frame, tile, firstBlock, endBlock;
    int r, c;
    int qBefore;  // CurrentQIndex before this SB (delta-q chain, computed per tile on the device)
};

// a dummy instance whose field addresses define the cdf ids (as in the CPU TeeSink)
__device__ av1g_cdfs g_cdf_layout;
#define CID(expr) ((uint32_t)((const uint16_t *)&(g_cdf_layout.expr) - (const uint16_t *)&g_cdf_layout))

struct Out {
    uint32_t *p;
    uint32_t n;
    __device__ void put(uint32_t w) { if (p) p[n] = w; n++; }
    __device__ void sym(uint32_t cid, int s, int N) { put(av1g_sym_pack(AV1G_SYM_ADAPT, cid, (uint32_t)s, (uint32_t)N)); }
    __device__ void bit(int b) { put(av1g_sym_pack(AV1G_SYM_BOOL, 0, (uint32_t)b, 2)); }
};

// ---------------------------------------------------------------------------
__global__ void k_maps(const av1g_block *blocks, const uint32_t *blockFrame, const FrameG *frames, uint32_t nBlocks,
                       uint8_t *mMiSize, uint8_t *mYMode, uint8_t *mSkip, uint8_t *mTx) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nBlocks) return;
    const av1g_block k = blocks[b];
    const FrameG &F = frames[blockFrame[b]];
    const int bw4 = av1g_bw4(k.bsize), bh4 = av1g_bh4(k.bsize);
    const int rEnd = min(k.mi_row + bh4, F.MiRows), cEnd = min(k.mi_col + bw4, F.MiCols);
    const uint8_t tx = (uint8_t)av1g_luma_tx_size(k.bsize, k.tx_depth);
    for (int y = k.mi_row; y < rEnd; y++)
        for (int x = k.mi_col; x < cEnd; x++) {
            const uint64_t i = F.miBase + (uint64_t)y * F.MiCols + x;
            mMiSize[i] = k.bsize;
            mYMode[i] = k.y_mode;
            mSkip[i] = k.skip;
            mTx[i] = tx;
        }
}

template <bool LL>
__global__ void k_txctx(const av1g_block *blocks, const uint32_t *blockFrame, const FrameG *frames, const av1g_txb *txb,
                        const uint32_t *txbBlock, const int16_t *coeffs, uint32_t nTxb,
                        uint8_t *ctxLvl, uint8_t *ctxDc, uint16_t *txbEob, uint8_t *txbType) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nTxb) return;
    const uint32_t b = txbBlock[t];
    const av1g_block k = blocks[b];
    const FrameG &F = frames[blockFrame[b]];
    const av1g_txb x = txb[t];
    const int plane = x.plane, txSz = x.tx_size;
    int type = plane ? av1g_chroma_tx_type(txSz, k.uv_mode, F.reduced) : x.tx_type;
    if (LL && frame_lossless(F)) type = AV1G_DCT_DCT;   // lossless: compute_tx_type is DCT_DCT (the WHT) in every plane
    int eob = 0, cul = 0, dc = 0;
    if (!k.skip) {
        const int16_t *q = coeffs + F.coeffBase + x.coeff_off;
        const uint16_t *scan = av1g_get_scan(txSz, type);
        const int n = av1g_tx_coeffs(txSz);
        for (int c = n - 1; c >= 0; c--)
            if (q[scan[c]]) { eob = c + 1; break; }
        for (int c = 0; c < eob; c++) {
            int v = q[scan[c]];
            cul += v < 0 ? -v : v;
        }
        if (cul > 63) cul = 63;
        dc = q[0] < 0 ? 1 : q[0] > 0 ? 2 : 0;
    }
    if (!eob && plane == 0) type = AV1G_DCT_DCT;
    txbEob[t] = (uint16_t)eob;
    txbType[t] = (uint8_t)type;
    const int w4 = av1g_tx_w(txSz) >> 2, h4 = av1g_tx_h(txSz) >> 2;
    const int stride = F.ctxStride[plane], rows = F.ctxRows[plane];
    for (int i = 0; i < h4 && x.y4 + i < rows; i++)
        for (int j = 0; j < w4 && x.x4 + j < stride; j++) {
            const uint64_t o = F.ctxBase[plane] + (uint64_t)(x.y4 + i) * stride + x.x4 + j;
            ctxLvl[o] = (uint8_t)cul;
            ctxDc[o] = (uint8_t)dc;
        }
}

// coefficient syntax of one tx block (mirror of Encoder::code_coeffs, write mode)
template <bool LL>
__global__ void k_txsym(const av1g_block *blocks, const uint32_t *blockFrame, const uint32_t *blockTile, const FrameG *frames,
                        const TileG *tiles, const av1g_txb *txb, const uint32_t *txbBlock,
                        const int16_t *coeffs, uint32_t nTxb, const uint8_t *ctxLvl, const uint8_t *ctxDc,
                        const uint16_t *txbEob, const uint8_t *txbType, uint32_t *itemCount, const uint32_t *itemOff,
                        uint32_t *syms, const uint32_t *symTotal, uint64_t symCap) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nTxb) return;
    if (syms && *symTotal > symCap) return;  // emit pass with too small a symbol buffer (reported later)
    const uint32_t b = txbBlock[t];
    const uint32_t item = t + b + 1;
    const av1g_block k = blocks[b];
    Out o{syms ? syms + itemOff[item] : nullptr, 0};
    if (k.skip) {
        if (!syms) itemCount[item] = 0;
        return;
    }
    const FrameG &F = frames[blockFrame[b]];
    const TileG &T = tiles[blockTile[b]];
    const av1g_txb x = txb[t];
    const int plane = x.plane, txSz = x.tx_size, ptype = plane > 0;
    const int subX = plane ? F.ssx : 0, subY = plane ? F.ssy : 0;
    const int x4 = x.x4, y4 = x.y4;
    const int eob = txbEob[t], txType = txbType[t];
    const int16_t *q = coeffs + F.coeffBase + x.coeff_off;
    const int txSzCtx = (AV1G_T(Tx_Size_Sqr)[txSz] + AV1G_T(Tx_Size_Sqr_Up)[txSz] + 1) >> 1;
    const int w = AV1G_T(Tx_Width)[txSz], h = AV1G_T(Tx_Height)[txSz], w4 = w >> 2, h4 = h >> 2;
    const int maxX4 = F.MiCols >> subX, maxY4 = F.MiRows >> subY;
    const int stride = F.ctxStride[plane];
    const uint8_t *L = ctxLvl + F.ctxBase[plane], *Dc = ctxDc + F.ctxBase[plane];
    const bool hasAbove = y4 > (T.miRowStart >> subY), hasLeft = x4 > (T.miColStart >> subX);
    auto aLvl = [&](int k_) -> int { return hasAbove ? L[(uint64_t)(y4 - 1) * stride + x4 + k_] : 0; };
    auto aDc = [&](int k_) -> int { return hasAbove ? Dc[(uint64_t)(y4 - 1) * stride + x4 + k_] : 0; };
    auto lLvl = [&](int k_) -> int { return hasLeft ? L[(uint64_t)(y4 + k_) * stride + x4 - 1] : 0; };
    auto lDc = [&](int k_) -> int { return hasLeft ? Dc[(uint64_t)(y4 + k_) * stride + x4 - 1] : 0; };
    int ctx;
    {
        const int bsz = plane ? av1g_uv_bsize(k.bsize, F.ssx, F.ssy) : k.bsize;
        const int bw = 4 * av1g_bw4(bsz), bh = 4 * av1g_bh4(bsz);
        if (plane == 0) {
            int top = 0, left = 0;
            for (int i = 0; i < w4; i++) if (x4 + i < maxX4) top = max(top, aLvl(i));
            for (int i = 0; i < h4; i++) if (y4 + i < maxY4) left = max(left, lLvl(i));
            const int mx = max(top, left), mn = min(top, left);
            if (bw == w && bh == h) ctx = 0;
            else if (top == 0 && left == 0) ctx = 1;
            else if (top == 0 || left == 0) ctx = 2 + (mx > 3);
            else if (mx <= 3) ctx = 4;
            else if (mn <= 3) ctx = 5;
            else ctx = 6;
        } else {
            int above = 0, left = 0;
            for (int i = 0; i < w4; i++) if (x4 + i < maxX4) above |= aLvl(i) | aDc(i);
            for (int i = 0; i < h4; i++) if (y4 + i < maxY4) left |= lLvl(i) | lDc(i);
            ctx = (above != 0) + (left != 0) + 7;
            if (bw * bh > w * h) ctx += 3;
        }
    }
    o.sym(CID(txb_skip[txSzCtx][ctx]), eob == 0, 2);
    if (eob > 0) {
        if (plane == 0) {
            const int set = av1g_get_tx_set_intra(txSz, F.reduced);
            if (set > 0 && (!LL || F.qidx > 0)) {   // transform_type: coded only for base_q_idx > 0
                const int sqr = AV1G_T(Tx_Size_Sqr)[txSz];
                const int intraDir = k.filter_intra ? AV1G_T(Filter_Intra_Mode_To_Intra_Dir)[k.filter_intra - 1] : k.y_mode;
                int idx = 0;
                if (set == AV1G_TX_SET_INTRA_1) {
                    for (int i = 0; i < 7; i++) if (AV1G_T(Tx_Type_Intra_Inv_Set1)[i] == txType) idx = i;
                    o.sym(CID(intra_tx_set1[sqr][intraDir]), idx, 7);
                } else {
                    for (int i = 0; i < 5; i++) if (AV1G_T(Tx_Type_Intra_Inv_Set2)[i] == txType) idx = i;
                    o.sym(CID(intra_tx_set2[sqr][intraDir]), idx, 5);
                }
            }
        }
        const uint16_t *scan = av1g_get_scan(txSz, txType);
        const int txClass = av1g_tx_class(txType);
        const int eobMultisize = min((int)AV1G_T(Tx_Width_Log2)[txSz], 5) + min((int)AV1G_T(Tx_Height_Log2)[txSz], 5) - 4;
        int eobPt;
        if (eob <= 2) eobPt = eob;
        else { eobPt = 2; int v = eob - 1; while (v > 1) { v >>= 1; eobPt++; } }
        {
            const int ectx = txClass == AV1G_TX_CLASS_2D ? 0 : 1;
            uint32_t c;
            switch (eobMultisize) {
            case 0: c = CID(eob_pt_16[ptype][ectx]); break;
            case 1: c = CID(eob_pt_32[ptype][ectx]); break;
            case 2: c = CID(eob_pt_64[ptype][ectx]); break;
            case 3: c = CID(eob_pt_128[ptype][ectx]); break;
            case 4: c = CID(eob_pt_256[ptype][ectx]); break;
            case 5: c = CID(eob_pt_512[ptype]); break;
            default: c = CID(eob_pt_1024[ptype]); break;
            }
            o.sym(c, eobPt - 1, eobMultisize + 5);
        }
        if (eobPt >= 3) {
            const int rem = eob - ((1 << (eobPt - 2)) + 1);
            o.sym(CID(eob_extra[txSzCtx][ptype][eobPt - 3]), (rem >> (eobPt - 3)) & 1, 2);
            for (int i = 1; i < eobPt - 2; i++) o.bit((rem >> (eobPt - 3 - i)) & 1);
        }
        const int adj = AV1G_T(Adjusted_Tx_Size)[txSz];
        const int bwl = AV1G_T(Tx_Width_Log2)[adj];
        const int width = 1 << bwl, height = AV1G_T(Tx_Height)[adj];
        uint8_t lev[1024];
        for (int i = 0; i < width * height; i++) lev[i] = 0;
        for (int c = eob - 1; c >= 0; c--) {
            const int pos = scan[c];
            const int v = q[pos] < 0 ? -q[pos] : q[pos];
            const int level = v < 15 ? v : 15;
            const int row = pos >> bwl, col = pos - (row << bwl);
            if (c == eob - 1) {
                int bctx = c == 0 ? 0 : c <= (height << bwl) / 8 ? 1 : c <= (height << bwl) / 4 ? 2 : 3;
                o.sym(CID(coeff_base_eob[txSzCtx][ptype][bctx]), min(level, 3) - 1, 3);
            } else {
                int mag = 0;
                for (int idx = 0; idx < AV1G_SIG_REF_DIFF_OFFSET_NUM; idx++) {
                    const int rr = row + AV1G_T(Sig_Ref_Diff_Offset)[txClass][idx][0];
                    const int cc = col + AV1G_T(Sig_Ref_Diff_Offset)[txClass][idx][1];
                    if (rr >= 0 && cc >= 0 && rr < height && cc < width) mag += min((int)lev[(rr << bwl) + cc], 3);
                }
                int bctx = min((mag + 1) >> 1, 4);
                if (txClass == AV1G_TX_CLASS_2D) {
                    if (row == 0 && col == 0) bctx = 0;
                    else bctx += AV1G_T(Coeff_Base_Ctx_Offset)[txSz][min(row, 4)][min(col, 4)];
                } else {
                    const int id = txClass == AV1G_TX_CLASS_VERT ? row : col;
                    bctx += AV1G_T(Coeff_Base_Pos_Ctx_Offset)[min(id, 2)];
                }
                o.sym(CID(coeff_base[txSzCtx][ptype][bctx]), min(level, 3), 4);
            }
            if (level > AV1G_NUM_BASE_LEVELS) {
                int mag = 0;
                for (int idx = 0; idx < 3; idx++) {
                    const int rr = row + AV1G_T(Mag_Ref_Offset_With_Tx_Class)[txClass][idx][0];
                    const int cc = col + AV1G_T(Mag_Ref_Offset_With_Tx_Class)[txClass][idx][1];
                    if (rr >= 0 && cc >= 0 && rr < height && cc < width) mag += min((int)lev[rr * width + cc], 15);
                }
                mag = min((mag + 1) >> 1, 6);
                int brctx;
                if (pos == 0) brctx = mag;
                else if (txClass == AV1G_TX_CLASS_2D) brctx = (row < 2 && col < 2) ? mag + 7 : mag + 14;
                else if (txClass == AV1G_TX_CLASS_HORIZ) brctx = col == 0 ? mag + 7 : mag + 14;
                else brctx = row == 0 ? mag + 7 : mag + 14;
                const int brsz = min(txSzCtx, (int)AV1G_TX_32X32);
                int rem = level - 3;
                for (int idx = 0; idx < AV1G_COEFF_BASE_RANGE / (AV1G_BR_CDF_SIZE - 1); idx++) {
                    const int kk = min(rem, 3);
                    o.sym(CID(coeff_br[brsz][ptype][brctx]), kk, AV1G_BR_CDF_SIZE);
                    rem -= kk;
                    if (kk < 3) break;
                }
            }
            lev[pos] = (uint8_t)level;
        }
        for (int c = 0; c < eob; c++) {
            const int pos = scan[c];
            const int v = q[pos];
            if (!v) continue;
            const int sign = v < 0, a = sign ? -v : v;
            if (c == 0) {
                int dcSign = 0;
                for (int i = 0; i < w4; i++)
                    if (x4 + i < maxX4) { int s = aDc(i); dcSign += s == 1 ? -1 : s == 2 ? 1 : 0; }
                for (int i = 0; i < h4; i++)
                    if (y4 + i < maxY4) { int s = lDc(i); dcSign += s == 1 ? -1 : s == 2 ? 1 : 0; }
                o.sym(CID(dc_sign[ptype][dcSign < 0 ? 1 : dcSign > 0 ? 2 : 0]), sign, 2);
            } else {
                o.bit(sign);
            }
            if (a > AV1G_NUM_BASE_LEVELS + AV1G_COEFF_BASE_RANGE) {
                const uint32_t xg = (uint32_t)(a - (AV1G_NUM_BASE_LEVELS + AV1G_COEFF_BASE_RANGE));
                int len = 0;
                while ((xg >> len) > 1) len++;
                for (int i = 0; i < len; i++) o.bit(0);
                o.bit(1);
                for (int i = len - 1; i >= 0; i--) o.bit((xg >> i) & 1);
            }
        }
    }
    if (!syms) itemCount[item] = o.n;
}

// ---------------------------------------------------------------------------
// superblock walk: partition + mode info (mirror of encode_partition / encode_block)
struct Walk {
    const FrameG *F;
    const TileG *T;
    const av1g_block *blocks;
    const uint32_t *blockTxbG;
    const uint8_t *mMiSize, *mYMode, *mSkip, *mTx;  // frame-relative
    uint32_t cursor, end;
    uint32_t *itemCount;
    const uint32_t *itemOff;
    uint32_t *syms;
    uint32_t pend[8];
    int npend;
    int curQ, firstInSb, cdefDone;
    DevStatus *st;
    __device__ bool inside(int r, int c) const {
        return c >= T->miColStart && c < T->miColEnd && r >= T->miRowStart && r < T->miRowEnd;
    }
    __device__ int map(const uint8_t *m, int r, int c) const { return m[(uint64_t)r * F->MiCols + c]; }
};

__device__ int w_derive(Walk &w, int r, int c, int bSize, int hasRows, int hasCols) {
    const av1g_block &b = w.blocks[w.cursor];
    const int half = av1g_bw4(bSize) >> 1;
    const int hz = AV1G_T(Partition_Subsize)[AV1G_PARTITION_HORZ][bSize];
    const int vt = AV1G_T(Partition_Subsize)[AV1G_PARTITION_VERT][bSize];
    const int sp = AV1G_T(Partition_Subsize)[AV1G_PARTITION_SPLIT][bSize];
    const int h4 = AV1G_T(Partition_Subsize)[AV1G_PARTITION_HORZ_4][bSize];
    const int v4 = AV1G_T(Partition_Subsize)[AV1G_PARTITION_VERT_4][bSize];
    auto at = [&](uint32_t k, int rr, int cc, int bs) {
        if (w.cursor + k >= w.end) return false;
        const av1g_block &n = w.blocks[w.cursor + k];
        return n.mi_row == rr && n.mi_col == cc && n.bsize == bs;
    };
    const int bs = b.bsize;
    if (bs == bSize) return AV1G_PARTITION_NONE;
    if (bs == hz) return at(1, r + half, c, sp) ? AV1G_PARTITION_HORZ_B : AV1G_PARTITION_HORZ;
    if (bs == vt) return at(1, r, c + half, sp) ? AV1G_PARTITION_VERT_B : AV1G_PARTITION_VERT;
    if (bs == sp && bSize > AV1G_BLOCK_8X8 && hasRows && hasCols && at(1, r + half, c, sp)) return AV1G_PARTITION_VERT_A;
    if (bs == sp && bSize > AV1G_BLOCK_8X8 && hasRows && hasCols && at(1, r, c + half, sp) && at(2, r + half, c, hz))
        return AV1G_PARTITION_HORZ_A;
    if (bSize > AV1G_BLOCK_8X8 && bs == h4) return AV1G_PARTITION_HORZ_4;
    if (bSize > AV1G_BLOCK_8X8 && bs == v4) return AV1G_PARTITION_VERT_4;
    return AV1G_PARTITION_SPLIT;
}

template <bool LL>
__device__ void w_block(Walk &w, int r, int c, int bsize) {
    if (w.cursor >= w.end) { atomicOr(&w.st->error, AV1G_GPU_ERR_ORDER); return; }
    const uint32_t bi = w.cursor++;
    const av1g_block k = w.blocks[bi];
    const FrameG &F = *w.F;
    const uint32_t item = bi + w.blockTxbG[bi];
    Out o{w.syms ? w.syms + w.itemOff[item] : nullptr, 0};
    for (int i = 0; i < w.npend; i++) o.put(w.pend[i]);
    w.npend = 0;
    const int bw4 = av1g_bw4(bsize), bh4 = av1g_bh4(bsize);
    const int hasChroma = av1g_has_chroma(bsize, r, c, F.mono, F.ssx, F.ssy);
    const int aU = w.inside(r - 1, c), aL = w.inside(r, c - 1);
    o.sym(CID(skip[(aU ? w.map(w.mSkip, r - 1, c) : 0) + (aL ? w.map(w.mSkip, r, c - 1) : 0)]), k.skip, 2);
    if (F.cdefBits >= 0 && !k.skip && !w.cdefDone) {  // read_cdef: first non-skip block of the 64x64
        for (int i = F.cdefBits - 1; i >= 0; i--) o.bit((k.cdef_idx >> i) & 1);
        w.cdefDone = 1;
    }
    if (F.dq && w.firstInSb && !(bsize == AV1G_BLOCK_64X64 && k.skip)) {  // read_delta_qindex
        const int reduced = ((int)k.sb_qindex - w.curQ) / (1 << F.dqRes);
        const int a = reduced < 0 ? -reduced : reduced;
        o.sym(CID(delta_q), min(a, (int)AV1G_DELTA_Q_SMALL), AV1G_DELTA_Q_SMALL + 1);
        if (a >= AV1G_DELTA_Q_SMALL) {
            int rem = 1;
            while (((a - 1) >> (rem + 1)) > 0) rem++;
            for (int i = 2; i >= 0; i--) o.bit(((rem - 1) >> i) & 1);
            const int bits = (a - 1) - (1 << rem);
            for (int i = rem - 1; i >= 0; i--) o.bit((bits >> i) & 1);
        }
        if (a) o.bit(reduced < 0);
        w.curQ = k.sb_qindex;
    }
    w.firstInSb = 0;
    {
        const int am = AV1G_T(Intra_Mode_Context)[aU ? w.map(w.mYMode, r - 1, c) : AV1G_DC_PRED];
        const int lm = AV1G_T(Intra_Mode_Context)[aL ? w.map(w.mYMode, r, c - 1) : AV1G_DC_PRED];
        o.sym(CID(y_mode[am][lm]), k.y_mode, AV1G_INTRA_MODES);
    }
    if (bsize >= AV1G_BLOCK_8X8 && av1g_is_directional(k.y_mode))
        o.sym(CID(angle_delta[k.y_mode - AV1G_V_PRED]), k.angle_delta_y + 3, 7);
    if (hasChroma) {
        if (av1g_cfl_allowed_ex(bsize, LL && frame_lossless(F), F.ssx, F.ssy)) o.sym(CID(uv_mode_cfl_allowed[k.y_mode]), k.uv_mode, 14);
        else o.sym(CID(uv_mode_cfl_not_allowed[k.y_mode]), k.uv_mode, 13);
        if (k.uv_mode == AV1G_UV_CFL_PRED) {
            const int sU = k.cfl_alpha_u == 0 ? 0 : k.cfl_alpha_u < 0 ? 1 : 2;
            const int sV = k.cfl_alpha_v == 0 ? 0 : k.cfl_alpha_v < 0 ? 1 : 2;
            o.sym(CID(cfl_sign), sU * 3 + sV - 1, 8);
            if (sU) o.sym(CID(cfl_alpha[(sU - 1) * 3 + sV]), (k.cfl_alpha_u < 0 ? -k.cfl_alpha_u : k.cfl_alpha_u) - 1, 16);
            if (sV) o.sym(CID(cfl_alpha[(sV - 1) * 3 + sU]), (k.cfl_alpha_v < 0 ? -k.cfl_alpha_v : k.cfl_alpha_v) - 1, 16);
        }
        if (bsize >= AV1G_BLOCK_8X8 && av1g_is_directional(k.uv_mode))
            o.sym(CID(angle_delta[k.uv_mode - AV1G_V_PRED]), k.angle_delta_uv + 3, 7);
    }
    if (F.sct && bsize >= AV1G_BLOCK_8X8 && bw4 <= 16 && bh4 <= 16) {
        const int bsizeCtx = AV1G_T(Mi_Width_Log2)[bsize] + AV1G_T(Mi_Height_Log2)[bsize] - 2;
        if (k.y_mode == AV1G_DC_PRED) o.sym(CID(palette_y_mode[bsizeCtx][0]), 0, 2);
        if (hasChroma && k.uv_mode == AV1G_DC_PRED) o.sym(CID(palette_uv_mode[0]), 0, 2);
    }
    if (F.fi && k.y_mode == AV1G_DC_PRED && bw4 <= 8 && bh4 <= 8) {  // filter_intra_mode_info
        o.sym(CID(filter_intra[bsize]), k.filter_intra != 0, 2);
        if (k.filter_intra) o.sym(CID(filter_intra_mode), k.filter_intra - 1, 5);
    }
    if (F.txsel && bsize > AV1G_BLOCK_4X4) {
        const int maxRect = AV1G_T(Max_Tx_Size_Rect)[bsize];
        const int maxTxDepth = AV1G_T(Max_Tx_Depth)[bsize];
        const int aboveW = aU ? AV1G_T(Tx_Width)[w.map(w.mTx, r - 1, c)] : 0;
        const int leftH = aL ? AV1G_T(Tx_Height)[w.map(w.mTx, r, c - 1)] : 0;
        const int ctx = (aboveW >= AV1G_T(Tx_Width)[maxRect]) + (leftH >= AV1G_T(Tx_Height)[maxRect]);
        if (maxTxDepth == 4) o.sym(CID(tx_64x64[ctx]), k.tx_depth, 3);
        else if (maxTxDepth == 3) o.sym(CID(tx_32x32[ctx]), k.tx_depth, 3);
        else if (maxTxDepth == 2) o.sym(CID(tx_16x16[ctx]), k.tx_depth, 3);
        else o.sym(CID(tx_8x8[ctx]), k.tx_depth, 2);
    }
    if (!w.syms) w.itemCount[item] = o.n;
}

template <bool LL>
__device__ void w_partition(Walk &w, int r, int c, int bSize) {
    const FrameG &F = *w.F;
    if (r >= F.MiRows || c >= F.MiCols) return;
    const int half = av1g_bw4(bSize) >> 1, quarter = half >> 1;
    const int hasRows = (r + half) < F.MiRows, hasCols = (c + half) < F.MiCols;
    int p;
    if (w.cursor >= w.end) { atomicOr(&w.st->error, AV1G_GPU_ERR_ORDER); return; }
    if (bSize < AV1G_BLOCK_8X8) {
        p = AV1G_PARTITION_NONE;
    } else {
        p = w_derive(w, r, c, bSize, hasRows, hasCols);
        const int bsl = AV1G_T(Mi_Width_Log2)[bSize];
        const int above = w.inside(r - 1, c) && AV1G_T(Mi_Width_Log2)[w.map(w.mMiSize, r - 1, c)] < bsl;
        const int left = w.inside(r, c - 1) && AV1G_T(Mi_Height_Log2)[w.map(w.mMiSize, r, c - 1)] < bsl;
        const int ctx = left * 2 + above;
        const uint32_t cid = bsl == 1 ? CID(partition_w8[ctx]) : bsl == 2 ? CID(partition_w16[ctx])
                           : bsl == 3 ? CID(partition_w32[ctx]) : CID(partition_w64[ctx]);
        if (hasRows && hasCols) w.pend[w.npend++] = av1g_sym_pack(AV1G_SYM_ADAPT, cid, p, bsl == 1 ? 4 : 10);
        else if (hasCols) w.pend[w.npend++] = av1g_sym_pack(AV1G_SYM_SPLIT_OR_HORZ, cid, p == AV1G_PARTITION_SPLIT, 2);
        else if (hasRows) w.pend[w.npend++] = av1g_sym_pack(AV1G_SYM_SPLIT_OR_VERT, cid, p == AV1G_PARTITION_SPLIT, 2);
    }
    const int sub = AV1G_T(Partition_Subsize)[p][bSize];
    const int sp = AV1G_T(Partition_Subsize)[AV1G_PARTITION_SPLIT][bSize];
    switch (p) {
    case AV1G_PARTITION_NONE: w_block<LL>(w, r, c, sub); break;
    case AV1G_PARTITION_HORZ: w_block<LL>(w, r, c, sub); if (hasRows) w_block<LL>(w, r + half, c, sub); break;
    case AV1G_PARTITION_VERT: w_block<LL>(w, r, c, sub); if (hasCols) w_block<LL>(w, r, c + half, sub); break;
    case AV1G_PARTITION_SPLIT:
        w_partition<LL>(w, r, c, sub); w_partition<LL>(w, r, c + half, sub);
        w_partition<LL>(w, r + half, c, sub); w_partition<LL>(w, r + half, c + half, sub);
        break;
    case AV1G_PARTITION_HORZ_A: w_block<LL>(w, r, c, sp); w_block<LL>(w, r, c + half, sp); w_block<LL>(w, r + half, c, sub); break;
    case AV1G_PARTITION_HORZ_B: w_block<LL>(w, r, c, sub); w_block<LL>(w, r + half, c, sp); w_block<LL>(w, r + half, c + half, sp); break;
    case AV1G_PARTITION_VERT_A: w_block<LL>(w, r, c, sp); w_block<LL>(w, r + half, c, sp); w_block<LL>(w, r, c + half, sub); break;
    case AV1G_PARTITION_VERT_B: w_block<LL>(w, r, c, sub); w_block<LL>(w, r, c + half, sp); w_block<LL>(w, r + half, c + half, sp); break;
    case AV1G_PARTITION_HORZ_4:
        for (int i = 0; i < 4; i++) if (i < 3 || r + quarter * 3 < F.MiRows) w_block<LL>(w, r + quarter * i, c, sub);
        break;
    default:
        for (int i = 0; i < 4; i++) if (i < 3 || c + quarter * 3 < F.MiCols) w_block<LL>(w, r, c + quarter * i, sub);
        break;
    }
}

template <bool LL>
__global__ void k_sbsym(const SbG *sbs, uint32_t nSb, const FrameG *frames, const TileG *tiles, const av1g_block *blocks,
                        const uint32_t *blockTxbG, const uint8_t *mMiSize, const uint8_t *mYMode, const uint8_t *mSkip,
                        const uint8_t *mTx, uint32_t *itemCount, const uint32_t *itemOff, uint32_t *syms,
                        const uint32_t *symTotal, uint64_t symCap, DevStatus *st) {
    const uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nSb) return;
    if (syms && *symTotal > symCap) return;
    const SbG sb = sbs[s];
    Walk w;
    w.F = &frames[sb.frame];
    w.T = &tiles[sb.tile];
    w.blocks = blocks;
    w.blockTxbG = blockTxbG;
    w.mMiSize = mMiSize + w.F->miBase;
    w.mYMode = mYMode + w.F->miBase;
    w.mSkip = mSkip + w.F->miBase;
    w.mTx = mTx + w.F->miBase;
    w.cursor = sb.firstBlock;
    w.end = sb.endBlock;
    w.itemCount = itemCount;
    w.itemOff = itemOff;
    w.syms = syms;
    w.npend = 0;
    w.curQ = sb.qBefore;
    w.firstInSb = 1;
    w.cdefDone = 0;
    w.st = st;
    if (sb.firstBlock >= sb.endBlock) { atomicOr(&st->error, AV1G_GPU_ERR_ORDER); return; }
    w_partition<LL>(w, sb.r, sb.c, AV1G_BLOCK_64X64);
    if (w.cursor != w.end) atomicOr(&st->error, AV1G_GPU_ERR_ORDER);
}

// ---------------------------------------------------------------------------
__global__ void k_entropy(const uint32_t *syms, const uint64_t *tBegin, const uint64_t *tEnd, const int *tQ,
                          const uint64_t *capOff, av1g_cdfs *scratch, uint16_t *precarry, uint8_t *outBytes,
                          int *outSize, uint32_t nTiles) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nTiles) return;
    av1g_cdfs *cd = scratch + t;
    av1g_cdfs_init(cd, tQ[t]);
    const uint64_t n = tEnd[t] - tBegin[t];
    const uint32_t cap = (uint32_t)(capOff[t + 1] - capOff[t]);
    av1g_ec ec;
    av1g_ec_init(&ec, precarry + capOff[t], cap);
    av1g_code_symbols(syms + tBegin[t], (uint32_t)n, cd, &ec);
    outSize[t] = av1g_ec_done(&ec, outBytes + capOff[t]);
}

// Same, with the tile's CDF state in shared memory (AV1G_ENT_TPB tiles per CUDA block).
#define AV1G_ENT_TPB 4
__global__ void k_entropy_smem(const uint32_t *syms, const uint64_t *tBegin, const uint64_t *tEnd, const int *tQ,
                               const uint64_t *capOff, uint16_t *precarry, uint8_t *outBytes, int *outSize,
                               uint32_t nTiles) {
    extern __shared__ av1g_cdfs sh[];
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nTiles) return;
    av1g_cdfs *cd = sh + threadIdx.x;
    av1g_cdfs_init(cd, tQ[t]);
    const uint64_t n = tEnd[t] - tBegin[t];
    const uint32_t cap = (uint32_t)(capOff[t + 1] - capOff[t]);
    av1g_ec ec;
    av1g_ec_init(&ec, precarry + capOff[t], cap);
    av1g_code_symbols(syms + tBegin[t], (uint32_t)n, cd, &ec);
    outSize[t] = av1g_ec_done(&ec, outBytes + capOff[t]);
}

// One warp per tile: lane 0 runs od_ec, the N-1 CDF entries are adapted by N-1
// lanes in parallel (same arithmetic as av1g_update_cdf).  CDF state in shared memory.
__global__ void k_entropy_warp(const uint32_t *syms, const uint64_t *tBegin, const uint64_t *tEnd, const int *tQ,
                               const uint64_t *capOff, uint16_t *precarry, uint8_t *outBytes, int *outSize,
                               uint32_t nTiles) {
    extern __shared__ av1g_cdfs sh[];
    const uint32_t t = blockIdx.x;
    const int lane = threadIdx.x;
    if (t >= nTiles) return;
    av1g_cdfs *cd = sh;
    uint16_t *base = (uint16_t *)cd;
    {   // cooperative init: copy the defaults (prepared by lane 0 into shared memory, then spread)
        if (lane == 0) av1g_cdfs_init(cd, tQ[t]);
        __syncwarp();
    }
    const uint64_t b = tBegin[t], n = tEnd[t] - b;
    const uint32_t cap = (uint32_t)(capOff[t + 1] - capOff[t]);
    av1g_ec ec;
    if (lane == 0) av1g_ec_init(&ec, precarry + capOff[t], cap);
    for (uint64_t i = 0; i < n; i++) {
        const uint32_t w = syms[b + i];
        const uint32_t kind = w >> 24, s = (w >> 16) & 15u, N = ((w >> 20) & 15u) + 1, off = w & 0xFFFFu;
        uint16_t *c = base + off;
        if (kind == AV1G_SYM_ADAPT) {
            const uint32_t cnt = c[N];
            const uint32_t mine = lane < (int)N - 1 ? c[lane] : 0;
            if (lane == 0) av1g_ec_encode_cdf(&ec, (int)s, c, (int)N);
            __syncwarp();
            if (lane < (int)N - 1) {
                const int rate = 3 + (cnt > 15) + (cnt > 31) + (N >= 4 ? 2 : 1);
                c[lane] = (uint16_t)((uint32_t)lane < s ? mine - (mine >> rate) : mine + ((32768u - mine) >> rate));
            }
            if (lane == 31) c[N] = (uint16_t)(cnt + (cnt < 32));
            __syncwarp();
        } else if (lane == 0) {
            if (kind == AV1G_SYM_BOOL) av1g_ec_bool(&ec, (int)s);
            else {
                const uint32_t psum = av1g_split_or_psum(c, kind == AV1G_SYM_SPLIT_OR_VERT);
                uint16_t tmp[2] = {(uint16_t)(32768u - psum), 32768u};
                av1g_ec_encode_cdf(&ec, (int)s, tmp, 2);
            }
        }
    }
    if (lane == 0) outSize[t] = av1g_ec_done(&ec, outBytes + capOff[t]);
}

// ---------------------------------------------------------------------------
// prep (all on the device; the host only lays out frame/tile tables from fp)
template <class T, int K>
struct Chunk {
    T v[K];
    uint32_t n, first;
};
template <class T, int K>
__global__ void k_put(const Chunk<T, K> c, T *dst) {
    for (uint32_t i = threadIdx.x; i < c.n; i += blockDim.x) dst[c.first + i] = c.v[i];
}

__global__ void k_cdf_defaults(av1g_cdfs *img) {  // one default CDF image per coefficient q context
    const int q = blockIdx.x;
    if (threadIdx.x == 0) av1g_cdfs_init(img + q, q == 0 ? 20 : q == 1 ? 60 : q == 2 ? 120 : 255);
}

__device__ uint32_t frame_of_block(const FrameG *F, uint32_t nF, uint32_t b) {
    uint32_t lo = 0, hi = nF;
    while (hi - lo > 1) {
        const uint32_t mid = (lo + hi) >> 1;
        if (F[mid].blockBase <= b) lo = mid; else hi = mid;
    }
    return lo;
}
__device__ uint32_t tile_of_sb(const TileG *T, uint32_t nT, uint32_t s) {
    uint32_t lo = 0, hi = nT;
    while (hi - lo > 1) {
        const uint32_t mid = (lo + hi) >> 1;
        if (T[mid].sbBase <= s) lo = mid; else hi = mid;
    }
    return lo;
}
// frame-relative decode-order SB index of a block + its tile (global)
__device__ uint32_t sb_key(const FrameG &g, const TileG *T, const av1g_block &k, uint32_t &tile) {
    const int sbR = k.mi_row >> 4, sbC = k.mi_col >> 4;
    const int tr = min(sbR / g.tileHsb, g.tileRows - 1), tc = min(sbC / g.tileWsb, g.tileCols - 1);
    tile = g.tileBase + (uint32_t)(tr * g.tileCols + tc);
    const TileG &tl = T[tile];
    return tl.sbBase - g.sbBase + (uint32_t)((sbR - tl.sbRow0) * tl.sbCols + (sbC - tl.sbCol0));
}

__global__ void k_blocks(const FrameG *F, uint32_t nF, const TileG *T, const av1g_block *blocks, const uint32_t *blockTxb,
                         uint32_t nB, uint32_t nT, uint32_t *blockFrame, uint32_t *blockTile, uint32_t *blockTxbG,
                         uint32_t *txbBlock, uint32_t *sbFirst, DevStatus *st) {
    const uint32_t b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nB) return;
    const uint32_t f = frame_of_block(F, nF, b);
    const FrameG &g = F[f];
    const av1g_block k = blocks[b];
    const uint32_t lb = b - g.blockBase;
    if (k.mi_row >= g.MiRows || k.mi_col >= g.MiCols) {
        atomicOr(&st->error, AV1G_GPU_ERR_ORDER);
        blockFrame[b] = f; blockTile[b] = g.tileBase; blockTxbG[b] = g.txbBase;
        return;
    }
    uint32_t tile;
    const uint32_t key = sb_key(g, T, k, tile);
    blockFrame[b] = f;
    blockTile[b] = tile;
    const uint32_t t0 = blockTxb[b], t1 = lb + 1 < g.nBlocks ? blockTxb[b + 1] : g.nTxb;
    bool bad = t1 < t0 || t1 > g.nTxb || (lb == 0 && t0 != 0);
    blockTxbG[b] = g.txbBase + min(t0, g.nTxb);
    if (!bad)
        for (uint32_t t = t0; t < t1; t++) txbBlock[g.txbBase + t] = b;
    if (lb == 0) {
        if (key != 0) bad = true;
        else sbFirst[g.sbBase] = b;
    } else {
        uint32_t pt;
        const uint32_t pkey = sb_key(g, T, blocks[b - 1], pt);
        if (key == pkey + 1) sbFirst[g.sbBase + key] = b;
        else if (key != pkey) bad = true;
    }
    if (lb + 1 == g.nBlocks && key + 1 != g.nSb) bad = true;
    if (bad) atomicOr(&st->error, AV1G_GPU_ERR_ORDER);
    if (b + 1 == nB) blockTxbG[nB] = nT;
}

__global__ void k_sbs(const FrameG *F, const TileG *T, uint32_t nTiles, uint32_t nSb, const uint32_t *sbFirst, SbG *sbs,
                      DevStatus *st) {
    const uint32_t s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= nSb) return;
    const uint32_t tile = tile_of_sb(T, nTiles, s);
    const TileG &tl = T[tile];
    const FrameG &g = F[tl.frame];
    const uint32_t local = s - tl.sbBase;
    SbG o;
    o.frame = tl.frame;
    o.tile = tile;
    o.r = (tl.sbRow0 + (int)(local / tl.sbCols)) * 16;
    o.c = (tl.sbCol0 + (int)(local % tl.sbCols)) * 16;
    const uint32_t fEnd = g.blockBase + g.nBlocks;
    o.firstBlock = sbFirst[s];
    o.endBlock = s + 1 < g.sbBase + g.nSb ? sbFirst[s + 1] : fEnd;
    if (o.firstBlock == 0xFFFFFFFFu || o.endBlock == 0xFFFFFFFFu || o.firstBlock >= o.endBlock || o.endBlock > fEnd) {
        atomicOr(&st->error, AV1G_GPU_ERR_ORDER);
        o.firstBlock = o.endBlock = g.blockBase;
    }
    o.qBefore = g.qidx;
    sbs[s] = o;
}

// one warp per tile: tile block range + CurrentQIndex before every SB (reset to base_q_idx at
// the tile start, updated by the first block of each SB unless it is a skipped 64x64)
__global__ void k_tiles(const FrameG *F, TileG *T, SbG *sbs, const av1g_block *blocks) {
    const uint32_t t = blockIdx.x;
    const int lane = threadIdx.x;
    const TileG tl = T[t];
    const FrameG &g = F[tl.frame];
    if (lane == 0) {
        T[t].firstBlock = sbs[tl.sbBase].firstBlock;
        T[t].endBlock = sbs[tl.sbBase + tl.nSb - 1].endBlock;
    }
    if (!g.dq) return;
    int carry = g.qidx;
    for (uint32_t base = 0; base < tl.nSb; base += 32) {
        const uint32_t i = base + lane;
        bool coded = false;
        int val = 0;
        if (i < tl.nSb) {
            const SbG s = sbs[tl.sbBase + i];
            if (s.firstBlock < s.endBlock) {
                const av1g_block fb = blocks[s.firstBlock];
                coded = !(fb.bsize == AV1G_BLOCK_64X64 && fb.skip);
                val = fb.sb_qindex;
            }
        }
        const uint32_t mask = __ballot_sync(0xFFFFFFFFu, coded);
        const uint32_t lower = mask & ((1u << lane) - 1u);
        const int v = __shfl_sync(0xFFFFFFFFu, val, lower ? 31 - __clz((int)lower) : 0);
        if (i < tl.nSb) sbs[tl.sbBase + i].qBefore = lower ? v : carry;
        if (mask) carry = __shfl_sync(0xFFFFFFFFu, val, 31 - __clz((int)mask));
    }
}

// per tile: symbol range, q, payload capacity (symbols / 2 + 256 bytes)
__global__ void k_tilerange(const TileG *T, uint32_t nTiles, const FrameG *F, const uint32_t *blockTxbG, const uint32_t *off,
                            uint64_t nItems, uint32_t nB, uint64_t symCap, uint64_t *tb, uint64_t *te, int *tq,
                            uint64_t *capArr, uint64_t *sizeArr, DevStatus *st) {
    const uint32_t t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= nTiles) return;
    const uint64_t total = off[nItems];
    const bool ovf = total > symCap;
    auto itemOf = [&](uint32_t b) -> uint64_t { return b < nB ? off[(uint64_t)b + blockTxbG[b]] : total; };
    uint64_t b = itemOf(T[t].firstBlock), e = itemOf(T[t].endBlock);
    if (ovf || e < b) b = e = 0;
    tb[t] = b;
    te[t] = e;
    tq[t] = F[T[t].frame].qidx;
    capArr[t] = (e - b) / 2 + 256;
    if (t == 0) {
        capArr[nTiles] = 0;
        sizeArr[nTiles] = 0;
        st->symbols = total;
        st->symbolCapacity = symCap;
        if (ovf) atomicOr(&st->error, AV1G_GPU_ERR_SYMBOLS);
    }
}

// ---------------------------------------------------------------------------
// windowed warp entropy coder
//
// Range coder of the consumer thread: libaom's current od_ec formulation (aom_dsp/entenc.c with
// the 64-bit window: up to 7 bytes flushed at once when 40+ bits are pending, carries propagated
// backwards into the output at flush time), fed with precomputed (fl, fh) words.  libaom documents
// it as bit-identical to the 32-bit precarry encoder (av1g_ec in entenc.h, used by the CPU writer).
struct Rc64 {
    uint64_t low;
    uint32_t rng, offs, cap;
    int cnt, ovf;
    uint8_t *out;
    __device__ void init(uint8_t *o, uint32_t c) { low = 0; rng = 0x8000; cnt = -9; offs = 0; cap = c; ovf = 0; out = o; }
    __device__ void carry_bwd(uint32_t k) {  // propagate_carry_bwd(out, k)
        for (;;) {
            const uint32_t x = out[k] + 1u;
            out[k] = (uint8_t)x;
            if (x <= 0xFFu || k == 0) break;
            k--;
        }
    }
    __device__ void flush(int &c, int s) {  // the s >= 40 branch of od_ec_enc_normalize
        const int nb = (s >> 3) + 1;
        c += 24 - (nb << 3);
        uint64_t output = low >> c;
        low &= ((uint64_t)1 << c) - 1;
        const uint64_t mask = (uint64_t)1 << (nb << 3);
        const bool carry = (output & mask) != 0;
        output &= mask - 1;
        if (offs + (uint32_t)nb > cap) { ovf = 1; offs = 0; }
        for (int i = 0; i < nb; i++) out[offs + i] = (uint8_t)(output >> ((nb - 1 - i) << 3));
        if (carry && offs > 0) carry_bwd(offs - 1);
        offs += (uint32_t)nb;
    }
    __device__ int done() {
        const uint64_t m = 0x3FFF;
        uint64_t e = ((low + m) & ~m) | (m + 1);
        int c = cnt, s = 10 + c;
        if (s > 0) {
            uint64_t n = ((uint64_t)1 << (c + 16)) - 1;
            do {
                if (offs >= cap) { ovf = 1; break; }
                const uint32_t val = (uint32_t)(e >> (c + 16)) & 0xFFFFu;
                out[offs] = (uint8_t)val;
                if ((val & 0x100u) && offs > 0) carry_bwd(offs - 1);
                offs++;
                e &= n;
                s -= 8;
                c -= 8;
                n >>= 8;
            } while (s > 0);
        }
        return ovf ? -1 : (int)offs;
    }
};

#ifdef AV1G_ENTROPY_PROFILE
// per tile: [0] total cycles, [1] phase A cycles, [2] phase B cycles, [3] max group size summed over
// windows, [4] windows, [5] bytes emitted, [6] carries, [7] carry propagation steps
__device__ unsigned long long g_prof[4096][8];
#define PROF(...) __VA_ARGS__
#else
#define PROF(...)
#endif

__device__ __forceinline__ uint32_t coded_word(uint32_t fl, uint32_t fh, uint32_t nms) {
    return (fl < 32768u ? 0x80000000u : 0u) | ((fl >> 6) << 21) | ((fh >> 6) << 11) | nms;
}

// The same encoder in the 32-bit precarry formulation (av1g_ec) with the most recent byte held in a
// register (a carry normally lands there) - kept for A/B measurements (g_entropy_mode 4).
struct Rc32 {
    uint32_t low, rng, offs, cap;
    int cnt, ovf, pend;
    uint8_t *out;
    __device__ void init(uint8_t *o, uint32_t c) { low = 0; rng = 0x8000; cnt = -9; offs = 0; cap = c; ovf = 0; pend = -1; out = o; }
    __device__ __forceinline__ void emit(uint32_t v) {  // v = byte + carry (<= 0x1FF)
        if (v > 0xFFu && pend >= 0) {
            pend++;
            if (pend > 0xFF) {
                pend = 0;
                uint32_t k = offs;
                while (k > 0) {
                    k--;
                    const uint32_t x = out[k] + 1u;
                    out[k] = (uint8_t)x;
                    if (x <= 0xFFu) break;
                }
            }
        }
        if (pend >= 0) {
            if (offs < cap) out[offs] = (uint8_t)pend;
            else ovf = 1;
            offs++;
        }
        pend = (int)(v & 0xFFu);
    }
    __device__ int done() {
        int c = cnt, s = 10 + c;
        const uint32_t m = 0x3FFF;
        uint32_t en = ((low + m) & ~m) | (m + 1);
        if (s > 0) {
            uint32_t n = (1u << (c + 16)) - 1;
            do {
                emit(en >> (c + 16));
                en &= n;
                s -= 8;
                c -= 8;
                n >>= 8;
            } while (s > 0);
        }
        if (pend >= 0) {
            if (offs < cap) out[offs] = (uint8_t)pend;
            else ovf = 1;
            offs++;
        }
        return ovf ? -1 : (int)offs;
    }
};

// od_ec_encode_q15 + normalize; u = r when fl >= 32768 (then low is unchanged and r -= v).
// cw: bit 31 = (fl < 32768), bits 21..30 fl >> 6, bits 11..20 fh >> 6, bits 0..4 nsyms - 1 - s.
__device__ __forceinline__ void rc_code(Rc64 &rc, uint32_t cw PROF(, unsigned long long *pc)) {
    uint32_t r = rc.rng;
    const uint32_t r8 = r >> 8, nms4 = 4u * (cw & 31u);
    const uint32_t v = ((r8 * ((cw >> 11) & 1023u)) >> 1) + nms4;
    const uint32_t uq = ((r8 * ((cw >> 21) & 1023u)) >> 1) + nms4 + 4u;
    const uint32_t u = (cw >> 31) ? uq : r;
    uint64_t l = rc.low + (r - u);
    r = u - v;
    const int d = __clz((int)r) - 16;
    int c = rc.cnt;
    int s = c + d;
    if (s >= 40) {
        PROF(pc[0]++;)
        rc.low = l;
        rc.flush(c, s);
        l = rc.low;
        s = c + d - 24;
    }
    rc.low = l << d;
    rc.rng = r << d;
    rc.cnt = s;
}
__device__ __forceinline__ void rc_code(Rc32 &rc, uint32_t cw PROF(, unsigned long long *pc)) {
    uint32_t l = rc.low, r = rc.rng;
    const uint32_t r8 = r >> 8, nms4 = 4u * (cw & 31u);
    const uint32_t v = ((r8 * ((cw >> 11) & 1023u)) >> 1) + nms4;
    const uint32_t uq = ((r8 * ((cw >> 21) & 1023u)) >> 1) + nms4 + 4u;
    const uint32_t u = (cw >> 31) ? uq : r;
    l += r - u;
    r = u - v;
    const int d = __clz((int)r) - 16;
    int c = rc.cnt, s = c + d;
    if (s >= 0) {
        c += 16;
        uint32_t m = (1u << c) - 1;
        if (s >= 8) {
            PROF(pc[0]++;)
            rc.emit(l >> c);
            l &= m;
            c -= 8;
            m >>= 8;
        }
        PROF(pc[0]++;)
        rc.emit(l >> c);
        s = c + d - 24;
        l &= m;
    }
    rc.low = l << d;
    rc.rng = r << d;
    rc.cnt = s;
}

// value of cr[idx] for a runtime idx without local memory
__device__ __forceinline__ uint32_t sel16(const uint32_t (&cr)[16], uint32_t idx) {
    uint32_t v = 0;
#pragma unroll
    for (int i = 0; i < 16; i++) v = (uint32_t)i == idx ? cr[i] : v;
    return v;
}

// Two warps per tile: warp 0 computes the coded words of 32-symbol windows (phase A, CDF
// adaptation) into a shared ring, warp 1 lane 0 range-codes them (phase B), so the per-window
// cost is max(A, B) instead of A + B.
#define AV1G_WIN_RING 8
struct WinShared {
    uint32_t ring[AV1G_WIN_RING][32];  // coded words
    uint32_t sw[32];                   // producer: per lane s | kind << 8
    int prod, cons;                    // windows produced / consumed
};
constexpr size_t kWinSmem = sizeof(av1g_cdfs) + sizeof(WinShared);

__device__ __forceinline__ int vload(const int *p) { return *(const volatile int *)p; }
__device__ __forceinline__ void vstore(int *p, int v) { *(volatile int *)p = v; }

template <class Coder>
__global__ void __launch_bounds__(64) k_entropy_win(const uint32_t *syms, const uint64_t *tb, const uint64_t *te, const int *tq,
                                                     const uint64_t *capOff, const av1g_cdfs *cdfDef, uint8_t *out,
                                                     uint64_t *sizeArr, DevStatus *st) {
    extern __shared__ uint4 sh4[];
    uint16_t *cdf = (uint16_t *)sh4;
    WinShared &W = *(WinShared *)(sh4 + sizeof(av1g_cdfs) / 16);
    const uint32_t t = blockIdx.x;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    {
        const uint4 *src = (const uint4 *)(cdfDef + av1g_coeff_cdf_q_ctx(tq[t]));
        for (int i = threadIdx.x; i < (int)(sizeof(av1g_cdfs) / 16); i += 64) sh4[i] = src[i];
        if (threadIdx.x == 0) { W.prod = 0; W.cons = 0; }
        __syncthreads();
    }
    const uint64_t b = tb[t], e = te[t];
    const uint32_t nWin = (uint32_t)((e - b + 31) / 32);
    PROF(unsigned long long pT0 = clock64(), pA = 0, pB = 0, pG = 0, pW = 0, pc[2] = {0, 0};)
    if (warp == 0) {
        // ---------------- phase A (producer) ----------------
        uint32_t next = b + lane < e ? syms[b + lane] : 0u;
        for (uint32_t w = 0; w < nWin; w++) {
            const uint64_t w0 = b + 32ull * w;
            const uint32_t nv = (uint32_t)min((uint64_t)32, e - w0);
            const uint32_t word = next;
            next = w0 + 32 + lane < e ? syms[w0 + 32 + lane] : 0u;  // prefetch the next window
            if (lane == 0)
                while ((int)w - vload(&W.cons) >= AV1G_WIN_RING) __nanosleep(32);
            __syncwarp();
            PROF(const unsigned long long pa0 = clock64();)
            uint32_t *sc = W.ring[w % AV1G_WIN_RING];
            const uint32_t kind = (uint32_t)lane < nv ? word >> 24 : AV1G_SYM_BOOL;
            const uint32_t s = (word >> 16) & 15u, off = word & 0xFFFFu;
            // symbols sharing a CDF form a group; its first lane applies the group's updates in
            // stream order with the CDF in registers
            const uint32_t key = kind == AV1G_SYM_BOOL ? 0x10000u | (uint32_t)lane : off;
            const uint32_t peers = __match_any_sync(0xFFFFFFFFu, key);
            W.sw[lane] = s | (kind << 8);
            if (kind == AV1G_SYM_BOOL) sc[lane] = coded_word(s ? 16384u : 32768u, s ? 0u : 16384u, 1u - (s & 1u));
            __syncwarp();
            PROF(pG += __reduce_max_sync(0xFFFFFFFFu, (uint32_t)__popc(peers)); pW++;)
            if (kind != AV1G_SYM_BOOL && (peers & ((1u << lane) - 1u)) == 0) {
                // alphabet of the CDF: a split_or_* word carries the bool's N; its partition CDF has 10
                const uint32_t N = kind == AV1G_SYM_ADAPT ? ((word >> 20) & 15u) + 1 : 10u;
                uint16_t *c = cdf + off;
                uint32_t cr[16];
#pragma unroll
                for (int i = 0; i < 16; i++) cr[i] = (uint32_t)i < N ? c[i] : 0u;
                uint32_t cnt = c[N];
                uint32_t m = peers;
                while (m) {
                    const int j = __ffs((int)m) - 1;
                    m &= m - 1;
                    const uint32_t x = W.sw[j], sj = x & 15u;
                    if ((x >> 8) == AV1G_SYM_ADAPT) {
                        sc[j] = coded_word(sj ? 32768u - sel16(cr, sj - 1) : 32768u, 32768u - sel16(cr, sj), N - 1 - sj);
                        const int rate = 3 + (cnt > 15) + (cnt > 31) + (N >= 4 ? 2 : 1);
                        // all 16 entries unconditionally (straight-line code): cr[N-1] == 32768 stays
                        // 32768, entries >= N are never read nor written back
#pragma unroll
                        for (int i = 0; i < 16; i++) {
                            const uint32_t y = cr[i];
                            cr[i] = (uint32_t)i < sj ? y - (y >> rate) : y + ((32768u - y) >> rate);
                        }
                        cnt += cnt < 32;
                    } else {  // split_or_horz / split_or_vert: read-only use of the partition CDF
#define P_(k) (cr[k] - ((k) > 0 ? cr[(k) > 0 ? (k) - 1 : 0] : 0u))
                        const uint32_t psum = (x >> 8) == AV1G_SYM_SPLIT_OR_HORZ
                            ? P_(AV1G_PARTITION_VERT) + P_(AV1G_PARTITION_SPLIT) + P_(AV1G_PARTITION_HORZ_A) +
                                  P_(AV1G_PARTITION_VERT_A) + P_(AV1G_PARTITION_VERT_B) + P_(AV1G_PARTITION_VERT_4)
                            : P_(AV1G_PARTITION_HORZ) + P_(AV1G_PARTITION_SPLIT) + P_(AV1G_PARTITION_HORZ_A) +
                                  P_(AV1G_PARTITION_HORZ_B) + P_(AV1G_PARTITION_VERT_A) + P_(AV1G_PARTITION_HORZ_4);
#undef P_
                        sc[j] = coded_word(sj ? psum : 32768u, sj ? 0u : psum, 1u - sj);
                    }
                }
#pragma unroll
                for (int i = 0; i < 15; i++)
                    if ((uint32_t)i + 1 < N) c[i] = (uint16_t)cr[i];
                c[N] = (uint16_t)cnt;
            }
            __syncwarp();
            PROF(pA += clock64() - pa0;)
            if (lane == 0) {
                __threadfence_block();
                vstore(&W.prod, (int)w + 1);
            }
        }
    } else if (lane == 0) {
        // ---------------- phase B (consumer, one thread) ----------------
        Coder rc;
        rc.init(out + capOff[t], (uint32_t)(capOff[t + 1] - capOff[t]));
        for (uint32_t w = 0; w < nWin; w++) {
            while (vload(&W.prod) <= (int)w) {}
            __threadfence_block();
            PROF(const unsigned long long pb0 = clock64();)
            const uint32_t *sc = W.ring[w % AV1G_WIN_RING];
            const uint32_t nv = (uint32_t)min((uint64_t)32, (uint64_t)(e - (b + 32ull * w)));
            uint32_t cw = sc[0];
            for (uint32_t j = 0; j < nv; j++) {
                const uint32_t cur = cw;
                cw = sc[j + 1 < 32 ? j + 1 : 31];
                rc_code(rc, cur PROF(, pc));
            }
            __threadfence_block();
            vstore(&W.cons, (int)w + 1);
            PROF(pB += clock64() - pb0;)
        }
        const int n = rc.done();
        PROF(if (t < 4096) { g_prof[t][0] = clock64() - pT0; g_prof[t][2] = pB; g_prof[t][5] = pc[0]; g_prof[t][6] = pc[1]; })
        sizeArr[t] = n < 0 ? 0 : (uint64_t)n;
        if (n < 0) atomicOr(&st->error, AV1G_GPU_ERR_OUTPUT);
    }
#ifdef AV1G_ENTROPY_PROFILE
    if (warp == 0 && lane == 0 && t < 4096) { g_prof[t][1] = pA; g_prof[t][3] = pG; g_prof[t][4] = pW; }
#endif
}

// tile payloads (at capOff) -> packed (at tileOff); one CTA per tile
__global__ void k_pack(const uint8_t *staging, const uint64_t *capOff, const uint64_t *sizeArr, const uint64_t *tileOff,
                       uint8_t *packed, uint32_t nTiles, DevStatus *st) {
    const uint32_t t = blockIdx.x;
    const uint8_t *src = staging + capOff[t];
    uint8_t *dst = packed + tileOff[t];
    const uint64_t n = sizeArr[t];
    for (uint64_t i = threadIdx.x; i < n; i += blockDim.x) dst[i] = src[i];
    if (t == 0 && threadIdx.x == 0) st->bytes = tileOff[nTiles];
}

int g_entropy_mode = 3;  // 3 two-warp windowed coder (64-bit od_ec window), 4 same with the 32-bit precarry
                         // coder; 2 M4 warp/tile; 1 shared 4 tiles/warp; 0 global-memory CDFs

static void launch_entropy_m4(bool smem, const uint32_t *syms, const uint64_t *b, const uint64_t *e, const int *q,
                              const uint64_t *cap, av1g_cdfs *scr, uint16_t *pre, uint8_t *out, int *sz, uint32_t n) {
    if (smem && g_entropy_mode == 2)
        k_entropy_warp<<<n, 32, sizeof(av1g_cdfs)>>>(syms, b, e, q, cap, pre, out, sz, n);
    else if (smem)
        k_entropy_smem<<<(n + AV1G_ENT_TPB - 1) / AV1G_ENT_TPB, AV1G_ENT_TPB, AV1G_ENT_TPB * sizeof(av1g_cdfs)>>>(
            syms, b, e, q, cap, pre, out, sz, n);
    else
        k_entropy<<<(n + 31) / 32, 32>>>(syms, b, e, q, cap, scr, pre, out, sz, n);
}

bool g_entropy_smem = true;

// ---------------------------------------------------------------------------
// host: scratch plan (shared by the size query and the call)
// ---------------------------------------------------------------------------
namespace {

constexpr int kFrameChunk = 128, kTileChunk = 512;
static_assert(sizeof(Chunk<FrameG, kFrameChunk>) < 32000, "kernel parameter limit");
static_assert(sizeof(Chunk<TileG, kTileChunk>) < 32000, "kernel parameter limit");

size_t al(size_t x) { return (x + 255) & ~(size_t)255; }

struct Plan {
    std::vector<FrameG> F;
    std::vector<TileG> T;
    std::vector<uint32_t> frameTile0;
    uint32_t nF = 0, nB = 0, nT = 0, nSb = 0, nTiles = 0;
    uint64_t nItems = 0, miTotal = 0, ctxAll = 0, symCap = 0, capBytes = 0;
    size_t cubItems = 0, cubTiles = 0;
    // scratch offsets
    size_t oF, oT, oSb, oBFrame, oBTile, oBTxbG, oTxbBlock, oSbFirst, oMap[4], oCtxL, oCtxD, oEob, oType, oCount, oOff, oCub,
        oSyms, oTb, oTe, oTq, oCapArr, oCapOff, oSizes, oStaging, oPacked, oResult, oCdfDef, total;
};

int build_plan(const DevBatch &b, Plan &P, std::string &err) {
    P = Plan();
    if (!b.frames || !b.nFrames) { err = "no frames"; return -1; }
    P.nF = b.nFrames;
    P.F.resize(P.nF);
    P.frameTile0.assign(P.nF + 1, 0);
    const uint32_t b0 = b.frames[0].blockBase, t0 = b.frames[0].txbBase;
    uint64_t miTotal = 0, ctxTotal[3] = {0, 0, 0}, pixels = 0, llSamples = 0;
    for (uint32_t f = 0; f < P.nF; f++) {
        const DevFrame &d = b.frames[f];
        const av1g_frame_params &fp = d.fp;
        if (d.blockBase != b0 + P.nB || d.txbBase != t0 + P.nT) {
            err = "frames must be stored in order and contiguously in blocks/blockTxb and txb (frame " + std::to_string(f) + ")";
            return -1;
        }
        if (!d.nBlocks || !d.nTxb) { err = "frame " + std::to_string(f) + " has no blocks / tx blocks"; return -1; }
        FrameG &g = P.F[f];
        memset(&g, 0, sizeof g);
        g.W = fp.width; g.H = fp.height;
        g.MiCols = av1g_mi_cols(g.W); g.MiRows = av1g_mi_rows(g.H);
        g.mono = fp.monochrome; g.qidx = fp.base_q_idx; g.reduced = fp.reduced_tx_set;
        g.ssx = av1g_ssx(&fp); g.ssy = av1g_ssy(&fp);
        g.txsel = fp.tx_mode_select; g.sct = fp.allow_screen_content_tools;
        g.dq = fp.delta_q_present; g.dqRes = fp.delta_q_res; g.fi = fp.enable_filter_intra;
        g.cdefBits = fp.enable_cdef ? fp.cdef_bits : -1;
        if (av1g_lossless(&fp)) llSamples += (uint64_t)fp.width * fp.height * (fp.monochrome ? 2 : g.ssx ? 3 : 6) / 2;
        g.miBase = miTotal;
        miTotal += (uint64_t)g.MiCols * g.MiRows;
        for (int p = 0; p < 3; p++) {
            g.ctxStride[p] = g.MiCols >> (p ? g.ssx : 0);
            g.ctxRows[p] = g.MiRows >> (p ? g.ssy : 0);
            g.ctxBase[p] = ctxTotal[p];
            if (p == 0 || !g.mono) ctxTotal[p] += (uint64_t)g.ctxStride[p] * g.ctxRows[p];
        }
        g.blockBase = P.nB; g.nBlocks = d.nBlocks; g.txbBase = P.nT; g.nTxb = d.nTxb; g.coeffBase = d.coeffBase;
        P.nB += d.nBlocks;
        P.nT += d.nTxb;
        pixels += (uint64_t)fp.width * fp.height * (g.ssx ? 1 : 2);   // 4:4:4: twice the samples of 4:2:0
        av1g::TileInfo ti;
        if (!av1g::compute_tiles(fp.width, fp.height, fp.tile_cols_log2, fp.tile_rows_log2, ti)) {
            err = "bad tiling in frame " + std::to_string(f);
            return -1;
        }
        const int sbCols = (g.MiCols + 15) >> 4, sbRows = (g.MiRows + 15) >> 4;
        g.tileCols = (int)ti.cols; g.tileRows = (int)ti.rows;
        g.tileWsb = ti.cols > 1 ? (int)(ti.miColStarts[1] >> 4) : sbCols;
        g.tileHsb = ti.rows > 1 ? (int)(ti.miRowStarts[1] >> 4) : sbRows;
        g.sbBase = P.nSb; g.tileBase = (uint32_t)P.T.size(); g.nTiles = ti.cols * ti.rows;
        P.frameTile0[f] = g.tileBase;
        for (uint32_t tr = 0; tr < ti.rows; tr++)
            for (uint32_t tc = 0; tc < ti.cols; tc++) {
                TileG tg;
                memset(&tg, 0, sizeof tg);
                tg.frame = f;
                tg.miRowStart = ti.miRowStarts[tr]; tg.miRowEnd = ti.miRowStarts[tr + 1];
                tg.miColStart = ti.miColStarts[tc]; tg.miColEnd = ti.miColStarts[tc + 1];
                if ((int)ti.miColStarts[tc] != (int)tc * g.tileWsb * 16 || (int)ti.miRowStarts[tr] != (int)tr * g.tileHsb * 16) {
                    err = "non-uniform tiling";
                    return -1;
                }
                tg.sbRow0 = tg.miRowStart >> 4; tg.sbCol0 = tg.miColStart >> 4;
                tg.sbCols = ((tg.miColEnd + 15) >> 4) - tg.sbCol0;
                tg.nSb = (uint32_t)(tg.sbCols * (((tg.miRowEnd + 15) >> 4) - tg.sbRow0));
                tg.sbBase = P.nSb;
                P.nSb += tg.nSb;
                P.T.push_back(tg);
            }
        g.nSb = P.nSb - g.sbBase;
    }
    for (auto &g : P.F) { g.ctxBase[1] += ctxTotal[0]; g.ctxBase[2] += ctxTotal[0] + ctxTotal[1]; }
    P.nTiles = (uint32_t)P.T.size();
    P.frameTile0[P.nF] = P.nTiles;
    P.miTotal = miTotal;
    P.ctxAll = ctxTotal[0] + ctxTotal[1] + ctxTotal[2];
    P.nItems = (uint64_t)P.nB + P.nT;
    if (P.nItems + 1 > 0x7FFFFFFFull) { err = "batch too large (items)"; return -1; }
    // default bound; lossless frames code every sample's WHT coefficient (sign, base / range levels, Golomb bits)
    P.symCap = b.symbolCapacity ? b.symbolCapacity : pixels * 3 / 4 + 16ull * P.nB + 8ull * P.nT + 6ull * llSamples;
    P.capBytes = P.symCap / 2 + 256ull * P.nTiles;
    size_t t1 = 0, t2 = 0;
    cub::DeviceScan::ExclusiveSum(nullptr, t1, (const uint32_t *)nullptr, (uint32_t *)nullptr, (int)(P.nItems + 1));
    cub::DeviceScan::ExclusiveSum(nullptr, t2, (const uint64_t *)nullptr, (uint64_t *)nullptr, (int)(P.nTiles + 1));
    P.cubItems = t1;
    P.cubTiles = t2;
    size_t o = 0;
    auto take = [&](size_t bytes) { const size_t at = o; o = al(o + bytes); return at; };
    P.oF = take(sizeof(FrameG) * P.nF);
    P.oT = take(sizeof(TileG) * P.nTiles);
    P.oSb = take(sizeof(SbG) * P.nSb);
    P.oBFrame = take(4ull * P.nB);
    P.oBTile = take(4ull * P.nB);
    P.oBTxbG = take(4ull * (P.nB + 1));
    P.oTxbBlock = take(4ull * P.nT);
    P.oSbFirst = take(4ull * P.nSb);
    for (int i = 0; i < 4; i++) P.oMap[i] = take(P.miTotal);
    P.oCtxL = take(P.ctxAll);
    P.oCtxD = take(P.ctxAll);
    P.oEob = take(2ull * P.nT);
    P.oType = take(P.nT);
    P.oCount = take(4ull * (P.nItems + 1));
    P.oOff = take(4ull * (P.nItems + 1));
    P.oCub = take(std::max(t1, t2));
    P.oSyms = take(4ull * P.symCap);
    P.oTb = take(8ull * P.nTiles);
    P.oTe = take(8ull * P.nTiles);
    P.oTq = take(4ull * P.nTiles);
    P.oCapArr = take(8ull * (P.nTiles + 1));
    P.oCapOff = take(8ull * (P.nTiles + 1));
    P.oSizes = take(8ull * (P.nTiles + 1));
    P.oStaging = take(P.capBytes);
    P.oPacked = take(P.capBytes);
    P.oResult = take(sizeof(DevStatus) + 8ull * (P.nTiles + 1));
    P.oCdfDef = take(4 * sizeof(av1g_cdfs));
    P.total = o;
    return 0;
}

}  // namespace

size_t run_batch_device_scratch_bytes(const DevBatch &b) {
    Plan P;
    std::string err;
    return build_plan(b, P, err) ? 0 : P.total;
}

int run_batch_device(const DevBatch &b, cudaStream_t s, void *scratch, size_t scratchBytes, DevResult &out, std::string &err,
                     const DevEvents *ev) {
    Plan P;
    if (build_plan(b, P, err)) return -1;
    if (!scratch || scratchBytes < P.total) {
        err = "scratch too small: " + std::to_string(scratchBytes) + " < " + std::to_string(P.total);
        return -1;
    }
    if (((uintptr_t)scratch & 255) != 0) { err = "scratch must be 256-byte aligned"; return -1; }
    if (!b.blocks || !b.blockTxb || !b.txb || !b.coeffs) { err = "null device array"; return -1; }
    uint8_t *S = (uint8_t *)scratch;
    FrameG *dF = (FrameG *)(S + P.oF);
    TileG *dT = (TileG *)(S + P.oT);
    SbG *dSb = (SbG *)(S + P.oSb);
    uint32_t *dBFrame = (uint32_t *)(S + P.oBFrame), *dBTile = (uint32_t *)(S + P.oBTile), *dBTxbG = (uint32_t *)(S + P.oBTxbG);
    uint32_t *dTxbBlock = (uint32_t *)(S + P.oTxbBlock), *dSbFirst = (uint32_t *)(S + P.oSbFirst);
    uint8_t *mMiSize = S + P.oMap[0], *mYMode = S + P.oMap[1], *mSkip = S + P.oMap[2], *mTx = S + P.oMap[3];
    uint8_t *cL = S + P.oCtxL, *cD = S + P.oCtxD;
    uint16_t *dEob = (uint16_t *)(S + P.oEob);
    uint8_t *dType = S + P.oType;
    uint32_t *dCount = (uint32_t *)(S + P.oCount), *dOff = (uint32_t *)(S + P.oOff), *dSyms = (uint32_t *)(S + P.oSyms);
    void *dCub = S + P.oCub;
    uint64_t *dTb = (uint64_t *)(S + P.oTb), *dTe = (uint64_t *)(S + P.oTe);
    int *dTq = (int *)(S + P.oTq);
    uint64_t *dCapArr = (uint64_t *)(S + P.oCapArr), *dCapOff = (uint64_t *)(S + P.oCapOff), *dSizes = (uint64_t *)(S + P.oSizes);
    uint8_t *dStaging = S + P.oStaging, *dPacked = S + P.oPacked;
    DevStatus *dSt = (DevStatus *)(S + P.oResult);
    uint64_t *dTileOff = (uint64_t *)(dSt + 1);
    av1g_cdfs *dCdfDef = (av1g_cdfs *)(S + P.oCdfDef);
    // caller's arrays, shifted so that frame 0 starts at index 0
    const av1g_block *blocks = b.blocks + b.frames[0].blockBase;
    const uint32_t *blockTxb = b.blockTxb + b.frames[0].blockBase;
    const av1g_txb *txb = b.txb + b.frames[0].txbBase;
    const uint32_t nB = P.nB, nT = P.nT, nSb = P.nSb, nTiles = P.nTiles;
    const uint64_t nItems = P.nItems;
    auto rec = [&](int i) { if (ev) cudaEventRecord(ev->e[i], s); };
    const int TB = 128;
    auto grid = [](uint64_t n, int tb) { return (unsigned)((n + tb - 1) / tb); };

    rec(0);
    CK(cudaMemsetAsync(dSt, 0, sizeof(DevStatus), s));
    CK(cudaMemsetAsync(dSbFirst, 0xFF, 4ull * nSb, s));
    for (uint32_t i = 0; i < P.nF; i += kFrameChunk) {
        Chunk<FrameG, kFrameChunk> c;
        c.n = std::min<uint32_t>(kFrameChunk, P.nF - i);
        c.first = i;
        std::copy(P.F.begin() + i, P.F.begin() + i + c.n, c.v);
        k_put<FrameG, kFrameChunk><<<1, 128, 0, s>>>(c, dF);
    }
    for (uint32_t i = 0; i < nTiles; i += kTileChunk) {
        Chunk<TileG, kTileChunk> c;
        c.n = std::min<uint32_t>(kTileChunk, nTiles - i);
        c.first = i;
        std::copy(P.T.begin() + i, P.T.begin() + i + c.n, c.v);
        k_put<TileG, kTileChunk><<<1, 256, 0, s>>>(c, dT);
    }
    k_cdf_defaults<<<4, 1, 0, s>>>(dCdfDef);
    k_blocks<<<grid(nB, TB), TB, 0, s>>>(dF, P.nF, dT, blocks, blockTxb, nB, nT, dBFrame, dBTile, dBTxbG, dTxbBlock, dSbFirst, dSt);
    k_sbs<<<grid(nSb, TB), TB, 0, s>>>(dF, dT, nTiles, nSb, dSbFirst, dSb, dSt);
    k_tiles<<<nTiles, 32, 0, s>>>(dF, dT, dSb, blocks);
    rec(1);
    k_maps<<<grid(nB, TB), TB, 0, s>>>(blocks, dBFrame, dF, nB, mMiSize, mYMode, mSkip, mTx);
    bool anyLL = false;   // lossless frames in the batch: the LL instantiations (handle lossy frames too)
    for (const FrameG &g : P.F) anyLL |= g.qidx == 0;
    if (anyLL) k_txctx<true><<<grid(nT, TB), TB, 0, s>>>(blocks, dBFrame, dF, txb, dTxbBlock, b.coeffs, nT, cL, cD, dEob, dType);
    else k_txctx<false><<<grid(nT, TB), TB, 0, s>>>(blocks, dBFrame, dF, txb, dTxbBlock, b.coeffs, nT, cL, cD, dEob, dType);
    rec(2);
    CK(cudaMemsetAsync(dCount + nItems, 0, 4, s));
    if (anyLL) k_txsym<true><<<grid(nT, TB), TB, 0, s>>>(blocks, dBFrame, dBTile, dF, dT, txb, dTxbBlock, b.coeffs, nT, cL, cD, dEob, dType,
                                        dCount, nullptr, nullptr, nullptr, 0);
    else k_txsym<false><<<grid(nT, TB), TB, 0, s>>>(blocks, dBFrame, dBTile, dF, dT, txb, dTxbBlock, b.coeffs, nT, cL, cD, dEob, dType,
                                        dCount, nullptr, nullptr, nullptr, 0);
    if (anyLL) k_sbsym<true><<<grid(nSb, 32), 32, 0, s>>>(dSb, nSb, dF, dT, blocks, dBTxbG, mMiSize, mYMode, mSkip, mTx, dCount, nullptr, nullptr,
                                          nullptr, 0, dSt);
    else k_sbsym<false><<<grid(nSb, 32), 32, 0, s>>>(dSb, nSb, dF, dT, blocks, dBTxbG, mMiSize, mYMode, mSkip, mTx, dCount, nullptr, nullptr,
                                          nullptr, 0, dSt);
    rec(3);
    size_t cubBytes = P.cubItems;
    CK(cub::DeviceScan::ExclusiveSum(dCub, cubBytes, dCount, dOff, (int)(nItems + 1), s));
    rec(4);
    if (anyLL) k_txsym<true><<<grid(nT, TB), TB, 0, s>>>(blocks, dBFrame, dBTile, dF, dT, txb, dTxbBlock, b.coeffs, nT, cL, cD, dEob, dType,
                                        dCount, dOff, dSyms, dOff + nItems, P.symCap);
    else k_txsym<false><<<grid(nT, TB), TB, 0, s>>>(blocks, dBFrame, dBTile, dF, dT, txb, dTxbBlock, b.coeffs, nT, cL, cD, dEob, dType,
                                        dCount, dOff, dSyms, dOff + nItems, P.symCap);
    if (anyLL) k_sbsym<true><<<grid(nSb, 32), 32, 0, s>>>(dSb, nSb, dF, dT, blocks, dBTxbG, mMiSize, mYMode, mSkip, mTx, dCount, dOff, dSyms,
                                          dOff + nItems, P.symCap, dSt);
    else k_sbsym<false><<<grid(nSb, 32), 32, 0, s>>>(dSb, nSb, dF, dT, blocks, dBTxbG, mMiSize, mYMode, mSkip, mTx, dCount, dOff, dSyms,
                                          dOff + nItems, P.symCap, dSt);
    rec(5);
    k_tilerange<<<grid(nTiles, 64), 64, 0, s>>>(dT, nTiles, dF, dBTxbG, dOff, nItems, nB, P.symCap, dTb, dTe, dTq, dCapArr,
                                                dSizes, dSt);
    cubBytes = P.cubTiles;
    CK(cub::DeviceScan::ExclusiveSum(dCub, cubBytes, dCapArr, dCapOff, (int)(nTiles + 1), s));
    if (g_entropy_mode == 4)
        k_entropy_win<Rc32><<<nTiles, 64, kWinSmem, s>>>(dSyms, dTb, dTe, dTq, dCapOff, dCdfDef, dStaging, dSizes, dSt);
    else
        k_entropy_win<Rc64><<<nTiles, 64, kWinSmem, s>>>(dSyms, dTb, dTe, dTq, dCapOff, dCdfDef, dStaging, dSizes, dSt);
    rec(6);
    cubBytes = P.cubTiles;
    CK(cub::DeviceScan::ExclusiveSum(dCub, cubBytes, dSizes, dTileOff, (int)(nTiles + 1), s));
    k_pack<<<nTiles, 128, 0, s>>>(dStaging, dCapOff, dSizes, dTileOff, dPacked, nTiles, dSt);
    rec(7);
    CK(cudaGetLastError());
    out = DevResult();
    out.result = dSt;
    out.resultBytes = sizeof(DevStatus) + 8ull * (nTiles + 1);
    out.status = dSt;
    out.tileOffset = dTileOff;
    out.bytes = dPacked;
    out.bytesCapacity = P.capBytes;
    out.nTiles = nTiles;
    out.frameTile0 = P.frameTile0;
    out.syms = dSyms;
    out.symBegin = dTb;
    out.symEnd = dTe;
    return 0;
}

// ---------------------------------------------------------------------------
// host convenience wrapper: upload records -> run_batch_device -> download
// ---------------------------------------------------------------------------
template <class T>
static int up(T *&d, const std::vector<T> &h, size_t &bytes, std::string &err) {
    CK(cudaMalloc(&d, std::max<size_t>(1, h.size()) * sizeof(T)));
    if (!h.empty()) CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    bytes += h.size() * sizeof(T);
    return 0;
}

int run_batch(const std::vector<const av1g_decision_record *> &frames, bool keepSyms, BatchResult &res, std::string &err) {
    res = BatchResult();
    const uint32_t nF = (uint32_t)frames.size();
    std::vector<DevFrame> df(nF);
    std::vector<av1g_block> blocks;
    std::vector<uint32_t> blockTxb;
    std::vector<av1g_txb> txb;
    std::vector<int16_t> coeffs;
    for (uint32_t f = 0; f < nF; f++) {
        const av1g_decision_record &R = *frames[f];
        DevFrame &d = df[f];
        d.fp = R.fp;
        d.blockBase = (uint32_t)blocks.size();
        d.nBlocks = R.n_blocks;
        d.txbBase = (uint32_t)txb.size();
        d.nTxb = R.n_txb;
        d.coeffBase = coeffs.size();
        blocks.insert(blocks.end(), R.blocks, R.blocks + R.n_blocks);
        blockTxb.insert(blockTxb.end(), R.block_txb, R.block_txb + R.n_blocks);
        txb.insert(txb.end(), R.txb, R.txb + R.n_txb);
        coeffs.insert(coeffs.end(), R.coeffs, R.coeffs + R.n_coeffs);
    }
    cudaEvent_t eu0, eu1, ed0, ed1;
    cudaEventCreate(&eu0); cudaEventCreate(&eu1); cudaEventCreate(&ed0); cudaEventCreate(&ed1);
    DevEvents ev;
    for (auto &e : ev.e) cudaEventCreate(&e);
    size_t bytes = 0;
    cudaEventRecord(eu0);
    av1g_block *dB; uint32_t *dBT; av1g_txb *dT; int16_t *dC;
    if (up(dB, blocks, bytes, err) || up(dBT, blockTxb, bytes, err) || up(dT, txb, bytes, err) || up(dC, coeffs, bytes, err))
        return -1;
    cudaEventRecord(eu1);
    DevBatch b;
    b.frames = df.data();
    b.nFrames = nF;
    b.blocks = dB; b.blockTxb = dBT; b.txb = dT; b.coeffs = dC;
    void *scratch = nullptr;
    DevResult r;
    DevStatus st;
    for (int attempt = 0; attempt < 2; attempt++) {
        const size_t need = run_batch_device_scratch_bytes(b);
        if (!need) { std::string e2; Plan P; build_plan(b, P, e2); err = e2; return -1; }
        CK(cudaMalloc(&scratch, need));
        if (run_batch_device(b, 0, scratch, need, r, err, &ev)) return -1;
        CK(cudaMemcpy(&st, r.status, sizeof st, cudaMemcpyDeviceToHost));
        bytes += need;
        if (!(st.error & AV1G_GPU_ERR_SYMBOLS)) break;
        cudaFree(scratch);
        scratch = nullptr;
        bytes -= need;
        b.symbolCapacity = st.symbols;  // exact count, rerun
    }
    if (st.error) { err = "run_batch_device error flags " + std::to_string(st.error); return -1; }
    cudaEventRecord(ed0);
    std::vector<uint64_t> toff(r.nTiles + 1);
    CK(cudaMemcpy(toff.data(), r.tileOffset, 8ull * (r.nTiles + 1), cudaMemcpyDeviceToHost));
    std::vector<uint8_t> hout(toff[r.nTiles]);
    if (!hout.empty()) CK(cudaMemcpy(hout.data(), r.bytes, hout.size(), cudaMemcpyDeviceToHost));
    std::vector<uint32_t> hsyms;
    std::vector<uint64_t> sb(r.nTiles), se(r.nTiles);
    if (keepSyms) {
        hsyms.resize(st.symbols);
        if (st.symbols) CK(cudaMemcpy(hsyms.data(), r.syms, 4ull * st.symbols, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(sb.data(), r.symBegin, 8ull * r.nTiles, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(se.data(), r.symEnd, 8ull * r.nTiles, cudaMemcpyDeviceToHost));
    }
    cudaEventRecord(ed1);
    CK(cudaEventSynchronize(ed1));
    float m;
    cudaEventElapsedTime(&m, eu0, eu1); res.msUpload = m;
    double *dst[7] = {&res.msPrep, &res.msMaps, &res.msCount, &res.msScan, &res.msEmit, &res.msEntropy, &res.msPack};
    for (int i = 0; i < 7; i++) { cudaEventElapsedTime(&m, ev.e[i], ev.e[i + 1]); *dst[i] = m; }
    cudaEventElapsedTime(&m, ed0, ed1); res.msDownload = m;
    res.tileBytes.assign(nF, {});
    if (keepSyms) res.tileSyms.assign(nF, {});
    for (uint32_t f = 0; f < nF; f++)
        for (uint32_t i = r.frameTile0[f]; i < r.frameTile0[f + 1]; i++) {
            res.tileBytes[f].emplace_back(hout.begin() + toff[i], hout.begin() + toff[i + 1]);
            if (keepSyms) res.tileSyms[f].emplace_back(hsyms.begin() + sb[i], hsyms.begin() + se[i]);
        }
    res.totalSyms = st.symbols;
    res.nTiles = r.nTiles;
    {
        Plan P;
        build_plan(b, P, err);
        res.nSb = P.nSb;
        res.nBlocks = P.nB;
        res.nTxb = P.nT;
    }
    res.deviceBytes = bytes;
    cudaFree(dB); cudaFree(dBT); cudaFree(dT); cudaFree(dC); cudaFree(scratch);
    for (auto &e : ev.e) cudaEventDestroy(e);
    cudaEventDestroy(eu0); cudaEventDestroy(eu1); cudaEventDestroy(ed0); cudaEventDestroy(ed1);
    return 0;
}

// Entropy-code given per-tile symbol streams on the GPU only (benchmark utility).
int code_streams(const std::vector<const std::vector<uint32_t> *> &streams, const std::vector<int> &qidx,
                 std::vector<std::vector<uint8_t>> &out, double &ms, std::string &err) {
    const uint32_t n = (uint32_t)streams.size();
    std::vector<uint32_t> syms;
    std::vector<uint64_t> b(n), e(n), cap(n + 1, 0);
    std::vector<std::pair<const void *, uint64_t>> seen;  // identical stream objects share device data
    for (uint32_t i = 0; i < n; i++) {
        uint64_t at = UINT64_MAX;
        for (auto &pr : seen) if (pr.first == streams[i]) at = pr.second;
        if (at == UINT64_MAX) {
            at = syms.size();
            seen.push_back({streams[i], at});
            syms.insert(syms.end(), streams[i]->begin(), streams[i]->end());
        }
        b[i] = at;
        e[i] = at + streams[i]->size();
        cap[i + 1] = cap[i] + streams[i]->size() / 2 + 256;
    }
    size_t bytes = 0;
    uint32_t *dS; uint64_t *dB, *dE, *dCap, *dSz64; int *dQ, *dSize; av1g_cdfs *dScr = nullptr, *dDef; uint16_t *dPre = nullptr;
    uint8_t *dOut; DevStatus *dSt;
    if (up(dS, syms, bytes, err) || up(dB, b, bytes, err) || up(dE, e, bytes, err) || up(dCap, cap, bytes, err) ||
        up(dQ, qidx, bytes, err)) return -1;
    const bool win = g_entropy_mode >= 3;
    if (!win) {
        CK(cudaMalloc(&dScr, (size_t)n * sizeof(av1g_cdfs)));
        CK(cudaMalloc(&dPre, cap[n] * 2));
    }
    CK(cudaMalloc(&dDef, 4 * sizeof(av1g_cdfs)));
    CK(cudaMalloc(&dOut, cap[n]));
    CK(cudaMalloc(&dSize, n * sizeof(int)));
    CK(cudaMalloc(&dSz64, n * 8));
    CK(cudaMalloc(&dSt, sizeof(DevStatus)));
    CK(cudaMemset(dSt, 0, sizeof(DevStatus)));
    k_cdf_defaults<<<4, 1>>>(dDef);
    CK(cudaDeviceSynchronize());
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0); cudaEventCreate(&e1);
    cudaEventRecord(e0);
    if (g_entropy_mode == 4) k_entropy_win<Rc32><<<n, 64, kWinSmem>>>(dS, dB, dE, dQ, dCap, dDef, dOut, dSz64, dSt);
    else if (win) k_entropy_win<Rc64><<<n, 64, kWinSmem>>>(dS, dB, dE, dQ, dCap, dDef, dOut, dSz64, dSt);
    else launch_entropy_m4(g_entropy_smem, dS, dB, dE, dQ, dCap, dScr, dPre, dOut, dSize, n);
    cudaEventRecord(e1);
    CK(cudaEventSynchronize(e1));
    CK(cudaGetLastError());
    float f;
    cudaEventElapsedTime(&f, e0, e1);
    ms = f;
    std::vector<int> sz(n);
    if (win) {
        std::vector<uint64_t> s64(n);
        DevStatus st;
        CK(cudaMemcpy(s64.data(), dSz64, n * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(&st, dSt, sizeof st, cudaMemcpyDeviceToHost));
        for (uint32_t i = 0; i < n; i++) sz[i] = (st.error & AV1G_GPU_ERR_OUTPUT) && !s64[i] ? -1 : (int)s64[i];
    } else {
        CK(cudaMemcpy(sz.data(), dSize, n * sizeof(int), cudaMemcpyDeviceToHost));
    }
    std::vector<uint8_t> hout(cap[n]);
    CK(cudaMemcpy(hout.data(), dOut, cap[n], cudaMemcpyDeviceToHost));
    out.assign(n, {});
    for (uint32_t i = 0; i < n; i++) {
        if (sz[i] < 0) { err = "entropy output buffer overflow (stream " + std::to_string(i) + ")"; return -1; }
        out[i].assign(hout.begin() + cap[i], hout.begin() + cap[i] + sz[i]);
    }
    cudaFree(dS); cudaFree(dB); cudaFree(dE); cudaFree(dCap); cudaFree(dQ); cudaFree(dScr); cudaFree(dPre); cudaFree(dDef);
    cudaFree(dOut); cudaFree(dSize); cudaFree(dSz64); cudaFree(dSt);
    cudaEventDestroy(e0); cudaEventDestroy(e1);
    return 0;
}

}  // namespace av1g_gpu
