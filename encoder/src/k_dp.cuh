// per-64x64 partition DP over the analysis J grids (one thread per SB), then the leaf
// blocks in AV1 decode order (decode_partition pre-order).
#pragma once
#include "dev_common.cuh"

__device__ __forceinline__ int sq_bsize(int n) {
    return n == 4 ? AV1G_BLOCK_4X4 : n == 8 ? AV1G_BLOCK_8X8 : n == 16 ? AV1G_BLOCK_16X16 : n == 32 ? AV1G_BLOCK_32X32 : AV1G_BLOCK_64X64;
}
// block size of partition p's main sub-block for a node of size n (Partition_Subsize)
__device__ __forceinline__ int sub_bsize(int p, int n) { return AV1G_T(Partition_Subsize)[p][sq_bsize(n)]; }

__device__ __forceinline__ float dp_blockJ(const GpBatch &B, const GpImage &I, int bs, int x, int y) {
    if (bs >= AV1G_BLOCK_INVALID || I.shNbx[bs] == 0) return GP_INF;
    const int w = bw_px(bs), h = bh_px(bs);
    return B.blk[I.shOff[bs] + (uint32_t)(y / h) * I.shNbx[bs] + x / w].J;
}

struct DpNode { float J; int8_t p; };

// Safety net of the leaf walk (GpEncoder::init's dp_complete keeps it unreached; B.stats[5] counts what it codes).
// A node whose every option has a non-finite J keeps PARTITION_SPLIT, down to 8x8 nodes whose 4x4 leaves have no
// analysis record (J = GP_INF, ncand 0, cand / tx never written).  Every other leaf belongs to a partition of finite
// cost (NONE only for a block with a record), so these are the only leaves k_recon could code from unwritten memory.
// The node is coded with the first of these shapes the analysis has record slots for (I.shNbx), each record without a
// finite J made codable: DC_PRED, delta 0, DCT_DCT, ncand 1 (J stays GP_INF; k_recon reads only cand[0 .. ncand-1]
// and tx of a leaf's record, k_recon.cuh: br, and nothing after it reads the records):
// - 4x4: the DP's own SPLIT; counts the records it made up (at least one: else SPLIT would have been finite);
// - else 8x8 (NONE), 8x4 (HORZ: two leaves) or 4x8 (VERT: two leaves), counting every leaf (a 4x4 leaf's record index
//   would land in another shape's records).  An 8x8 node is never cut by the frame, so all four partitions are legal
//   there whatever --part8 says (the mask only restricts the search; the backend derives the partition from the leaves).
// Options with none of the four shapes are refused at init (dp_complete: a frame corner of 8x8 pixels needs one), so
// the last return is not reached.  Returns the partition | count << 4.
static __device__ __noinline__ int dp_net8(const GpBatch &B, const GpImage &I, int x, int y, int FW, int FH) {
    auto fix = [&](int bs, int bx, int by) {   // 1: the record had no finite J (its candidates, if any, are not trusted)
        GpBlockRes &r = B.blk[I.shOff[bs] + (uint32_t)(by / bh_px(bs)) * I.shNbx[bs] + bx / bw_px(bs)];
        if (r.J < GP_INF) return 0;
        r.cand[0] = (uint8_t)cand_pack(AV1G_DC_PRED, 0);
        r.ncand = 1;
        r.tx = AV1G_DCT_DCT;
        return 1;
    };
    if (I.shNbx[AV1G_BLOCK_4X4]) {
        int n = 0;
        for (int k = 0; k < 4; k++) {
            const int xx = x + (k & 1) * 4, yy = y + (k >> 1) * 4;
            if (xx < FW && yy < FH) n += fix(AV1G_BLOCK_4X4, xx, yy);
        }
        return AV1G_PARTITION_SPLIT | n << 4;
    }
    if (I.shNbx[AV1G_BLOCK_8X8]) {
        fix(AV1G_BLOCK_8X8, x, y);
        return AV1G_PARTITION_NONE | 1 << 4;
    }
    if (I.shNbx[AV1G_BLOCK_8X4]) {
        fix(AV1G_BLOCK_8X4, x, y);
        fix(AV1G_BLOCK_8X4, x, y + 4);
        return AV1G_PARTITION_HORZ | 2 << 4;
    }
    if (I.shNbx[AV1G_BLOCK_4X8]) {
        fix(AV1G_BLOCK_4X8, x, y);
        fix(AV1G_BLOCK_4X8, x + 4, y);
        return AV1G_PARTITION_VERT | 2 << 4;
    }
    return AV1G_PARTITION_SPLIT;
}

template <int BD = 8>
__global__ void k_dp(GpBatch B, GpParams P) {
    const uint32_t sb = blockIdx.x * blockDim.x + threadIdx.x;
    if (sb >= B.nSb) return;
    const GpImage &I = B.img[B.sbImg[sb]];
    const uint32_t local = sb - I.sbBase;
    const int sbx = (int)(local % I.sbCols) * 64, sby = (int)(local / I.sbCols) * 64;
    const int FW = I.MiCols * 4, FH = I.MiRows * 4;
    const float lambda = gp_lambda<BD>(sb_qindex(I, sbx, sby), P.lambdaMul);
    DpNode node[85];
    const int lvBase[4] = {0, 1, 5, 21};
    for (int L = 3; L >= 0; L--) {
        const int n = 64 >> L, half = n >> 1, q4 = n >> 2, per = 1 << L;
        const int bsl = L == 3 ? 1 : L == 2 ? 2 : L == 1 ? 3 : 4;
        const int pmask = n == 8 ? P.partMask8 : P.partMaskN;
        for (int ny = 0; ny < per; ny++)
            for (int nx = 0; nx < per; nx++) {
                DpNode &nd = node[lvBase[L] + ny * per + nx];
                const int x = sbx + nx * n, y = sby + ny * n;
                if (x >= FW || y >= FH) { nd.J = 0.f; nd.p = -1; continue; }
                const bool hasRows = (y + half) < FH, hasCols = (x + half) < FW;
                // SPLIT cost
                float splitJ = 0.f;
                if (n == 8) {
                    for (int k = 0; k < 4; k++) {
                        const int xx = x + (k & 1) * 4, yy = y + (k >> 1) * 4;
                        if (xx < FW && yy < FH) splitJ += dp_blockJ(B, I, AV1G_BLOCK_4X4, xx, yy);
                    }
                } else {
                    const int cb = lvBase[L + 1], cper = per * 2;
                    for (int k = 0; k < 4; k++) splitJ += node[cb + (ny * 2 + (k >> 1)) * cper + nx * 2 + (k & 1)].J;
                }
                float best = GP_INF;
                int bp = AV1G_PARTITION_SPLIT;
                const float lamN = lambda * ssim_scale(I, x, y, n, n);
                auto consider = [&](int p, float J, float bits) {
                    const float t = J + lamN * bits;
                    if (t < best) { best = t; bp = p; }
                };
                const bool big = n > P.maxPart;  // only SPLIT above the largest allowed partition
                if (hasRows && hasCols) {
                    const float *pb = g_avgPart[bsl];
                    consider(AV1G_PARTITION_SPLIT, splitJ, pb[AV1G_PARTITION_SPLIT]);
                    if (!big) {
                        const int hz = sub_bsize(AV1G_PARTITION_HORZ, n), vt = sub_bsize(AV1G_PARTITION_VERT, n);
                        const int sp = sub_bsize(AV1G_PARTITION_SPLIT, n);
                        if (pmask & (1 << AV1G_PARTITION_NONE)) {
                            // a block without a record stays GP_INF (a --split-bias below 1 would make it look finite)
                            const float jn = dp_blockJ(B, I, sq_bsize(n), x, y);
                            if (jn < GP_INF) consider(AV1G_PARTITION_NONE, jn * P.splitBias, pb[AV1G_PARTITION_NONE]);
                        }
                        if (pmask & (1 << AV1G_PARTITION_HORZ))
                            consider(AV1G_PARTITION_HORZ, dp_blockJ(B, I, hz, x, y) + dp_blockJ(B, I, hz, x, y + half), pb[AV1G_PARTITION_HORZ]);
                        if (pmask & (1 << AV1G_PARTITION_VERT))
                            consider(AV1G_PARTITION_VERT, dp_blockJ(B, I, vt, x, y) + dp_blockJ(B, I, vt, x + half, y), pb[AV1G_PARTITION_VERT]);
                        if (n > 8) {
                            const float s00 = dp_blockJ(B, I, sp, x, y), s01 = dp_blockJ(B, I, sp, x + half, y);
                            const float s10 = dp_blockJ(B, I, sp, x, y + half), s11 = dp_blockJ(B, I, sp, x + half, y + half);
                            if (pmask & (1 << AV1G_PARTITION_HORZ_A))
                                consider(AV1G_PARTITION_HORZ_A, s00 + s01 + dp_blockJ(B, I, hz, x, y + half), pb[AV1G_PARTITION_HORZ_A]);
                            if (pmask & (1 << AV1G_PARTITION_HORZ_B))
                                consider(AV1G_PARTITION_HORZ_B, dp_blockJ(B, I, hz, x, y) + s10 + s11, pb[AV1G_PARTITION_HORZ_B]);
                            if (pmask & (1 << AV1G_PARTITION_VERT_A))
                                consider(AV1G_PARTITION_VERT_A, s00 + s10 + dp_blockJ(B, I, vt, x + half, y), pb[AV1G_PARTITION_VERT_A]);
                            if (pmask & (1 << AV1G_PARTITION_VERT_B))
                                consider(AV1G_PARTITION_VERT_B, dp_blockJ(B, I, vt, x, y) + s01 + s11, pb[AV1G_PARTITION_VERT_B]);
                            const int h4 = sub_bsize(AV1G_PARTITION_HORZ_4, n), v4 = sub_bsize(AV1G_PARTITION_VERT_4, n);
                            if (pmask & (1 << AV1G_PARTITION_HORZ_4)) {
                                float J = 0.f;
                                for (int i = 0; i < 4; i++) if (i < 3 || y + 3 * q4 < FH) J += dp_blockJ(B, I, h4, x, y + i * q4);
                                consider(AV1G_PARTITION_HORZ_4, J, pb[AV1G_PARTITION_HORZ_4]);
                            }
                            if (pmask & (1 << AV1G_PARTITION_VERT_4)) {
                                float J = 0.f;
                                for (int i = 0; i < 4; i++) if (i < 3 || x + 3 * q4 < FW) J += dp_blockJ(B, I, v4, x + i * q4, y);
                                consider(AV1G_PARTITION_VERT_4, J, pb[AV1G_PARTITION_VERT_4]);
                            }
                        }
                    }
                } else if (hasCols) {  // bottom half outside: split_or_horz
                    consider(AV1G_PARTITION_SPLIT, splitJ, 1.f);
                    if (!big) consider(AV1G_PARTITION_HORZ, dp_blockJ(B, I, sub_bsize(AV1G_PARTITION_HORZ, n), x, y), 1.f);
                } else if (hasRows) {  // right half outside: split_or_vert
                    consider(AV1G_PARTITION_SPLIT, splitJ, 1.f);
                    if (!big) consider(AV1G_PARTITION_VERT, dp_blockJ(B, I, sub_bsize(AV1G_PARTITION_VERT, n), x, y), 1.f);
                } else {
                    consider(AV1G_PARTITION_SPLIT, splitJ, 0.f);
                }
                nd.J = best;
                nd.p = (int8_t)bp;
            }
    }
    B.sbJ[sb] = node[0].J;   // (not finite exactly where the safety net acts: the host names those frames)
    // ---- leaves in decode order ----
    uint32_t nRep = 0;   // blocks the safety net (dp_net8) coded
    GpLeaf *out = B.leaves + (size_t)sb * GP_MAX_LEAVES;
    int nl = 0;
    auto leaf = [&](int bs, int x, int y) {
        if (x >= FW || y >= FH) return;
        GpLeaf lf;
        lf.bsize = (uint8_t)bs;
        lf.x4 = (uint8_t)((x - sbx) >> 2);
        lf.y4 = (uint8_t)((y - sby) >> 2);
        lf.pad = 0;
        out[nl++] = lf;
    };
    // explicit stack of (level, nx, ny)
    int stk[64][3];
    int sp = 0;
    stk[sp][0] = 0; stk[sp][1] = 0; stk[sp][2] = 0; sp++;
    while (sp > 0) {
        sp--;
        const int L = stk[sp][0], nx = stk[sp][1], ny = stk[sp][2];
        const int n = 64 >> L, half = n >> 1, q4 = n >> 2, per = 1 << L;
        const DpNode &nd = node[lvBase[L] + ny * per + nx];
        const int x = sbx + nx * n, y = sby + ny * n;
        if (nd.p < 0) continue;
        const int p = nd.p;
        const int sub = sub_bsize(p, n), sp2 = sub_bsize(AV1G_PARTITION_SPLIT, n);
        const bool hasRows = (y + half) < FH, hasCols = (x + half) < FW;
        switch (p) {
        case AV1G_PARTITION_NONE: leaf(sub, x, y); break;
        case AV1G_PARTITION_HORZ: leaf(sub, x, y); if (hasRows) leaf(sub, x, y + half); break;
        case AV1G_PARTITION_VERT: leaf(sub, x, y); if (hasCols) leaf(sub, x + half, y); break;
        case AV1G_PARTITION_SPLIT:
            if (n == 8) {
                if (!(nd.J < GP_INF)) {   // safety net (dp_net8)
                    const int r = dp_net8(B, I, x, y, FW, FH), pt = r & 15;
                    nRep += (uint32_t)(r >> 4);
                    if (pt != AV1G_PARTITION_SPLIT) {
                        const int bs = pt == AV1G_PARTITION_NONE ? AV1G_BLOCK_8X8 : pt == AV1G_PARTITION_HORZ ? AV1G_BLOCK_8X4 : AV1G_BLOCK_4X8;
                        leaf(bs, x, y);
                        if (pt != AV1G_PARTITION_NONE) leaf(bs, x + (pt == AV1G_PARTITION_VERT ? 4 : 0), y + (pt == AV1G_PARTITION_HORZ ? 4 : 0));
                        break;
                    }
                }
                leaf(AV1G_BLOCK_4X4, x, y); leaf(AV1G_BLOCK_4X4, x + 4, y);
                leaf(AV1G_BLOCK_4X4, x, y + 4); leaf(AV1G_BLOCK_4X4, x + 4, y + 4);
            } else {  // push children in reverse decode order
                for (int k = 3; k >= 0; k--) {
                    stk[sp][0] = L + 1; stk[sp][1] = nx * 2 + (k & 1); stk[sp][2] = ny * 2 + (k >> 1); sp++;
                }
            }
            break;
        case AV1G_PARTITION_HORZ_A: leaf(sp2, x, y); leaf(sp2, x + half, y); leaf(sub, x, y + half); break;
        case AV1G_PARTITION_HORZ_B: leaf(sub, x, y); leaf(sp2, x, y + half); leaf(sp2, x + half, y + half); break;
        case AV1G_PARTITION_VERT_A: leaf(sp2, x, y); leaf(sp2, x, y + half); leaf(sub, x + half, y); break;
        case AV1G_PARTITION_VERT_B: leaf(sub, x, y); leaf(sp2, x + half, y); leaf(sp2, x + half, y + half); break;
        case AV1G_PARTITION_HORZ_4:
            for (int i = 0; i < 4; i++) if (i < 3 || y + 3 * q4 < FH) leaf(sub, x, y + i * q4);
            break;
        default:
            for (int i = 0; i < 4; i++) if (i < 3 || x + 3 * q4 < FW) leaf(sub, x + i * q4, y);
            break;
        }
    }
    B.leafCountW[sb] = (uint16_t)nl;
    if (nRep) atomicAdd(&B.stats[5], nRep);
}
