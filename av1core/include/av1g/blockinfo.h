/* av1g/blockinfo.h - block / transform geometry rules of the AV1 syntax that every
 * engine needs (host + device).  4:2:0, 4:4:4 or monochrome, no 128x128 blocks.
 * Chroma rules take the frame's subsampling (ssx, ssy) = (subsampling_x, subsampling_y):
 * (1, 1) for 4:2:0, (0, 0) for 4:4:4 (av1g_ssx / av1g_ssy); monochrome frames carry no
 * chroma, any value works for them. */
#ifndef AV1G_BLOCKINFO_H
#define AV1G_BLOCKINFO_H

#include "av1g/hd.h"
#include "av1g/enums_gen.h"
#include "av1g/spec_tables.h"
#include "av1g/decisions.h"

/* subsampling_x / subsampling_y of a frame (monochrome: 1, as in the spec) */
AV1G_HD int av1g_ssx(const av1g_frame_params *fp) { return (!fp->monochrome && fp->chroma_format == AV1G_CHROMA_444) ? 0 : 1; }
AV1G_HD int av1g_ssy(const av1g_frame_params *fp) { return (!fp->monochrome && fp->chroma_format == AV1G_CHROMA_444) ? 0 : 1; }

AV1G_HD int av1g_bw4(int bsize) { return AV1G_T(Num_4x4_Blocks_Wide)[bsize]; }
AV1G_HD int av1g_bh4(int bsize) { return AV1G_T(Num_4x4_Blocks_High)[bsize]; }

/* MiCols / MiRows of the coded area (spec compute_image_size) */
AV1G_HD int av1g_mi_cols(int width) { return 2 * ((width + 7) >> 3); }
AV1G_HD int av1g_mi_rows(int height) { return 2 * ((height + 7) >> 3); }

/* HasChroma (spec decode_block): 4:4:4 blocks always carry chroma; in 4:2:0 a 4xN / Nx4
 * block at an even mi_col / mi_row does not (its chroma is coded with the next block) */
AV1G_HD int av1g_has_chroma(int bsize, int mi_row, int mi_col, int mono, int ssx, int ssy) {
    if (mono) return 0;
    if (ssy && av1g_bh4(bsize) == 1 && (mi_row & 1) == 0) return 0;
    if (ssx && av1g_bw4(bsize) == 1 && (mi_col & 1) == 0) return 0;
    return 1;
}

/* luma TxSize of an intra block for a given tx_depth (read_tx_size) */
AV1G_HD int av1g_luma_tx_size(int bsize, int tx_depth) {
    int t = AV1G_T(Max_Tx_Size_Rect)[bsize];
    for (int i = 0; i < tx_depth; i++) t = AV1G_T(Split_Tx_Size)[t];
    return t;
}
/* largest legal tx_depth for an intra block */
AV1G_HD int av1g_max_tx_depth(int bsize) {
    int d = AV1G_T(Max_Tx_Depth)[bsize];
    return d > 2 ? 2 : d;
}

/* chroma block size (get_plane_residual_size, plane > 0); 4:4:4: the block size itself */
AV1G_HD int av1g_uv_bsize(int bsize, int ssx, int ssy) { return AV1G_T(Subsampled_Size)[bsize][ssx][ssy]; }

/* chroma TxSize (get_tx_size for plane > 0): the largest tx of the chroma block, at most 32
 * samples per side (4:4:4 blocks with a 64-sample side get several chroma tx blocks) */
AV1G_HD int av1g_uv_tx_size(int bsize, int ssx, int ssy) {
    int uvTx = AV1G_T(Max_Tx_Size_Rect)[av1g_uv_bsize(bsize, ssx, ssy)];
    int w = AV1G_T(Tx_Width)[uvTx], h = AV1G_T(Tx_Height)[uvTx];
    if (w == 64 || h == 64) {
        if (w == 16) return AV1G_TX_16X32;
        if (h == 16) return AV1G_TX_32X16;
        return AV1G_TX_32X32;
    }
    return uvTx;
}

/* CflAllowed for a non-lossless block */
AV1G_HD int av1g_cfl_allowed(int bsize) {
    return av1g_bw4(bsize) <= 8 && av1g_bh4(bsize) <= 8;
}

/* number of coefficients stored for a tx size (tw * th) */
AV1G_HD int av1g_tx_coeffs(int txSz) {
    int w = AV1G_T(Tx_Width)[txSz], h = AV1G_T(Tx_Height)[txSz];
    return (w > 32 ? 32 : w) * (h > 32 ? 32 : h);
}

/* Block-level neighbour availability (spec decode_block): AvailU/AvailL for luma and
 * AvailUChroma/AvailLChroma; tile bounds in luma mi units. */
AV1G_HD void av1g_block_avail(int bsize, int mi_row, int mi_col, int mono, int ssx, int ssy, int tileR0, int tileR1, int tileC0,
                              int tileC1, int *availU, int *availL, int *availUC, int *availLC) {
#define AV1G_INSIDE(r, c) ((c) >= tileC0 && (c) < tileC1 && (r) >= tileR0 && (r) < tileR1)
    *availU = AV1G_INSIDE(mi_row - 1, mi_col);
    *availL = AV1G_INSIDE(mi_row, mi_col - 1);
    if (av1g_has_chroma(bsize, mi_row, mi_col, mono, ssx, ssy)) {
        *availUC = (ssy && av1g_bh4(bsize) == 1) ? AV1G_INSIDE(mi_row - 2, mi_col) : *availU;
        *availLC = (ssx && av1g_bw4(bsize) == 1) ? AV1G_INSIDE(mi_row, mi_col - 2) : *availL;
    } else {
        *availUC = *availLC = 0;
    }
#undef AV1G_INSIDE
}

/* Edge availability of a coded tx block for intra prediction (spec transform_block:
 * haveLeft, haveAbove, haveAboveRight, haveBelowLeft) WITHOUT the decoder's sequential
 * BlockDecoded state: ordmap[y4 * stride + x4] (plane units) holds the index (layout
 * order) of the coded tx block of this plane covering that 4x4, or 0xFFFFFFFF.  The map
 * must cover the whole 64x64 superblock grid (stride >= (sbCols * 16) >> ss, rows >=
 * (sbRows * 16) >> ss) and every coded tx block marks its FULL area, also the part
 * beyond the frame edge (as the spec's BlockDecoded marking does).  A neighbour is "decoded" iff it lies in the current superblock and its
 * covering tx block precedes t, or it is in the row above / column left of the SB
 * inside the tile (spec clear_block_decoded_flags).  For GPU closed-loop engines.
 *   plane/x4/y4/txSz/t: the tx block; xi/yi: its 4x4 offset inside the block;
 *   blockAvailL/U: AvailL/AvailU (plane 0) or AvailLChroma/AvailUChroma; ssx/ssy: the frame's. */
AV1G_HD void av1g_txb_edges(int plane, int ssx, int ssy, int x4, int y4, int txSz, uint32_t t, int xi, int yi, int blockAvailL,
                            int blockAvailU, int tileR1, int tileC1, const uint32_t *ordmap, int stride,
                            int *haveLeft, int *haveAbove, int *haveAboveRight, int *haveBelowLeft) {
    const int subX = plane ? ssx : 0, subY = plane ? ssy : 0;
    const int row = y4 << subY, col = x4 << subX;          /* luma mi */
    const int sbR = row & ~15, sbC = col & ~15;            /* SB origin, luma mi */
    const int by = (row & 15) >> subY, bx = (col & 15) >> subX;
    const int stepX = AV1G_T(Tx_Width)[txSz] >> 2, stepY = AV1G_T(Tx_Height)[txSz] >> 2;
    const int sbSizeX = 16 >> subX, sbSizeY = 16 >> subY;
    const int sbW4 = (tileC1 - sbC) >> subX, sbH4 = (tileR1 - sbR) >> subY;
    const int x0 = sbC >> subX, y0 = sbR >> subY;          /* SB origin, plane 4x4 units */
    *haveLeft = blockAvailL || xi > 0;
    *haveAbove = blockAvailU || yi > 0;
    /* BlockDecoded[plane][yy][xx] in SB-relative plane units */
#define AV1G_BD(yy, xx) ((yy) < 0 ? ((xx) < sbW4) : (xx) < 0 ? ((yy) < sbH4 && (yy) != sbSizeY) : \
                         ((yy) >= sbSizeY || (xx) >= sbSizeX) ? 0 : \
                         (ordmap[(size_t)(y0 + (yy)) * stride + x0 + (xx)] < t))
    *haveAboveRight = AV1G_BD(by - 1, bx + stepX);
    *haveBelowLeft = AV1G_BD(by + stepY, bx - 1);
#undef AV1G_BD
}

/* get_filter_type (spec 7.11.2.8) for the intra edge filter: 1 if the above or left block
 * (4:2:0 chroma: the chroma-carrying block of that 8x8 unit) uses SMOOTH / SMOOTH_V / SMOOTH_H.
 * yModes/uvModes: 4x4 maps (stride = MiCols) of YMode and UVMode (UVMode written by blocks
 * that carry chroma); availU/availL: AvailU/AvailL (plane 0) or AvailUChroma/AvailLChroma. */
AV1G_HD int av1g_get_filter_type(int plane, int mi_row, int mi_col, int availU, int availL, const uint8_t *yModes,
                                 const uint8_t *uvModes, int miCols, int ssx, int ssy) {
#define AV1G_SMOOTH(m) ((m) == AV1G_SMOOTH_PRED || (m) == AV1G_SMOOTH_V_PRED || (m) == AV1G_SMOOTH_H_PRED)
    int aboveSmooth = 0, leftSmooth = 0;
    const uint8_t *m = plane ? uvModes : yModes;
    if (availU) {
        int r = mi_row - 1, c = mi_col;
        if (plane > 0) {
            if (ssx && !(mi_col & 1)) c++;
            if (ssy && (mi_row & 1)) r--;
        }
        aboveSmooth = AV1G_SMOOTH(m[(size_t)r * miCols + c]);
    }
    if (availL) {
        int r = mi_row, c = mi_col - 1;
        if (plane > 0) {
            if (ssx && (mi_col & 1)) c--;
            if (ssy && !(mi_row & 1)) r++;
        }
        leftSmooth = AV1G_SMOOTH(m[(size_t)r * miCols + c]);
    }
    return aboveSmooth || leftSmooth;
#undef AV1G_SMOOTH
}

/* Iterate the coded tx blocks of one block in bitstream order.  Calls
 * fn(plane, txSz, x4, y4, xi, yi) where (x4,y4) is the tx block position in its
 * plane (4-sample units) and (xi,yi) its index within the block (4x4 units of the
 * plane, as the spec's x/y in transform_block).  Returns the count. */
template <typename F>
AV1G_HDI int av1g_for_each_txb(int bsize, int mi_row, int mi_col, int tx_depth, int mono, int ssx, int ssy,
                               int mi_rows, int mi_cols, F &&fn) {
    int n = 0;
    const int hasChroma = av1g_has_chroma(bsize, mi_row, mi_col, mono, ssx, ssy);
    for (int plane = 0; plane < 1 + 2 * hasChroma; plane++) {
        const int subX = plane ? ssx : 0, subY = plane ? ssy : 0;
        const int txSz = plane ? av1g_uv_tx_size(bsize, ssx, ssy) : av1g_luma_tx_size(bsize, tx_depth);
        const int stepX = AV1G_T(Tx_Width)[txSz] >> 2, stepY = AV1G_T(Tx_Height)[txSz] >> 2;
        const int planeSz = plane ? av1g_uv_bsize(bsize, ssx, ssy) : bsize;
        const int num4x4W = av1g_bw4(planeSz), num4x4H = av1g_bh4(planeSz);
        const int baseX4 = mi_col >> subX, baseY4 = mi_row >> subY;
        const int maxX4 = mi_cols >> subX, maxY4 = mi_rows >> subY; /* (MiCols*MI_SIZE)>>sub in 4-units */
        for (int y = 0; y < num4x4H; y += stepY)
            for (int x = 0; x < num4x4W; x += stepX) {
                int x4 = baseX4 + x, y4 = baseY4 + y;
                if (x4 >= maxX4 || y4 >= maxY4) continue;
                fn(plane, txSz, x4, y4, x, y);
                n++;
            }
    }
    return n;
}

#endif
