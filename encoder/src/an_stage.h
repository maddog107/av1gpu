// Staged open-loop analysis (opt-in, GpConfig::anTopdown; off by default): the shapes are analysed group by group, and
// a partition DP over the groups done so far decides on the device which blocks of the next group are analysed at all.
// Device-side lists and fixed worst-case launch sizes: no host synchronisation.
//
//   top-down (anTopdown): group 1 = the shapes whose smaller side is 16 or more; DP; the shapes with a smaller side of 8
//     only in the 16x16 cells whose covering leaf of that DP has a smaller side of 16 (or a neighbouring cell's has);
//     DP; the shapes with a smaller side of 4 likewise in 8x8 cells.  Cells that the frame cuts are always analysed.
//
// Kept out of gp.h / kernels.h (and out of GpParams), so that the closed loop and the default analysis kernels do not
// change at all: only the analysis translation units, k_misc.cu and pipeline.cpp include this header.
#ifndef GP_AN_STAGE_H
#define GP_AN_STAGE_H

#include <stdint.h>

// Per-launch options of the MODE 1 analysis kernels (k_analysis.cuh, k_analysis_small.cuh).
struct GpAnX {
    const uint32_t *list;   // non-null: the blocks to analyse, as shape-global indices (the shape's raster order of
                            // GpBatch::blk, as GpBatch::shImgStart counts them); the others were filled by k_an_list
    const uint32_t *count;  // [1] entries of list
};
// the shape needs the MODE 1 analysis kernel (a zero GpAnX: none)
inline bool an_x_on(const GpAnX *x) { return x && x->list; }

// Options of a DP stage (k_dp<BD, true>, k_dp.cuh): which shapes the DP may use, and what it writes.
struct GpDpStage {
    uint32_t shapeMask;     // AV1G_BLOCK_* bits: shapes analysed so far (the others count as GP_INF)
    uint8_t *core;          // non-null: per SB, one byte per cell: 1 where the covering leaf's smaller side <= coreSide
    int coreCell;           // 16 (16 bytes per SB, raster in the SB) or 8 (64 bytes)
    int coreSide;
};

// What k_an_list tests for one shape of a stage.
struct GpAnListShape {
    int bs;
    uint32_t total;         // blocks of the shape in the batch (shTotal)
    uint32_t *list, *count;
    int flat;               // apply the flat-area pruning (prune_flat)
    int cell;               // 16 / 8: alive only where a covered cell of that size is marked in core (or a 4-neighbour
                            // is, with neigh), or the frame cuts the 16x16 cell; 0: no cell test
};
#define GP_AN_LIST_MAX 8
struct GpAnListStage {
    int n;
    GpAnListShape s[GP_AN_LIST_MAX];
    const uint8_t *core16, *core8;
    int neigh;              // core cells also make their 4 neighbours alive (T3)
};

// Stage of a w x h shape: 1 (analysed first, every block); with the top-down stages 2 (smaller side 8) and 3 (side 4).
// A DP stage before each but the first.
inline int an_stage_of(int topdown, int w, int h) {
    const int m = w < h ? w : h, g = m >= 16 ? 1 : m == 8 ? 2 : 3;
    return topdown ? g : 1;
}

// One batch's staged analysis (pipeline.cpp fills it when GpConfig asks for any of it): the switches and the device
// buffers.
struct GpAnRun {
    int topdown;            // 1: top-down refinement
    uint32_t *list[22];     // per shape of stages 2..3: [shTotal] (GP_NSHAPES)
    uint32_t *count;        // [GP_NSHAPES] (zeroed by gp_launch_analysis_staged)
    uint8_t *core16, *core8;   // [nSb * 16], [nSb * 64]
};

#endif
