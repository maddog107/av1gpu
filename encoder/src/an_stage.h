// Staged open-loop analysis (GpAnOpts in pipeline.h, --topdown / --prune-41; on in the presets base and turbo): the shapes are
// analysed group by group, and a partition DP over the groups done so far decides on the device which blocks of the next
// group are analysed at all.  Device-side lists and fixed worst-case launch sizes: no host synchronisation.
//
//   top-down (--topdown): group 1 = the shapes whose smaller side is 16 or more; DP; the shapes with a smaller side of 8
//     only in the 16x16 cells whose covering leaf of that DP has a smaller side of 16 (or a neighbouring cell's has);
//     DP; the shapes with a smaller side of 4 likewise in 8x8 cells.  Cells that the frame cuts are always analysed.
//   4:1 gating (--prune-41): a node's n x n/4 blocks are analysed only where its HORZ partition costs at most
//     (1 + theta) times the best of NONE, VERT and SPLIT (into square-only children) in a DP over the shapes done so far
//     (VERT likewise), after everything else or, with the top-down stages, right after their own group.
//
// Kept out of gp.h / kernels.h (and out of GpParams), so that the closed loop and the default analysis kernels do not
// change at all: only the analysis translation units, k_misc.cu and pipeline.cpp include this header.  The staged
// kernels (MODE 1 analysis, k_an_list, the DP stages) are a module of their own, k_an_x.cu, whose tables are uploaded
// only for an encoder with a switch on (gp_staged_prepare): without the switches a process never loads that code
// (lazy module loading, the default outside batch mode).
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
// The shapes a staged analysis lists (an_stage_of >= 2 with some switch): smaller side 8 or 4, and the 4:1 shapes.  Only
// they have MODE 1 kernels: every kernel in the binary is loaded at start-up in batch mode (eager module loading), and
// MODE 1 instances of every shape cost 80 ms there on an L4 with the switches off.  Of the shapes up to 64 pixels
// (k_an_a.cu) only the small-kernel versions have one: with --small-kernel 0 an_stage_of puts them in stage 1.
constexpr bool an_staged_shape(int w, int h) { return (w < h ? w : h) <= 8 || w == 4 * h || h == 4 * w; }

// Options of a DP stage (k_dp<BD, true>, k_dp.cuh): which shapes the DP may use, and what it writes.
struct GpDpStage {
    uint32_t shapeMask;     // AV1G_BLOCK_* bits: shapes analysed so far (the others count as GP_INF)
    uint8_t *core;          // non-null: per SB, one byte per cell: 1 where the covering leaf's smaller side <= coreSide
    int coreCell;           // 16 (16 bytes per SB, raster in the SB) or 8 (64 bytes)
    int coreSide;
    uint8_t *gate;          // non-null: per SB, 21 node bytes (64x64, the 4 32x32, the 16 16x16 in raster order): bit 0
                            // = analyse the node's n x n/4 blocks (HORZ_4), bit 1 its n/4 x n blocks (VERT_4)
    float gateMul;          // 1 + theta
};
#define GP_DP_GATE_NODES 21

// What k_an_list tests for one shape of a stage.
struct GpAnListShape {
    int bs;
    uint32_t total;         // blocks of the shape in the batch (shTotal)
    uint32_t *list, *count;
    int flat;               // apply the flat-area pruning (prune_flat)
    int cell;               // 16 / 8: alive only where a covered cell of that size is marked in core (or a 4-neighbour
                            // is, with neigh), or the frame cuts the 16x16 cell; 0: no cell test
    int gate;               // 1: alive only where the node's gate bit is set
};
#define GP_AN_LIST_MAX 8
struct GpAnListStage {
    int n;
    GpAnListShape s[GP_AN_LIST_MAX];
    const uint8_t *core16, *core8, *gate;
    int neigh;              // core cells also make their 4 neighbours alive (T3)
};

// Stage of a w x h shape: 1 (analysed first, every block); with the top-down stages 2 (smaller side 8) and 3 (side 4);
// the 4:1 shapes with the gating: 4 without the top-down stages (after everything else), else right after their own
// group, before the DP that maps the next group (5: 64x16 / 16x64 after 1; 6: 32x8 / 8x32 after 2; 7: 16x4 / 4x16
// after 3).  Order: 1, 5, 2, 6, 3, 7 (or 1, 4), a DP stage before each but the first.  smallKernel 0 (GpParams): the
// shapes of up to 64 pixels go through k_an_a's warp kernels, which have no MODE 1 version, so they stay in stage 1
// (every block analysed, as without the switches).
inline int an_stage_of(int topdown, float gateMul, int smallKernel, int w, int h) {
    if (!smallKernel && w * h <= 64) return 1;
    const int m = w < h ? w : h, g = m >= 16 ? 1 : m == 8 ? 2 : 3;
    if (gateMul > 0.f && (w == 4 * h || h == 4 * w)) return topdown ? 4 + g : 4;
    return topdown ? g : 1;
}

// One batch's staged analysis (pipeline.cpp fills it when GpConfig asks for any of it): the switches and the device
// buffers.
struct GpAnRun {
    int topdown;            // 1: top-down refinement
    float gateMul;          // > 0: 4:1 gating, 1 + theta
    uint32_t *list[22];     // per shape of stages 2..7: [shTotal] (GP_NSHAPES)
    uint32_t *count;        // [GP_NSHAPES] (zeroed by gp_launch_analysis_staged)
    uint8_t *core16, *core8, *gate;   // [nSb * 16], [nSb * 64], [nSb * GP_DP_GATE_NODES]
};

#endif
