// Staged analysis (an_stage.h): which blocks of a stage's shapes are analysed (k_an_list, k_an_a.cu).  Device code that
// the research CPU emulation also compiles, so the rule it costs is this one.
#pragma once
#include "dev_common.cuh"
#include "an_stage.h"

// core value of the cell (cx, cy) of size c (16 or 8) of image I: the per-SB maps of the DP stages (k_dp<BD, true>)
__device__ __forceinline__ bool an_core_at(const GpImage &I, const uint8_t *core, int c, int cx, int cy) {
    const int px = cx * c, py = cy * c, cpr = 64 / c;
    const uint32_t sb = I.sbBase + (uint32_t)((py >> 6) * I.sbCols + (px >> 6));
    return core[(size_t)sb * cpr * cpr + ((py & 63) / c) * cpr + (px & 63) / c] != 0;
}

// the cell (cx, cy) of size c is to be refined: the frame cuts its 16x16 cell, or the DP marked it (or, with neigh, one
// of its 4 neighbours inside the image's SB grid)
__device__ __forceinline__ bool an_cell_ok(const GpImage &I, const uint8_t *core, int c, int neigh, int cx, int cy) {
    const int x16 = (cx * c) & ~15, y16 = (cy * c) & ~15;
    if (x16 + 16 > I.MiCols * 4 || y16 + 16 > I.MiRows * 4) return true;
    if (an_core_at(I, core, c, cx, cy)) return true;
    if (!neigh) return false;
    const int ncx = I.sbCols * (64 / c), ncy = I.sbRows * (64 / c);
    return (cx > 0 && an_core_at(I, core, c, cx - 1, cy)) || (cx + 1 < ncx && an_core_at(I, core, c, cx + 1, cy)) ||
           (cy > 0 && an_core_at(I, core, c, cx, cy - 1)) || (cy + 1 < ncy && an_core_at(I, core, c, cx, cy + 1));
}

// block (x, y) of shape sh.bs (inside the frame) is analysed in this stage
template <int BD>
__device__ __forceinline__ bool an_list_alive(const GpParams &P, const GpAnListStage &L, const GpAnListShape &sh, const GpImage &I,
                                              int x, int y) {
    const int w = bw_px(sh.bs), h = bh_px(sh.bs);
    if (sh.flat && prune_flat<BD>(I, P, x, y, w, h, sb_qindex(I, x, y))) return false;
    if (sh.cell) {
        const uint8_t *core = sh.cell == 16 ? L.core16 : L.core8;
        bool ok = false;
        for (int yy = y; yy < y + h && !ok; yy += sh.cell)
            for (int xx = x; xx < x + w && !ok; xx += sh.cell) ok = an_cell_ok(I, core, sh.cell, L.neigh, xx / sh.cell, yy / sh.cell);
        if (!ok) return false;
    }
    if (sh.gate) {
        // the node of a 4:1 block: n x n/4 (bit 0) at (x, y rounded down to n), n/4 x n (bit 1) at (x rounded down, y)
        const bool hz = w > h;
        const int n = hz ? w : h, nx0 = hz ? x : x & ~(n - 1), ny0 = hz ? y & ~(n - 1) : y;
        const int lv = n == 64 ? 0 : n == 32 ? 1 : 2, per = 1 << lv, base = lv == 0 ? 0 : lv == 1 ? 1 : 5;
        const uint32_t sb = I.sbBase + (uint32_t)((ny0 >> 6) * I.sbCols + (nx0 >> 6));
        const int node = base + ((ny0 & 63) / n) * per + (nx0 & 63) / n;
        if (!(L.gate[(size_t)sb * GP_DP_GATE_NODES + node] & (hz ? 1 : 2))) return false;
    }
    return true;
}
