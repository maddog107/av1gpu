/* av1f/varboost_gpu.h - GPU variance-boost delta-q (libaom tune=iq, deltaq-mode 6), batched.
 *
 * Output per 64x64 superblock (raster over the image's SB grid, sbCols = ceil(W8 / 64),
 * sbRows = ceil(H8 / 64), W8/H8 = size padded to 8):
 *   sb_var     libaom's SB variance statistic (octile-5 of the 64 8x8 variances, 1:2:1)
 *   sb_target  av1_get_sbq_variance_boost() qindex (before delta_q_res rounding)
 *   sb_qindex  the qindex the SB is coded with: legal for delta_q_res = av1f_delta_q_res(base),
 *              ready for av1g_block.sb_qindex (av1g_apply_sb_qindex); use it for that SB's
 *              quantizers and lambda.
 * Rounding order (order argument):
 *   AV1F_VB_TILE     libaom with one thread (avifenc -j 1, the reference anchor): dead-zone rounding
 *                    toward the previous SB, predictor reset at each tile start  -> bit-exact
 *   AV1F_VB_ROW      libaom with row multithreading (avifenc default -j all): reset at every SB
 *                    row of every tile -> bit-exact for that mode
 *   AV1F_VB_NEAREST  fully parallel nearest legal value (av1g_legalize_sb_qindex rule)
 * Input luma must be readable for the whole SB grid (ingest pads to 64 by edge replication); the
 * kernel uses 8-byte loads: luma pointer 8-byte aligned and pitch a multiple of 8 (ingest: 64).
 * 10-bit luma (bit_depth = 10): uint16_t samples, pitch in samples, 16-byte loads (16-byte aligned,
 * pitch a multiple of 8); variances as aom_highbd_10_variance8x8 (8-bit scale) and the 10-bit q tables
 * (LUTs from av1f_vb_build_luts_bd(..., 10, ...)); descriptors of one call share the bit depth.
 * VRAM: luts 256 KB (once) + 6 B per SB.  One kernel pass reads the luma plane once.
 */
#ifndef AV1F_VARBOOST_GPU_H
#define AV1F_VARBOOST_GPU_H

#include <stdint.h>
#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

enum { AV1F_VB_TILE = 0, AV1F_VB_ROW = 1, AV1F_VB_NEAREST = 2 };

typedef struct av1f_vb_desc {
    const uint8_t *y;          /* device luma (top-left of the picture) */
    int32_t pitch;
    int32_t width, height;     /* visible size */
    int32_t base_q_idx;        /* 1..255 */
    int32_t tile_cols_log2;    /* actual uniform tiling (av1g_tile_layout(...).cols_log2) */
    int32_t tile_rows_log2;
    uint32_t sb_off;           /* first SB of this image in the output arrays */
    int32_t bit_depth;         /* 8 or 10 (0 = 8): y holds uint16_t samples at 10 bits (appended field) */
} av1f_vb_desc;

/* SB count of a picture (for sizing / sb_off) */
static inline uint32_t av1f_vb_num_sbs(int width, int height) {
    const uint32_t c = (uint32_t)((((width + 7) >> 3) * 8 + 63) >> 6), r = (uint32_t)((((height + 7) >> 3) * 8 + 63) >> 6);
    return c * r;
}

/* d_luts: 256 x 1024 bytes, entry [base][var] = SB qindex for variance var < 1024 (8-bit / bit_depth q tables). */
int av1f_vb_build_luts(uint8_t *d_luts, int deltaq_strength_pct, cudaStream_t stream);
int av1f_vb_build_luts_bd(uint8_t *d_luts, int deltaq_strength_pct, int bit_depth, cudaStream_t stream);

/* n images (descriptors in DEVICE memory), total_sbs = sum of SB counts.  d_sb_var/d_sb_target may
 * be NULL.  Returns 0 or a cudaError_t. */
int av1f_varboost(const av1f_vb_desc *d_desc, int n, uint32_t total_sbs, const uint8_t *d_luts, int order,
                  uint32_t *d_sb_var, uint8_t *d_sb_target, uint8_t *d_sb_qindex, cudaStream_t stream);
/* bit_depth: of every descriptor (8 or 10); av1f_varboost = bit_depth 8 */
int av1f_varboost_bd(const av1f_vb_desc *d_desc, int n, uint32_t total_sbs, const uint8_t *d_luts, int order, int bit_depth,
                     uint32_t *d_sb_var, uint8_t *d_sb_target, uint8_t *d_sb_qindex, cudaStream_t stream);

/* Host convenience (tests / Python): one image from HOST memory; plane must cover the SB grid. */
int av1f_varboost_host(const uint8_t *y, int pitch, int width, int height, int base_q_idx, int tile_cols_log2,
                       int tile_rows_log2, int order, int deltaq_strength_pct, uint32_t *sb_var, uint8_t *sb_target,
                       uint8_t *sb_qindex);
/* the same with a bit depth: 10 -> y is uint16_t samples (pitch in samples) */
int av1f_varboost_host_bd(const void *y, int pitch, int width, int height, int base_q_idx, int tile_cols_log2,
                          int tile_rows_log2, int order, int deltaq_strength_pct, int bit_depth, uint32_t *sb_var,
                          uint8_t *sb_target, uint8_t *sb_qindex);

#ifdef __cplusplus
}
#endif
#endif
