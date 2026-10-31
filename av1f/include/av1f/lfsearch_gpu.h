/* av1f/lfsearch_gpu.h - GPU loop-filter strength search (deblocking levels + CDEF presets and
 * per-64x64 cdef_idx) on top of av1core's bit-exact GPU filters (av1g/gpu_postfilter.h).
 *
 *   size_t n = av1f_lf_scratch_size(&fp, &opts);  cudaMalloc(&scratch, n);
 *   av1f_lf_search(&fp, d_blocks, n_blocks, d_recon, recon_stride, d_src, src_stride,
 *                  &opts, &res, d_sb_cdef_idx, scratch, stream);          // host-synchronising, timed
 *   av1f_lf_search_async(..., d_res, ...)                                   // no host sync, result in device memory
 *
 * Inputs (device): the record's blocks (skip / tx_depth / bsize decide edges and CDEF skip),
 * the PADDED pre-filter reconstruction (av1g_encode AV1G_ENC_PREFILTER_RECON: MiCols*4 x
 * MiRows*4, chroma >> ssx / ssy) and the source planes padded at least as far.  fp supplies width,
 * height, monochrome, chroma_format and base_q_idx.
 * On return: d_recon is DEBLOCKED with the chosen levels (CDEF not applied), res holds the frame
 * header values (deblocking levels/sharpness, CDEF damping/bits/presets with secondary strengths
 * as applied values 0/1/2/4), d_sb_cdef_idx[sbRows * sbCols] the preset of every 64x64
 * (-1 = all-skip SB).  Write them into av1g_frame_params / av1g_block.cdef_idx.
 *
 * Modes
 *   lf_mode   AV1F_LF_OFF | AV1F_LF_FROM_Q (libaom s6: av1_pick_filter_level LPF_PICK_FROM_Q, exact;
 *             recommended) | AV1F_LF_SEARCH (all levels 0..63, luma both directions, U and V
 *             separately; fused per-SB kernel, ~3 ms/MP, measured -0.1 % BD - not worth it)
 *   cdef_mode AV1F_CDEF_OFF | AV1F_CDEF_AOM (libaom s6 CDEF_FAST_SEARCH_LVL4 + tune=iq CDEF_ADAPTIVE,
 *             bit-exact incl. the RD choice of cdef_bits; ~0.3 ms/MP) | AV1F_CDEF_SEARCH (16 x 4 luma
 *             and 16 x 4 chroma strengths, one fused SSE pass, prune to cdef_prune strengths per
 *             plane, libaom's joint greedy preset search, no strength halving; ~1.8 ms/MP,
 *             -0.42 % BD vs libaom's choice)
 *   cdef_damping -2 additionally searches the damping (sync API only).
 * Distortion: SSE (8-bit samples, libaom) or a variance-masked SSE (AV1F_DIST_VARW, a cheap
 * SSIMULACRA2 proxy - not better than SSE for CDEF); a caller with its own perceptual per-SB
 * loss can choose presets with av1f_cdef_select_tables() below.
 */
#ifndef AV1F_LFSEARCH_GPU_H
#define AV1F_LFSEARCH_GPU_H

#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime_api.h>
#include "av1g/decisions.h"

#ifdef __cplusplus
extern "C" {
#endif

enum { AV1F_LF_OFF = 0, AV1F_LF_FROM_Q = 1, AV1F_LF_SEARCH = 2 };
enum { AV1F_CDEF_OFF = 0, AV1F_CDEF_AOM = 1, AV1F_CDEF_SEARCH = 2 };

typedef struct av1f_lf_opts {
    int lf_mode;
    int sharpness;          /* -1: tune=iq rule (7 / 1 / 0 for base <= 120 / <= 160 / above), else 0..7 */
    int cdef_mode;
    int cdef_max_bits;      /* 0..3 */
    float cdef_rd_scale;    /* multiplies libaom's frame rdmult in the cdef_bits RD decision (1 = libaom) */
    int cdef_refine;        /* SEARCH mode: refinement rounds per preset (libaom: 4) */
    int cdef_damping;       /* -1: libaom 3 + (base >> 6); 3..6 fixed; -2 (SEARCH mode): also search 3..6 by RD */
    int dist;               /* AV1F_DIST_SSE (libaom) | AV1F_DIST_VARW: SSIMULACRA2 proxy = SSE weighted per 8x8
                               by 1 / (source variance + dist_c), normalised to a frame mean of 1 */
    float dist_c;           /* AV1F_DIST_VARW constant (default 58.5 = SSIM's C2 for 8-bit) */
    int cdef_prune;         /* SEARCH mode: keep this many strengths per plane (greedy single-plane coverage, <= 16)
                               before the joint preset search; 0 = all 64 x 64 pairs */
} av1f_lf_opts;
enum { AV1F_DIST_SSE = 0, AV1F_DIST_VARW = 1,
       AV1F_DIST_GRAD = 2 /* deblocking only: SSE of the first differences of the error image (blocking-sensitive);
                             CDEF then uses SSE */ };

typedef struct av1f_lf_result {
    int lf_level[4];        /* effective levels (Y vert, Y horz, U, V) */
    int sharpness;
    int cdef_enable;        /* sequence-header enable_cdef (1 unless cdef_mode == OFF) */
    int cdef_damping, cdef_bits;
    int y_pri[8], y_sec[8], uv_pri[8], uv_sec[8];
    uint64_t lf_sse[3];     /* chosen deblocking SSE per plane (visible area) */
    uint64_t cdef_sse;      /* chosen CDEF total SSE over filtered blocks (search distortion) */
    float ms_lf, ms_cdef_dir, ms_cdef_sse, ms_cdef_select;  /* device time per stage (CUDA events) */
    int n_cand;             /* CDEF_SEARCH with pruning: the kept strength indices (pri * 4 + sec index) */
    int cand_y[16], cand_uv[16];
} av1f_lf_result;

size_t av1f_lf_scratch_size(const av1g_frame_params *fp, const av1f_lf_opts *o);
int av1f_lf_search(const av1g_frame_params *fp, const av1g_block *d_blocks, uint32_t n_blocks, uint8_t *const d_recon[3],
                   const int recon_stride[3], const uint8_t *const d_src[3], const int src_stride[3], const av1f_lf_opts *o,
                   av1f_lf_result *res, int8_t *d_sb_cdef_idx, void *d_scratch, cudaStream_t stream);

/* Same search, fully asynchronous: no host synchronisation, no timers; the result struct is written
 * to DEVICE memory (d_res, sizeof(av1f_lf_result)); copy it after your own sync.  Identical decisions
 * (tests/lfsearch_async_check via py/check_gpu_tools.py --async). */
int av1f_lf_search_async(const av1g_frame_params *fp, const av1g_block *d_blocks, uint32_t n_blocks, uint8_t *const d_recon[3],
                         const int recon_stride[3], const uint8_t *const d_src[3], const int src_stride[3], const av1f_lf_opts *o,
                         av1f_lf_result *d_res, int8_t *d_sb_cdef_idx, void *d_scratch, cudaStream_t stream);

/* Preset selection on caller-supplied per-SB cost tables (e.g. a perceptual loss instead of SSE):
 * d_my / d_muv: [nsb][nc] uint32 costs of nc luma / chroma strengths (any per-SB constant offset is
 * harmless; d_muv may be all 0 for monochrome), d_active[nsb] >= 0 for SBs that carry a cdef_idx.
 * Runs libaom's joint greedy search (+ refine rounds) for cdef_bits 0..max_bits after pruning each
 * plane to prune_k strengths (0 = all), then picks bits by rd = total cost + lambda * signalling bits.
 * Outputs strength INDICES (into the caller's nc candidates) and the per-SB preset index. */
typedef struct av1f_cdef_sel {
    int bits;
    int lev_y[8], lev_uv[8];
    double rd, cost;
} av1f_cdef_sel;
int av1f_cdef_select_tables(const uint32_t *d_my, const uint32_t *d_muv, int nc, int nsb, const int8_t *d_active, int nact, int mono,
                            int max_bits, int refine, int prune_k, double lambda, av1f_cdef_sel *out, int8_t *d_sb_idx,
                            cudaStream_t stream);

static inline av1f_lf_opts av1f_lf_opts_default(void) {
    av1f_lf_opts o = {AV1F_LF_FROM_Q, -1, AV1F_CDEF_AOM, 3, 1.0f, 4, -1, AV1F_DIST_SSE, 58.5f, 16};
    return o;
}

#ifdef __cplusplus
}
#endif
#endif
