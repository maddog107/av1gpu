// host-callable kernel launchers (implemented in kernels.cu).
#ifndef GP_KERNELS_H
#define GP_KERNELS_H

#include <cuda_runtime.h>
#include <stdint.h>
#include "gp.h"

// Fill the device tables (forward transform matrices, distortion scales, symbol costs).  Call once
// per process after the CUDA device is selected.
int gp_init_tables(char *err, int errlen);

// Pack offsets per SB (computed on the host in decode order).
struct GpPackOff {
    uint32_t gBlk;      // global block index of the SB's first block
    uint32_t gTxb;      // global txb index of the SB's first txb
    uint64_t gCoef;     // global coefficient index
    uint32_t rTxb;      // image-relative txb index (block_txb values)
    uint32_t rCoef;     // image-relative coefficient offset (txb.coeff_off)
    uint32_t img, tile; // image index, global tile index (entropy auxiliaries)
};
// optional auxiliary outputs of k_pack for the device entropy path (entropy_dev.h)
struct GpPackAux {
    uint32_t *blockFrame, *blockTile, *blockTxbG, *txbBlock;
    uint64_t *txbCoeff;
};

// bd: bit depth of the batch (8 or 10): the kernels are instantiated per bit depth.  The images' sample planes
// (GpImage::src / rec) are uint16_t at 10 bits.
void gp_launch_analysis(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st, int bd = 8);
// y: luma plane (bd = 10: uint16_t samples, pitch in samples)
void gp_launch_ssim_map(const uint8_t *y, int pitch, int MiCols, int MiRows, float *out, double *sum, cudaStream_t st, int bd = 8);
void gp_launch_var16(const uint8_t *y, int pitch, int MiCols, int MiRows, float *out, cudaStream_t st, int bd = 8);
void gp_launch_dp(const GpBatch &B, const GpParams &P, cudaStream_t st, int bd = 8);
// 8-bit plane -> 10-bit plane (full-range rescaling, see k_misc.cu); chroma = Cb / Cr (512-centred)
void gp_launch_to10(const uint8_t *src, int spitch, uint16_t *dst, int dpitch, int w, int h, int chroma, cudaStream_t st);
// one wavefront step: the first n420 SBs of dSbList belong to 4:2:0 / monochrome frames, the next n444 to 4:4:4 frames
// (monochrome frames of batches with 4:4:4 frames may be in the second group)
void gp_launch_recon(const GpBatch &B, const GpParams &P, const uint32_t *dSbList, uint32_t n420, uint32_t n444, cudaStream_t st,
                     int bd = 8);
void gp_launch_pack(const GpBatch &B, const GpPackOff *dOff, av1g_block *oB, uint32_t *oBT, av1g_txb *oT, int16_t *oC,
                    const GpPackAux &aux, cudaStream_t st);
// cdef_idx of every block from the per-SB map (av1f search); frames with cdefOn[f] == 0 keep 0
void gp_launch_set_cdef(av1g_block *blocks, uint32_t nB, const uint32_t *blockFrame, const uint32_t *frameSbBase,
                        const int *frameSbCols, const int *cdefOn, const int8_t *sbCdef, cudaStream_t st);

// lossless (GpConfig::lossless, k_lossless.cu): cost tables + per-shape J (then gp_launch_dp), and the per-SB
// decisions + levels (no wavefront: the reconstruction is the source).  edgeFilter: enable_intra_edge_filter,
// fiOn: enable_filter_intra.
void gp_launch_ll_analysis(const GpBatch &B, const GpLossless &L, const uint32_t *shTotal, int edgeFilter, int fiOn, cudaStream_t st,
                           int bd);
void gp_launch_ll_recon(const GpBatch &B, const GpLossless &L, int edgeFilter, int fiOn, cudaStream_t st, int bd);

#endif
