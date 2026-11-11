// host-callable kernel launchers (implemented in kernels.cu).
#ifndef GP_KERNELS_H
#define GP_KERNELS_H

#include <cuda_runtime.h>
#include <stdint.h>
#include "gp.h"

// Fill the device tables (forward transform matrices, distortion scales, symbol costs).  Call once
// per process after the CUDA device is selected.  The closed loop's tables are left to gp_recon_prepare.
int gp_init_tables(char *err, int errlen);

// Pack offsets per SB (computed on the host in decode order).
struct GpPackOff {
    uint32_t gBlk;      // global block index of the SB's first block
    uint32_t gTxb;      // global txb index of the SB's first txb
    uint64_t gCoef;     // global coefficient index
    uint32_t rTxb;      // image-relative txb index (block_txb values)
    uint32_t rCoef;     // image-relative coefficient offset (txb.coeff_off)
    uint32_t img, tile; // image index, global tile index (entropy auxiliaries)
    uint32_t gPal;      // global palette index of the SB's first palette
    uint32_t rBlk;      // image-relative index of the SB's first block (palette block fields)
    uint64_t gMap;      // global colour index map offset
    uint64_t rMap;      // image-relative colour index map offset (palette map_off)
};
// optional auxiliary outputs of k_pack for the device entropy path (entropy_dev.h)
struct GpPackAux {
    uint32_t *blockFrame, *blockTile, *blockTxbG, *txbBlock;
    uint64_t *txbCoeff;
};

// bd: bit depth of the batch (8 or 10): the kernels are instantiated per bit depth.  The images' sample planes
// (GpImage::src / rec) are uint16_t at 10 bits.
// live: the batch's live 16x16 cells (gp_launch_live16; all zero without the flat-area pruning)
void gp_launch_analysis(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st, int bd, const GpLive *live);
// the live 16x16 cells of B into L (nothing when L.cell is null), after the image upload and before gp_launch_analysis
// (k_an_a.cu)
void gp_launch_live16(const GpBatch &B, const GpParams &P, const GpLive &L, cudaStream_t st, int bd = 8);
// palette candidates (k_pal.cu) of the screen-content frames, after gp_launch_analysis; nothing when B.oPal is null
void gp_launch_pal_analysis(const GpBatch &B, const GpParams &P, const uint32_t *shTotal, cudaStream_t st, int bd = 8);
// y: luma plane (bd = 10: uint16_t samples, pitch in samples)
void gp_launch_ssim_map(const uint8_t *y, int pitch, int MiCols, int MiRows, float *out, double *sum, cudaStream_t st, int bd = 8);
void gp_launch_var16(const uint8_t *y, int pitch, int MiCols, int MiRows, float *out, cudaStream_t st, int bd = 8);
void gp_launch_dp(const GpBatch &B, const GpParams &P, cudaStream_t st, int bd = 8);
// 8-bit plane -> 10-bit plane (full-range rescaling, see k_misc.cu); chroma = Cb / Cr (512-centred)
void gp_launch_to10(const uint8_t *src, int spitch, uint16_t *dst, int dpitch, int w, int h, int chroma, cudaStream_t st);
// one plane of gp_frames_transform: source columns [sx0, sx0 + tw) filtered vertically into tmp (oh x tw floats; with
// alpha 3 planes of them: plain, alpha-weighted colour, alpha), then horizontally into dst (ow x oh, edge-replicated to
// aw x ah).  in10 / out10: uint16_t samples (10-bit).  alpha (may be NULL): the frame's alpha plane (its own pitch and
// depth, the colour's), as = 1 for 4:2:0 chroma (2x2 alpha means)
void gp_launch_resize(const void *src, int spitch, int in10, int sx0, int tw, const int *ystart, const float *yw, int ytaps,
                      const int *xstart, const float *xw, int xtaps, float *tmp, void *dst, int dpitch, int ow, int oh, int aw,
                      int ah, int out10, int chroma, const void *alpha, int apitch, int as, cudaStream_t st);
// one wavefront step: the first n420 SBs of dSbList belong to 4:2:0 / monochrome frames, the next n444 to 4:4:4 frames
// (monochrome frames of batches with 4:4:4 frames may be in the second group)
void gp_launch_recon(const GpBatch &B, const GpParams &P, const uint32_t *dSbList, uint32_t n420, uint32_t n444, cudaStream_t st,
                     int bd = 8);
// Upload the tables of the k_recon instantiations for bd with 4:2:0 / 4:4:4 groups, once per process each, before
// their first launch (after gp_init_tables).  Each instantiation is its own kernel module, 39 MB of sm_120 code: with
// lazy module loading (the default outside batch mode) a single image loads only the one it uses.  0, or -1.
int gp_recon_prepare(int bd, bool k420, bool k444);
// open-loop shortlists for the closed loop (GpParams::uvShortlist / txShortlist) into B.uvShort / B.txShort (whichever is
// allocated), after the partition DP and before the wavefront (k_leafprep.cu; its scratch is stream-ordered): 0, or a
// cudaError_t.  gp_leafprep_prepare uploads that module's tables, once per process, before its first launch: 0, or -1
int gp_launch_leafprep(const GpBatch &B, const GpParams &P, cudaStream_t st, int bd = 8);
int gp_leafprep_prepare();
void gp_launch_pack(const GpBatch &B, const GpPackOff *dOff, av1g_block *oB, uint32_t *oBT, av1g_txb *oT, int16_t *oC,
                    const GpPackAux &aux, cudaStream_t st, av1g_palette *oP = nullptr, uint8_t *oM = nullptr);
// screen-content detection (libaom's anti-aliasing-aware mode 2 on 16x16 luma blocks, k_misc.cu): sets dImg[i].sct (the
// device copy of the batch's images, hImg its host copy with scBlk0 / W / H) and *any if some image passes.  cnt:
// device scratch of 3 counters per image.  No host synchronisation (GP_SC_DEBUG=1: prints the counts, synchronising)
void gp_launch_sc_detect(GpImage *dImg, const GpImage *hImg, int n, int bd, unsigned long long *cnt, uint32_t *any, cudaStream_t st);
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
