// k_pack with the tx blocks' symbol counts for the device entropy path (k_misc.cu).  (Declared here and not in
// kernels.h, which every k_recon translation unit includes.)
#ifndef GP_PACK_H
#define GP_PACK_H
#include "kernels.h"

// gp_launch_pack, and with itemCount (lossy batches) the symbol count of each tx block at its item index (t + b + 1 for
// tx block t of block b, batch-global) for av1g_gpu::DevBatch::itemCount; the blocks' items are left to the symbolizer
void gp_launch_pack_counts(const GpBatch &B, const GpPackOff *dOff, av1g_block *oB, uint32_t *oBT, av1g_txb *oT, int16_t *oC,
                           const GpPackAux &aux, cudaStream_t st, av1g_palette *oP, uint8_t *oM, uint32_t *itemCount, int reducedTxSet);
#endif
