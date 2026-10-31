/* av1g/gpu_postfilter.h - GPU deblocking + CDEF (spec 7.14 / 7.15), bit-exact with the
 * decoder (same av1g/postfilter.h code as the CPU backend).  All pointers are device
 * pointers; every call is asynchronous on `stream` and returns a cudaError_t value (0 = ok).
 *
 *   size_t n = av1g_gpu_pf_scratch_size(&fp);                  // maps + CDEF directions
 *   av1g_pf_frame f;
 *   av1g_gpu_pf_prepare(&fp, d_blocks, n_blocks, d_scratch, &f, stream);  // maps from blocks
 *   f.plane[p] = d_plane[p]; f.stride[p] = stride[p];          // pre-filter recon, padded:
 *                                                              // (MiCols*4 x MiRows*4) >> ss
 *   av1g_gpu_deblock(&f, stream);                              // in place
 *   av1g_gpu_cdef_dir(&f, stream);                             // directions/variance (any strengths)
 *   av1g_gpu_cdef(&f, d_out, out_stride, stream);              // CdefFrame (separate buffers)
 *
 * Strength searches: av1g_gpu_cdef_dir() depends only on the deblocked luma, so run it once
 * and call av1g_gpu_cdef() per candidate set (change f.cdefYPri[] etc. / f.cdefIdx).
 *
 * 10-bit frames (fp.bit_depth = 10, f.bitDepth): the planes and d_out are uint16_t sample buffers
 * passed as uint8_t pointers, strides in samples. */
#ifndef AV1G_GPU_POSTFILTER_H
#define AV1G_GPU_POSTFILTER_H

#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime_api.h>
#include "av1g/decisions.h"
#include "av1g/postfilter.h"

#ifdef __cplusplus
extern "C" {
#endif

size_t av1g_gpu_pf_scratch_size(const av1g_frame_params *fp);
/* f receives the frame parameters and the map pointers inside d_scratch (plane pointers zeroed) */
int av1g_gpu_pf_prepare(const av1g_frame_params *fp, const av1g_block *d_blocks, uint32_t n_blocks, void *d_scratch,
                        av1g_pf_frame *f, cudaStream_t stream);
int av1g_gpu_deblock(const av1g_pf_frame *f, cudaStream_t stream);
int av1g_gpu_cdef_dir(const av1g_pf_frame *f, cudaStream_t stream);
int av1g_gpu_cdef(const av1g_pf_frame *f, uint8_t *const d_out[3], const int out_stride[3], cudaStream_t stream);

#ifdef __cplusplus
}
#endif
#endif
