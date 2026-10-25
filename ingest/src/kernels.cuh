// GPU kernels of the ingest: dequant + islow IDCT, resample/pad/orient, activity remap.
#pragma once
#include <cstdint>
#include <cuda_runtime.h>

namespace ingest {

// One native (as-coded) component plane.  Blocks of a plane are contiguous in the batch
// coefficient buffer, planes are sorted by blk0.
struct PlaneDesc {
  long long blk0;      // first block (64 x int16) index in the batch coefficient buffer
  long long nat_off;   // byte offset of the native plane in the native buffer
  long long act_off;   // signed byte offset of the float4 activity map from act_base (if has_act)
  int wb, hb;          // blocks
  int qt;              // dequant table index (64 x uint16 each)
  int nat_pitch;       // bytes, multiple of 8
  int act_pitch;       // float4 elements per row
  int has_act;
};

// One output plane (Y, Cb or Cr of an image).  Work is split in 64x16 pixel tiles.
struct OutDesc {
  long long out_off;   // byte offset in the pool
  long long tile0;     // first global tile index
  int out_pitch, out_w, out_h;  // fill region (aw x ah or aw/2 x ah/2)
  int tiles_x;
  int chroma;          // 0: direct (luma / mono), 1: 2x2 box average to 4:2:0
  int src;             // PlaneDesc index of the native source component
  int W, H;            // source visible size (luma units, stored orientation)
  int Wo, Ho;          // output visible size (after orientation)
  int hs, vs, hmax, vmax;  // sampling factors of the source component / maxima
  int orient;          // EXIF orientation applied (1 = none)
  int pad_;
};

struct ActDesc {       // remap of an activity map into the oriented block grid
  long long src_off, dst_off;   // bytes (src in native buffer, dst in pool)
  long long t0;                 // first global element index
  int src_w, src_h, dst_w, dst_h;
  int W, H, Wo, Ho, orient, pad_;
};

// Activity from decoded luma pixels (hardware-decoded images have no coefficients): orthonormal
// float 8x8 DCT of each native luma block (edge-replicated beyond the visible area), same 4 stats
// as the coefficient path; "nnz" counts |F| >= q/2 with the image's luma quant table.
struct ActPixDesc {
  long long nat_off;   // native luma plane
  long long act_off;   // signed byte offset from act_base
  long long blk0;      // first global block index (prefix)
  int wb, hb, nat_pitch, act_pitch;
  int vis_w, vis_h;    // visible size of the luma component
  int qt, pad_;
};

void launch_act_pixels(const ActPixDesc *d, int n, long long total, const uint16_t *qts, const uint8_t *native,
                       uint8_t *act_base, cudaStream_t s);
cudaError_t kernels_warmup();
void launch_idct(const short *coef, long long total_blocks, const PlaneDesc *planes, int nplanes,
                 const uint16_t *qts, uint8_t *native, uint8_t *act_base, cudaStream_t s);
void launch_finalize(const OutDesc *outs, int nouts, long long total_tiles, const PlaneDesc *planes,
                     const uint8_t *native, uint8_t *pool, cudaStream_t s);
void launch_act_remap(const ActDesc *d, int n, long long total, const uint8_t *native, uint8_t *pool,
                      cudaStream_t s);

}  // namespace ingest
