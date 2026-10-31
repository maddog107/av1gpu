// GPU kernels of the ingest.
//
// idct_kernel   : one thread per 8x8 block; dequantize + libjpeg-turbo jpeg_idct_islow
//                 (jidctint.c, CONST_BITS=13, PASS1_BITS=2, same range-limit table semantics),
//                 writes the native component plane; for luma also the activity map.
// finalize_kernel: native planes -> output planes: replicate visible edges, resample native
//                 sampling to 4:2:0 (2x2 box, libjpeg h2v2_downsample rounding bias 1,2,1,2..)
//                 or to 4:4:4 (libjpeg-turbo's decoder upsampling, jdsample.c), optional EXIF
//                 orientation.  4 output pixels per thread, 64x16 tiles.
// act_remap_kernel: activity map into the oriented block grid (only when rotating on GPU).
#include "kernels.cuh"

namespace ingest {

#define CONST_BITS 13
#define PASS1_BITS 2
#define FIX_0_298631336 2446
#define FIX_0_390180644 3196
#define FIX_0_541196100 4433
#define FIX_0_765366865 6270
#define FIX_0_899976223 7373
#define FIX_1_175875602 9633
#define FIX_1_501321110 12299
#define FIX_1_847759065 15137
#define FIX_1_961570560 16069
#define FIX_2_053119869 16819
#define FIX_2_562915447 20995
#define FIX_3_072711026 25172
#define DESCALE(x, n) (((x) + (1 << ((n)-1))) >> (n))

// libjpeg-turbo IDCT_range_limit(cinfo)[x & RANGE_MASK] for 8-bit samples.
__device__ __forceinline__ unsigned range_limit(int x) {
  int v = x & 1023;
  return v < 128 ? (unsigned)(v + 128) : (v < 512 ? 255u : (v < 896 ? 0u : (unsigned)(v - 896)));
}

// 1-D islow butterfly shared by both passes (inputs already dequantized / from workspace).
// Returns outputs before descaling in o[0..7] as (sum, diff) pairs per jidctint.c.
struct Islow8 {
  int tmp10, tmp11, tmp12, tmp13, tmp0, tmp1, tmp2, tmp3;
};

__device__ __forceinline__ Islow8 islow_1d(int i0, int i1, int i2, int i3, int i4, int i5, int i6, int i7) {
  Islow8 r;
  int z1, z2, z3, z4, z5, tmp0, tmp1, tmp2, tmp3;
  z2 = i2;
  z3 = i6;
  z1 = (z2 + z3) * FIX_0_541196100;
  tmp2 = z1 + z3 * (-FIX_1_847759065);
  tmp3 = z1 + z2 * FIX_0_765366865;
  tmp0 = (i0 + i4) * (1 << CONST_BITS);
  tmp1 = (i0 - i4) * (1 << CONST_BITS);
  r.tmp10 = tmp0 + tmp3;
  r.tmp13 = tmp0 - tmp3;
  r.tmp11 = tmp1 + tmp2;
  r.tmp12 = tmp1 - tmp2;
  tmp0 = i7;
  tmp1 = i5;
  tmp2 = i3;
  tmp3 = i1;
  z1 = tmp0 + tmp3;
  z2 = tmp1 + tmp2;
  z3 = tmp0 + tmp2;
  z4 = tmp1 + tmp3;
  z5 = (z3 + z4) * FIX_1_175875602;
  tmp0 = tmp0 * FIX_0_298631336;
  tmp1 = tmp1 * FIX_2_053119869;
  tmp2 = tmp2 * FIX_3_072711026;
  tmp3 = tmp3 * FIX_1_501321110;
  z1 = z1 * (-FIX_0_899976223);
  z2 = z2 * (-FIX_2_562915447);
  z3 = z3 * (-FIX_1_961570560);
  z4 = z4 * (-FIX_0_390180644);
  z3 += z5;
  z4 += z5;
  r.tmp0 = tmp0 + z1 + z3;
  r.tmp1 = tmp1 + z2 + z4;
  r.tmp2 = tmp2 + z2 + z3;
  r.tmp3 = tmp3 + z1 + z4;
  return r;
}

__global__ void __launch_bounds__(128) idct_kernel(const short *__restrict__ coef, long long total,
                                                   const PlaneDesc *__restrict__ planes, int nplanes,
                                                   const uint16_t *__restrict__ qts,
                                                   uint8_t *__restrict__ native, uint8_t *__restrict__ act_base) {
  long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= total) return;
  int lo = 0, hi = nplanes - 1;
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (planes[mid].blk0 <= g) lo = mid;
    else hi = mid - 1;
  }
  const PlaneDesc P = planes[lo];
  const int local = (int)(g - P.blk0);
  const int by = local / P.wb, bx = local - by * P.wb;
  const int4 *src = reinterpret_cast<const int4 *>(coef) + g * 8;
  const uint16_t *q = qts + (size_t)P.qt * 64;

  int ws[64];
  int nnz = 0;
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    int4 v = __ldg(src + r);
    const short *s = reinterpret_cast<const short *>(&v);
#pragma unroll
    for (int u = 0; u < 8; ++u) {
      int c = s[u];
      nnz += (c != 0);
      ws[r * 8 + u] = c * (int)__ldg(q + r * 8 + u);
    }
  }

  if (P.has_act) {
    float e_all = 0.f, e_h = 0.f, e_v = 0.f;
#pragma unroll
    for (int k = 1; k < 64; ++k) {
      float d = (float)ws[k];
      e_all += d * d;
    }
#pragma unroll
    for (int k = 1; k < 8; ++k) {
      float a = (float)ws[k], b = (float)ws[k * 8];
      e_h += a * a;
      e_v += b * b;
    }
    nnz -= (ws[0] != 0);
    float4 *am = reinterpret_cast<float4 *>(act_base + P.act_off);
    am[(size_t)by * P.act_pitch + bx] = make_float4(e_all * (1.f / 64.f), e_h * (1.f / 64.f),
                                                    e_v * (1.f / 64.f), (float)nnz);
  }

  // Pass 1: columns -> workspace (scaled by 2^PASS1_BITS)
#pragma unroll
  for (int u = 0; u < 8; ++u) {
    if (ws[8 + u] == 0 && ws[16 + u] == 0 && ws[24 + u] == 0 && ws[32 + u] == 0 && ws[40 + u] == 0 &&
        ws[48 + u] == 0 && ws[56 + u] == 0) {
      int dc = ws[u] * (1 << PASS1_BITS);
#pragma unroll
      for (int v = 0; v < 8; ++v) ws[v * 8 + u] = dc;
      continue;
    }
    Islow8 r = islow_1d(ws[u], ws[8 + u], ws[16 + u], ws[24 + u], ws[32 + u], ws[40 + u], ws[48 + u], ws[56 + u]);
    const int n = CONST_BITS - PASS1_BITS;
    ws[0 * 8 + u] = DESCALE(r.tmp10 + r.tmp3, n);
    ws[7 * 8 + u] = DESCALE(r.tmp10 - r.tmp3, n);
    ws[1 * 8 + u] = DESCALE(r.tmp11 + r.tmp2, n);
    ws[6 * 8 + u] = DESCALE(r.tmp11 - r.tmp2, n);
    ws[2 * 8 + u] = DESCALE(r.tmp12 + r.tmp1, n);
    ws[5 * 8 + u] = DESCALE(r.tmp12 - r.tmp1, n);
    ws[3 * 8 + u] = DESCALE(r.tmp13 + r.tmp0, n);
    ws[4 * 8 + u] = DESCALE(r.tmp13 - r.tmp0, n);
  }

  // Pass 2: rows -> samples
  uint8_t *out = native + P.nat_off + (size_t)(by * 8) * P.nat_pitch + bx * 8;
#pragma unroll
  for (int v = 0; v < 8; ++v) {
    const int *w = ws + v * 8;
    unsigned o[8];
    if (w[1] == 0 && w[2] == 0 && w[3] == 0 && w[4] == 0 && w[5] == 0 && w[6] == 0 && w[7] == 0) {
      unsigned dc = range_limit(DESCALE(w[0], PASS1_BITS + 3));
#pragma unroll
      for (int k = 0; k < 8; ++k) o[k] = dc;
    } else {
      Islow8 r = islow_1d(w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]);
      const int n = CONST_BITS + PASS1_BITS + 3;
      o[0] = range_limit(DESCALE(r.tmp10 + r.tmp3, n));
      o[7] = range_limit(DESCALE(r.tmp10 - r.tmp3, n));
      o[1] = range_limit(DESCALE(r.tmp11 + r.tmp2, n));
      o[6] = range_limit(DESCALE(r.tmp11 - r.tmp2, n));
      o[2] = range_limit(DESCALE(r.tmp12 + r.tmp1, n));
      o[5] = range_limit(DESCALE(r.tmp12 - r.tmp1, n));
      o[3] = range_limit(DESCALE(r.tmp13 + r.tmp0, n));
      o[4] = range_limit(DESCALE(r.tmp13 - r.tmp0, n));
    }
    uint2 pk;
    pk.x = o[0] | (o[1] << 8) | (o[2] << 16) | (o[3] << 24);
    pk.y = o[4] | (o[5] << 8) | (o[6] << 16) | (o[7] << 24);
    *reinterpret_cast<uint2 *>(out + (size_t)v * P.nat_pitch) = pk;
  }
}

// output (oriented) luma-space coordinate -> source (stored) coordinate
__device__ __forceinline__ void orient_map(int orient, int ox, int oy, int W, int H, int &sx, int &sy) {
  switch (orient) {
    default: sx = ox; sy = oy; break;
    case 2: sx = W - 1 - ox; sy = oy; break;
    case 3: sx = W - 1 - ox; sy = H - 1 - oy; break;
    case 4: sx = ox; sy = H - 1 - oy; break;
    case 5: sx = oy; sy = ox; break;
    case 6: sx = oy; sy = H - 1 - ox; break;
    case 7: sx = W - 1 - oy; sy = H - 1 - ox; break;
    case 8: sx = W - 1 - oy; sy = ox; break;
  }
}

__device__ __forceinline__ unsigned fetch(const OutDesc &O, const uint8_t *nat, int pitch, int ox, int oy) {
  ox = min(ox, O.Wo - 1);
  oy = min(oy, O.Ho - 1);
  int sx, sy;
  orient_map(O.orient, ox, oy, O.W, O.H, sx, sy);
  int nx = O.hs == O.hmax ? sx : sx * O.hs / O.hmax;
  int ny = O.vs == O.vmax ? sy : sy * O.vs / O.vmax;
  return __ldg(nat + (size_t)ny * pitch + nx);
}

// Full-resolution sample of a native component at output (oriented) luma position (ox, oy), as
// libjpeg-turbo's decoder produces it (jdsample.c, do_fancy_upsampling = TRUE, unscaled): fullsize
// copy; h2v1 / h2v2 triangle filters ("fancy", ordered-dither rounding biases, first / last column
// special cases) when the component is more than 2 samples wide, else replication; h1v2 triangle
// filter; replication (int_upsample) for the other integral ratios.  Vertical context beyond the
// component (jdmainct.c): the row above row 0 is row 0, rows below the last real row repeat it.
__device__ __forceinline__ unsigned fetch_full(const OutDesc &O, const uint8_t *nat, int pitch, int ox, int oy) {
  ox = min(ox, O.Wo - 1);
  oy = min(oy, O.Ho - 1);
  int sx, sy;
  orient_map(O.orient, ox, oy, O.W, O.H, sx, sy);
  const int hs = O.hs, vs = O.vs, hm = O.hmax, vm = O.vmax;
  auto px = [&](int x, int y) -> int { return __ldg(nat + (size_t)y * pitch + x); };
  if (hs == hm && vs == vm) return (unsigned)px(sx, sy);
  const int dw = (O.W * hs + hm - 1) / hm, dh = (O.H * vs + vm - 1) / vm;   // downsampled_width / height
  const bool h2 = hs * 2 == hm, v2 = vs * 2 == vm;
  if (h2 && vs == vm && dw > 2) {                    // h2v1_fancy_upsample
    const int c = sx >> 1, v = px(c, sy);
    if (sx & 1) return c == dw - 1 ? (unsigned)v : (unsigned)((v * 3 + px(c + 1, sy) + 2) >> 2);
    return c == 0 ? (unsigned)v : (unsigned)((v * 3 + px(c - 1, sy) + 1) >> 2);
  }
  if (hs == hm && v2) {                              // h1v2_fancy_upsample
    const int r = sy >> 1, r1 = (sy & 1) ? min(r + 1, dh - 1) : max(r - 1, 0);
    return (unsigned)((px(sx, r) * 3 + px(sx, r1) + ((sy & 1) ? 2 : 1)) >> 2);
  }
  if (h2 && v2 && dw > 2) {                          // h2v2_fancy_upsample
    const int r = sy >> 1, r1 = (sy & 1) ? min(r + 1, dh - 1) : max(r - 1, 0), c = sx >> 1;
    auto colsum = [&](int k) { return px(k, r) * 3 + px(k, r1); };
    const int t = colsum(c);
    if (sx & 1) return c == dw - 1 ? (unsigned)((t * 4 + 7) >> 4) : (unsigned)((t * 3 + colsum(c + 1) + 7) >> 4);
    return c == 0 ? (unsigned)((t * 4 + 8) >> 4) : (unsigned)((t * 3 + colsum(c - 1) + 8) >> 4);
  }
  return (unsigned)px(sx * hs / hm, sy * vs / vm);   // h2v1 / h2v2 / int_upsample replication
}

__global__ void __launch_bounds__(256) finalize_kernel(const OutDesc *__restrict__ outs, int nouts,
                                                       const PlaneDesc *__restrict__ planes,
                                                       const uint8_t *__restrict__ native, uint8_t *__restrict__ pool) {
  const long long t = blockIdx.x;
  int lo = 0, hi = nouts - 1;
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (outs[mid].tile0 <= t) lo = mid;
    else hi = mid - 1;
  }
  const OutDesc O = outs[lo];
  const int lt = (int)(t - O.tile0);
  const int tyi = lt / O.tiles_x, txi = lt - tyi * O.tiles_x;
  const int x0 = txi * 64 + (threadIdx.x & 15) * 4;
  const int y = tyi * 16 + (threadIdx.x >> 4);
  if (x0 >= O.out_w || y >= O.out_h) return;
  const PlaneDesc P = planes[O.src];
  const uint8_t *nat = native + P.nat_off;
  const int np = P.nat_pitch;
  unsigned packed = 0;
  if (!O.chroma) {
#pragma unroll
    for (int i = 0; i < 4; ++i) packed |= fetch(O, nat, np, x0 + i, y) << (8 * i);
  } else if (O.chroma == 2) {
#pragma unroll
    for (int i = 0; i < 4; ++i) packed |= fetch_full(O, nat, np, x0 + i, y) << (8 * i);
  } else {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const int x = x0 + i;
      unsigned s = fetch(O, nat, np, 2 * x, 2 * y) + fetch(O, nat, np, 2 * x + 1, 2 * y) +
                   fetch(O, nat, np, 2 * x, 2 * y + 1) + fetch(O, nat, np, 2 * x + 1, 2 * y + 1);
      packed |= ((s + ((x & 1) ? 2u : 1u)) >> 2) << (8 * i);
    }
  }
  *reinterpret_cast<unsigned *>(pool + O.out_off + (size_t)y * O.out_pitch + x0) = packed;
}

__global__ void act_remap_kernel(const ActDesc *__restrict__ ds, int n, long long total,
                                 const uint8_t *__restrict__ native, uint8_t *__restrict__ pool) {
  long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= total) return;
  int lo = 0, hi = n - 1;
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (ds[mid].t0 <= g) lo = mid;
    else hi = mid - 1;
  }
  const ActDesc D = ds[lo];
  const int k = (int)(g - D.t0);
  const int by = k / D.dst_w, bx = k - by * D.dst_w;
  int ox = min(bx * 8 + 4, D.Wo - 1), oy = min(by * 8 + 4, D.Ho - 1), sx, sy;
  orient_map(D.orient, ox, oy, D.W, D.H, sx, sy);
  const int sbx = min(sx >> 3, D.src_w - 1), sby = min(sy >> 3, D.src_h - 1);
  float4 a = reinterpret_cast<const float4 *>(native + D.src_off)[(size_t)sby * D.src_w + sbx];
  if (D.orient >= 5) {  // transposing orientations swap horizontal / vertical frequency energy
    float t = a.y;
    a.y = a.z;
    a.z = t;
  }
  reinterpret_cast<float4 *>(pool + D.dst_off)[k] = a;
}

__constant__ float kDct8[8][8];  // kDct8[k][n] = c_k cos((2n+1) k pi / 16), orthonormal

__global__ void __launch_bounds__(128) act_pixel_kernel(const ActPixDesc *__restrict__ ds, int n, long long total,
                                                        const uint16_t *__restrict__ qts,
                                                        const uint8_t *__restrict__ native, uint8_t *__restrict__ act_base) {
  long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= total) return;
  int lo = 0, hi = n - 1;
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (ds[mid].blk0 <= g) lo = mid;
    else hi = mid - 1;
  }
  const ActPixDesc D = ds[lo];
  const int k = (int)(g - D.blk0);
  const int by = k / D.wb, bx = k - by * D.wb;
  const uint8_t *src = native + D.nat_off;
  float t[64];
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    const int y = min(by * 8 + r, D.vis_h - 1);
    float x[8];
#pragma unroll
    for (int c = 0; c < 8; ++c) x[c] = (float)src[(size_t)y * D.nat_pitch + min(bx * 8 + c, D.vis_w - 1)];
#pragma unroll
    for (int u = 0; u < 8; ++u) {  // row DCT
      float a = 0.f;
#pragma unroll
      for (int c = 0; c < 8; ++c) a += kDct8[u][c] * x[c];
      t[r * 8 + u] = a;
    }
  }
  const uint16_t *q = qts + (size_t)D.qt * 64;
  float e_all = 0.f, e_h = 0.f, e_v = 0.f, nnz = 0.f;
#pragma unroll
  for (int u = 0; u < 8; ++u) {  // column DCT
#pragma unroll
    for (int v = 0; v < 8; ++v) {
      float a = 0.f;
#pragma unroll
      for (int r = 0; r < 8; ++r) a += kDct8[v][r] * t[r * 8 + u];
      if (u | v) {
        const float a2 = a * a;
        e_all += a2;
        if (v == 0) e_h += a2;
        if (u == 0) e_v += a2;
        nnz += (fabsf(a) * 2.f >= (float)__ldg(q + v * 8 + u)) ? 1.f : 0.f;
      }
    }
  }
  float4 *am = reinterpret_cast<float4 *>(act_base + D.act_off);
  am[(size_t)by * D.act_pitch + bx] = make_float4(e_all * (1.f / 64.f), e_h * (1.f / 64.f), e_v * (1.f / 64.f), nnz);
}

void launch_act_pixels(const ActPixDesc *d, int n, long long total, const uint16_t *qts, const uint8_t *native,
                       uint8_t *act_base, cudaStream_t s) {
  if (total <= 0 || n <= 0) return;
  act_pixel_kernel<<<(unsigned)((total + 127) / 128), 128, 0, s>>>(d, n, total, qts, native, act_base);
}

// Eager module load + constant upload, called from ingest_create.  With CUDA lazy loading the
// first launch / cudaMemcpyToSymbol would otherwise load the module mid-run, which takes the driver
// lock and stalled a co-running encoder's launches for ~200 ms (measured with tools/loadgen).
cudaError_t kernels_warmup() {
  float m[8][8];
  for (int k = 0; k < 8; ++k)
    for (int i = 0; i < 8; ++i) m[k][i] = (float)((k == 0 ? sqrt(0.125) : 0.5) * cos((2 * i + 1) * k * M_PI / 16.0));
  cudaError_t e = cudaMemcpyToSymbol(kDct8, m, sizeof(m));
  cudaFuncAttributes a;
  if (e == cudaSuccess) e = cudaFuncGetAttributes(&a, idct_kernel);
  if (e == cudaSuccess) e = cudaFuncGetAttributes(&a, finalize_kernel);
  if (e == cudaSuccess) e = cudaFuncGetAttributes(&a, act_pixel_kernel);
  if (e == cudaSuccess) e = cudaFuncGetAttributes(&a, act_remap_kernel);
  return e;
}

void launch_idct(const short *coef, long long total_blocks, const PlaneDesc *planes, int nplanes,
                 const uint16_t *qts, uint8_t *native, uint8_t *act_base, cudaStream_t s) {
  if (total_blocks <= 0) return;
  const int tpb = 128;
  idct_kernel<<<(unsigned)((total_blocks + tpb - 1) / tpb), tpb, 0, s>>>(coef, total_blocks, planes, nplanes,
                                                                         qts, native, act_base);
}

void launch_finalize(const OutDesc *outs, int nouts, long long total_tiles, const PlaneDesc *planes,
                     const uint8_t *native, uint8_t *pool, cudaStream_t s) {
  if (total_tiles <= 0) return;
  finalize_kernel<<<(unsigned)total_tiles, 256, 0, s>>>(outs, nouts, planes, native, pool);
}

void launch_act_remap(const ActDesc *d, int n, long long total, const uint8_t *native, uint8_t *pool,
                      cudaStream_t s) {
  if (total <= 0 || n <= 0) return;
  act_remap_kernel<<<(unsigned)((total + 255) / 256), 256, 0, s>>>(d, n, total, native, pool);
}

}  // namespace ingest
