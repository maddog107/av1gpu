// av1core: GPU deblocking + CDEF on the reconstruction, bit-exact with the decoder.
// The kernels are thin wrappers around the shared host/device code in av1g/postfilter.h, instantiated per
// bit depth (f.bitDepth: 8-bit planes, or uint16_t planes at 10 bits).
#include "av1g/gpu_postfilter.h"

#include <cuda_runtime.h>

namespace {

size_t align256(size_t x) { return (x + 255) & ~(size_t)255; }

struct ScratchLayout {
    size_t skip, txY, txU, txV, cdef, dirvar, total;
};
ScratchLayout layout(const av1g_frame_params *fp) {
    av1g_pf_frame f;
    av1g_pf_frame_init(fp, &f);
    const size_t nmi = (size_t)f.MiRows * f.MiCols, nuv = (size_t)(f.MiRows >> f.ssy) * (f.MiCols >> f.ssx);
    const size_t n8 = (size_t)(f.MiRows / 2) * (f.MiCols / 2);   // 8x8 luma blocks
    ScratchLayout L;
    size_t o = 0;
    L.skip = o; o = align256(o + nmi);
    L.txY = o; o = align256(o + nmi);
    L.txU = o; o = align256(o + nuv);
    L.txV = o; o = align256(o + nuv);
    L.cdef = o; o = align256(o + (size_t)f.sbCols * f.sbRows);
    L.dirvar = o; o = align256(o + 4 * n8);   // one int32 per 8x8 luma block
    L.total = o;
    return L;
}

__global__ void k_mark(av1g_pf_frame f, const av1g_block *blocks, uint32_t n) {
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) av1g_pf_mark_block(&f, &blocks[i]);
}

// One thread per (plane, 4x4 edge position); blockIdx.z = plane.  Chroma positions are in
// chroma 4x4 units (luma row/col = index << ssy / ssx: 2 * index in 4:2:0, the same in 4:4:4).
template <int BD>
__global__ void k_deblock(av1g_pf_frame f, int pass) {
    const int plane = blockIdx.z;
    const int subX = plane ? f.ssx : 0, subY = plane ? f.ssy : 0;
    const int col = blockIdx.x * blockDim.x + threadIdx.x, row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= (f.MiCols >> subX) || row >= (f.MiRows >> subY)) return;
    av1g_lf_edge<BD>(&f, plane, pass, row << subY, col << subX);
}

template <int BD>
__global__ void k_cdef_dir(av1g_pf_frame f) {
    const int bx = blockIdx.x * blockDim.x + threadIdx.x, by = blockIdx.y * blockDim.y + threadIdx.y;
    const int n8c = f.MiCols >> 1;
    if (bx >= n8c || by >= (f.MiRows >> 1)) return;
    int yDir = 0, var = 0;
    if (av1g_cdef_active_idx(&f, by * 2, bx * 2) >= 0)
        av1g_cdef_direction<BD>((const av1g_pixel_t(BD) *)f.plane[0] + (size_t)(by * 8) * f.stride[0] + bx * 8, f.stride[0], &yDir,
                                &var);
    f.cdefDirVar[(size_t)by * n8c + bx] = (var << 3) | yDir;
}

// One thread per sample of one plane (blockIdx.z = plane: no luma/chroma divergence).
struct OutPlanes {
    uint8_t *p[3];
    int s[3];
};
template <int BD>
__global__ void k_cdef(av1g_pf_frame f, OutPlanes o) {
    typedef av1g_pixel_t(BD) P;
    const int plane = blockIdx.z, subX = plane ? f.ssx : 0, subY = plane ? f.ssy : 0;
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= (f.MiCols * 4) >> subX || y >= (f.MiRows * 4) >> subY) return;
    const int bx = x >> (3 - subX), by = y >> (3 - subY);  // 8x8 luma block index
    const int idx = av1g_cdef_active_idx(&f, by * 2, bx * 2);
    const P *src = (const P *)f.plane[plane];
    const int ss = f.stride[plane];
    int v = src[(size_t)y * ss + x];
    if (idx >= 0) {
        const int dv = f.cdefDirVar[(size_t)by * (f.MiCols >> 1) + bx];
        const av1g_cdef_blk b = av1g_cdef_params<BD>(&f, idx, dv & 7, dv >> 3);
        // damping: CdefDamping + (BitDepth - 8), minus 1 for chroma (av1g_cdef_damping)
        v = plane == 0 ? av1g_cdef_pixel<BD>(&f, src, ss, 0, x, y, b.yPri, b.ySec, f.cdefDamping + (BD - 8), b.yDir)
                       : av1g_cdef_pixel<BD>(&f, src, ss, plane, x, y, b.uvPri, b.uvSec, f.cdefDamping + (BD - 8) - 1, b.uvDir);
    }
    ((P *)o.p[plane])[(size_t)y * o.s[plane] + x] = (P)v;
}

}  // namespace

extern "C" size_t av1g_gpu_pf_scratch_size(const av1g_frame_params *fp) { return layout(fp).total; }

extern "C" int av1g_gpu_pf_prepare(const av1g_frame_params *fp, const av1g_block *d_blocks, uint32_t n_blocks, void *d_scratch,
                                   av1g_pf_frame *f, cudaStream_t stream) {
    const ScratchLayout L = layout(fp);
    *f = av1g_pf_frame();
    av1g_pf_frame_init(fp, f);
    uint8_t *base = (uint8_t *)d_scratch;
    f->skip = base + L.skip;
    f->lfTx[0] = base + L.txY;
    f->lfTx[1] = base + L.txU;
    f->lfTx[2] = base + L.txV;
    f->cdefIdx = (int8_t *)(base + L.cdef);
    f->cdefDirVar = (int32_t *)(base + L.dirvar);
    cudaError_t e = cudaMemsetAsync(f->cdefIdx, 0xFF, (size_t)f->sbCols * f->sbRows, stream);
    if (e != cudaSuccess) return (int)e;
    if (n_blocks) k_mark<<<(n_blocks + 127) / 128, 128, 0, stream>>>(*f, d_blocks, n_blocks);
    return (int)cudaGetLastError();
}

extern "C" int av1g_gpu_deblock(const av1g_pf_frame *f, cudaStream_t stream) {
    if (!f->lfLevel[0] && !f->lfLevel[1]) return 0;
    const dim3 blk(32, 4);
    const dim3 grid((f->MiCols + 31) / 32, (f->MiRows + 3) / 4, f->mono ? 1 : 3);
    for (int pass = 0; pass < 2; pass++) {
        if (f->bitDepth == 10) k_deblock<10><<<grid, blk, 0, stream>>>(*f, pass);
        else k_deblock<8><<<grid, blk, 0, stream>>>(*f, pass);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) return (int)e;
    }
    return 0;
}

extern "C" int av1g_gpu_cdef_dir(const av1g_pf_frame *f, cudaStream_t stream) {
    const dim3 blk(16, 8);
    const dim3 grid(((f->MiCols >> 1) + 15) / 16, ((f->MiRows >> 1) + 7) / 8);
    if (f->bitDepth == 10) k_cdef_dir<10><<<grid, blk, 0, stream>>>(*f);
    else k_cdef_dir<8><<<grid, blk, 0, stream>>>(*f);
    return (int)cudaGetLastError();
}

extern "C" int av1g_gpu_cdef(const av1g_pf_frame *f, uint8_t *const d_out[3], const int out_stride[3], cudaStream_t stream) {
    OutPlanes o;
    for (int p = 0; p < 3; p++) {
        o.p[p] = d_out[p];
        o.s[p] = out_stride[p];
    }
    const dim3 blk(32, 8);
    const dim3 grid((f->MiCols * 4 + 31) / 32, (f->MiRows * 4 + 7) / 8, f->mono ? 1 : 3);
    if (f->bitDepth == 10) k_cdef<10><<<grid, blk, 0, stream>>>(*f, o);
    else k_cdef<8><<<grid, blk, 0, stream>>>(*f, o);
    return (int)cudaGetLastError();
}
