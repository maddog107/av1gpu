// av1core: GPU deblocking + CDEF on the reconstruction, bit-exact with the decoder.
// The kernels are thin wrappers around the shared host/device code in av1g/postfilter.h (CDEF: a 64x64 tile in
// shared memory around av1g_cdef_filter), instantiated per bit depth (f.bitDepth: 8-bit planes, or uint16_t
// planes at 10 bits).
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

struct OutPlanes {
    uint8_t *p[3];
    int s[3];
};
// One CTA per 64x64 (SS: chroma subsampling, 1 = 4:2:0 / monochrome, 0 = 4:4:4).  The SB's deblocked samples + a
// 2-sample halo are staged in shared memory (int16, -1 = outside the frame), the per-8x8 parameters and the tap
// offsets of every direction once; then one thread per sample (av1g_cdef_filter, the same arithmetic as
// av1g_cdef_pixel).  An SB without CDEF is copied.
template <int BD, int SS>
__global__ void __launch_bounds__(256) k_cdef(av1g_pf_frame f, OutPlanes o) {
    typedef av1g_pixel_t(BD) P;
    constexpr int TL = 68, CN = 64 >> SS, TC = CN + 4, CPX = CN * CN;
    __shared__ int16_t tl[TL * TL];
    __shared__ int16_t tc[2][TC * TC];
    __shared__ int16_t off[2][8][12];   // tap offsets per direction in the luma / chroma tile
    __shared__ int16_t bp[64][6];       // per 8x8: yPri, ySec, yDir, uvPri, uvSec, uvDir; yPri -1 = not filtered
    const int sb = blockIdx.x, sbr = sb / f.sbCols, sbc = sb % f.sbCols, tid = threadIdx.x;
    const int np = f.mono ? 1 : 3;
    const int lw = f.MiCols * 4, lh = f.MiRows * 4, cw = lw >> SS, ch = lh >> SS;
    // plane pointers by select (an indexed parameter array would be copied to local memory)
    auto src = [&](int p) { return (const P *)(p == 0 ? f.plane[0] : p == 1 ? f.plane[1] : f.plane[2]); };
    auto sst = [&](int p) { return p == 0 ? f.stride[0] : p == 1 ? f.stride[1] : f.stride[2]; };
    auto dst = [&](int p) { return (P *)(p == 0 ? o.p[0] : p == 1 ? o.p[1] : o.p[2]); };
    auto dst_s = [&](int p) { return p == 0 ? o.s[0] : p == 1 ? o.s[1] : o.s[2]; };
    if (f.cdefIdx[sb] < 0) {
        for (int i = tid; i < 4096 + (np > 1 ? 2 * CPX : 0); i += 256) {
            const int p = i < 4096 ? 0 : 1 + (i - 4096) / CPX, j = i < 4096 ? i : (i - 4096) % CPX, n = p ? CN : 64;
            const int x = sbc * n + j % n, y = sbr * n + j / n;
            if (x >= (p ? cw : lw) || y >= (p ? ch : lh)) continue;
            dst(p)[(size_t)y * dst_s(p) + x] = src(p)[(size_t)y * sst(p) + x];
        }
        return;
    }
    {
        const int x0 = sbc * 64 - 2, y0 = sbr * 64 - 2;
        for (int i = tid; i < TL * TL; i += 256) {
            const int X = x0 + i % TL, Y = y0 + i / TL;
            tl[i] = X >= 0 && Y >= 0 && X < lw && Y < lh ? (int)src(0)[(size_t)Y * sst(0) + X] : -1;
        }
        if (np > 1) {
            const int cx0 = sbc * CN - 2, cy0 = sbr * CN - 2;
            for (int i = tid; i < 2 * TC * TC; i += 256) {
                const int pl = i / (TC * TC), j = i % (TC * TC);
                const int X = cx0 + j % TC, Y = cy0 + j / TC;
                tc[pl][j] = X >= 0 && Y >= 0 && X < cw && Y < ch ? (int)src(1 + pl)[(size_t)Y * sst(1 + pl) + X] : -1;
            }
        }
        if (tid < 192) {
            const int w = tid < 96 ? TL : TC, dir = (tid % 96) / 12, i = tid % 12, k = i / 6, sign = (i / 3) & 1 ? 1 : -1, j = i % 3;
            const int dd = j == 0 ? dir : (dir + (j == 1 ? -2 : 2)) & 7;
            off[tid / 96][dir][i] = (int16_t)(sign * (AV1G_T(Cdef_Directions)[dd][k][0] * w + AV1G_T(Cdef_Directions)[dd][k][1]));
        } else {
            const int b = tid - 192, by = sbr * 8 + (b >> 3), bx = sbc * 8 + (b & 7);
            const int idx = by * 2 < f.MiRows && bx * 2 < f.MiCols ? av1g_cdef_active_idx(&f, by * 2, bx * 2) : -1;
            if (idx >= 0) {
                const int dv = f.cdefDirVar[(size_t)by * (f.MiCols >> 1) + bx];
                const av1g_cdef_blk c = av1g_cdef_params<BD>(&f, idx, dv & 7, dv >> 3);
                bp[b][0] = (int16_t)c.yPri; bp[b][1] = (int16_t)c.ySec; bp[b][2] = (int16_t)c.yDir;
                bp[b][3] = (int16_t)c.uvPri; bp[b][4] = (int16_t)c.uvSec; bp[b][5] = (int16_t)c.uvDir;
            } else {
                bp[b][0] = -1;
            }
        }
    }
    __syncthreads();
    // one sample at SB-local (lx, ly) of plane p (n x n per SB, block b = its 8x8 luma block)
    auto px = [&](int p, int n, int sub, int lx, int ly) {
        const int x = sbc * n + lx, y = sbr * n + ly;
        if (x >= (p ? cw : lw) || y >= (p ? ch : lh)) return;
        const int b = ((ly << sub) >> 3) * 8 + ((lx << sub) >> 3);
        const int16_t *tile = p == 0 ? tl : tc[p - 1];
        const int c = (ly + 2) * (p ? TC : TL) + lx + 2;
        int v = tile[c];
        if (bp[b][0] >= 0) {
            const int16_t *o12 = off[p > 0][bp[b][p ? 5 : 2]];
            int t[12];
#pragma unroll
            for (int k = 0; k < 12; k++) t[k] = tile[c + o12[k]];
            // damping: CdefDamping + (BitDepth - 8), minus 1 for chroma (av1g_cdef_damping)
            v = av1g_cdef_filter<BD>(v, t, bp[b][p ? 3 : 0], bp[b][p ? 4 : 1], f.cdefDamping + (BD - 8) - (p > 0));
        }
        dst(p)[(size_t)y * dst_s(p) + x] = (P)v;
    };
    for (int i = tid; i < 4096; i += 256) px(0, 64, 0, i & 63, i >> 6);
    if (np > 1)
        for (int i = tid; i < 2 * CPX; i += 256) px(1 + (i >= CPX), CN, SS, i % CN, (i % CPX) / CN);
}

}  // namespace

extern "C" size_t av1g_gpu_pf_scratch_size(const av1g_frame_params *fp) { return layout(fp).total; }

extern "C" void av1g_gpu_pf_attach(const av1g_frame_params *fp, void *d_scratch, av1g_pf_frame *f) {
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
}

extern "C" int av1g_gpu_pf_prepare(const av1g_frame_params *fp, const av1g_block *d_blocks, uint32_t n_blocks, void *d_scratch,
                                   av1g_pf_frame *f, cudaStream_t stream) {
    av1g_gpu_pf_attach(fp, d_scratch, f);
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
    const unsigned nsb = (unsigned)(f->sbCols * f->sbRows);
    const bool s444 = !f->mono && !f->ssx;
    if (f->bitDepth == 10) {
        if (s444) k_cdef<10, 0><<<nsb, 256, 0, stream>>>(*f, o);
        else k_cdef<10, 1><<<nsb, 256, 0, stream>>>(*f, o);
    } else {
        if (s444) k_cdef<8, 0><<<nsb, 256, 0, stream>>>(*f, o);
        else k_cdef<8, 1><<<nsb, 256, 0, stream>>>(*f, o);
    }
    return (int)cudaGetLastError();
}
