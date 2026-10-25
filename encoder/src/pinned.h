// per-thread pinned staging for device-to-host downloads.
// A cudaMemcpyAsync into PAGEABLE host memory is synchronous and (measured with nsys, 2026-10-02)
// keeps the driver busy while it waits for its stream, blocking other threads' launches/frees: the
// encoder, entropy and ingest threads serialised on it.  Downloads go into a pinned arena instead:
// stage() enqueues the copy, flush() synchronises the stream (blocking sync, no lock held) and copies
// every staged block to its destination.
#ifndef GP_PINNED_H
#define GP_PINNED_H
#include <cuda_runtime.h>
#include <cstring>
#include <vector>

struct GpPinnedStage {
    struct Item { void *dst; size_t off, n; };
    uint8_t *p = nullptr;
    size_t cap = 0, used = 0;
    std::vector<Item> items;
    // outgrown buffers are kept until thread exit: cudaFreeHost synchronises the whole device (nsys: 0.4-0.5 s
    // calls mid-run that drained the GPU while other threads waited on the driver)
    std::vector<uint8_t *> retired;
    ~GpPinnedStage() {
        if (p) cudaFreeHost(p);
        for (uint8_t *q : retired) cudaFreeHost(q);
    }
    // returns non-zero on CUDA error
    int stage(void *dst, const void *src, size_t n, cudaStream_t st) {
        if (!n) return 0;
        const size_t need = ((used + 255) & ~(size_t)255) + n;
        if (need > cap) {
            if (int e = flush(st)) return e;
            size_t nc = cap ? cap : (size_t)4 << 20;
            while (nc < n + 256) nc *= 2;
            if (p) retired.push_back(p);
            p = nullptr; cap = 0;
            if (cudaHostAlloc((void **)&p, nc, cudaHostAllocDefault) != cudaSuccess) return 1;
            cap = nc;
        }
        used = (used + 255) & ~(size_t)255;
        if (cudaMemcpyAsync(p + used, src, n, cudaMemcpyDeviceToHost, st) != cudaSuccess) return 1;
        items.push_back({dst, used, n});
        used += n;
        return 0;
    }
    int stage2d(void *dst, size_t dpitch, const void *src, size_t spitch, size_t w, size_t h, cudaStream_t st) {
        // packed rows (dpitch == w) only: one staged block
        if (dpitch != w) return 1;
        if (!w || !h) return 0;
        const size_t n = w * h;
        const size_t need = ((used + 255) & ~(size_t)255) + n;
        if (need > cap) {
            if (int e = flush(st)) return e;
            size_t nc = cap ? cap : (size_t)4 << 20;
            while (nc < n + 256) nc *= 2;
            if (p) retired.push_back(p);
            p = nullptr; cap = 0;
            if (cudaHostAlloc((void **)&p, nc, cudaHostAllocDefault) != cudaSuccess) return 1;
            cap = nc;
        }
        used = (used + 255) & ~(size_t)255;
        if (cudaMemcpy2DAsync(p + used, w, src, spitch, w, h, cudaMemcpyDeviceToHost, st) != cudaSuccess) return 1;
        items.push_back({dst, used, n});
        used += n;
        return 0;
    }
    int flush(cudaStream_t st) {
        if (items.empty()) { used = 0; return 0; }
        if (cudaStreamSynchronize(st) != cudaSuccess) return 1;
        for (const Item &it : items) memcpy(it.dst, p + it.off, it.n);
        items.clear();
        used = 0;
        return 0;
    }
};

inline GpPinnedStage &gp_pinned() {
    static thread_local GpPinnedStage s;
    return s;
}
#endif
