// cubsweep.cu: the uint32 scan of gpu_entropy.cu:1734 alone, for every n in [N0, N1] (or the n given):
// the temp size is queried with a const input pointer type (as gpu_entropy.cu:1569 does), the scratch is a buffer of
// exactly that size at a 256-byte-aligned offset in a larger allocation (av1gpu's arena), and the scan runs with a
// non-const input pointer on a non-blocking stream: cub::DeviceScan::ExclusiveSum<unsigned*, unsigned*, int>.
// Stops at the first CUDA error or wrong result and prints that n.
//   cubsweep N0 N1 [STEP]
#include <cub/cub.cuh>
#include <cstdio>
#include <cstdlib>
#include <vector>

int main(int argc, char **argv) {
    const int n0 = argc > 1 ? atoi(argv[1]) : 1, n1 = argc > 2 ? atoi(argv[2]) : 100000, step = argc > 3 ? atoi(argv[3]) : 1;
    std::vector<uint32_t> h(n1), o(n1);
    for (int i = 0; i < n1; i++) h[i] = (i * 2654435761u) % 1000;
    uint32_t *dIn, *dOut; char *arena;
    cudaMalloc(&dIn, 4ull * n1); cudaMalloc(&dOut, 4ull * n1); cudaMalloc(&arena, 1 << 22);
    cudaMemcpy(dIn, h.data(), 4ull * n1, cudaMemcpyHostToDevice);
    cudaStream_t s; cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    size_t maxtb = 0; long tested = 0;
    for (int n = n0; n <= n1; n += step) {
        size_t tb = 0;
        cub::DeviceScan::ExclusiveSum(nullptr, tb, (const uint32_t *)nullptr, (uint32_t *)nullptr, n);
        if (tb > maxtb) maxtb = tb;
        void *tmp = arena + 256 * 7;   // an aligned sub-buffer of exactly tb bytes
        size_t bytes = tb;
        cudaError_t e = cub::DeviceScan::ExclusiveSum(tmp, bytes, dIn, dOut, n, s);
        cudaError_t e2 = cudaStreamSynchronize(s);
        tested++;
        if (e || e2) { printf("n %d: launch %s, sync %s\n", n, cudaGetErrorString(e), cudaGetErrorString(e2)); return 1; }
        if (n % 9973 == 0 || n == n1) {   // spot-check the result
            cudaMemcpy(o.data(), dOut, 4ull * n, cudaMemcpyDeviceToHost);
            uint32_t acc = 0; int wrong = 0;
            for (int i = 0; i < n; i++) { wrong += o[i] != acc; acc += h[i]; }
            if (wrong) { printf("n %d: %d wrong\n", n, wrong); return 1; }
        }
    }
    printf("CUB %d.%d.%d: n %d..%d step %d, %ld scans, no error (temp up to %zu B)\n", CUB_MAJOR_VERSION, CUB_MINOR_VERSION,
           CUB_SUBMINOR_VERSION, n0, n1, step, tested, maxtb);
    return 0;
}
