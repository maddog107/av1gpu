// cubscan.cu: cub::DeviceScan::ExclusiveSum on uint32 arrays of several lengths, alone, as gpu_entropy.cu calls it
// (in place of av1gpu: is the Blackwell memcheck report CUB's?).  Checks the result against the CPU.
//   nvcc -O2 -arch=sm_120 cubscan.cu -o cubscan && compute-sanitizer --tool memcheck ./cubscan
#include <cub/cub.cuh>
#include <cstdio>
#include <vector>

int main() {
    const int ns[] = {2, 3, 17, 100, 129, 1000, 4097, 65537, 1000001};
    int bad = 0;
    for (int n : ns) {
        std::vector<uint32_t> h(n), o(n), ref(n);
        for (int i = 0; i < n; i++) h[i] = (i * 2654435761u) % 1000;
        uint32_t s = 0;
        for (int i = 0; i < n; i++) { ref[i] = s; s += h[i]; }
        uint32_t *dIn, *dOut; void *tmp = nullptr; size_t tb = 0;
        cudaMalloc(&dIn, n * 4); cudaMalloc(&dOut, n * 4);
        cudaMemcpy(dIn, h.data(), n * 4, cudaMemcpyHostToDevice);
        cub::DeviceScan::ExclusiveSum(nullptr, tb, dIn, dOut, n);
        cudaMalloc(&tmp, tb);
        cudaError_t e = cub::DeviceScan::ExclusiveSum(tmp, tb, dIn, dOut, n);
        cudaError_t e2 = cudaDeviceSynchronize();
        cudaMemcpy(o.data(), dOut, n * 4, cudaMemcpyDeviceToHost);
        int wrong = 0;
        for (int i = 0; i < n; i++) wrong += o[i] != ref[i];
        printf("n %8d temp %8zu B: launch %s, sync %s, %d wrong\n", n, tb, cudaGetErrorString(e), cudaGetErrorString(e2), wrong);
        bad += wrong || e || e2;
        cudaFree(dIn); cudaFree(dOut); cudaFree(tmp);
        if (e2) break;
    }
    printf("CUB %d.%d.%d, %s\n", CUB_MAJOR_VERSION, CUB_MINOR_VERSION, CUB_SUBMINOR_VERSION, bad ? "FAILED" : "all correct");
    return bad != 0;
}
