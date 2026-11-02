#include <cstdio>
__global__ void k_small(float *a, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float s = 0.f;
  for (int j = 0; j < n; j++) s += __sinf(a[(i + j) & 1023] * j) * __expf(-j * 0.001f);
  a[i & 1023] = s;
}
int main() {
  float *a; cudaMalloc(&a, 4096 * 4);
  k_small<<<4, 256>>>(a, 100);
  cudaError_t e = cudaDeviceSynchronize();
  printf("%s\n", cudaGetErrorString(e));
  return e != cudaSuccess;
}
