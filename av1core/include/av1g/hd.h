/* av1g/hd.h - host/device portability macros shared by all av1core headers.
 *
 * AV1G_HD      marks functions callable from both host and CUDA device code.
 * AV1G_TABLE   defines a constant lookup table twice when compiled as CUDA: a host
 *              copy (name_host) and a device copy (name_dev).  Plain C/C++ builds
 *              only get the host copy.
 *              With Clang as the CUDA compiler the device copy is not `const`: Clang
 *              puts const __device__ data with a constant initializer in the 64 KB
 *              constant bank (.const), which these tables overflow (ptxas: "too much
 *              global constant data").  nvcc keeps them const in .global and reads
 *              them through the read-only data cache.
 * AV1G_T(n)    selects the right copy at the use site (device pass vs host pass).
 * AV1G_HOST_CLZ(x)  count of leading zero bits of a nonzero 32-bit x in host code (GCC / Clang builtin,
 *              MSVC intrinsic); device code uses __clz.
 */
#ifndef AV1G_HD_H
#define AV1G_HD_H

#if defined(__CUDACC__)
#define AV1G_HD __host__ __device__ __forceinline__
#define AV1G_HDI __host__ __device__ inline
#if defined(__clang__) && !defined(__NVCC__)
#define AV1G_DEV_CONST
#else
#define AV1G_DEV_CONST const
#endif
#define AV1G_TABLE(T, name, dims, ...)                 \
    static const T name##_host dims = __VA_ARGS__;     \
    static __device__ AV1G_DEV_CONST T name##_dev dims = __VA_ARGS__;
#else
#define AV1G_HD static inline
#define AV1G_HDI static inline
#define AV1G_TABLE(T, name, dims, ...) static const T name##_host dims = __VA_ARGS__;
#endif

#if defined(__CUDA_ARCH__)
#define AV1G_T(name) name##_dev
#else
#define AV1G_T(name) name##_host
#endif

#if defined(_MSC_VER) && !defined(__clang__)
#ifdef __cplusplus
extern "C" unsigned char _BitScanReverse(unsigned long *index, unsigned long mask);
#else
unsigned char _BitScanReverse(unsigned long *index, unsigned long mask);
#endif
#pragma intrinsic(_BitScanReverse)
static __inline int av1g_host_clz32(unsigned x) {
    unsigned long i;
    _BitScanReverse(&i, x);
    return 31 - (int)i;
}
#define AV1G_HOST_CLZ(x) av1g_host_clz32(x)
#else
#define AV1G_HOST_CLZ(x) __builtin_clz(x)
#endif

#endif
