/* av1g/hd.h - host/device portability macros shared by all av1core headers.
 *
 * AV1G_HD      marks functions callable from both host and CUDA device code.
 * AV1G_TABLE   defines a constant lookup table twice when compiled as CUDA: a host
 *              copy (name_host) and a device copy (name_dev).  Plain C/C++ builds
 *              only get the host copy.
 *              The device copy is not `const`: Clang/LLVM would otherwise put it in
 *              the 64 KB constant bank (.const), which these tables overflow.
 *              nvcc already emits non-__constant__ __device__ data in .global.
 * AV1G_T(n)    selects the right copy at the use site (device pass vs host pass).
 */
#ifndef AV1G_HD_H
#define AV1G_HD_H

#if defined(__CUDACC__)
#define AV1G_HD __host__ __device__ __forceinline__
#define AV1G_HDI __host__ __device__ inline
#define AV1G_TABLE(T, name, dims, ...)                 \
    static const T name##_host dims = __VA_ARGS__;     \
    static __device__ T name##_dev dims = __VA_ARGS__;
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

#endif
