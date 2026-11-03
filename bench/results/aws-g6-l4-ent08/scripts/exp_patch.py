#!/usr/bin/env python3
# speed-of-light experiment (NOT byte-identical, never committed): env AV1X_SKIP_SEL=1 replaces k_select_small (lfsearch
# async path) by memsets; AV1X_SKIP_ENT=1 makes k_entropy_win write a 1-byte tile without coding.
import os, sys
T = sys.argv[1]
p = f"{T}/av1f/src/lfsearch.cu"
s = open(p).read()
old = "        k_select_small<<<1, 1024, 0, s>>>(smy, smuv, ncs, g.nsb, g.mono, lastBits, refine, fast_mono, lev, tots, cur);\n"
assert s.count(old) == 1
s = s.replace(old, "        if (getenv(\"AV1X_SKIP_SEL\")) { cudaMemsetAsync(lev, 0, 4 * 16 * sizeof(int), s); cudaMemsetAsync(tots, 0, 32, s); }\n"
              "        else\n" + old)
open(p, "w").write(s)
p = f"{T}/av1core/src/gpu_entropy.cu"
s = open(p).read()
old = "    extern __shared__ uint4 sh4[];\n    uint16_t *cdf = (uint16_t *)sh4;\n"
assert s.count(old) == 1
s = s.replace(old, "    if (g_skip_ent) { if (threadIdx.x == 0) sizeArr[blockIdx.x] = 1; return; }\n" + old)
old = "template <class Coder>\n__global__ void __launch_bounds__(64) k_entropy_win("
assert s.count(old) == 1
s = s.replace(old, "__device__ int g_skip_ent;\n" + old)
old = "int run_batch_device(const DevBatch &b, cudaStream_t s, void *scratch, size_t scratchBytes, DevResult &out, std::string &err,\n                     const DevEvents *ev) {\n    ensure_walk_stack();\n"
assert s.count(old) == 1
s = s.replace(old, old + "    { static std::once_flag o2; std::call_once(o2, [] { int v = getenv(\"AV1X_SKIP_ENT\") ? 1 : 0; cudaMemcpyToSymbol(g_skip_ent, &v, 4); }); }\n")
open(p, "w").write(s)
print("patched")
