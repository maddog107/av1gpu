#!/usr/bin/env python3
"""instrument.py TREE: temporary start-up timestamps (stderr "[tt] label ms", CLOCK_MONOTONIC) in a copy of the source;
never committed.  The launcher (ttrun.py) prints its own CLOCK_MONOTONIC before the exec."""
import re, sys
T = sys.argv[1]
TT = ('#include <time.h>\n#include <stdio.h>\nstatic inline void tt_(const char *l) { timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); '
      'fprintf(stderr, "[tt] %s %.3f\\n", l, ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6); }\n')
def patch(path, subs, head=True):
    s = open(path).read()
    if head:
        s = TT + s
    for a, b in subs:
        if a not in s:
            sys.exit(f"{path}: not found: {a[:60]}")
        s = s.replace(a, b, 1)
    open(path, "w").write(s)
patch(f"{T}/encoder/apps/av1gpu.cpp", [
    ("    GpConfig cfg;\n    Opts o;\n", "    tt_(\"main\");\n    atexit([] { tt_(\"atexit\"); });\n    GpConfig cfg;\n    Opts o;\n"),
    ("    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);\n    if (cudaSetDevice(0)", "    tt_(\"before setdevice\");\n    cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);\n    if (cudaSetDevice(0)"),
    ("#endif\n    if (!o.batch.empty()) {\n        if (o.debugOutputs)", "#endif\n    tt_(\"after setdevice\");\n    if (!o.batch.empty()) {\n        if (o.debugOutputs)"),
    ("    GpEncoder enc;\n    if (enc.init(cfg, err)) { fprintf(stderr, \"init: %s\\n\", err.c_str()); return 1; }\n    GpFrameIn fin, afin;",
     "    GpEncoder enc;\n    tt_(\"enc.init start\");\n    if (enc.init(cfg, err)) { fprintf(stderr, \"init: %s\\n\", err.c_str()); return 1; }\n    tt_(\"enc.init done\");\n    GpFrameIn fin, afin;"),
    ("        ingest_submit(ictx, paths, 1);", "        tt_(\"ingest_create done\");\n        ingest_submit(ictx, paths, 1);"),
    ("        if (r != 1) {\n            ingest_fallback fb;", "        tt_(\"ingest_next_batch done\");\n        if (r != 1) {\n            ingest_fallback fb;"),
    ("    auto t1 = std::chrono::steady_clock::now();\n    GpEncoder encS;", "    auto t1 = std::chrono::steady_clock::now();\n    tt_(\"t1\");\n    GpEncoder encS;"),
    ("    if (E->encode({fin}, outs, tm, err)) return failed(\"encode\");", "    tt_(\"encoders ready\");\n    if (E->encode({fin}, outs, tm, err)) return failed(\"encode\");\n    tt_(\"encode done\");"),
    ("    auto t2 = std::chrono::steady_clock::now();\n    GpFrameOut &f = outs[0];", "    auto t2 = std::chrono::steady_clock::now();\n    tt_(\"ingest destroyed, buffers freed\");\n    GpFrameOut &f = outs[0];"),
    ("    auto t3 = std::chrono::steady_clock::now();\n    if (!gp_write_file", "    auto t3 = std::chrono::steady_clock::now();\n    tt_(\"entropy done\");\n    if (!gp_write_file"),
    ("    if (cfg.P.profile) {\n        const char *nm[9]", "    tt_(\"written\");\n    if (cfg.P.profile) {\n        const char *nm[9]"),
])
patch(f"{T}/ingest/src/pipeline.cu", [
    ("  if (cudaFree(nullptr) != cudaSuccess) return fail(\"CUDA context creation failed (GPU full?)\");\n  if (kernels_warmup() != cudaSuccess) return fail(\"loading the ingest kernels failed\");",
     "  tt_(\"ic: start\");\n  if (cudaFree(nullptr) != cudaSuccess) return fail(\"CUDA context creation failed (GPU full?)\");\n  if (kernels_warmup() != cudaSuccess) return fail(\"loading the ingest kernels failed\");\n  tt_(\"ic: kernels_warmup\");"),
    ("  cudaMemPool_t mp;\n  if (cudaDeviceGetDefaultMemPool(&mp, k.device) == cudaSuccess) {\n    uint64_t thr = 1ull << 30;",
     "  tt_(\"ic: nvjpeg done\");\n  cudaMemPool_t mp;\n  if (cudaDeviceGetDefaultMemPool(&mp, k.device) == cudaSuccess) {\n    uint64_t thr = 1ull << 30;"),
    ("  if (!c->arena.init((size_t)k.pinned_bytes)) return fail(\"cudaHostAlloc of the pinned arena failed\");",
     "  if (!c->arena.init((size_t)k.pinned_bytes)) return fail(\"cudaHostAlloc of the pinned arena failed\");\n  tt_(\"ic: arena\");"),
    ("void ingest_destroy(ingest_ctx *c) {\n  if (!c) return;", "void ingest_destroy(ingest_ctx *c) {\n  if (!c) return;\n  tt_(\"ingest_destroy start\");"),
])
patch(f"{T}/encoder/src/pipeline.cpp", [
    ("    if (gp_init_tables(buf, sizeof buf)) { err = buf; return -1; }", "    tt_(\"  init: start\");\n    if (gp_init_tables(buf, sizeof buf)) { err = buf; return -1; }\n    tt_(\"  init: tables\");"),
    ("        GPCK(cudaDeviceSynchronize());\n    }\n    // keep freed stream-ordered", "        GPCK(cudaDeviceSynchronize());\n        tt_(\"  init: vb luts\");\n    }\n    // keep freed stream-ordered"),
])
print("instrumented", T)
