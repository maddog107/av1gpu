// modload.c: how long the driver takes to load each kernel module (cubin) of av1gpu on this GPU.
//   modload CUBIN...
// Prints the context creation time, then per cubin its size, the cuModuleLoad time and the time to load all its
// functions (cuModuleEnumerateFunctions + cuFuncLoad) in ms; two passes (the second with the file in the page cache
// and the driver warm).  Under CUDA_MODULE_LOADING=LAZY (the default) cuModuleLoad defers the functions' code, so the
// two columns split a module's data from its code; under EAGER the functions are loaded by cuModuleLoad.
#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <time.h>

static double now_ms(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

int main(int argc, char **argv) {
    double t0 = now_ms();
    CUdevice d; CUcontext c;
    if (cuInit(0) || cuDeviceGet(&d, 0) || cuDevicePrimaryCtxRetain(&c, d) || cuCtxSetCurrent(c)) { puts("init failed"); return 1; }
    cuCtxSynchronize();
    printf("context %.1f ms (CUDA_MODULE_LOADING=%s)\n", now_ms() - t0, getenv("CUDA_MODULE_LOADING") ? getenv("CUDA_MODULE_LOADING") : "");
    for (int pass = 0; pass < 2; pass++) {
        double tot = 0, totf = 0, bytes = 0;
        for (int i = 1; i < argc; i++) {
            struct stat st; stat(argv[i], &st);
            CUmodule m;
            double a = now_ms();
            CUresult r = cuModuleLoad(&m, argv[i]);
            cuCtxSynchronize();
            double dt = now_ms() - a, df = 0;
            unsigned int nf = 0;
            if (!r) {
                cuModuleGetFunctionCount(&nf, m);
                CUfunction *fs = (CUfunction *)malloc(sizeof(CUfunction) * (nf ? nf : 1));
                double b = now_ms();
                cuModuleEnumerateFunctions(fs, nf, m);
                for (unsigned k = 0; k < nf; k++) cuFuncLoad(fs[k]);
                cuCtxSynchronize();
                df = now_ms() - b;
                free(fs);
            }
            printf("pass %d %-40s %9.1f KB %3u functions load %8.1f ms functions %8.1f ms%s\n", pass, argv[i], st.st_size / 1024.0,
                   nf, dt, df, r ? " LOAD FAILED" : "");
            tot += dt; totf += df; bytes += st.st_size;
            if (!r) cuModuleUnload(m);
        }
        printf("pass %d total %.1f MB load %.1f ms functions %.1f ms\n", pass, bytes / 1048576.0, tot, totf);
    }
    return 0;
}
