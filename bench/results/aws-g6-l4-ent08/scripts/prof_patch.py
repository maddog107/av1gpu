#!/usr/bin/env python3
# k_entropy_win per-tile cycle profile (AV1G_ENTROPY_PROFILE) printed by run_batch (single-file path); experiment only
import sys
T = sys.argv[1]
p = f"{T}/av1core/src/gpu_entropy.cu"
s = open(p).read()
s = s.replace('#include "gpu_entropy.h"\n', '#define AV1G_ENTROPY_PROFILE 1\n#include "gpu_entropy.h"\n', 1)
old = "    cudaEventRecord(ed1);\n    CK(cudaEventSynchronize(ed1));\n"
assert s.count(old) == 1
s = s.replace(old, old + r'''    {
        static unsigned long long hp[4096][8];
        cudaMemcpyFromSymbol(hp, g_prof, sizeof hp);
        std::vector<uint64_t> b2(r.nTiles), e2(r.nTiles);
        cudaMemcpy(b2.data(), r.symBegin, 8ull * r.nTiles, cudaMemcpyDeviceToHost);
        cudaMemcpy(e2.data(), r.symEnd, 8ull * r.nTiles, cudaMemcpyDeviceToHost);
        for (uint32_t t = 0; t < r.nTiles && t < 4096; t++) {
            const double n = (double)(e2[t] - b2[t]);
            fprintf(stderr, "PROF tile %u syms %.0f cyc %.0f cyc/sym %.1f A/sym %.1f B/sym %.1f C/sym %.1f maxgrp/win %.2f wins %llu flush/sym %.3f\n", t, n,
                    (double)hp[t][0], hp[t][0] / n, hp[t][1] / n, hp[t][2] / n, hp[t][6] / n, (double)hp[t][3] / (hp[t][4] ? hp[t][4] : 1),
                    hp[t][4], hp[t][5] / n);
        }
    }
''')
open(p, "w").write(s)
print("patched")
