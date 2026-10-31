// gpu_postfilter_test - GPU deblocking + CDEF (av1g/gpu_postfilter.h) against the decoder.
//
//   gpu_postfilter_test --seed S --count N [--yuv444 P]
//                                                     random records (fuzzer generator, ~half
//                                                     with deblocking and/or CDEF, P % of the
//                                                     colour cases 4:4:4): GPU output
//                                                     == dav1d output frame, and == the CPU
//                                                     av1g_postfilter on the padded area
//   gpu_postfilter_test --obu F.obu --yuv F.yuv [--iters K]
//                                                     a real stream (e.g. libaom's): parse,
//                                                     re-encode pre-filter, GPU post-filter,
//                                                     compare with the decoder's frame F.yuv;
//                                                     prints CUDA-event kernel timings
#define AV1G_FUZZ_NO_MAIN
#include "fuzz.cpp"
#include "av1g/gpu_postfilter.h"

#include <algorithm>
#include <cuda_runtime.h>

#define CKX(x) do { cudaError_t e_ = (cudaError_t)(x); if (e_ != cudaSuccess) { fprintf(stderr, "CUDA %s at %s\n", cudaGetErrorString(e_), #x); exit(3); } } while (0)

struct GpuRun {
    std::vector<uint8_t> out[3];  // padded CdefFrame (or deblocked frame when CDEF is off)
    int w[3] = {0, 0, 0}, h[3] = {0, 0, 0};
    double msMaps = 0, msDeblock = 0, msDir = 0, msCdef = 0;
};

// pre: padded pre-filter planes (av1g_encode with AV1G_ENC_PREFILTER_RECON)
static void gpu_postfilter(const av1g_decision_record &rec, const av1g_output &pre, int iters, GpuRun &g) {
    const int np = rec.fp.monochrome ? 1 : 3;
    av1g_block *dBlocks = nullptr;
    void *dScratch = nullptr;
    uint8_t *dIn[3] = {nullptr, nullptr, nullptr}, *dWork[3] = {nullptr, nullptr, nullptr}, *dOut[3] = {nullptr, nullptr, nullptr};
    int st[3] = {0, 0, 0};
    CKX(cudaMalloc(&dBlocks, sizeof(av1g_block) * std::max<uint32_t>(rec.n_blocks, 1)));
    CKX(cudaMemcpy(dBlocks, rec.blocks, sizeof(av1g_block) * rec.n_blocks, cudaMemcpyHostToDevice));
    const size_t ss = av1g_gpu_pf_scratch_size(&rec.fp);
    CKX(cudaMalloc(&dScratch, ss));
    for (int p = 0; p < np; p++) {
        g.w[p] = (int)pre.recon_w[p];
        g.h[p] = (int)pre.recon_h[p];
        st[p] = g.w[p];
        const size_t n = (size_t)g.w[p] * g.h[p];
        CKX(cudaMalloc(&dIn[p], n));
        CKX(cudaMalloc(&dWork[p], n));
        CKX(cudaMalloc(&dOut[p], n));
        CKX(cudaMemcpy(dIn[p], pre.recon[p], n, cudaMemcpyHostToDevice));
    }
    cudaEvent_t e[5];
    for (auto &x : e) CKX(cudaEventCreate(&x));
    std::vector<double> tm[4];
    av1g_pf_frame f;
    for (int it = 0; it < iters; it++) {
        for (int p = 0; p < np; p++) CKX(cudaMemcpyAsync(dWork[p], dIn[p], (size_t)g.w[p] * g.h[p], cudaMemcpyDeviceToDevice, 0));
        CKX(cudaEventRecord(e[0], 0));
        CKX(av1g_gpu_pf_prepare(&rec.fp, dBlocks, rec.n_blocks, dScratch, &f, 0));
        for (int p = 0; p < np; p++) { f.plane[p] = dWork[p]; f.stride[p] = st[p]; }
        CKX(cudaEventRecord(e[1], 0));
        CKX(av1g_gpu_deblock(&f, 0));
        CKX(cudaEventRecord(e[2], 0));
        if (f.enableCdef) {
            CKX(av1g_gpu_cdef_dir(&f, 0));
            CKX(cudaEventRecord(e[3], 0));
            CKX(av1g_gpu_cdef(&f, dOut, st, 0));
        } else {
            CKX(cudaEventRecord(e[3], 0));
        }
        CKX(cudaEventRecord(e[4], 0));
        CKX(cudaEventSynchronize(e[4]));
        for (int k = 0; k < 4; k++) {
            float ms = 0;
            CKX(cudaEventElapsedTime(&ms, e[k], e[k + 1]));
            tm[k].push_back(ms);
        }
    }
    auto med = [](std::vector<double> v) { std::sort(v.begin(), v.end()); return v[v.size() / 2]; };
    g.msMaps = med(tm[0]); g.msDeblock = med(tm[1]); g.msDir = med(tm[2]); g.msCdef = med(tm[3]);
    for (int p = 0; p < np; p++) {
        g.out[p].resize((size_t)g.w[p] * g.h[p]);
        CKX(cudaMemcpy(g.out[p].data(), f.enableCdef ? dOut[p] : dWork[p], g.out[p].size(), cudaMemcpyDeviceToHost));
        cudaFree(dIn[p]); cudaFree(dWork[p]); cudaFree(dOut[p]);
    }
    for (auto &x : e) cudaEventDestroy(x);
    cudaFree(dBlocks);
    cudaFree(dScratch);
}

// cropped GPU planes vs a decoder's I420 / I444 frame
static bool cmp_cropped(const GpuRun &g, const av1g_decision_record &rec, const std::vector<uint8_t> &yuv, std::string &why) {
    const int W = (int)rec.fp.width, H = (int)rec.fp.height, np = rec.fp.monochrome ? 1 : 3;
    const int ssx = av1g_ssx(&rec.fp), ssy = av1g_ssy(&rec.fp);
    size_t off = 0;
    for (int p = 0; p < np; p++) {
        const int w = p ? (W + ssx) >> ssx : W, h = p ? (H + ssy) >> ssy : H;
        for (int y = 0; y < h; y++) {
            if (off + (size_t)(y + 1) * w > yuv.size()) { why = "decoder frame too small"; return false; }
            const uint8_t *a = g.out[p].data() + (size_t)y * g.w[p], *b = yuv.data() + off + (size_t)y * w;
            if (memcmp(a, b, w)) {
                int x = 0;
                while (a[x] == b[x]) x++;
                char m[160];
                snprintf(m, sizeof m, "plane %d x %d y %d: gpu %d decoder %d", p, x, y, a[x], b[x]);
                why = m;
                return false;
            }
        }
        off += (size_t)w * h;
    }
    return true;
}

int main(int argc, char **argv) {
    uint64_t seed = 1;
    int count = 100, iters = 1, pct444 = 40;
    std::string obuPath, yuvPath, tmp = default_out_dir("av1g_gpu_pf");
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "--seed") seed = strtoull(nx().c_str(), 0, 10);
        else if (a == "--count") count = atoi(nx().c_str());
        else if (a == "--obu") obuPath = nx();
        else if (a == "--yuv") yuvPath = nx();
        else if (a == "--iters") iters = atoi(nx().c_str());
        else if (a == "--yuv444") pct444 = atoi(nx().c_str());
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    if (!obuPath.empty()) {  // ---- real stream ----
        std::vector<uint8_t> obu = read_file(obuPath), yuv = read_file(yuvPath);
        av1g_parsed *pr = nullptr;
        if (av1g_parse(obu.data(), obu.size(), &pr) != AV1G_OK) { printf("parse failed: %s\n", pr ? pr->error : "?"); return 1; }
        av1g_decision_record &rec = pr->rec;
        av1g_encode_opts eo = {AV1G_ENC_PREFILTER_RECON, -1};
        av1g_output pre;
        if (av1g_encode(&rec, &eo, &pre) != AV1G_OK) { printf("encode failed: %s\n", pre.error); return 1; }
        GpuRun g;
        gpu_postfilter(rec, pre, std::max(iters, 1), g);
        std::string why;
        const bool ok = cmp_cropped(g, rec, yuv, why);
        printf("%s %ux%u lf=%d/%d/%d/%d sharp=%d cdef=%d bits=%d damping=%d blocks=%u: GPU post-filter vs decoder frame: %s%s\n",
               obuPath.c_str(), rec.fp.width, rec.fp.height, rec.fp.loop_filter_level[0], rec.fp.loop_filter_level[1],
               rec.fp.loop_filter_level[2], rec.fp.loop_filter_level[3], rec.fp.loop_filter_sharpness, rec.fp.enable_cdef,
               rec.fp.cdef_bits, rec.fp.cdef_damping_minus_3 + 3, rec.n_blocks, ok ? "IDENTICAL" : "MISMATCH ", why.c_str());
        printf("GPU_PF_TIMING iters=%d median ms: maps %.4f deblock %.4f cdef_dir %.4f cdef %.4f total %.4f\n", iters, g.msMaps,
               g.msDeblock, g.msDir, g.msCdef, g.msMaps + g.msDeblock + g.msDir + g.msCdef);
        av1g_output_free(&pre);
        av1g_parsed_free(pr);
        return ok ? 0 : 1;
    }
    mkdir(tmp.c_str(), 0755);
    int bad = 0, withPf = 0, n444 = 0, pf444 = 0;
    for (int ci = 0; ci < count; ci++) {
        const uint64_t cs = seed + ci;
        FuzzCase fc;
        if (!make_case(cs, false, 300, 2, fc, pct444)) return 1;
        av1g_decision_record &rec = fc.rec;
        const bool pf = rec.fp.loop_filter_level[0] || rec.fp.loop_filter_level[1] || rec.fp.enable_cdef;
        const bool is444 = rec.fp.chroma_format == AV1G_CHROMA_444;
        withPf += pf;
        n444 += is444;
        pf444 += pf && is444;
        av1g_encode_opts e1 = {AV1G_ENC_FIX_RANGE | AV1G_ENC_TD, -1};
        av1g_output o;
        if (av1g_encode(&rec, &e1, &o) != AV1G_OK) { printf("seed %llu: encode error %s\n", (unsigned long long)cs, o.error); bad++; continue; }
        av1g_encode_opts e2 = {AV1G_ENC_PREFILTER_RECON, -1};
        av1g_output pre;
        if (av1g_encode(&rec, &e2, &pre) != AV1G_OK) { printf("seed %llu: encode error %s\n", (unsigned long long)cs, pre.error); bad++; continue; }
        GpuRun g;
        gpu_postfilter(rec, pre, 1, g);
        // decoder frame
        const std::string ob = tmp + "/c.obu", dy = tmp + "/c.yuv";
        write_file(ob, o.obu, o.obu_size);
        const std::string cmd = DAV1D + " -q -i " + ob + " --demuxer section5 --muxer yuv -o " + dy + " >/dev/null 2>&1";
        std::string why;
        bool ok = system(cmd.c_str()) == 0 && cmp_cropped(g, rec, read_file(dy), why);
        if (ok) {  // padded area vs the CPU API
            const int np = rec.fp.monochrome ? 1 : 3;
            std::vector<uint8_t> cpu[3];
            uint8_t *pp[3] = {nullptr, nullptr, nullptr};
            int st[3] = {0, 0, 0};
            for (int p = 0; p < np; p++) {
                cpu[p].assign(pre.recon[p], pre.recon[p] + (size_t)pre.recon_w[p] * pre.recon_h[p]);
                pp[p] = cpu[p].data();
                st[p] = (int)pre.recon_w[p];
            }
            av1g_postfilter(&rec, pp, st);
            for (int p = 0; p < np; p++)
                if (cpu[p] != g.out[p]) { ok = false; why = "padded area differs from CPU av1g_postfilter, plane " + std::to_string(p); }
        }
        if (!ok) {
            bad++;
            printf("seed %llu %ux%u%s lf=%d/%d/%d/%d cdef=%d: MISMATCH %s\n", (unsigned long long)cs, rec.fp.width, rec.fp.height,
                   is444 ? " 444" : rec.fp.monochrome ? " mono" : "",
                   rec.fp.loop_filter_level[0], rec.fp.loop_filter_level[1], rec.fp.loop_filter_level[2], rec.fp.loop_filter_level[3],
                   rec.fp.enable_cdef, why.c_str());
        }
        av1g_output_free(&o);
        av1g_output_free(&pre);
    }
    printf("GPU POSTFILTER seeds %llu..%llu: %d cases (%d with deblocking/CDEF; 4:4:4: %d, %d with deblocking/CDEF), %d mismatches -> %s\n",
           (unsigned long long)seed, (unsigned long long)(seed + count - 1), count, withPf, n444, pf444, bad, bad ? "FAILED" : "PASSED");
    return bad ? 1 : 0;
}
