// gpu_fuzz_test - random DecisionRecords (same generator as av1g_fuzz: all partitions,
// modes, angle deltas, CFL, tx sizes/types, tx-select, reduced tx set, delta-q,
// monochrome, 4:4:4, screen-content flag, forced/extra tiles, odd sizes) through the GPU
// symbolizer + entropy coder through av1g_gpu::run_batch_device (device arrays); streams and
// tile bytes must equal the CPU writer's.  Batches mix 4:2:0, 4:4:4 and monochrome frames.
//
//   gpu_fuzz_test --seed S --count N [--batch B] [--yuv444 P]   (P: % of colour cases in 4:4:4, default 40)
#define AV1G_FUZZ_NO_MAIN
#include "fuzz.cpp"
#include "gpu_entropy.h"
#include "av1g/symstream.h"

int main(int argc, char **argv) {
    uint64_t seed = 1;
    int count = 200, batch = 40, pct444 = 40;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--seed" && i + 1 < argc) seed = strtoull(argv[++i], 0, 10);
        else if (a == "--count" && i + 1 < argc) count = atoi(argv[++i]);
        else if (a == "--batch" && i + 1 < argc) batch = atoi(argv[++i]);
        else if (a == "--yuv444" && i + 1 < argc) pct444 = atoi(argv[++i]);
    }
    int bad = 0, done = 0, tilesChecked = 0, capacityErrors = 0, orderCheck = 0, scratchAllocs = 0, n444 = 0, nMono = 0;
    uint64_t syms = 0;
    void *scratch = nullptr;
    size_t scratchBytes = 0;
    cudaStream_t stream;
    cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);
    for (int c0 = 0; c0 < count; c0 += batch) {
        const int n = std::min(batch, count - c0);
        std::vector<FuzzCase> cases(n);
        std::vector<std::vector<std::vector<uint32_t>>> cpuSyms(n);
        std::vector<std::vector<std::vector<uint8_t>>> cpuBytes(n);
        std::vector<const av1g_decision_record *> frames;
        for (int i = 0; i < n; i++) {
            const uint64_t cs = seed + c0 + i;
            if (!make_case(cs, false, 300, 2, cases[i], pct444)) return 1;
            n444 += cases[i].rec.fp.chroma_format == AV1G_CHROMA_444;
            nMono += cases[i].rec.fp.monochrome;
            // make the record conformant (FIX_RANGE), then record the reference streams
            av1g_encode_opts eo = {AV1G_ENC_FIX_RANGE | AV1G_ENC_NO_RECON, -1};
            av1g_output o;
            if (av1g_encode(&cases[i].rec, &eo, &o) != AV1G_OK) { fprintf(stderr, "seed %llu: %s\n", (unsigned long long)cs, o.error); return 1; }
            av1g_output_free(&o);
            eo.flags = AV1G_ENC_BITSTREAM_ONLY | AV1G_ENC_NO_RECON;
            if (av1g::encode_with_hook(&cases[i].rec, &eo, &o, nullptr, &cpuSyms[i]) != AV1G_OK) return 1;
            av1g_parsed *p = nullptr;
            if (av1g_parse(o.obu, o.obu_size, &p) != AV1G_OK) { fprintf(stderr, "parse failed: %s\n", p->error); return 1; }
            for (uint32_t t = 0; t < p->n_tiles; t++)
                cpuBytes[i].emplace_back(o.obu + p->tile_offsets[t], o.obu + p->tile_offsets[t] + p->tile_bytes[t]);
            av1g_parsed_free(p);
            av1g_output_free(&o);
            frames.push_back(&cases[i].rec);
        }
        // ---- the device entry point, driven like a pipeline would: caller-owned device arrays with
        // non-zero bases and gaps between frames' coefficients, a non-blocking stream, scratch reused
        // across batches; every 3rd batch first runs with too small a symbol capacity ----
        av1g_gpu::BatchResult br;
        std::string err;
        {
            Rng g(seed + c0 + 777);
            std::vector<av1g_gpu::DevFrame> df(n);
            std::vector<av1g_block> hb((size_t)g.range(0, 5));   // leading junk blocks
            std::vector<uint32_t> hbt(hb.size(), 0xDEADBEEFu);
            std::vector<av1g_txb> ht((size_t)g.range(0, 7));
            std::vector<int16_t> hc((size_t)g.range(0, 100), 12345);
            for (auto &x : hb) memset(&x, 0x5A, sizeof x);
            for (auto &x : ht) memset(&x, 0xA5, sizeof x);
            for (int i = 0; i < n; i++) {
                const av1g_decision_record &R = cases[i].rec;
                av1g_gpu::DevFrame &d = df[i];
                d.fp = R.fp;
                d.blockBase = (uint32_t)hb.size(); d.nBlocks = R.n_blocks;
                d.txbBase = (uint32_t)ht.size(); d.nTxb = R.n_txb;
                hc.resize(hc.size() + g.range(0, 40), -7);                  // gap
                d.coeffBase = hc.size();
                hb.insert(hb.end(), R.blocks, R.blocks + R.n_blocks);
                hbt.insert(hbt.end(), R.block_txb, R.block_txb + R.n_blocks);
                ht.insert(ht.end(), R.txb, R.txb + R.n_txb);
                hc.insert(hc.end(), R.coeffs, R.coeffs + R.n_coeffs);
            }
            av1g_block *dB; uint32_t *dBT; av1g_txb *dT; int16_t *dC;
            cudaMalloc(&dB, hb.size() * sizeof(av1g_block)); cudaMalloc(&dBT, hbt.size() * 4);
            cudaMalloc(&dT, ht.size() * sizeof(av1g_txb)); cudaMalloc(&dC, hc.size() * 2);
            // on `stream`: a pageable cudaMemcpy runs on the legacy stream and may return before its DMA
            // lands, and the non-blocking stream does not wait for it
            cudaMemcpyAsync(dB, hb.data(), hb.size() * sizeof(av1g_block), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(dBT, hbt.data(), hbt.size() * 4, cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(dT, ht.data(), ht.size() * sizeof(av1g_txb), cudaMemcpyHostToDevice, stream);
            cudaMemcpyAsync(dC, hc.data(), hc.size() * 2, cudaMemcpyHostToDevice, stream);
            cudaStreamSynchronize(stream);
            av1g_gpu::DevBatch b;
            b.frames = df.data(); b.nFrames = (uint32_t)n;
            b.blocks = dB; b.blockTxb = dBT; b.txb = dT; b.coeffs = dC;
            const bool squeeze = (c0 / batch) % 3 == 1;
            if (squeeze) b.symbolCapacity = 1000;
            av1g_gpu::DevResult r;
            std::vector<uint8_t> head;
            for (int attempt = 0; attempt < 2; attempt++) {
                const size_t need = av1g_gpu::run_batch_device_scratch_bytes(b);
                if (need > scratchBytes) {
                    cudaFree(scratch);
                    if (cudaMalloc(&scratch, need) != cudaSuccess) { fprintf(stderr, "scratch alloc %zu failed\n", need); return 2; }
                    scratchBytes = need;
                    scratchAllocs++;
                }
                if (av1g_gpu::run_batch_device(b, stream, scratch, scratchBytes, r, err)) { fprintf(stderr, "run_batch_device: %s\n", err.c_str()); return 2; }
                head.resize(r.resultBytes);
                cudaMemcpyAsync(head.data(), r.result, r.resultBytes, cudaMemcpyDeviceToHost, stream);
                cudaStreamSynchronize(stream);
                const av1g_gpu::DevStatus &stt = *(const av1g_gpu::DevStatus *)head.data();
                if (attempt == 0 && squeeze) {
                    if (stt.error != av1g_gpu::AV1G_GPU_ERR_SYMBOLS || stt.symbols <= 1000) {
                        printf("batch at %llu: expected a symbol-capacity error, got error %d symbols %llu\n",
                               (unsigned long long)(seed + c0), stt.error, (unsigned long long)stt.symbols);
                        bad++;
                    } else capacityErrors++;
                    b.symbolCapacity = stt.symbols;  // exactly what is needed
                    continue;
                }
                if (stt.error) { fprintf(stderr, "batch at %llu: device error %d\n", (unsigned long long)(seed + c0), stt.error); return 2; }
                break;
            }
            const av1g_gpu::DevStatus &stt = *(const av1g_gpu::DevStatus *)head.data();
            const uint64_t *toff = (const uint64_t *)(head.data() + sizeof(av1g_gpu::DevStatus));
            std::vector<uint8_t> bytes(stt.bytes);
            std::vector<uint32_t> hs(stt.symbols);
            std::vector<uint64_t> sb(r.nTiles), se(r.nTiles);
            cudaMemcpyAsync(bytes.data(), r.bytes, bytes.size(), cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(hs.data(), r.syms, hs.size() * 4, cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(sb.data(), r.symBegin, sb.size() * 8, cudaMemcpyDeviceToHost, stream);
            cudaMemcpyAsync(se.data(), r.symEnd, se.size() * 8, cudaMemcpyDeviceToHost, stream);
            if (cudaStreamSynchronize(stream) != cudaSuccess) { fprintf(stderr, "CUDA error\n"); return 2; }
            br.tileBytes.assign(n, {});
            br.tileSyms.assign(n, {});
            for (int f = 0; f < n; f++)
                for (uint32_t i = r.frameTile0[f]; i < r.frameTile0[f + 1]; i++) {
                    br.tileBytes[f].emplace_back(bytes.begin() + toff[i], bytes.begin() + toff[i + 1]);
                    br.tileSyms[f].emplace_back(hs.begin() + sb[i], hs.begin() + se[i]);
                }
            // one broken record per run: swapped blocks must be reported (AV1G_GPU_ERR_ORDER), not crash
            if (c0 == 0 && n > 0 && cases[0].rec.n_blocks > 2) {
                std::vector<av1g_block> sw(cases[0].rec.blocks, cases[0].rec.blocks + cases[0].rec.n_blocks);
                std::swap(sw[0], sw.back());
                cudaMemcpyAsync(dB + df[0].blockBase, sw.data(), sw.size() * sizeof(av1g_block), cudaMemcpyHostToDevice, stream);
                cudaStreamSynchronize(stream);
                b.symbolCapacity = 0;
                const size_t need = av1g_gpu::run_batch_device_scratch_bytes(b);
                if (need > scratchBytes) { cudaFree(scratch); cudaMalloc(&scratch, need); scratchBytes = need; scratchAllocs++; }
                if (av1g_gpu::run_batch_device(b, stream, scratch, scratchBytes, r, err) == 0) {
                    av1g_gpu::DevStatus s2;
                    cudaMemcpyAsync(&s2, r.status, sizeof s2, cudaMemcpyDeviceToHost, stream);
                    cudaStreamSynchronize(stream);
                    orderCheck = (s2.error & av1g_gpu::AV1G_GPU_ERR_ORDER) ? 1 : -1;
                } else orderCheck = -1;
                if (orderCheck < 0) { printf("swapped-block record was not reported as an order error\n"); bad++; }
            }
            cudaFree(dB); cudaFree(dBT); cudaFree(dT); cudaFree(dC);
        }
        for (int i = 0; i < n; i++) {
            bool ok = br.tileBytes[i].size() == cpuBytes[i].size();
            for (size_t t = 0; ok && t < cpuBytes[i].size(); t++) {
                tilesChecked++;
                syms += cpuSyms[i][t].size();
                ok = br.tileSyms[i][t] == cpuSyms[i][t] && br.tileBytes[i][t] == cpuBytes[i][t];
            }
            if (!ok) {
                bad++;
                const av1g_frame_params &fp = cases[i].rec.fp;
                printf("seed %llu %ux%u mono%d 444%d txsel%d red%d sct%d: GPU MISMATCH\n", (unsigned long long)(seed + c0 + i), fp.width,
                       fp.height, fp.monochrome, fp.chroma_format, fp.tx_mode_select, fp.reduced_tx_set, fp.allow_screen_content_tools);
                for (size_t t = 0; t < cpuSyms[i].size() && t < br.tileSyms[i].size(); t++) {
                    const auto &a = br.tileSyms[i][t], &b = cpuSyms[i][t];
                    size_t k = 0;
                    while (k < a.size() && k < b.size() && a[k] == b[k]) k++;
                    if (k < a.size() || k < b.size())
                        printf("   tile %zu: first symbol difference at %zu/%zu: gpu %08x cpu %08x\n", t, k, b.size(),
                               k < a.size() ? a[k] : 0, k < b.size() ? b[k] : 0);
                }
            }
        }
        done += n;
    }
    printf("GPU FUZZ (run_batch_device, device arrays, non-blocking stream, %d scratch allocations) seeds %llu..%llu: %d cases "
           "(%d 4:4:4, %d monochrome), %d tiles, %llu symbols, %d symbol-capacity errors reported + rerun, order check %s, %d mismatches -> %s\n",
           scratchAllocs, (unsigned long long)seed, (unsigned long long)(seed + count - 1), done, n444, nMono, tilesChecked,
           (unsigned long long)syms, capacityErrors, orderCheck > 0 ? "ok" : orderCheck < 0 ? "FAILED" : "not run", bad,
           bad ? "FAILED" : "PASSED");
    cudaFree(scratch);
    cudaStreamDestroy(stream);
    return bad ? 1 : 0;
}
