// GPU entropy path = av1core GPU entropy (GPU symbolization + od_ec per tile, av1g_gpu::run_batch) +
// OBU assembly here, with av1core's own sequence / frame header writer (av1g::write_headers, internal.h);
// the output is checked byte-for-byte against the CPU writer (av1gpu --check-entropy, tests/decoder_gate.py).
#include "pipeline.h"
#include "gpu_entropy.h"
#include "internal.h"
#include "av1g/av1g.h"
#include "av1g/blockinfo.h"
#include "av1g/levels_gen.h"

#include <cstring>

// OBUs ([TD] + sequence header + OBU_FRAME) from per-tile payloads.
int gp_assemble_obus(const av1g_frame_params *fp, const std::vector<std::vector<uint8_t>> &tileData, bool td,
                     std::vector<uint8_t> &obu, std::string &err) {
    av1g::TileInfo tiles;
    if (!av1g::compute_tiles(fp->width, fp->height, fp->tile_cols_log2, fp->tile_rows_log2, tiles)) { err = "tiles"; return -1; }
    const int ntiles = (int)tileData.size();
    if ((uint32_t)ntiles != tiles.cols * tiles.rows) { err = "tile count mismatch"; return -1; }
    uint64_t maxSz = 0;
    for (int i = 0; i + 1 < ntiles; i++)
        if (tileData[i].size() - 1 > maxSz) maxSz = tileData[i].size() - 1;
    tiles.tileSizeBytes = maxSz < (1u << 8) ? 1 : maxSz < (1u << 16) ? 2 : maxSz < (1u << 24) ? 3 : 4;
    const int level = av1g::choose_level(fp->width, fp->height, ntiles, (int)tiles.cols);
    av1g::BitWriter seq, fh;
    av1g::write_headers(fp, tiles, level, seq, fh);
    std::vector<uint8_t> frame = fh.buf;
    if (ntiles > 1) frame.push_back(0);
    for (int i = 0; i < ntiles; i++) {
        if (i + 1 < ntiles) {
            const uint64_t v = tileData[i].size() - 1;
            for (int k = 0; k < tiles.tileSizeBytes; k++) frame.push_back((uint8_t)(v >> (8 * k)));
        }
        frame.insert(frame.end(), tileData[i].begin(), tileData[i].end());
    }
    obu.clear();
    if (td) { obu.push_back(0x12); obu.push_back(0x00); }
    obu.push_back(0x0A);
    av1g::put_leb128(obu, seq.buf.size());
    obu.insert(obu.end(), seq.buf.begin(), seq.buf.end());
    obu.push_back(0x32);
    av1g::put_leb128(obu, frame.size());
    obu.insert(obu.end(), frame.begin(), frame.end());
    return 0;
}

int gp_entropy_gpu(std::vector<GpFrameOut *> &frames, const std::vector<const std::vector<uint8_t> *> &icc,
                   std::vector<std::vector<uint8_t>> &obus, std::vector<std::vector<uint8_t>> &avifs, double *gpuMs,
                   std::string &err) {
    const size_t n = frames.size();
    std::vector<av1g_decision_record> recs(n);
    std::vector<const av1g_decision_record *> rp(n);
    for (size_t i = 0; i < n; i++) { recs[i] = frames[i]->record(); rp[i] = &recs[i]; }
    av1g_gpu::BatchResult br;
    if (av1g_gpu::run_batch(rp, false, br, err)) return -1;
    if (gpuMs) *gpuMs = br.msUpload + br.msMaps + br.msCount + br.msScan + br.msEmit + br.msEntropy;
    obus.assign(n, {});
    avifs.assign(n, {});
    for (size_t i = 0; i < n; i++) {
        if (gp_assemble_obus(&frames[i]->fp, br.tileBytes[i], true, obus[i], err)) return -1;
        av1g_avif_meta meta;
        memset(&meta, 0, sizeof meta);
        const std::vector<uint8_t> *ic = i < icc.size() ? icc[i] : nullptr;
        meta.icc = ic && !ic->empty() ? ic->data() : nullptr;
        meta.icc_size = ic ? ic->size() : 0;
        av1g_exif_orientation_to_irot_imir(frames[i]->orientation, &meta.irot, &meta.imir);
        uint8_t *a = nullptr;
        size_t an = 0;
        if (av1g_mux_avif_ex(obus[i].data() + 2, obus[i].size() - 2, frames[i]->fp.width, frames[i]->fp.height, &meta, &a,
                             &an) != AV1G_OK) {
            err = "AVIF mux failed";
            return -1;
        }
        avifs[i].assign(a, a + an);
        free(a);
    }
    return 0;
}
