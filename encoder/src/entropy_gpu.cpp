// GPU entropy path = av1core GPU entropy (GPU symbolization + od_ec per tile, av1g_gpu::run_batch) +
// OBU assembly here.  The sequence/frame header writer mirrors av1core Encoder::write_headers (not
// exported by av1core); the output is checked byte-for-byte against the CPU writer
// (av1gpu --check-entropy, tests/decoder_gate.py).
#include "pipeline.h"
#include "gpu_entropy.h"
#include "internal.h"
#include "av1g/av1g.h"
#include "av1g/levels_gen.h"

#include <cstring>

namespace {

int choose_level(uint32_t W, uint32_t H, int ntiles, int tcols) {
    if (W < 16 || H < 16) return 31;
    for (int i = 0; i < AV1G_NUM_LEVELS; i++) {
        const uint32_t *L = av1g_levels[i];
        if ((uint64_t)W * H <= L[1] && W <= L[2] && H <= L[3] && (uint32_t)ntiles <= L[4] && (uint32_t)tcols <= L[5])
            return (int)L[0];
    }
    return 31;
}

void write_headers(const av1g_frame_params *fp, const av1g::TileInfo &tiles, int level, av1g::BitWriter &seq,
                   av1g::BitWriter &fh) {
    const int W = (int)fp->width, H = (int)fp->height, mono = fp->monochrome;
    const int diff_uv = (fp->delta_q_v_dc != fp->delta_q_u_dc || fp->delta_q_v_ac != fp->delta_q_u_ac);
    const int separate_uv = !mono && (diff_uv || (fp->using_qmatrix && fp->qm_v != fp->qm_u));
    seq.put(0, 3);
    seq.put(1, 1);
    seq.put(1, 1);
    seq.put(level, 5);
    int wb = 1, hb = 1;
    while ((uint32_t)(W - 1) >> wb) wb++;
    while ((uint32_t)(H - 1) >> hb) hb++;
    seq.put(wb - 1, 4);
    seq.put(hb - 1, 4);
    seq.put(W - 1, wb);
    seq.put(H - 1, hb);
    seq.put(0, 1);
    seq.put(fp->enable_filter_intra, 1);
    seq.put(fp->enable_intra_edge_filter, 1);
    seq.put(0, 1);
    seq.put(fp->enable_cdef, 1);
    seq.put(0, 1);
    seq.put(0, 1);
    seq.put(mono, 1);
    seq.put(1, 1);
    seq.put(1, 8);
    seq.put(13, 8);
    seq.put(6, 8);
    seq.put(1, 1);
    if (!mono) {
        seq.put(0, 2);
        seq.put(separate_uv, 1);
    }
    seq.put(0, 1);
    seq.trailing_bits();

    fh.put(0, 1);
    fh.put(fp->allow_screen_content_tools, 1);
    if (fp->allow_screen_content_tools) fh.put(0, 1);
    fh.put(0, 1);
    if (fp->allow_screen_content_tools) fh.put(0, 1);
    fh.put(1, 1);
    for (int k = tiles.minLog2TileCols; k < tiles.maxLog2TileCols; k++) {
        if (k < tiles.colsLog2) fh.put(1, 1);
        else { fh.put(0, 1); break; }
    }
    for (int k = tiles.minLog2TileRows; k < tiles.maxLog2TileRows; k++) {
        if (k < tiles.rowsLog2) fh.put(1, 1);
        else { fh.put(0, 1); break; }
    }
    if (tiles.colsLog2 > 0 || tiles.rowsLog2 > 0) {
        fh.put(0, tiles.colsLog2 + tiles.rowsLog2);
        fh.put(tiles.tileSizeBytes - 1, 2);
    }
    fh.put(fp->base_q_idx, 8);
    auto delta_q = [&](int v) {
        if (v) { fh.put(1, 1); fh.su(v, 7); } else fh.put(0, 1);
    };
    delta_q(fp->delta_q_y_dc);
    if (!mono) {
        if (separate_uv) fh.put(diff_uv, 1);
        delta_q(fp->delta_q_u_dc);
        delta_q(fp->delta_q_u_ac);
        if (diff_uv) {
            delta_q(fp->delta_q_v_dc);
            delta_q(fp->delta_q_v_ac);
        }
    }
    fh.put(fp->using_qmatrix, 1);
    if (fp->using_qmatrix) {
        fh.put(fp->qm_y, 4);
        fh.put(fp->qm_u, 4);
        if (separate_uv) fh.put(fp->qm_v, 4);
    }
    fh.put(0, 1);
    fh.put(fp->delta_q_present, 1);
    if (fp->delta_q_present) {
        fh.put(fp->delta_q_res, 2);
        fh.put(0, 1);
    }
    fh.put(fp->loop_filter_level[0], 6);
    fh.put(fp->loop_filter_level[1], 6);
    if (!mono && (fp->loop_filter_level[0] || fp->loop_filter_level[1])) {
        fh.put(fp->loop_filter_level[2], 6);
        fh.put(fp->loop_filter_level[3], 6);
    }
    fh.put(fp->loop_filter_sharpness, 3);
    fh.put(0, 1);
    if (fp->enable_cdef) {
        fh.put(fp->cdef_damping_minus_3, 2);
        fh.put(fp->cdef_bits, 2);
        for (int i = 0; i < (1 << fp->cdef_bits); i++) {
            fh.put(fp->cdef_y_pri[i], 4);
            fh.put(fp->cdef_y_sec[i] == 4 ? 3 : fp->cdef_y_sec[i], 2);
            if (!mono) {
                fh.put(fp->cdef_uv_pri[i], 4);
                fh.put(fp->cdef_uv_sec[i] == 4 ? 3 : fp->cdef_uv_sec[i], 2);
            }
        }
    }
    fh.put(fp->tx_mode_select ? 1 : 0, 1);
    fh.put(fp->reduced_tx_set ? 1 : 0, 1);
    fh.byte_align();
}

}  // namespace

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
    const int level = choose_level(fp->width, fp->height, ntiles, (int)tiles.cols);
    av1g::BitWriter seq, fh;
    write_headers(fp, tiles, level, seq, fh);
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
