// DecisionRecord file format (little endian, packed as in memory):
//   "AV1GREC1" | av1g_frame_params | u32 n_blocks | u32 n_txb | u64 n_coeffs |
//   blocks[n_blocks] | block_txb[n_blocks+1] | txb[n_txb] | coeffs[n_coeffs] (int16)
#include "tools.h"

#include <cstdio>
#include <cstring>

namespace av1g {

bool save_record(const av1g_decision_record &r, const char *path) {
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    bool ok = fwrite("AV1GREC1", 1, 8, f) == 8;
    ok &= fwrite(&r.fp, sizeof r.fp, 1, f) == 1;
    ok &= fwrite(&r.n_blocks, 4, 1, f) == 1 && fwrite(&r.n_txb, 4, 1, f) == 1 && fwrite(&r.n_coeffs, 8, 1, f) == 1;
    ok &= fwrite(r.blocks, sizeof(av1g_block), r.n_blocks, f) == r.n_blocks;
    ok &= fwrite(r.block_txb, 4, r.n_blocks + 1, f) == r.n_blocks + 1;
    ok &= fwrite(r.txb, sizeof(av1g_txb), r.n_txb, f) == r.n_txb;
    ok &= fwrite(r.coeffs, 2, r.n_coeffs, f) == r.n_coeffs;
    fclose(f);
    return ok;
}

bool load_record(const char *path, OwnedRecord &o) {
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    char magic[8];
    memset(&o.rec, 0, sizeof o.rec);
    bool ok = fread(magic, 1, 8, f) == 8 && !memcmp(magic, "AV1GREC1", 8);
    ok = ok && fread(&o.rec.fp, sizeof o.rec.fp, 1, f) == 1;
    ok = ok && fread(&o.rec.n_blocks, 4, 1, f) == 1 && fread(&o.rec.n_txb, 4, 1, f) == 1 &&
         fread(&o.rec.n_coeffs, 8, 1, f) == 1;
    if (ok) {
        o.blocks.resize(o.rec.n_blocks);
        o.block_txb.resize(o.rec.n_blocks + 1);
        o.txb.resize(o.rec.n_txb);
        o.coeffs.resize(o.rec.n_coeffs + 1);
        ok = fread(o.blocks.data(), sizeof(av1g_block), o.rec.n_blocks, f) == o.rec.n_blocks &&
             fread(o.block_txb.data(), 4, o.rec.n_blocks + 1, f) == o.rec.n_blocks + 1 &&
             fread(o.txb.data(), sizeof(av1g_txb), o.rec.n_txb, f) == o.rec.n_txb &&
             fread(o.coeffs.data(), 2, o.rec.n_coeffs, f) == o.rec.n_coeffs;
        uint64_t nc = o.rec.n_coeffs;
        o.bind();
        o.rec.n_coeffs = nc;
    }
    fclose(f);
    return ok;
}

}  // namespace av1g
