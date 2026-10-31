// Random-decision fuzzer for av1core.
//
// Generates random *legal* DecisionRecords (random sizes, partitions, modes,
// angle deltas, CFL alphas, tx sizes/types, sparse coefficients incl. Golomb-range
// magnitudes, skip blocks, tiles, delta-q, QM, filter intra, edge filter,
// deblocking + CDEF; 4:2:0, 4:4:4 and monochrome; 8-bit and 10-bit), encodes them with av1core and
// checks that dav1d AND aomdec reproduce av1core's reconstruction byte for byte (10-bit: 16-bit
// little-endian samples from both decoders).
//
//   av1g_fuzz --seed S --count N [--mode m2|full] [--out DIR] [--keep] [--maxdim D]
//             [--big P] [--yuv444 P] [--depth10 P] [--lossless P] [--no-aom] [--quiet]
// --yuv444 P: P % of the non-monochrome cases are 4:4:4 (default 40).  That choice comes from
// its own random stream, so every other case of a seed is the same record as with
// --yuv444 0, which reproduces the pre-4:4:4 (4:2:0 / monochrome only) case stream exactly.
// --depth10 P: P % of the cases are 10-bit (default 40), again from a separate stream: a 10-bit
// case is the record its seed has in 8 bits with fp.bit_depth = 10, and --depth10 0 reproduces
// the 8-bit-only case stream (every 8-bit case of a seed is the same with any P).
// --lossless P: P % of the seeds are lossless cases (default 25), chosen by a separate stream: every other seed's case
// is the same record as with --lossless 0, which reproduces the pre-lossless case stream exactly.  A lossless case
// (base_q_idx 0; 8 / 10 bit, 4:2:0 / 4:4:4 / monochrome, every block size, mode, CfL, filter intra, edge filter, tiles,
// colour descriptions incl. identity 1/13/0) has a random SOURCE picture: random decisions are coded closed-loop
// through av1core's TxbHook (levels = av1g_fwht4x4(source - exact prediction)), blocks without residual then get
// skip = 1 (most of them) and the record is re-encoded.  Checked: av1core's reconstruction == the source, dav1d and
// aomdec == av1core, the parser round trip.
// Exit status 0 iff every case matched.  Failing cases are kept in DIR as
// case_<seed>.{obu,yuv,txt}.
// post-filter path coverage (counted when the fuzzer re-runs the shared filters itself), per bit depth
static unsigned long long g_pfStat[2][16];
static int g_pfBd10 = 0;   // index into g_pfStat: the case being filtered is 10-bit
#if defined(__CUDA_ARCH__)
#define AV1G_PF_STAT(kind) ((void)0)
#else
#define AV1G_PF_STAT(kind) (g_pfStat[g_pfBd10][(kind)]++)
#endif
#include "av1g/postfilter.h"
#include "av1g/av1g.h"
#include "av1g/blockinfo.h"
#include "av1g/dsp.h"
#include "internal.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <sys/stat.h>

// Reference decoders: dav1d and aomdec from PATH, or AV1G_DAV1D / AV1G_AOMDEC.
static std::string tool_path(const char *env, const char *dflt) {
    const char *v = getenv(env);
    return v && *v ? v : dflt;
}
static const std::string DAV1D = tool_path("AV1G_DAV1D", "dav1d");
static const std::string AOMDEC = tool_path("AV1G_AOMDEC", "aomdec");
static std::string default_out_dir(const char *leaf) {
    const char *t = getenv("TMPDIR");
    return std::string(t && *t ? t : "/tmp") + "/" + leaf;
}

struct Rng {
    uint64_t s;
    static uint64_t mix(uint64_t z) {
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    // hash the seed so that streams of consecutive seeds are not shifted copies
    explicit Rng(uint64_t seed) : s(mix(mix(seed + 0x632BE59BD9B4E019ull))) { next(); }
    uint64_t next() {  // splitmix64
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    int range(int lo, int hi) { return lo + (int)(next() % (uint64_t)(hi - lo + 1)); }  // inclusive
    bool chance(int pct) { return (int)(next() % 100) < pct; }
};

struct Coverage {
    uint64_t part[22][10] = {};        // [node bsize][partition]
    uint64_t bsize[22] = {};
    uint64_t ymode[13][7] = {};        // [mode][angle delta + 3]
    uint64_t uvmode[14][7] = {};
    uint64_t cflSigns[9] = {};
    uint64_t txLuma[19][16] = {};      // coded (eob > 0) luma tx blocks
    uint64_t txChroma[19] = {};
    uint64_t depth[22][3] = {};
    uint64_t fiMode[5] = {}, fiFrames = 0, efFrames = 0, efDirBlocks = 0;
    uint64_t sct = 0, qm = 0, qmSep = 0, qmLevel[16] = {}, dq = 0, dqRes[4] = {}, dqBigJump = 0, dqSkipSb = 0;
    uint64_t lf = 0, lfChromaOff = 0, lfSharp[8] = {}, cdef = 0, cdefBits[4] = {}, cdefDamp[4] = {}, cdefSec4 = 0, pfBoth = 0;
    uint64_t golomb = 0, maxMag = 0, skip = 0, mono = 0, deltaq = 0, sepuv = 0, multiTile = 0, edgeForced = 0;
    uint32_t maxTileCols = 0, maxTileRows = 0;
    // 4:4:4 cases (counted separately: everything 4:2:0 exercises must also be hit in 4:4:4)
    uint64_t c444 = 0, c444Tiles = 0, c444Lf = 0, c444Cdef = 0, c444Dq = 0, c444Qm = 0, c444Fi = 0, c444Ef = 0, c444TxSel = 0,
             c444Reduced = 0, c444Sct = 0, c444DeltaQ = 0, c444Big = 0;
    uint64_t txChroma444[19] = {};       // coded (eob > 0) chroma tx blocks per size
    uint64_t txChromaType444[19][16] = {};  // derived chroma tx types coded per size
    uint64_t cfl444[22] = {};            // CfL blocks per bsize
    uint64_t uvmode444[14][7] = {};      // uv mode x angle delta
    uint64_t subChroma444 = 0;           // 4xN / Nx4 blocks at even mi positions (no chroma in 4:2:0)
    uint64_t multiTxChroma444 = 0;       // blocks with several chroma tx blocks per plane (64-sample sides)
    uint64_t bsize444[22] = {};
    // 10-bit cases (counted separately: every tool the 8-bit cases exercise must also be hit at 10 bits)
    uint64_t c10 = 0, c10Fmt[3] = {}, c10Tiles = 0, c10Lf = 0, c10Cdef = 0, c10Dq = 0, c10Qm = 0, c10Fi = 0, c10Ef = 0,
             c10TxSel = 0, c10Reduced = 0, c10Sct = 0, c10DeltaQ = 0, c10Big = 0, c10Fixed = 0, c10Golomb = 0, c10Clamp = 0;
    uint64_t bsize10[22] = {}, ymode10[13][7] = {}, uvmode10[14][7] = {}, cflSigns10[9] = {}, cfl10[22] = {};
    uint64_t txLuma10[19][16] = {}, txChroma10[19] = {}, fiMode10[5] = {}, efDir10 = 0, qmLevel10[16] = {}, dqRes10[4] = {};
    uint64_t lfSharp10[8] = {}, cdefBits10[4] = {}, cdefDamp10[4] = {}, cdefSec4_10 = 0, pfBoth10 = 0, lfChromaOff10 = 0;
    uint64_t maxMag10 = 0;
    void report() const {
        static const int sq[4] = {AV1G_BLOCK_8X8, AV1G_BLOCK_16X16, AV1G_BLOCK_32X32, AV1G_BLOCK_64X64};
        int missing = 0;
        printf("COVERAGE partitions (node 8/16/32/64 x NONE..VERT_4):\n");
        for (int s : sq) {
            printf("  %2dx%-2d:", 4 * av1g_bw4(s), 4 * av1g_bh4(s));
            for (int p = 0; p < (s == AV1G_BLOCK_8X8 ? 4 : 10); p++) { printf(" %llu", (unsigned long long)part[s][p]); missing += !part[s][p]; }
            printf("\n");
        }
        printf("  block sizes:");
        for (int b = 0; b < 22; b++) if (b < 13 || b > 15) { printf(" %llu", (unsigned long long)bsize[b]); missing += !bsize[b]; }
        printf("\n  y_mode x angle (directional rows show 7 deltas):\n");
        for (int m = 0; m < 13; m++) {
            printf("   %2d:", m);
            for (int d = 0; d < 7; d++) {
                bool need = av1g_is_directional(m) || d == 3;
                if (need) { printf(" %llu", (unsigned long long)ymode[m][d]); missing += !ymode[m][d]; }
            }
            printf("\n");
        }
        printf("  uv_mode x angle:");
        for (int m = 0; m < 14; m++) for (int d = 0; d < 7; d++) {
            bool need = (m < 13 && av1g_is_directional(m)) || d == 3;
            if (need) missing += !uvmode[m][d];
        }
        for (int m = 0; m < 14; m++) printf(" %llu", (unsigned long long)uvmode[m][3]);
        printf("\n  cfl sign combos:");
        for (int i = 0; i < 8; i++) { printf(" %llu", (unsigned long long)cflSigns[i]); missing += !cflSigns[i]; }
        int txAllowed = 0, txHit = 0;
        for (int t = 0; t < 19; t++) for (int ty = 0; ty < 16; ty++) {
            bool allowed = av1g_tx_type_in_set_intra(av1g_get_tx_set_intra(t, 0), ty) || av1g_tx_type_in_set_intra(av1g_get_tx_set_intra(t, 1), ty);
            if (!allowed) continue;
            txAllowed++;
            if (txLuma[t][ty]) txHit++; else { missing++; printf("\n  MISSING luma tx size %d type %d", t, ty); }
        }
        printf("\n  luma (tx size, type) combos coded: %d / %d allowed\n  chroma tx sizes:", txHit, txAllowed);
        for (int t = 0; t < 19; t++) printf(" %llu", (unsigned long long)txChroma[t]);
        printf("\n  tx_depth per bsize (0/1/2):");
        for (int b = 0; b < 22; b++) if (b < 13 || b > 15) printf(" [%llu/%llu/%llu]", (unsigned long long)depth[b][0], (unsigned long long)depth[b][1], (unsigned long long)depth[b][2]);
        printf("\n  golomb coeffs %llu, max |coeff| %llu, skip blocks %llu, mono cases %llu, delta-q cases %llu (separate uv %llu),"
               " multi-tile cases %llu (max %ux%u), edge-forced partitions %llu\n",
               (unsigned long long)golomb, (unsigned long long)maxMag, (unsigned long long)skip, (unsigned long long)mono,
               (unsigned long long)deltaq, (unsigned long long)sepuv, (unsigned long long)multiTile, maxTileCols, maxTileRows,
               (unsigned long long)edgeForced);
        printf("  allow_screen_content_tools cases %llu\n", (unsigned long long)sct);
        printf("  filter intra: %llu cases, modes 0..4: %llu %llu %llu %llu %llu; intra edge filter: %llu cases, %llu directional blocks\n",
               (unsigned long long)fiFrames, (unsigned long long)fiMode[0], (unsigned long long)fiMode[1], (unsigned long long)fiMode[2],
               (unsigned long long)fiMode[3], (unsigned long long)fiMode[4], (unsigned long long)efFrames, (unsigned long long)efDirBlocks);
        for (int i = 0; i < 5; i++) missing += !fiMode[i];
        missing += !efDirBlocks;
        printf("  quantizer matrices: %llu cases (%llu with qm_v != qm_u), levels used:", (unsigned long long)qm, (unsigned long long)qmSep);
        for (int l = 0; l < 16; l++) { printf(" %llu", (unsigned long long)qmLevel[l]); missing += !qmLevel[l]; }
        printf("\n  delta-q: %llu cases, res 0..3: %llu %llu %llu %llu, |reduced delta| >= 3 (large form): %llu SBs, skipped 64x64 SBs: %llu\n",
               (unsigned long long)dq, (unsigned long long)dqRes[0], (unsigned long long)dqRes[1], (unsigned long long)dqRes[2],
               (unsigned long long)dqRes[3], (unsigned long long)dqBigJump, (unsigned long long)dqSkipSb);
        for (int r = 0; r < 4; r++) missing += !dqRes[r];
        missing += !dqBigJump + !qmSep;
        printf("  deblocking: %llu cases (%llu with a chroma plane off), sharpness 0..7:", (unsigned long long)lf, (unsigned long long)lfChromaOff);
        for (int i = 0; i < 8; i++) { printf(" %llu", (unsigned long long)lfSharp[i]); missing += !lfSharp[i]; }
        printf("\n  CDEF: %llu cases (%llu with deblocking too), cdef_bits 0..3: %llu %llu %llu %llu, damping 3..6: %llu %llu %llu %llu,"
               " sec strength 4 used %llu\n", (unsigned long long)cdef, (unsigned long long)pfBoth, (unsigned long long)cdefBits[0],
               (unsigned long long)cdefBits[1], (unsigned long long)cdefBits[2], (unsigned long long)cdefBits[3],
               (unsigned long long)cdefDamp[0], (unsigned long long)cdefDamp[1], (unsigned long long)cdefDamp[2], (unsigned long long)cdefDamp[3],
               (unsigned long long)cdefSec4);
        for (int i = 0; i < 4; i++) missing += !cdefBits[i] + !cdefDamp[i];
        missing += !lfChromaOff + !cdefSec4 + !pfBoth;
        static const char *pfName[AV1G_PF_STAT_COUNT] = {"lf masked", "narrow hev", "narrow", "wide6 (chroma)", "wide8", "wide16",
                                                         "cdef all-skip 8x8", "cdef filtered 8x8", "cdef clamped px"};
        printf("  post-filter paths (lines / blocks / px):");
        for (int i = 0; i < AV1G_PF_STAT_COUNT; i++) {
            printf(" %s %llu,", pfName[i], g_pfStat[0][i] + g_pfStat[1][i]);
            missing += !(g_pfStat[0][i] + g_pfStat[1][i]);
        }
        printf("\n");
        // ---- 4:4:4 ----
        printf("COVERAGE 4:4:4: %llu cases (tiles %llu, deblocking %llu, CDEF %llu, delta-q %llu, QM %llu, filter intra %llu,"
               " edge filter %llu, tx-select %llu, reduced set %llu, screen content %llu, frame delta-q %llu, forced tiles %llu)\n",
               (unsigned long long)c444, (unsigned long long)c444Tiles, (unsigned long long)c444Lf, (unsigned long long)c444Cdef,
               (unsigned long long)c444Dq, (unsigned long long)c444Qm, (unsigned long long)c444Fi, (unsigned long long)c444Ef,
               (unsigned long long)c444TxSel, (unsigned long long)c444Reduced, (unsigned long long)c444Sct,
               (unsigned long long)c444DeltaQ, (unsigned long long)c444Big);
        missing += !c444 + !c444Tiles + !c444Lf + !c444Cdef + !c444Dq + !c444Qm + !c444Fi + !c444Ef + !c444TxSel + !c444Reduced +
                   !c444Sct + !c444DeltaQ;
        printf("  block sizes:");
        for (int b = 0; b < 22; b++) if (b < 13 || b > 15) { printf(" %llu", (unsigned long long)bsize444[b]); missing += !bsize444[b]; }
        printf("\n  chroma tx sizes coded:");
        int cAllowed = 0, cHit = 0;
        for (int t = 0; t < 19; t++) {
            printf(" %llu", (unsigned long long)txChroma444[t]);
            if (av1g_tx_w(t) <= 32 && av1g_tx_h(t) <= 32) { cAllowed++; cHit += txChroma444[t] > 0; missing += !txChroma444[t]; }
        }
        int ctAllowed = 0, ctHit = 0;
        for (int t = 0; t < 19; t++) {
            if (av1g_tx_w(t) > 32 || av1g_tx_h(t) > 32) continue;
            for (int ty = 0; ty < 16; ty++) {
                bool derivable = false;   // some uv_mode / reduced_tx_set maps to this type
                for (int m = 0; m <= AV1G_UV_CFL_PRED && !derivable; m++)
                    derivable = av1g_chroma_tx_type(t, m, 0) == ty || av1g_chroma_tx_type(t, m, 1) == ty;
                if (!derivable) continue;
                ctAllowed++;
                if (txChromaType444[t][ty]) ctHit++; else { missing++; printf("\n  MISSING 4:4:4 chroma tx size %d type %d", t, ty); }
            }
        }
        printf("\n  chroma tx sizes: %d / %d, (size, derived type) combos: %d / %d\n  CfL per bsize:", cHit, cAllowed, ctHit, ctAllowed);
        for (int b = 0; b < 22; b++)
            if ((b < 13 || b > 15) && av1g_cfl_allowed(b)) { printf(" %llu", (unsigned long long)cfl444[b]); missing += !cfl444[b]; }
        printf("\n  uv_mode x angle:");
        for (int m = 0; m < 14; m++) for (int d = 0; d < 7; d++) {
            bool need = (m < 13 && av1g_is_directional(m)) || d == 3;
            if (need) missing += !uvmode444[m][d];
        }
        for (int m = 0; m < 14; m++) printf(" %llu", (unsigned long long)uvmode444[m][3]);
        printf("\n  4xN/Nx4 chroma at even positions %llu, blocks with several chroma tx per plane %llu\n",
               (unsigned long long)subChroma444, (unsigned long long)multiTxChroma444);
        missing += !subChroma444 + !multiTxChroma444;
        // ---- 10-bit ----
        if (c10) {
            const int m0 = missing;
            printf("COVERAGE 10-bit: %llu cases (4:2:0 %llu, 4:4:4 %llu, mono %llu; tiles %llu, deblocking %llu, CDEF %llu, delta-q %llu,"
                   " QM %llu, filter intra %llu, edge filter %llu, tx-select %llu, reduced set %llu, screen content %llu, frame delta-q %llu,"
                   " forced tiles %llu)\n",
                   (unsigned long long)c10, (unsigned long long)c10Fmt[0], (unsigned long long)c10Fmt[1], (unsigned long long)c10Fmt[2],
                   (unsigned long long)c10Tiles, (unsigned long long)c10Lf, (unsigned long long)c10Cdef, (unsigned long long)c10Dq,
                   (unsigned long long)c10Qm, (unsigned long long)c10Fi, (unsigned long long)c10Ef, (unsigned long long)c10TxSel,
                   (unsigned long long)c10Reduced, (unsigned long long)c10Sct, (unsigned long long)c10DeltaQ, (unsigned long long)c10Big);
            for (int i = 0; i < 3; i++) missing += !c10Fmt[i];
            missing += !c10Tiles + !c10Lf + !c10Cdef + !c10Dq + !c10Qm + !c10Fi + !c10Ef + !c10TxSel + !c10Reduced + !c10Sct + !c10DeltaQ;
            printf("  block sizes:");
            for (int b = 0; b < 22; b++) if (b < 13 || b > 15) { printf(" %llu", (unsigned long long)bsize10[b]); missing += !bsize10[b]; }
            int ym = 0, uvm = 0;
            for (int m = 0; m < 13; m++) for (int d = 0; d < 7; d++) if (av1g_is_directional(m) || d == 3) ym += !ymode10[m][d];
            for (int m = 0; m < 14; m++) for (int d = 0; d < 7; d++) if ((m < 13 && av1g_is_directional(m)) || d == 3) uvm += !uvmode10[m][d];
            missing += ym + uvm;
            printf("\n  y_mode x angle missing %d, uv_mode x angle missing %d, cfl sign combos:", ym, uvm);
            for (int i = 0; i < 8; i++) { printf(" %llu", (unsigned long long)cflSigns10[i]); missing += !cflSigns10[i]; }
            printf("\n  CfL per bsize:");
            for (int b = 0; b < 22; b++)
                if ((b < 13 || b > 15) && av1g_cfl_allowed(b)) { printf(" %llu", (unsigned long long)cfl10[b]); missing += !cfl10[b]; }
            int txAllowed = 0, txHit = 0;
            for (int t = 0; t < 19; t++) for (int ty = 0; ty < 16; ty++) {
                bool allowed = av1g_tx_type_in_set_intra(av1g_get_tx_set_intra(t, 0), ty) || av1g_tx_type_in_set_intra(av1g_get_tx_set_intra(t, 1), ty);
                if (!allowed) continue;
                txAllowed++;
                if (txLuma10[t][ty]) txHit++; else { missing++; printf("\n  MISSING 10-bit luma tx size %d type %d", t, ty); }
            }
            int cAllowed = 0, cHit = 0;
            for (int t = 0; t < 19; t++)
                if (av1g_tx_w(t) <= 32 && av1g_tx_h(t) <= 32) { cAllowed++; cHit += txChroma10[t] > 0; missing += !txChroma10[t]; }
            printf("\n  luma (tx size, type) combos coded: %d / %d, chroma tx sizes coded: %d / %d\n", txHit, txAllowed, cHit, cAllowed);
            printf("  filter intra modes:");
            for (int i = 0; i < 5; i++) { printf(" %llu", (unsigned long long)fiMode10[i]); missing += !fiMode10[i]; }
            printf(", edge-filter directional blocks %llu, QM levels:", (unsigned long long)efDir10);
            missing += !efDir10;
            for (int l = 0; l < 16; l++) { printf(" %llu", (unsigned long long)qmLevel10[l]); missing += !qmLevel10[l]; }
            printf("\n  delta-q res 0..3:");
            for (int r = 0; r < 4; r++) { printf(" %llu", (unsigned long long)dqRes10[r]); missing += !dqRes10[r]; }
            printf(", deblocking sharpness 0..7:");
            for (int i = 0; i < 8; i++) { printf(" %llu", (unsigned long long)lfSharp10[i]); missing += !lfSharp10[i]; }
            printf(" (chroma plane off %llu), CDEF bits:", (unsigned long long)lfChromaOff10);
            for (int i = 0; i < 4; i++) { printf(" %llu", (unsigned long long)cdefBits10[i]); missing += !cdefBits10[i] + !cdefDamp10[i]; }
            printf(" damping:");
            for (int i = 0; i < 4; i++) printf(" %llu", (unsigned long long)cdefDamp10[i]);
            printf(", sec 4 %llu, with deblocking %llu\n", (unsigned long long)cdefSec4_10, (unsigned long long)pfBoth10);
            missing += !lfChromaOff10 + !cdefSec4_10 + !pfBoth10;
            static const char *pfName10[AV1G_PF_STAT_COUNT] = {"lf masked", "narrow hev", "narrow", "wide6 (chroma)", "wide8", "wide16",
                                                               "cdef all-skip 8x8", "cdef filtered 8x8", "cdef clamped px"};
            printf("  post-filter paths at 10 bits:");
            for (int i = 0; i < AV1G_PF_STAT_COUNT; i++) { printf(" %s %llu,", pfName10[i], g_pfStat[1][i]); missing += !g_pfStat[1][i]; }
            printf("\n  golomb coeffs %llu, max |coeff| %llu, dequantized values clamped to +-2^17: %llu, range-fixed txb %llu\n",
                   (unsigned long long)c10Golomb, (unsigned long long)maxMag10, (unsigned long long)c10Clamp, (unsigned long long)c10Fixed);
            missing += !c10Golomb + !c10Clamp + !c10Fixed;
            printf("  10-bit items missing: %d\n", missing - m0);
        }
        printf("COVERAGE missing items: %d\n", missing);
    }
};
static Coverage cov;
static bool g_cov10 = false;   // the case being generated is 10-bit (coverage only)

struct Gen {
    Rng &rng;
    bool m2;
    int W, H, MiCols, MiRows, mono, txSelect, reduced;
    int ssx = 1, ssy = 1;   // 4:4:4: 0, 0
    std::vector<av1g_block> blocks;
    int maxDepthBias;
    int dqBias = 0;
    int pfSmooth = 0;  // post-filter cases: more skip blocks (flat areas reach the wide filters)
    int fi = 0;  // enable_filter_intra  // delta-q cases: produce whole-SB skip blocks (no delta coded for them)
    av1g_tiles tiles;
    Coverage *C = &cov;     // generation-time coverage (lossless cases count into their own instance)
    bool lossless = false;  // lossless case: no skip (set after the closed loop), CfL only for 4x4 chroma blocks

    Gen(Rng &r, bool m2_) : rng(r), m2(m2_) {}

    void make_block(int r, int c, int bs) {
        av1g_block b;
        memset(&b, 0, sizeof b);
        b.mi_row = (uint16_t)r;
        b.mi_col = (uint16_t)c;
        b.bsize = (uint8_t)bs;
        if (m2) {
            b.y_mode = AV1G_DC_PRED;
            b.uv_mode = AV1G_DC_PRED;
            b.skip = rng.chance(10);
            blocks.push_back(b);
            return;
        }
        b.skip = rng.chance(bs == AV1G_BLOCK_64X64 && dqBias ? 60 : pfSmooth ? 45 : 15);
        C->skip += b.skip;
        b.y_mode = (uint8_t)rng.range(0, 12);
        if (bs >= AV1G_BLOCK_8X8 && av1g_is_directional(b.y_mode) && rng.chance(70)) b.angle_delta_y = (int8_t)rng.range(-3, 3);
        if (txSelect && bs > AV1G_BLOCK_4X4) b.tx_depth = (uint8_t)rng.range(0, av1g_max_tx_depth(bs));
        if (lossless) b.skip = 0;
        if (av1g_has_chroma(bs, r, c, mono, ssx, ssy)) {
            const bool cflOk = lossless ? av1g_cfl_allowed_ex(bs, 1, ssx, ssy) != 0 : av1g_cfl_allowed(bs) != 0;
            int maxuv = cflOk ? 13 : 12;
            b.uv_mode = (uint8_t)rng.range(0, maxuv);
            if (cflOk && rng.chance(lossless ? 35 : 20)) b.uv_mode = AV1G_UV_CFL_PRED;
            if (b.uv_mode == AV1G_UV_CFL_PRED) {
                do {
                    b.cfl_alpha_u = (int8_t)(rng.chance(25) ? 0 : rng.range(-16, 16));
                    b.cfl_alpha_v = (int8_t)(rng.chance(25) ? 0 : rng.range(-16, 16));
                } while (b.cfl_alpha_u == 0 && b.cfl_alpha_v == 0);
            }
            if (bs >= AV1G_BLOCK_8X8 && av1g_is_directional(b.uv_mode) && rng.chance(70)) b.angle_delta_uv = (int8_t)rng.range(-3, 3);
            C->uvmode[b.uv_mode][b.angle_delta_uv + 3]++;
            if (g_cov10) {
                C->uvmode10[b.uv_mode][b.angle_delta_uv + 3]++;
                if (b.uv_mode == AV1G_UV_CFL_PRED) C->cfl10[bs]++;
            }
            if (!ssx) {
                C->uvmode444[b.uv_mode][b.angle_delta_uv + 3]++;
                if (b.uv_mode == AV1G_UV_CFL_PRED) C->cfl444[bs]++;
                if ((av1g_bw4(bs) == 1 && !(c & 1)) || (av1g_bh4(bs) == 1 && !(r & 1))) C->subChroma444++;
                if (av1g_bw4(bs) == 16 || av1g_bh4(bs) == 16) C->multiTxChroma444++;
            }
            if (b.uv_mode == AV1G_UV_CFL_PRED) {
                int su = b.cfl_alpha_u == 0 ? 0 : b.cfl_alpha_u < 0 ? 1 : 2, sv = b.cfl_alpha_v == 0 ? 0 : b.cfl_alpha_v < 0 ? 1 : 2;
                C->cflSigns[su * 3 + sv - 1]++;
                if (g_cov10) C->cflSigns10[su * 3 + sv - 1]++;
            }
        }
        if (fi && b.y_mode == AV1G_DC_PRED && av1g_bw4(bs) <= 8 && av1g_bh4(bs) <= 8 && rng.chance(50)) {
            b.filter_intra = (uint8_t)rng.range(1, 5);
            C->fiMode[b.filter_intra - 1]++;
            if (g_cov10) C->fiMode10[b.filter_intra - 1]++;
        }
        C->ymode[b.y_mode][b.angle_delta_y + 3]++;
        C->bsize[bs]++;
        if (g_cov10) { C->ymode10[b.y_mode][b.angle_delta_y + 3]++; C->bsize10[bs]++; }
        if (!ssx && !mono) C->bsize444[bs]++;
        C->depth[bs][b.tx_depth]++;
        blocks.push_back(b);
    }

    int pick_partition(int bs, int hasRows, int hasCols, int depth) {
        if (m2) {
            if (!(hasRows && hasCols)) return AV1G_PARTITION_SPLIT;
            return bs > AV1G_BLOCK_16X16 ? AV1G_PARTITION_SPLIT : AV1G_PARTITION_NONE;
        }
        if (hasRows && hasCols) {
            if (bs == AV1G_BLOCK_8X8) return rng.range(0, 3);
            if (bs == AV1G_BLOCK_64X64 && dqBias && rng.chance(25)) return AV1G_PARTITION_NONE;
            // bias: split often at large sizes, use every type
            int pSplit = bs == AV1G_BLOCK_64X64 ? 55 : bs == AV1G_BLOCK_32X32 ? 40 : 25;
            pSplit = pSplit * maxDepthBias / 2;
            if (rng.chance(pSplit)) return AV1G_PARTITION_SPLIT;
            int p = rng.range(0, 9);
            return p;
        }
        if (hasCols) return rng.chance(50) ? AV1G_PARTITION_HORZ : AV1G_PARTITION_SPLIT;
        if (hasRows) return rng.chance(50) ? AV1G_PARTITION_VERT : AV1G_PARTITION_SPLIT;
        return AV1G_PARTITION_SPLIT;
    }

    void partition(int r, int c, int bSize, int depth) {
        if (r >= MiRows || c >= MiCols) return;
        const int num4x4 = av1g_bw4(bSize), half = num4x4 >> 1, quarter = half >> 1;
        const int hasRows = (r + half) < MiRows, hasCols = (c + half) < MiCols;
        int p = bSize < AV1G_BLOCK_8X8 ? AV1G_PARTITION_NONE : pick_partition(bSize, hasRows, hasCols, depth);
        if (bSize >= AV1G_BLOCK_8X8) { C->part[bSize][p]++; if (!(hasRows && hasCols)) C->edgeForced++; }
        const int sub = AV1G_T(Partition_Subsize)[p][bSize];
        const int sp = AV1G_T(Partition_Subsize)[AV1G_PARTITION_SPLIT][bSize];
        switch (p) {
        case AV1G_PARTITION_NONE: make_block(r, c, sub); break;
        case AV1G_PARTITION_HORZ: make_block(r, c, sub); if (hasRows) make_block(r + half, c, sub); break;
        case AV1G_PARTITION_VERT: make_block(r, c, sub); if (hasCols) make_block(r, c + half, sub); break;
        case AV1G_PARTITION_SPLIT:
            partition(r, c, sub, depth + 1); partition(r, c + half, sub, depth + 1);
            partition(r + half, c, sub, depth + 1); partition(r + half, c + half, sub, depth + 1);
            break;
        case AV1G_PARTITION_HORZ_A: make_block(r, c, sp); make_block(r, c + half, sp); make_block(r + half, c, sub); break;
        case AV1G_PARTITION_HORZ_B: make_block(r, c, sub); make_block(r + half, c, sp); make_block(r + half, c + half, sp); break;
        case AV1G_PARTITION_VERT_A: make_block(r, c, sp); make_block(r + half, c, sp); make_block(r, c + half, sub); break;
        case AV1G_PARTITION_VERT_B: make_block(r, c, sub); make_block(r, c + half, sp); make_block(r + half, c + half, sp); break;
        case AV1G_PARTITION_HORZ_4:
            for (int i = 0; i < 4; i++) if (i < 3 || r + quarter * 3 < MiRows) make_block(r + quarter * i, c, sub);
            break;
        default:
            for (int i = 0; i < 4; i++) if (i < 3 || c + quarter * 3 < MiCols) make_block(r, c + quarter * i, sub);
            break;
        }
    }

    void gen_blocks() {
        for (uint32_t tr = 0; tr < tiles.rows; tr++)
            for (uint32_t tc = 0; tc < tiles.cols; tc++)
                for (uint32_t r = tiles.mi_row_starts[tr]; r < tiles.mi_row_starts[tr + 1]; r += 16)
                    for (uint32_t c = tiles.mi_col_starts[tc]; c < tiles.mi_col_starts[tc + 1]; c += 16)
                        partition(r, c, AV1G_BLOCK_64X64, 0);
    }

    // random sparse coefficients for one tx block
    void fill_coeffs(int16_t *q, int txSz, int plane, int txType, int style) {
        const int n = av1g_tx_coeffs(txSz);
        memset(q, 0, sizeof(int16_t) * n);
        if (rng.chance(25)) return;  // all zero
        const uint16_t *scan = av1g_get_scan(txSz, txType);
        int eob;
        int r = rng.range(0, 99);
        if (r < 30) eob = rng.range(1, n < 4 ? n : 4);
        else if (r < 70) eob = rng.range(1, n < 32 ? n : 32);
        else eob = rng.range(1, n);
        int density = rng.range(5, 100);
        for (int c = 0; c < eob; c++) {
            if (c != eob - 1 && !rng.chance(density)) continue;
            int mag;
            int m = rng.range(0, 999);
            if (m < 600) mag = 1;
            else if (m < 800) mag = rng.range(2, 3);
            else if (m < 920) mag = rng.range(4, 15);
            else if (m < 985) mag = rng.range(16, 300);
            else if (m < 997) mag = rng.range(300, 5000);
            else mag = rng.range(5000, 32767);
            if (style == 1 && mag > 40) mag = 40;
            q[scan[c]] = (int16_t)(rng.chance(50) ? -mag : mag);
        }
    }
};

static bool write_file(const std::string &path, const void *p, size_t n) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return false;
    fwrite(p, 1, n, f);
    fclose(f);
    return true;
}
static std::vector<uint8_t> read_file(const std::string &path) {
    std::vector<uint8_t> v;
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return v;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    v.resize(n > 0 ? n : 0);
    if (n > 0 && fread(v.data(), 1, n, f) != (size_t)n) v.clear();
    fclose(f);
    return v;
}

struct FuzzCase {
    av1g_decision_record rec;
    std::vector<av1g_block> blocks;
    std::vector<uint32_t> btx;
    std::vector<av1g_txb> txb;
    std::vector<int16_t> coeffs;
};

// Build the random legal record of seed cs (deterministic).  pct444: % of the non-monochrome
// cases that are 4:4:4, drawn from a separate stream (0: the 4:2:0 / monochrome stream as before).
// pct10: % of the cases that are 10-bit, from another separate stream (0: 8-bit only).
static bool is_10bit_case(uint64_t cs, bool m2, int pct10) {
    Rng drng(cs ^ 0x10B17D3B7110B17Dull);
    return !m2 && pct10 > 0 && drng.chance(pct10);
}
static bool make_case(uint64_t cs, bool m2, int maxdim, int bigPct, FuzzCase &fc, int pct444 = 40, int pct10 = 0) {
    Rng rng(cs);
    Gen g(rng, m2);
    g_cov10 = is_10bit_case(cs, m2, pct10);
    // ---- frame parameters ----
    int W, H;
    int sz = rng.range(0, 99);
    static const int special[][2] = {{1, 1}, {7, 13}, {65, 65}, {64, 64}, {8, 8}, {2, 3}, {13, 7}, {16, 16}, {72, 40}, {129, 1}};
    if (sz < 10) { int k = rng.range(0, 9); W = special[k][0]; H = special[k][1]; }
    else if (sz < 30) { W = rng.range(1, 40); H = rng.range(1, 40); }
    else if (sz < 100 - bigPct) { W = rng.range(1, maxdim); H = rng.range(1, maxdim); }
    else {  // forced tiles: wider than 4096 or area > 4096*2304
        if (rng.chance(70)) { W = rng.range(4097, 9000); H = rng.range(1, 80); }
        else { W = rng.range(3000, 4096); H = rng.range(2400, 2600); }
    }
    av1g_decision_record &rec = fc.rec;
    memset(&rec, 0, sizeof rec);
    rec.fp.version = AV1G_DECISIONS_VERSION;
    rec.fp.width = W;
    rec.fp.height = H;
    rec.fp.bit_depth = g_cov10 ? 10 : 8;
    rec.fp.monochrome = m2 ? 0 : rng.chance(15);
    {
        Rng yrng(cs ^ 0x4444C0FFEE444444ull);   // own stream: the other cases keep their records
        if (!m2 && !rec.fp.monochrome && pct444 > 0 && yrng.chance(pct444)) {
            rec.fp.chroma_format = AV1G_CHROMA_444;
            g.ssx = g.ssy = 0;
        }
    }
    rec.fp.base_q_idx = (uint8_t)rng.range(1, 255);
    rec.fp.tx_mode_select = m2 ? 0 : rng.chance(60);
    rec.fp.reduced_tx_set = m2 ? 0 : rng.chance(30);
    rec.fp.tile_cols_log2 = -1;
    rec.fp.tile_rows_log2 = -1;
    rec.fp.allow_screen_content_tools = m2 ? 0 : rng.chance(15);
    if (!m2 && rng.chance(15)) {
        rec.fp.delta_q_y_dc = (int8_t)rng.range(-20, 20);
        rec.fp.delta_q_u_dc = (int8_t)rng.range(-20, 20);
        rec.fp.delta_q_u_ac = (int8_t)rng.range(-20, 20);
        if (rng.chance(50)) { rec.fp.delta_q_v_dc = (int8_t)rng.range(-20, 20); rec.fp.delta_q_v_ac = (int8_t)rng.range(-20, 20); }
        else { rec.fp.delta_q_v_dc = rec.fp.delta_q_u_dc; rec.fp.delta_q_v_ac = rec.fp.delta_q_u_ac; }
    }
    if (!m2 && rng.chance(30)) {
        rec.fp.using_qmatrix = 1;
        rec.fp.qm_y = (uint8_t)rng.range(0, 15);
        rec.fp.qm_u = (uint8_t)rng.range(0, 15);
        rec.fp.qm_v = rng.chance(50) ? rec.fp.qm_u : (uint8_t)rng.range(0, 15);
    }
    if (!m2 && rng.chance(35)) rec.fp.enable_filter_intra = 1;
    if (!m2 && rng.chance(35)) rec.fp.enable_intra_edge_filter = 1;
    if (!m2 && rng.chance(30)) {
        rec.fp.delta_q_present = 1;
        rec.fp.delta_q_res = (uint8_t)rng.range(0, 3);
    }
    // optional extra tiles for moderately sized pictures
    av1g_tiles mt;
    av1g_tile_layout(W, H, -1, -1, &mt);
    if (!m2 && rng.chance(15)) {
        av1g_tiles t2;
        int cl = mt.cols_log2 + rng.range(0, 2), rl = rng.range(0, 2);
        if (av1g_tile_layout(W, H, cl, rl, &t2) == AV1G_OK) { rec.fp.tile_cols_log2 = (int8_t)cl; rec.fp.tile_rows_log2 = (int8_t)rl; }
    }
    av1g_tile_layout(W, H, rec.fp.tile_cols_log2, rec.fp.tile_rows_log2, &g.tiles);
    // deblocking / CDEF parameters from their own stream (keeps the pre-M5.5 cases of each seed)
    Rng prng(cs ^ 0x5DEECE66Dull);
    if (!m2 && prng.chance(45)) {
        static const uint8_t hi[] = {63, 50, 40, 32, 24, 16, 8, 4};
        rec.fp.loop_filter_level[0] = (uint8_t)(prng.chance(50) ? hi[prng.range(0, 7)] : prng.range(0, 63));
        rec.fp.loop_filter_level[1] = (uint8_t)(prng.chance(20) ? 0 : prng.chance(50) ? rec.fp.loop_filter_level[0] : prng.range(0, 63));
        if (!rec.fp.loop_filter_level[0] && !rec.fp.loop_filter_level[1]) rec.fp.loop_filter_level[1] = 1;
        if (!rec.fp.monochrome) {
            rec.fp.loop_filter_level[2] = (uint8_t)(prng.chance(15) ? 0 : prng.range(0, 63));
            rec.fp.loop_filter_level[3] = (uint8_t)(prng.chance(15) ? 0 : prng.range(0, 63));
        }
        rec.fp.loop_filter_sharpness = (uint8_t)(prng.chance(50) ? 0 : prng.range(0, 7));
    }
    if (!m2 && prng.chance(45)) {
        static const uint8_t sec[4] = {0, 1, 2, 4};
        rec.fp.enable_cdef = 1;
        rec.fp.cdef_damping_minus_3 = (uint8_t)prng.range(0, 3);
        rec.fp.cdef_bits = (uint8_t)prng.range(0, 3);
        for (int i = 0; i < (1 << rec.fp.cdef_bits); i++) {
            rec.fp.cdef_y_pri[i] = (uint8_t)prng.range(0, 15);
            rec.fp.cdef_y_sec[i] = sec[prng.range(0, 3)];
            if (!rec.fp.monochrome) {
                rec.fp.cdef_uv_pri[i] = (uint8_t)prng.range(0, 15);
                rec.fp.cdef_uv_sec[i] = sec[prng.range(0, 3)];
            }
        }
    }
    g.pfSmooth = (rec.fp.loop_filter_level[0] || rec.fp.loop_filter_level[1] || rec.fp.enable_cdef) && prng.chance(50);
    g.W = W; g.H = H;
    g.MiCols = av1g_mi_cols(W); g.MiRows = av1g_mi_rows(H);
    g.mono = rec.fp.monochrome;
    g.txSelect = rec.fp.tx_mode_select;
    g.reduced = rec.fp.reduced_tx_set;
    g.maxDepthBias = rng.range(1, 3);
    g.dqBias = rec.fp.delta_q_present;
    g.fi = rec.fp.enable_filter_intra;
    g.gen_blocks();
    fc.blocks.swap(g.blocks);
    rec.n_blocks = (uint32_t)fc.blocks.size();
    rec.blocks = fc.blocks.data();
    fc.btx.assign(rec.n_blocks + 1, 0);
    rec.block_txb = fc.btx.data();
    av1g_layout(&rec, 0, 0);
    fc.txb.assign(rec.n_txb, av1g_txb());
    fc.coeffs.assign(rec.n_coeffs + 1, 0);
    rec.txb = fc.txb.data();
    rec.coeffs = fc.coeffs.data();
    if (av1g_layout(&rec, rec.n_txb, rec.n_coeffs) != AV1G_OK) { fprintf(stderr, "seed %llu: layout failed\n", (unsigned long long)cs); return false; }
    if (rec.fp.delta_q_present) {  // random per-SB qindex map, legalized
        const int sbCols = (g.MiCols + 15) >> 4, sbRows = (g.MiRows + 15) >> 4;
        std::vector<uint8_t> target((size_t)sbCols * sbRows), legal(target.size());
        const int mode = rng.range(0, 2);
        for (auto &t : target)
            t = (uint8_t)(mode == 0 ? rng.range(1, 255) : mode == 1 ? std::max(1, std::min(255, rec.fp.base_q_idx + rng.range(-12, 12)))
                                                              : (rng.chance(80) ? rec.fp.base_q_idx : rng.range(1, 255)));
        av1g_legalize_sb_qindex(rec.fp.base_q_idx, rec.fp.delta_q_res, target.data(), target.size(), legal.data());
        av1g_apply_sb_qindex(&rec, legal.data());
    }
    if (rec.fp.enable_cdef) {  // one random cdef_idx per 64x64 (all blocks of an SB share it)
        const int sbCols = (g.MiCols + 15) >> 4, sbRows = (g.MiRows + 15) >> 4;
        std::vector<int8_t> sbc((size_t)sbCols * sbRows);
        for (auto &v : sbc) v = (int8_t)prng.range(0, (1 << rec.fp.cdef_bits) - 1);
        for (uint32_t b = 0; b < rec.n_blocks; b++)
            rec.blocks[b].cdef_idx = sbc[(size_t)(rec.blocks[b].mi_row >> 4) * sbCols + (rec.blocks[b].mi_col >> 4)];
    }
    const int style = rng.chance(50) ? 1 : 0;
    for (uint32_t b = 0; b < rec.n_blocks; b++) {
        for (uint32_t t = rec.block_txb[b]; t < rec.block_txb[b + 1]; t++) {
            av1g_txb &x = rec.txb[t];
            int type = AV1G_DCT_DCT;
            if (x.plane == 0 && !m2) {
                int set = av1g_get_tx_set_intra(x.tx_size, rec.fp.reduced_tx_set);
                if (set > 0) {
                    do type = rng.range(0, 15); while (!av1g_tx_type_in_set_intra(set, type));
                }
                x.tx_type = (uint8_t)type;
            } else if (x.plane > 0) {
                type = av1g_chroma_tx_type(x.tx_size, rec.blocks[b].uv_mode, rec.fp.reduced_tx_set);
            }
            g.fill_coeffs(rec.coeffs + x.coeff_off, x.tx_size, x.plane, type, style);
        }
    }
    return true;
}

// ---------------------------------------------------------------------------
// lossless cases
// ---------------------------------------------------------------------------
static bool is_lossless_case(uint64_t cs, bool m2, int pctLL) {
    Rng lr(cs ^ 0x7105E1E55C0DEull);
    return !m2 && pctLL > 0 && lr.chance(pctLL);
}

// the source picture of a lossless case, every plane padded to the coded area (MiCols * 4 x MiRows * 4 >> ss)
struct LLSource {
    int bd = 8, np = 1, w[3] = {0, 0, 0}, h[3] = {0, 0, 0};
    std::vector<uint16_t> p[3];
    int at(int pl, int x, int y) const { return p[pl][(size_t)y * w[pl] + x]; }
};

// random content: per 16x16 cell one of noise / smooth gradient / flat region / extremes (or one style for all)
static void gen_source(Rng &r, LLSource &S) {
    const int mx = (1 << S.bd) - 1;
    const int caseStyle = r.range(0, 5);   // 0..3: one style everywhere, 4..5: mixed cells
    for (int pl = 0; pl < S.np; pl++) {
        const int w = S.w[pl], h = S.h[pl];
        S.p[pl].assign((size_t)w * h, 0);
        const double gx = (r.range(-200, 200) / 100.0) * (mx / 255.0), gy = (r.range(-200, 200) / 100.0) * (mx / 255.0);
        const int off = r.range(0, mx), noiseAmp = r.range(0, 3) << (S.bd - 8);
        const int cw = (w + 15) / 16, ch = (h + 15) / 16, fw = (w + 31) / 32, fh = (h + 31) / 32;
        std::vector<int> cellStyle((size_t)cw * ch), flatVal((size_t)fw * fh);
        for (auto &c : cellStyle) c = caseStyle < 4 ? caseStyle : (caseStyle == 4 ? r.range(0, 3) : (r.chance(60) ? 2 : r.range(0, 3)));
        for (auto &v : flatVal) v = r.chance(20) ? (r.chance(50) ? 0 : mx) : r.range(0, mx);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                int v;
                switch (cellStyle[(size_t)(y / 16) * cw + x / 16]) {
                case 0: v = r.range(0, mx); break;                                                              // noise
                case 1: v = off + (int)(gx * x + gy * y) + (noiseAmp ? r.range(-noiseAmp, noiseAmp) : 0); break;  // smooth
                case 2: v = flatVal[(size_t)(y / 32) * fw + x / 32]; break;                                      // flat
                default: v = r.chance(50) ? 0 : mx; break;                                                     // extremes
                }
                S.p[pl][(size_t)y * w + x] = (uint16_t)(v < 0 ? 0 : v > mx ? mx : v);
            }
    }
}

// closed loop: the levels of every tx block are the WHT of (source - av1core's exact prediction)
struct LLHook : av1g::TxbHook {
    const LLSource &S;
    int bad = 0;
    explicit LLHook(const LLSource &s) : S(s) {}
    void on_txb(const av1g_block &, uint32_t, av1g_txb &t, int16_t *q, const uint8_t *recon, int stride, int dcQ, int acQ, int,
                int) override {
        if (t.tx_size != AV1G_TX_4X4 || dcQ != 4 || acQ != 4) { bad++; return; }
        int32_t res[16];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++) {
                const int pred = S.bd == 8 ? recon[i * stride + j] : ((const uint16_t *)recon)[i * stride + j];
                res[i * 4 + j] = S.at(t.plane, t.x4 * 4 + j, t.y4 * 4 + i) - pred;
            }
        av1g_fwht4x4(res, q);
        if (t.plane == 0) t.tx_type = AV1G_DCT_DCT;
    }
};

// lossless coverage (generation-time counts in gen, coding-time ones here)
struct LLCov {
    Coverage gen;
    uint64_t cases = 0, fmt[2][3] = {}, tiles = 0, big = 0, sct = 0, reduced = 0, fiFrames = 0, efFrames = 0, efDir = 0;
    uint64_t colour[4] = {};      // default 1/13/6, identity 1/13/0, YCgCo-Re x/x/16, other
    uint64_t skip = 0, zeroNonSkip = 0, txbCoded = 0, txbZero = 0, golomb = 0, maxMag = 0, cflBlocks[2][22] = {};
    uint64_t srcMismatch = 0;
    void report() const {
        int missing = 0;
        printf("COVERAGE lossless: %llu cases (8-bit 4:2:0 %llu, 4:4:4 %llu, mono %llu; 10-bit 4:2:0 %llu, 4:4:4 %llu, mono %llu),"
               " tiles %llu, forced tiles %llu, screen content %llu, reduced set %llu, filter intra %llu, edge filter %llu"
               " (%llu directional blocks)\n",
               (unsigned long long)cases, (unsigned long long)fmt[0][0], (unsigned long long)fmt[0][1], (unsigned long long)fmt[0][2],
               (unsigned long long)fmt[1][0], (unsigned long long)fmt[1][1], (unsigned long long)fmt[1][2], (unsigned long long)tiles,
               (unsigned long long)big, (unsigned long long)sct, (unsigned long long)reduced, (unsigned long long)fiFrames,
               (unsigned long long)efFrames, (unsigned long long)efDir);
        for (int b = 0; b < 2; b++) for (int f = 0; f < 3; f++) missing += !fmt[b][f];
        missing += !tiles + !sct + !reduced + !fiFrames + !efDir;
        static const int sq[4] = {AV1G_BLOCK_8X8, AV1G_BLOCK_16X16, AV1G_BLOCK_32X32, AV1G_BLOCK_64X64};
        printf("  partitions (node 8/16/32/64 x NONE..VERT_4):");
        for (int k : sq) {
            printf(" [");
            for (int p = 0; p < (k == AV1G_BLOCK_8X8 ? 4 : 10); p++) {
                printf("%s%llu", p ? " " : "", (unsigned long long)gen.part[k][p]);
                missing += !gen.part[k][p];
            }
            printf("]");
        }
        printf("\n  block sizes:");
        for (int b = 0; b < 22; b++) if (b < 13 || b > 15) { printf(" %llu", (unsigned long long)gen.bsize[b]); missing += !gen.bsize[b]; }
        int ym = 0, uvm = 0;
        for (int m = 0; m < 13; m++) for (int d = 0; d < 7; d++) if (av1g_is_directional(m) || d == 3) ym += !gen.ymode[m][d];
        for (int m = 0; m < 14; m++) for (int d = 0; d < 7; d++) if ((m < 13 && av1g_is_directional(m)) || d == 3) uvm += !gen.uvmode[m][d];
        missing += ym + uvm;
        printf("\n  y_mode x angle missing %d, uv_mode x angle missing %d, cfl sign combos:", ym, uvm);
        for (int i = 0; i < 8; i++) { printf(" %llu", (unsigned long long)gen.cflSigns[i]); missing += !gen.cflSigns[i]; }
        printf("\n  CfL blocks 4:2:0 (4x4 4x8 8x4 8x8):");
        const int c420[4] = {AV1G_BLOCK_4X4, AV1G_BLOCK_4X8, AV1G_BLOCK_8X4, AV1G_BLOCK_8X8};
        for (int b : c420) { printf(" %llu", (unsigned long long)cflBlocks[0][b]); missing += !cflBlocks[0][b]; }
        printf(", 4:4:4 (4x4): %llu, filter intra modes:", (unsigned long long)cflBlocks[1][AV1G_BLOCK_4X4]);
        missing += !cflBlocks[1][AV1G_BLOCK_4X4];
        for (int i = 0; i < 5; i++) { printf(" %llu", (unsigned long long)gen.fiMode[i]); missing += !gen.fiMode[i]; }
        printf("\n  colour descriptions (default, identity 1/13/0, YCgCo-Re x/x/16, other): %llu %llu %llu %llu\n",
               (unsigned long long)colour[0], (unsigned long long)colour[1], (unsigned long long)colour[2], (unsigned long long)colour[3]);
        for (int i = 0; i < 4; i++) missing += !colour[i];
        printf("  skip blocks %llu, non-skip blocks without residual %llu, tx blocks coded %llu / all-zero %llu, golomb coeffs %llu,"
               " max |coeff| %llu\n", (unsigned long long)skip, (unsigned long long)zeroNonSkip, (unsigned long long)txbCoded,
               (unsigned long long)txbZero, (unsigned long long)golomb, (unsigned long long)maxMag);
        missing += !skip + !zeroNonSkip + !txbZero + !golomb;
        printf("  reconstruction != source: %llu cases\n", (unsigned long long)srcMismatch);
        printf("COVERAGE lossless missing items: %d\n", missing);
    }
};
static LLCov covLL;

// Build a lossless case of seed cs: random legal decisions (coefficients come from the closed loop) and its source.
static bool make_lossless_case(uint64_t cs, int maxdim, int bigPct, FuzzCase &fc, LLSource &S, int pct444, int pct10) {
    Rng rng(cs ^ 0x1055C1E5500AB1Eull);
    Gen g(rng, false);
    g.C = &covLL.gen;
    g.lossless = true;
    g_cov10 = false;
    int W, H;
    const int sz = rng.range(0, 99);
    static const int special[][2] = {{1, 1}, {7, 13}, {65, 65}, {64, 64}, {8, 8}, {2, 3}, {13, 7}, {16, 16}, {72, 40}, {129, 1}};
    if (sz < 10) { int k = rng.range(0, 9); W = special[k][0]; H = special[k][1]; }
    else if (sz < 35) { W = rng.range(1, 40); H = rng.range(1, 40); }
    else if (sz < 100 - bigPct) { W = rng.range(1, maxdim); H = rng.range(1, maxdim); }
    else { W = rng.range(4097, 6000); H = rng.range(1, 24); }   // forced tiles (kept short: closed loop on the CPU)
    av1g_decision_record &rec = fc.rec;
    memset(&rec, 0, sizeof rec);
    rec.fp.version = AV1G_DECISIONS_VERSION;
    rec.fp.width = W;
    rec.fp.height = H;
    rec.fp.bit_depth = (pct10 > 0 && rng.chance(pct10)) ? 10 : 8;
    rec.fp.monochrome = rng.chance(15);
    if (!rec.fp.monochrome && pct444 > 0 && rng.chance(pct444)) {
        rec.fp.chroma_format = AV1G_CHROMA_444;
        g.ssx = g.ssy = 0;
    }
    rec.fp.base_q_idx = 0;
    rec.fp.reduced_tx_set = rng.chance(30);
    rec.fp.tile_cols_log2 = -1;
    rec.fp.tile_rows_log2 = -1;
    rec.fp.allow_screen_content_tools = rng.chance(20);
    if (rng.chance(40)) rec.fp.enable_filter_intra = 1;
    if (rng.chance(40)) rec.fp.enable_intra_edge_filter = 1;
    // colour description (headers / AVIF only; the identity form 1/13/0 omits color_range in the sequence header)
    {
        const int k = rng.range(0, 99);
        const bool rgb444 = !rec.fp.monochrome && rec.fp.chroma_format == AV1G_CHROMA_444;
        if (k < 30) {
        } else if (k < 65 && rgb444) {
            rec.fp.color_description = 1; rec.fp.color_primaries = 1; rec.fp.transfer_characteristics = 13; rec.fp.matrix_coefficients = 0;
        } else if (k < 85) {
            rec.fp.color_description = 1; rec.fp.color_primaries = 1; rec.fp.transfer_characteristics = 13; rec.fp.matrix_coefficients = 16;
        } else {
            static const uint8_t cps[] = {1, 2, 9, 12}, tcs[] = {1, 2, 13, 16, 18}, mcs[] = {1, 2, 5, 6, 9, 16};
            rec.fp.color_description = 1;
            rec.fp.color_primaries = cps[rng.range(0, 3)];
            rec.fp.transfer_characteristics = tcs[rng.range(0, 4)];
            rec.fp.matrix_coefficients = rgb444 && rng.chance(30) ? 0 : mcs[rng.range(0, 5)];
        }
    }
    av1g_tiles mt;
    av1g_tile_layout(W, H, -1, -1, &mt);
    if (rng.chance(20)) {
        av1g_tiles t2;
        int cl = mt.cols_log2 + rng.range(0, 2), rl = rng.range(0, 2);
        if (av1g_tile_layout(W, H, cl, rl, &t2) == AV1G_OK) { rec.fp.tile_cols_log2 = (int8_t)cl; rec.fp.tile_rows_log2 = (int8_t)rl; }
    }
    av1g_tile_layout(W, H, rec.fp.tile_cols_log2, rec.fp.tile_rows_log2, &g.tiles);
    g.W = W; g.H = H;
    g.MiCols = av1g_mi_cols(W); g.MiRows = av1g_mi_rows(H);
    g.mono = rec.fp.monochrome;
    g.txSelect = 0;
    g.reduced = rec.fp.reduced_tx_set;
    g.maxDepthBias = rng.range(1, 3);
    g.fi = rec.fp.enable_filter_intra;
    g.gen_blocks();
    fc.blocks.swap(g.blocks);
    rec.n_blocks = (uint32_t)fc.blocks.size();
    rec.blocks = fc.blocks.data();
    fc.btx.assign(rec.n_blocks + 1, 0);
    rec.block_txb = fc.btx.data();
    av1g_layout(&rec, 0, 0);
    fc.txb.assign(rec.n_txb, av1g_txb());
    fc.coeffs.assign(rec.n_coeffs + 1, 0);
    rec.txb = fc.txb.data();
    rec.coeffs = fc.coeffs.data();
    if (av1g_layout(&rec, rec.n_txb, rec.n_coeffs) != AV1G_OK) {
        fprintf(stderr, "seed %llu: lossless layout failed\n", (unsigned long long)cs);
        return false;
    }
    S.bd = av1g_bit_depth(&rec.fp);
    S.np = rec.fp.monochrome ? 1 : 3;
    for (int p = 0; p < S.np; p++) {
        const int sx = p ? g.ssx : 0, sy = p ? g.ssy : 0;
        S.w[p] = (g.MiCols * 4) >> sx;
        S.h[p] = (g.MiRows * 4) >> sy;
    }
    Rng srng(cs ^ 0x5005CEC0DE5EEDull);
    gen_source(srng, S);
    return true;
}

// Closed-loop lossless encode of a case: levels from the source, then skip = 1 for most blocks without residual and
// a re-encode (the reconstruction does not change: their prediction is the source already).  Returns the
// av1g_encode status of the final encode (o holds its output); recon == source is checked by the caller.
static int encode_lossless_case(FuzzCase &fc, const LLSource &S, uint64_t cs, const av1g_encode_opts &eo, av1g_output &o,
                                std::string &why) {
    av1g_decision_record &rec = fc.rec;
    LLHook hook(S);
    int rc = av1g::encode_with_hook(&rec, &eo, &o, &hook);
    if (rc != AV1G_OK) { why = o.error; return rc; }
    av1g_output_free(&o);
    if (hook.bad) { why = "hook: not a 4x4 / qindex 0 tx block"; return AV1G_ERR_INTERNAL; }
    Rng k(cs ^ 0x5C1BB10C5ull);
    for (uint32_t b = 0; b < rec.n_blocks; b++) {
        bool zero = true;
        for (uint32_t t = rec.block_txb[b]; t < rec.block_txb[b + 1] && zero; t++) zero = rec.txb[t].eob == 0;
        if (zero && k.chance(85)) rec.blocks[b].skip = 1;
    }
    rc = av1g_encode(&rec, &eo, &o);
    if (rc != AV1G_OK) why = o.error;
    return rc;
}

// av1core's reconstruction (cropped planes) == the visible source?
static bool recon_is_source(const av1g_output &o, const LLSource &S, std::string &why) {
    for (int p = 0; p < S.np; p++) {
        const int w = (int)o.recon_w[p], h = (int)o.recon_h[p];
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++) {
                const int v = S.bd == 8 ? o.recon[p][(size_t)y * w + x] : ((const uint16_t *)o.recon[p])[(size_t)y * w + x];
                if (v != S.at(p, x, y)) {
                    char b[160];
                    snprintf(b, sizeof b, "recon != source: plane %d (%d,%d) recon %d source %d", p, x, y, v, S.at(p, x, y));
                    why = b;
                    return false;
                }
            }
    }
    return true;
}

// lossless 4x4 WHT self-test: forward -> dequantize at qindex 0 -> spec inverse == input, at both bit depths
static bool wht_selftest() {
    Rng r(12345);
    for (int it = 0; it < 200000; it++) {
        const int bd = (it & 1) ? 10 : 8, mx = (1 << bd) - 1;
        int32_t res[16], dq[16], back[16];
        int16_t q[16];
        const int style = r.range(0, 2);
        for (int i = 0; i < 16; i++) res[i] = style == 0 ? r.range(-mx, mx) : style == 1 ? (r.chance(50) ? mx : -mx) : r.range(-3, 3);
        av1g_fwht4x4(res, q);
        for (int i = 0; i < 16; i++) dq[i] = q[i] ? (bd == 8 ? av1g_dequant_coef<8>(q[i], 4, 1) : av1g_dequant_coef<10>(q[i], 4, 1)) : 0;
        uint32_t err = 0;
        if (bd == 8) av1g_inv_wht4x4<8>(dq, back, &err); else av1g_inv_wht4x4<10>(dq, back, &err);
        if (err || memcmp(back, res, sizeof res)) { printf("WHT self-test FAILED (iteration %d, %d bits)\n", it, bd); return false; }
    }
    return true;
}

#ifndef AV1G_FUZZ_NO_MAIN
int main(int argc, char **argv) {
    uint64_t seed = 1;
    int count = 100, maxdim = 300, bigPct = 2, pct444 = 40, pct10 = 40, pctLL = 25;
    bool m2 = false, keep = false, useAom = true, quiet = false;
    std::string out = default_out_dir("av1g_fuzz");
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto nx = [&]() { return std::string(argv[++i]); };
        if (a == "--seed") seed = strtoull(nx().c_str(), 0, 10);
        else if (a == "--count") count = atoi(nx().c_str());
        else if (a == "--mode") m2 = nx() == "m2";
        else if (a == "--out") out = nx();
        else if (a == "--keep") keep = true;
        else if (a == "--maxdim") maxdim = atoi(nx().c_str());
        else if (a == "--big") bigPct = atoi(nx().c_str());
        else if (a == "--yuv444") pct444 = atoi(nx().c_str());
        else if (a == "--depth10") pct10 = atoi(nx().c_str());
        else if (a == "--lossless") pctLL = atoi(nx().c_str());
        else if (a == "--no-aom") useAom = false;
        else if (a == "--quiet") quiet = true;
        else { fprintf(stderr, "unknown arg %s\n", a.c_str()); return 2; }
    }
    mkdir(out.c_str(), 0755);
    if (pctLL > 0 && !wht_selftest()) return 1;
    int fails = 0, encFails = 0, parseFails = 0, llCases = 0;
    uint64_t totalFixed = 0, totalBlocks = 0, totalTxb = 0;
    for (int ci = 0; ci < count; ci++) {
        const uint64_t cs = seed + ci;
        FuzzCase fc;
        LLSource llSrc;
        const bool ll = is_lossless_case(cs, m2, pctLL);
        if (ll ? !make_lossless_case(cs, maxdim, bigPct, fc, llSrc, pct444, pct10) : !make_case(cs, m2, maxdim, bigPct, fc, pct444, pct10))
            return 1;
        av1g_decision_record &rec = fc.rec;
        const bool is10 = av1g_bit_depth(&rec.fp) == 10;
        const int bps = is10 ? 2 : 1;   // recon bytes per sample
        g_pfBd10 = is10;
        totalBlocks += rec.n_blocks;
        totalTxb += rec.n_txb;
        av1g_encode_opts eo = {AV1G_ENC_FIX_RANGE | AV1G_ENC_TD | AV1G_ENC_CHECK_EDGES, -1};
        av1g_output o;
        int rc;
        bool okSrc = true;
        std::string llWhy;
        if (ll) {   // closed loop on the source, no range fixing: true residuals are conformant
            eo.flags = AV1G_ENC_TD | AV1G_ENC_CHECK_EDGES;
            rc = encode_lossless_case(fc, llSrc, cs, eo, o, llWhy);
            if (rc == AV1G_OK) okSrc = recon_is_source(o, llSrc, llWhy);
        } else {
            rc = av1g_encode(&rec, &eo, &o);
        }
        char base[512];
        snprintf(base, sizeof base, "%s/case_%llu", out.c_str(), (unsigned long long)cs);
        if (rc != AV1G_OK) {
            fprintf(stderr, "seed %llu: %sencode error %d: %s\n", (unsigned long long)cs, ll ? "lossless " : "", rc,
                    ll ? llWhy.c_str() : o.error);
            encFails++;
            continue;
        }
        if (ll) {
            llCases++;
            covLL.cases++;
            covLL.fmt[is10 ? 1 : 0][rec.fp.monochrome ? 2 : rec.fp.chroma_format == AV1G_CHROMA_444 ? 1 : 0]++;
            covLL.tiles += o.tile_cols * o.tile_rows > 1;
            covLL.big += rec.fp.width > 4096;
            covLL.sct += rec.fp.allow_screen_content_tools;
            covLL.reduced += rec.fp.reduced_tx_set;
            covLL.fiFrames += rec.fp.enable_filter_intra;
            covLL.efFrames += rec.fp.enable_intra_edge_filter;
            {
                const int mc = rec.fp.matrix_coefficients;
                covLL.colour[!rec.fp.color_description ? 0 : (mc == 0 && rec.fp.color_primaries == 1 && rec.fp.transfer_characteristics == 13) ? 1
                             : mc == 16 ? 2 : 3]++;
            }
            const int lssx = av1g_ssx(&rec.fp), lssy = av1g_ssy(&rec.fp);
            for (uint32_t b = 0; b < rec.n_blocks; b++) {
                const av1g_block &k = rec.blocks[b];
                if (rec.fp.enable_intra_edge_filter) covLL.efDir += av1g_is_directional(k.y_mode);
                if (k.uv_mode == AV1G_UV_CFL_PRED && av1g_has_chroma(k.bsize, k.mi_row, k.mi_col, rec.fp.monochrome, lssx, lssy))
                    covLL.cflBlocks[lssx ? 0 : 1][k.bsize]++;
                if (k.skip) { covLL.skip++; continue; }
                bool any = false;
                for (uint32_t t = rec.block_txb[b]; t < rec.block_txb[b + 1]; t++) {
                    const av1g_txb &x = rec.txb[t];
                    if (!x.eob) { covLL.txbZero++; continue; }
                    any = true;
                    covLL.txbCoded++;
                    const int16_t *q = rec.coeffs + x.coeff_off;
                    for (int i = 0; i < 16; i++) {
                        const uint64_t a = q[i] < 0 ? -q[i] : q[i];
                        covLL.golomb += a > 14;
                        if (a > covLL.maxMag) covLL.maxMag = a;
                    }
                }
                covLL.zeroNonSkip += !any;
            }
            if (!okSrc) { covLL.srcMismatch++; printf("seed %llu: LOSSLESS %s\n", (unsigned long long)cs, llWhy.c_str()); }
        }
        totalFixed += o.n_fixed_txb;
        if (!ll) cov.mono += rec.fp.monochrome;
        const bool is444 = rec.fp.chroma_format == AV1G_CHROMA_444;
        const int ssx = av1g_ssx(&rec.fp), ssy = av1g_ssy(&rec.fp);
        const bool lfOn10 = rec.fp.loop_filter_level[0] || rec.fp.loop_filter_level[1];
        if (is10 && !ll) {
            cov.c10++;
            cov.c10Fmt[rec.fp.monochrome ? 2 : is444 ? 1 : 0]++;
            cov.c10Tiles += o.tile_cols * o.tile_rows > 1;
            cov.c10Lf += lfOn10;
            cov.c10Cdef += rec.fp.enable_cdef;
            cov.c10Dq += rec.fp.delta_q_present;
            cov.c10Qm += rec.fp.using_qmatrix;
            cov.c10Fi += rec.fp.enable_filter_intra;
            cov.c10Ef += rec.fp.enable_intra_edge_filter;
            cov.c10TxSel += rec.fp.tx_mode_select;
            cov.c10Reduced += rec.fp.reduced_tx_set;
            cov.c10Sct += rec.fp.allow_screen_content_tools;
            cov.c10DeltaQ += rec.fp.delta_q_y_dc || rec.fp.delta_q_u_dc || rec.fp.delta_q_u_ac || rec.fp.delta_q_v_dc || rec.fp.delta_q_v_ac;
            cov.c10Big += rec.fp.width > 4096 || (uint64_t)rec.fp.width * rec.fp.height > 4096ull * 2304;
            cov.c10Fixed += o.n_fixed_txb;
            if (lfOn10) {
                cov.lfSharp10[rec.fp.loop_filter_sharpness]++;
                if (!rec.fp.monochrome && (!rec.fp.loop_filter_level[2] || !rec.fp.loop_filter_level[3])) cov.lfChromaOff10++;
            }
            if (rec.fp.enable_cdef) {
                cov.pfBoth10 += lfOn10;
                cov.cdefBits10[rec.fp.cdef_bits]++;
                cov.cdefDamp10[rec.fp.cdef_damping_minus_3]++;
                for (int i = 0; i < (1 << rec.fp.cdef_bits); i++) cov.cdefSec4_10 += rec.fp.cdef_y_sec[i] == 4 || rec.fp.cdef_uv_sec[i] == 4;
            }
            if (rec.fp.enable_intra_edge_filter)
                for (uint32_t b = 0; b < rec.n_blocks; b++) cov.efDir10 += av1g_is_directional(rec.blocks[b].y_mode);
            if (rec.fp.using_qmatrix) { cov.qmLevel10[rec.fp.qm_y]++; cov.qmLevel10[rec.fp.qm_u]++; cov.qmLevel10[rec.fp.qm_v]++; }
            if (rec.fp.delta_q_present) cov.dqRes10[rec.fp.delta_q_res]++;
        }
        if (is444 && !ll) {
            cov.c444++;
            cov.c444Tiles += o.tile_cols * o.tile_rows > 1;
            cov.c444Lf += rec.fp.loop_filter_level[0] || rec.fp.loop_filter_level[1];
            cov.c444Cdef += rec.fp.enable_cdef;
            cov.c444Dq += rec.fp.delta_q_present;
            cov.c444Qm += rec.fp.using_qmatrix;
            cov.c444Fi += rec.fp.enable_filter_intra;
            cov.c444Ef += rec.fp.enable_intra_edge_filter;
            cov.c444TxSel += rec.fp.tx_mode_select;
            cov.c444Reduced += rec.fp.reduced_tx_set;
            cov.c444Sct += rec.fp.allow_screen_content_tools;
            cov.c444DeltaQ += rec.fp.delta_q_u_dc || rec.fp.delta_q_u_ac || rec.fp.delta_q_v_dc || rec.fp.delta_q_v_ac;
            cov.c444Big += rec.fp.width > 4096 || (uint64_t)rec.fp.width * rec.fp.height > 4096ull * 2304;
        }
        if (!ll) {   // lossy coverage (lossless cases have their own, covLL)
        cov.sct += rec.fp.allow_screen_content_tools;
        cov.fiFrames += rec.fp.enable_filter_intra;
        const bool lfOn = rec.fp.loop_filter_level[0] || rec.fp.loop_filter_level[1];
        if (lfOn) {
            cov.lf++;
            cov.lfSharp[rec.fp.loop_filter_sharpness]++;
            if (!rec.fp.monochrome && (!rec.fp.loop_filter_level[2] || !rec.fp.loop_filter_level[3])) cov.lfChromaOff++;
        }
        if (rec.fp.enable_cdef) {
            cov.cdef++;
            cov.pfBoth += lfOn;
            cov.cdefBits[rec.fp.cdef_bits]++;
            cov.cdefDamp[rec.fp.cdef_damping_minus_3]++;
            for (int i = 0; i < (1 << rec.fp.cdef_bits); i++) cov.cdefSec4 += rec.fp.cdef_y_sec[i] == 4 || rec.fp.cdef_uv_sec[i] == 4;
        }
        if (rec.fp.enable_intra_edge_filter) {
            cov.efFrames++;
            for (uint32_t b = 0; b < rec.n_blocks; b++) cov.efDirBlocks += av1g_is_directional(rec.blocks[b].y_mode);
        }
        if (rec.fp.using_qmatrix) {
            cov.qm++;
            cov.qmSep += rec.fp.qm_v != rec.fp.qm_u && !rec.fp.monochrome;
            cov.qmLevel[rec.fp.qm_y]++; cov.qmLevel[rec.fp.qm_u]++; cov.qmLevel[rec.fp.qm_v]++;
        }
        if (rec.fp.delta_q_present) {
            cov.dq++;
            cov.dqRes[rec.fp.delta_q_res]++;
            int cur = rec.fp.base_q_idx;
            for (uint32_t b = 0; b < rec.n_blocks; b++) {
                const av1g_block &k = rec.blocks[b];
                if (b > 0 && rec.blocks[b - 1].mi_row >> 4 == k.mi_row >> 4 && rec.blocks[b - 1].mi_col >> 4 == k.mi_col >> 4) continue;
                if (k.bsize == AV1G_BLOCK_64X64 && k.skip) { cov.dqSkipSb++; continue; }
                int red = ((int)k.sb_qindex - cur) / (1 << rec.fp.delta_q_res);
                if (red >= 3 || red <= -3) cov.dqBigJump++;
                cur = k.sb_qindex;  // (tile resets ignored: coverage only)
            }
        }
        if (rec.fp.delta_q_y_dc || rec.fp.delta_q_u_dc || rec.fp.delta_q_u_ac || rec.fp.delta_q_v_dc || rec.fp.delta_q_v_ac) cov.deltaq++;
        if (rec.fp.delta_q_v_dc != rec.fp.delta_q_u_dc || rec.fp.delta_q_v_ac != rec.fp.delta_q_u_ac) cov.sepuv++;
        if (o.tile_cols * o.tile_rows > 1) cov.multiTile++;
        if (o.tile_cols > cov.maxTileCols) cov.maxTileCols = o.tile_cols;
        if (o.tile_rows > cov.maxTileRows) cov.maxTileRows = o.tile_rows;
        for (uint32_t b = 0; b < rec.n_blocks; b++) {
            if (rec.blocks[b].skip) continue;
            for (uint32_t t = rec.block_txb[b]; t < rec.block_txb[b + 1]; t++) {
                const av1g_txb &x = rec.txb[t];
                if (!x.eob) continue;
                if (x.plane == 0) cov.txLuma[x.tx_size][x.tx_type]++; else cov.txChroma[x.tx_size]++;
                if (x.plane && is444) { cov.txChroma444[x.tx_size]++; cov.txChromaType444[x.tx_size][x.tx_type]++; }
                if (is10) { if (x.plane == 0) cov.txLuma10[x.tx_size][x.tx_type]++; else cov.txChroma10[x.tx_size]++; }
                const int16_t *q = rec.coeffs + x.coeff_off;
                // 10 bits: dequantized values that hit the +-2^17 clamp (spec 7.12.3 1f)
                int dcQ = 0, acQ = 0, den = 1;
                const uint8_t *qmw = nullptr;
                if (is10) {
                    const av1g_block &k = rec.blocks[b];
                    const int qi = rec.fp.delta_q_present ? k.sb_qindex : rec.fp.base_q_idx;
                    const int ddc = x.plane == 0 ? rec.fp.delta_q_y_dc : x.plane == 1 ? rec.fp.delta_q_u_dc : rec.fp.delta_q_v_dc;
                    const int dac = x.plane == 0 ? 0 : x.plane == 1 ? rec.fp.delta_q_u_ac : rec.fp.delta_q_v_ac;
                    dcQ = av1g_dc_q<10>(qi + ddc);
                    acQ = av1g_ac_q<10>(qi + dac);
                    den = av1g_dq_denom(x.tx_size);
                    const int lvl = x.plane == 0 ? rec.fp.qm_y : x.plane == 1 ? rec.fp.qm_u : rec.fp.qm_v;
                    qmw = av1g_qm(rec.fp.using_qmatrix, lvl, x.plane, x.tx_size, x.tx_type);
                }
                for (int i = 0, n = av1g_tx_coeffs(x.tx_size); i < n; i++) {
                    uint64_t a = q[i] < 0 ? -q[i] : q[i];
                    if (a > 14) cov.golomb++;
                    if (a > cov.maxMag) cov.maxMag = a;
                    if (is10 && a) {
                        if (a > 14) cov.c10Golomb++;
                        if (a > cov.maxMag10) cov.maxMag10 = a;
                        const int64_t raw = ((int64_t)a * av1g_qm_step(i == 0 ? dcQ : acQ, qmw, i)) & 0xFFFFFF;
                        if (raw / den > (1 << 17) - 1) cov.c10Clamp++;
                    }
                }
            }
        }
        }   // !ll coverage
        // ---- parser round trip: OBUs -> record -> OBUs must be identical ----
        bool okParse = true;
        {
            av1g_parsed *pr = nullptr;
            int prc = av1g_parse(o.obu, o.obu_size, &pr);
            std::string why;
            if (prc != AV1G_OK) why = std::string("parse error: ") + pr->error;
            else {
                av1g_decision_record &q = pr->rec;
                if (q.n_blocks != rec.n_blocks || q.n_txb != rec.n_txb) why = "block/txb count";
                if (q.fp.chroma_format != rec.fp.chroma_format || q.fp.monochrome != rec.fp.monochrome)
                    why = "frame params (chroma format)";
                if (av1g_bit_depth(&q.fp) != av1g_bit_depth(&rec.fp)) why = "frame params (bit depth)";
                {   // effective colour description (the parser maps 1/13/6 to color_description 0)
                    auto cicp = [](const av1g_frame_params &f) {
                        return f.color_description ? (f.color_primaries << 16 | f.transfer_characteristics << 8 | f.matrix_coefficients)
                                                   : (1 << 16 | 13 << 8 | 6);
                    };
                    if (q.fp.base_q_idx != rec.fp.base_q_idx || cicp(q.fp) != cicp(rec.fp))
                        why = "frame params (base_q_idx / colour description)";
                }
                if (q.fp.using_qmatrix != rec.fp.using_qmatrix || q.fp.delta_q_present != rec.fp.delta_q_present ||
                    q.fp.enable_filter_intra != rec.fp.enable_filter_intra ||
                    q.fp.enable_intra_edge_filter != rec.fp.enable_intra_edge_filter ||
                    (rec.fp.using_qmatrix && (q.fp.qm_y != rec.fp.qm_y || q.fp.qm_u != rec.fp.qm_u ||
                                              (!rec.fp.monochrome && q.fp.qm_v != rec.fp.qm_v))) ||
                    (rec.fp.delta_q_present && q.fp.delta_q_res != rec.fp.delta_q_res))
                    why = "frame params (qm / delta-q)";
                {
                    av1g_pf_frame a, bb;
                    memset(&a, 0, sizeof a);
                    memset(&bb, 0, sizeof bb);
                    av1g_pf_frame_init(&rec.fp, &a);
                    av1g_pf_frame_init(&q.fp, &bb);
                    bool same = a.sharpness == bb.sharpness && a.enableCdef == bb.enableCdef;
                    for (int i = 0; i < 4; i++) same &= a.lfLevel[i] == bb.lfLevel[i];
                    if (a.enableCdef) {
                        same &= a.cdefDamping == bb.cdefDamping && rec.fp.cdef_bits == q.fp.cdef_bits;
                        for (int i = 0; i < (1 << rec.fp.cdef_bits); i++)
                            same &= a.cdefYPri[i] == bb.cdefYPri[i] && a.cdefYSec[i] == bb.cdefYSec[i] &&
                                    (rec.fp.monochrome || (a.cdefUVPri[i] == bb.cdefUVPri[i] && a.cdefUVSec[i] == bb.cdefUVSec[i]));
                    }
                    if (!same && why.empty()) why = "frame params (deblocking / CDEF)";
                }
                for (uint32_t b = 0; why.empty() && b < rec.n_blocks; b++) {
                    const av1g_block &x = rec.blocks[b], &y = q.blocks[b];
                    bool hc = av1g_has_chroma(x.bsize, x.mi_row, x.mi_col, rec.fp.monochrome, ssx, ssy);
                    if (x.mi_row != y.mi_row || x.mi_col != y.mi_col || x.bsize != y.bsize || x.skip != y.skip ||
                        x.tx_depth != y.tx_depth || x.y_mode != y.y_mode || x.angle_delta_y != y.angle_delta_y ||
                        (hc && (x.uv_mode != y.uv_mode || x.angle_delta_uv != y.angle_delta_uv ||
                                x.cfl_alpha_u != y.cfl_alpha_u || x.cfl_alpha_v != y.cfl_alpha_v)) ||
                        (rec.fp.delta_q_present && !(x.bsize == AV1G_BLOCK_64X64 && x.skip) && x.sb_qindex != y.sb_qindex) ||
                        x.filter_intra != y.filter_intra || (rec.fp.enable_cdef && !x.skip && x.cdef_idx != y.cdef_idx))
                        why = "block fields " + std::to_string(b);
                    for (uint32_t t = rec.block_txb[b]; why.empty() && t < rec.block_txb[b + 1]; t++) {
                        const av1g_txb &tx = rec.txb[t], &ty = q.txb[t];
                        if (tx.eob != ty.eob || (tx.eob && tx.tx_type != ty.tx_type)) why = "txb " + std::to_string(t);
                        if (!x.skip && memcmp(rec.coeffs + tx.coeff_off, q.coeffs + ty.coeff_off, 2 * av1g_tx_coeffs(tx.tx_size)))
                            why = "coeffs of txb " + std::to_string(t);
                    }
                }
                if (why.empty()) {
                    av1g_encode_opts e2 = {AV1G_ENC_TD | AV1G_ENC_NO_RECON, -1};
                    av1g_output o2;
                    int rc2 = av1g_encode(&q, &e2, &o2);
                    if (rc2 != AV1G_OK) why = std::string("re-encode: ") + o2.error;
                    else if (o2.obu_size != o.obu_size || memcmp(o2.obu, o.obu, o.obu_size)) why = "re-encoded bytes differ";
                    av1g_output_free(&o2);
                }
            }
            av1g_parsed_free(pr);
            if (!why.empty()) {
                okParse = false;
                parseFails++;
                printf("seed %llu: PARSE ROUND TRIP FAILED: %s\n", (unsigned long long)cs, why.c_str());
            }
        }
        std::string obuPath = std::string(base) + ".obu", yuvPath = std::string(base) + ".yuv";
        write_file(obuPath, o.obu, o.obu_size);
        std::vector<uint8_t> ref;
        for (int p = 0; p < 3; p++)   // the decoders write 10-bit samples as 16-bit little endian (this host's order)
            if (o.recon[p]) ref.insert(ref.end(), o.recon[p], o.recon[p] + (size_t)o.recon_w[p] * o.recon_h[p] * bps);
        // dav1d
        std::string d1 = std::string(base) + ".dav1d.yuv", d2 = std::string(base) + ".aom.yuv";
        std::string cmd = DAV1D + " -q -i " + obuPath + " --demuxer section5 --muxer yuv -o " + d1 + " >/dev/null 2>&1";
        int r1 = system(cmd.c_str());
        std::vector<uint8_t> v1 = read_file(d1);
        bool ok1 = r1 == 0 && v1 == ref;
        bool ok2 = true;
        std::vector<uint8_t> v2;
        if (useAom) {
            cmd = AOMDEC + " --rawvideo -o " + d2 + " " + obuPath + " >/dev/null 2>&1";
            int r2 = system(cmd.c_str());
            v2 = read_file(d2);
            if (rec.fp.monochrome && v2.size() > ref.size()) v2.resize(ref.size());  // aomdec appends neutral chroma
            ok2 = r2 == 0 && v2 == ref;
        }
        // ---- post-filter API cross-check + path coverage: pre-filter recon -> shared filters ----
        bool okPf = true;
        if (rec.fp.loop_filter_level[0] || rec.fp.loop_filter_level[1] || rec.fp.enable_cdef) {
            av1g_encode_opts e3 = {AV1G_ENC_PREFILTER_RECON, -1};
            av1g_output o3;
            if (av1g_encode(&rec, &e3, &o3) != AV1G_OK) okPf = false;
            else {
                const int np = rec.fp.monochrome ? 1 : 3;
                std::vector<uint8_t> pl[3], pl2[3];   // bytes; 10-bit: uint16_t samples (strides in samples)
                uint8_t *pp[3] = {nullptr, nullptr, nullptr}, *pp2[3] = {nullptr, nullptr, nullptr};
                int st[3] = {0, 0, 0};
                for (int p = 0; p < np; p++) {
                    pl[p].assign(o3.recon[p], o3.recon[p] + (size_t)o3.recon_w[p] * o3.recon_h[p] * bps);
                    pl2[p] = pl[p];
                    pp[p] = pl[p].data();
                    pp2[p] = pl2[p].data();
                    st[p] = (int)o3.recon_w[p];
                }
                // (a) the library API
                if (is10) {
                    uint16_t *pp16[3] = {(uint16_t *)pp2[0], (uint16_t *)pp2[1], (uint16_t *)pp2[2]};
                    if (av1g_postfilter16(&rec, pp16, st) != AV1G_OK) okPf = false;
                } else if (av1g_postfilter(&rec, pp2, st) != AV1G_OK) okPf = false;
                // (b) the header functions here (counts paths), maps from av1g_pf_mark_block
                av1g_pf_frame f;
                memset(&f, 0, sizeof f);
                av1g_pf_frame_init(&rec.fp, &f);
                std::vector<uint8_t> sk((size_t)f.MiRows * f.MiCols), t0(sk.size()), t1((size_t)(f.MiRows >> ssy) * (f.MiCols >> ssx)),
                    t2(t1.size());
                std::vector<int8_t> cd((size_t)f.sbCols * f.sbRows, -1);
                f.skip = sk.data(); f.lfTx[0] = t0.data(); f.lfTx[1] = t1.data(); f.lfTx[2] = t2.data(); f.cdefIdx = cd.data();
                for (uint32_t b = 0; b < rec.n_blocks; b++) av1g_pf_mark_block(&f, &rec.blocks[b]);
                for (int p = 0; p < np; p++) { f.plane[p] = pp[p]; f.stride[p] = st[p]; }
                if (is10) av1g_deblock_frame<10>(&f); else av1g_deblock_frame<8>(&f);
                std::vector<uint16_t> outp[3];
                uint8_t *dst[3] = {pp[0], pp[1], pp[2]};
                if (f.enableCdef) {
                    for (int p = 0; p < np; p++) { outp[p].assign(pl[p].size(), 0); dst[p] = (uint8_t *)outp[p].data(); }
                    if (is10) av1g_cdef_frame<10>(&f, dst, st); else av1g_cdef_frame<8>(&f, dst, st);
                }
                size_t off = 0;
                for (int p = 0; p < np && okPf; p++) {
                    const int w = (int)o.recon_w[p], h = (int)o.recon_h[p];
                    for (int y = 0; y < h && okPf; y++) {
                        if (memcmp(dst[p] + (size_t)y * st[p] * bps, ref.data() + off + (size_t)y * w * bps, (size_t)w * bps) ||
                            memcmp(pp2[p] + (size_t)y * st[p] * bps, ref.data() + off + (size_t)y * w * bps, (size_t)w * bps)) {
                            okPf = false;
                            printf("   post-filter API mismatch: plane %d row %d\n", p, y);
                        }
                    }
                    off += (size_t)w * h * bps;
                }
            }
            av1g_output_free(&o3);
        }
        ok1 = ok1 && okPf;
        if (!quiet || !(ok1 && ok2 && okSrc))
            printf("seed %llu %dx%d%s%s%s q%d lf%d/%d/%d/%d cdef%d txsel%d red%d tiles %ux%u blocks %u txb %u fixed %u bytes %zu: dav1d %s aomdec %s\n",
                   (unsigned long long)cs, (int)rec.fp.width, (int)rec.fp.height, rec.fp.monochrome ? " mono" : is444 ? " 444" : "",
                   is10 ? " 10bit" : "", ll ? (okSrc ? " lossless" : " lossless(RECON!=SOURCE)") : "",
                   rec.fp.base_q_idx,
                   rec.fp.loop_filter_level[0], rec.fp.loop_filter_level[1], rec.fp.loop_filter_level[2], rec.fp.loop_filter_level[3],
                   rec.fp.enable_cdef ? 1 + rec.fp.cdef_bits : 0, rec.fp.tx_mode_select, rec.fp.reduced_tx_set, o.tile_cols, o.tile_rows, rec.n_blocks, rec.n_txb,
                   o.n_fixed_txb, o.obu_size, ok1 ? "OK" : (r1 ? "ERR" : "MISMATCH"),
                   useAom ? (ok2 ? "OK" : "MISMATCH/ERR") : "skip");
        if (!(ok1 && ok2 && okParse && okSrc)) {
            fails++;
            write_file(yuvPath, ref.data(), ref.size());
            if (!ok1 && v1.size() == ref.size()) {
                size_t i = 0;   // sample index
                auto smp = [&](const std::vector<uint8_t> &v, size_t k) { return bps == 2 ? v[2 * k] | v[2 * k + 1] << 8 : v[k]; };
                while (i < ref.size() / bps && smp(v1, i) == smp(ref, i)) i++;
                size_t ysz = (size_t)o.recon_w[0] * o.recon_h[0];
                int p = i < ysz ? 0 : (i < ysz + (size_t)o.recon_w[1] * o.recon_h[1] ? 1 : 2);
                size_t off = p == 0 ? i : p == 1 ? i - ysz : i - ysz - (size_t)o.recon_w[1] * o.recon_h[1];
                printf("   first dav1d mismatch: plane %d x %zu y %zu (ours %d theirs %d)\n", p, off % o.recon_w[p],
                       off / o.recon_w[p], smp(ref, i), smp(v1, i));
            } else if (!ok1) {
                printf("   dav1d output size %zu vs %zu\n", v1.size(), ref.size());
            }
        } else if (!keep) {
            remove(obuPath.c_str());
        }
        remove(d1.c_str());
        remove(d2.c_str());
        av1g_output_free(&o);
        fflush(stdout);
    }
    if (count > llCases) cov.report();
    if (llCases) covLL.report();
    printf("SUMMARY seeds %llu..%llu mode %s: %d cases (%d lossy: %llu 4:4:4, %llu monochrome, %llu 10-bit; %d lossless, %llu with "
           "reconstruction != source), %d mismatches (%d parser round-trip failures), %d encode errors, blocks %llu, txb %llu, "
           "range-fixed txb %llu\n",
           (unsigned long long)seed, (unsigned long long)(seed + count - 1), m2 ? "m2" : "full", count, count - llCases,
           (unsigned long long)cov.c444, (unsigned long long)cov.mono, (unsigned long long)cov.c10, llCases,
           (unsigned long long)covLL.srcMismatch, fails, parseFails, encFails, (unsigned long long)totalBlocks,
           (unsigned long long)totalTxb, (unsigned long long)totalFixed);
    return (fails || encFails) ? 1 : 0;
}
#endif  // AV1G_FUZZ_NO_MAIN
