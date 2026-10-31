// AVIF alpha muxer test (av1g_mux_avif_alpha, CPU only).
//
// Colour and alpha AV1 streams come from av1core (random legal DecisionRecords: 4:2:0 / 4:4:4 / monochrome colour,
// 8 / 10 bits, lossy and lossless; the alpha stream monochrome with the unspecified 2/2/2 colour description).  Each
// AVIF is parsed back with the small ISOBMFF reader below and checked against the layout libavif 1.4 writes:
//   - ftyp, meta = hdlr, pitm (1), iinf (items 1 and 2, 'av01', not hidden), iloc, iref, iprp, then mdat;
//   - iloc: item 1 = the colour stream, item 2 = the alpha stream, byte for byte, alpha first in mdat, nothing else;
//   - iref: exactly one 'auxl' reference, item 2 -> item 1 (no 'prem');
//   - ipma item 1: ispe!, av1C!, pixi, [colr prof], colr nclx, [irot!], [imir!] (as without alpha);
//     item 2: the same ispe, its own av1C! (mono_chrome, the colour's bit depth), pixi (1 channel), auxC
//     "urn:mpeg:mpegB:cicp:systems:auxiliary:alpha" (version 0, NUL-terminated), the colour item's irot / imir, no colr;
//   - identical properties are stored once (monochrome colour shares av1C / pixi with the alpha item);
//   - the alpha stream's sequence header: mono_chrome 1, no colour description (2/2/2), color_range 1;
//   - without alpha the output is byte-identical to av1g_mux_avif_ex and has one item, no iref;
//   - streams that break the AVIF alpha rules (not monochrome, other bit depth or size) are rejected.
// Both item payloads are extracted from the file and decoded with dav1d and aomdec (PATH, or AV1G_DAV1D /
// AV1G_AOMDEC), which must reproduce av1core's reconstruction.  --write DIR keeps every AVIF (t<N>.avif) with the
// alpha reconstruction (t<N>.alpha.raw, 8-bit or 16-bit little-endian samples) for external readers (libavif).
#include "av1g/av1g.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <unistd.h>
#include <vector>

typedef std::vector<uint8_t> Bytes;

static std::string tool_path(const char *env, const char *dflt) {
    const char *v = getenv(env);
    return v && *v ? v : dflt;
}

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
    uint64_t next() {
        uint64_t z = (s += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
    int range(int lo, int hi) { return lo + (int)(next() % (uint64_t)(hi - lo + 1)); }
    bool chance(int pct) { return (int)(next() % 100) < pct; }
};

// ---------------------------------------------------------------------------------------------------------------
// random legal record: every 64x64 split down to 8x8 blocks (always legal), random intra modes, a few coefficients
struct Rec {
    av1g_decision_record r;
    std::vector<av1g_block> blocks;
    std::vector<uint32_t> btx;
    std::vector<av1g_txb> txb;
    std::vector<int16_t> coeffs;
};
struct Stream {
    Bytes obu;              // sequence header + frame (no temporal delimiter)
    Bytes recon;            // visible planes (8-bit: bytes, 10-bit: uint16_t little endian)
    int mono = 0, bd = 8;
};

static void split8(std::vector<av1g_block> &v, int r, int c, int sz, int miRows, int miCols, Rng &g, bool mono) {
    if (r >= miRows || c >= miCols) return;
    if (sz == 2) {
        av1g_block b;
        memset(&b, 0, sizeof b);
        b.mi_row = (uint16_t)r;
        b.mi_col = (uint16_t)c;
        b.bsize = AV1G_BLOCK_8X8;
        b.skip = g.chance(40);
        b.y_mode = (uint8_t)g.range(AV1G_DC_PRED, AV1G_PAETH_PRED);
        b.uv_mode = mono ? 0 : (uint8_t)g.range(AV1G_DC_PRED, AV1G_PAETH_PRED);
        v.push_back(b);
        return;
    }
    const int h = sz / 2;
    split8(v, r, c, h, miRows, miCols, g, mono);
    split8(v, r, c + h, h, miRows, miCols, g, mono);
    split8(v, r + h, c, h, miRows, miCols, g, mono);
    split8(v, r + h, c + h, h, miRows, miCols, g, mono);
}

// fmt: 0 = 4:2:0, 1 = 4:4:4, 2 = monochrome; cicp: -1 default (1/13/6), else {cp, tc, mc} packed as cp << 16 | tc << 8 | mc
static bool make_stream(Rng &g, int W, int H, int fmt, int bd, bool lossless, int cicp, Stream &out) {
    Rec R;
    av1g_decision_record &r = R.r;
    memset(&r, 0, sizeof r);
    av1g_frame_params &fp = r.fp;
    fp.version = AV1G_DECISIONS_VERSION;
    fp.width = (uint32_t)W;
    fp.height = (uint32_t)H;
    fp.monochrome = fmt == 2;
    fp.chroma_format = fmt == 1 ? AV1G_CHROMA_444 : AV1G_CHROMA_420;
    fp.bit_depth = (uint8_t)bd;
    fp.base_q_idx = lossless ? 0 : (uint8_t)g.range(20, 220);
    fp.tile_cols_log2 = fp.tile_rows_log2 = -1;
    if (cicp >= 0) {
        fp.color_description = 1;
        fp.color_primaries = (uint8_t)(cicp >> 16);
        fp.transfer_characteristics = (uint8_t)(cicp >> 8);
        fp.matrix_coefficients = (uint8_t)cicp;
    }
    const int miCols = 2 * ((W + 7) >> 3), miRows = 2 * ((H + 7) >> 3);
    for (int sr = 0; sr < miRows; sr += 16)
        for (int sc = 0; sc < miCols; sc += 16) split8(R.blocks, sr, sc, 16, miRows, miCols, g, fp.monochrome);
    r.n_blocks = (uint32_t)R.blocks.size();
    r.blocks = R.blocks.data();
    R.btx.assign(r.n_blocks + 1, 0);
    r.block_txb = R.btx.data();
    av1g_layout(&r, 0, 0);
    R.txb.assign(r.n_txb, av1g_txb());
    R.coeffs.assign(r.n_coeffs + 1, 0);
    r.txb = R.txb.data();
    r.coeffs = R.coeffs.data();
    if (av1g_layout(&r, r.n_txb, r.n_coeffs) != AV1G_OK) { printf("layout failed\n"); return false; }
    for (uint32_t t = 0; t < r.n_txb; t++) {
        av1g_txb &x = R.txb[t];
        x.tx_type = AV1G_DCT_DCT;
        if (g.chance(30)) R.coeffs[x.coeff_off] = (int16_t)(lossless ? g.range(-3, 3) : g.range(-12, 12));
        if (!lossless && g.chance(20)) R.coeffs[x.coeff_off + 1] = (int16_t)g.range(-4, 4);
    }
    av1g_encode_opts eo = {0, -1};
    av1g_output o;
    memset(&o, 0, sizeof o);
    if (av1g_encode(&r, &eo, &o) != AV1G_OK) { printf("av1g_encode: %s\n", o.error); av1g_output_free(&o); return false; }
    out.obu.assign(o.obu, o.obu + o.obu_size);
    out.mono = fp.monochrome;
    out.bd = bd;
    out.recon.clear();
    for (int p = 0; p < (fp.monochrome ? 1 : 3); p++)
        out.recon.insert(out.recon.end(), o.recon[p], o.recon[p] + (size_t)o.recon_w[p] * o.recon_h[p] * (bd > 8 ? 2 : 1));
    av1g_output_free(&o);
    return true;
}

// ---------------------------------------------------------------------------------------------------------------
// minimal ISOBMFF reader
struct Box { std::string type; size_t start, body, end; };
static uint32_t rd32(const Bytes &d, size_t p) { return (uint32_t)d[p] << 24 | (uint32_t)d[p + 1] << 16 | (uint32_t)d[p + 2] << 8 | d[p + 3]; }
static uint32_t rd16(const Bytes &d, size_t p) { return (uint32_t)d[p] << 8 | d[p + 1]; }
static bool children(const Bytes &d, size_t from, size_t to, std::vector<Box> &out) {
    out.clear();
    for (size_t p = from; p < to;) {
        if (p + 8 > to) return false;
        const uint32_t sz = rd32(d, p);
        if (sz < 8 || p + sz > to) return false;
        out.push_back({std::string((const char *)&d[p + 4], 4), p, p + 8, p + sz});
        p += sz;
    }
    return true;
}

struct Parsed {
    std::vector<std::string> top, metaKids;
    std::string hdlr;
    int pitm = -1;
    struct Infe { int id, version, flags; std::string type, name; };
    std::vector<Infe> items;
    std::map<int, std::vector<std::pair<uint32_t, uint32_t>>> iloc;   // item -> extents (offset, length)
    struct Ref { std::string type; int from; std::vector<int> to; };
    std::vector<Ref> refs;
    std::vector<std::pair<std::string, Bytes>> props;                 // ipco: type, full box bytes
    std::map<int, std::vector<std::pair<bool, int>>> ipma;            // item -> (essential, 1-based index)
    std::vector<int> ipmaOrder;
    size_t mdatBody = 0, mdatEnd = 0;
};

static std::string parse_avif(const Bytes &d, Parsed &P) {
    std::vector<Box> top;
    if (!children(d, 0, d.size(), top)) return "top-level box sizes";
    for (const Box &b : top) P.top.push_back(b.type);
    for (const Box &b : top) {
        if (b.type == "mdat") { P.mdatBody = b.body; P.mdatEnd = b.end; }
        if (b.type != "meta") continue;
        if (rd32(d, b.body) != 0) return "meta version / flags";
        std::vector<Box> mk;
        if (!children(d, b.body + 4, b.end, mk)) return "meta children";
        for (const Box &m : mk) {
            P.metaKids.push_back(m.type);
            if (m.type == "hdlr") P.hdlr = std::string((const char *)&d[m.body + 8], 4);
            else if (m.type == "pitm") P.pitm = (int)rd16(d, m.body + 4);
            else if (m.type == "iinf") {
                const size_t n = rd16(d, m.body + 4);
                std::vector<Box> ie;
                if (!children(d, m.body + 6, m.end, ie) || ie.size() != n) return "iinf entries";
                for (const Box &e : ie) {
                    if (e.type != "infe") return "iinf child " + e.type;
                    Parsed::Infe f;
                    f.version = d[e.body];
                    f.flags = (int)(rd32(d, e.body) & 0xFFFFFF);
                    f.id = (int)rd16(d, e.body + 4);
                    f.type = std::string((const char *)&d[e.body + 8], 4);
                    size_t q = e.body + 12;
                    while (q < e.end && d[q]) f.name += (char)d[q++];
                    if (q + 1 != e.end) return "infe name / trailing bytes";
                    P.items.push_back(f);
                }
            } else if (m.type == "iloc") {
                if (d[m.body] != 0 || d[m.body + 4] != 0x44 || d[m.body + 5] != 0) return "iloc version / field sizes";
                const size_t n = rd16(d, m.body + 6);
                size_t q = m.body + 8;
                for (size_t k = 0; k < n; k++) {
                    const int id = (int)rd16(d, q);
                    if (rd16(d, q + 2) != 0) return "iloc data_reference_index";
                    const size_t ec = rd16(d, q + 4);
                    q += 6;
                    for (size_t e = 0; e < ec; e++, q += 8) P.iloc[id].push_back({rd32(d, q), rd32(d, q + 4)});
                }
                if (q != m.end) return "iloc size";
            } else if (m.type == "iref") {
                if (rd32(d, m.body) != 0) return "iref version";
                std::vector<Box> rk;
                if (!children(d, m.body + 4, m.end, rk)) return "iref children";
                for (const Box &r : rk) {
                    Parsed::Ref ref;
                    ref.type = r.type;
                    ref.from = (int)rd16(d, r.body);
                    const size_t c = rd16(d, r.body + 2);
                    if (r.body + 4 + 2 * c != r.end) return "iref entry size";
                    for (size_t k = 0; k < c; k++) ref.to.push_back((int)rd16(d, r.body + 4 + 2 * k));
                    P.refs.push_back(ref);
                }
            } else if (m.type == "iprp") {
                std::vector<Box> pk, pc;
                if (!children(d, m.body, m.end, pk) || pk.size() != 2 || pk[0].type != "ipco" || pk[1].type != "ipma")
                    return "iprp children";
                if (!children(d, pk[0].body, pk[0].end, pc)) return "ipco children";
                for (const Box &p : pc) P.props.push_back({p.type, Bytes(d.begin() + p.start, d.begin() + p.end)});
                const Box &im = pk[1];
                if (rd32(d, im.body) != 0) return "ipma version / flags";
                const size_t n = rd32(d, im.body + 4);
                size_t q = im.body + 8;
                for (size_t k = 0; k < n; k++) {
                    const int id = (int)rd16(d, q);
                    const int c = d[q + 2];
                    q += 3;
                    P.ipmaOrder.push_back(id);
                    for (int a = 0; a < c; a++, q++) P.ipma[id].push_back({(d[q] & 0x80) != 0, d[q] & 0x7F});
                }
                if (q != im.end) return "ipma size";
            }
        }
    }
    return "";
}

// sequence header fields of a reduced still-picture header: mono, colour description present, colour range, bit depth
struct SeqHdr { int ok = 0, profile = 0, mono = 0, hbd = 0, cdp = 0, cp = 2, tc = 2, mc = 2, range = 0; uint32_t w = 0, h = 0; };
static SeqHdr seq_header(const uint8_t *p, size_t n) {
    SeqHdr s;
    size_t pos = 0;
    while (pos < n) {
        const uint8_t h = p[pos++];
        const int type = (h >> 3) & 15;
        if ((h >> 2) & 1) pos++;
        uint64_t sz = 0;
        for (int i = 0; i < 8; i++) {
            const uint8_t b = p[pos++];
            sz |= (uint64_t)(b & 0x7F) << (7 * i);
            if (!(b & 0x80)) break;
        }
        if (type == 1) {
            size_t bit = 0;
            auto f = [&](int k) {
                uint32_t v = 0;
                for (int i = 0; i < k; i++, bit++) v = v << 1 | ((p[pos + (bit >> 3)] >> (7 - (bit & 7))) & 1);
                return v;
            };
            s.profile = (int)f(3);
            f(1);
            if (!f(1)) return s;
            f(5);
            const int wb = (int)f(4) + 1, hb = (int)f(4) + 1;
            s.w = f(wb) + 1;
            s.h = f(hb) + 1;
            f(6);
            s.hbd = (int)f(1);
            s.mono = s.profile == 1 ? 0 : (int)f(1);
            s.cdp = (int)f(1);
            if (s.cdp) { s.cp = (int)f(8); s.tc = (int)f(8); s.mc = (int)f(8); }
            if (s.mono) s.range = (int)f(1);
            else if (s.cp == 1 && s.tc == 13 && s.mc == 0) s.range = 1;
            else s.range = (int)f(1);
            s.ok = 1;
            return s;
        }
        pos += (size_t)sz;
    }
    return s;
}

// ---------------------------------------------------------------------------------------------------------------
static int g_fail = 0;
#define CHECK(cond, ...)                                                    \
    do {                                                                    \
        if (!(cond)) {                                                      \
            printf("  FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond);        \
            printf(__VA_ARGS__);                                            \
            printf("\n");                                                   \
            ok = false;                                                     \
        }                                                                   \
    } while (0)

static bool write_file(const std::string &path, const void *p, size_t n) {
    FILE *f = fopen(path.c_str(), "wb");
    if (!f) return false;
    const bool ok = fwrite(p, 1, n, f) == n;
    return fclose(f) == 0 && ok;
}
static Bytes read_file(const std::string &path) {
    Bytes v;
    FILE *f = fopen(path.c_str(), "rb");
    if (!f) return v;
    uint8_t b[65536];
    size_t n;
    while ((n = fread(b, 1, sizeof b, f)) > 0) v.insert(v.end(), b, b + n);
    fclose(f);
    return v;
}

// both decoders must reproduce `want` from the item payload (a temporal delimiter is prepended for section 5)
static bool decode_check(const Bytes &payload, const Stream &s, const std::string &tmp, const char *what) {
    Bytes td = {0x12, 0x00};
    td.insert(td.end(), payload.begin(), payload.end());
    const std::string obu = tmp + ".obu", y1 = tmp + ".dav1d.yuv", y2 = tmp + ".aom.yuv";
    write_file(obu, td.data(), td.size());
    remove(y1.c_str());
    remove(y2.c_str());
    const std::string c1 = tool_path("AV1G_DAV1D", "dav1d") + " -q -i " + obu + " --demuxer section5 --muxer yuv -o " + y1 + " >/dev/null 2>&1";
    const std::string c2 = tool_path("AV1G_AOMDEC", "aomdec") + " --rawvideo -o " + y2 + " " + obu + " >/dev/null 2>&1";
    const int r1 = system(c1.c_str()), r2 = system(c2.c_str());
    Bytes v1 = read_file(y1), v2 = read_file(y2);
    if (s.mono && v2.size() > s.recon.size()) v2.resize(s.recon.size());   // aomdec appends neutral chroma
    const bool ok = r1 == 0 && r2 == 0 && v1 == s.recon && v2 == s.recon;
    if (!ok) printf("  FAIL %s payload: dav1d rc %d %s, aomdec rc %d %s\n", what, r1, v1 == s.recon ? "identical" : "DIFFERS", r2,
                    v2 == s.recon ? "identical" : "DIFFERS");
    remove(obu.c_str());
    remove(y1.c_str());
    remove(y2.c_str());
    return ok;
}

struct Case {
    const char *name;
    int W, H, fmt, bd;
    bool llColour, llAlpha;
    int colourCicp;       // -1 default
    int icc, irot, imir;  // icc: bytes of a fake profile (0 = none)
};

static bool run_case(int idx, const Case &c, Rng &g, const std::string &tmp, const std::string &writeDir) {
    bool ok = true;
    Stream col, alp;
    if (!make_stream(g, c.W, c.H, c.fmt, c.bd, c.llColour, c.colourCicp, col)) return false;
    if (!make_stream(g, c.W, c.H, 2, c.bd, c.llAlpha, 2 << 16 | 2 << 8 | 2, alp)) return false;
    // the alpha stream: monochrome, no colour description (2/2/2 is not coded), full range
    const SeqHdr as = seq_header(alp.obu.data(), alp.obu.size());
    CHECK(as.ok && as.mono == 1 && as.cdp == 0 && as.range == 1 && as.hbd == (c.bd == 10), "alpha sequence header");
    {   // av1g_parse reads the absent description back as 2/2/2, and the record re-encodes to the same bytes
        av1g_parsed *pr = nullptr;
        const int prc = av1g_parse(alp.obu.data(), alp.obu.size(), &pr);
        CHECK(prc == AV1G_OK, "parse alpha stream: %s", pr ? pr->error : "?");
        if (prc == AV1G_OK) {
            const av1g_frame_params &q = pr->rec.fp;
            CHECK(q.color_description == 1 && q.color_primaries == 2 && q.transfer_characteristics == 2 && q.matrix_coefficients == 2 &&
                      q.monochrome == 1,
                  "parsed colour description %d %d/%d/%d", q.color_description, q.color_primaries, q.transfer_characteristics,
                  q.matrix_coefficients);
            av1g_encode_opts eo = {AV1G_ENC_BITSTREAM_ONLY, (int32_t)pr->seq_level_idx};
            av1g_output o;
            memset(&o, 0, sizeof o);
            CHECK(av1g_encode(&pr->rec, &eo, &o) == AV1G_OK && o.obu_size == alp.obu.size() && !memcmp(o.obu, alp.obu.data(), o.obu_size),
                  "alpha stream does not re-encode identically");
            av1g_output_free(&o);
        }
        av1g_parsed_free(pr);
    }
    Bytes icc((size_t)c.icc);
    for (size_t k = 0; k < icc.size(); k++) icc[k] = (uint8_t)(k * 7 + 3);
    av1g_avif_meta md;
    memset(&md, 0, sizeof md);
    md.icc = icc.empty() ? nullptr : icc.data();
    md.icc_size = icc.size();
    md.irot = c.irot;
    md.imir = c.imir;
    uint8_t *a = nullptr, *b = nullptr;
    size_t an = 0, bn = 0;
    // without alpha: byte-identical to av1g_mux_avif_ex, one item, no iref
    CHECK(av1g_mux_avif_alpha(col.obu.data(), col.obu.size(), nullptr, 0, c.W, c.H, &md, &a, &an) == AV1G_OK, "mux (no alpha)");
    CHECK(av1g_mux_avif_ex(col.obu.data(), col.obu.size(), c.W, c.H, &md, &b, &bn) == AV1G_OK, "mux_ex");
    CHECK(a && b && an == bn && !memcmp(a, b, an), "no-alpha output differs from av1g_mux_avif_ex");
    Bytes noAlpha(a, a + an);
    free(a);
    free(b);
    Parsed P0;
    std::string e = parse_avif(noAlpha, P0);
    CHECK(e.empty(), "parse (no alpha): %s", e.c_str());
    CHECK(P0.items.size() == 1 && P0.refs.empty() && P0.ipma.size() == 1, "no-alpha file has %zu items, %zu refs", P0.items.size(),
          P0.refs.size());
    // with alpha
    a = nullptr;
    CHECK(av1g_mux_avif_alpha(col.obu.data(), col.obu.size(), alp.obu.data(), alp.obu.size(), c.W, c.H, &md, &a, &an) == AV1G_OK,
          "mux (alpha)");
    if (!a) return false;
    Bytes f(a, a + an);
    free(a);
    Parsed P;
    e = parse_avif(f, P);
    CHECK(e.empty(), "parse: %s", e.c_str());
    if (!e.empty()) return false;
    CHECK((P.top == std::vector<std::string>{"ftyp", "meta", "mdat"}), "top-level boxes");
    CHECK((P.metaKids == std::vector<std::string>{"hdlr", "pitm", "iinf", "iloc", "iref", "iprp"}), "meta children");
    CHECK(P.hdlr == "pict" && P.pitm == 1, "hdlr %s pitm %d", P.hdlr.c_str(), P.pitm);
    CHECK(P.items.size() == 2, "%zu items", P.items.size());
    for (size_t k = 0; k < P.items.size(); k++)
        CHECK(P.items[k].id == (int)k + 1 && P.items[k].version == 2 && P.items[k].flags == 0 && P.items[k].type == "av01",
              "infe %zu: id %d v%d flags %d type %s", k, P.items[k].id, P.items[k].version, P.items[k].flags, P.items[k].type.c_str());
    // iloc: item payloads are exactly the two streams, alpha first, and they fill mdat
    CHECK(P.iloc.size() == 2 && P.iloc[1].size() == 1 && P.iloc[2].size() == 1, "iloc entries");
    if (P.iloc[1].size() == 1 && P.iloc[2].size() == 1) {
        const auto ce = P.iloc[1][0], ae = P.iloc[2][0];
        CHECK(ae.first == P.mdatBody && ae.first + ae.second == ce.first && ce.first + ce.second == P.mdatEnd && P.mdatEnd == f.size(),
              "extents colour (%u, %u) alpha (%u, %u), mdat %zu..%zu", ce.first, ce.second, ae.first, ae.second, P.mdatBody, P.mdatEnd);
        CHECK(ce.second == col.obu.size() && !memcmp(&f[ce.first], col.obu.data(), col.obu.size()), "colour payload");
        CHECK(ae.second == alp.obu.size() && !memcmp(&f[ae.first], alp.obu.data(), alp.obu.size()), "alpha payload");
        if (ok) {
            ok &= decode_check(Bytes(f.begin() + ce.first, f.begin() + ce.first + ce.second), col, tmp, "colour");
            ok &= decode_check(Bytes(f.begin() + ae.first, f.begin() + ae.first + ae.second), alp, tmp, "alpha");
        }
    }
    CHECK(P.refs.size() == 1 && P.refs[0].type == "auxl" && P.refs[0].from == 2 && P.refs[0].to == std::vector<int>{1}, "iref");
    // ipma: entries in item order; colour item as without alpha
    CHECK((P.ipmaOrder == std::vector<int>{1, 2}), "ipma item order");
    CHECK(P.ipma[1] == P0.ipma[1], "colour associations differ from the no-alpha file");
    for (size_t k = 0; k < P0.props.size() && k < P.props.size(); k++) CHECK(P0.props[k] == P.props[k], "colour property %zu moved", k);
    for (size_t k = 0; k < P.props.size(); k++)
        for (size_t j = k + 1; j < P.props.size(); j++) CHECK(P.props[k] != P.props[j], "properties %zu and %zu are duplicates", k + 1, j + 1);
    auto prop = [&](int idx1) -> const std::pair<std::string, Bytes> & { return P.props.at((size_t)idx1 - 1); };
    std::vector<std::string> ct, at;
    for (auto &x : P.ipma[1]) ct.push_back(prop(x.second).first + (x.first ? "!" : ""));
    for (auto &x : P.ipma[2]) at.push_back(prop(x.second).first + (x.first ? "!" : ""));
    std::vector<std::string> wantC = {"ispe!", "av1C!", "pixi"}, wantA = {"ispe!", "av1C!", "pixi", "auxC"};
    if (c.icc) wantC.push_back("colr");
    wantC.push_back("colr");
    if (c.irot >= 0) { wantC.push_back("irot!"); wantA.push_back("irot!"); }
    if (c.imir >= 0) { wantC.push_back("imir!"); wantA.push_back("imir!"); }
    CHECK(ct == wantC, "colour associations");
    CHECK(at == wantA, "alpha associations");
    if (ok) {
        const auto &A = P.ipma[2], &C = P.ipma[1];
        CHECK(A[0].second == C[0].second, "alpha ispe is not the colour's");
        const size_t nx = (c.irot >= 0) + (c.imir >= 0);
        for (size_t k = 0; k < nx; k++) CHECK(A[4 + k].second == C[C.size() - nx + k].second, "alpha irot/imir not shared");
        const Bytes &av1c = prop(A[1].second).second, &pixi = prop(A[2].second).second, &auxc = prop(A[3].second).second;
        const int bpc = c.bd;
        CHECK(av1c.size() == 12 && av1c[8] == 0x81 && (av1c[10] & 0x10) && ((av1c[10] >> 6) & 1) == (c.bd == 10) &&
                  (av1c[10] & 0x0C) == 0x0C && (av1c[10] & 3) == 0,
              "alpha av1C");
        CHECK(pixi.size() == 14 && rd32(pixi, 8) == 0 && pixi[12] == 1 && pixi[13] == bpc, "alpha pixi");
        static const char urn[] = "urn:mpeg:mpegB:cicp:systems:auxiliary:alpha";
        CHECK(auxc.size() == 12 + sizeof urn && rd32(auxc, 8) == 0 && !memcmp(&auxc[12], urn, sizeof urn), "auxC");
        // monochrome colour: av1C and pixi are shared (same bytes)
        if (c.fmt == 2) CHECK(A[1].second == C[1].second && A[2].second == C[2].second, "mono colour: av1C / pixi not shared");
        else CHECK(A[1].second != C[1].second && A[2].second != C[2].second, "alpha av1C / pixi shared with colour");
    }
    // the alpha rules: not monochrome, other bit depth, other size -> rejected
    {
        Stream bad;
        uint8_t *x = nullptr;
        size_t xn = 0;
        if (c.fmt != 2 && c.W >= 8) {
            CHECK(av1g_mux_avif_alpha(col.obu.data(), col.obu.size(), col.obu.data(), col.obu.size(), c.W, c.H, &md, &x, &xn) ==
                      AV1G_ERR_INVALID, "colour stream accepted as alpha");
        }
        if (make_stream(g, c.W, c.H, 2, c.bd == 8 ? 10 : 8, true, 2 << 16 | 2 << 8 | 2, bad))
            CHECK(av1g_mux_avif_alpha(col.obu.data(), col.obu.size(), bad.obu.data(), bad.obu.size(), c.W, c.H, &md, &x, &xn) ==
                      AV1G_ERR_INVALID, "alpha of another bit depth accepted");
        if (make_stream(g, c.W + 8, c.H, 2, c.bd, true, 2 << 16 | 2 << 8 | 2, bad))
            CHECK(av1g_mux_avif_alpha(col.obu.data(), col.obu.size(), bad.obu.data(), bad.obu.size(), c.W, c.H, &md, &x, &xn) ==
                      AV1G_ERR_INVALID, "alpha of another size accepted");
        CHECK(av1g_mux_avif_alpha(col.obu.data(), col.obu.size(), alp.obu.data(), 0, c.W, c.H, &md, &x, &xn) == AV1G_ERR_INVALID,
              "empty alpha stream accepted");
    }
    if (!writeDir.empty()) {
        const std::string base = writeDir + "/t" + std::to_string(idx);
        write_file(base + ".avif", f.data(), f.size());
        write_file(base + ".alpha.raw", alp.recon.data(), alp.recon.size());
    }
    printf("%-3d %-34s %4dx%-4d %s %2d-bit colour %s alpha %s icc %d irot %d imir %d: %zu bytes (alpha %zu): %s\n", idx, c.name,
           c.W, c.H, c.fmt == 0 ? "4:2:0" : c.fmt == 1 ? "4:4:4" : "mono ", c.bd, c.llColour ? "lossless" : "lossy   ",
           c.llAlpha ? "lossless" : "lossy   ", c.icc, c.irot, c.imir, f.size(), alp.obu.size(), ok ? "ok" : "FAILED");
    return ok;
}

int main(int argc, char **argv) {
    std::string writeDir;
    uint64_t seed = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--write") && i + 1 < argc) writeDir = argv[++i];
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = strtoull(argv[++i], nullptr, 10);
        else { fprintf(stderr, "usage: avif_alpha_test [--write DIR] [--seed N]\n"); return 2; }
    }
    const char *t = getenv("TMPDIR");
    const std::string tmp = std::string(t && *t ? t : "/tmp") + "/avif_alpha_test_" + std::to_string((long)getpid());
    static const Case cases[] = {
        {"4:2:0 lossy + lossless alpha", 97, 61, 0, 8, false, true, -1, 0, -1, -1},
        {"4:2:0 lossy + lossy alpha", 64, 64, 0, 8, false, false, -1, 0, -1, -1},
        {"4:4:4 GBR lossless + ICC + rotation", 130, 67, 1, 8, true, true, 1 << 16 | 13 << 8 | 0, 1000, 1, 0},
        {"4:4:4 YCgCo-Re 10-bit lossless", 33, 47, 1, 10, true, true, 1 << 16 | 13 << 8 | 16, 0, -1, 1},
        {"4:2:0 10-bit lossy + irot", 200, 120, 0, 10, false, true, -1, 0, 3, -1},
        {"4:4:4 10-bit lossy + imir + ICC", 72, 40, 1, 10, false, false, -1, 333, -1, 0},
        {"monochrome lossless + lossless alpha", 61, 97, 2, 8, true, true, -1, 0, -1, -1},
        {"monochrome 10-bit + all metadata", 50, 50, 2, 10, false, true, -1, 64, 2, 1},
        {"1x1", 1, 1, 0, 8, false, true, -1, 0, -1, -1},
        {"1x1 4:4:4 10-bit", 1, 1, 1, 10, false, true, -1, 0, 1, 1},
        {"wide", 1031, 9, 0, 8, false, true, -1, 0, -1, -1},
    };
    Rng g(seed);
    int nok = 0, n = 0;
    for (const Case &c : cases) {
        n++;
        if (run_case(n, c, g, tmp, writeDir)) nok++;
        else g_fail++;
    }
    printf("avif_alpha_test: %d/%d cases passed\n", nok, n);
    return g_fail ? 1 : 0;
}
