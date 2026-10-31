// AVIF (ISOBMFF) writer: a byte-exact C++ port of
// avif_conversion/hybrid_nvenc_libavif/avif_muxer.py:mux_avif() for still images, plus an optional alpha
// auxiliary image item laid out as libavif 1.4 writes it (av1g_mux_avif_alpha, see av1g.h).
// av1C, pixi and the nclx colr box are derived from the sequence header OBU contained in each stream.
#include "av1g/av1g.h"

#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

typedef std::vector<uint8_t> Bytes;

void be32(Bytes &b, uint32_t v) { for (int i = 3; i >= 0; i--) b.push_back((uint8_t)(v >> (8 * i))); }
void be16(Bytes &b, uint32_t v) { b.push_back((uint8_t)(v >> 8)); b.push_back((uint8_t)v); }
void app(Bytes &b, const Bytes &x) { b.insert(b.end(), x.begin(), x.end()); }
void app(Bytes &b, const char *s, size_t n) { b.insert(b.end(), s, s + n); }

Bytes box(const char *type, const Bytes &data) {
    Bytes b;
    be32(b, (uint32_t)(8 + data.size()));
    app(b, type, 4);
    app(b, data);
    return b;
}
Bytes fullbox(const char *type, int version, uint32_t flags, const Bytes &data) {
    Bytes b;
    be32(b, (uint32_t)(12 + data.size()));
    app(b, type, 4);
    be32(b, ((uint32_t)version << 24) | (flags & 0xFFFFFF));
    app(b, data);
    return b;
}

struct BitReader {
    const uint8_t *d; size_t n; size_t pos = 0;
    BitReader(const uint8_t *d_, size_t n_) : d(d_), n(n_) {}
    uint32_t get(int k) {
        uint32_t v = 0;
        for (int i = 0; i < k; i++, pos++) {
            if ((pos >> 3) >= n) return v;
            v = (v << 1) | ((d[pos >> 3] >> (7 - (pos & 7))) & 1);
        }
        return v;
    }
};

struct Cfg {
    int profile = 0, level = 0, tier = 0, hbd = 0, twelve = 0, mono = 0, ssx = 1, ssy = 1, csp = 0;
    int cp = 2, tc = 2, mc = 2, fullRange = 0;   // colour description (CICP) and color_range of the sequence header
    uint32_t w = 0, h = 0;                       // max_frame_width / height (the still picture's size)
};

bool read_leb128(const uint8_t *p, size_t n, size_t &pos, uint64_t &v) {
    v = 0;
    for (int i = 0; i < 8; i++) {
        if (pos >= n) return false;
        uint8_t b = p[pos++];
        v |= (uint64_t)(b & 0x7f) << (7 * i);
        if (!(b & 0x80)) return true;
    }
    return true;
}

bool parse_seq(const uint8_t *obus, size_t n, Cfg &c) {
    size_t pos = 0;
    while (pos < n) {
        uint8_t h = obus[pos++];
        int type = (h >> 3) & 15, ext = (h >> 2) & 1, has_size = (h >> 1) & 1;
        if (ext) pos++;
        uint64_t sz = n - pos;
        if (has_size && !read_leb128(obus, n, pos, sz)) return false;
        if (pos + sz > n) return false;
        if (type == 1) {
            BitReader br(obus + pos, (size_t)sz);
            c.profile = br.get(3);
            br.get(1);  // still_picture
            if (!br.get(1)) return false;  // only reduced_still_picture_header streams
            c.level = br.get(5);
            int wb = br.get(4) + 1, hb = br.get(4) + 1;
            c.w = br.get(wb) + 1;
            c.h = br.get(hb) + 1;
            br.get(1); br.get(1); br.get(1);  // 128 sb, filter intra, edge filter
            br.get(1); br.get(1); br.get(1);  // superres, cdef, restoration
            c.hbd = br.get(1);
            if (c.profile == 2 && c.hbd) c.twelve = br.get(1);
            c.mono = c.profile == 1 ? 0 : br.get(1);
            int cp = 2, tc = 2, mc = 2;
            if (br.get(1)) { cp = br.get(8); tc = br.get(8); mc = br.get(8); }
            c.cp = cp; c.tc = tc; c.mc = mc;
            if (c.mono) { c.fullRange = br.get(1); c.ssx = c.ssy = 1; c.csp = 0; }
            else if (cp == 1 && tc == 13 && mc == 0) { c.fullRange = 1; c.ssx = c.ssy = 0; c.csp = 0; }
            else {
                c.fullRange = br.get(1);  // color_range
                if (c.profile == 0) { c.ssx = c.ssy = 1; }
                else if (c.profile == 1) { c.ssx = c.ssy = 0; }
                else { c.ssx = 1; c.ssy = 0; }
                c.csp = (c.ssx && c.ssy) ? br.get(2) : 0;
            }
            return true;
        }
        pos += (size_t)sz;
    }
    return false;
}

}  // namespace

extern "C" void av1g_exif_orientation_to_irot_imir(int orientation, int32_t *irot, int32_t *imir) {
    *irot = -1;
    *imir = -1;
    switch (orientation) {
    case 2: *imir = 1; break;
    case 3: *irot = 2; break;
    case 4: *imir = 0; break;
    case 5: *irot = 1; *imir = 0; break;
    case 6: *irot = 3; break;
    case 7: *irot = 3; *imir = 0; break;
    case 8: *irot = 1; break;
    default: break;  // 1 = identity, others reserved
    }
}

extern "C" int av1g_mux_avif(const uint8_t *obus, size_t obu_size, uint32_t width, uint32_t height,
                             uint8_t **avif, size_t *avif_size) {
    return av1g_mux_avif_ex(obus, obu_size, width, height, nullptr, avif, avif_size);
}

extern "C" int av1g_mux_avif_ex(const uint8_t *obus, size_t obu_size, uint32_t width, uint32_t height,
                                const av1g_avif_meta *md, uint8_t **avif, size_t *avif_size) {
    return av1g_mux_avif_alpha(obus, obu_size, nullptr, 0, width, height, md, avif, avif_size);
}

extern "C" int av1g_mux_avif_alpha(const uint8_t *obus, size_t obu_size, const uint8_t *alpha_obus, size_t alpha_size,
                                   uint32_t width, uint32_t height, const av1g_avif_meta *md, uint8_t **avif,
                                   size_t *avif_size) {
    Cfg cfg;
    if (!parse_seq(obus, obu_size, cfg)) return AV1G_ERR_INVALID;
    const bool alpha = alpha_obus != nullptr;
    Cfg acfg;
    if (alpha) {
        // AVIF section 4: the alpha image is monochrome and full range, with the master image's bit depth (and size)
        if (!alpha_size || !parse_seq(alpha_obus, alpha_size, acfg)) return AV1G_ERR_INVALID;
        if (!acfg.mono || !acfg.fullRange || acfg.hbd != cfg.hbd || acfg.twelve != cfg.twelve || acfg.w != width ||
            acfg.h != height)
            return AV1G_ERR_INVALID;
    }
    Bytes hdlr_d = {0, 0, 0, 0};
    app(hdlr_d, "pict", 4);
    hdlr_d.resize(hdlr_d.size() + 12, 0);
    hdlr_d.push_back(0);
    Bytes hdlr = fullbox("hdlr", 0, 0, hdlr_d);
    Bytes pitm_d; be16(pitm_d, 1);
    Bytes pitm = fullbox("pitm", 0, 0, pitm_d);
    Bytes ispe_d; be32(ispe_d, width); be32(ispe_d, height);
    Bytes ispe = fullbox("ispe", 0, 0, ispe_d);
    auto av1c_box = [](const Cfg &c) {
        Bytes d = {0x81, (uint8_t)(((c.profile & 7) << 5) | (c.level & 31)),
                   (uint8_t)(((c.tier & 1) << 7) | ((c.hbd & 1) << 6) | ((c.twelve & 1) << 5) | ((c.mono & 1) << 4) |
                             ((c.ssx & 1) << 3) | ((c.ssy & 1) << 2) | (c.csp & 3)),
                   0x00};
        return box("av1C", d);
    };
    auto pixi_box = [](const Cfg &c) {   // bits_per_channel = BitDepth of the sequence header (8, 10 or 12)
        Bytes d;
        const uint8_t bpc = (uint8_t)(c.hbd ? (c.twelve ? 12 : 10) : 8);
        if (c.mono) { d.push_back(1); d.push_back(bpc); }
        else { d.push_back(3); d.push_back(bpc); d.push_back(bpc); d.push_back(bpc); }
        return fullbox("pixi", 0, 0, d);
    };
    Bytes colr_d;   // the stream's colour description (av1core default: BT.709 primaries, sRGB TC, BT.601 matrix, full range)
    app(colr_d, "nclx", 4);
    be16(colr_d, (uint32_t)cfg.cp); be16(colr_d, (uint32_t)cfg.tc); be16(colr_d, (uint32_t)cfg.mc);
    colr_d.push_back(cfg.fullRange ? 0x80 : 0x00);          // full_range_flag
    Bytes colr = box("colr", colr_d);
    // ipco: the colour item's properties ispe, av1C, pixi, [colr prof], colr nclx, [irot], [imir] (descriptive before
    // transformative; irot before imir as in libavif), then the alpha item's own ones.  An identical property is stored
    // once (as libavif deduplicates them): the alpha item shares ispe, irot and imir with the colour item.
    std::vector<Bytes> props;
    auto prop = [&](const Bytes &b) -> uint8_t {   // 1-based index of b in ipco
        for (size_t k = 0; k < props.size(); k++)
            if (props[k] == b) return (uint8_t)(k + 1);
        props.push_back(b);
        return (uint8_t)props.size();
    };
    std::vector<uint8_t> assoc;   // essential bit | 1-based property index
    assoc.push_back((uint8_t)(0x80 | prop(ispe)));
    assoc.push_back((uint8_t)(0x80 | prop(av1c_box(cfg))));
    assoc.push_back(prop(pixi_box(cfg)));
    if (md && md->icc && md->icc_size) {
        Bytes prof_d;
        app(prof_d, "prof", 4);
        prof_d.insert(prof_d.end(), md->icc, md->icc + md->icc_size);
        assoc.push_back(prop(box("colr", prof_d)));
    }
    assoc.push_back(prop(colr));
    std::vector<uint8_t> xform;   // irot / imir (essential): associated with the alpha item as well (libavif 1.4)
    if (md && md->irot >= 0) xform.push_back((uint8_t)(0x80 | prop(box("irot", Bytes{(uint8_t)(md->irot & 3)}))));
    if (md && md->imir >= 0) xform.push_back((uint8_t)(0x80 | prop(box("imir", Bytes{(uint8_t)(md->imir ? 1 : 0)}))));
    assoc.insert(assoc.end(), xform.begin(), xform.end());
    std::vector<uint8_t> aassoc;
    if (alpha) {
        // alpha item: ispe, av1C, pixi (one channel), auxC, then the transformative properties.  No colr: CICP and ICC do
        // not apply to an alpha plane (AVIF section 4: the colr box should be omitted, readers ignore it).
        static const char kAlphaUrn[] = "urn:mpeg:mpegB:cicp:systems:auxiliary:alpha";   // aux_type, with its NUL
        Bytes auxc_d;
        app(auxc_d, kAlphaUrn, sizeof kAlphaUrn);
        aassoc.push_back((uint8_t)(0x80 | prop(ispe)));
        aassoc.push_back((uint8_t)(0x80 | prop(av1c_box(acfg))));
        aassoc.push_back(prop(pixi_box(acfg)));
        aassoc.push_back(prop(fullbox("auxC", 0, 0, auxc_d)));
        aassoc.insert(aassoc.end(), xform.begin(), xform.end());
    }
    Bytes ipco_d;
    for (const Bytes &b : props) app(ipco_d, b);
    Bytes ipco = box("ipco", ipco_d);
    Bytes ipma_d; be32(ipma_d, alpha ? 2 : 1); be16(ipma_d, 1); ipma_d.push_back((uint8_t)assoc.size());
    app(ipma_d, assoc);
    if (alpha) { be16(ipma_d, 2); ipma_d.push_back((uint8_t)aassoc.size()); app(ipma_d, aassoc); }
    Bytes ipma = fullbox("ipma", 0, 0, ipma_d);
    Bytes iprp_d; app(iprp_d, ipco); app(iprp_d, ipma);
    Bytes iprp = box("iprp", iprp_d);
    Bytes iinf_d; be16(iinf_d, alpha ? 2 : 1);
    for (uint32_t id = 1; id <= (alpha ? 2u : 1u); id++) {   // item 1: colour (primary), 2: alpha; 'av01', no name, not hidden
        Bytes infe_d; be16(infe_d, id); be16(infe_d, 0); app(infe_d, "av01", 4); infe_d.push_back(0);
        app(iinf_d, fullbox("infe", 2, 0, infe_d));
    }
    Bytes iinf = fullbox("iinf", 0, 0, iinf_d);
    Bytes iref;   // alpha: 'auxl' from the alpha item to the colour item (straight alpha: no 'prem' reference)
    if (alpha) {
        Bytes auxl_d; be16(auxl_d, 2); be16(auxl_d, 1); be16(auxl_d, 1);
        iref = fullbox("iref", 0, 0, box("auxl", auxl_d));
    }
    Bytes ftyp_d; app(ftyp_d, "avif", 4); be32(ftyp_d, 0); app(ftyp_d, "avifmif1", 8);
    Bytes ftyp = box("ftyp", ftyp_d);
    // mdat = [alpha payload] colour payload: libavif writes alpha first, so a progressive reader has it before the colour
    const uint32_t alphaBytes = alpha ? (uint32_t)alpha_size : 0;
    auto iloc = [&](uint32_t off) {
        Bytes d = {0x44, 0x00};
        be16(d, alpha ? 2 : 1);
        be16(d, 1); be16(d, 0); be16(d, 1);
        be32(d, off + alphaBytes); be32(d, (uint32_t)obu_size);
        if (alpha) { be16(d, 2); be16(d, 0); be16(d, 1); be32(d, off); be32(d, alphaBytes); }
        return fullbox("iloc", 0, 0, d);
    };
    auto meta_box = [&](uint32_t off) {
        Bytes d; app(d, hdlr); app(d, pitm); app(d, iinf); app(d, iloc(off)); app(d, iref); app(d, iprp);
        return fullbox("meta", 0, 0, d);
    };
    Bytes meta1 = meta_box(0);
    const uint32_t off = (uint32_t)(ftyp.size() + meta1.size() + 8);
    Bytes meta = meta_box(off);
    Bytes file;
    app(file, ftyp);
    app(file, meta);
    be32(file, (uint32_t)(8 + obu_size + alphaBytes));
    app(file, "mdat", 4);
    if (alpha) file.insert(file.end(), alpha_obus, alpha_obus + alpha_size);
    file.insert(file.end(), obus, obus + obu_size);
    *avif = (uint8_t *)malloc(file.size());
    memcpy(*avif, file.data(), file.size());
    *avif_size = file.size();
    return AV1G_OK;
}
