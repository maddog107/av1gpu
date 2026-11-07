#!/usr/bin/env python3
"""AVIF output checking, independent of the encoder: parse the container, decode both items with dav1d and aomdec,
insist on identical pixels, return planes / RGB."""
import os, subprocess, tempfile, struct, shutil
import numpy as np

class Bad(Exception):
    pass

def seq_info(obu):
    """(format, w, h, depth, matrix, full_range) from the first sequence header"""
    pos = 0
    while pos < len(obu):
        h = obu[pos]; pos += 1
        typ, ext, has_size = (h >> 3) & 15, (h >> 2) & 1, (h >> 1) & 1
        pos += ext
        size = len(obu) - pos
        if has_size:
            size, sh = 0, 0
            while True:
                b = obu[pos]; pos += 1
                size |= (b & 0x7F) << sh; sh += 7
                if not b & 0x80: break
        if typ == 1:
            bits = int.from_bytes(obu[pos:pos + size], "big"); nbits = 8 * size; at = [0]
            def f(n):
                at[0] += n
                return (bits >> (nbits - at[0])) & ((1 << n) - 1)
            profile = f(3); still = f(1); red = f(1)
            if not red: raise Bad("not a reduced still-picture header")
            f(5)
            wb, hb = f(4) + 1, f(4) + 1
            w, hgt = f(wb) + 1, f(hb) + 1
            f(1); f(1); f(1)  # 128, filter_intra, intra_edge
            f(1); f(1); f(1)  # superres, cdef, restoration
            hbd = f(1); depth = 10 if hbd else 8
            mono = 0 if profile == 1 else f(1)
            cdp = f(1)
            mc = 2; cp = tc = 2
            if cdp:
                cp, tc, mc = f(8), f(8), f(8)
            if mono:
                rng = f(1); fmt = "mono"
            elif cp == 1 and tc == 13 and mc == 0:
                rng = 1; fmt = "444"
            else:
                rng = f(1)
                fmt = "420" if profile == 0 else "444"
            return fmt, w, hgt, depth, mc, rng
        pos += size
    raise Bad("no sequence header")

def parse(path):
    d = open(path, "rb").read()
    def boxes(off, end):
        while off + 8 <= end:
            size, typ = int.from_bytes(d[off:off + 4], "big"), d[off + 4:off + 8]
            if size < 8 or off + size > end: raise Bad("box size")
            yield typ, off + 8, off + size
            off += size
    r = dict(irot=None, imir=None, icc=None, alpha=None, nclx=None)
    tops = {t: (b, e) for t, b, e in boxes(0, len(d))}
    if b"ftyp" not in tops or b"meta" not in tops or b"mdat" not in tops: raise Bad("top-level boxes " + str(list(tops)))
    meta = tops[b"meta"]
    pitm, iloc, auxl, props, assoc = None, {}, {}, [], {}
    for t, b, e in boxes(meta[0] + 4, meta[1]):
        if t == b"pitm": pitm = int.from_bytes(d[b + 4:b + 6], "big")
        elif t == b"iloc":
            if d[b + 4] != 0x44 or d[b + 5] != 0: raise Bad("iloc field sizes")
            n, q = int.from_bytes(d[b + 6:b + 8], "big"), b + 8
            for _ in range(n):
                iid, ec = int.from_bytes(d[q:q + 2], "big"), int.from_bytes(d[q + 4:q + 6], "big")
                q += 6; ext = []
                for _ in range(ec):
                    ext.append((int.from_bytes(d[q:q + 4], "big"), int.from_bytes(d[q + 4:q + 8], "big"))); q += 8
                iloc[iid] = ext
        elif t == b"iref":
            for rt, rb, re_ in boxes(b + 4, e):
                if rt == b"auxl":
                    fr, cnt = int.from_bytes(d[rb:rb + 2], "big"), int.from_bytes(d[rb + 2:rb + 4], "big")
                    auxl[fr] = [int.from_bytes(d[rb + 4 + 2 * k:rb + 6 + 2 * k], "big") for k in range(cnt)]
        elif t == b"iprp":
            for pt, pb, pe in boxes(b, e):
                if pt == b"ipco":
                    props = [(ct, d[cb:ce]) for ct, cb, ce in boxes(pb, pe)]
                elif pt == b"ipma":
                    n, q = int.from_bytes(d[pb + 4:pb + 8], "big"), pb + 8
                    for _ in range(n):
                        iid, c = int.from_bytes(d[q:q + 2], "big"), d[q + 2]
                        assoc[iid] = [x & 0x7F for x in d[q + 3:q + 3 + c]]
                        q += 3 + c
    def payload(iid):
        ext = iloc.get(iid)
        if not ext or len(ext) != 1: raise Bad(f"item {iid}: extents {ext}")
        o, n = ext[0]
        if o + n > len(d): raise Bad("extent outside the file")
        return d[o:o + n]
    r["colour"] = payload(pitm)
    for k in assoc.get(pitm, []):
        t, body = props[k - 1]
        if t == b"ispe": r["ispe"] = struct.unpack(">II", body[4:12])
        elif t == b"irot": r["irot"] = body[0] & 3
        elif t == b"imir": r["imir"] = body[0] & 1
        elif t == b"pixi": r["pixi"] = list(body[5:5 + body[4]])
        elif t == b"colr":
            if body[:4] == b"nclx": r["nclx"] = struct.unpack(">HHH", body[4:10]) + (body[10] >> 7,)
            elif body[:4] in (b"prof", b"rICC"): r["icc"] = body[4:]
    for fr, to in auxl.items():
        urn = [b for k in assoc.get(fr, []) for t, b in [props[k - 1]] if t == b"auxC"]
        if to == [pitm] and urn and urn[0][4:].rstrip(b"\0") == b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha":
            r["alpha"] = payload(fr)
            for k in assoc.get(fr, []):
                t, body = props[k - 1]
                if t == b"ispe": r["alpha_ispe"] = struct.unpack(">II", body[4:12])
    return r

def plane_sizes(fmt, w, h):
    if fmt == "mono": return [(w, h)]
    if fmt == "444": return [(w, h)] * 3
    return [(w, h), ((w + 1) // 2, (h + 1) // 2), ((w + 1) // 2, (h + 1) // 2)]

def split(buf, fmt, w, h, depth):
    dt = np.dtype("<u2") if depth > 8 else np.uint8
    a = np.frombuffer(buf, dtype=dt); o = 0; pl = []
    for pw, ph in plane_sizes(fmt, w, h):
        pl.append(a[o:o + pw * ph].reshape(ph, pw)); o += pw * ph
    return pl, a[o:]

def decode_stream(payload, td, tag="c"):
    """both decoders on TD + payload; returns (seq, planes); raises Bad if a decoder fails or they differ"""
    si = seq_info(payload)
    fmt, w, h, depth = si[:4]
    obu = f"{td}/{tag}.obu"
    with open(obu, "wb") as f:
        f.write(b"\x12\x00" + payload)
    dy, ay = f"{td}/{tag}.d.yuv", f"{td}/{tag}.a.yuv"
    for p in (dy, ay):
        if os.path.exists(p): os.remove(p)
    r = subprocess.run(["dav1d", "-q", "-i", obu, "--demuxer", "section5", "-o", dy, "--muxer", "yuv"], capture_output=True, timeout=120)
    if r.returncode or not os.path.exists(dy): raise Bad("dav1d failed: " + r.stderr.decode()[-200:])
    r = subprocess.run(["aomdec", "--rawvideo", "-o", ay, obu], capture_output=True, timeout=120)
    if r.returncode or not os.path.exists(ay): raise Bad("aomdec failed: " + r.stderr.decode()[-200:])
    db, ab = open(dy, "rb").read(), open(ay, "rb").read()
    n = sum(pw * ph for pw, ph in plane_sizes(fmt, w, h)) * (2 if depth > 8 else 1)
    if len(db) != n: raise Bad(f"dav1d wrote {len(db)} bytes, {fmt} {w}x{h} {depth}-bit needs {n}")
    if fmt == "mono":
        extra = ab[n:]
        neutral = b"\x80" if depth == 8 else (512).to_bytes(2, "little")
        if extra != neutral * (len(extra) // len(neutral)): raise Bad("aomdec's extra planes are not neutral")
        ab = ab[:n]
    if db != ab:
        pd, _ = split(db, fmt, w, h, depth); pa, _ = split(ab, fmt, w, h, depth)
        raise Bad("dav1d != aomdec " + str([int((x != y).sum()) for x, y in zip(pd, pa)]))
    pl, _ = split(db, fmt, w, h, depth)
    return si, [p.copy() for p in pl]

def upsample(c, w, h):
    return np.repeat(np.repeat(c, 2, 0), 2, 1)[:h, :w]

def to_rgb(si, pl):
    """float RGB 0..1 from the decoded planes (nearest chroma), as the stream's CICP says"""
    fmt, w, h, depth, mc, rng = si
    m = float((1 << depth) - 1)
    if fmt == "mono":
        y = pl[0].astype(np.float64) / m
        return np.dstack([y, y, y])
    u, v = pl[1], pl[2]
    if fmt == "420": u, v = upsample(u, w, h), upsample(v, w, h)
    Y, U, V = pl[0].astype(np.float64), u.astype(np.float64), v.astype(np.float64)
    if mc == 0:
        return np.dstack([V, Y, U]) / m
    if mc == 16:
        cg, co = U - 512, V - 512
        t = Y - np.floor(cg / 2); G = cg + t; B = t - np.floor(co / 2); R = B + co
        return np.dstack([R, G, B]) / 255.0
    mid = float(1 << (depth - 1))
    yy, cb, cr = Y / m, (U - mid) / m, (V - mid) / m
    if mc in (6, 5, 2):   # BT.601 (2 unspecified: the encoder's default planes are BT.601)
        r = yy + 1.402 * cr; g = yy - 0.344136286 * cb - 0.714136286 * cr; b = yy + 1.772 * cb
    elif mc == 1:
        r = yy + 1.5748 * cr; g = yy - 0.1873 * cb - 0.4681 * cr; b = yy + 1.8556 * cb
    else:
        raise Bad(f"matrix {mc}")
    return np.clip(np.dstack([r, g, b]), 0, 1)

def apply_transforms(a, irot, imir):
    """displayed from stored: irot (anti-clockwise 90 deg units), then imir"""
    if irot: a = np.rot90(a, irot)
    if imir is not None: a = a[::-1] if imir == 0 else a[:, ::-1]   # libavif: axis 0 exchanges top and bottom, axis 1 left and right
    return a

def check(path, td=None):
    """parse + decode both items in both decoders; dict(si, planes, rgb, alpha, ...) or raises Bad"""
    own = td is None
    if own: td = tempfile.mkdtemp(prefix="avt_", dir=os.environ.get("AVT_TMP"))
    try:
        r = parse(path)
        si, pl = decode_stream(r["colour"], td, "c")
        fmt, w, h, depth, mc, rng = si
        if r.get("ispe") != (w, h): raise Bad(f"ispe {r.get('ispe')} != stream {w}x{h}")
        ch = 1 if fmt == "mono" else 3
        if r.get("pixi") and (len(r["pixi"]) != ch or any(b != depth for b in r["pixi"])): raise Bad(f"pixi {r['pixi']} vs {fmt} {depth}")
        r.update(si=si, planes=pl)
        if r["nclx"] is not None and r["nclx"][2] != mc: raise Bad(f"colr matrix {r['nclx']} vs stream {mc}")
        if r["alpha"] is not None:
            ai, apl = decode_stream(r["alpha"], td, "a")
            if ai[0] != "mono" or ai[1:4] != si[1:4]: raise Bad(f"alpha stream {ai} vs colour {si}")
            r.update(asi=ai, aplane=apl[0])
        return r
    finally:
        if own: shutil.rmtree(td, ignore_errors=True)
