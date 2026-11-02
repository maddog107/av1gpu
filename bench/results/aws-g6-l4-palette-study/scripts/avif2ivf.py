#!/usr/bin/env python3
"""avif2ivf.py IN.avif OUT.ivf : the primary item's AV1 OBUs (sequence header + frame) as a one-frame IVF file, so
libaom's tools (pstat, aomdec) can read an AVIF still image.  Handles what avifenc and av1gpu write: one primary item
in the meta box, data in mdat (iloc construction method 0) or idat (method 1)."""
import struct
import sys


def boxes(b, off, end):
    while off + 8 <= end:
        size, typ = struct.unpack(">I4s", b[off:off + 8])
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", b[off + 8:off + 16])[0]
            hdr = 16
        elif size == 0:
            size = end - off
        yield typ.decode("latin1"), off + hdr, off + size
        off += size


def uint(b, off, n):
    return int.from_bytes(b[off:off + n], "big") if n else 0, off + n


def primary_item(b):
    meta = next((s, e) for t, s, e in boxes(b, 0, len(b)) if t == "meta")
    pitm, iloc, idat = None, None, None
    for t, s, e in boxes(b, meta[0] + 4, meta[1]):   # meta is a FullBox
        if t == "pitm":
            pitm = struct.unpack(">H", b[s + 4:s + 6])[0] if b[s] == 0 else struct.unpack(">I", b[s + 4:s + 8])[0]
        elif t == "iloc":
            iloc = (s, e)
        elif t == "idat":
            idat = s
    s = iloc[0]
    ver = b[s]
    off = s + 4
    v = b[off] << 8 | b[off + 1]
    off += 2
    osz, lsz, bsz, isz = v >> 12, (v >> 8) & 15, (v >> 4) & 15, v & 15 if ver in (1, 2) else 0
    cnt, off = uint(b, off, 2 if ver < 2 else 4)
    for _ in range(cnt):
        iid, off = uint(b, off, 2 if ver < 2 else 4)
        method = 0
        if ver in (1, 2):
            cm, off = uint(b, off, 2)
            method = cm & 15
        _, off = uint(b, off, 2)   # data_reference_index
        base, off = uint(b, off, bsz)
        n, off = uint(b, off, 2)
        data = b""
        for _ in range(n):
            if ver in (1, 2) and isz:
                _, off = uint(b, off, isz)
            eo, off = uint(b, off, osz)
            el, off = uint(b, off, lsz)
            start = (idat if method == 1 else 0) + base + eo
            data += b[start:start + el]
        if iid == pitm:
            return data
    raise ValueError("primary item not found")


def ispe(b):
    for t, s, e in boxes(b, 0, len(b)):
        if t == "meta":
            for t2, s2, e2 in boxes(b, s + 4, e):
                if t2 == "iprp":
                    for t3, s3, e3 in boxes(b, s2, e2):
                        if t3 == "ipco":
                            for t4, s4, e4 in boxes(b, s3, e3):
                                if t4 == "ispe":
                                    return struct.unpack(">II", b[s4 + 4:s4 + 12])
    return 0, 0


def main():
    b = open(sys.argv[1], "rb").read()
    obus = primary_item(b)
    w, h = ispe(b)
    hdr = b"DKIF" + struct.pack("<HH4sHHIIII", 0, 32, b"AV01", w & 0xFFFF, h & 0xFFFF, 30, 1, 1, 0)
    open(sys.argv[2], "wb").write(hdr + struct.pack("<IQ", len(obus), 0) + obus)


if __name__ == "__main__":
    main()
