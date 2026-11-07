import os,struct,zlib
import numpy as np
def chunk(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
def wpng(path, a, ct=None, bits=8, extra=b""):
    a = np.asarray(a)
    if a.ndim == 2: a2 = a; c = 1
    else: a2 = a; c = a.shape[2]
    if ct is None: ct = {1: 0, 2: 4, 3: 2, 4: 6}[c]
    dt = ">u2" if bits == 16 else "u1"
    h, w = a2.shape[:2]
    raw = b"".join(b"\x00" + a2[y].astype(dt).tobytes() for y in range(h))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, bits, ct, 0, 0, 0)) + extra +
                           chunk(b"IDAT", zlib.compress(raw, 3)) + chunk(b"IEND", b""))

