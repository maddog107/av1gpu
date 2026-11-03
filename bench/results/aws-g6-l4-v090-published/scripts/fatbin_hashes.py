#!/usr/bin/env python3
"""SHA-256 (first 6 hex digits) of every GPU code entry of one source file's fatbin, per package.

  fatbin_hashes.py SOURCE.cu PACKAGE...   (Linux .tar.gz or Windows .zip release packages)

Prints one line per package: E<arch> for SASS (ELF) entries, P<arch> for PTX. Compressed entries are hashed as
stored. Used for v0.9.0 to compare the tag build's k_an_c.cu with the build that was tested."""
import hashlib, struct, sys, tarfile, zipfile


def load(f):
    if f.endswith('.zip'):
        z = zipfile.ZipFile(f)
        return z.read([x for x in z.namelist() if x.endswith('av1gpu.exe')][0])
    t = tarfile.open(f)
    return t.extractfile([x for x in t.getmembers() if x.name.endswith('/av1gpu')][0]).read()


def entries(b):
    out, off = [], 0
    while True:
        off = b.find(b'\x50\xed\x55\xba', off)
        if off < 0:
            return out
        _, ver, hsz, fsz = struct.unpack_from('<IHHQ', b, off)
        if ver != 1 or hsz != 16 or fsz > len(b):
            off += 4
            continue
        p, pend = off + hsz, off + hsz + fsz
        while p < pend:
            kind, _, ehs, size = struct.unpack_from('<HHIQ', b, p)
            arch, = struct.unpack_from('<I', b, p + 28)
            no, nl = struct.unpack_from('<II', b, p + 32)
            src = b[p + no: p + no + nl].split(b'\0')[0].decode(errors='replace') if no and nl else '?'
            out.append(('P' if kind == 1 else 'E', arch, src.replace('\\', '/').split('/')[-1], b[p + ehs: p + ehs + size]))
            p += ehs + size
        off = pend


want = sys.argv[1]
for f in sys.argv[2:]:
    e = [x for x in entries(load(f)) if x[2] == want]
    print(f.split('/')[-1].ljust(44), ' '.join(f'{k}{a}:{hashlib.sha256(p).hexdigest()[:6]}' for k, a, _, p in e))
