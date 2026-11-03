#!/usr/bin/env python3
# fatbin_compare.py TESTED13 PUBLISHED13 TESTED12 PUBLISHED12: every GPU code entry of two package pairs, by source file
# and architecture (E = SASS, P = PTX); the entry parser is ../../aws-g6-l4-v090-published/scripts/fatbin_hashes.py's.
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


import collections
def table(f):
    d = collections.Counter(); h = {}
    for k, a, src, p in entries(load(f)):
        key = (src, k, a); d[key] += 1; h[(key, d[key])] = hashlib.sha256(p).hexdigest()
    return h
for a, b in ((sys.argv[1], sys.argv[2]), (sys.argv[3], sys.argv[4])):
    ha, hb = table(a), table(b)
    same = sum(1 for k in ha if hb.get(k) == ha[k]); diff = [k for k in ha if k in hb and hb[k] != ha[k]]
    only = set(ha) ^ set(hb)
    print(a.split('/')[-1], 'vs', b.split('/')[-1], f': {len(ha)} / {len(hb)} entries, {same} identical, {len(diff)} differ, {len(only)} unmatched')
    for (src, k, ar), n in sorted(diff): print('   differ:', src, f'{k}{ar}')
