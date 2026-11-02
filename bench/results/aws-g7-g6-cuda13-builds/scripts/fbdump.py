# fbdump.py BINARY|TARBALL: every fatbin entry's header fields in .nv_fatbin
import sys, tarfile, struct
a = sys.argv[1]
if a.endswith('.tar.gz'):
    t = tarfile.open(a); m = [x for x in t.getmembers() if x.name.endswith('/av1gpu')][0]; b = t.extractfile(m).read()
else:
    b = open(a, 'rb').read()
shoff, = struct.unpack_from('<Q', b, 0x28); shentsize, shnum, shstrndx = struct.unpack_from('<HHH', b, 0x3a)
secs = []
for i in range(shnum):
    name, typ, flags, addr, off, size = struct.unpack_from('<IIQQQQ', b, shoff + i * shentsize)
    secs.append((name, off, size))
strtab = secs[shstrndx]
def nm(n): s = b[strtab[1] + n:]; return s[:s.index(0)].decode()
fb = [(nm(n), o, s) for n, o, s in secs if nm(n) == '.nv_fatbin'][0]
off, end = fb[1], fb[1] + fb[2]
tot = {}
mi = 0
while off < end:
    magic, ver, hsz, fsz = struct.unpack_from('<IHHQ', b, off)
    if magic != 0xBA55ED50: off += 8; continue
    p, pend = off + hsz, off + hsz + fsz
    print(f'fatbin {mi} ver={ver} size={fsz}')
    while p < pend:
        kind, u1, ehs, size = struct.unpack_from('<HHIQ', b, p)
        f = struct.unpack_from('<IIHHIIIQQQ', b, p + 16)
        csize, u2, minor, major, arch, non, nol, flags, z, dsize = f
        extra = b[p + u2: p + ehs] if u2 else b""
        oname = b[p + non: p + non + nol].split(b'\0')[0].decode(errors='replace') if non else ''
        print(f'  {"PTX" if kind == 1 else "ELF"} sm={arch} u1={u1:#x} hsz={ehs} size={size} csize={csize} u2={u2:#x} '
              f'ver={major}.{minor} flags={flags:#x} z={z:#x} dsize={dsize} name={oname!r} extra={extra[:32].hex()}')
        tot[(kind, arch)] = tot.get((kind, arch), 0) + (dsize or size)
        p += ehs + size
    off = (pend + 7) & ~7; mi += 1
for k in sorted(tot): print('total', 'PTX' if k[0] == 1 else 'ELF', k[1], f'{tot[k]/1e6:.1f} MB')
