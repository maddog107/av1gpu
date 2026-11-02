#!/usr/bin/env python3
"""sass.py TREE WRITE_LINE: the SASS of k_recon<1,8> (and k_recon<1,10>) of ~/TREE/build/av1gpu at the U / V eob
reads after the concurrent chroma finals (rc_chroma, k_recon.cuh).  cuobjdump -xelf, then nvdisasm -g -c on the
k_recon cubin.  WRITE_LINE is the k_recon.cuh line of `if (tid == 0) { S.nTxb += 2; ... }` (1374 on d7d94b3, 1375
on 78e374c).  For each kernel: the code from the barrier before the eob reads to the barrier after the nTxb update,
every shared load in it, and every wide (.64 / .128) shared load in the whole kernel whose address range covers
the nTxb word that this STS writes.  ~/results/sass_TREE.txt (and the full disassembly in ~/work/sass_TREE/)."""
import glob, os, re, shutil, subprocess, sys

H = os.path.expanduser("~")
tree, wline = sys.argv[1], int(sys.argv[2])
TEST = sys.argv[3] if len(sys.argv) > 3 else None   # a saved disassembly instead (local test)
B = f"{H}/{tree}/build/av1gpu"
W = f"{H}/work/sass_{tree}"
# the AMI's CUDA 13.2.51 has no cuobjdump / nvdisasm: the same version's .debs, unpacked with dpkg -x
CT = os.environ.get("CUDA_TOOLS", f"{H}/cudatools/usr/local/cuda-13.2/bin")
if TEST:
    cub, OUT = [TEST], TEST + ".out"
    out = [f"# test {TEST}"]
else:
    shutil.rmtree(W, ignore_errors=True)
    os.makedirs(W)
    os.chdir(W)
    subprocess.run([f"{CT}/cuobjdump", "-xelf", "all", B], capture_output=True)
    cub, OUT = sorted(glob.glob("*k_recon*.cubin")), f"{H}/results/sass_{tree}.txt"
    out = [f"# {tree} ({open(f'{H}/{tree}/COMMIT').read().strip()}): cuobjdump -xelf all {B}; nvdisasm -g -c {cub}"]
FN = re.compile(r'^\s*\.section\s+\.text\.([A-Za-z0-9_$.]+)')
LN = re.compile(r'//## File "([^"]+)", line (\d+)')
INS = re.compile(r'/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;')
WANT = {"_Z7k_reconILi1ELi8EEv7GpBatch8GpParamsPKj": "k_recon<1,8>", "_Z7k_reconILi1ELi10EEv7GpBatch8GpParamsPKj": "k_recon<1,10>"}
for c in cub:
    if TEST:
        sass = open(c).read()
    else:
        sass = subprocess.run([f"{CT}/nvdisasm", "-g", "-c", c], capture_output=True, text=True).stdout
        open(f"{c}.sass", "w").write(sass)
    funcs, cur, rows, line = {}, None, [], None   # rows: (raw text, kind, src line, addr, instr)
    for raw in sass.splitlines():
        m = FN.match(raw)
        if m:
            cur = m.group(1); funcs[cur] = rows = []; line = None
            continue
        if cur is None:
            continue
        m = LN.search(raw)
        if m:
            line = (os.path.basename(m.group(1)), int(m.group(2)))
            rows.append((raw, "line", line, None, None))
            continue
        m = INS.search(raw)
        if m:
            rows.append((raw, "ins", line, int(m.group(1), 16), m.group(2)))
        elif raw.strip().startswith(".L_x"):
            rows.append((raw, "label", line, None, None))
    out.append(f"# {c}: {len(funcs)} functions; k_recon ones: {', '.join(n for n in funcs if 'k_recon' in n)}")
    for name in [n for n in WANT if n not in funcs]:
        out.append(f"== {WANT[name]} {name}: not in {c}")
    for name in [n for n in funcs if "k_recon" in n or "rc_chroma" in n]:
        label = WANT.get(name, name)
        rows = funcs[name]
        ins = [i for i, r in enumerate(rows) if r[1] == "ins"]
        # the nTxb write: an STS on WRITE_LINE of k_recon.cuh
        w = [i for i in ins if rows[i][2] == ("k_recon.cuh", wline) and re.match(r'(@!?P\w+\s+)?STS', rows[i][4])]
        out.append(f"== {label} {name}: {len(ins)} instructions; STS on k_recon.cuh:{wline}: "
                   + "; ".join(f"/*{rows[i][3]:x}*/ {rows[i][4]}" for i in w))
        if not w:
            continue
        # the addresses those STS write (nTxb and nCoef)
        addrs = []
        for i in w:
            m = re.search(r'\[(?:R\w+(?:\.X\d+)?\+)?(0x[0-9a-f]+)\]', rows[i][4])
            if m:
                addrs.append((i, int(m.group(1), 16), "R" in rows[i][4].split("[")[1].split("]")[0]))
        i0 = w[0]
        bars = [i for i in ins if "BAR.SYNC" in rows[i][4] or "BAR.SYNC.DEFER_BLOCKING" in rows[i][4]]
        before = [i for i in bars if i < i0]
        after = [i for i in bars if i > i0]
        lo = before[-1] if before else max(0, i0 - 40)
        lo = max(0, lo - 16)
        hi = (after[0] if after else i0 + 12) + 1
        out.append(f"-- code from the barrier before the eob reads to the barrier after the nTxb update")
        for r in rows[lo:hi + 1]:
            out.append(r[0].rstrip())
        out.append("-- shared loads in that range")
        for i in range(lo, hi + 1):
            if rows[i][1] == "ins" and re.match(r'(@!?P\w+\s+)?LDS', rows[i][4]):
                out.append(f"   /*{rows[i][3]:x}*/ {rows[i][4]}    ({rows[i][2][0]}:{rows[i][2][1]})")
        for (_, a, regbased) in addrs:
            if regbased:
                continue
            cov = []
            for i in ins:
                m = re.match(r'(@!?P\w+\s+)?LDS(\.U)?(\.\w+)*\s+\S+,\s*\[(0x[0-9a-f]+)\]', rows[i][4])
                if not m:
                    continue
                wd = 16 if ".128" in rows[i][4] else 8 if ".64" in rows[i][4] else 4
                base = int(m.group(4), 16)
                if base <= a < base + wd and wd > 4:
                    cov.append(f"   /*{rows[i][3]:x}*/ {rows[i][4]}    ({rows[i][2][0]}:{rows[i][2][1]})" if rows[i][2]
                               else f"   /*{rows[i][3]:x}*/ {rows[i][4]}")
            out.append(f"-- wide shared loads anywhere in {label} covering [0x{a:x}] (absolute address): {len(cov)}")
            out.extend(cov)
open(OUT, "w").write("\n".join(out) + "\nSASS_DONE\n")
print("\n".join(out))
