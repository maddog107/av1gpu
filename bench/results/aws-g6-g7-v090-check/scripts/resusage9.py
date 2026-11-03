#!/usr/bin/env python3
"""resusage9.py: k_recon's resources in the CUDA 13 packages of v0.8.0 (~/bin/A13) and v0.9.0 (~/bin/B13), for sm_89
and sm_120, read from the binaries themselves (cuobjdump / nvdisasm 13.4, apt cuda-cuobjdump-13-4 cuda-nvdisasm-13-4):
  - cuobjdump --dump-resource-usage: REG, STACK (bytes of stack frame), SHARED, LOCAL of each k_recon kernel;
  - nvdisasm of the four k_recon cubins per architecture (cuobjdump -xelf): per function (the kernel and its
    non-inlined device functions: rc_luma, rc_chroma, cta_inverse_add, ...) the number of local-memory stores and
    loads (STL / LDL instructions: register spills and stack arrays) and of instructions.
~/results/resusage/{A13,B13}_{usage,elf}.txt, ~/results/resusage.txt (the table).  Prints RESUSAGE_DONE."""
import collections, os, re, subprocess

H = os.path.expanduser("~")
CU = "/usr/local/cuda-13.4/bin"
O = f"{H}/results/resusage"; os.makedirs(O, exist_ok=True)
ARCHS = ("sm_89", "sm_120")
CUBINS = ("k_recon", "k_recon_444_8", "k_recon_420_10", "k_recon_444_10")


def dem(n):
    last = "_Z" + n.split("$_Z")[-1] if "$_Z" in n else n
    p = subprocess.run(["c++filt", last], capture_output=True, text=True).stdout.strip()
    p = re.sub(r"\(.*\)$", "", p).replace("av1g_gpu::", "")
    return p[:80]


def usage(b):
    txt = subprocess.run([f"{CU}/cuobjdump", "--dump-resource-usage", f"{H}/bin/{b}"], capture_output=True, text=True).stdout
    open(f"{O}/{b}_usage.txt", "w").write(txt)
    out, arch, ident, fn = {}, None, None, None
    for line in txt.splitlines():
        m = re.match(r"\s*arch = (sm_\d+)", line)
        if m:
            arch = m.group(1)
        m = re.match(r"\s*identifier = (\S+)", line)
        if m:
            ident = os.path.basename(m.group(1))
        m = re.match(r"\s*Function (\S+):", line)
        if m:
            fn = m.group(1)
        m = re.match(r"\s*REG:(\d+) STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)", line)
        if m and fn and arch in ARCHS and "k_recon" in fn and ident and ident.startswith("k_recon"):
            out[(arch, ident.replace(".cu", ""), fn)] = tuple(int(x) for x in m.groups())
    return out


def sass(b):
    d = f"{H}/work/elf_{b}"
    os.makedirs(d, exist_ok=True)
    lst = subprocess.run([f"{CU}/cuobjdump", "-lelf", f"{H}/bin/{b}"], capture_output=True, text=True).stdout
    open(f"{O}/{b}_elf.txt", "w").write(lst)
    counts = {}
    for arch in ARCHS:
        for cb in CUBINS:
            n = f"{cb}.{arch}.cubin"
            if n not in lst:
                print("missing", b, n)
                continue
            subprocess.run([f"{CU}/cuobjdump", "-xelf", n, f"{H}/bin/{b}"], cwd=d, capture_output=True)
            txt = subprocess.run([f"{CU}/nvdisasm", os.path.join(d, n)], capture_output=True, text=True).stdout
            fn = None
            for line in txt.splitlines():
                m = re.match(r"\s*\.type\s+(\S+),@function", line)
                if m:
                    fn = m.group(1)
                    continue
                if fn is None:
                    continue
                m = re.match(r"\s*/\*[0-9a-f]+\*/\s+(?:@!?U?P\w+\s+)?([A-Z][A-Z0-9_.]*)", line)
                if m:
                    c = counts.setdefault((arch, cb, fn), collections.Counter())
                    op = m.group(1).split(".")[0]
                    c["instr"] += 1
                    if op in ("STL", "LDL"):
                        c[op] += 1
            os.remove(os.path.join(d, n))
    return counts


res = {b: (usage(b), sass(b)) for b in ("A13", "B13")}
with open(f"{H}/results/resusage.txt", "w") as f:
    f.write("# kernels (cuobjdump --dump-resource-usage): arch | cubin | kernel | v0.8.0 REG/STACK/SHARED/LOCAL | v0.9.0\n")
    for k in sorted(set(res["A13"][0]) | set(res["B13"][0])):
        a, b = res["A13"][0].get(k), res["B13"][0].get(k)
        f.write(f"{k[0]} | {k[1]} | {dem(k[2])} | {'/'.join(map(str, a)) if a else '-'} | {'/'.join(map(str, b)) if b else '-'}\n")
    f.write("# functions (nvdisasm): arch | cubin | function | v0.8.0 STL / LDL / instructions | v0.9.0 STL / LDL / instructions\n")
    keys = sorted(set(res["A13"][1]) | set(res["B13"][1]))
    for k in keys:
        a, b = res["A13"][1].get(k), res["B13"][1].get(k)
        if "__internal" in k[2] or "cuda_sm" in k[2]:
            continue
        fa = f"{a['STL']} / {a['LDL']} / {a['instr']}" if a else "-"
        fb = f"{b['STL']} / {b['LDL']} / {b['instr']}" if b else "-"
        f.write(f"{k[0]} | {k[1]} | {dem(k[2])} | {fa} | {fb}\n")
print(open(f"{H}/results/resusage.txt").read())
print("RESUSAGE_DONE")
