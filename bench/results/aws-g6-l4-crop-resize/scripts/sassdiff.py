#!/usr/bin/env python3
"""sassdiff.py: SASS of every kernel in ~/main/build/av1gpu and ~/cr/build/av1gpu (cuobjdump -sass), compared per
kernel with addresses and encodings removed: which kernels exist only in one, which differ."""
import os, re, subprocess

H = os.path.expanduser("~")


def funcs(b):
    t = subprocess.run(["/usr/local/cuda/bin/cuobjdump", "-sass", b], capture_output=True, text=True).stdout
    parts = re.split(r"\n\s*Function : (\S+)\n", t)
    d = {}
    for i in range(1, len(parts) - 1, 2):
        body = re.sub(r"/\*[0-9a-f]{4,}\*/", "", parts[i + 1])
        body = re.sub(r"/\* 0x[0-9a-f]+ \*/", "", body)
        body = " ".join(x.strip() for x in body.splitlines() if x.strip().endswith(";"))
        d.setdefault(re.sub(r"_GLOBAL__N__[0-9a-f]+_", "_GLOBAL__N__", parts[i]), []).append(body)
    return d


a, b = funcs(f"{H}/main/build/av1gpu"), funcs(f"{H}/cr/build/av1gpu")
print("kernels: main", sum(len(v) for v in a.values()), "branch", sum(len(v) for v in b.values()))
print("only in the branch:", sorted(set(b) - set(a)))
print("only in main:", sorted(set(a) - set(b)))
diff = [k for k in a if k in b and a[k] != b[k]]
print("same name, different SASS:", len(diff), diff[:10])
