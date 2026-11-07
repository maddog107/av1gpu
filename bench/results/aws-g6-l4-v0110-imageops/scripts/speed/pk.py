#!/usr/bin/env python3
"""pk.py N: kernel time of each prototype against the PR on one stream, N traces per case, order rotating.  Kernel ms from nsys for the
kernels the prototype touches; host wait and any-kernel span too.  Prints medians, one line per case."""
import collections, os, sqlite3, statistics as st, subprocess, sys
H = os.path.expanduser("~"); N = int(sys.argv[1]); D = f"{H}/prof/pk"; os.makedirs(D, exist_ok=True)
C444 = "--yuv 444 --depth 10 --tiles 2,2,2 --max-mp 100 --max-input-mp 100".split()
CONV = "--trim --fuzz 10% --trim-margin 1% --trim-limits 1%,20% --trim-lightness 80 --skip-blank 99".split()
NV = "--trim --fuzz 10% --trim-limits 100%,100%".split(); T10 = "--trim --fuzz 10%".split()
SETS = {"b444": (f"{H}/sets/border", C444), "border": (f"{H}/sets/border", []), "comics": (f"{H}/sets/comics", []), "sr": (f"{H}/images", []),
        "thumbs": (f"{H}/sets/thumbs", []), "blank": (f"{H}/sets/blank", [])}
BIN = lambda p: f"{H}/pr/build/av1gpu" if p == "pr" else f"{H}/proto/{p}/build/av1gpu"
CASES = [("never", NV, ["pr", "p1", "p2", "pc"], ["b444", "border", "comics", "sr", "thumbs", "blank"], "k_trim_measure"),
         ("trim10", T10, ["pr", "p2", "p5", "pc"], ["b444", "border"], "k_trim_measure,k_rz_v,k_rz_h,k_crop"),
         ("auto", ["--auto-level"], ["pr", "p3", "p6", "pc"], ["b444", "border", "comics", "sr"], "k_tone_range"),
         ("level", ["--level", "5%,95%,1.1"], ["pr", "p4"], ["b444", "border", "sr"], "k_tone")]
env = dict(os.environ, GP_ENCODERS="1")
def trace(b, src, opts):
    subprocess.run("rm -rf /dev/shm/out", shell=True)
    r = f"{D}/t"
    subprocess.run(["nsys", "profile", "--trace=cuda", "--sample=none", "--cpuctxsw=none", "--force-overwrite=true", "-o", r, b, "--batch", src, "--outdir", "/dev/shm/out"] + opts,
                   env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.run(["nsys", "export", "--type", "sqlite", "--force-overwrite=true", "-o", r + ".sqlite", r + ".nsys-rep"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    db = sqlite3.connect(r + ".sqlite"); names = dict(db.execute("select id, value from StringIds"))
    t = collections.Counter(); n = collections.Counter()
    for a, b_, s in db.execute("select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL"):
        t[names[s]] += (b_ - a) / 1e6; n[names[s]] += 1
    return t, n
for cname, opts, protos, sets, keys in CASES:
    keys = keys.split(",")
    for s in sets:
        src, ex = SETS[s]
        if not os.path.exists(src): continue
        res = {p: [] for p in protos}
        for r in range(N):
            o = protos[r % len(protos):] + protos[:r % len(protos)]
            for p in o:
                t, n = trace(BIN(p), src, ex + opts)
                res[p].append((sum(t[k] for k in keys), {k: (round(t[k], 2), n[k]) for k in keys if n[k]}))
        base = st.median(x[0] for x in res["pr"])
        for p in protos:
            m = st.median(x[0] for x in res[p])
            print(f"{cname:7s} {s:7s} {p:4s} {m:8.2f} ms ({100 * (m / base - 1):+6.1f} %) [{min(x[0] for x in res[p]):.2f}-{max(x[0] for x in res[p]):.2f}] {res[p][-1][1]}", flush=True)
print("PK_DONE", flush=True)
