"""Compare encoded file sizes with the published results.json rows, per codec, image and quality point.
bytes_check.py REPO FILESDIR NVDIR OUT.json   (FILESDIR = bench.py --keep's OUT/files; NVDIR = NVENC outputs)
Only (image, q) pairs present in a published file are compared; extra quality points are listed as such."""
import json, os, sys
REPO, FILES, NV, OUT = sys.argv[1:5]
sys.path.insert(0, os.path.join(REPO, "bench"))
import bench
R = os.path.join(REPO, "bench/results")
pub = {}  # codec -> {(image, q): {source: bytes}}
for src, codecs in (("aws-g6-l4-gamut", ("av1gpu", "aom_s6", "aom_s9", "webp")),
                    ("aws-c6a-gamut-svt", ("aom_s6", "aom_s9", "svt_p4", "svt_p6", "svt_p9", "webp")),
                    ("aws-g6-l4-nvenc", ("nvenc",))):
    q = json.load(open(os.path.join(R, src, "results.json")))["quality"]
    for c in codecs:
        for r in q[c]:
            pub.setdefault(c, {}).setdefault((r["image"], r["q"]), {})[src] = r["bytes"]


def local(codec):
    d = NV if codec == "nvenc" else os.path.join(FILES, codec)
    out = {}
    if not os.path.isdir(d):
        return out
    inv = {}
    for f in os.listdir(d):
        if not f.endswith((".avif", ".webp")) or ".q" not in f:
            continue
        img, q = f.rsplit(".q", 1)
        q = int(q.split(".")[0])
        if codec == "av1gpu":  # av1gpu names files by qindex
            inv = inv or {bench.qindex_for_quality(x): x for x in (30, 40, 50, 60, 70, 80, 85, 90, 95, 98)}
            q = inv.get(q, -q)
        out[(img, q)] = os.path.getsize(os.path.join(d, f))
    return out


res = {}
for codec, p in pub.items():
    loc = local(codec)
    cmp = [(k, v) for k, v in loc.items() if k in p]
    bad = [dict(image=k[0], q=k[1], bytes=v, published=p[k]) for k, v in cmp if any(b != v for b in p[k].values())]
    extra = sorted({k[1] for k in loc if k not in p})
    missing = sorted(k for k in p if k not in loc and k[0] in {x[0] for x in loc})
    res[codec] = dict(local_files=len(loc), compared=len(cmp), identical=len(cmp) - len(bad), mismatched=bad[:50],
                      n_mismatched=len(bad), extra_q=extra, published_not_found=[list(k) for k in missing][:50],
                      sources=sorted({s for v in p.values() for s in v}))
    print(f"{codec:8s} files {len(loc):5d} compared {len(cmp):5d} identical {len(cmp) - len(bad):5d} "
          f"mismatched {len(bad)} extra q {extra} missing {len(missing)}", flush=True)
json.dump(res, open(OUT, "w"), indent=1)
