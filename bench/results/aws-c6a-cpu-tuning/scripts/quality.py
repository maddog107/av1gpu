#!/usr/bin/env python3
"""Stage 5: SSIMULACRA2 scoring and BD-rates, with bench.py's own functions (bench.encode_cpu_set, bench.score,
bench.bd_rate) and the stock tools (~/opt/bin/avifdec, ssimulacra2 from libjxl v0.11.1, built as in setup.sh).

  quality.py check              score the stock -j1 q60 speed-run outputs (one per image) and compare bytes and
                                scores with the published rows (anchor: aws-g6-l4-gamut aom_s6; aws-c6a-gamut-svt)
  quality.py splits GROUP_LABEL...  score the q60 outputs of those timing runs (~/keep/<GROUP_LABEL>)
  quality.py sweep BUILD ENC[,ENC...] [THREADS] [tiles|notiles]
                                6 quality points (30..80) on the gamut set, scored; BD-rate vs libaom speed 6
                                (aws-g6-l4-gamut) and vs the published stock rows of the same encoder

Results accumulate in ~/out/quality.json."""
import concurrent.futures as cf
import glob
import json
import os
import statistics
import sys
import types

H = os.path.expanduser("~")
sys.path.insert(0, f"{H}/av1gpu/bench")
import bench  # noqa: E402
from PIL import Image  # noqa: E402

IMG = f"{H}/av1gpu/bench/images_gamut"
KEEP, OUT, WORK = f"{H}/keep", f"{H}/out", f"{H}/qwork"
QJ = os.path.join(OUT, "quality.json")
BUILDS = {"stock": f"{H}/opt/bin/avifenc", "tuned": f"{H}/opt-tuned/bin/avifenc", "pgo": f"{H}/opt-pgo/bin/avifenc"}
QS = [30, 40, 50, 60, 70, 80]
ANCHOR = json.load(open(f"{H}/av1gpu/bench/results/aws-g6-l4-gamut/results.json"))["quality"]["aom_s6"]
PUB = json.load(open(f"{H}/av1gpu/bench/results/aws-c6a-gamut-svt/results.json"))["quality"]
TOOLS = types.SimpleNamespace(avifdec=f"{H}/opt/bin/avifdec", dwebp="dwebp", ssimulacra2=f"{H}/opt/bin/ssimulacra2")


def load():
    return json.load(open(QJ)) if os.path.exists(QJ) else {}


def save(res):
    json.dump(res, open(QJ + ".tmp", "w"), indent=1)
    os.replace(QJ + ".tmp", QJ)


def score_many(pairs):
    os.makedirs(WORK, exist_ok=True)
    with cf.ThreadPoolExecutor(os.cpu_count()) as ex:
        return list(ex.map(lambda t: bench.score(TOOLS, t[0], t[1], WORK), pairs))


def score_keep(group_label):
    d = os.path.join(KEEP, group_label)
    outs = sorted(f for f in os.listdir(d) if f.startswith("r0_"))
    bases = [f.split("_", 1)[1][:-len(".avif")] for f in outs]
    scores = score_many([(os.path.join(IMG, b), os.path.join(d, f)) for b, f in zip(bases, outs)])
    return {b: dict(bytes=os.path.getsize(os.path.join(d, f)), ssimulacra2=s) for b, f, s in zip(bases, outs, scores)}


def per_image(rows):
    per = {}
    for r in rows:
        per.setdefault(r["image"], []).append((r["bytes"], r["ssimulacra2"]))
    return per


def bd_summary(test_rows, anchor_rows, sizes):
    t, an = per_image(test_rows), per_image(anchor_rows)
    vals = {im: bench.bd_rate(t[im], an[im]) for im in t if im in an}
    vals = {k: v for k, v in vals.items() if v is not None}
    if not vals:
        return None
    out = dict(n=len(vals), median=statistics.median(vals.values()), mean=statistics.fmean(vals.values()), by_size={})
    for cls, lo, hi in bench.SIZE_CLASSES:
        v = [x for im, x in vals.items() if lo <= sizes[im] < hi]
        if v:
            out["by_size"][cls] = dict(n=len(v), median=statistics.median(v))
    return out


def main():
    mode = sys.argv[1]
    res = load()
    files = sorted(glob.glob(os.path.join(IMG, "*.jpg")))
    sizes = {os.path.basename(f): (lambda im: im.width * im.height / 1e6)(Image.open(f)) for f in files}
    if mode == "check":
        chk = {}
        for enc in ("aom_s6", "aom_s9", "svt_p2", "svt_p4", "svt_p6", "svt_p8", "svt_p9"):
            got = score_keep(f"stock_j1_stock_{enc}_j1")
            pub = {r["image"]: r for r in (ANCHOR if enc == "aom_s6" else PUB[enc]) if r["q"] == 60}
            common = [b for b in got if b in pub]
            same_bytes = sum(got[b]["bytes"] == pub[b]["bytes"] for b in common)
            ds = [got[b]["ssimulacra2"] - pub[b]["ssimulacra2"] for b in common
                  if got[b]["ssimulacra2"] is not None and pub[b]["ssimulacra2"] is not None]
            chk[enc] = dict(images=len(got), compared=len(common), same_bytes=same_bytes,
                            max_abs_score_diff=max(abs(x) for x in ds) if ds else None,
                            source="aws-g6-l4-gamut" if enc == "aom_s6" else "aws-c6a-gamut-svt", per_image=got)
            print(f"check {enc}: {same_bytes}/{len(common)} same bytes as published q60, "
                  f"max |score diff| {chk[enc]['max_abs_score_diff']}", flush=True)
        res["check"] = chk
    elif mode == "splits":
        res.setdefault("splits", {})
        for gl in sys.argv[2:]:
            res["splits"][gl] = score_keep(gl)
            v = res["splits"][gl]
            print(f"splits {gl}: {sum(x['bytes'] for x in v.values())} B, mean score "
                  f"{statistics.fmean(x['ssimulacra2'] for x in v.values() if x['ssimulacra2'] is not None):.4f}",
                  flush=True)
    elif mode == "sweep":
        build, encs = sys.argv[2], sys.argv[3].split(",")
        threads = int(sys.argv[4]) if len(sys.argv) > 4 else 1
        tiles = len(sys.argv) > 5 and sys.argv[5] == "tiles"
        notiles = len(sys.argv) > 5 and sys.argv[5] == "notiles"
        res.setdefault("sweeps", {})
        for enc in encs:
            key = (f"{build}_{enc}" + (f"_j{threads}" if threads != 1 else "") + ("_tiles" if tiles else "")
                   + ("_notiles" if notiles else ""))
            if key in res["sweeps"]:
                continue
            a = types.SimpleNamespace(avifenc=BUILDS[build], svt_tune="3", avifenc_threads=threads, autotiling=tiles,
                                      no_tiling=notiles, jobs=os.cpu_count(), cwebp="cwebp")
            outdir = os.path.join(WORK, key)
            outs = bench.encode_cpu_set(a, enc, files, QS, outdir, True)
            scores = score_many([(src, dst) for src, q, dst in outs])
            rows = [dict(image=os.path.basename(src), q=q, bytes=os.path.getsize(dst), ssimulacra2=s)
                    for (src, q, dst), s in zip(outs, scores)]
            pub_rows = ANCHOR if enc == "aom_s6" else PUB[enc]
            pub = {(r["image"], r["q"]): r["bytes"] for r in pub_rows}
            same = sum(pub.get((r["image"], r["q"])) == r["bytes"] for r in rows)
            e = dict(build=build, encoder=enc, threads=threads, autotiling=tiles, no_tiling=notiles, files=len(rows),
                     same_bytes_as_published_stock=same, rows=rows,
                     bd_vs_aom_s6=bd_summary(rows, ANCHOR, sizes),
                     bd_vs_published_stock_same_encoder=bd_summary(rows, pub_rows, sizes))
            res["sweeps"][key] = e
            save(res)
            b = e["bd_vs_aom_s6"]
            print(f"sweep {key}: {len(rows)} files, {same} same size as published stock; BD vs aom_s6 median "
                  f"{100 * b['median']:+.2f}% mean {100 * b['mean']:+.2f}% (n={b['n']}); vs stock same encoder "
                  f"median {100 * e['bd_vs_published_stock_same_encoder']['median']:+.2f}%", flush=True)
    save(res)
    print(f"QUALITY_DONE {mode}", flush=True)


if __name__ == "__main__":
    main()
