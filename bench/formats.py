#!/usr/bin/env python3
"""av1gpu and libaom in four formats: 8-bit 4:2:0 (the default), 8-bit 4:4:4, 10-bit 4:2:0 and 10-bit 4:4:4.

  GP_ENCODERS=4 bench/formats.py --images bench/images_gamut --out bench/results/my-formats --av1gpu build/av1gpu \\
      --avifdec-codec dav1d --anchor bench/results/aws-g6-l4-gamut/results.json [--svt-preset 6 --svt-tune 3]

bench.py's methodology and code, once per format (avifenc -y 444 / -d 10, av1gpu --yuv 444 / --depth 10):
- Speed at quality 60 on the set repeated up to --min-mp megapixels.  av1gpu runs --rounds interleaved rounds of
  the four formats (alternating order; each a warm-up and --repeats timed runs; reported: the median over all
  timed runs).  libaom speed 6 runs one avifenc -j 1 per CPU, once per format.
- Quality at bench.py's six points.  Outputs are decoded with avifdec (16-bit PNGs for 10-bit) and scored against
  the source JPEG with SSIMULACRA2; per-image BD-rates (median and mean over images): every av1gpu format against
  libaom in the same format; 4:4:4 against 4:2:0 and 10-bit against 8-bit within each encoder; and everything
  against the libaom speed 6 8-bit 4:2:0 encodes in --anchor (a published bench.py results.json).  The new
  8-bit 4:2:0 libaom encodes are checked against --anchor file by file (bytes and scores).
--svt-preset P adds SVT-AV1 preset P at 10-bit 4:2:0 (SVT-AV1 encodes 4:2:0 only);
--svt-anchor (a bench.py results.json with svt_pP rows) compares it with SVT-AV1's 8-bit encodes.
Writes OUT/results.json and OUT/summary.md.  With --skip-speed or --skip-quality, the other phase's results of an
earlier run in OUT are kept.
"""
import argparse
import concurrent.futures as cf
import copy
import glob
import json
import os
import shutil
import statistics
import sys

import numpy as np

import bench

FORMATS = [("420_8", "420", 8), ("444_8", "444", 8), ("420_10", "420", 10), ("444_10", "444", 10)]
LABEL = {"420_8": "8-bit 4:2:0", "444_8": "8-bit 4:4:4", "420_10": "10-bit 4:2:0", "444_10": "10-bit 4:4:4"}
NAME = {"av1gpu": "av1gpu", "aom_s6": "libaom s6"}


def with_format(a, yuv, depth):
    f = copy.copy(a)
    f.yuv, f.depth = yuv, depth
    return f


def per_image(rows):
    out = {}
    for r in rows:
        out.setdefault(r["image"], []).append((r["bytes"], r["ssimulacra2"]))
    return out


def bd_summary(test, anchor, images=None):
    vals = [bench.bd_rate(test[im], anchor[im]) for im in test if im in anchor and (images is None or im in images)]
    vals = [v for v in vals if v is not None]
    return dict(n=len(vals), median=statistics.median(vals), mean=statistics.fmean(vals)) if vals else None


def label(name):
    """'av1gpu_444_10' -> 'av1gpu 10-bit 4:4:4'; 'anchor' -> the published anchor"""
    if name == "anchor":
        return "libaom s6 8-bit 4:2:0 (published)"
    if name == "svt_anchor":
        return "SVT-AV1 8-bit 4:2:0 (published)"
    enc, k = name.rsplit("_", 2)[0], "_".join(name.rsplit("_", 2)[1:])
    enc = NAME.get(enc, enc.replace("svt_p", "SVT-AV1 p"))
    return f"{enc} {LABEL[k]}"


def comparisons(names, svt):
    """(test, anchor) pairs for the BD-rate table, in order"""
    pairs = [(f"av1gpu_{k}", f"aom_s6_{k}") for k, _, _ in FORMATS]
    for enc in ("av1gpu", "aom_s6"):
        pairs += [(f"{enc}_444_{d}", f"{enc}_420_{d}") for d in (8, 10)]
        pairs += [(f"{enc}_{y}_10", f"{enc}_{y}_8") for y in ("420", "444")]
    if svt:
        pairs += [(svt, "aom_s6_420_10"), (svt, "svt_anchor")]
    pairs += [(n, "anchor") for n in names]
    return [(t, r) for t, r in pairs if t in names]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=os.path.join(bench.HERE, "images_gamut"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--av1gpu", default=os.path.join(bench.ROOT, "build", "av1gpu"))
    ap.add_argument("--gpu-label", default="default settings")
    ap.add_argument("--avifenc", default="avifenc")
    ap.add_argument("--avifdec", default="avifdec")
    ap.add_argument("--avifdec-codec", default="", help="avifdec -c CODEC (default: avifdec's own choice)")
    ap.add_argument("--ssimulacra2", default="ssimulacra2")
    ap.add_argument("--quality", default="30,40,50,60,70,80")
    ap.add_argument("--formats", default=",".join(k for k, _, _ in FORMATS), help="subset of " +
                    ", ".join(k for k, _, _ in FORMATS))
    ap.add_argument("--anchor", default="", help="published bench.py results.json with aom_s6 (8-bit 4:2:0) rows")
    ap.add_argument("--svt-preset", type=int, default=None, help="add SVT-AV1 at this preset, 10-bit 4:2:0")
    ap.add_argument("--svt-tune", default="")
    ap.add_argument("--svt-anchor", default="", help="bench.py results.json with svt_pP rows (8-bit 4:2:0)")
    ap.add_argument("--min-mp", type=float, default=1000)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=3, help="interleaved av1gpu speed rounds over the formats")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--keep", action="store_true", help="keep the encoded files under OUT/files")
    a = ap.parse_args()
    a.cwebp = "cwebp"
    fmts = [f for f in FORMATS if f[0] in a.formats.split(",")]
    files = sorted(f for f in glob.glob(os.path.join(a.images, "*")) if f.lower().endswith(bench.EXTS))
    if not files:
        sys.exit(f"no JPEGs in {a.images}")
    os.makedirs(a.out, exist_ok=True)
    work = os.path.join(a.out, "files")
    os.makedirs(work, exist_ok=True)
    from PIL import Image
    sizes = {os.path.basename(f): float(np.prod(Image.open(f).size)) / 1e6 for f in files}
    tune_iq = bench.avifenc_supports_tune_iq(a, files[0], work)
    rpath = os.path.join(a.out, "results.json")
    res = json.load(open(rpath)) if os.path.exists(rpath) else {}
    env = bench.environment(a)
    env.pop("cwebp", None)
    env["avifdec"] = "avifdec " + bench.tool_version([a.avifdec, "--version"]).removeprefix("Version: ")
    env["avifdec_codec"] = a.avifdec_codec or "default"
    env["GP_ENCODERS"] = os.environ.get("GP_ENCODERS", "default")
    res.update(environment=env, images=len(files), megapixels=sum(sizes.values()), avifenc_tune_iq=tune_iq,
               quality_points=a.quality, formats=[k for k, _, _ in fmts])
    svt = f"svt_p{a.svt_preset}_420_10" if a.svt_preset is not None else None
    if svt:
        res["svt_tune"] = a.svt_tune or "default"
    save = lambda: json.dump(res, open(rpath, "w"), indent=1)
    print(f"{len(files)} images, {res['megapixels']:.1f} MP; tune=iq {tune_iq}; GP_ENCODERS={env['GP_ENCODERS']}", flush=True)

    if not a.skip_speed:
        sdir, sfiles, smp, reps = bench.speed_set(files, sizes, a.min_mp, work)
        res["speed_workload"] = dict(images=len(sfiles), megapixels=smp, repeats=reps)
        res["speed"], res["speed_av1gpu_rounds"] = [], {}
        print(f"speed workload: the set {reps}x, {len(sfiles)} images, {smp:.0f} MP", flush=True)
        for rnd in range(a.rounds):   # interleaved, alternating order: drift hits every format alike
            for k, y, d in (fmts if rnd % 2 == 0 else fmts[::-1]):
                r = bench.gpu_speed(with_format(a, y, d), sdir, work)
                res["speed_av1gpu_rounds"].setdefault(k, []).append(r)
                runs = ", ".join("%.1f" % x["mp_s"] for x in r["runs"])
                print(f"speed round {rnd} av1gpu {LABEL[k]}: {r['mp_s']:.1f} MP/s (runs {runs})", flush=True)
        for k, _, _ in fmts:
            rs = res["speed_av1gpu_rounds"][k]
            runs = [x for r in rs for x in r["runs"]]
            med = lambda key: statistics.median(x[key] for x in runs)
            res["speed"].append(dict(codec=rs[0]["codec"], encoder="av1gpu", format=k, images=rs[0]["images"], mp=rs[0]["mp"],
                                     mp_s=med("mp_s"), img_s=med("img_s"), cpu_cores=med("cpu_cores"), wall_s=med("wall_ms") / 1e3,
                                     bytes=rs[0]["bytes"], timed_runs=len(runs), round_mp_s=[r["mp_s"] for r in rs],
                                     min_mp_s=min(x["mp_s"] for x in runs), max_mp_s=max(x["mp_s"] for x in runs)))
        save()
        for k, y, d in fmts:
            r = bench.cpu_speed(with_format(a, y, d), sfiles, 6, tune_iq, smp, work)
            r.update(encoder="aom_s6", format=k)
            res["speed"].append(r)
            print(f"speed {r['codec']}: {r['mp_s']:.1f} MP/s", flush=True)
            save()
        if svt:
            r = bench.cpu_speed(with_format(a, "420", 10), sfiles, a.svt_preset, tune_iq, smp, work, "svt")
            r.update(encoder=f"svt_p{a.svt_preset}", format="420_10")
            res["speed"].append(r)
            print(f"speed {r['codec']}: {r['mp_s']:.1f} MP/s", flush=True)
            save()
        shutil.rmtree(sdir, ignore_errors=True)

    if not a.skip_quality:
        qs = [int(x) for x in a.quality.split(",")]
        res["quality"] = {}
        jobs = []
        for k, y, d in fmts:
            jobs += [(f"av1gpu_{k}", with_format(a, y, d), "av1gpu"), (f"aom_s6_{k}", with_format(a, y, d), "aom_s6")]
        if svt:
            jobs.append((svt, with_format(a, "420", 10), f"svt_p{a.svt_preset}"))
        for name, f, codec in jobs:
            outdir = os.path.join(work, name)
            outs = bench.encode_gpu_set(f, files, qs, outdir) if codec == "av1gpu" else \
                bench.encode_cpu_set(f, codec, files, qs, outdir, tune_iq)
            with cf.ThreadPoolExecutor(a.jobs) as ex:
                scores = list(ex.map(lambda t: bench.score(f, t[0], t[2], work), outs))
            res["quality"][name] = [dict(image=os.path.basename(src), q=q, bytes=os.path.getsize(dst), ssimulacra2=s)
                                    for (src, q, dst), s in zip(outs, scores)]
            print(f"quality {name}: {len(outs)} files, {sum(s is None for s in scores)} not scored", flush=True)
            save()
            if not a.keep:
                shutil.rmtree(outdir, ignore_errors=True)

    if res.get("quality"):
        per = {n: per_image(rows) for n, rows in res["quality"].items()}
        if a.anchor:
            pub = json.load(open(a.anchor))
            per["anchor"] = per_image(pub["quality"]["aom_s6"])
            # the new 8-bit 4:2:0 libaom encodes against the published ones, file by file
            for name, pubname in (("aom_s6_420_8", "aom_s6"), ("av1gpu_420_8", "av1gpu")):
                if name not in res["quality"] or pubname not in pub["quality"]:
                    continue
                old = {(r["image"], r["q"]): r for r in pub["quality"][pubname]}
                new = {(r["image"], r["q"]): r for r in res["quality"][name]}
                both = sorted(set(old) & set(new))
                res.setdefault("anchor_check", {})[name] = dict(
                    published=a.anchor, published_codec=pubname, files_new=len(new), files_published=len(old), common=len(both),
                    same_bytes=sum(old[x]["bytes"] == new[x]["bytes"] for x in both),
                    same_score=sum(old[x]["ssimulacra2"] is not None and new[x]["ssimulacra2"] is not None and
                                   abs(old[x]["ssimulacra2"] - new[x]["ssimulacra2"]) < 1e-6 for x in both),
                    differ=[dict(image=x[0], q=x[1], bytes=[old[x]["bytes"], new[x]["bytes"]],
                                 ssimulacra2=[old[x]["ssimulacra2"], new[x]["ssimulacra2"]])
                            for x in both if old[x]["bytes"] != new[x]["bytes"]][:50])
        if svt and a.svt_anchor:
            per["svt_anchor"] = per_image(json.load(open(a.svt_anchor))["quality"][f"svt_p{a.svt_preset}"])
        names = [n for n in res["quality"]]
        res["bd_rate"], res["bd_rate_by_size"] = {}, {}
        for t, r in comparisons(names, svt):
            if r not in per:
                continue
            v = bd_summary(per[t], per[r])
            if v:
                res["bd_rate"][f"{t}_vs_{r}"] = v
            if r.startswith("aom_s6") and t.startswith("av1gpu") and t[7:] == r[7:]:
                for cls, lo, hi in bench.SIZE_CLASSES:
                    v = bd_summary(per[t], per[r], {im for im, s in sizes.items() if lo <= s < hi})
                    if v:
                        res["bd_rate_by_size"].setdefault(f"{t}_vs_{r}", {})[cls] = v
        res["totals"] = {}
        for n, rows in res["quality"].items():
            for q in sorted({r["q"] for r in rows}):
                rq = [r for r in rows if r["q"] == q]
                sc = [r["ssimulacra2"] for r in rq if r["ssimulacra2"] is not None]
                res["totals"].setdefault(n, {})[str(q)] = dict(files=len(rq), bytes=sum(r["bytes"] for r in rq),
                                                               mean_ssimulacra2=statistics.fmean(sc) if sc else None)
    save()
    write_summary(res, os.path.join(a.out, "summary.md"))
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
    print(open(os.path.join(a.out, "summary.md")).read())


def write_summary(res, path):
    e = res["environment"]
    L = ["# av1gpu formats benchmark", "",
         f"- GPU: {e['gpu']}; `GP_ENCODERS={e['GP_ENCODERS']}`", f"- CPU: {e['cpu']} ({e['cpus']} logical CPUs)",
         f"- {e['avifenc']}; tune=iq: {'yes' if res['avifenc_tune_iq'] else 'no'}; libaom speed 6, one avifenc -j 1 per CPU",
         f"- decoding: {e['avifdec']}, -c {e['avifdec_codec']} (16-bit PNGs for 10-bit); SSIMULACRA2 against the source JPEG",
         f"- {res['images']} images, {res['megapixels']:.1f} MP; quality points {res['quality_points']}"]
    if res.get("svt_tune"):
        L.append(f"- SVT-AV1: tune {res['svt_tune']}")
    if res.get("speed_workload"):
        w = res["speed_workload"]
        L.append(f"- speed runs: the set repeated {w['repeats']}x ({w['images']} images, {w['megapixels']:.0f} MP), quality 60")
    L.append("")
    if res.get("speed"):
        L += ["## Throughput (quality 60)", "",
              "av1gpu: median of all timed runs over the interleaved rounds (round medians in brackets).", "",
              "| Encoder | Format | MP/s | images/s | CPU cores | output MB |", "|---|---|---:|---:|---:|---:|"]
        for s in res["speed"]:
            enc = NAME.get(s["encoder"], s["encoder"].replace("svt_p", "SVT-AV1 preset "))
            rounds = " (%s)" % ", ".join("%.1f" % x for x in s["round_mp_s"]) if s.get("round_mp_s") else ""
            L.append(f"| {enc} | {LABEL[s['format']]} | {s['mp_s']:.1f}{rounds} | {s['img_s']:.1f} | {s['cpu_cores']:.1f} | "
                     f"{s['bytes'] / 1e6:.2f} |")
        L.append("")
    if res.get("bd_rate"):
        L += ["## BD-rate (SSIMULACRA2, per image; negative = smaller files at equal quality)", "",
              "| Test | vs | images | median | mean |", "|---|---|---:|---:|---:|"]
        for k, v in res["bd_rate"].items():
            t, r = k.split("_vs_")
            L.append(f"| {label(t)} | {label(r)} | {v['n']} | {100 * v['median']:+.2f}% | {100 * v['mean']:+.2f}% |")
        L.append("")
    if res.get("bd_rate_by_size"):
        L += ["## av1gpu vs libaom s6 in the same format, by image size (median; images in brackets)", "",
              "| Format | " + " | ".join(c for c, _, _ in bench.SIZE_CLASSES) + " |", "|---|" + "---:|" * len(bench.SIZE_CLASSES)]
        for k, by in res["bd_rate_by_size"].items():
            cells = [f"{100 * by[c]['median']:+.1f}% ({by[c]['n']})" if c in by else "—" for c, _, _ in bench.SIZE_CLASSES]
            L.append(f"| {LABEL['_'.join(k.split('_vs_')[0].split('_')[1:])]} | " + " | ".join(cells) + " |")
        L.append("")
    if res.get("totals"):
        qs = sorted({int(q) for t in res["totals"].values() for q in t})
        L += ["## Bytes and mean SSIMULACRA2 per quality point (all images)", "",
              "| Encoder | " + " | ".join(f"q{q}" for q in qs) + " |", "|---|" + "---:|" * len(qs)]
        for n, t in res["totals"].items():
            cells = [f"{t[str(q)]['bytes'] / 1e6:.2f} MB, {t[str(q)]['mean_ssimulacra2']:.2f}" if str(q) in t else "—" for q in qs]
            L.append(f"| {label(n)} | " + " | ".join(cells) + " |")
        L.append("")
    for n, c in res.get("anchor_check", {}).items():
        L.append(f"- {label(n)} against the published `{c['published_codec']}` rows ({c['published']}): {c['common']} common files, "
                 f"{c['same_bytes']} with the same bytes, {c['same_score']} with the same SSIMULACRA2 score")
    open(path, "w").write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
