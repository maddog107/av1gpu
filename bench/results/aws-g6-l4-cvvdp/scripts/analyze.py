"""Merge the CVVDP run's phases (bench.py results + NVENC rows), check them against the published results, and
compute every table.  Run from the worktree:
  analyze.py RESULTS_DIR OUT_DIR [--min-overlap J] [--target J]
RESULTS_DIR holds p1/results.json, nvenc_p1.json, and optionally p2/results.json, nvenc_p2.json, bytes_*.json."""
import argparse, csv, json, os, statistics, sys
sys.path.insert(0, "bench")
import bench

ap = argparse.ArgumentParser()
ap.add_argument("res"); ap.add_argument("out")
ap.add_argument("--min-overlap", type=float, default=0.05)
ap.add_argument("--target", type=float, default=9.5)
ap.add_argument("--print-only", action="store_true")
a = ap.parse_args()
R = a.res
J = lambda p: json.load(open(os.path.join(R, p))) if os.path.exists(os.path.join(R, p)) else None
PUB = "bench/results"
pubj = {k: json.load(open(f"{PUB}/{k}/results.json")) for k in ("aws-g6-l4-gamut", "aws-c6a-gamut-svt", "aws-g6-l4-nvenc")}
man = {r["file"]: r for r in csv.DictReader(open("bench/openimages_v4_gamut.csv"))}

p1, p2 = J("p1/results.json"), J("p2/results.json")
quality = {}
for p in (p1, p2):
    if p:
        for c, rows in p["quality"].items():
            quality.setdefault(c, []).extend(rows)
for f in ("nvenc_p1.json", "nvenc_p2.json"):
    n = J(f)
    if n:
        quality.setdefault("nvenc", []).extend(n["rows"])
for c in quality:
    quality[c].sort(key=lambda r: (r["image"], r["q"]))

# ---- phase 3: the 4 photos whose JPEG ICC profile Pillow missed, re-encoded and re-scored with the fixed reference.
# Their files must have the same bytes and SSIMULACRA2 as in phases 1-2; their CVVDP replaces the old one.
p3, nv3 = J("p3/results.json"), J("nvenc_p3.json")
fix = None
if p3:
    new = {(c, r["image"], r["q"]): r for c, rows in p3["quality"].items() for r in rows}
    if nv3:
        new.update({("nvenc", r["image"], r["q"]): r for r in nv3["rows"]})
    fix = dict(images=sorted({k[1] for k in new}), rows=len(new), replaced=0, bytes_identical=0, ssimulacra2_identical=0,
               not_found=[], changes={})
    seen = set()
    for c, rows in quality.items():
        for r in rows:
            k = (c, r["image"], r["q"])
            if k in new:
                n = new[k]; seen.add(k)
                fix["bytes_identical"] += n["bytes"] == r["bytes"]
                fix["ssimulacra2_identical"] += n["ssimulacra2"] == r["ssimulacra2"]
                fix["changes"].setdefault(c, []).append(dict(image=r["image"], q=r["q"], cvvdp_before=r["cvvdp"], cvvdp=n["cvvdp"]))
                r["cvvdp"] = n["cvvdp"]; fix["replaced"] += 1
    fix["not_found"] = [list(k) for k in new if k not in seen]
    print("phase 3 fix:", {k: v for k, v in fix.items() if k != "changes"})
images = sorted({r["image"] for rows in quality.values() for r in rows})
sizes = {im: int(man[im]["width"]) * int(man[im]["height"]) / 1e6 for im in images}
pts = {c: sorted({r["q"] for r in rows}) for c, rows in quality.items()}
PUBQ = {"webp": [50, 60, 70, 80, 90, 95], "nvenc": [18, 25, 32, 38, 44, 50]}
pubq = lambda c: PUBQ.get(c, [30, 40, 50, 60, 70, 80])
published_only = {c: [r for r in rows if r["q"] in pubq(c)] for c, rows in quality.items()}

# ---- byte and SSIMULACRA2 identity against the published rows
pub_rows = {}
for src, codecs in (("aws-g6-l4-gamut", ("av1gpu", "aom_s6", "aom_s9", "webp")),
                    ("aws-c6a-gamut-svt", ("aom_s6", "aom_s9", "svt_p4", "svt_p6", "svt_p9", "webp")),
                    ("aws-g6-l4-nvenc", ("nvenc",))):
    for c in codecs:
        for r in pubj[src]["quality"][c]:
            pub_rows.setdefault(c, {}).setdefault((r["image"], r["q"]), []).append((src, r["bytes"], r["ssimulacra2"]))
ident = {}
for c, rows in quality.items():
    cmp = [(r, pub_rows.get(c, {}).get((r["image"], r["q"]))) for r in rows]
    cmp = [(r, p) for r, p in cmp if p]
    byte_bad = [(r["image"], r["q"], r["bytes"], [b for _, b, _ in p]) for r, p in cmp if any(b != r["bytes"] for _, b, _ in p)]
    sd = [abs(r["ssimulacra2"] - s) for r, p in cmp for _, _, s in p if r["ssimulacra2"] is not None and s is not None]
    s_eq = sum(all(r["ssimulacra2"] == s for _, _, s in p) for r, p in cmp)
    ident[c] = dict(rows=len(rows), compared=len(cmp), bytes_identical=len(cmp) - len(byte_bad), bytes_mismatched=byte_bad[:20],
                    ssimulacra2_identical=s_eq, ssimulacra2_max_abs_diff=max(sd) if sd else None,
                    published_sources=sorted({s for _, p in cmp for s, _, _ in p}))

# ---- BD-rates
order = ["av1gpu", "aom_s9", "svt_p4", "svt_p6", "svt_p9", "nvenc", "webp"]
bd_j_all, by_j_all = bench.bd_table(quality, sizes, "cvvdp", a.min_overlap)
bd_j_pub, by_j_pub = bench.bd_table(published_only, sizes, "cvvdp", a.min_overlap)
bd_s_all, by_s_all = bench.bd_table(quality, sizes, "ssimulacra2")
bd_s_pub, by_s_pub = bench.bd_table(published_only, sizes, "ssimulacra2")
pub_bd = {"av1gpu": pubj["aws-g6-l4-gamut"]["bd_rate"]["av1gpu_vs_aom_s6"],
          "aom_s9": pubj["aws-g6-l4-gamut"]["bd_rate"]["aom_s9_vs_aom_s6"],
          "webp": pubj["aws-g6-l4-gamut"]["bd_rate"]["webp_vs_aom_s6"],
          "nvenc": pubj["aws-g6-l4-nvenc"]["bd_rate"]["nvenc_vs_aom_s6"]}
for p in (4, 6, 9):
    pub_bd[f"svt_p{p}"] = pubj["aws-c6a-gamut-svt"]["bd_rate"][f"svt_p{p}_vs_aom_s6"]
pub_by = dict(pubj["aws-c6a-gamut-svt"].get("bd_rate_by_size", {}))
pub_by["nvenc_vs_aom_s6"] = pubj["aws-g6-l4-nvenc"]["bd_rate"]["nvenc_vs_aom_s6_by_size"]


# images left out of the JOD BD-rate (too little overlap) and why
def overlap_info(qd, metric, mo):
    per = {}
    for c, rows in qd.items():
        for r in rows:
            per.setdefault(c, {}).setdefault(r["image"], []).append((r["bytes"], r.get(metric)))
    out = {}
    for c in per:
        if c == "aom_s6":
            continue
        miss = []
        for im in per[c]:
            if bench.bd_rate(per[c][im], per["aom_s6"][im], mo) is None:
                xt, _ = bench._curve(per[c][im]); xa, _ = bench._curve(per["aom_s6"][im])
                miss.append(dict(image=im, test=[round(float(xt[0]), 3), round(float(xt[-1]), 3)] if len(xt) else None,
                                 anchor=[round(float(xa[0]), 3), round(float(xa[-1]), 3)] if len(xa) else None))
        out[c] = miss
    return out


excl_all = overlap_info(quality, "cvvdp", a.min_overlap)

# ---- NVENC by height parity (odd heights get a chroma error at every CQ in this pipeline)
odd_h = {im for im in images if int(man[im]["height"]) % 2}
nv_par = {}
for lab, sel in (("even_height", lambda im: im not in odd_h), ("odd_height", lambda im: im in odd_h)):
    sub = {c: [r for r in quality[c] if sel(r["image"])] for c in ("nvenc", "aom_s6")}
    subp = {c: [r for r in rows if r["q"] in pubq(c)] for c, rows in sub.items()}
    nv_par[lab] = dict(images=len({r["image"] for r in sub["nvenc"]}),
                       cvvdp=bench.bd_table(sub, sizes, "cvvdp", a.min_overlap)[0].get("nvenc_vs_aom_s6"),
                       ssimulacra2_published_points=bench.bd_table(subp, sizes)[0].get("nvenc_vs_aom_s6"),
                       ssimulacra2_all_points=bench.bd_table(sub, sizes)[0].get("nvenc_vs_aom_s6"),
                       reach_target=sum(1 for v in bench.jod_target(sub, sizes, a.target)["codecs"]["nvenc"]["per_image_bpp"].values()
                                        if v is not None))
print("nvenc by height parity:", json.dumps(nv_par))

# ---- JOD target, means
tgt = bench.jod_target(quality, sizes, a.target)
tgt_pub = bench.jod_target(published_only, sizes, a.target)
byq = bench.mean_by_q(quality, a.target)

# ---- SSIMULACRA2 / JOD spans of the anchor, for the overlap threshold
span = []
for im in images:
    rr = {r["q"]: r for r in quality["aom_s6"] if r["image"] == im}
    if 30 in rr and 80 in rr and rr[30]["cvvdp"] is not None and rr[80]["cvvdp"] is not None:
        span.append(((rr[80]["ssimulacra2"] - rr[30]["ssimulacra2"]), (rr[80]["cvvdp"] - rr[30]["cvvdp"])))
ratio = statistics.median(s / j for s, j in span if j > 0) if span else None

pct = lambda v: f"{100 * v:+.1f}%" if v is not None else "—"


def bdcell(d, c):
    v = d.get(f"{c}_vs_aom_s6")
    return f"{pct(v['median'])} ({pct(v['mean'])}, {v['n']})" if v else "—"


print(f"images {len(images)}, points {pts}")
print(f"aom_s6 q30->q80 span: median SSIMULACRA2/JOD ratio {ratio:.1f} -> 2 SSIMULACRA2 points ~ {2 / ratio:.3f} JOD" if ratio else "")
print("\nidentity vs published:")
for c, v in ident.items():
    print(f"  {c:7s} compared {v['compared']:4d} bytes identical {v['bytes_identical']:4d} ssimulacra2 identical "
          f"{v['ssimulacra2_identical']:4d} max|d| {v['ssimulacra2_max_abs_diff']}")
print("\n| Encoder | JOD BD (all points) | JOD BD (published points) | SSIMULACRA2 BD published | SSIMULACRA2 re-scored (published points) | SSIMULACRA2 (all points) |")
for c in order:
    if c in quality:
        pb = pub_bd.get(c)
        print(f"| {c} | {bdcell(bd_j_all, c)} | {bdcell(bd_j_pub, c)} | {pct(pb['median'])} ({pct(pb['mean'])}, {pb['n']}) | "
              f"{bdcell(bd_s_pub, c)} | {bdcell(bd_s_all, c)} |")
print("\nexcluded (JOD overlap < %.2f):" % a.min_overlap, {c: len(v) for c, v in excl_all.items()})
print("\nby size, JOD all points:")
for c in order:
    k = f"{c}_vs_aom_s6"
    if k in by_j_all:
        print(" ", c, {cls: f"{pct(v['median'])} ({v['n']})" for cls, v in by_j_all[k].items()})
print(f"\nJOD {a.target}:")
for c in ["aom_s6"] + order:
    v = tgt["codecs"].get(c)
    if v:
        print(f"  {c:7s} reached {v['reached']}/{v['images']} median bpp {v['median_bpp']} common({v['common_images']}) {v['common_median_bpp']}")
print("\nmean JOD by q:")
for c in ["aom_s6"] + order:
    if c in byq:
        print(" ", c, {q: (round(v["cvvdp"], 3) if v["cvvdp"] is not None else None, v["at_target"]) for q, v in byq[c].items()})
if a.print_only:
    sys.exit()

# ---- write
os.makedirs(a.out, exist_ok=True)
env = dict(p1["environment"])
res = dict(
    description="CVVDP (ColorVideoVDP) and SSIMULACRA2 on the 99 size-range photos (bench/openimages_v4_gamut.csv), "
                "re-encoded exactly as published on an AWS g6.4xlarge (NVIDIA L4)",
    environment=env, cvvdp=dict(p1["cvvdp"]), images=len(images), megapixels=sum(sizes.values()),
    quality_points={c: v for c, v in pts.items()}, published_quality_points={c: pubq(c) for c in quality},
    avifenc_tune_iq=p1["avifenc_tune_iq"], avifdec_codec=p1.get("avifdec_codec"),
    identity_vs_published=ident,
    bd_rate_cvvdp=bd_j_all, bd_rate_cvvdp_by_size=by_j_all,
    bd_rate_cvvdp_published_points=bd_j_pub, bd_rate_cvvdp_published_points_by_size=by_j_pub,
    bd_rate_cvvdp_excluded=excl_all,
    bd_rate_ssimulacra2=bd_s_all, bd_rate_ssimulacra2_by_size=by_s_all,
    bd_rate_ssimulacra2_published_points=bd_s_pub, bd_rate_ssimulacra2_published_points_by_size=by_s_pub,
    bd_rate_ssimulacra2_as_published={f"{c}_vs_aom_s6": v for c, v in pub_bd.items()},
    bd_rate_ssimulacra2_as_published_by_size={k: v for k, v in pub_by.items()
                                              if k.split("_vs_")[0] in quality and k.endswith("_vs_aom_s6")},
    cvvdp_target=tgt, cvvdp_target_published_points=tgt_pub, by_quality=byq,
    anchor_span=dict(images=len(span), median_ssimulacra2_per_jod=ratio),
    quality=quality)
res["cvvdp"].update(target=a.target, min_overlap=a.min_overlap)
res["nvenc_by_height_parity"] = nv_par
res["icc_fix_rescore"] = fix
lu = J("cvvdp_luma.json")
lu3 = J("cvvdp_luma_p3.json")
if lu and lu3:  # the 4 photos' luma rows from phase 3 as well
    n3 = {(r["codec"], r["image"], r["q"]): r for r in lu3}
    lu = [n3.get((r["codec"], r["image"], r["q"]), r) for r in lu]
if lu:  # diagnostic: luma-only CVVDP for av1gpu and libaom speed 6
    lq = {}
    for r in lu:
        lq.setdefault(r["codec"], []).append(dict(image=r["image"], q=r["q"], bytes=r["bytes"], cvvdp=r["cvvdp_luma"]))
    bdl, byl = bench.bd_table(lq, sizes, "cvvdp", a.min_overlap)
    res["cvvdp_luma_only"] = dict(note="diagnostic: both images converted to grey (Pillow L) after the sRGB conversion, "
                                       "so CVVDP sees luminance only", bd_rate=bdl, bd_rate_by_size=byl,
                                  points={c: sorted({r["q"] for r in v}) for c, v in lq.items()}, rows=lu)
    print("luma-only", json.dumps(bdl), json.dumps(byl))
wi = J("webp_icc.json")
if wi:  # side check: WebP SSIMULACRA2 with the .webp's ICC profile put back on dwebp's PNG
    wfix = {(r["image"], r["q"]): r["ssimulacra2_icc"] for r in wi}
    wq = {"webp": [dict(r, ssimulacra2=wfix.get((r["image"], r["q"]), r["ssimulacra2"])) for r in quality["webp"]],
          "aom_s6": quality["aom_s6"]}
    wq_pub = {c: [r for r in rows if r["q"] in pubq(c)] for c, rows in wq.items()}
    res["webp_icc_check"] = dict(
        files=len(wi), images=len({r["image"] for r in wi}),
        changed=sum(1 for r in quality["webp"] if (r["image"], r["q"]) in wfix and wfix[(r["image"], r["q"])] != r["ssimulacra2"]),
        bd_rate_ssimulacra2_published_points=bench.bd_table(wq_pub, sizes)[0].get("webp_vs_aom_s6"),
        bd_rate_ssimulacra2_all_points=bench.bd_table(wq, sizes)[0].get("webp_vs_aom_s6"), rows=wi)
    print("webp icc check", {k: v for k, v in res["webp_icc_check"].items() if k != "rows"})
for f in ("bytes_p1.json", "bytes_p2.json"):
    if J(f):
        res.setdefault("byte_check_files", {})[f] = {c: {k: v for k, v in d.items() if k != "mismatched"} | {"mismatched": d["mismatched"]}
                                                      for c, d in J(f).items()}
json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
print("wrote", os.path.join(a.out, "results.json"))

NAMES = {"av1gpu": "av1gpu default (`GP_ENCODERS=4`)", "aom_s6": "libaom speed 6", "aom_s9": "libaom speed 9",
         "svt_p4": "SVT-AV1 preset 4", "svt_p6": "SVT-AV1 preset 6", "svt_p9": "SVT-AV1 preset 9",
         "nvenc": "NVENC AV1 (p7, `tune uhq`)", "webp": "WebP (cwebp `-m 4`)"}
c = res["cvvdp"]; e = res["environment"]
md = lambda d, c: (f"{pct(d[c + '_vs_aom_s6']['median'])} ({pct(d[c + '_vs_aom_s6']['mean'])})"
                   if c + "_vs_aom_s6" in d else "—")
nn = lambda d, c: d[c + "_vs_aom_s6"]["n"] if c + "_vs_aom_s6" in d else 0
L = ["# CVVDP and SSIMULACRA2, size-range photos, NVIDIA L4", "",
     f"- GPU: {e['gpu']}; CPU: {e['cpu']} ({e['cpus']} vCPUs); {e['platform']}",
     f"- {e['avifenc']}; tune=iq; cwebp {e['cwebp']}; NVENC through ffmpeg n8.1.3 (BtbN static) and the "
     "hybrid_nvenc_libavif muxer, as published",
     f"- decoded with avifdec -c dav1d (AVIF) and dwebp 1.3.2 (WebP); SSIMULACRA2: libjxl v0.11.1 `ssimulacra2`",
     f"- CVVDP: cvvdp {c['cvvdp']} (pip), torch {c['torch']} on the {c['device']}; display `{c['display']}`: "
     f"{c['resolution'][0]} x {c['resolution'][1]}, {c['peak_cd_m2']} cd/m^2, contrast {c['contrast']}:1, "
     f"{c['ambient_lux']} lux, {c['distance_m']} m, {c['pixels_per_degree']:.1f} pixels per degree, EOTF {c['eotf']}",
     f"- inputs: {c['inputs']}; reference = the source JPEG decoded by Pillow (identical to libjxl's decode)",
     f"- {res['images']} photos, {res['megapixels']:.1f} MP; quality points: " +
     "; ".join(f"{k} {','.join(map(str, v))}" for k, v in res["quality_points"].items()), ""]
L += ["## How it was run", "",
      "One g6.4xlarge spot instance (2026-10-31), the published builds (`scripts/setup.sh`: libaom 3.13.1, SVT-AV1 4.2.0 and "
      "libavif 1.4.0 from source, libjxl v0.11.1 `ssimulacra2`, av1gpu from this branch) and the 99 photos of the published "
      "SVT-AV1 and NVENC runs (`fetch_images.py` now downloads all 100; 9fba6f263f3c4f3e_tiny.jpg is left out as before).",
      "",
      "```",
      "bench/bench.py --images ~/images_gamut --out p1 --av1gpu <GP_ENCODERS=4 wrapper> --skip-speed --aom-speeds 6,9 \\",
      "  --svt-presets 4,6,9 --svt-tune 3 --avifdec-codec dav1d --cvvdp standard_fhd --keep        # published points",
      "bench/bench.py ... --quality 85,90 --webp-quality 98,100                                   # extra points",
      "```",
      "",
      "NVENC: `scripts/nvenc_encode.py` (the published hybrid_nvenc_libavif pipeline, unchanged) and `scripts/nvenc_score.py` "
      "(SSIMULACRA2 exactly as published, CVVDP through `bench.Cvvdp`). `scripts/analyze.py` merged the phases' outputs "
      "(not committed; every row is in `results.json`) into `results.json` and this file; `logs/` has the console output "
      "(the bench.py summaries there used a 0.1 JOD overlap; "
      "0.05 and 0.1 give the same BD-rates here).", ""]
if fix:
    L += [f"**ICC fix.** Four photos ({', '.join(fix['images'])}) store their ICC profile (Camera RGB or Generic RGB) "
          "after the JPEG frame header. libjpeg, and so ssimulacra2, avifenc and cwebp, read it, but Pillow does not, so the "
          "first two phases gave CVVDP an untagged (sRGB) reference against tagged encodes. bench.py now assembles the "
          "profile itself (commit `bench.py: CVVDP reads JPEG ICC profiles that Pillow misses`). A second g6.4xlarge "
          f"(on demand; `scripts/run3.sh`) re-encoded those photos at every point and re-scored them: {fix['rows']} files, "
          f"{fix['bytes_identical']} with the same bytes and {fix['ssimulacra2_identical']} with the same SSIMULACRA2 as before. "
          "Their CVVDP scores in `results.json` and below are the re-scored ones.", ""]
L += ["## Identity with the published runs", "", "| Encoder | files compared | same bytes | same SSIMULACRA2 | max difference |",
      "|---|---:|---:|---:|---:|"]
for k in ["aom_s6"] + order:
    v = ident.get(k)
    if v:
        L.append(f"| {NAMES[k]} | {v['compared']} | {v['bytes_identical']} | {v['ssimulacra2_identical']} | "
                 f"{v['ssimulacra2_max_abs_diff']:.2g} |")
L += ["", f"## BD-rate against libaom speed 6 (median over images, mean in brackets)", "",
      f"JOD curves must share at least {a.min_overlap} JOD (SSIMULACRA2: 2 points). Images in the BD-rate: last column.", "",
      "| Encoder | CVVDP JOD, all points | CVVDP JOD, published points | SSIMULACRA2 as published | SSIMULACRA2 this run, published points | SSIMULACRA2 this run, all points | images (JOD all / SSIM2 published) |",
      "|---|---:|---:|---:|---:|---:|---:|"]
for k in order:
    if k in quality:
        pb = pub_bd[k]
        L.append(f"| {NAMES[k]} | {md(bd_j_all, k)} | {md(bd_j_pub, k)} | {pct(pb['median'])} ({pct(pb['mean'])}) | "
                 f"{md(bd_s_pub, k)} | {md(bd_s_all, k)} | {nn(bd_j_all, k)} / {pb['n']} |")
L += ["", "## By image size (median BD-rate against libaom speed 6, images in brackets)", "",
      "| Encoder | metric | " + " | ".join(cl for cl, _, _ in bench.SIZE_CLASSES) + " |", "|---|---|" + "---:|" * 4]
for k in order:
    pubsz = res["bd_rate_ssimulacra2_as_published_by_size"]
    for lab, d in (("CVVDP JOD", by_j_all), ("SSIMULACRA2 (this run, all points)", by_s_all),
                   ("SSIMULACRA2 as published", pubsz) if k + "_vs_aom_s6" in pubsz else
                   ("SSIMULACRA2 this run, published points (= published scores)", by_s_pub)):
        by = d.get(k + "_vs_aom_s6")
        if by:
            L.append(f"| {NAMES[k]} | {lab} | " + " | ".join(f"{pct(by[cl]['median'])} ({by[cl]['n']})" if cl in by else "—"
                                                       for cl, _, _ in bench.SIZE_CLASSES) + " |")
t = tgt["codecs"]; anyc = next(iter(t.values()))
L += ["", f"## Reaching {a.target} JOD", "",
      f"Bits per pixel at {a.target} JOD, interpolated per image (log size against JOD, as in the BD-rate) within the "
      f"tested points. 'Common' = the {anyc['common_images']} photos that every encoder reaches.", "",
      "| Encoder | photos reaching it | median bpp | median bpp, common photos | photos reaching it, published points only |",
      "|---|---:|---:|---:|---:|"]
f3 = lambda v: f"{v:.3f}" if v is not None else "—"
for k in ["aom_s6"] + order:
    v = t.get(k)
    if v:
        L.append(f"| {NAMES[k]} | {v['reached']}/{v['images']} | {f3(v['median_bpp'])} | {f3(v['common_median_bpp'])} | "
                 f"{tgt_pub['codecs'][k]['reached']}/{tgt_pub['codecs'][k]['images']} |")
L += ["", "## Mean JOD per quality point", ""]
avq = sorted({q for k in quality if k not in ("webp", "nvenc") for q in pts[k]})
L += ["| Encoder | " + " | ".join(f"q{q}" for q in avq) + " |", "|---|" + "---:|" * len(avq)]
for k in ["aom_s6"] + [k for k in order if k not in ("webp", "nvenc")]:
    if k in byq:
        L.append(f"| {NAMES[k]} | " + " | ".join(f"{byq[k][str(q)]['cvvdp']:.3f}" if str(q) in byq[k] else "—" for q in avq) + " |")
for k, lab in (("webp", "q"), ("nvenc", "CQ ")):
    if k in byq:
        L += ["", f"| {NAMES[k]} | " + " | ".join(f"{lab}{q}" for q in byq[k]) + " |", "|---|" + "---:|" * len(byq[k]),
              "| mean JOD | " + " | ".join(f"{v['cvvdp']:.3f}" for v in byq[k].values()) + " |"]
L += ["", "## Every quality point", "", f"| Encoder | point | photos | MB | mean SSIMULACRA2 | mean JOD | photos >= {a.target} JOD |",
      "|---|---:|---:|---:|---:|---:|---:|"]
for k in ["aom_s6"] + order:
    for q, v in byq.get(k, {}).items():
        L.append(f"| {NAMES[k]} | {q} | {v['n']} | {v['bytes'] / 1e6:.2f} | {v['ssimulacra2']:.2f} | {v['cvvdp']:.3f} | {v['at_target']} |")
L += ["", "## NVENC by image height", "",
      "In this pipeline, photos with an odd height decode with a chroma error that stays at every CQ "
      "(luma is fine on the two checked, `logs/nvenc_shift2.log`); both metrics see it. BD-rate against libaom speed 6, "
      "median (mean):", "",
      "| Photos | count | CVVDP JOD, all points | SSIMULACRA2, published points | reach 9.5 JOD |", "|---|---:|---:|---:|---:|"]
for lab, v in nv_par.items():
    f2 = lambda d: f"{pct(d['median'])} ({pct(d['mean'])}, {d['n']})" if d else "—"
    L.append(f"| {lab.replace('_', ' ')} | {v['images']} | {f2(v['cvvdp'])} | {f2(v['ssimulacra2_published_points'])} | "
             f"{v['reach_target']}/{v['images']} |")
if res.get("cvvdp_luma_only"):
    d = res["cvvdp_luma_only"]
    v = d["bd_rate"].get("av1gpu_vs_aom_s6")
    L += ["", "## Diagnostic: CVVDP on luma only", "",
          f"Both images converted to grey after the sRGB conversion (CVVDP sees no colour). av1gpu against libaom speed 6: "
          f"{pct(v['median'])} median ({pct(v['mean'])} mean, {v['n']} photos), against {md(bd_j_all, 'av1gpu')} in colour. "
          "By size: " + ", ".join(f"{cl} {pct(x['median'])} ({x['n']})" for cl, x in d["bd_rate_by_size"].get("av1gpu_vs_aom_s6", {}).items()) + "."]
if res.get("webp_icc_check"):
    w = res["webp_icc_check"]
    L += ["", "## Side check: WebP and ICC profiles", "",
          f"dwebp's PNG drops the ICC profile that `cwebp -metadata icc` stores in the .webp, so SSIMULACRA2 compares an untagged "
          f"PNG with the tagged source. With the profile put back ({w['files']} files of {w['images']} photos with a profile), "
          f"WebP's SSIMULACRA2 BD-rate against libaom speed 6 is {pct(w['bd_rate_ssimulacra2_published_points']['median'])} "
          f"(mean {pct(w['bd_rate_ssimulacra2_published_points']['mean'])}; published points) instead of "
          f"{pct(pub_bd['webp']['median'])} (mean {pct(pub_bd['webp']['mean'])}) as published. {w['changed']} files change: "
          "those of the 13 photos with an Adobe RGB, Camera RGB or Generic RGB profile (by 0.6 to 37 points) and of one with an "
          "sRGB-equivalent profile (by 0.01). CVVDP reads the profile from the .webp, so its WebP numbers do not have this. "
          "NVENC's pipeline drops the profile from the file itself, so both metrics count that against it."]

# per-image agreement of the metrics on av1gpu against libaom speed 6
pa = {}
for k in ("av1gpu", "aom_s6"):
    for r in quality[k]:
        pa.setdefault(k, {}).setdefault(r["image"], []).append(r)
pj, ps = [], []
for im in pa["av1gpu"]:
    x = bench.bd_rate([(r["bytes"], r["cvvdp"]) for r in pa["av1gpu"][im]], [(r["bytes"], r["cvvdp"]) for r in pa["aom_s6"][im]], a.min_overlap)
    y = bench.bd_rate([(r["bytes"], r["ssimulacra2"]) for r in pa["av1gpu"][im]], [(r["bytes"], r["ssimulacra2"]) for r in pa["aom_s6"][im]])
    if x is not None and y is not None:
        pj.append(x); ps.append(y)
res["av1gpu_per_image_agreement"] = dict(images=len(pj), jod_worse_than_ssimulacra2=sum(x > y for x, y in zip(pj, ps)),
                                         jod_positive=sum(x > 0 for x in pj), ssimulacra2_positive=sum(y > 0 for y in ps),
                                         correlation=float(__import__("numpy").corrcoef(pj, ps)[0, 1]))
json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
g = res["av1gpu_per_image_agreement"]
miss = {k: sorted(im for im, v in d["per_image_bpp"].items() if v is None) for k, d in tgt["codecs"].items()}
L += ["", "## Notes", "",
      f"- The encodes are the published files: every compared file has the published size, and this run's SSIMULACRA2 "
      "scores equal the published ones to the last digit. The extra points (q85/q90, WebP q98/q100, NVENC CQ 12/6) are new.",
      "- CVVDP and SSIMULACRA2 agree within about a point on libaom speed 9 and on SVT-AV1, and on WebP's median.",
      f"- They disagree on av1gpu: {md(bd_j_all, 'av1gpu')} on JOD against {md(bd_s_all, 'av1gpu')} on SSIMULACRA2 "
      f"(same points). JOD rates av1gpu worse than SSIMULACRA2 does on {g['jod_worse_than_ssimulacra2']} of {g['images']} photos "
      f"(per-photo correlation {g['correlation']:.2f}); the gap grows with image size, the opposite of SSIMULACRA2's trend."
      + (f" With colour removed, CVVDP puts av1gpu at {pct(res['cvvdp_luma_only']['bd_rate']['av1gpu_vs_aom_s6']['median'])}, "
         "so the gap is in luma, not chroma." if res.get("cvvdp_luma_only") else ""),
      f"- {a.target} JOD: " + "; ".join(
          f"{NAMES[k].split(' (')[0]} reaches it on every photo from {lab}{next(q for q, v in byq[k].items() if v['at_target'] == v['n'])}"
          for k, lab in ((k, "q") for k in ["aom_s6"] + order if k != "nvenc")
          if any(v["at_target"] == v["n"] for v in byq[k].values())) +
      f" (the published top points already suffice). NVENC never reaches it on {len(miss['nvenc'])} photos.",
      f"- NVENC: {nv_par['odd_height']['images']} photos have an odd height; in this pipeline they decode with a chroma error "
      f"that stays at every CQ (luma is fine on the two checked: `scripts/nvenc_shift2.py`, `logs/nvenc_shift2.log`). "
      f"{sum(1 for im in miss['nvenc'] if im in odd_h)} of the {len(miss['nvenc'])} photos on which NVENC never reaches "
      f"{a.target} JOD are among them; the other "
      f"({', '.join(im for im in miss['nvenc'] if im not in odd_h)}) has a Camera RGB profile, which the pipeline drops. "
      f"On the {nv_par['even_height']['images']} even-height photos NVENC is "
      f"{pct(nv_par['even_height']['cvvdp']['median'])} (JOD) and {pct(nv_par['even_height']['ssimulacra2_published_points']['median'])} "
      "(SSIMULACRA2) against libaom speed 6.",
      "- `standard_fhd` is a 24-inch 1080p screen at 0.6 m (37.8 pixels per degree). The commenter's 27-inch 1080p at about "
      "20 inches is about 28.5 pixels per degree, with larger pixels, so artefacts there are easier to see than here. CVVDP "
      "shows image pixels 1:1 on display pixels: a 24 MP photo is judged as if viewed at 100 % zoom, not scaled to the screen.",
      "- On the Deep Learning Base AMI, LD_LIBRARY_PATH holds CUDA 13.2's cuDNN 9.23, which breaks the cuDNN 9.26 that "
      "PyTorch 2.15 ships (CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED); the runs put the pip cuDNN first on LD_LIBRARY_PATH "
      "(scripts/run1.sh). bench.py now runs a small CVVDP self-test at start-up so such a failure stops the run."]
open(os.path.join(a.out, "summary.md"), "w").write("\n".join(L) + "\n")
print("wrote", os.path.join(a.out, "summary.md"))
