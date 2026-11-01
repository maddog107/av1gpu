# exec'd by analyze4x4.py --write DIR (uses its A, O, N, NO, RUN): bench/results/aws-g6-l4-tiles4x4/{results.json,summary.md}
D = arg("--write")
os.makedirs(D, exist_ok=True)
chk = json.load(open(os.path.join(RUN, "checks.json")))
cmpj = json.load(open(os.path.join(RUN, "compare.json")))["compare"]
env = open(os.path.join(RUN, "env.txt")).read()
cvinfo = json.load(open(os.path.join(RUN, "cvvdp_info.json")))
alpha = json.load(open(os.path.join(RUN, "alpha.json"))) if os.path.exists(os.path.join(RUN, "alpha.json")) else None
gates = {}
for g in sorted(os.listdir(os.path.join(RUN, "gates"))) if os.path.isdir(os.path.join(RUN, "gates")) else []:
    lines = [x for x in open(os.path.join(RUN, "gates", g)).read().splitlines() if x.startswith("[gate]")]
    gates[g[:-4]] = lines
strip = lambda rows: [{k: r[k] for k in ("image", "q", "bytes", "ssimulacra2", "cvvdp") if k in r} for r in rows]  # noqa: E731
res = dict(
    description="av1gpu's side of the L4 compression results, re-encoded with the 4x4-tile default (perf/tiles-4x4: turbo, "
                "turbo2 and turbo3 use 4x4 tiles for images of 2 MP and more, instead of 2x2 / 4x2) and re-scored on an AWS "
                "g6.4xlarge (NVIDIA L4), with v0.3.0 (2x2) encoded alongside for the byte checks. The CPU baselines "
                "(libaom, SVT-AV1, WebP, NVENC) are the stored ones in bench/results/; below 2 MP the files are "
                "byte-identical to v0.3.0's, so those rows equal the published ones.",
    date="2026-10-31",
    code=dict(new="50b62cb (perf/tiles-4x4)", old="f8f1ecc (v0.3.0)"),
    environment=env.strip().splitlines(),
    cvvdp=cvinfo,
    images=100, megapixels=sum(SIZES.values()),
    baselines=dict(size_range="aws-g6-l4-gamut (libaom speed 6 / 9, WebP with aws-g6-l4-cvvdp's ICC re-scores)",
                   cvvdp="aws-g6-l4-cvvdp (99 photos, 8 quality points)", formats="aws-g6-l4-formats",
                   one_tile_anchor="aws-c6a-cpu-tuning (stock_aom_s6_notiles)", nvenc="aws-g6-l4-nvenc",
                   svt="aws-c6a-gamut-svt"),
    quality={"av1gpu": strip(A["default6"]), "av1gpu_nosmall": strip(A["nosmall6"]), "av1gpu_turbo3_nosmall": strip(A["turbo3ns"]),
             "av1gpu_turbo3": strip(A["turbo3"]), "av1gpu_8pts": strip(A["default8"]),
             **{f"av1gpu_{f}": strip(A["fmt"][f]) for f in FMTS[1:]}},
    quality_v030_on_l4={"av1gpu_turbo3_nosmall": strip(O["turbo3ns"]), "av1gpu_turbo3": strip(O["turbo3"])},
    cvvdp_luma_only=[{k: r[k] for k in ("image", "q", "bytes", "cvvdp_luma")} for r in A["luma"]],
    bd_rate={"av1gpu_vs_aom_s6": N["default"], "av1gpu_vs_webp": N["default_vs_webp"],
             "av1gpu_nosmall_vs_aom_s6": N["nosmall"], "av1gpu_turbo3_nosmall_vs_aom_s6": N["turbo3ns"],
             "av1gpu_turbo3_vs_aom_s6": N["turbo3"], "av1gpu_vs_aom_s6_notiles": N["default_notiles"],
             "av1gpu_nosmall_vs_aom_s6_notiles": N["nosmall_notiles"], "av1gpu_vs_nvenc": N["vs_nvenc"],
             "av1gpu_vs_nvenc_even_height": N["vs_nvenc_even"], **N["formats"]},
    bd_rate_by_size={"av1gpu_vs_aom_s6": N["default_by_size"], "av1gpu_nosmall_vs_aom_s6": N["nosmall_by_size"]},
    bd_rate_cvvdp={"av1gpu_vs_aom_s6": N["cvvdp"]["bd"], "av1gpu_vs_aom_s6_published_points": N["cvvdp"]["bd_published_points"],
                   "av1gpu_vs_aom_s6_luma_only": N["cvvdp"]["luma_only"]},
    bd_rate_cvvdp_by_size={"av1gpu_vs_aom_s6": N["cvvdp"]["by_size"]},
    cvvdp_target={k: v for k, v in N["cvvdp"]["target"].items()},
    av1gpu_per_image_agreement=dict(images=N["cvvdp"]["images"], jod_worse_than_ssimulacra2=N["cvvdp"]["jod_worse_than_ssimulacra2"]),
    totals=N["totals"],
    v030_on_l4={"bd_rate": {"av1gpu_vs_aom_s6": NO["default"], "av1gpu_nosmall_vs_aom_s6": NO["nosmall"],
                            "av1gpu_turbo3_nosmall_vs_aom_s6": NO["turbo3ns"], "av1gpu_turbo3_vs_aom_s6": NO["turbo3"],
                            **NO["formats"]},
                "bd_rate_cvvdp": {"av1gpu_vs_aom_s6": NO["cvvdp"]["bd"]}, "totals": NO["totals"]},
    byte_identity_vs_v030=cmpj, checks=chk, decoder_gate=gates, alpha=alpha)
json.dump(res, open(os.path.join(D, "results.json"), "w"), indent=1)

p = lambda v: f"{100 * v['median']:+.1f} %".replace("-", "−")  # noqa: E731
pm = lambda v: f"{100 * v['median']:+.1f} % ({100 * v['mean']:+.1f} %)".replace("-", "−")  # noqa: E731
lines = ["# 4x4-tile default: av1gpu re-encoded (NVIDIA L4)", "",
         "av1gpu 50b62cb (perf/tiles-4x4: 4x4 tiles for images of 2 MP and more in every turbo preset) against the "
         "stored CPU baselines, 100 size-range photos unless noted. Median BD-rate on SSIMULACRA2 (mean in brackets); "
         "v0.3.0 (2x2 tiles; turbo3 4x2) for comparison.", "",
         "| av1gpu | 4x4 (new default) | v0.3.0 |", "|---|---:|---:|",
         f"| default vs libaom speed 6 | {pm(N['default'])} | {pm(NO['default'])} |",
         f"| `--small-preset none` | {pm(N['nosmall'])} | {pm(NO['nosmall'])} |",
         f"| `--preset turbo3 --small-preset none` | {pm(N['turbo3ns'])} | {pm(NO['turbo3ns'])} |",
         f"| `--preset turbo3` | {pm(N['turbo3'])} | {pm(NO['turbo3'])} |",
         f"| default vs one-tile libaom speed 6 (99 photos) | {pm(N['default_notiles'])} | {pm(NO['default_notiles'])} |",
         f"| default vs WebP (ICC re-scored) | {pm(N['default_vs_webp'])} | {pm(NO['default_vs_webp'])} |",
         f"| default vs NVENC AV1 ({N['vs_nvenc']['n']} photos) | {pm(N['vs_nvenc'])} | {pm(NO['vs_nvenc'])} |",
         f"| CVVDP (JOD), default vs libaom speed 6 (99 photos, 8 points) | {pm(N['cvvdp']['bd'])} | {pm(NO['cvvdp']['bd'])} |"]
for f in FMTS:
    k = f"av1gpu_{f}_vs_aom_s6_{f}"
    lines.append(f"| {f.replace('_', ' / ')}-bit vs libaom speed 6 in the same format | {pm(N['formats'][k])} | {pm(NO['formats'][k])} |")
lines += ["", "Byte identity with v0.3.0 (files below / at or above 2 MP):", ""]
for k, v in cmpj.items():
    lines.append(f"- {k}: below 2 MP {v['lt2']['identical']}/{v['lt2']['files']} identical; 2 MP and more "
                 f"{v['ge2']['identical']}/{v['ge2']['files']} identical, {v['ge2']['bytes_new'] - v['ge2']['bytes_old']:+,} bytes")
lines += ["", "Decoder gates (NEW, `tests/decoder_gate.py`, 100 size-range photos):", ""]
lines += [f"- {k}: {x}" for k, v in gates.items() for x in v]
lines += [""]
open(os.path.join(D, "summary.md"), "w").write("\n".join(lines) + "\n")
print("wrote", D)
