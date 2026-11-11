#!/usr/bin/env python3
"""--uv-shortlist / --tx-shortlist (and --uvK 1) against the default: BD-rate, bytes at fixed quality, speed.

  bench/shortlist_bd.py --av1gpu build/av1gpu --images bench/images_gamut --out OUT [--avifdec-codec dav1d]

Variants (--variants NAME=OPTS,... to change them; OPTS space-separated): off (no option), uv3 (--uv-shortlist 3),
tx4 (--tx-shortlist 4), uv3_tx4 (both), uvk1 (--uvK 1, an existing option, for comparison).
- Quality (bench.py's method): every variant encodes every image at bench.py's quality points (--quality, one
  av1gpu --batch run with every --qindex); outputs are decoded with avifdec and scored against the source JPEG with
  SSIMULACRA2.  Per image the BD-rate of each variant against off (bytes at equal SSIMULACRA2, bench.bd_rate);
  reported: median and mean over images, by size class too.  Also the total bytes and the mean SSIMULACRA2 at each
  quality point against off (at quality 60, the default, that is "bytes at fixed quality").
- Speed: the set repeated to --min-mp megapixels; --rounds rounds, each running every variant once with
  GP_ENCODERS=3 (the default) and once with GP_ENCODERS=1, the variant order alternating between rounds, after one
  warm-up run of each; reported: the median MP/s, CPU cores, recon and dp ms (--stats) per variant and encoder count,
  and the change against off.  k_leafprep (the shortlists' open-loop pass) runs inside the stats' dp_ms.
- Recon profile (one run per variant, GP_ENCODERS=1 --prof): k_recon CTA-cycles and phase shares.
Writes OUT/results.json and OUT/summary.md.  --skip-quality / --skip-speed / --skip-prof leave a phase out.
Requires avifdec and ssimulacra2 (see bench.py), numpy, Pillow.
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import re
import shutil
import statistics
import sys
import tempfile

import bench

VARIANTS = [("off", []), ("uv3", ["--uv-shortlist", "3"]), ("tx4", ["--tx-shortlist", "4"]),
            ("uv3_tx4", ["--uv-shortlist", "3", "--tx-shortlist", "4"]), ("uvk1", ["--uvK", "1"])]


def stats_line(p):
    for line in reversed(p.stdout.strip().splitlines()):
        if line.startswith("{"):
            return json.loads(line)
    sys.exit(f"av1gpu failed:\n{p.stderr[-3000:]}")


def run_env(cmd, env):
    e = dict(os.environ)
    e.update(env)
    return bench.run(cmd, env=e)


def quality_phase(a, files, variants, work):
    rows = {}
    qi = {q: bench.qindex_for_quality(q) for q in a.qualities}
    for name, opts in variants:
        out = os.path.join(work, "q_" + name)
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out)
        lst = os.path.join(out, "inputs.txt")
        open(lst, "w").write("\n".join(files) + "\n")
        p = bench.run([a.av1gpu, "--batch", lst, "--outdir", out, "--qindex", ",".join(str(v) for v in qi.values()), "--name-q"]
                      + bench.gpu_format(a) + opts)
        if p.returncode:
            sys.exit(f"av1gpu {name} failed:\n{p.stderr[-3000:]}")
        outs = [(src, q, os.path.join(out, f"{os.path.basename(src)}.q{v}.avif")) for src in files for q, v in qi.items()]
        outs = [o for o in outs if os.path.exists(o[2])]
        with cf.ThreadPoolExecutor(a.jobs) as ex:
            scores = list(ex.map(lambda t: bench.score(a, t[0], t[2], work), outs))
        rows[name] = [dict(image=os.path.basename(src), q=q, bytes=os.path.getsize(dst), ssimulacra2=s)
                      for (src, q, dst), s in zip(outs, scores)]
        print(f"[quality] {name}: {len(outs)} files, {sum(r['bytes'] for r in rows[name])} bytes", flush=True)
        if not a.keep:
            shutil.rmtree(out, ignore_errors=True)
    return rows


def quality_summary(rows, sizes):
    per = {n: {} for n in rows}
    for n, rs in rows.items():
        for r in rs:
            per[n].setdefault(r["image"], []).append((r["bytes"], r["ssimulacra2"]))
    res = {}
    for n in rows:
        if n == "off":
            continue
        vals = {im: bench.bd_rate(per[n][im], per["off"][im]) for im in per[n] if im in per["off"]}
        vals = {im: v for im, v in vals.items() if v is not None}
        d = dict(n=len(vals), median=statistics.median(vals.values()), mean=statistics.fmean(vals.values())) if vals else {}
        d["by_size"] = {}
        for cls, lo, hi in bench.SIZE_CLASSES:
            v = [x for im, x in vals.items() if lo <= sizes[im] < hi]
            if v:
                d["by_size"][cls] = dict(n=len(v), median=statistics.median(v), mean=statistics.fmean(v))
        d["by_q"] = {}
        for q in sorted({r["q"] for r in rows[n]}):
            b = sum(r["bytes"] for r in rows[n] if r["q"] == q)
            b0 = sum(r["bytes"] for r in rows["off"] if r["q"] == q)
            s = [r["ssimulacra2"] for r in rows[n] if r["q"] == q and r["ssimulacra2"] is not None]
            s0 = [r["ssimulacra2"] for r in rows["off"] if r["q"] == q and r["ssimulacra2"] is not None]
            d["by_q"][q] = dict(bytes=b, bytes_off=b0, bytes_change=b / b0 - 1, ssimulacra2=statistics.fmean(s),
                                ssimulacra2_off=statistics.fmean(s0))
        res[n] = d
    return res


def speed_phase(a, files, sizes, variants, work):
    d, lst, mp, rep = bench.speed_set(files, sizes, a.min_mp, work)
    out = os.path.join(work, "speed_out")
    runs = {(n, e): [] for n, _ in variants for e in (3, 1)}
    for rnd in range(a.rounds + 1):
        order = variants if rnd % 2 == 0 else list(reversed(variants))
        for name, opts in order:
            for enc in (3, 1):
                shutil.rmtree(out, ignore_errors=True)
                p = run_env([a.av1gpu, "--batch", d, "--outdir", out, "--stats"] + bench.gpu_format(a) + opts, {"GP_ENCODERS": str(enc)})
                st = stats_line(p)
                if rnd:   # round 0: warm-up
                    runs[(name, enc)].append(st)
                print(f"[speed] round {rnd} {name} enc {enc}: {st['mp_s']:.1f} MP/s{' (warm-up)' if not rnd else ''}", flush=True)
    res = {}
    for (name, enc), rs in runs.items():
        med = lambda k: statistics.median(r[k] for r in rs)
        res[f"{name}_enc{enc}"] = dict(mp=rs[0]["mp"], mp_s=med("mp_s"), mp_s_runs=[r["mp_s"] for r in rs], cpu_cores=med("cpu_cores"),
                                       recon_ms=med("recon_ms"), dp_ms=med("dp_ms"), analysis_ms=med("analysis_ms"), bytes=rs[0]["bytes"])
    for name, _ in variants:
        for enc in (3, 1):
            r, o = res[f"{name}_enc{enc}"], res[f"off_enc{enc}"]
            r["mp_s_change"] = r["mp_s"] / o["mp_s"] - 1
            r["recon_ms_change"] = r["recon_ms"] / o["recon_ms"] - 1
    return dict(set_mp=mp, repeats=rep, rounds=a.rounds, runs=res)


def prof_phase(a, files, variants, work):
    lst = os.path.join(work, "prof.txt")
    open(lst, "w").write("\n".join(files) + "\n")
    res = {}
    for name, opts in variants:
        out = os.path.join(work, "prof_out")
        shutil.rmtree(out, ignore_errors=True)
        p = run_env([a.av1gpu, "--batch", lst, "--outdir", out, "--stats", "--prof"] + bench.gpu_format(a) + opts, {"GP_ENCODERS": "1"})
        st = stats_line(p)
        ph = re.search(r"recon phases: (.*)", p.stderr)
        cyc = re.search(r"recon CTA-cycles ([0-9.e+]+)", p.stderr)
        res[name] = dict(phases=ph.group(1).strip() if ph else "", cta_cycles=float(cyc.group(1)) if cyc else None,
                         recon_ms=st["recon_ms"], dp_ms=st["dp_ms"])
        print(f"[prof] {name}: CTA-cycles {res[name]['cta_cycles']}, {res[name]['phases']}", flush=True)
    for name in res:
        if res[name]["cta_cycles"] and res["off"]["cta_cycles"]:
            res[name]["cta_cycles_change"] = res[name]["cta_cycles"] / res["off"]["cta_cycles"] - 1
    return res


def write_summary(res, path):
    L = [f"# Shortlists against the default ({res['gpu']})", "", f"av1gpu: `{res['av1gpu']}`, images: {res['n_images']} ({res['mp']:.0f} MP)", ""]
    q = res.get("quality")
    if q:
        L += ["## Compression (BD-rate on SSIMULACRA2 against off, bench.py's quality points; negative = smaller)", "",
              "| Variant | Options | median | mean | n | bytes at q60 vs off | SSIMULACRA2 at q60 (off) |", "|---|---|---:|---:|---:|---:|---:|"]
        for n, d in q.items():
            b = d["by_q"].get(60) or d["by_q"].get("60") or {}
            L.append(f"| {n} | `{' '.join(res['variants'][n])}` | {100 * d.get('median', 0):+.2f} % | {100 * d.get('mean', 0):+.2f} % | {d.get('n', 0)} | "
                     f"{100 * b.get('bytes_change', 0):+.2f} % | {b.get('ssimulacra2', 0):.2f} ({b.get('ssimulacra2_off', 0):.2f}) |")
        L += [""]
    s = res.get("speed")
    if s:
        L += [f"## Speed (set repeated to {s['set_mp']:.0f} MP, median of {s['rounds']} rounds)", "",
              "| Variant | MP/s 3 enc | vs off | MP/s 1 enc | vs off | recon ms 1 enc | vs off | dp ms 1 enc | CPU cores 3 enc |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for n in res["variants"]:
            r3, r1 = s["runs"][f"{n}_enc3"], s["runs"][f"{n}_enc1"]
            L.append(f"| {n} | {r3['mp_s']:.1f} | {100 * r3['mp_s_change']:+.1f} % | {r1['mp_s']:.1f} | {100 * r1['mp_s_change']:+.1f} % | "
                     f"{r1['recon_ms']:.0f} | {100 * r1['recon_ms_change']:+.1f} % | {r1['dp_ms']:.1f} | {r3['cpu_cores']:.2f} |")
        L += [""]
    p = res.get("prof")
    if p:
        L += ["## k_recon (GP_ENCODERS=1 --prof)", "", "| Variant | CTA-cycles | vs off | phases |", "|---|---:|---:|---|"]
        for n, d in p.items():
            L.append(f"| {n} | {d['cta_cycles']:.4g} | {100 * d.get('cta_cycles_change', 0):+.1f} % | {d['phases']} |")
        L += [""]
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=os.path.join(bench.HERE, "images_gamut"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--av1gpu", default=os.path.join(bench.ROOT, "build", "av1gpu"))
    ap.add_argument("--avifdec", default="avifdec")
    ap.add_argument("--avifdec-codec", default="")
    ap.add_argument("--ssimulacra2", default="ssimulacra2")
    ap.add_argument("--quality", default="30,40,50,60,70,80")
    ap.add_argument("--variants", default="", help="NAME=OPTS,... (first must be off=)")
    ap.add_argument("--yuv", choices=("420", "444"), default="420")
    ap.add_argument("--depth", type=int, choices=(8, 10), default=8)
    ap.add_argument("--min-mp", type=float, default=1000)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--skip-prof", action="store_true")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    a.qualities = [int(x) for x in a.quality.split(",")]
    variants = [(v.split("=", 1)[0], v.split("=", 1)[1].split()) for v in a.variants.split(",")] if a.variants else VARIANTS
    if variants[0][0] != "off":
        sys.exit("the first variant must be off")
    files = sorted(f for f in glob.glob(os.path.join(a.images, "*")) if f.lower().endswith(bench.EXTS))
    from PIL import Image
    sizes = {os.path.basename(f): Image.open(f).size[0] * Image.open(f).size[1] / 1e6 for f in files}
    os.makedirs(a.out, exist_ok=True)
    work = tempfile.mkdtemp(prefix="av1gpu-shortlist-")
    res = dict(av1gpu=a.av1gpu, gpu=bench.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).stdout.strip(),
               n_images=len(files), mp=sum(sizes.values()), variants={n: o for n, o in variants}, format=f"{a.yuv} {a.depth}-bit")
    if not a.skip_prof:
        res["prof"] = prof_phase(a, files, variants, work)
    if not a.skip_speed:
        res["speed"] = speed_phase(a, files, sizes, variants, work)
    if not a.skip_quality:
        rows = quality_phase(a, files, variants, work)
        res["quality_rows"] = rows
        res["quality"] = quality_summary(rows, sizes)
    json.dump(res, open(os.path.join(a.out, "results.json"), "w"), indent=1)
    write_summary(res, os.path.join(a.out, "summary.md"))
    print(open(os.path.join(a.out, "summary.md")).read())
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
