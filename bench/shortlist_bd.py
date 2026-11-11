#!/usr/bin/env python3
"""--uv-shortlist / --tx-shortlist (and --uvK 1) against the default: speed, BD-rate, bytes at fixed quality.

  bench/shortlist_bd.py --av1gpu build/av1gpu [--ref RELEASE_BINARY] --images bench/images_gamut --out OUT \\
      [--common "--preset turbo2"] [--avifdec-codec dav1d]

Variants (--variants NAME=OPTS,... to change them; OPTS space-separated; the first must be off=): off (no option; run
with --ref when given, e.g. the release, else with --av1gpu), uv3 (--uv-shortlist 3), tx4 (--tx-shortlist 4), uv3_tx4
(both), uvk1 (--uvK 1, an existing option, for comparison).  --common adds options to every run of every variant: the
switches behave differently per preset (--uv-shortlist replaces --uvK, so with turbo2 / turbo3's --uvK 1 only N = 1
saves time; --tx-shortlist does nothing in turbo2, turbo3 and slow), so time each preset you care about.
- Speed (the headline): the set repeated to --min-mp megapixels; one warm-up run of each variant, then --rounds rounds
  (default 6), each running every variant with GP_ENCODERS=3 and =1, in forward order on even rounds and reverse order
  (encoder counts reversed too) on odd ones, so every pair is timed in both orders.  Reported: median MP/s per variant
  and encoder count, its change against off as the median of the per-round ratios, CPU cores, and recon / dp ms from
  --stats.  Judge by MP/s: the shortlists' open-loop pass (k_leafprep) is booked in dp_ms, so recon ms alone overstates.
- Quality (bench.py's method): every variant encodes every image at bench.py's quality points (--quality, one av1gpu
  --batch run with every --qindex); outputs are decoded with avifdec and scored against the source JPEG with
  SSIMULACRA2.  Per image the BD-rate of each variant against off (bytes at equal SSIMULACRA2, bench.bd_rate);
  reported: median and mean over images, by size class too, and the total bytes and mean SSIMULACRA2 at each quality
  point against off (at quality 60, the default, that is "bytes at fixed quality").
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


def pct(x):
    return "n/a" if x is None else f"{100 * x:+.2f} %"


def fmean_or_none(v):
    v = [x for x in v if x is not None]
    return statistics.fmean(v) if v else None


def quality_phase(a, files, variants, work):
    rows = {}
    qi = {q: bench.qindex_for_quality(q) for q in a.qualities}
    for name, binary, opts in variants:
        out = os.path.join(work, "q_" + name)
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out)
        lst = os.path.join(out, "inputs.txt")
        open(lst, "w").write("\n".join(files) + "\n")
        p = bench.run([binary, "--batch", lst, "--outdir", out, "--qindex", ",".join(str(v) for v in qi.values()), "--name-q"]
                      + bench.gpu_format(a) + a.common_opts + opts)
        if p.returncode:
            sys.exit(f"av1gpu {name} failed:\n{p.stderr[-3000:]}")
        outs = [(src, q, os.path.join(out, f"{os.path.basename(src)}.q{v}.avif")) for src in files for q, v in qi.items()]
        outs = [o for o in outs if os.path.exists(o[2])]
        with cf.ThreadPoolExecutor(a.jobs) as ex:
            scores = list(ex.map(lambda t: bench.score(a, t[0], t[2], work), outs))
        rows[name] = [dict(image=os.path.basename(src), q=q, bytes=os.path.getsize(dst), ssimulacra2=s)
                      for (src, q, dst), s in zip(outs, scores)]
        print(f"[quality] {name}: {len(outs)} files, {sum(r['bytes'] for r in rows[name])} bytes, "
              f"{sum(r['ssimulacra2'] is None for r in rows[name])} unscored", flush=True)
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
        d = dict(n=len(vals), median=statistics.median(vals.values()) if vals else None,
                 mean=statistics.fmean(vals.values()) if vals else None, by_size={}, by_q={})
        for cls, lo, hi in bench.SIZE_CLASSES:
            v = [x for im, x in vals.items() if lo <= sizes[im] < hi]
            if v:
                d["by_size"][cls] = dict(n=len(v), median=statistics.median(v), mean=statistics.fmean(v))
        for q in sorted({r["q"] for r in rows[n]}):
            b = sum(r["bytes"] for r in rows[n] if r["q"] == q)
            b0 = sum(r["bytes"] for r in rows["off"] if r["q"] == q)
            d["by_q"][q] = dict(bytes=b, bytes_off=b0, bytes_change=b / b0 - 1 if b0 else None,
                                ssimulacra2=fmean_or_none(r["ssimulacra2"] for r in rows[n] if r["q"] == q),
                                ssimulacra2_off=fmean_or_none(r["ssimulacra2"] for r in rows["off"] if r["q"] == q))
        res[n] = d
    return res


def speed_phase(a, files, sizes, variants, work):
    d, lst, mp, rep = bench.speed_set(files, sizes, a.min_mp, work)
    out = os.path.join(work, "speed_out")
    runs = {(n, e): {} for n, _, _ in variants for e in (3, 1)}

    def one(name, binary, opts, enc, rnd):
        shutil.rmtree(out, ignore_errors=True)
        st = stats_line(run_env([binary, "--batch", d, "--outdir", out, "--stats"] + bench.gpu_format(a) + a.common_opts + opts,
                                {"GP_ENCODERS": str(enc)}))
        if rnd >= 0:
            runs[(name, enc)][rnd] = st
        print(f"[speed] {'warm-up' if rnd < 0 else f'round {rnd}'} {name} enc {enc}: {st['mp_s']:.1f} MP/s", flush=True)

    for name, binary, opts in variants:
        one(name, binary, opts, 3, -1)
    for rnd in range(a.rounds):
        fwd = rnd % 2 == 0
        for name, binary, opts in (variants if fwd else list(reversed(variants))):
            for enc in ((3, 1) if fwd else (1, 3)):
                one(name, binary, opts, enc, rnd)
    res = {}
    for (name, enc), rs in runs.items():
        med = lambda k: statistics.median(r[k] for r in rs.values())
        off = runs[("off", enc)]
        ratio = lambda k: statistics.median(rs[r][k] / off[r][k] for r in rs if r in off and off[r][k]) - 1
        res[f"{name}_enc{enc}"] = dict(mp=next(iter(rs.values()))["mp"], mp_s=med("mp_s"), mp_s_runs=[rs[r]["mp_s"] for r in sorted(rs)],
                                       mp_s_change=ratio("mp_s"), cpu_cores=med("cpu_cores"), recon_ms=med("recon_ms"),
                                       dp_ms=med("dp_ms"), analysis_ms=med("analysis_ms"), bytes=next(iter(rs.values()))["bytes"])
    return dict(set_mp=mp, repeats=rep, rounds=a.rounds, runs=res)


def prof_phase(a, files, variants, work):
    lst = os.path.join(work, "prof.txt")
    open(lst, "w").write("\n".join(files) + "\n")
    res = {}
    for name, binary, opts in variants:
        out = os.path.join(work, "prof_out")
        shutil.rmtree(out, ignore_errors=True)
        p = run_env([binary, "--batch", lst, "--outdir", out, "--stats", "--prof"] + bench.gpu_format(a) + a.common_opts + opts,
                    {"GP_ENCODERS": "1"})
        st = stats_line(p)
        ph = re.search(r"recon phases: (.*)", p.stderr)
        cyc = re.search(r"recon CTA-cycles ([0-9.e+]+)", p.stderr)
        res[name] = dict(phases=ph.group(1).strip() if ph else "", cta_cycles=float(cyc.group(1)) if cyc else None,
                         recon_ms=st["recon_ms"], dp_ms=st["dp_ms"])
        print(f"[prof] {name}: CTA-cycles {res[name]['cta_cycles']}, {res[name]['phases']}", flush=True)
    for name in res:
        c, c0 = res[name]["cta_cycles"], res["off"]["cta_cycles"]
        res[name]["cta_cycles_change"] = c / c0 - 1 if c and c0 else None
    return res


def write_summary(res, path):
    common = " ".join(res["common"]) or "(none)"
    L = [f"# Shortlists against off ({res['gpu']})", "",
         f"av1gpu: `{res['av1gpu']}`; off: `{res['ref']}`; options on every run: `{common}`; images: {res['n_images']} "
         f"({res['mp']:.0f} MP)", ""]
    s = res.get("speed")
    if s:
        L += [f"## Speed (set repeated to {s['set_mp']:.0f} MP; median of {s['rounds']} rounds in alternating order; "
              f"change = median of the per-round ratios to off)", "",
              "| Variant | Options | MP/s 3 enc | vs off | MP/s 1 enc | vs off | CPU cores 3 / 1 enc | recon + dp ms 1 enc |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
        for n, opts in res["variants"].items():
            r3, r1 = s["runs"][f"{n}_enc3"], s["runs"][f"{n}_enc1"]
            L.append(f"| {n} | `{' '.join(opts)}` | {r3['mp_s']:.1f} | {pct(r3['mp_s_change'])} | {r1['mp_s']:.1f} | "
                     f"{pct(r1['mp_s_change'])} | {r3['cpu_cores']:.2f} / {r1['cpu_cores']:.2f} | "
                     f"{r1['recon_ms']:.0f} + {r1['dp_ms']:.0f} |")
        L += ["", "(dp ms includes the shortlists' open-loop pass, k_leafprep.)", ""]
    q = res.get("quality")
    if q:
        L += ["## Compression (BD-rate on SSIMULACRA2 against off, bench.py's quality points; negative = smaller)", "",
              "| Variant | median | mean | n | bytes at q60 vs off | SSIMULACRA2 at q60 (off) |", "|---|---:|---:|---:|---:|---:|"]
        for n, d in q.items():
            b = d["by_q"].get(60) or {}
            ss = lambda x: "n/a" if x is None else f"{x:.2f}"
            L.append(f"| {n} | {pct(d['median'])} | {pct(d['mean'])} | {d['n']} | {pct(b.get('bytes_change'))} | "
                     f"{ss(b.get('ssimulacra2'))} ({ss(b.get('ssimulacra2_off'))}) |")
        L += [""]
    p = res.get("prof")
    if p:
        L += ["## k_recon (GP_ENCODERS=1 --prof)", "", "| Variant | CTA-cycles | vs off | phases |", "|---|---:|---:|---|"]
        for n, d in p.items():
            cyc = "n/a" if d["cta_cycles"] is None else f"{d['cta_cycles']:.4g}"
            L.append(f"| {n} | {cyc} | {pct(d['cta_cycles_change'])} | {d['phases']} |")
        L += [""]
    open(path, "w").write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", default=os.path.join(bench.HERE, "images_gamut"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--av1gpu", default=os.path.join(bench.ROOT, "build", "av1gpu"))
    ap.add_argument("--ref", default="", help="binary for the off variant (e.g. the release); default: --av1gpu")
    ap.add_argument("--common", default="", help="options for every run, e.g. \"--preset turbo2\"")
    ap.add_argument("--avifdec", default="avifdec")
    ap.add_argument("--avifdec-codec", default="")
    ap.add_argument("--ssimulacra2", default="ssimulacra2")
    ap.add_argument("--quality", default="30,40,50,60,70,80")
    ap.add_argument("--variants", default="", help="NAME=OPTS,... (the first must be off=)")
    ap.add_argument("--yuv", choices=("420", "444"), default="420")
    ap.add_argument("--depth", type=int, choices=(8, 10), default=8)
    ap.add_argument("--min-mp", type=float, default=1000)
    ap.add_argument("--rounds", type=int, default=6, help="measured speed rounds (an even number balances the order)")
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--skip-prof", action="store_true")
    ap.add_argument("--keep", action="store_true")
    a = ap.parse_args()
    a.qualities = [int(x) for x in a.quality.split(",")]
    a.common_opts = a.common.split()
    vs = [(v.split("=", 1)[0], v.split("=", 1)[1].split()) for v in a.variants.split(",")] if a.variants else VARIANTS
    if vs[0][0] != "off":
        sys.exit("the first variant must be off")
    variants = [(n, (a.ref or a.av1gpu) if n == "off" else a.av1gpu, o) for n, o in vs]
    files = sorted(f for f in glob.glob(os.path.join(a.images, "*")) if f.lower().endswith(bench.EXTS))
    from PIL import Image
    sizes = {}
    for f in files:
        with Image.open(f) as im:
            sizes[os.path.basename(f)] = im.size[0] * im.size[1] / 1e6
    os.makedirs(a.out, exist_ok=True)
    work = tempfile.mkdtemp(prefix="av1gpu-shortlist-")
    res = dict(av1gpu=a.av1gpu, ref=a.ref or a.av1gpu, common=a.common_opts,
               gpu=bench.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).stdout.strip(),
               n_images=len(files), mp=sum(sizes.values()), variants={n: o for n, _, o in variants}, format=f"{a.yuv} {a.depth}-bit")
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
