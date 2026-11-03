#!/usr/bin/env python3
"""Tables of the palette study from the scored rows (study.py score).

  analyze.py ROWS_DIR OUT_DIR       (ROWS_DIR: scored_lossy.jsonl, scored_ll.jsonl, sizes.json)

BD-rates use bench/bench.py's bd_rate (per image, piecewise-cubic log-size against the score over the
overlapping quality range, at least 2 SSIMULACRA2 points or 0.1 butteraugli 3-norm), then the median (and mean)
over the images of a group.  Negative = smaller files at equal score.  Writes OUT_DIR/tables.md and results.json."""
import collections
import gzip
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "..", "bench"))
sys.path.insert(0, os.path.join(HERE, "..", "..", ".."))
from bench import bd_rate  # noqa: E402


def _open(p):
    """a rows file, or its .gz"""
    if not os.path.exists(p) and os.path.exists(p + ".gz"):
        return gzip.open(p + ".gz", "rt")
    return open(p)


GROUPS = [  # (label, set, image filter)
    ("comics (scanned pages, JPEG)", "comics", None),
    ("flat-colour art, PNG", "flat_png", None),
    ("flat-colour art, JPEG q90", "flat_jpg", None),
    ("painted webcomic (Pepper&Carrot), JPEG", "painted", None),
    ("screen: all", "screen", None),
    ("screen: 23 graphics of the lossless comparison", "screen", lambda f: f.startswith("screen_synth_")),
    ("screen: real screenshots", "screen", lambda f: not f.startswith("screen_synth_") and not f.startswith("screen_wikipedia_")),
    ("screen: Wikipedia page renders", "screen", lambda f: f.startswith("screen_wikipedia_")),
    ("photos: all", "photo", None),
    ("photos: Kodak (PNG)", "photo", lambda f: f.startswith("photo_kodim")),
    ("photos: size-range JPEGs", "photo", lambda f: f.endswith(".jpg")),
]
LL_GROUPS = [g for g in GROUPS if g[1] in ("flat_png", "screen", "photo") and g[0] != "photos: size-range JPEGs"]


def codec(r):
    if r["enc"] == "gpu":
        return f"gpu{r['fmt'][:3]}"
    return f"aom{r['fmt'][:3]}_{r['cfg']}"


def curves(rows, metric):
    per = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in rows:
        if r.get("missing") or r.get(metric) is None:
            continue
        v = r[metric] if metric == "ssimulacra2" else -r[metric]
        per[codec(r)][(r["set"], r["image"])].append((r["bytes"], v))
    return per


def bd_group(per, test, anchor, sel, metric):
    ov = 2.0 if metric == "ssimulacra2" else 0.1
    vals = []
    for k in sel:
        if k in per[test] and k in per[anchor]:
            v = bd_rate(per[test][k], per[anchor][k], ov)
            if v is not None:
                vals.append(v)
    if not vals:
        return None
    return dict(n=len(vals), of=len(sel), median=statistics.median(vals), mean=statistics.fmean(vals))


def fmt_bd(d):
    if not d:
        return "–"
    s = f"{100 * d['median']:+.1f} % ({100 * d['mean']:+.1f})"
    return s + (f" [{d['n']}/{d['of']}]" if d["n"] < d["of"] else "")


def main():
    rows_dir, out = sys.argv[1], sys.argv[2]
    lossy = [json.loads(l) for l in _open(os.path.join(rows_dir, "scored_lossy.jsonl"))]
    ll = [json.loads(l) for l in _open(os.path.join(rows_dir, "scored_ll.jsonl"))]
    sizes = json.load(open(os.path.join(rows_dir, "sizes.json")))
    res = dict(bd={}, detection={}, palette_use={}, enc_time={}, lossless={}, quality={})
    md = []
    images = collections.defaultdict(set)
    for r in lossy:
        images[r["set"]].add(r["image"])

    def sel_of(st, flt):
        return sorted((st, i) for i in images[st] if flt is None or flt(i))

    # ---- lossy BD-rates
    for metric in ("ssimulacra2", "butteraugli"):
        per = curves(lossy, metric)
        md.append(f"\n### BD-rate on {metric}, median over images (mean), against libaom A\n")
        if metric == "ssimulacra2":
            cols = [("aom420_B", "aom420_A"), ("aom420_C", "aom420_A"), ("aom420_D", "aom420_A"), ("aom420_C0", "aom420_A"),
                    ("aom420_C", "aom420_C0"), ("gpu420", "aom420_A"), ("aom444_C", "aom444_A"), ("aom444_D", "aom444_A"),
                    ("gpu444", "aom444_A")]
        else:
            cols = [("aom420_C", "aom420_A"), ("aom420_D", "aom420_A"), ("gpu420", "aom420_A")]
        md.append("| Images | n | " + " | ".join(f"{t} vs {a}" for t, a in cols) + " |")
        md.append("|---|---:|" + "---:|" * len(cols))
        for label, st, flt in GROUPS:
            sel = sel_of(st, flt)
            cells = []
            for t, a in cols:
                d = bd_group(per, t, a, sel, metric)
                res["bd"].setdefault(metric, {}).setdefault(label, {})[f"{t}_vs_{a}"] = d
                cells.append(fmt_bd(d))
            md.append(f"| {label} | {len(sel)} | " + " | ".join(cells) + " |")
    # ---- mean quality and bytes per q (8-bit 4:2:0)
    md.append("\n### Bytes (sum) and mean SSIMULACRA2 per quality point, 8-bit 4:2:0\n")
    for label, st, flt in GROUPS:
        if label.endswith(": all") and st != "screen":
            continue
        line = [f"**{label}**"]
        for c in ("aom420_A", "aom420_B", "aom420_C", "aom420_D", "gpu420"):
            parts = []
            for q in (30, 45, 60, 75, 90):
                rs = [r for r in lossy if codec(r) == c and r["set"] == st and r["q"] == q and not r.get("missing") and
                      (flt is None or flt(r["image"])) and r.get("ssimulacra2") is not None]
                if rs:
                    b, s = sum(r["bytes"] for r in rs), statistics.fmean(r["ssimulacra2"] for r in rs)
                    parts.append(f"q{q} {b / 1e6:.2f} MB / {s:.1f}")
                    res["quality"].setdefault(label, {}).setdefault(c, {})[q] = dict(bytes=b, ssimulacra2=s, n=len(rs))
            line.append(f"{c}: " + ", ".join(parts))
        md.append("- " + "<br>".join(line))
    # ---- libaom's own decision (B) and tool use, from pstat
    md.append("\n### What libaom decided (pstat of every file): images with screen-content tools / IntraBC allowed, "
              "share of 4x4 units in palette / IntraBC blocks (mean over images, 8-bit 4:2:0)\n")
    md.append("| Images | n | B: sct on (q30..q90) | B: IntraBC on | A: sct on (detection) | palette area B / C / D | "
              "IntraBC area B / D | chroma palette area C |")
    md.append("|---|---:|---|---|---|---|---|---:|")
    for label, st, flt in GROUPS:
        sel = set(sel_of(st, flt))
        rs = [r for r in lossy if r["enc"] == "aom" and (r["set"], r["image"]) in sel and r.get("pstat")]

        def frac(cfg, key, fmt="420"):
            v = [r["pstat"][key] / r["pstat"]["mi"] for r in rs if r["cfg"] == cfg and r["fmt"] == fmt]
            return 100 * statistics.fmean(v) if v else None

        def on(cfg, key):
            per_q = collections.Counter(r["q"] for r in rs if r["cfg"] == cfg and r["fmt"] == "420" and r["pstat"][key])
            return "/".join(str(per_q.get(q, 0)) for q in (30, 45, 60, 75, 90))

        d = dict(b_sct=on("B", "sct"), b_ibc=on("B", "ibc_allowed"), a_sct=on("A", "sct"),
                 pal=[frac(c, "pal_y") for c in ("B", "C", "D")], ibc=[frac(c, "ibc") for c in ("B", "D")],
                 pal_uv_c=frac("C", "pal_uv"))
        res["detection"][label] = d
        f1 = lambda v: "–" if v is None else f"{v:.1f}"
        md.append(f"| {label} | {len(sel)} | {d['b_sct']} | {d['b_ibc']} | {d['a_sct']} | "
                  f"{' / '.join(f1(v) for v in d['pal'])} % | {' / '.join(f1(v) for v in d['ibc'])} % | {f1(d['pal_uv_c'])} % |")
    # ---- libaom encode time per config (sum of single-threaded avifenc wall times)
    md.append("\n### libaom encode time (sum of the single-threaded avifenc runs, all 5 quality points), relative to A\n")
    md.append("| Images | " + " | ".join(["A (s)", "B", "C", "D", "C0", "A 4:4:4 10-bit (s)", "C 4:4:4", "D 4:4:4"]) + " |")
    md.append("|---|" + "---:|" * 8)
    for label, st, flt in GROUPS:
        sel = set(sel_of(st, flt))
        t = collections.defaultdict(float)
        for r in lossy:
            if r["enc"] == "aom" and (r["set"], r["image"]) in sel and r.get("enc_s"):
                t[codec(r)] += r["enc_s"]
        res["enc_time"][label] = dict(t)
        a, a4 = t.get("aom420_A"), t.get("aom444_A")
        if not a:
            continue
        md.append(f"| {label} | {a:.0f} | " + " | ".join(f"{t[c] / a:.2f}×" for c in ("aom420_B", "aom420_C", "aom420_D", "aom420_C0")) +
                  f" | {a4:.0f} | " + " | ".join(f"{t[c] / a4:.2f}×" for c in ("aom444_C", "aom444_D")) + " |")
    # ---- lossless
    md.append("\n### Lossless: total bytes (exact decodes / files)\n")
    llc = [("aom", "gbr", "A"), ("aom", "gbr", "B"), ("aom", "gbr", "C"), ("aom", "gbr", "D"), ("aom", "ycgco", "A"),
           ("aom", "ycgco", "D"), ("gpu", "gbr", "lossless"), ("gpu", "ycgco", "lossless"), ("webp", "argb", "default"),
           ("webp", "argb", "z9")]
    names = {("aom", "gbr", "A"): "avifenc -l A", ("aom", "gbr", "B"): "B", ("aom", "gbr", "C"): "C", ("aom", "gbr", "D"): "D",
             ("aom", "ycgco", "A"): "A YCgCo-Re", ("aom", "ycgco", "D"): "D YCgCo-Re", ("gpu", "gbr", "lossless"): "av1gpu --lossless",
             ("gpu", "ycgco", "lossless"): "av1gpu --lossless --ycgco-re", ("webp", "argb", "default"): "cwebp -lossless",
             ("webp", "argb", "z9"): "cwebp -lossless -z 9"}
    md.append("| Images | n | PNG | " + " | ".join(names[k] for k in llc) + " |")
    md.append("|---|---:|---:|" + "---:|" * len(llc))
    for label, st, flt in LL_GROUPS:
        rs = [r for r in ll if r["set"] == st and (flt is None or flt(r["image"]))]
        imgs = sorted({r["image"] for r in rs})
        png = sum(sizes[st][i] for i in imgs)
        cells, d = [], dict(n=len(imgs), png=png)
        for k in llc:
            sel = [r for r in rs if (r["enc"], r["fmt"], r["cfg"]) == k and not r.get("missing")]
            b = sum(r["bytes"] for r in sel)
            ex = sum(1 for r in sel if r.get("exact"))
            d[names[k]] = dict(bytes=b, exact=ex, files=len(sel), enc_s=sum(r.get("enc_s", 0) for r in sel))
            cells.append(f"{b / 1e6:.2f} MB ({100 * b / png:.0f} %)" + ("" if ex == len(imgs) else f" [{ex}/{len(imgs)} exact]"))
        res["lossless"][label] = d
        md.append(f"| {label} | {len(imgs)} | {png / 1e6:.2f} MB | " + " | ".join(cells) + " |")
    # lossless: what libaom's detection did
    md.append("\n### Lossless: libaom's decisions (images with sct on / IntraBC on; palette and IntraBC area, mean)\n")
    md.append("| Images | B sct | B IntraBC | palette area B / C / D | IntraBC area B / D | chroma palette area C |")
    md.append("|---|---|---|---|---|---:|")
    for label, st, flt in LL_GROUPS:
        rs = [r for r in ll if r["enc"] == "aom" and r["fmt"] == "gbr" and r["set"] == st and (flt is None or flt(r["image"]))
              and r.get("pstat")]

        def fr(cfg, key):
            v = [r["pstat"][key] / r["pstat"]["mi"] for r in rs if r["cfg"] == cfg]
            return f"{100 * statistics.fmean(v):.1f}" if v else "–"
        nb = [r for r in rs if r["cfg"] == "B"]
        md.append(f"| {label} | {sum(r['pstat']['sct'] for r in nb)}/{len(nb)} | {sum(r['pstat']['ibc_allowed'] for r in nb)}/{len(nb)} | "
                  f"{fr('B', 'pal_y')} / {fr('C', 'pal_y')} / {fr('D', 'pal_y')} % | {fr('B', 'ibc')} / {fr('D', 'ibc')} % | {fr('C', 'pal_uv')} % |")
    os.makedirs(out, exist_ok=True)
    open(os.path.join(out, "tables.md"), "w").write("\n".join(md) + "\n")
    json.dump(res, open(os.path.join(out, "results.json"), "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
