#!/usr/bin/env python3
"""BD-rates of the palette branch's av1gpu against main's (and against libaom A / C), from the scored rows.

  analyze_br.py MAIN_ROWS BR_ROWS OUT_MD      (rows: scored_lossy.jsonl of study.py score)"""
import collections
import gzip
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "bench"))
from bench import bd_rate  # noqa: E402


def _open(p):
    """a rows file, or its .gz"""
    if not os.path.exists(p) and os.path.exists(p + ".gz"):
        return gzip.open(p + ".gz", "rt")
    return open(p)


GROUPS = [
    ("flat-colour art, PNG", "flat_png", None),
    ("flat-colour art, JPEG q90", "flat_jpg", None),
    ("screen: all", "screen", None),
    ("screen: 23 graphics", "screen", lambda f: f.startswith("screen_synth_")),
    ("screen: real screenshots", "screen", lambda f: not f.startswith("screen_synth_") and not f.startswith("screen_wikipedia_")),
    ("screen: Wikipedia page renders", "screen", lambda f: f.startswith("screen_wikipedia_")),
]


def load(p):
    return [json.loads(l) for l in _open(p)]


def key(r, tag):
    if r["enc"] == "gpu":
        return f"{tag}gpu{r['fmt'][:3]}"
    return f"aom{r['fmt'][:3]}_{r['cfg']}"


def main():
    main_rows, br_rows, out = load(sys.argv[1]), load(sys.argv[2]), sys.argv[3]
    md = []
    for metric in ("ssimulacra2", "butteraugli"):
        per = collections.defaultdict(lambda: collections.defaultdict(list))
        for tag, rows in (("", main_rows), ("br_", br_rows)):
            for r in rows:
                if r.get("missing") or r.get(metric) is None:
                    continue
                if tag == "br_" and r["enc"] != "gpu":
                    continue
                v = r[metric] if metric == "ssimulacra2" else -r[metric]
                per[key(r, tag)][(r["set"], r["image"])].append((r["bytes"], v))
        cols = [("br_gpu420", "gpu420"), ("br_gpu420", "aom420_A"), ("br_gpu420", "aom420_C"), ("br_gpu420", "aom420_D"), ("gpu420", "aom420_A")]
        if metric == "ssimulacra2":
            cols += [("br_gpu444", "gpu444"), ("br_gpu444", "aom444_A")]
        md.append(f"\n### BD-rate on {metric}, median over images (mean)\n")
        md.append("| Images | n | " + " | ".join(f"{t.replace('br_', 'palette ')} vs {a}" for t, a in cols) + " |")
        md.append("|---|---:|" + "---:|" * len(cols))
        ov = 2.0 if metric == "ssimulacra2" else 0.1
        for label, st, flt in GROUPS + [("flat-colour art, PNG: the images palette is tried on", "flat_png", "changed"),
                                        ("screen: the images palette is tried on", "screen", "changed")]:
            if flt == "changed":   # the branch's detection passed: some file differs from main's
                mb = {(r["set"], r["image"], r["fmt"], r["q"]): r["bytes"] for r in main_rows if r["enc"] == "gpu"}
                ch = {(r["set"], r["image"]) for r in br_rows if r.get("bytes") and mb.get((r["set"], r["image"], r["fmt"], r["q"])) != r["bytes"]}
                sel = sorted(k for k in ch if k[0] == st)
            else:
                sel = sorted({(r["set"], r["image"]) for r in br_rows if r["set"] == st and (flt is None or flt(r["image"]))})
            cells = []
            for t, a in cols:
                vals = [v for k in sel if k in per[t] and k in per[a] for v in [bd_rate(per[t][k], per[a][k], ov)] if v is not None]
                cells.append(f"{100 * statistics.median(vals):+.1f} % ({100 * statistics.fmean(vals):+.1f})" if vals else "–")
            md.append(f"| {label} | {len(sel)} | " + " | ".join(cells) + " |")
    open(out, "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
