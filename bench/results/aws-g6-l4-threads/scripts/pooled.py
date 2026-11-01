"""pooled.py DIR: medians over every sweep / confirm / qtest run per instance size (main and the branch at their defaults)."""
import glob, json, statistics, sys
from collections import defaultdict
g = defaultdict(list)
for f in glob.glob(f"{sys.argv[1]}/*/*.jsonl"):
    if not any(x in f for x in ("sweep.jsonl", "confirm.jsonl", "qtest.jsonl")):
        continue
    for l in open(f):
        r = json.loads(l)
        if r["rc"] == 0 and r.get("mp_s") and not r["env"] or r["env"] == {"GP_INGEST_DECODER": "1"}:
            if r["rc"] == 0 and r.get("mp_s"):
                g[(r["n"], r["list"], " ".join(r["args"]), r["bin"], json.dumps(r["env"]))].append(r)
for tiles in ("--tiles 2,2,2", ""):
    print("tiles:", tiles or "2x2 (main's default)")
    for name, lst, a, e in (("JPEG", "jpeg", "", {}), ("JPEG, CPU entropy decode (GP_INGEST_DECODER=1)", "jpeg", "", {"GP_INGEST_DECODER": "1"}),
                            ("PNG", "png", "", {}), ("PNG to 10-bit 4:4:4", "png", "--yuv 444 --depth 10", {}), ("mixed JPEG / PNG / WebP", "mix", "", {})):
        args = " ".join(x for x in (tiles, a) if x)
        cells = []
        for n in (8, 16, 32):
            m = g.get((n, lst, args, "main", json.dumps(e)), [])
            w = g.get((n, lst, args, "new", json.dumps(e)), [])
            if m and w:
                mm, ww = statistics.median(x["mp_s"] for x in m), statistics.median(x["mp_s"] for x in w)
                cells.append(f"{mm:.1f} → {ww:.1f} ({(ww / mm - 1) * 100:+.0f} %)")
            else:
                cells.append("–")
            cells[-1] += f" [{len(m)}/{len(w)}]"
        print(f"| {name} | " + " | ".join(cells) + " |")
    # CPU cores used
    for lst, a in (("png", ""), ("png", "--yuv 444 --depth 10"), ("mix", ""), ("jpeg", "")):
        args = " ".join(x for x in (tiles, a) if x)
        print(lst, a, [(n, round(statistics.median(x["cores"] for x in g[(n, lst, args, "main", "{}")]), 1),
                        round(statistics.median(x["cores"] for x in g[(n, lst, args, "new", "{}")]), 1),
                        max(x.get("gpu_mem_max_mb") or 0 for x in g[(n, lst, args, "main", "{}")]),
                        max(x.get("gpu_mem_max_mb") or 0 for x in g[(n, lst, args, "new", "{}")])) for n in (8, 16, 32)])
