import json, statistics, sys
from collections import defaultdict
g = defaultdict(list)
for f in sys.argv[1:]:
    for l in open(f):
        r = json.loads(l)
        if r["rc"] == 0 and r.get("mp_s"):
            g[(r.get("cpuset", "all"), r["list"], " ".join(r["args"]), json.dumps(r["env"]), r["bin"])].append(r)
names = (("JPEG", "jpeg", "--tiles 2,2,2", "{}"), ("JPEG, CPU entropy decode", "jpeg", "--tiles 2,2,2", '{"GP_INGEST_DECODER": "1"}'),
         ("PNG", "png", "--tiles 2,2,2", "{}"), ("PNG to 10-bit 4:4:4", "png", "--tiles 2,2,2 --yuv 444 --depth 10", "{}"), ("Mixed", "mix", "--tiles 2,2,2", "{}"))
for name, lst, a, e in names:
    cells = []
    for cs in ("0-3,16-19", "0-7,16-23", "all"):
        m, p = g.get((cs, lst, a, e, "main")), g.get((cs, lst, a, e, sys.argv[0] and "p1"))
        if m and p:
            mm, pp = statistics.median(x["mp_s"] for x in m), statistics.median(x["mp_s"] for x in p)
            cells.append(f"{mm:.1f} → {pp:.1f} ({(pp / mm - 1) * 100:+.0f} %)")
        else:
            cells.append("–")
    print(f"| {name} | " + " | ".join(cells) + " |")
for k in sorted(g):
    print(k, [round(x["mp_s"], 1) for x in g[k]], "gmem", max(x.get("gpu_mem_max_mb") or 0 for x in g[k]), "cores", round(statistics.median(x["cores"] for x in g[k]), 2), "mism", sum(x.get("md5_mismatch") or 0 for x in g[k]))
