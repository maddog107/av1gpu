#!/usr/bin/env python3
"""analyze.py RUN_DIR: the speed table from RUN_DIR/race.jsonl and noise.jsonl (spd.py): per setting and binary, the
median MP/s of the 6 timed runs (av1gpu's --stats), each round's median, wall MP/s (incl. start-up), and the
change new against old (noise: old2 against old).  Markdown on stdout; RUN_DIR/speed.json."""
import json, statistics, sys

d = sys.argv[1]
res = {}
rows = []
for ph, a, b in (("race", "new", "old"), ("noise", "old2", "old")):
    try:
        rs = [json.loads(l) for l in open(f"{d}/{ph}.jsonl")]
    except FileNotFoundError:
        continue
    rs = [r for r in rs if r["timed"]]
    bad = [r for r in rs if r["rc"]]
    tags = list(dict.fromkeys(r["tag"] for r in rs))
    for t in tags:
        e = {}
        for x in (a, b):
            q = [r for r in rs if r["tag"] == t and r["bin"] == x]
            mps = [r["stats"]["mp_s"] for r in q]
            wall = [r["stats"]["mp"] / r["wall_s"] for r in q]
            rnd = [statistics.median([r["stats"]["mp_s"] for r in q if r["rnd"] == k]) for k in (0, 1)]
            e[x] = dict(n=len(q), mps=statistics.median(mps), rounds=rnd, wall=statistics.median(wall), runs=mps,
                        mp=q[0]["stats"]["mp"], files=q[0]["files"])
        ch = 100 * (e[a]["mps"] / e[b]["mps"] - 1)
        chw = 100 * (e[a]["wall"] / e[b]["wall"] - 1)
        res[f"{ph}:{t}"] = dict(e, change_pct=round(ch, 2), wall_change_pct=round(chw, 2), failed=len(bad))
        rows.append(f"| {ph} | {t} | {e[a]['mps']:.1f} | {e[b]['mps']:.1f} | {ch:+.2f} % | "
                    f"{e[a]['rounds'][0]:.1f} / {e[a]['rounds'][1]:.1f} | {e[b]['rounds'][0]:.1f} / {e[b]['rounds'][1]:.1f} | "
                    f"{e[a]['wall']:.1f} | {e[b]['wall']:.1f} | {chw:+.2f} % | {e[a]['n']} / {e[b]['n']} | {e[a]['mp']:.0f} |")
print("| phase | set / setting | A MP/s | B MP/s | change | A rounds | B rounds | A wall | B wall | wall change | runs | MP/run |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
print("\n".join(rows))
print("\nA/B: race = 78e374c / d7d94b3; noise = d7d94b3 / d7d94b3 (the same binary twice)")
json.dump(res, open(f"{d}/speed.json", "w"), indent=1)
