#!/usr/bin/env python3
"""Summarise the speed rerun (RUN_DIR = the instance's ~/results, pulled): median MP/s, img/s and CPU cores per phase /
setting / binary over the timed runs, and the L4's board power during them (nvidia-smi dmon).

  analyze_spd.py RUN_DIR [--json OUT]"""
import calendar, json, os, statistics, sys, time

RUN = sys.argv[1]
OUT = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None


def load(phase):
    p = os.path.join(RUN, f"{phase}.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


power = []   # (epoch, W, SM %)
if os.path.exists(os.path.join(RUN, "dmon.txt")):
    for l in open(os.path.join(RUN, "dmon.txt")):
        f = l.split()
        if len(f) < 6 or l.startswith("#"):
            continue
        try:
            t = calendar.timegm(time.strptime(f[0] + f[1], "%Y%m%d%H:%M:%S"))
            power.append((t, float(f[3]), float(f[6])))
        except (ValueError, IndexError):
            pass


def watts(runs):
    s = [(w, u) for t, w, u in power for r in runs if r["t0"] + 1 <= t <= r["t1"] - 1]
    return (statistics.fmean(w for w, _ in s), statistics.fmean(u for _, u in s), len(s)) if s else None


res = {}
for phase in ("jpeg", "formats", "lossless", "alpha", "kodak"):
    rows = load(phase)
    for key in sorted({(r["tag"], r["bin"]) for r in rows}):
        rs = [r for r in rows if (r["tag"], r["bin"]) == key and r["timed"] and r["rc"] == 0]
        if not rs:
            continue
        st = lambda k: statistics.median(r["stats"][k] for r in rs)  # noqa: E731
        d = dict(n=len(rs), mp_s=st("mp_s"), img_s=st("img_s"), cpu_cores=st("cpu_cores"), mp=rs[0]["stats"]["mp"],
                 images=rs[0]["stats"]["images"], bytes=rs[0]["stats"]["bytes"], wall_s=statistics.median(r["wall_s"] for r in rs),
                 mp_s_wall=statistics.median(r["stats"]["mp"] / r["wall_s"] for r in rs),
                 runs=[round(r["stats"]["mp_s"], 1) for r in rs], peak_dev_mb=max(r["stats"].get("peak_dev_mb", 0) for r in rs),
                 workers=(rs[0]["stats"].get("ingest_workers"), rs[0]["stats"].get("fallback_workers")),
                 bytes_same=len({r["stats"]["bytes"] for r in rs}) == 1,
                 rounds=[statistics.median(r["stats"]["mp_s"] for r in rs[i:i + 3]) for i in range(0, len(rs), 3)],
                 round_cores=[statistics.median(r["stats"]["cpu_cores"] for r in rs[i:i + 3]) for i in range(0, len(rs), 3)],
                 rounds_wall=[statistics.median(r["stats"]["mp"] / r["wall_s"] for r in rs[i:i + 3]) for i in range(0, len(rs), 3)])
        pw = watts(rs)
        if pw:
            d["watts"], d["sm_util"], d["power_samples"] = round(pw[0], 1), round(pw[1], 1), pw[2]
            d["j_per_mp"] = round(pw[0] / d["mp_s"], 3)
        res[f"{phase}/{key[0]}/{key[1]}"] = d
for k, d in res.items():
    extra = f" {d['watts']} W {d['sm_util']} % SM {d['j_per_mp']} J/MP" if "watts" in d else ""
    print(f"{k:42s} {d['mp_s']:7.1f} MP/s (wall {d['mp_s_wall']:6.1f}) {d['img_s']:7.2f} img/s {d['cpu_cores']:5.2f} cores "
          f"n={d['n']} runs {d['runs']} bytes {d['bytes']}{'' if d['bytes_same'] else ' (varies!)'} workers {d['workers']}{extra}")
if OUT:
    json.dump(res, open(OUT, "w"), indent=1)
