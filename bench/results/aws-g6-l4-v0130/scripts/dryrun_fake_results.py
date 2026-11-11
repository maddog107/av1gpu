#!/usr/bin/env python3
"""Dry run of analyze.py's run path on the Mac: a fake instance ~/results built from stored data (tiles4x4's av1gpu rows
as both NEW and OLD, the v0.9.0 check's docs speed logs / dmon / start-up / inputs, tiles4x4's gate logs and alpha rows),
then analyze.py on it (--allow-failed-runs: v0.9.0's lossless list had 3 failed OLD runs); the bd_rate / speed it writes must equal the stored ones.

  dryrun_fake_results.py WORKTREE FAKE_DIR OUT_DIR"""
import glob, hashlib, json, os, shutil, subprocess, sys

WT, F, OUT = sys.argv[1:4]
RES = os.path.join(WT, "bench", "results")
sys.path.insert(0, os.path.join(WT, "bench"))
import bench  # noqa: E402
L = lambda n: json.load(open(os.path.join(RES, n, "results.json")))  # noqa: E731
t4, v9 = L("aws-g6-l4-tiles4x4"), L("aws-g6-g7-v090-check")
D9 = os.path.join(RES, "aws-g6-g7-v090-check", "logs", "l4")
shutil.rmtree(F, ignore_errors=True)
for d in ("speed", "quality/gates"):
    os.makedirs(os.path.join(F, d))
q = t4["quality"]
CFG = {"gamut_default": q["av1gpu_8pts"], "gamut_nosmall": q["av1gpu_nosmall"], "gamut_turbo3ns": q["av1gpu_turbo3_nosmall"],
       "gamut_turbo3": q["av1gpu_turbo3"], "fmt_444_8": q["av1gpu_444_8"], "fmt_420_10": q["av1gpu_420_10"],
       "fmt_444_10": q["av1gpu_444_10"], "std_default": L("aws-g6-l4")["quality"]["av1gpu"],
       "std_nosmall": L("aws-g6-l4-turbo-nosmall")["quality"]["av1gpu"]}
md5 = lambda *a: hashlib.md5(repr(a).encode()).hexdigest()  # noqa: E731
manifests, compare, sc_new, sc_cv = {}, {}, [], []
for cfg, rows in CFG.items():
    files = {f"{r['image']}.q{bench.qindex_for_quality(r['q'])}.avif": dict(bytes=r["bytes"], md5=md5(cfg, r["image"], r["q"], r["bytes"]))
             for r in rows}
    for b in ("new", "old"):
        manifests[f"{b}/{cfg}"] = dict(rc=0, files=files)
    tot = dict(files=len(files), identical=len(files), bytes_new=sum(v["bytes"] for v in files.values()),
               bytes_old=sum(v["bytes"] for v in files.values()))
    compare[cfg] = dict(new=len(files), old=len(files), all=tot, lt1=tot, **{"1to2": tot}, ge2=tot,
                        unknown_size=dict(files=0, identical=0))
    for r in rows:
        x = dict(config=cfg, bin="new", image=r["image"], q=r["q"], bytes=r["bytes"], md5=md5(cfg, r["image"], r["q"], r["bytes"]),
                 ssimulacra2=r["ssimulacra2"])
        sc_new.append(x)
        if cfg == "gamut_default" and r["q"] <= 80:
            sc_new.append(dict(x, bin="old"))
        if cfg == "gamut_default" and r.get("cvvdp") is not None:
            sc_cv.append(dict(x, cvvdp=r["cvvdp"]))
J = lambda p, o: json.dump(o, open(os.path.join(F, p), "w"))  # noqa: E731
J("quality/manifests.json", manifests)
J("quality/compare.json", dict(megapixels={}, compare=compare))
J("quality/scores_new.json", sc_new)
J("quality/scores_old.json", [])
J("quality/scores_cvvdp.json", sc_cv)
J("quality/cvvdp_info.json", t4["cvvdp"])
J("quality/alpha.json", t4["alpha"])
for g in glob.glob(os.path.join(RES, "aws-g6-l4-tiles4x4", "logs", "*_q*.log")):
    shutil.copy(g, os.path.join(F, "quality", "gates"))
grp = {"core": lambda r: (r["phase"] == "jpeg" and r["tag"] in ("size_range/default_enc4", "size_range/default_enc3", "size_range/nosmall_enc4",
                                                                 "size_range/turbo3ns_enc4", "size_range/turbo3_enc4")) or r["phase"] == "formats",
       "small": lambda r: r["phase"] in ("thumbs", "kodak"), "std": lambda r: r["phase"] == "jpeg" and r["tag"].startswith("standard/"),
       "ll": lambda r: r["phase"] == "lossless", "alpha": lambda r: r["phase"] == "alpha"}
grp["extra"] = lambda r: r["phase"] == "jpeg" and r["tag"].startswith("size_range/") and not grp["core"](r)
rows = [json.loads(l) for p in sorted(glob.glob(os.path.join(D9, "docs", "*.jsonl"))) for l in open(p)]
for g, sel in grp.items():
    with open(os.path.join(F, "speed", f"{g}.jsonl"), "w") as f:
        for r in rows:
            if sel(r):
                f.write(json.dumps(r) + "\n")
shutil.copy(os.path.join(D9, "dmon.txt"), F)
shutil.copy(os.path.join(D9, "inputs.sha256"), F)
s = open(os.path.join(D9, "spd9_single.txt")).read()
s = s.replace("/home/ubuntu/bin/B13", "/home/ubuntu/bin/new").replace("/home/ubuntu/bin/A13c", "/home/ubuntu/bin/old2") \
     .replace("/home/ubuntu/bin/A13", "/home/ubuntu/bin/old")
open(os.path.join(F, "startup.txt"), "w").write(s)
open(os.path.join(F, "env.txt"), "w").write(open(os.path.join(D9, "env.txt")).read() + "\nnproc 16\nrgba: 39 files\n")
open(os.path.join(F, "binaries.txt"), "w").write("new /home/ubuntu/rel/fake/av1gpu 0000 av1gpu 0.13.0 (dry run)\n"
                                                 "old /home/ubuntu/rel/fake/av1gpu 0000 av1gpu 0.12.0 (dry run)\n")
open(os.path.join(F, "progress.log"), "w").write("".join(f"## {p}: OK, 1 s\n" for p in (
    "speed", "quality", "speed_small", "speed_std", "speed_ll", "gates", "startup", "alpha", "cvvdp", "quality_old", "speed_extra")))
p = subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "analyze.py"), WT, F, "--out", OUT, "--allow-failed-runs"],
                   capture_output=True, text=True)
print(p.stdout[-1500:], p.stderr[-3000:])
if p.returncode:
    sys.exit("analyze.py failed")
r = json.load(open(os.path.join(OUT, "results.json")))
bad = [k for k in t4["bd_rate"] if k in r["bd_rate"] and r["bd_rate"][k] != t4["bd_rate"][k]]
bad += [k for k in t4["bd_rate"] if k not in r["bd_rate"]]
bad += [k for k in v9["speed"] if r["speed"].get(k) != json.loads(json.dumps(v9["speed"][k]))]
t4c = {k: v for k, v in t4["bd_rate_cvvdp"].items() if k != "av1gpu_vs_aom_s6_luma_only"}   # luma-only: not re-measured
bad += [] if r["bd_rate_cvvdp"] == t4c else ["bd_rate_cvvdp"]
bad += [k for k in ("cvvdp_target", "totals") if json.loads(json.dumps(r[k])) != json.loads(json.dumps(t4[k]))]
print("dry run:", "results.json reproduces tiles4x4 bd_rate / bd_rate_cvvdp / cvvdp_target / totals and v0.9.0 speed"
      if not bad else f"DIFFERENCES: {bad[:20]}")
sys.exit(1 if bad else 0)
