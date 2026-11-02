#!/usr/bin/env python3
"""mk_results_json.py RUN_DIR ANALYSIS_JSON OUT: bench/results/aws-g6-l4-speed-v06/results.json (as -v04's: description,
code, environment, sets, speed, demo; plus the release check's identity / gate / sanitizer summaries)."""
import json, os, re, sys
RUN, AJ, OUT = sys.argv[1:4]
rd = lambda f: open(os.path.join(RUN, f)).read() if os.path.exists(os.path.join(RUN, f)) else ""  # noqa: E731
env = [l for l in rd("env.txt").splitlines() if l and not l.startswith("+")]
env = [l for l in env if not l.startswith("new ef2522f")] + ["new 6a4956eaee2c6ffa2ec05d955afc1720031ceb52 (rebuilt, logs/pipe6.log) "
                                                              "old b6545c1a5a5d2105ead1e8db5b23d01ccb40b56c"]
ident = []
for l in rd("identity.txt").splitlines():
    m = re.match(r"(\w+): inputs (\d+), files (\d+) / (\d+), differ (\d+).*opts \[([^\]]*)\]\s*(\S*)", l)
    if m:
        ident.append(dict(set=m[1], inputs=int(m[2]), files_v050=int(m[3]), files_v06=int(m[4]), differ=int(m[5]),
                          opts=m[6].strip(), env=m[7]))
sets = {}
for f in ("jpeg_sets.json", "lossless_set.json", "alpha_set.json", "thumbs_set.json", "kodak_set.json"):
    if rd(f):
        sets[f[:-9] if f.endswith("_set.json") else "jpeg"] = json.loads(rd(f))
out = dict(
    description="av1gpu speed on an NVIDIA L4 (AWS g6.4xlarge, 16 vCPUs), 2026-11-01: v0.6 (main 6a4956e, #32-#40) against "
                "v0.5.0 (b6545c1) in the same session, after a byte-identity check of every output. Method as -v04: each set "
                "repeated to >= 1,000 MP (lossless and RGBA sets as published), a warm-up then 3 timed runs (--stats), 2 "
                "interleaved rounds with the binary order alternating; median of the 6 timed runs. mp_s_wall / img_s_wall: "
                "megapixels / images over the whole process's wall time (CUDA start-up included).",
    date="2026-11-01", code=dict(new="6a4956eaee2c6ffa2ec05d955afc1720031ceb52", old="b6545c1a5a5d2105ead1e8db5b23d01ccb40b56c"),
    environment=env, sets=sets, speed=json.load(open(AJ)),
    identity=dict(runs=len(ident), files=sum(x["files_v06"] for x in ident), differ=sum(x["differ"] for x in ident), sets=ident),
    gate=[l.strip() for l in rd("gate.txt").splitlines() if l.startswith(("== ", "[gate]"))],
    demo=json.loads(rd("rec.json")) if rd("rec.json").strip() else None)
json.dump(out, open(OUT, "w"), indent=1)
print("wrote", OUT, len(out["speed"]), "speed entries,", out["identity"]["runs"], "identity runs")
