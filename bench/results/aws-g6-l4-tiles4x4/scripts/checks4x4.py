#!/usr/bin/env python3
"""Checks on the 4x4 run (RUN_DIR = the instance's ~/results, pulled):
1. OLD (v0.3.0) files have the published sizes (so OLD = the published encoder),
2. NEW vs OLD bytes by size class (identical below 2 MP and for lossless),
3. the scorer reproduces the published SSIMULACRA2 / CVVDP / luma-only CVVDP on unchanged and OLD files,
and picks the luma-only reference variant that reproduces the published one (luma_variant.json).

  checks4x4.py WORKTREE RUN_DIR"""
import json, os, sys

WT, RUN = sys.argv[1], sys.argv[2]
sys.path.insert(0, os.path.join(WT, "bench"))
import bench  # noqa: E402

RES = os.path.join(WT, "bench", "results")
L = lambda n: json.load(open(os.path.join(RES, n, "results.json")))  # noqa: E731
man = json.load(open(os.path.join(RUN, "manifests.json")))
cmpj = json.load(open(os.path.join(RUN, "compare.json")))
sc = json.load(open(os.path.join(RUN, "scores.json"))) if os.path.exists(os.path.join(RUN, "scores.json")) else []
inv = {bench.qindex_for_quality(q): q for q in (30, 40, 50, 60, 70, 80, 85, 90)}
out = {}


def stored_sizes(rows):
    return {(r["image"], r["q"]): r["bytes"] for r in rows}


def run_sizes(key):
    o = {}
    for f, m in man[key]["files"].items():
        img, q = f.rsplit(".q", 1)
        o[(img, inv[int(q[:-5])])] = m["bytes"]
    return o


# 1. OLD = published
pairs = [("old/gamut_default", L("aws-g6-l4-gamut")["quality"]["av1gpu"]),
         ("old/gamut_default", L("aws-g6-l4-cvvdp")["quality"]["av1gpu"]),
         ("old/gamut_nosmall", L("aws-g6-l4-gamut-turbo-nosmall")["quality"]["av1gpu"]),
         ("old/std_default", L("aws-g6-l4")["quality"]["av1gpu"]),
         ("old/std_nosmall", L("aws-g6-l4-turbo-nosmall")["quality"]["av1gpu"])]
fm = L("aws-g6-l4-formats")["quality"]
pairs += [(f"old/fmt_{f}", fm[f"av1gpu_{f}"]) for f in ("444_8", "420_10", "444_10")]
pairs += [("old/gamut_default", fm["av1gpu_420_8"])]
o1 = []
for key, rows in pairs:
    s, r = stored_sizes(rows), run_sizes(key)
    common = [k for k in s if k in r]
    o1.append(dict(run=key, stored_rows=len(s), common=len(common), same_size=sum(s[k] == r[k] for k in common),
                   stored_not_in_run=len([k for k in s if k not in r])))
out["old_vs_published"] = o1

# 2. NEW vs OLD
out["new_vs_old"] = {k: {x: y for x, y in v.items() if x not in ("only_new", "only_old")} for k, v in cmpj["compare"].items()}

# 3. scorer controls
if sc:
    pub = {(r["image"], r["q"]): r for r in L("aws-g6-l4-cvvdp")["quality"]["av1gpu"]}
    lum = {(r["image"], r["q"]): r["cvvdp_luma"] for r in L("aws-g6-l4-cvvdp")["cvvdp_luma_only"]["rows"] if r["codec"] == "av1gpu"}
    gpub = {(r["image"], r["q"]): r for r in L("aws-g6-l4-gamut")["quality"]["av1gpu"]}
    ctrl = dict(rows=0, same_bytes=0, ssim_equal=0, ssim_maxdiff=0.0, cvvdp_rows=0, cvvdp_maxdiff=0.0,
                luma_rows=0, luma_maxdiff=0.0, luma_noicc_rows=0, luma_noicc_maxdiff=0.0, luma_icc_maxdiff_icc4=0.0)
    unchanged = dict(rows=0, ssim_equal=0, ssim_maxdiff=0.0, cvvdp_rows=0, cvvdp_maxdiff=0.0)
    for r in sc:
        if r["config"] != "gamut_default":
            continue
        k = (r["image"], r["q"])
        p = pub.get(k) or gpub.get(k)
        if not p or p["bytes"] != r["bytes"]:
            continue
        tgt = ctrl if r["bin"] == "old" else unchanged
        if r["bin"] == "new":
            # NEW files with the published size: below 2 MP (identical bytes), scored again
            pass
        tgt["rows"] += 1
        if r["bin"] == "old":
            tgt["same_bytes"] += 1
        d = abs(r["ssimulacra2"] - p["ssimulacra2"])
        tgt["ssim_equal"] += d == 0
        tgt["ssim_maxdiff"] = max(tgt["ssim_maxdiff"], d)
        if "cvvdp" in r and r["cvvdp"] is not None and "cvvdp" in p:
            tgt["cvvdp_rows"] += 1
            tgt["cvvdp_maxdiff"] = max(tgt["cvvdp_maxdiff"], abs(r["cvvdp"] - p["cvvdp"]))
        if r["bin"] == "old" and k in lum:
            if r.get("cvvdp_luma") is not None:
                ctrl["luma_rows"] += 1
                ctrl["luma_maxdiff"] = max(ctrl["luma_maxdiff"], abs(r["cvvdp_luma"] - lum[k]))
            if r.get("cvvdp_luma_noicc") is not None:
                ctrl["luma_noicc_rows"] += 1
                ctrl["luma_noicc_maxdiff"] = max(ctrl["luma_noicc_maxdiff"], abs(r["cvvdp_luma_noicc"] - lum[k]))
                ctrl["luma_icc_maxdiff_icc4"] = max(ctrl["luma_icc_maxdiff_icc4"], abs(r["cvvdp_luma"] - lum[k]))
    out["score_controls"] = dict(old_files=ctrl, new_unchanged_files=unchanged)
    v = "cvvdp_luma_noicc" if ctrl["luma_noicc_rows"] and ctrl["luma_noicc_maxdiff"] < ctrl["luma_icc_maxdiff_icc4"] else "cvvdp_luma"
    json.dump(dict(variant=v, why=ctrl), open(os.path.join(RUN, "luma_variant.json"), "w"), indent=1)
    out["luma_variant"] = v
json.dump(out, open(os.path.join(RUN, "checks.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
