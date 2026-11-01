#!/usr/bin/env python3
"""t4.py: 8-vCPU emulation (taskset 0-3,16-19): batch logs, then main vs p1 vs p1 with niced fallback decoding"""
import os, tthr
H = tthr.H
tthr.BIN.update(p1=f"{H}/bin/av1gpu-p1", p1n=f"{H}/bin/av1gpu-p1n")
T = ["--tiles", "2,2,2"]
pre = ["taskset", "-c", "0-3,16-19"]
L = f"{H}/results/blog"
for b, e, tag in (("main", {}, "main8"), ("new", {"GP_FB_POLICY": "1"}, "p1_8"), ("main", {}, "main8b"), ("new", {"GP_FB_POLICY": "1"}, "p1_8b")):
    r = tthr.run(b, "mix", T, {**e, "GP_BATCH_LOG": f"{L}/mix_{tag}.txt"}, prefix=pre, tag="t4")
    tthr.log("t4_instr", r)
cs = [("main", "mix", T, {}), ("p1", "mix", T, {}), ("p1n", "mix", T, {"GP_FALLBACK_NICE": "10"}), ("p1n", "mix", T, {"GP_FALLBACK_NICE": "19"}),
      ("main", "png", T, {}), ("p1", "png", T, {}), ("p1n", "png", T, {"GP_FALLBACK_NICE": "10"}), ("p1n", "png", T, {"GP_FALLBACK_NICE": "19"})]
for rd in range(4):
    for b, l, a, e in (cs if rd % 2 == 0 else cs[::-1]):
        r = tthr.run(b, l, a, e, prefix=pre, tag="t4")
        r["round"] = rd
        r["cpuset"] = "0-3,16-19"
        tthr.log("t4", r)
print("T4_DONE", flush=True)
