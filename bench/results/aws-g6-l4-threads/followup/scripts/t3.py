#!/usr/bin/env python3
"""t3.py CPULIST|all ROUNDS: main (instr-main, v0.3.0 + #21/#22/#27) vs p1 (the final branch) on every list; prefix taskset"""
import sys, tthr
tthr.BIN["p1"] = f"{tthr.H}/bin/av1gpu-p1"
T = ["--tiles", "2,2,2"]
P4 = T + ["--yuv", "444", "--depth", "10"]
cpus, n = sys.argv[1], int(sys.argv[2])
prefix = [] if cpus == "all" else ["taskset", "-c", cpus]
cs = []
for lst, a, e in (("mix", T, {}), ("png", T, {}), ("png", P4, {}), ("jpeg", T, {}), ("jpeg", T, {"GP_INGEST_DECODER": "1"})):
    cs += [("main", lst, a, e), ("p1", lst, a, e)]
for rd in range(n):
    for b, l, a, e in (cs if rd % 2 == 0 else cs[::-1]):
        r = tthr.run(b, l, a, e, prefix=prefix, tag="t3")
        r["round"] = rd
        r["cpuset"] = cpus
        tthr.log("t3", r)
print("T3_DONE", cpus, flush=True)
