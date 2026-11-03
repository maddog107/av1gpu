#!/usr/bin/env python3
"""wall.py N BIN [BIN2 ...] -- ARGS...: cold-process wall time of `BIN ARGS` (one process per run), N runs per binary,
binaries alternating run by run (A B A B ...) after one warm-up each.  ARGS may contain {bin} (replaced by the
binary's index) for per-binary output names.  Prints per-binary median / min / max in ms and every run (JSON line)."""
import json, statistics, subprocess, sys, time

n = int(sys.argv[1])
k = sys.argv.index("--")
bins, args = sys.argv[2:k], sys.argv[k + 1:]
runs = {b: [] for b in bins}
def run(i, b):
    a = [x.replace("{bin}", str(i)) for x in args]
    t = time.perf_counter()
    r = subprocess.run([b] + a, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    dt = (time.perf_counter() - t) * 1e3
    if r.returncode:
        print("FAIL", b, r.returncode, r.stderr.decode()[-300:], file=sys.stderr)
    return dt
for i, b in enumerate(bins):
    run(i, b)
for r in range(n):
    order = list(enumerate(bins)) if r % 2 == 0 else list(enumerate(bins))[::-1]
    for i, b in order:
        runs[b].append(run(i, b))
for b in bins:
    v = runs[b]
    print(f"{b}: median {statistics.median(v):.1f} ms, min {min(v):.1f}, max {max(v):.1f}, n {len(v)}")
print(json.dumps({"args": args, "runs": runs}))
