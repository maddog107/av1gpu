#!/usr/bin/env python3
"""wall9.py N BIN [BIN2 ...] -- ARGS...: cold-process wall time of `BIN ARGS` (one process per run), N runs per binary,
binaries alternating run by run (A B C ... then reversed) after one warm-up each.  pipe08's wall.py, plus: a binary
may be given as PATH@VAR=VALUE[,VAR=VALUE] to run it with those environment variables.  ARGS may contain {bin}
(replaced by the binary's index) for per-binary output names.  Prints per-binary median / min / max in ms and every
run (JSON line)."""
import json, os, statistics, subprocess, sys, time

n = int(sys.argv[1])
k = sys.argv.index("--")
specs, args = sys.argv[2:k], sys.argv[k + 1:]
runs = {s: [] for s in specs}


def run(i, s):
    path, _, ev = s.partition("@")
    env = dict(os.environ)
    for kv in filter(None, ev.split(",")):
        a, _, b = kv.partition("=")
        env[a] = b
    a = [x.replace("{bin}", str(i)) for x in args]
    t = time.perf_counter()
    r = subprocess.run([path] + a, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, env=env)
    dt = (time.perf_counter() - t) * 1e3
    if r.returncode:
        print("FAIL", s, r.returncode, r.stderr.decode()[-300:], file=sys.stderr)
    return dt


for i, s in enumerate(specs):
    run(i, s)
for r in range(n):
    order = list(enumerate(specs)) if r % 2 == 0 else list(enumerate(specs))[::-1]
    for i, s in order:
        runs[s].append(run(i, s))
for s in specs:
    v = runs[s]
    print(f"{s}: median {statistics.median(v):.1f} ms, min {min(v):.1f}, max {max(v):.1f}, n {len(v)}")
print(json.dumps({"args": args, "runs": runs}))
