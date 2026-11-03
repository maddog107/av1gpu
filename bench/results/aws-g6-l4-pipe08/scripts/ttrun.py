#!/usr/bin/env python3
"""ttrun.py N BIN ARGS...: run an instrumented binary N times; per label the median ms since the exec (CLOCK_MONOTONIC
before Popen) and the process end; prints the phase table (median of N)."""
import statistics, subprocess, sys, time
n = int(sys.argv[1]); cmd = sys.argv[2:]
rows = {}
order = []
for _ in range(n):
    t0 = time.clock_gettime(time.CLOCK_MONOTONIC) * 1e3
    r = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    t1 = time.clock_gettime(time.CLOCK_MONOTONIC) * 1e3
    seen = {}
    for ln in r.stderr.splitlines():
        if ln.startswith("[tt] "):
            lab, v = ln[5:].rsplit(" ", 1)
            k = lab
            i = 2
            while k in seen:
                k = f"{lab} #{i}"; i += 1
            seen[k] = float(v) - t0
            if k not in order: order.append(k)
    seen["process end"] = t1 - t0
    if "process end" not in order: order.append("process end")
    for k, v in seen.items(): rows.setdefault(k, []).append(v)
prev = 0
for k in order:
    m = statistics.median(rows[k])
    print(f"{m:8.1f} ms  (+{m - prev:6.1f})  {k}")
    prev = m
