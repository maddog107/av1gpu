#!/usr/bin/env python3
"""PGO training for SVT-AV1: avifenc -c svt (the instrumented library via LD_LIBRARY_PATH) on the 100-image
~1 MP set, presets 2/4/6/8/9, tune=3, q 30/60/80, 16 single-threaded encodes at once (as in the benchmark)."""
import concurrent.futures as cf
import glob
import os
import subprocess
import sys
import tempfile
import time

avifenc, imgdir = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(os.path.join(imgdir, "*.jpg")))
out = tempfile.mkdtemp()
jobs = []
for p in (2, 4, 6, 8, 9):
    for q in (30, 60, 80):
        for f in files:
            dst = os.path.join(out, f"p{p}q{q}_{os.path.basename(f)}.avif")
            jobs.append([avifenc, "-q", str(q), "-y", "420", "-j", "1", "--ignore-exif", "--ignore-xmp", "-c", "svt",
                         "-a", f"preset={p}", "-a", "tune=3", f, dst])
t0 = time.time()
with cf.ThreadPoolExecutor(os.cpu_count()) as ex:
    rcs = list(ex.map(lambda c: subprocess.run(c, capture_output=True).returncode, jobs))
print(f"training: {len(files)} images, {len(jobs)} encodes, {rcs.count(0)} ok, {time.time() - t0:.0f} s")
sys.exit(0 if rcs.count(0) == len(jobs) else 1)
