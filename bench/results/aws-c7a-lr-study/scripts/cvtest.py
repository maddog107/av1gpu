#!/usr/bin/env python3
"""Time and peak memory of one CVVDP score (cvscore.py's setup) on a 1 MP and the largest size-range photo."""
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cvscore  # noqa: E402

H = os.path.expanduser("~")
cvscore.init()
big = max((os.path.join(H, "img/gamut", f) for f in os.listdir(f"{H}/img/gamut")), key=os.path.getsize)
small = sorted(os.path.join(H, "img/std", f) for f in os.listdir(f"{H}/img/std"))[0]
for src, s in ((small, "std"), (big, "gamut")):
    p = f"{H}/lr/out/420/s6/{s}/{os.path.basename(src)}.q60.avif"
    t0 = time.time()
    r = cvscore.one(("s6", s, src, 60, p))
    print(os.path.basename(src), f"{cvscore.mp(src):.1f} MP", r["jod"], f"{time.time() - t0:.1f} s",
          f"maxrss {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB", flush=True)
