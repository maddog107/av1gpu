#!/usr/bin/env python3
"""palette area (share of 4x4 units in palette blocks, pstat) of av1gpu outputs vs libaom C, per set at one q.
  paluse.py OUTDIR [Q]       (OUTDIR: a study.py OUT with gpu/420/<set>/*.q<qindex>.avif)"""
import glob
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import study  # noqa: E402

out, q = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 60
qi = study.qindex_for_quality(q)
for s in ("screen", "flat_png"):
    g, a = [], []
    for f in sorted(glob.glob(f"{out}/gpu/420/{s}/*.q{qi}.avif")):
        p = study.pstat(f)
        img = os.path.basename(f).rsplit(".q", 1)[0]
        pa = study.pstat(f"{study.PAL}/out/aom/420_C/{s}/{img}.q{q}.avif")
        if p and pa:
            g.append(p["pal_y"] / p["mi"])
            a.append(pa["pal_y"] / pa["mi"])
    print(f"{s}: n={len(g)} av1gpu palette area {100 * statistics.fmean(g):.1f} %, libaom C {100 * statistics.fmean(a):.1f} % (q{q})")
