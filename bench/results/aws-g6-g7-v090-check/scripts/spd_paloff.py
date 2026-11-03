#!/usr/bin/env python3
"""spd_paloff.py: the RGBA set (39 PNGs x 20) and the lossless comparison set (47 PNGs x 4) of speed-v06's spd.py
with --palette off added, v0.9.0 (new) and v0.8.0 (old): how much of the RGBA set's speed palette mode (on by default
since v0.8.0, --palette auto) takes.  Same method (a warm-up and 3 timed runs, 2 rounds with the order alternating,
GP_ENCODERS=4 for the RGBA set as published).  ~/results/paloff/<phase>.jsonl"""
import os, sys

H = os.path.expanduser("~")
sys.argv = [sys.argv[0]]
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("spd", f"{H}/spd_docs.py")
spd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spd)
spd.R = f"{H}/results/paloff"
os.makedirs(spd.R, exist_ok=True)
import glob  # noqa: E402

files = sorted(glob.glob(f"{H}/data/rgba/set/*.png"))
d = spd.rep_dir("rgba20", files, 20)
cmp = sorted(glob.glob(f"{H}/ll/cmp/*.png"))
four = spd.rep_dir("ll4", cmp, 4)
X = [("default_paloff", d, ["--palette", "off"], {"GP_ENCODERS": "4"}),
     ("no_alpha_paloff", d, ["--no-alpha", "--palette", "off"], {"GP_ENCODERS": "4"}),
     ("q100_paloff", d, ["--alpha-quality", "100", "--palette", "off"], {"GP_ENCODERS": "4"}),
     ("ll_cmp_4x_paloff", four, ["--lossless", "--palette", "off"], {}),
     ("ll_cmp_4x_ycgco_paloff", four, ["--lossless", "--ycgco-re", "--palette", "off"], {})]
for rnd in range(2):
    for b in (("new", "old") if rnd % 2 == 0 else ("old", "new")):
        for tag, src, opts, env in X:
            spd.speed_call("alpha" if not tag.startswith("ll") else "lossless", b, src, opts, env, tag)
