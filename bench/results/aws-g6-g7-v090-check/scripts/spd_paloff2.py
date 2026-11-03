#!/usr/bin/env python3
"""spd_paloff2.py: spd_paloff.py's RGBA set with --lossless --palette off (the transparency table's lossless row),
same method; ~/results/paloff/alpha.jsonl"""
import os, sys

H = os.path.expanduser("~")
sys.argv = [sys.argv[0]]
import importlib.util  # noqa: E402
spec = importlib.util.spec_from_file_location("spd", f"{H}/spd_docs.py")
spd = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spd)
spd.R = f"{H}/results/paloff"
import glob  # noqa: E402

d = spd.rep_dir("rgba20", sorted(glob.glob(f"{H}/data/rgba/set/*.png")), 20)
for rnd in range(2):
    for b in (("new", "old") if rnd % 2 == 0 else ("old", "new")):
        spd.speed_call("alpha", b, d, ["--lossless", "--palette", "off"], {"GP_ENCODERS": "4"}, "lossless_paloff")
print("PALOFF2_DONE")
