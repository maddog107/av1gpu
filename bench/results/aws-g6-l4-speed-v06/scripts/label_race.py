#!/usr/bin/env python3
"""label_race.py SUMMARY: relabel race_sum.py's '== X_K.log' headers with version and options, in place."""
import re, sys
L = {"old": "v0.5.0 b6545c1", "new": "v0.6 main 6a4956e", "m36": "63683ef (#36 merged, before #37)", "m37": "980cc10 (#37 merged)"}
K = {"0": "default", "1": "--yuv 444 --depth 10"}
p = sys.argv[1]
s = open(p).read()
s = re.sub(r"^== (?:\S*/)?(old|new|m36|m37)_([01])\.log:", lambda m: f"== {L[m[1]]}, {K[m[2]]}:", s, flags=re.M)
open(p, "w").write("racecheck (analysis mode) on av1gpu --batch, 3 size-range photos: hazards per pair of source locations\n\n" + s)
print(s[:200])
