#!/usr/bin/env python3
"""append_interp.py SUMMARY.md: insert the reading of the speed numbers after the single-JPEG table."""
import sys

p = sys.argv[1]
s = open(p).read()
TXT = """
**Reading the numbers.** The noise run's medians agree within 0.3 %, but the rounds of one binary differ by up to
1.3 % (size range) and 2.2 % (standard set), so:

- **Standard set +4.2 %: outside the noise**; both branch rounds (76.2, 75.3) are above both main rounds (73.3, 72.4).
  Change 1 +1.8 % and change 2 +2.3 % each, at the edge of the round spread on their own.
- **Thumbnails +3.1 %** on the median of 6; round 1 +4.5 %, round 2 +0.6 %. The median is above the noise run's
  0.3 %, the second round alone is not; the ncu kernel times (k_analysis -7.7 % from change 1) support a real gain.
- **Size range +1.1 %: inside the noise.** The size range is mostly `turbo` (K 1, no tx search), where the analysis is
  10-13 % of the GPU time; the gain there is real at kernel level but too small to separate from run-to-run noise.
- **Single JPEG cold process -0.2 % / -0.5 %: inside the noise.** About 1.1-1.3 s per process, mostly CUDA start-up.
"""
key = "Speed sets (files, MP, repeats)"
i = s.index(key)
s = s[:i] + TXT.lstrip("\n") + "\n" + s[i:]
open(p, "w").write(s)
