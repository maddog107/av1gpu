#!/usr/bin/env python3
# cases.tmpl -> cases.txt with real file names (run in ~/data after mkdata.py)
import glob, os
pick = {
    "JPG0": max(glob.glob("jpg/*.jpg"), key=os.path.getsize),
    "png40/PNG0": sorted(glob.glob("png40/*.png"))[0],
    "mix/WEBP0": sorted(glob.glob("mix/*.webp"))[0],
    "alpha/RGBA0": sorted(glob.glob("alpha/*.png"))[0],
    "few/PNGF0": sorted(glob.glob("few/*.png"))[0],
}
s = open(os.path.expanduser("~/test/cases.tmpl")).read()
for k, v in pick.items():
    s = s.replace(k, v)
open("cases.txt", "w", newline="\n").write(s)
print(s)
