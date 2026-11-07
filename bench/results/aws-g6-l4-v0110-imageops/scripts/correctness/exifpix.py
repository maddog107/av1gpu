#!/usr/bin/env python3
"""EXIF orientation 1..8 with trim / resize / tone / gray: the displayed picture (decoded, irot / imir applied) against the upright picture processed by magick -auto-orient ..."""
import os, subprocess, sys, tempfile, shutil
import numpy as np
from PIL import Image, ImageOps
sys.path.insert(0, os.path.expanduser("~/t"))
import avt
H = os.path.expanduser("~"); BIN = os.environ.get("BIN", f"{H}/pr/build/av1gpu"); MAG = f"{H}/im/bin/magick"
env = dict(os.environ, LD_LIBRARY_PATH=f"{H}/im/lib")
T = tempfile.mkdtemp(prefix="exp_", dir="/dev/shm")
cases = [
 ([], []),
 (["--trim", "--fuzz", "10%"], ["-fuzz", "10%", "-trim", "+repage"]),
 (["--trim", "--fuzz", "10%", "--resize", "100x"], ["-fuzz", "10%", "-trim", "+repage", "-resize", "100x"]),
 (["--trim", "--fuzz", "10%", "--resize", "x60"], ["-fuzz", "10%", "-trim", "+repage", "-resize", "x60"]),
 (["--trim", "--fuzz", "10%", "--resize", "70x40!"], ["-fuzz", "10%", "-trim", "+repage", "-resize", "70x40!"]),
 (["--trim", "--fuzz", "10%", "--resize", "90x50"], ["-fuzz", "10%", "-trim", "+repage", "-resize", "90x50"]),
 (["--trim", "--fuzz", "10%", "--resize", "90x50^"], ["-fuzz", "10%", "-trim", "+repage", "-resize", "90x50^"]),
 (["--trim", "--fuzz", "10%", "--resize", "@2000"], ["-fuzz", "10%", "-trim", "+repage", "-resize", "@2000"]),
 (["--trim", "--fuzz", "10%", "--max-size", "90x50"], ["-fuzz", "10%", "-trim", "+repage", "-resize", "90x50>"]),
 (["--negate"], ["-negate"]),
 (["--grayscale", "Rec709Luma", "--resize", "200x"], ["-resize", "200x", "-grayscale", "Rec709Luma"]),
 (["--auto-level", "--gamma", "1.3"], ["-auto-level", "-gamma", "1.3"]),
]
bad = 0; n = 0
for w, h in [(160, 96), (161, 97)]:
  for ss in ("444", "420"):
    for o in range(1, 9):
      f = f"{H}/adv/jpeg/ex{o}_{ss}_{w}x{h}.jpg"
      for av, im in cases:
        out = f"{T}/o.avif"; m = f"{T}/m.png"
        p = subprocess.run([BIN, f, out, "--yuv", "444", "--small-preset", "none", "--qindex", "20"] + av, capture_output=True, text=True, timeout=60)
        r = subprocess.run([MAG, f, "-auto-orient"] + im + ["-depth", "8", f"png:{m}"], capture_output=True, text=True, env=env)
        n += 1
        if p.returncode: print("rc", p.returncode, f, av, p.stderr[-100:]); bad += 1; continue
        c = avt.check(out, T)
        rgb = avt.to_rgb(c["si"], c["planes"]) * 255
        d = avt.apply_transforms(rgb, c["irot"], c["imir"])
        a = np.array(Image.open(m).convert("RGB")).astype(float)
        if d.shape != a.shape:
            print("SIZE", os.path.basename(f), " ".join(av), d.shape[:2], a.shape[:2]); bad += 1; continue
        e = np.abs(d - a).mean()
        if e > 6:
            print(f"PIX {os.path.basename(f)} {' '.join(av)}: mean diff {e:.1f}"); bad += 1
print(f"{n - bad}/{n} displayed pictures match magick -auto-orient")
shutil.rmtree(T, ignore_errors=True)
