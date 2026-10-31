"""NVENC odd heights: PSNR per horizontal band (top..bottom) at row offsets 0 and 1, luma and chroma separately (YCbCr
from Pillow), to tell a stretch from a shift or a chroma misplacement.  nvenc_shift2.py IMAGE..."""
import os, subprocess, sys, tempfile
import numpy as np
from PIL import Image
for im in sys.argv[1:]:
    s = Image.open(os.path.expanduser(f"~/images_gamut/{im}")).convert("RGB")
    with tempfile.TemporaryDirectory() as td:
        subprocess.run([os.path.expanduser("~/opt/bin/avifdec"), "-c", "dav1d", os.path.expanduser(f"~/nv/q/{im}.q18.avif"),
                        f"{td}/d.png"], check=True, capture_output=True)
        d = Image.open(f"{td}/d.png").convert("RGB")
    S = np.asarray(s.convert("YCbCr")).astype(float); D = np.asarray(d.convert("YCbCr")).astype(float)
    h = S.shape[0]
    print(im)
    for name, ch in (("Y", 0), ("Cb", 1), ("Cr", 2)):
        cells = []
        for dy in (0, 1):
            row = []
            for b in range(4):
                y0, y1 = b * h // 4, (b + 1) * h // 4 - 1
                mse = ((D[y0 + dy:y1 + dy, :S.shape[1], ch] - S[y0:y1, :, ch]) ** 2).mean()
                row.append(f"{10 * np.log10(255 ** 2 / max(mse, 1e-9)):.1f}")
            cells.append(f"off{dy} " + "/".join(row))
        print(f"  {name}: " + "   ".join(cells))
