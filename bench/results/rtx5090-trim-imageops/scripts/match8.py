"""--dump-source planes (4:4:4 8-bit) against ImageMagick's output for the same operation, converted to YCbCr as
av1gpu converts RGB (BT.601 full range).  usage: match.py AV1GPU SRC "IM ops" -- av1gpu options"""
import subprocess, sys, tempfile, os, math
import numpy as np
from PIL import Image
b, src, imops = sys.argv[1], sys.argv[2], sys.argv[3]
opts = sys.argv[sys.argv.index("--") + 1:]
t = tempfile.mkdtemp()
dump = os.path.join(t, "d.yuv")
r = subprocess.run([b, src, os.path.join(t, "o.avif"), "--yuv", "444", "--dump-source", dump, *opts], capture_output=True, text=True)
assert r.returncode == 0, r.stderr
ref = os.path.join(t, "ref.png")
subprocess.run(["magick", src, *imops.split(), "-depth", "8", "PNG24:" + ref], check=True)
im = Image.open(ref)
gray = im.mode == "L"
a = np.asarray(im.convert("RGB"), dtype=np.float64)
h, w = a.shape[:2]
d = np.frombuffer(open(dump, "rb").read(), np.uint8)
mono = d.size == w * h
if not mono and d.size != 3 * w * h:
    print("SIZE MISMATCH", d.size, w, h); sys.exit(1)
R, G, B = a[..., 0], a[..., 1], a[..., 2]
Y = np.clip(np.rint(0.299 * R + 0.587 * G + 0.114 * B), 0, 255)
Cb = np.clip(np.rint(-0.168735892 * R - 0.331264108 * G + 0.5 * B + 128), 0, 255)
Cr = np.clip(np.rint(0.5 * R - 0.418687589 * G - 0.081312411 * B + 128), 0, 255)
planes = [Y] if mono else [Y, Cb, Cr]
out = []
for i, p in enumerate(planes):
    q = d[i * w * h:(i + 1) * w * h].reshape(h, w).astype(np.float64)
    e = q - p
    mse = (e * e).mean()
    out.append("%s mad %.3f psnr %.1f max %d" % ("YUV"[i], np.abs(e).mean(), 10 * math.log10(255 ** 2 / mse) if mse else 99, np.abs(e).max()))
print("%dx%d%s: %s" % (w, h, " mono" if mono else "", " | ".join(out)))
