"""--dump-source planes (4:4:4, 10-bit) against ImageMagick at 16 bits, converted as av1gpu converts RGB.
  usage: match16.py WORKDIR AV1GPU "IM ops" "av1gpu options" SRC.png (a 16-bit PNG; WxH read from it)"""
import subprocess, numpy as np, sys
S, b, imop, avop = sys.argv[1], sys.argv[2], sys.argv[3].split(), sys.argv[4].split()
src = sys.argv[5]
subprocess.run([b, src, S + "/o.avif", "--yuv", "444", "--depth", "10", *avop, "--dump-source", S + "/d.yuv"], check=True, capture_output=True)
W, H = (int(v) for v in subprocess.run(["magick", "identify", "-format", "%w %h", src], capture_output=True, text=True).stdout.split())
raw = subprocess.run(["magick", src, *imop, "-depth", "16", "-endian", "MSB", "rgb:-"], capture_output=True).stdout
a = np.frombuffer(raw, ">u2").astype(np.float64).reshape(H, W, 3) / 65535
R, G, B = a[..., 0], a[..., 1], a[..., 2]
ref = [np.rint((0.299 * R + 0.587 * G + 0.114 * B) * 1023), np.rint((-0.168735892 * R - 0.331264108 * G + 0.5 * B) * 1023 + 512),
       np.rint((0.5 * R - 0.418687589 * G - 0.081312411 * B) * 1023 + 512)]
d = np.frombuffer(open(S + "/d.yuv", "rb").read(), "<u2").astype(np.float64)
n = W * H
print(" | ".join("%s mad %.3f max %d" % ("YUV"[i], np.abs(d[i * n:(i + 1) * n].reshape(H, W) - ref[i]).mean(),
                                         np.abs(d[i * n:(i + 1) * n].reshape(H, W) - ref[i]).max()) for i in range(3)))
