#!/usr/bin/env python3
"""Derived image sets for the 10-bit gates (run on the instance):
  ~/images_png16   16-bit RGB PNGs: 30 bench JPEGs through a float pipeline (decode, 2x2 box downscale in float,
                   x 65535 / 255, round to 16 bits) + 6 smooth synthetic gradients (banding stress: narrow ranges,
                   radial / diagonal / two-colour ramps, one with odd size)
  ~/images_png     8-bit PNG / WebP from the bench JPEGs (8-bit-source CPU path)
  ~/images_large   the gamut set's large + huge classes (symlinks)"""
import glob
import os
import struct
import zlib

import numpy as np
from PIL import Image, features

H = os.path.expanduser("~")


def write_png16(path, rgb):
    """rgb: HxWx3 uint16 -> 16-bit RGB PNG (filter 0 rows)"""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].astype(">u2").tobytes() for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 16, 2, 0, 0, 0)) + \
        chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")
    open(path, "wb").write(png)


os.makedirs(f"{H}/images_png16", exist_ok=True)
os.makedirs(f"{H}/images_png", exist_ok=True)
os.makedirs(f"{H}/images_large", exist_ok=True)
src = sorted(glob.glob(f"{H}/images/*.jpg"))
for i, f in enumerate(src):
    b = os.path.basename(f)[:-4]
    im = Image.open(f)
    im = im.convert("RGB") if im.mode != "RGB" else im
    if i % 3 == 0 and i < 90:   # 30 images
        a = np.asarray(im, dtype=np.float64)
        hh, ww = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
        a = a[:hh, :ww].reshape(hh // 2, 2, ww // 2, 2, 3).mean(axis=(1, 3))
        write_png16(f"{H}/images_png16/{b}_f16.png", np.clip(np.rint(a * (65535.0 / 255.0)), 0, 65535).astype(np.uint16))
    if i % 3 == 1:
        im.save(f"{H}/images_png/{b}.png", compress_level=3)
    if i % 10 == 2 and features.check("webp"):
        im.save(f"{H}/images_png/{b}.webp", quality=95)
# synthetic gradients (16-bit precision, values in [0, 1])
yy, xx = np.mgrid[0:768, 0:1024].astype(np.float64)
grads = {
    "grad_h_dark": np.stack([0.10 + 0.05 * xx / 1023] * 3, -1),                      # 12.75 8-bit codes over 1024 px
    "grad_v_sky": np.stack([0.35 + 0.10 * yy / 767, 0.55 + 0.12 * yy / 767, 0.80 + 0.08 * yy / 767], -1),
    "grad_radial": np.stack([0.5 + 0.08 * np.hypot(xx - 512, yy - 384) / 640] * 3, -1),
    "grad_diag_2col": np.stack([0.2 + 0.3 * (xx + yy) / 1790, 0.25 + 0.05 * (xx + yy) / 1790, 0.6 - 0.2 * (xx + yy) / 1790], -1),
    "grad_sine": np.stack([0.5 + 0.03 * np.sin(xx / 97.0) * np.cos(yy / 71.0)] * 3, -1),
}
for k, v in grads.items():
    write_png16(f"{H}/images_png16/{k}.png", np.clip(np.rint(v * 65535), 0, 65535).astype(np.uint16))
yo, xo = np.mgrid[0:501, 0:777].astype(np.float64)   # odd size
write_png16(f"{H}/images_png16/grad_odd.png",
            np.clip(np.rint(np.stack([0.3 + 0.1 * xo / 776, 0.3 + 0.1 * yo / 500, 0.32 + 0.02 * (xo + yo) / 1276], -1) * 65535),
                    0, 65535).astype(np.uint16))
for f in glob.glob(f"{H}/images_gamut/*_large.jpg") + glob.glob(f"{H}/images_gamut/*_huge.jpg"):
    d = f"{H}/images_large/{os.path.basename(f)}"
    if not os.path.exists(d):
        os.symlink(f, d)
print("png16", len(os.listdir(f"{H}/images_png16")), "png/webp", len(os.listdir(f"{H}/images_png")),
      "large", len(os.listdir(f"{H}/images_large")))
