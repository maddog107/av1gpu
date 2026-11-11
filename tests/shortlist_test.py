#!/usr/bin/env python3
"""--uv-shortlist / --tx-shortlist (k_leafprep's open-loop shortlists for the closed loop): deterministic, independent of
the batch, and with a tx shortlist no shorter than every tx set the same bytes as without one.

Inputs: the images of --images and two PNGs this script writes (1280 x 960 with gradients, edges at several angles, fine
texture and text-like blocks, so the turbo preset runs on it and not --small-preset's base; and a 160 x 120 RGBA one,
whose lossy alpha frame gets the luma shortlist too).  For each option set below, batch mode over all inputs:
- a second run gives the same bytes, and so do one image per GPU batch (GP_ENCODE_MP=0.01) with one encoder and the
  list in reverse order with two encoders (batch composition, encoder count);
- the options are on: some output differs from the run without them;
- --check on one image: the reconstruction equals av1core's decode of the record; --check-entropy: the device entropy
  path writes the CPU writer's bytes.
And --tx-shortlist 7 (every intra tx set has at most 7 types: no list removes a type) gives the bytes of the run without
it.  Decoder identity (dav1d + aomdec) of shortlist outputs: the gate_shortlist tests (tests/decoder_gate.py).

  tests/shortlist_test.py --bin build/av1gpu --images tests/gate_images
"""
import argparse
import glob
import hashlib
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

SETS = [
    ["--uv-shortlist", "3", "--tx-shortlist", "4"],
    ["--uv-shortlist", "3", "--tx-shortlist", "4", "--yuv", "444", "--depth", "10"],
    ["--uv-shortlist", "1", "--preset", "base", "--small-preset", "none"],
    ["--tx-shortlist", "1", "--preset", "fast"],
]


def png(path, w, h, row, alpha=False):
    """PNG from row(y) -> bytes of one row (RGB or RGBA, 8 bits)"""
    raw = b"".join(b"\0" + row(y) for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6 if alpha else 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def scene_row(y):
    out = bytearray()
    for x in range(1280):
        if y < 320:     # smooth gradients
            p = (x * 255 // 1279, y * 255 // 319, (x + y) * 255 // 1598)
        elif y < 640:   # edges at several angles, by vertical band
            band = x // 160
            d = (x * (band + 1) + y * (8 - band)) // 9
            v = 230 if (d // 23) % 2 else 30
            p = (v, (v + band * 25) % 256, 255 - v)
        else:           # texture (a fixed pseudo-random pattern) with text-like blocks
            h = (x * 73856093) ^ (y * 19349663)
            t = (h >> 7) & 63
            ink = (y // 12) % 3 == 0 and (x // 9) % 6 and x > 640
            p = (20, 20, 25) if ink else (120 + t, 110 + (t >> 1), 90 + (t >> 2))
        out += bytes(p)
    return bytes(out)


def rgba_row(y):
    return b"".join(bytes((200, (x * 3) % 256, 40, (x + y) * 255 // 278)) for x in range(160))


def run(binary, args, env=None):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run([binary] + args, capture_output=True, text=True, env=e, timeout=600)


def batch(binary, lst, out, opts, env=None):
    shutil.rmtree(out, ignore_errors=True)
    p = run(binary, ["--batch", lst, "--outdir", out] + opts, env)
    if p.returncode:
        sys.stdout.write(p.stderr[-2000:])
    return {f: hashlib.sha256(open(os.path.join(out, f), "rb").read()).hexdigest() for f in sorted(os.listdir(out))} \
        if p.returncode == 0 and os.path.isdir(out) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--images", required=True)
    a = ap.parse_args()
    t = tempfile.mkdtemp(prefix="av1g_shortlist_")
    fails = []

    def check(ok, what):
        print(f"{'ok  ' if ok else 'FAIL'} {what}", flush=True)
        if not ok:
            fails.append(what)

    try:
        src = sorted(glob.glob(os.path.join(a.images, "*")))
        png(os.path.join(t, "scene_1280x960.png"), 1280, 960, scene_row)
        png(os.path.join(t, "rgba_160x120.png"), 160, 120, rgba_row, alpha=True)
        files = src + [os.path.join(t, "scene_1280x960.png"), os.path.join(t, "rgba_160x120.png")]
        lst, rev = os.path.join(t, "list.txt"), os.path.join(t, "rev.txt")
        open(lst, "w").write("".join(f + "\n" for f in files))
        open(rev, "w").write("".join(f + "\n" for f in reversed(files)))
        for opts in SETS:
            name = " ".join(opts)
            base = [o for i, o in enumerate(opts) if not (o.endswith("-shortlist") or (i and opts[i - 1].endswith("-shortlist")))]
            off = batch(a.bin, lst, os.path.join(t, "off"), base)
            on = batch(a.bin, lst, os.path.join(t, "on"), opts)
            check(on is not None and off is not None and len(on) == len(files), f"{name}: {len(on or {})} outputs")
            if on is None or off is None:
                continue
            check(any(on[k] != off.get(k) for k in on), f"{name}: changes the output ({sum(on[k] != off.get(k) for k in on)} files)")
            check(batch(a.bin, lst, os.path.join(t, "again"), opts) == on, f"{name}: same bytes in a second run")
            check(batch(a.bin, lst, os.path.join(t, "one"), opts, {"GP_ENCODE_MP": "0.01", "GP_ENCODERS": "1"}) == on,
                  f"{name}: same bytes with one image per batch, one encoder")
            check(batch(a.bin, rev, os.path.join(t, "rev"), opts, {"GP_ENCODERS": "2"}) == on,
                  f"{name}: same bytes with the list reversed, two encoders")
            img = os.path.join(t, "scene_1280x960.png")
            p = run(a.bin, [img, os.path.join(t, "c.avif"), "--check"] + opts)
            check(p.returncode == 0 and "recon identical to av1core" in p.stderr, f"{name}: --check: {p.stderr.strip()[-160:]}")
            p = run(a.bin, [img, os.path.join(t, "e.avif"), "--check-entropy"] + opts)
            check(p.returncode == 0 and "device entropy == CPU writer" in p.stderr, f"{name}: --check-entropy: {p.stderr.strip()[-160:]}")
        off = batch(a.bin, lst, os.path.join(t, "off"), [])
        check(off is not None and batch(a.bin, lst, os.path.join(t, "tx7"), ["--tx-shortlist", "7"]) == off,
              "--tx-shortlist 7: the bytes of the run without a shortlist")
    finally:
        shutil.rmtree(t, ignore_errors=True)
    print("PASS" if not fails else f"FAIL: {len(fails)} checks")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
