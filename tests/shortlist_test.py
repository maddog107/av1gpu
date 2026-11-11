#!/usr/bin/env python3
"""--uv-shortlist / --tx-shortlist (k_leafprep's open-loop shortlists for the closed loop): the shortlists rank well, the
streams decode, the bytes are deterministic and independent of the batch, and the option values are checked.

Inputs: the images of --images and five PNGs this script writes: a 1280 x 960 scene (gradients, edges at several angles,
texture, text-like blocks) and a 640 x 480 photo-like one (smooth colour fields with fine texture), both above
--small-mp so the turbo preset runs on them; a 333 x 197 RGB image (odd size), a 160 x 120 RGBA one (lossy alpha frame)
and a 259 x 131 grey one.
- Ranking: total bytes at qindex 120 with --uv-shortlist 3, --tx-shortlist 4 and both, against the run without them,
  must stay within MAX_GROWTH.  A ranking that keeps the worst candidates instead of the best (both rankings negated in
  k_leafprep.cu) exceeds it for each switch on its own.
- Decoding: with both switches on, dav1d and aomdec reproduce the reconstruction (tests/decoder_gate.py's check) at the
  defaults, with forced 2 x 2 tiles, with the base preset, in 10-bit 4:4:4 and in monochrome (--grayscale).
- Determinism: a second run, one image per GPU batch with one encoder, and the list reversed with two encoders give the
  same bytes; --check (reconstruction = av1core's decode of the record) and --check-entropy pass.
- --tx-shortlist 7 gives the bytes of the run without it (no intra tx set has more than 7 types).
- Out-of-range and non-numeric values of both options are refused.

  tests/shortlist_test.py --bin build/av1gpu --images tests/gate_images
"""
import argparse
import glob
import hashlib
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import decoder_gate  # noqa: E402

BOTH = ["--uv-shortlist", "3", "--tx-shortlist", "4"]
# total bytes at qindex 120 against no shortlist: the most each setting may add.  On an L4: the correct build +0.06 % (uv),
# +0.06 % (tx), +0.19 % (both); with both rankings negated +14.6 %, +4.0 %, +18.5 %
MAX_GROWTH = {"uv": 0.010, "tx": 0.005, "both": 0.010}
GATES = [("defaults", []), ("tiles", ["--tiles", "0.01,1,1", "--small-preset", "none"]),
         ("base", ["--preset", "base", "--small-preset", "none"]), ("444_10", ["--yuv", "444", "--depth", "10"]),
         ("mono", ["--grayscale", "Rec709Luma"])]


def png(path, w, h, row, mode="rgb"):
    """PNG from row(y) -> bytes of one row; mode rgb, rgba or grey (8 bits)"""
    raw = b"".join(b"\0" + row(y) for y in range(h))
    ct = {"rgb": 2, "rgba": 6, "grey": 0}[mode]
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, ct, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b""))


def scene_row(y, w=1280):
    out = bytearray()
    for x in range(w):
        if y < 320:     # smooth gradients
            p = (x * 255 // (w - 1), y * 255 // 319, (x + y) * 255 // (w + 318))
        elif y < 640:   # edges at several angles, by vertical band
            band = x // 160
            d = (x * (band + 1) + y * (8 - band)) // 9
            v = 230 if (d // 23) % 2 else 30
            p = (v, (v + band * 25) % 256, 255 - v)
        else:           # texture (a fixed pseudo-random pattern) with text-like blocks
            h = (x * 73856093) ^ (y * 19349663)
            t = (h >> 7) & 63
            ink = (y // 12) % 3 == 0 and (x // 9) % 6 and x > w // 2
            p = (20, 20, 25) if ink else (120 + t, 110 + (t >> 1), 90 + (t >> 2))
        out += bytes(p)
    return bytes(out)


def photo_row(y, w=640):
    out = bytearray()
    for x in range(w):
        a = math.sin(x * 0.013 + y * 0.007) + math.sin(x * 0.031 - y * 0.023) * 0.6 + math.sin((x + 2 * y) * 0.0041) * 1.4
        b = math.sin(x * 0.009 - y * 0.017 + 1.3) + math.sin(math.hypot(x - 300, y - 200) * 0.05) * 0.7
        n = (((x * 1103515245 + y * 12345) ^ (y * 2654435761)) >> 9) & 15
        r = 128 + 50 * a + 25 * b + n - 8
        g = 120 + 45 * b - 20 * a + n - 8
        bl = 110 + 40 * math.sin(a + b) + 30 * math.sin(x * 0.05) * math.sin(y * 0.045) + n - 8
        out += bytes(max(0, min(255, int(v))) for v in (r, g, bl))
    return bytes(out)


def run(binary, args, env=None):
    e = dict(os.environ)
    e.update(env or {})
    return subprocess.run([binary] + args, capture_output=True, text=True, env=e, timeout=600)


def batch(binary, lst, out, opts, env=None):
    """{output name: sha256} of the AVIFs, or None"""
    shutil.rmtree(out, ignore_errors=True)
    p = run(binary, ["--batch", lst, "--outdir", out] + opts, env)
    if p.returncode:
        sys.stdout.write(p.stderr[-2000:])
        return None
    return {f: hashlib.sha256(open(os.path.join(out, f), "rb").read()).hexdigest() for f in sorted(os.listdir(out)) if f.endswith(".avif")}


def total(out):
    return sum(os.path.getsize(f) for f in glob.glob(os.path.join(out, "*.avif")))


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
        gen = [("scene_1280x960.png", 1280, 960, scene_row, "rgb"), ("photo_640x480.png", 640, 480, photo_row, "rgb"),
               ("odd_333x197.png", 333, 197, lambda y: scene_row(y + 300, 333)[:333 * 3], "rgb"),
               ("rgba_160x120.png", 160, 120, lambda y: b"".join(bytes((200, (x * 3) % 256, 40, (x + y) * 255 // 278)) for x in range(160)), "rgba"),
               ("grey_259x131.png", 259, 131, lambda y: bytes(((x * 7 + y * 3) ^ (x * y)) & 255 for x in range(259)), "grey")]
        for name, w, h, row, mode in gen:
            png(os.path.join(t, name), w, h, row, mode)
        files = sorted(glob.glob(os.path.join(a.images, "*"))) + [os.path.join(t, g[0]) for g in gen]
        lst, rev = os.path.join(t, "list.txt"), os.path.join(t, "rev.txt")
        open(lst, "w").write("".join(f + "\n" for f in files))
        open(rev, "w").write("".join(f + "\n" for f in reversed(files)))
        q = ["--qindex", "120"]

        # ---- ranking: bytes against no shortlist, each switch alone and both
        off = batch(a.bin, lst, os.path.join(t, "off"), q)
        check(off is not None and len(off) == len(files), f"no shortlist: {len(off or {})} outputs")
        b0 = total(os.path.join(t, "off"))
        for key, opts in (("uv", ["--uv-shortlist", "3"]), ("tx", ["--tx-shortlist", "4"]), ("both", BOTH)):
            d = os.path.join(t, "r_" + key)
            got = batch(a.bin, lst, d, q + opts + (["--debug-outputs"] if key == "both" else []))
            g = total(d) / b0 - 1 if got and b0 else float("inf")
            check(got is not None and g <= MAX_GROWTH[key],
                  f"{' '.join(opts)}: bytes {g:+.3%} against no shortlist (at most {MAX_GROWTH[key]:+.1%}); "
                  f"{sum(got[k] != off.get(k) for k in got) if got else 0} of {len(files)} files differ")

        # ---- decoding: both switches, several settings (the "both" run above is the defaults' one)
        for name, opts in GATES:
            d = os.path.join(t, "r_both") if name == "defaults" else os.path.join(t, "g_" + name)
            if name != "defaults" and batch(a.bin, lst, d, q + BOTH + opts + ["--debug-outputs"]) is None:
                check(False, f"gate {name}: encode failed")
                continue
            res = [decoder_gate.check_one(f, d) for f in sorted(glob.glob(os.path.join(d, "*.avif")))]
            bad = [r for r in res if not decoder_gate.ok_result(r)]
            fm = sorted({f"{r.get('format')} {r.get('depth')}-bit" for r in res})
            check(len(res) == len(files) and not bad,
                  f"gate {name}: {len(res) - len(bad)}/{len(res)} identical in dav1d and aomdec ({', '.join(fm)}; "
                  f"{sum(1 for r in res if r.get('alpha'))} with alpha)" + (f"; {bad[0]}" if bad else ""))

        # ---- determinism, batch independence
        ref = batch(a.bin, lst, os.path.join(t, "d0"), q + BOTH)
        check(ref is not None and batch(a.bin, lst, os.path.join(t, "d1"), q + BOTH) == ref, "both: same bytes in a second run")
        check(batch(a.bin, lst, os.path.join(t, "d2"), q + BOTH, {"GP_ENCODE_MP": "0.01", "GP_ENCODERS": "1"}) == ref,
              "both: same bytes with one image per batch, one encoder")
        check(batch(a.bin, rev, os.path.join(t, "d3"), q + BOTH, {"GP_ENCODERS": "2"}) == ref,
              "both: same bytes with the list reversed, two encoders")
        img = os.path.join(t, "scene_1280x960.png")
        p = run(a.bin, [img, os.path.join(t, "c.avif"), "--check"] + BOTH)
        check(p.returncode == 0 and "recon identical to av1core" in p.stderr, f"both: --check: {p.stderr.strip()[-120:]}")
        p = run(a.bin, [img, os.path.join(t, "e.avif"), "--check-entropy"] + BOTH)
        check(p.returncode == 0 and "device entropy == CPU writer" in p.stderr, f"both: --check-entropy: {p.stderr.strip()[-120:]}")

        # ---- --tx-shortlist 7 = no shortlist
        check(batch(a.bin, lst, os.path.join(t, "tx7"), q + ["--tx-shortlist", "7"]) == off,
              "--tx-shortlist 7: the bytes of the run without a shortlist")

        # ---- option values
        for opt, v in (("--uv-shortlist", "7"), ("--uv-shortlist", "-3"), ("--uv-shortlist", "abc"), ("--tx-shortlist", "17"),
                       ("--tx-shortlist", "4x"), ("--tx-shortlist", "")):
            p = run(a.bin, [img, os.path.join(t, "v.avif"), opt, v])
            check(p.returncode != 0 and opt in p.stderr and not os.path.exists(os.path.join(t, "v.avif")),
                  f"{opt} '{v}' refused (exit {p.returncode})")
    finally:
        shutil.rmtree(t, ignore_errors=True)
    print("PASS" if not fails else f"FAIL: {len(fails)} checks")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
