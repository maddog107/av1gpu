#!/usr/bin/env python3
"""The opt-in analysis switches (--topdown, --prune-41): deterministic, independent of the
batch, and decodable.

On generated PNGs (a 1.1 MP picture that the default encodes with its turbo preset, frame edges that cut 16x16 cells,
flat areas next to texture; a 640x480 and a 333x250 one for the base preset) plus tests/gate_images: every switch
alone and both, with the default settings and with --preset base, must give the same bytes with three encoders
and the list in order as with one encoder and the list reversed; the switches set to their off values must give the
bytes of a run without them; both together must change the output; then tests/decoder_gate.py (dav1d + aomdec reproduce the
encoder's reconstruction) on these pictures with both switches, both presets.

  tests/staged_analysis_test.py --bin build/av1gpu [--keep DIR]
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

HERE = os.path.dirname(os.path.abspath(__file__))
SWITCHES = {"topdown": ["--topdown", "1"], "prune41": ["--prune-41", "10"]}
ALL = sum(SWITCHES.values(), [])
OFF = ["--topdown", "0", "--prune-41", "-1"]


def png(path, w, h, px):
    rows = b"".join(b"\0" + b"".join(struct.pack("BBB", *px(x, y)) for x in range(w)) for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(rows, 6)) + chunk(b"IEND", b""))


def picture(x, y):
    """flat sky, a gradient, fine and coarse texture, diagonal edges: every shape group gets something to do"""
    n = ((x * 7919 + y * 104729) ^ (x * y)) % 61
    if y < 180:
        return (120, 160, 220)
    if x < 300:
        return ((x + y) % 256, (2 * x) % 256, (3 * y) % 256)
    if (x // 23 + y // 17) % 2:
        return (40 + n, 90 + n // 2, 30)
    v = 128 + ((x - y) % 64) - 32 + n // 4
    return (v, v, 255 - v)


def run(binary, args, env=None):
    e = dict(os.environ)
    e.update(env or {})
    p = subprocess.run([binary] + args, capture_output=True, text=True, env=e, timeout=600)
    return p.returncode, p.stderr


def hashes(d):
    return {os.path.basename(f): hashlib.sha256(open(f, "rb").read()).hexdigest() for f in sorted(glob.glob(os.path.join(d, "*.avif")))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--keep")
    a = ap.parse_args()
    t = a.keep or tempfile.mkdtemp(prefix="av1g_staged_")
    os.makedirs(t, exist_ok=True)
    fails = []
    try:
        src = os.path.join(t, "in")
        os.makedirs(src, exist_ok=True)
        for name, w, h in (("big_1100x1000.png", 1100, 1000), ("mid_640x480.png", 640, 480), ("odd_333x250.png", 333, 250)):
            if not os.path.exists(os.path.join(src, name)):
                png(os.path.join(src, name), w, h, picture)
        for f in glob.glob(os.path.join(HERE, "gate_images", "*")):
            shutil.copy(f, src)
        files = sorted(glob.glob(os.path.join(src, "*.png")))
        fwd, rev = os.path.join(t, "fwd.lst"), os.path.join(t, "rev.lst")
        open(fwd, "w").write("".join(f + "\n" for f in files))
        open(rev, "w").write("".join(f + "\n" for f in reversed(files)))
        for preset in ([], ["--preset", "base"]):
            pname = "base" if preset else "default"
            base = {}
            for name, opts in [("none", []), ("off", OFF)] + list(SWITCHES.items()) + [("all", ALL)]:
                out = {}
                for how, lst, env in (("enc3", fwd, {"GP_ENCODERS": "3"}), ("enc1rev", rev, {"GP_ENCODERS": "1"})):
                    od = os.path.join(t, f"{pname}_{name}_{how}")
                    shutil.rmtree(od, ignore_errors=True)
                    rc, err = run(a.bin, ["--batch", lst, "--outdir", od] + preset + opts, env)
                    out[how] = hashes(od)
                    if rc or len(out[how]) != len(files):
                        fails.append(f"{pname} {name} {how}: exit {rc}, {len(out[how])} of {len(files)} outputs: {err[-300:]}")
                if out["enc3"] != out["enc1rev"]:
                    diff = [k for k in out["enc3"] if out["enc3"].get(k) != out["enc1rev"].get(k)]
                    fails.append(f"{pname} {name}: three encoders in order and one encoder reversed differ on {diff}")
                base[name] = out["enc3"]
                print(f"{pname} {name}: {len(out['enc3'])} outputs, deterministic: {out['enc3'] == out['enc1rev']}", flush=True)
            if base["off"] != base["none"]:
                fails.append(f"{pname}: the switches at their off values change the output")
            if base["all"] == base["none"]:
                fails.append(f"{pname}: both switches together leave every output unchanged")
            g = subprocess.run([sys.executable, os.path.join(HERE, "decoder_gate.py"), "--bin", a.bin, "--images", src,
                                "--q", "60,120", "--keep", os.path.join(t, f"gate_{pname}"), "--"] + preset + ALL,
                               capture_output=True, text=True, timeout=1800)
            print(g.stdout.strip().splitlines()[-1] if g.stdout.strip() else g.stderr[-300:], flush=True)
            if g.returncode:
                fails.append(f"{pname}: decoder gate failed:\n{g.stdout[-2000:]}{g.stderr[-1000:]}")
    finally:
        if not a.keep:
            shutil.rmtree(t, ignore_errors=True)
    for f in fails:
        print("FAIL:", f)
    print("PASS" if not fails else f"FAIL ({len(fails)})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
