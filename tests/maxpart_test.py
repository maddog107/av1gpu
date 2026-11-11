#!/usr/bin/env python3
"""Partition options that leave the flat pruning nothing to fall back on: --maxpart 8 / 4, --partN 0 (turbo, faster).

The flat pruning (--prune-flat; turbo, turbo2, turbo3, faster) skips the shapes below 16x16 in flat 16x16 cells and
counts on a 16x16 or larger block to take them over.  With --maxpart below 16 (or --partN without any partition above
8x8) the DP had no partition of finite cost in such a cell and fell back to 4x4 leaves without analysis records, which
k_recon coded from uninitialised device memory: illegal memory accesses, streams that dav1d / aomdec decode
differently from the reconstruction, output that changed from run to run.  The encoder now analyses every block for
such options (GpEncoder::init, dp_complete), and refuses options that leave no partition at all.

Each case is encoded twice through tests/decoder_gate.py, with every device allocation filled with 0 and with 120
(GP_POISON; 120 made v0.12.0 fail to encode the flat image): both runs must pass the gate and give the same bytes.
Inputs (written here): a flat 64x64 frame, a 200x136 frame with flat and textured areas (partial SBs on both edges)
and a flat 72x40 RGBA frame (its alpha item is coded with the main preset).

  tests/maxpart_test.py --bin build/av1gpu
"""
import argparse
import filecmp
import glob
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
CASES = [
    ["--maxpart", "8"],
    ["--maxpart", "8", "--yuv", "444"],
    ["--maxpart", "4"],
    ["--partN", "0"],
    ["--preset", "turbo2", "--maxpart", "8"],
]


def png(path, w, h, px, alpha=False):
    ch = 4 if alpha else 3
    rows = b"".join(b"\0" + b"".join(struct.pack("B" * ch, *px(x, y)) for x in range(w)) for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6 if alpha else 2, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def mixed(x, y):
    if x < 96 or y < 48:
        return (90, 140, 200)                    # flat: pruned cells
    v = (x * 37 + y * 91 + (x * y) % 23) % 256   # texture: every shape analysed
    return (v, (v * 3) % 256, 255 - v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    a = ap.parse_args()
    t = tempfile.mkdtemp(prefix="av1g_maxpart_")
    fails = []
    try:
        img = os.path.join(t, "img")
        os.makedirs(img)
        png(os.path.join(img, "flat_64x64.png"), 64, 64, lambda x, y: (120, 130, 140))
        png(os.path.join(img, "mixed_200x136.png"), 200, 136, mixed)
        png(os.path.join(img, "alpha_72x40.png"), 72, 40, lambda x, y: (200, 60, 30, 255 if x < 40 else 128), alpha=True)
        for k, opts in enumerate(CASES):
            keep = []
            for poison in ("0", "120"):
                d = os.path.join(t, f"c{k}_{poison}")
                env = dict(os.environ, GP_POISON=poison)
                p = subprocess.run([sys.executable, os.path.join(HERE, "decoder_gate.py"), "--bin", a.bin, "--images", img,
                                    "--q", "100", "--jobs", "4", "--keep", d, "--", "--small-preset", "none"] + opts,
                                   capture_output=True, text=True, env=env, timeout=300)
                gate = [l for l in p.stdout.splitlines() if l.startswith("[gate]") or "BAD" in l]
                print(f"{' '.join(opts)} GP_POISON={poison}: " + " | ".join(gate), flush=True)
                if p.returncode:
                    fails.append(f"{' '.join(opts)} GP_POISON={poison}: decoder gate failed\n{p.stdout}{p.stderr}")
                keep.append(d)
            out0 = sorted(glob.glob(os.path.join(keep[0], "q100", "*.avif")))
            out1 = sorted(glob.glob(os.path.join(keep[1], "q100", "*.avif")))
            names = [os.path.basename(f) for f in out0]
            if len(out0) != 3 or names != [os.path.basename(f) for f in out1]:
                fails.append(f"{' '.join(opts)}: outputs {names} / {[os.path.basename(f) for f in out1]}, expected 3 each")
            else:
                diff = [n for f, g, n in zip(out0, out1, names) if not filecmp.cmp(f, g, shallow=False)]
                if diff:
                    fails.append(f"{' '.join(opts)}: output depends on uninitialised memory (GP_POISON 0 / 120): {diff}")
        # options that leave no partition the analysis evaluates (turbo2 has no 4x4: --maxpart 4 forces 4x4 leaves)
        p = subprocess.run([a.bin, os.path.join(img, "flat_64x64.png"), os.path.join(t, "x.avif"), "--preset", "turbo2",
                            "--maxpart", "4", "--small-preset", "none"], capture_output=True, text=True, timeout=300)
        print(f"--preset turbo2 --maxpart 4: exit {p.returncode}: {p.stderr.strip()}", flush=True)
        if p.returncode == 0 or "no partition" not in p.stderr:
            fails.append(f"--preset turbo2 --maxpart 4 must be refused (exit {p.returncode}): {p.stderr}")
    finally:
        shutil.rmtree(t, ignore_errors=True)
    for f in fails:
        print("FAIL", f)
    print("maxpart_test:", "FAIL" if fails else "PASS")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
