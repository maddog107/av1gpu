#!/usr/bin/env python3
"""Partition options that leave the flat / variance pruning nothing to fall back on, and k_dp's safety net.

The flat pruning (--prune-flat; turbo, turbo2, turbo3, faster) skips the shapes below 16x16 in flat 16x16 cells and
counts on a 16x16 or larger block to take them over; --prune-var skips blocks of --prune-area pixels and up whose
variance is high and counts on smaller ones.  With --maxpart below 16 (or --partN without NONE, a small --prune-area,
...) the DP had no partition of finite cost there and fell back to 4x4 leaves without analysis records, which k_recon
coded from uninitialised device memory: illegal memory accesses, streams that dav1d / aomdec decode differently from
the reconstruction, output that changed from run to run.  The encoder now analyses every block for such options
(GpEncoder::init, dp_complete), refuses options that leave no partition at all, and k_dp gives any leaf still
without a record a DC_PRED record (counted, reported, and a decoder-gate failure) or fails the frame.

- Every case in CASES is encoded twice through tests/decoder_gate.py, with every device allocation filled with 0 and
  with 120 (GP_POISON; 120 made v0.12.0 fail to encode the flat image): both runs must pass the gate (which also fails
  on any block coded by k_dp's safety net) and give the same bytes.  The cases cover interior superblocks (--maxpart),
  the frame-edge-only route (--partN 0x3FE: flat 16x16 / 48x48 frames) and the variance pruning.
- GP_DP_UNCHECKED=1 keeps the pruning that init would switch off: with turbo --maxpart 8 the safety net must act
  (reported, decodable, the same bytes for both fillings); with turbo2 (no 4x4 shape) the frame must fail cleanly.
- Options without any partition are refused at start, also for the small-image preset of a frame that is not small.
Inputs (written here): flat 64x64, 16x16 and 48x48 frames, a 200x136 frame with flat and textured areas (partial SBs on
both edges) and a flat 72x40 RGBA frame (its alpha item is coded with the main preset).

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
    ["--preset", "turbo", "--partN", "0x3FE"],                       # only frame edges at 16 / 48 px are exposed
    ["--prune-var", "0.0001", "--prune-area", "16", "--small-kernel", "0"],
    ["--preset", "base", "--prune-var", "0.5", "--prune-area", "64", "--prune-flat", "2"],
]
NET_CASE = ["--maxpart", "8"]   # with GP_DP_UNCHECKED=1: turbo has the 4x4 shape, the net gives records
IMAGES = 5


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


def gate_twice(a, img, t, tag, opts, fails, env_extra=None, allow_net=False):
    """The decoder gate with GP_POISON 0 and 120; returns the gate output of both runs."""
    keep, outs = [], []
    for poison in ("0", "120"):
        d = os.path.join(t, f"{tag}_{poison}")
        env = dict(os.environ, GP_POISON=poison, **(env_extra or {}))
        p = subprocess.run([sys.executable, os.path.join(HERE, "decoder_gate.py"), "--bin", a.bin, "--images", img,
                            "--q", "100", "--jobs", "4", "--keep", d] + (["--allow-dp-net"] if allow_net else [])
                           + ["--", "--small-preset", "none"] + opts,
                           capture_output=True, text=True, env=env, timeout=300)
        gate = [l for l in p.stdout.splitlines() if l.startswith("[gate]") or "BAD" in l or "dp safety net" in l]
        print(f"{' '.join(opts)} GP_POISON={poison}{' GP_DP_UNCHECKED=1' if env_extra else ''}: " + " | ".join(gate), flush=True)
        if p.returncode:
            fails.append(f"{' '.join(opts)} GP_POISON={poison}: decoder gate failed\n{p.stdout}{p.stderr}")
        keep.append(d)
        outs.append(p.stdout)
    out0 = sorted(glob.glob(os.path.join(keep[0], "q100", "*.avif")))
    out1 = sorted(glob.glob(os.path.join(keep[1], "q100", "*.avif")))
    names = [os.path.basename(f) for f in out0]
    if len(out0) != IMAGES or names != [os.path.basename(f) for f in out1]:
        fails.append(f"{' '.join(opts)}: outputs {names} / {[os.path.basename(f) for f in out1]}, expected {IMAGES} each")
    else:
        diff = [n for f, g, n in zip(out0, out1, names) if not filecmp.cmp(f, g, shallow=False)]
        if diff:
            fails.append(f"{' '.join(opts)}: output depends on uninitialised memory (GP_POISON 0 / 120): {diff}")
    return outs


def refused(a, args, what, fails):
    p = subprocess.run([a.bin] + args, capture_output=True, text=True, timeout=300)
    print(f"{what}: exit {p.returncode}: {p.stderr.strip()}", flush=True)
    if p.returncode == 0 or "no partition" not in p.stderr:
        fails.append(f"{what} must be refused (exit {p.returncode}): {p.stderr}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    a = ap.parse_args()
    t = tempfile.mkdtemp(prefix="av1g_maxpart_")
    fails = []
    try:
        img = os.path.join(t, "img")
        os.makedirs(img)
        flat = lambda x, y: (120, 130, 140)
        png(os.path.join(img, "flat_64x64.png"), 64, 64, flat)
        png(os.path.join(img, "flat_16x16.png"), 16, 16, flat)
        png(os.path.join(img, "flat_48x48.png"), 48, 48, flat)
        png(os.path.join(img, "mixed_200x136.png"), 200, 136, mixed)
        png(os.path.join(img, "alpha_72x40.png"), 72, 40, lambda x, y: (200, 60, 30, 255 if x < 40 else 128), alpha=True)
        for k, opts in enumerate(CASES):
            gate_twice(a, img, t, f"c{k}", opts, fails)
        # the safety net, made to act: the pruning kept although the partition check would switch it off
        outs = gate_twice(a, img, t, "net", NET_CASE, fails, {"GP_DP_UNCHECKED": "1"}, allow_net=True)
        if not all("dp safety net" in o for o in outs):
            fails.append("GP_DP_UNCHECKED=1 --maxpart 8: the safety net did not report the blocks it coded")
        # ... without the 4x4 shape (turbo2): the frame fails, non-zero exit in single-file and batch mode
        flat64 = os.path.join(img, "flat_64x64.png")
        one = os.path.join(t, "one")
        os.makedirs(one)
        shutil.copy(flat64, one)
        env = dict(os.environ, GP_DP_UNCHECKED="1", GP_POISON="120")
        for mode, args in (("single", [flat64, os.path.join(t, "n.avif")]),
                           ("batch", ["--batch", one, "--outdir", os.path.join(t, "nb")])):
            p = subprocess.run([a.bin] + args + ["--preset", "turbo2", "--maxpart", "8", "--small-preset", "none"],
                               capture_output=True, text=True, env=env, timeout=300)
            msg = next((l for l in p.stderr.splitlines() if "k_dp:" in l), "")
            print(f"GP_DP_UNCHECKED=1 --preset turbo2 --maxpart 8 ({mode}): exit {p.returncode}: {msg.strip()}", flush=True)
            if p.returncode == 0 or not msg:
                fails.append(f"turbo2 safety net ({mode}): the frame must fail with the k_dp message (exit {p.returncode})\n{p.stderr}")
        # options that leave no partition the analysis evaluates (turbo2 has no 4x4: --maxpart 4 forces 4x4 leaves),
        # for the main preset and for the small-image preset of a frame that is not small
        refused(a, [flat64, os.path.join(t, "x.avif"), "--preset", "turbo2", "--maxpart", "4", "--small-preset", "none"],
                "--preset turbo2 --maxpart 4", fails)
        refused(a, [flat64, os.path.join(t, "y.avif"), "--preset", "base", "--small-preset", "turbo2", "--maxpart", "4",
                    "--small-mp", "0.000001"], "--preset base --small-preset turbo2 --maxpart 4 (not a small frame)", fails)
    finally:
        shutil.rmtree(t, ignore_errors=True)
    for f in fails:
        print("FAIL", f)
    print("maxpart_test:", "FAIL" if fails else "PASS")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
