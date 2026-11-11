#!/usr/bin/env python3
"""The staged-analysis switches (--topdown, --prune-41; on in the presets base and turbo): they work, deterministically,
and decode.

Inputs, generated: a 1.1 MP picture (the default settings encode it with turbo; frame edges that cut 16x16 cells;
flat sky, a gradient, fine and coarse texture), a 640x480 and a 333x250 one (base below 1 MP), a 320x240 RGBA one
(alpha item); plus tests/gate_images.  Checked, with the default settings and with --preset base:

- function: each switch alone changes the output where it applies (--topdown in both presets, --prune-41 in base;
  in the default turbo preset --prune-41 has no 4:1 shape to prune); total bytes within 2 % of the run with both
  off (--topdown 0 --prune-41 off) at the same qindex; the work actually skipped (GP_DEBUG_STAGES: listed blocks per staged shape, a
  deterministic count) between bounds that both "refine nothing" and "refine everything" break: with --topdown in
  base, the side-4 shapes are listed for 8..92 % of their blocks, the side-8 shapes for 30..99 %; with --prune-41 in
  base, the 4:1 shapes for 5..95 %;
- determinism: both switches give the same bytes with three encoders in list order as with one encoder and the list
  reversed (and --topdown alone in the default preset); with every allocation poisoned (GP_POISON=255) the same bytes;
- a run without the options gives the bytes of --topdown 1 --prune-41 10 (the defaults of base and turbo);
- --small-kernel 0 (base, both switches): no shape of up to 64 pixels is listed (they have no list-driven kernel and
  are analysed in full), the larger staged shapes are, and the output changes;
- tests/decoder_gate.py (dav1d + aomdec reproduce the encoder's reconstruction, alpha items included) with each
  switch alone, and with both in 4:4:4 10-bit, with 4x4 tiles and with --small-kernel 0;
  (the routes without a fallback partition, with GP_POISON 0 / 120: tests/maxpart_test.py)
- option validation: --topdown 2 / -1 and --prune-41 abc / -0 / nan / inf / 1e30 / 1001 are refused.

  tests/staged_analysis_test.py --bin build/av1gpu [--keep DIR]
"""
import argparse
import collections
import glob
import hashlib
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import zlib

HERE = os.path.dirname(os.path.abspath(__file__))
# each switch alone (the presets base and turbo have both on: --topdown 1 --prune-41 10)
TD = ["--topdown", "1", "--prune-41", "off"]
G = ["--topdown", "0", "--prune-41", "10"]
BOTH = ["--topdown", "1", "--prune-41", "10"]
OFF = ["--topdown", "0", "--prune-41", "off"]


def png(path, w, h, px, alpha=False):
    ch = 4 if alpha else 3
    rows = b"".join(b"\0" + b"".join(struct.pack("B" * ch, *px(x, y)) for x in range(w)) for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6 if alpha else 2, 0, 0, 0))
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


def nbytes(d):
    return sum(os.path.getsize(f) for f in glob.glob(os.path.join(d, "*.avif")))


def listed(err):
    """GP_DEBUG_STAGES lines summed over batches: {WxH: [listed, blocks]}"""
    s = collections.defaultdict(lambda: [0, 0])
    for m in re.finditer(r"stages: (\d+)x(\d+) listed (\d+) of (\d+)", err):
        k = (int(m.group(1)), int(m.group(2)))
        s[k][0] += int(m.group(3))
        s[k][1] += int(m.group(4))
    return s


def fraction(s, pick):
    a = sum(v[0] for k, v in s.items() if pick(*k))
    b = sum(v[1] for k, v in s.items() if pick(*k))
    return a / b if b else None


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
        if not os.path.exists(os.path.join(src, "rgba_320x240.png")):
            png(os.path.join(src, "rgba_320x240.png"), 320, 240, lambda x, y: picture(x + 400, y + 200) + ((x * 255) // 319,), alpha=True)
        for f in glob.glob(os.path.join(HERE, "gate_images", "*")):
            shutil.copy(f, src)
        files = sorted(glob.glob(os.path.join(src, "*.png")))
        fwd, rev = os.path.join(t, "fwd.lst"), os.path.join(t, "rev.lst")
        open(fwd, "w").write("".join(f + "\n" for f in files))
        open(rev, "w").write("".join(f + "\n" for f in reversed(files)))

        def batch(tag, preset, opts, lst=fwd, env=None):
            od = os.path.join(t, tag)
            shutil.rmtree(od, ignore_errors=True)
            rc, err = run(a.bin, ["--batch", lst, "--outdir", od] + preset + opts, env)
            h = hashes(od)
            if rc or len(h) != len(files):
                fails.append(f"{tag}: exit {rc}, {len(h)} of {len(files)} outputs: {err[-300:]}")
            return h, nbytes(od), err

        for preset, pname, switches in (([], "default", {"topdown": TD}), (["--preset", "base"], "base", {"topdown": TD, "prune41": G})):
            none, b0, _ = batch(f"{pname}_none", preset, OFF)
            dflt, _, _ = batch(f"{pname}_dflt", preset, [])
            for name, opts in list(switches.items()) + [("both", BOTH)]:
                h, b, err = batch(f"{pname}_{name}", preset, opts, env={"GP_DEBUG_STAGES": "1"})
                if h == none:
                    fails.append(f"{pname} {name}: leaves every output unchanged")
                if name == "both" and h != dflt:
                    fails.append(f"{pname}: a run without the options differs from --topdown 1 --prune-41 10 (the preset's defaults)")
                if abs(b / b0 - 1) > 0.02:
                    fails.append(f"{pname} {name}: {b} bytes against {b0} without it ({100 * (b / b0 - 1):+.2f} %)")
                s = listed(err)
                print(f"{pname} {name}: {100 * (b / b0 - 1):+.2f} % bytes; listed " +
                      ", ".join(f"{w}x{hh} {v[0]}/{v[1]}" for (w, hh), v in sorted(s.items())), flush=True)
                if pname == "base" and name == "topdown":
                    f4 = fraction(s, lambda w, hh: min(w, hh) == 4)
                    f8 = fraction(s, lambda w, hh: min(w, hh) == 8)
                    if f4 is None or not 0.08 <= f4 <= 0.92:
                        fails.append(f"base topdown: side-4 shapes listed for {f4} of their blocks (expected 0.08 .. 0.92)")
                    if f8 is None or not 0.30 <= f8 <= 0.99:
                        fails.append(f"base topdown: side-8 shapes listed for {f8} of their blocks (expected 0.30 .. 0.99)")
                if pname == "base" and name == "prune41":
                    f41 = fraction(s, lambda w, hh: w == 4 * hh or hh == 4 * w)
                    if f41 is None or not 0.05 <= f41 <= 0.95:
                        fails.append(f"base prune41: 4:1 shapes listed for {f41} of their blocks (expected 0.05 .. 0.95)")
                if name == "both" or (pname == "default" and name == "topdown"):
                    r, _, _ = batch(f"{pname}_{name}_enc1rev", preset, opts, lst=rev, env={"GP_ENCODERS": "1"})
                    if r != h:
                        fails.append(f"{pname} {name}: one encoder with the list reversed differs on "
                                     f"{[k for k in h if h.get(k) != r.get(k)]}")
                if name == "both" and pname == "base":
                    p, _, _ = batch(f"{pname}_{name}_poison", preset, opts, env={"GP_POISON": "255"})
                    if p != h:
                        fails.append(f"{pname} {name}: GP_POISON=255 changes {[k for k in h if h.get(k) != p.get(k)]}")
        # --small-kernel 0: the shapes of up to 64 pixels have no list-driven kernel, so they are analysed in full (stage
        # 1); the larger staged shapes are still listed
        sk0 = ["--preset", "base", "--small-kernel", "0"]
        h0, _, _ = batch("base_sk0_none", sk0, OFF)
        h, _, err = batch("base_sk0_both", sk0, BOTH, env={"GP_DEBUG_STAGES": "1"})
        s = listed(err)
        print("base --small-kernel 0 both: listed " + ", ".join(f"{w}x{hh} {v[0]}/{v[1]}" for (w, hh), v in sorted(s.items())), flush=True)
        if not s or any(w * hh <= 64 for w, hh in s):
            fails.append(f"--small-kernel 0: lists for {sorted(s)} (expected the staged shapes above 64 pixels only)")
        if h == h0:
            fails.append("--small-kernel 0: the switches leave every output unchanged")
        # decoder identity: each switch alone, both in 4:4:4 10-bit, both with 4x4 tiles, both with --small-kernel 0
        # (alpha items in every run)
        gates = [("default_topdown", TD, []), ("base_prune41", ["--preset", "base"] + G, []),
                 ("base_both_444_10", ["--preset", "base", "--yuv", "444", "--depth", "10"] + BOTH,
                  ["--expect-format", "444", "--expect-depth", "10"]),
                 ("default_both_tiles", ["--tiles", "0,2,2"] + BOTH, []),
                 ("base_both_sk0", sk0 + BOTH, [])]
        for tag, opts, gopts in gates:
            g = subprocess.run([sys.executable, os.path.join(HERE, "decoder_gate.py"), "--bin", a.bin, "--images", src, "--q", "120",
                                "--keep", os.path.join(t, "gate_" + tag)] + gopts + ["--"] + opts,
                               capture_output=True, text=True, timeout=1800)
            print(f"gate {tag}: " + (g.stdout.strip().splitlines()[-1] if g.stdout.strip() else g.stderr[-300:]), flush=True)
            if g.returncode:
                fails.append(f"gate {tag} failed:\n{g.stdout[-2000:]}{g.stderr[-1000:]}")
        # option validation (refused before any encode)
        for bad in (["--topdown", "2"], ["--topdown", "-1"], ["--prune-41", "abc"], ["--prune-41", "-0"], ["--prune-41", "nan"],
                    ["--prune-41", "inf"], ["--prune-41", "1e30"], ["--prune-41", "1001"]):
            rc, err = run(a.bin, [files[0], os.path.join(t, "bad.avif")] + bad)
            if rc == 0 or "bad " + bad[0] not in err:
                fails.append(f"{' '.join(bad)}: exit {rc}, not refused ({err.strip()[-200:]})")
    finally:
        if not a.keep:
            shutil.rmtree(t, ignore_errors=True)
    for f in fails:
        print("FAIL:", f)
    print("PASS" if not fails else f"FAIL ({len(fails)})")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
