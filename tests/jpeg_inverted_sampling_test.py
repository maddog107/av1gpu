#!/usr/bin/env python3
"""3-component JPEGs whose luma has smaller sampling factors than a chroma component (Y 1x1 with Cb / Cr 2x2, ...) must
give the encoder libjpeg-turbo's planes (follow-up to #44).

Such files take the CPU route, with or without nvJPEG, and their luma was upsampled by replicating its samples instead
of libjpeg-turbo's fancy upsampling. jpeg_gray_gen --ycbcr writes them and controls, with libjpeg-turbo's YCbCr decode
(<name>.444) and the 4:2:0 planes av1gpu must make (<name>.420: libjpeg-turbo's luma, chroma by av1gpu's 4:2:0 rule).

  - single-image mode, default decoder and GP_INGEST_DECODER=1, default 4:2:0 and --yuv 444: the encoder's input
    planes (--dump-source) must equal <name>.420 / <name>.444 byte for byte;
  - batch mode, one list with every file, the same four combinations: batch mode has no source dump and encodes the
    same planes as single-image mode exactly, so every AVIF must be byte-identical to the single-image one.

  tests/jpeg_inverted_sampling_test.py --bin build/av1gpu --gen build/jpeg_gray_gen
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

FORMATS = (("420", []), ("444", ["--yuv", "444"]))
MODES = (("default", {}), ("cpu", {"GP_INGEST_DECODER": "1"}))


def read(p):
    with open(p, "rb") as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--gen", required=True)
    a = ap.parse_args()
    t = tempfile.mkdtemp(prefix="av1g_ycc_")
    fails = []
    try:
        src = os.path.join(t, "in")
        os.makedirs(src)
        g = subprocess.run([a.gen, "--ycbcr", src], capture_output=True, text=True, timeout=300)
        if g.returncode:
            print(f"FAIL jpeg_gray_gen --ycbcr: exit {g.returncode}: {g.stderr.strip()[-500:]}")
            return 1
        names = [l.split()[0] for l in g.stdout.splitlines()]
        lst = os.path.join(t, "list.txt")
        with open(lst, "w") as fl:
            fl.write("".join(os.path.join(src, n + ".jpg") + "\n" for n in names))
        for fmt, opts in FORMATS:
            for mode, env in MODES:
                e = dict(os.environ, **env)
                single = {}
                res = {}
                for n in names:
                    o, dump = os.path.join(t, n + ".avif"), os.path.join(t, "dump.yuv")
                    p = subprocess.run([a.bin, os.path.join(src, n + ".jpg"), o, "--dump-source", dump] + opts,
                                       capture_output=True, text=True, env=e, timeout=300)
                    ref = read(os.path.join(src, f"{n}.{fmt}"))
                    ok = p.returncode == 0 and os.path.exists(dump) and read(dump) == ref
                    single[n] = read(o) if p.returncode == 0 and os.path.exists(o) else None
                    res[n] = ["planes ok" if ok else "planes FAIL"]
                    if not ok:
                        fails.append(f"single {fmt} {mode} {n}")
                        sys.stdout.write(p.stderr[-500:])
                    for x in (o, dump):
                        if os.path.exists(x):
                            os.remove(x)
                out = os.path.join(t, f"out_{fmt}_{mode}")
                p = subprocess.run([a.bin, "--batch", lst, "--outdir", out] + opts, capture_output=True, text=True,
                                   env=e, timeout=600)
                if p.returncode:
                    sys.stdout.write(p.stderr[-2000:])
                for n in names:
                    o = os.path.join(out, n + ".jpg.avif")
                    same = single[n] is not None and os.path.exists(o) and read(o) == single[n]
                    res[n].append("batch AVIF same" if same else "batch AVIF FAIL")
                    if not same:
                        fails.append(f"batch {fmt} {mode} {n}")
                    print(f"{'ok  ' if 'FAIL' not in ' '.join(res[n]) else 'FAIL'} {fmt} {mode:7} {n:24} "
                          f"{', '.join(res[n])}")
    finally:
        shutil.rmtree(t, ignore_errors=True)
    print("PASS" if not fails else f"FAIL: {len(fails)} checks: {', '.join(fails[:20])}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
