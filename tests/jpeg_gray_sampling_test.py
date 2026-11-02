#!/usr/bin/env python3
"""Gray JPEGs whose one component is tagged 2x2, 2x1 or 1x2 in the SOF must encode like the 1x1 file (#44).

A one-component scan is non-interleaved, so its MCU is one block whatever the factors say, and libjpeg-turbo ignores
them; v0.6.0's nvJPEG route encoded such files scrambled (5-17 dB), with exit 0. jpeg_gray_gen writes baseline,
restart-interval and progressive gray JPEGs as 1x1, 2x2, 2x1 and 1x2, with libjpeg-turbo's decode of each.

  - single-image mode, nvJPEG (default) and GP_INGEST_DECODER=1: the encoder's input planes (--dump-source) must equal
    libjpeg-turbo's decode byte for byte;
  - batch mode, one list with every file, nvJPEG and GP_INGEST_DECODER=1: batch mode has no source dump, so every
    tagged file's AVIF must be byte-identical to its 1x1 file's, and the 1x1 file's reconstruction (--debug-outputs)
    at least 30 dB from libjpeg-turbo's decode (PSNR; scrambled planes gave 5-17 dB).

  tests/jpeg_gray_sampling_test.py --bin build/av1gpu --gen build/jpeg_gray_gen
"""
import argparse
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

TAGS = ("11", "22", "21", "12")
MODES = (("nvjpeg", {}), ("cpu", {"GP_INGEST_DECODER": "1"}))


def psnr(a, b):
    if len(a) != len(b):
        return -1.0
    se = sum((x - y) * (x - y) for x, y in zip(a, b))
    return math.inf if se == 0 else 10 * math.log10(255 * 255 * len(a) / se)


def read(p):
    with open(p, "rb") as f:
        return f.read()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--gen", required=True)
    a = ap.parse_args()
    t = tempfile.mkdtemp(prefix="av1g_gray_")
    fails = []
    try:
        src = os.path.join(t, "in")
        os.makedirs(src)
        g = subprocess.run([a.gen, src], capture_output=True, text=True, timeout=300)
        if g.returncode:
            print(f"FAIL jpeg_gray_gen: exit {g.returncode}: {g.stderr.strip()[-500:]}")
            return 1
        names = [(n, int(w), int(h)) for n, w, h in (l.split() for l in g.stdout.splitlines())]
        refs = {n: read(os.path.join(src, n + ".gray")) for n, _, _ in names}
        # single-image mode: input planes against libjpeg-turbo
        for mode, env in MODES:
            e = dict(os.environ, **env)
            for n, w, h in names:
                res = []
                for tag in TAGS:
                    f = os.path.join(src, f"{n}_{tag}.jpg")
                    dump = os.path.join(t, "dump.y")
                    p = subprocess.run([a.bin, f, os.path.join(t, "single.avif"), "--dump-source", dump],
                                       capture_output=True, text=True, env=e, timeout=300)
                    got = read(dump) if p.returncode == 0 and os.path.exists(dump) else b""
                    ok = got == refs[n]
                    res.append(f"{tag} {'ok' if ok else 'FAIL'}" + ("" if ok else f" ({psnr(got, refs[n]):.1f} dB)"))
                    if not ok:
                        fails.append(f"single {mode} {n}_{tag}")
                        if p.returncode:
                            sys.stdout.write(p.stderr[-1000:])
                    if os.path.exists(dump):
                        os.remove(dump)
                print(f"{'ok  ' if all(r.endswith(' ok') for r in res) else 'FAIL'} single {mode:6} {n:16} "
                      f"planes vs libjpeg-turbo: {', '.join(res)}")
        # batch mode: tagged AVIFs against the 1x1 file's, the 1x1 reconstruction against libjpeg-turbo
        lst = os.path.join(t, "list.txt")
        with open(lst, "w") as fl:
            for n, _, _ in names:
                for tag in TAGS:
                    fl.write(os.path.join(src, f"{n}_{tag}.jpg") + "\n")
        for mode, env in MODES:
            out = os.path.join(t, "out_" + mode)
            p = subprocess.run([a.bin, "--batch", lst, "--outdir", out, "--debug-outputs"], capture_output=True,
                               text=True, env=dict(os.environ, **env), timeout=600)
            m = re.search(r"skipped \(unsupported\): (\d+)", p.stderr)
            if p.returncode or not m or m.group(1) != "0":
                print(f"FAIL batch {mode}: exit {p.returncode}, skipped {m.group(1) if m else None}")
                sys.stdout.write(p.stderr[-2000:])
                fails.append(f"batch {mode}")
                continue
            for n, w, h in names:
                base = os.path.join(out, f"{n}_11.jpg.avif")
                if not os.path.exists(base):
                    print(f"FAIL batch {mode:6} {n}: no output for the 1x1 file")
                    fails.append(f"batch {mode} {n}")
                    continue
                db = psnr(read(base + ".recon.yuv"), refs[n])
                res, ok = [f"11 {db:.1f} dB"], db >= 30
                for tag in TAGS[1:]:
                    o = os.path.join(out, f"{n}_{tag}.jpg.avif")
                    same = os.path.exists(o) and read(o) == read(base)
                    if not same:
                        rd = psnr(read(o + ".recon.yuv"), refs[n]) if os.path.exists(o + ".recon.yuv") else -1.0
                        res.append(f"{tag} FAIL ({rd:.1f} dB)")
                        ok = False
                    else:
                        res.append(f"{tag} same")
                if not ok:
                    fails.append(f"batch {mode} {n}")
                print(f"{'ok  ' if ok else 'FAIL'} batch  {mode:6} {n:16} AVIF vs the 1x1 file's: {', '.join(res)}")
    finally:
        shutil.rmtree(t, ignore_errors=True)
    print("PASS" if not fails else f"FAIL: {len(fails)} checks: {', '.join(fails[:20])}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
