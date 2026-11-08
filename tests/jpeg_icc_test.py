#!/usr/bin/env python3
"""#67: a JPEG's ICC profile goes into the AVIF only if its APP2 segments are numbered correctly, as libjpeg-turbo's
jpeg_read_icc_profile requires: every segment gives the same count N, and the numbers 1..N each appear once, in any
order.  Otherwise the AVIF has no profile.  Before the fix the segments were sorted and joined whatever their numbers,
so a repeated segment doubled the profile and a missing one truncated it.

jpeg_gray_gen --pages writes a 4:2:0 JPEG; this script inserts APP2 segments after its APP0 and reads the profile back
from each AVIF's colr 'prof' box: in batch mode (nvJPEG, and the CPU entropy decoder with GP_INGEST_DECODER=1) and in
single-file mode.

  tests/jpeg_icc_test.py --bin build/av1gpu --gen build/jpeg_gray_gen
"""
import argparse
import os
import shutil
import struct
import subprocess
import sys
import tempfile


def fake_profile(n=588):
    """A profile-shaped blob (av1gpu copies the bytes, it does not parse them): header with size and 'acsp', then filler."""
    head = struct.pack(">I4sI4s4s4s", n, b"lcms", 0x02100000, b"mntr", b"RGB ", b"XYZ ") + bytes(12) + b"acsp"
    return head + bytes((i * 37 + 11) & 0xFF for i in range(n - len(head)))


def app2(ident, body):
    data = ident + body
    return b"\xff\xe2" + struct.pack(">H", len(data) + 2) + data


def icc_seg(seq, count, part):
    return app2(b"ICC_PROFILE\0", bytes([seq, count]) + part)


def insert_after_app0(jpeg, segs):
    assert jpeg[:2] == b"\xff\xd8"
    end = 2
    if jpeg[2:4] == b"\xff\xe0":
        end = 4 + struct.unpack(">H", jpeg[4:6])[0]
    return jpeg[:end] + b"".join(segs) + jpeg[end:]


def avif_profile(path):
    """The colr 'prof' box's payload, or None."""
    d = open(path, "rb").read()
    i = d.find(b"colrprof")
    if i < 4:
        return None
    size = struct.unpack(">I", d[i - 4:i])[0]
    return d[i + 8:i - 4 + size]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--gen", required=True)
    a = ap.parse_args()
    t = tempfile.mkdtemp(prefix="av1g_icc_")
    fails = []
    try:
        g = subprocess.run([a.gen, "--pages", t], capture_output=True, text=True, timeout=300)
        if g.returncode:
            print(f"FAIL jpeg_gray_gen --pages: exit {g.returncode}: {g.stderr.strip()[-500:]}")
            return 1
        base = open(os.path.join(t, "page.jpg"), "rb").read()
        if b"ICC_PROFILE" in base:
            print("FAIL page.jpg already has an ICC profile")
            return 1
        p = fake_profile()
        h, a3, b3 = len(p) // 2, len(p) // 3, 2 * len(p) // 3
        mpf = app2(b"MPF\0", b"MM\0\x2a\0\0\0\x08" + bytes(16))   # another APP2 (Multi-Picture Format), not ICC
        cases = {  # name: (segments, the profile libjpeg-turbo reads)
            "no_icc": ([], None),
            "single": ([icc_seg(1, 1, p)], p),
            "twice": ([icc_seg(1, 1, p), icc_seg(1, 1, p)], None),
            "split_reversed": ([icc_seg(2, 2, p[h:]), icc_seg(1, 2, p[:h])], p),
            "split_missing": ([icc_seg(1, 2, p[:h])], None),
            "split3_shuffled": ([icc_seg(3, 3, p[b3:]), icc_seg(1, 3, p[:a3]), icc_seg(2, 3, p[a3:b3])], p),
            "split3_repeated": ([icc_seg(1, 3, p[:a3]), icc_seg(2, 3, p[a3:b3]), icc_seg(2, 3, p[a3:b3]),
                                 icc_seg(3, 3, p[b3:])], None),
            "count_mismatch": ([icc_seg(1, 2, p[:h]), icc_seg(2, 3, p[h:])], None),
            "seq_zero": ([icc_seg(0, 1, p)], None),
            "seq_above_count": ([icc_seg(2, 1, p)], None),
            "empty_middle_part": ([icc_seg(1, 3, p[:h]), icc_seg(2, 3, b""), icc_seg(3, 3, p[h:])], p),
            "other_app2_too": ([mpf, icc_seg(1, 1, p), mpf], p),
        }
        src = os.path.join(t, "in")
        os.makedirs(src)
        names = sorted(cases)
        for n in names:
            open(os.path.join(src, n + ".jpg"), "wb").write(insert_after_app0(base, cases[n][0]))
        lst = os.path.join(t, "list.txt")
        open(lst, "w").write("".join(os.path.join(src, n + ".jpg") + "\n" for n in names))

        runs = []
        for label, env in (("batch", {}), ("batch GP_INGEST_DECODER=1", {"GP_INGEST_DECODER": "1"})):
            out = os.path.join(t, "out_" + str(len(runs)))
            r = subprocess.run([a.bin, "--batch", lst, "--outdir", out], capture_output=True, text=True, timeout=300,
                               env=dict(os.environ, **env))
            if r.returncode:
                fails.append(f"{label}: exit {r.returncode}: {r.stderr.strip()[-300:]}")
            runs.append((label, {n: os.path.join(out, n + ".jpg.avif") for n in names}))   # batch keeps the extension
        single = os.path.join(t, "out_single")
        os.makedirs(single)
        for n in names:
            r = subprocess.run([a.bin, os.path.join(src, n + ".jpg"), os.path.join(single, n + ".avif")],
                               capture_output=True, text=True, timeout=300)
            if r.returncode:
                fails.append(f"single {n}: exit {r.returncode}: {r.stderr.strip()[-300:]}")
        runs.append(("single", {n: os.path.join(single, n + ".avif") for n in names}))

        for label, files in runs:
            for n in names:
                want = cases[n][1]
                if not os.path.exists(files[n]):
                    fails.append(f"{label} {n}: no output")
                    continue
                got = avif_profile(files[n])
                ok = got == want
                desc = lambda x: "no profile" if x is None else f"{len(x)} bytes" + ("" if x == p else " (not the profile)")
                print(f"{'ok  ' if ok else 'FAIL'} {label:26s} {n:18s} {desc(got)}" + ("" if ok else f", want {desc(want)}"))
                if not ok:
                    fails.append(f"{label} {n}: {desc(got)}, want {desc(want)}")
    finally:
        shutil.rmtree(t, ignore_errors=True)
    if fails:
        print(f"FAIL: {len(fails)}")
        for f in fails:
            print("  " + f)
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
