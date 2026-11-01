#!/usr/bin/env python3
"""Inputs that cannot be read are skipped, and the rest is still encoded.

Batch mode gets a list with a good PNG, a directory, a directory with an image name, a missing file and an empty
file, then a directory holding the good PNG, the image-named subdirectory and the empty file. Both runs must exit 0,
write only the good image's AVIF into an --outdir whose parents do not exist yet, and report the others as skipped.
Single-image mode on a directory must fail with status 3. Before this test, a directory aborted av1gpu with
std::bad_alloc in both modes, and a nested --outdir was not created.

  tests/batch_skip_test.py --bin build/av1gpu --image tests/gate_images/cfl_edge_211x149.png
"""
import argparse
import os
import re
import shutil
import subprocess
import sys
import tempfile


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--image", required=True)
    a = ap.parse_args()
    t = tempfile.mkdtemp(prefix="av1g_skip_")
    fails = []
    try:
        src = os.path.join(t, "in")
        os.makedirs(os.path.join(src, "subdir.png"))
        shutil.copy(a.image, os.path.join(src, "good.png"))
        open(os.path.join(src, "empty.png"), "wb").close()
        lst = os.path.join(t, "list.txt")
        with open(lst, "w") as f:
            for p in ("good.png", "", "subdir.png", "missing.png", "empty.png"):
                f.write(os.path.join(src, p).rstrip("/") + "\n")
        for name, inp, skipped in (("list", lst, 4), ("dir", src, 2)):
            out = os.path.join(t, "out_" + name, "a", "b")   # --outdir and its missing parents are created

            p = subprocess.run([a.bin, "--batch", inp, "--outdir", out], capture_output=True, text=True, timeout=300)
            got = sorted(os.listdir(out)) if os.path.isdir(out) else []
            m = re.search(r"skipped \(unsupported\): (\d+)", p.stderr)
            n = int(m.group(1)) if m else None
            ok = p.returncode == 0 and got == ["good.png.avif"] and n == skipped
            print(f"{'ok  ' if ok else 'FAIL'} batch {name}: exit {p.returncode}, outputs {got}, skipped {n} (want {skipped})")
            if not ok:
                fails.append(name)
                sys.stdout.write(p.stderr[-2000:])
        p = subprocess.run([a.bin, src, os.path.join(t, "x.avif")], capture_output=True, text=True, timeout=300)
        ok = p.returncode == 3
        print(f"{'ok  ' if ok else 'FAIL'} single image, a directory: exit {p.returncode} (want 3): {p.stderr.strip()[-200:]}")
        if not ok:
            fails.append("single")
    finally:
        shutil.rmtree(t, ignore_errors=True)
    print("PASS" if not fails else f"FAIL: {', '.join(fails)}")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
