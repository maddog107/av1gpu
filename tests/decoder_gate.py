#!/usr/bin/env python3
"""Decoder identity gate for the batch (throughput) path.

Encodes every image in a directory with `av1gpu --batch ... --debug-outputs`, so each output also gets
<name>.avif.obu (temporal delimiter + OBUs) and <name>.avif.recon.yuv (the encoder's own reconstruction).
Every .obu is then decoded with dav1d and aomdec, and both decoders must reproduce the encoder's
reconstruction byte for byte (monochrome: aomdec's extra neutral chroma planes are ignored).
A run passes only with N/N.

  tests/decoder_gate.py --bin build/av1gpu --images DIR [--q 120,60] [--jobs 4] [--keep DIR] [-- <av1gpu options>]

Decoders come from PATH, or from $AV1G_DAV1D / $AV1G_AOMDEC.  Exit status 0 only if every encoded image passed.
Images the batch path does not encode (unsupported inputs) are reported separately.
"""
import argparse
import concurrent.futures as cf
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

DAV1D = os.environ.get("AV1G_DAV1D", "dav1d")
AOMDEC = os.environ.get("AV1G_AOMDEC", "aomdec")
EXTS = (".jpg", ".jpeg", ".jfif", ".png", ".webp")


def check_one(avif, workdir):
    obu, rec = avif + ".obu", avif + ".recon.yuv"
    name = os.path.basename(avif)[:-5]
    if not (os.path.exists(obu) and os.path.exists(rec)):
        return dict(file=name, dav1d=False, aomdec=False, err="missing .obu/.recon.yuv")
    want = open(rec, "rb").read()
    res = dict(file=name, bytes=os.path.getsize(avif))
    with tempfile.TemporaryDirectory(dir=workdir) as td:
        r = subprocess.run([DAV1D, "-q", "-i", obu, "--demuxer", "section5", "-o", f"{td}/d.yuv"], capture_output=True)
        got = open(f"{td}/d.yuv", "rb").read() if os.path.exists(f"{td}/d.yuv") else b""
        res["dav1d"] = r.returncode == 0 and got == want
        r = subprocess.run([AOMDEC, "--rawvideo", "-o", f"{td}/a.yuv", obu], capture_output=True)
        got = open(f"{td}/a.yuv", "rb").read() if os.path.exists(f"{td}/a.yuv") else b""
        if len(got) > len(want) and got[len(want):].count(0x80) == len(got) - len(want):
            got = got[:len(want)]   # monochrome: aomdec appends neutral chroma
        res["aomdec"] = r.returncode == 0 and got == want
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bin", default="build/av1gpu")
    ap.add_argument("--images", required=True, help="directory of input images")
    ap.add_argument("--q", default="120", help="comma-separated qindex values")
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--keep", help="keep outputs in this directory instead of a temporary one")
    ap.add_argument("--json", help="write a summary here")
    argv = sys.argv[1:]
    cut = argv.index("--") if "--" in argv else len(argv)
    a = ap.parse_args(argv[:cut])
    opts = argv[cut + 1:]
    files = sorted(f for f in glob.glob(os.path.join(a.images, "*")) if f.lower().endswith(EXTS))
    if not files:
        sys.exit(f"no images in {a.images}")
    root = a.keep or tempfile.mkdtemp(prefix="av1gpu_gate_")
    os.makedirs(root, exist_ok=True)
    summary = dict(bin=a.bin, images=a.images, opts=opts, n_inputs=len(files), runs=[])
    allok = True
    for q in [int(x) for x in a.q.split(",")]:
        out = os.path.join(root, f"q{q}")
        shutil.rmtree(out, ignore_errors=True)
        os.makedirs(out)
        cmd = [a.bin, "--batch", a.images, "--outdir", out, "--qindex", str(q), "--debug-outputs"] + opts
        t0 = time.time()
        p = subprocess.run(cmd, capture_output=True, text=True)
        line = next((l for l in p.stderr.splitlines() if "MP/s" in l), "")
        avifs = sorted(glob.glob(os.path.join(out, "*.avif")))
        with cf.ThreadPoolExecutor(a.jobs) as ex:
            results = list(ex.map(lambda f: check_one(f, out), avifs))
        bad = [r for r in results if not (r.get("dav1d") and r.get("aomdec"))]
        run = dict(q=q, rc=p.returncode, wall_s=round(time.time() - t0, 1), batch=line.strip(), n_encoded=len(avifs),
                   n_pass=len(results) - len(bad), n_not_encoded=len(files) - len(avifs), failures=bad)
        summary["runs"].append(run)
        ok = p.returncode == 0 and not bad and len(avifs) > 0
        allok &= ok
        print(f"[gate] q{q}: {run['n_pass']}/{run['n_encoded']} identical in dav1d and aomdec, "
              f"{run['n_not_encoded']} not encoded (unsupported input), rc={p.returncode} {'PASS' if ok else 'FAIL'}",
              flush=True)
        if p.returncode:
            print(p.stderr[-2000:], file=sys.stderr)
        for r in bad[:20]:
            print(f"   BAD {r['file']} dav1d={r.get('dav1d')} aomdec={r.get('aomdec')} {r.get('err', '')}", flush=True)
    summary["pass"] = allok
    if a.json:
        json.dump(summary, open(a.json, "w"), indent=1)
    if not a.keep:
        shutil.rmtree(root, ignore_errors=True)
    sys.exit(0 if allok else 1)


if __name__ == "__main__":
    main()
