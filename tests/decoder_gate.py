#!/usr/bin/env python3
"""Decoder identity gate for the batch (throughput) path.

Encodes every image in a directory with `av1gpu --batch ... --debug-outputs`, so each output also gets
<name>.avif.obu (temporal delimiter + OBUs) and <name>.avif.recon.yuv (the encoder's own reconstruction).
Every .obu is then decoded with dav1d and aomdec, and both decoders must reproduce the encoder's
reconstruction byte for byte (monochrome: aomdec's extra neutral chroma planes are ignored).  The chroma
format of every stream is read from its sequence header (seq_profile 0: 4:2:0 / monochrome, 1: 4:4:4)
and the reconstruction must have exactly that format's plane sizes; --expect-format 420|444 also
requires every colour image to be coded in that format (e.g. with `-- --yuv 444`).
A run passes only with N/N.

  tests/decoder_gate.py --bin build/av1gpu --images DIR [--q 120,60] [--jobs 4] [--keep DIR] [--expect-format 444]
                        [-- <av1gpu options>]

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


def seq_info(obu):
    """(format, width, height) from the reduced still-picture sequence header of an OBU stream:
    format '420', '444' or 'mono'; None if it cannot be read."""
    pos = 0
    while pos < len(obu):
        h = obu[pos]
        pos += 1
        typ, ext, has_size = (h >> 3) & 15, (h >> 2) & 1, (h >> 1) & 1
        pos += ext
        size = len(obu) - pos
        if has_size:
            size, sh = 0, 0
            while True:
                b = obu[pos]
                pos += 1
                size |= (b & 0x7F) << sh
                sh += 7
                if not b & 0x80:
                    break
        if typ == 1:
            bits = int.from_bytes(obu[pos:pos + size], "big")
            nbits = 8 * size
            at = [0]

            def f(n):
                at[0] += n
                return (bits >> (nbits - at[0])) & ((1 << n) - 1)
            profile = f(3)
            f(1)
            if not f(1):
                return None   # not a reduced still-picture header
            f(5)
            wb, hb = f(4) + 1, f(4) + 1
            w, hgt = f(wb) + 1, f(hb) + 1
            f(6)        # use_128x128_superblock .. enable_restoration
            if f(1):    # high_bitdepth
                return None
            mono = 0 if profile == 1 else f(1)
            fmt = "mono" if mono else "444" if profile == 1 else "420" if profile == 0 else None
            return fmt, w, hgt
        pos += size
    return None


def plane_bytes(fmt, w, h):
    if fmt == "mono":
        return w * h
    if fmt == "444":
        return 3 * w * h
    return w * h + 2 * ((w + 1) // 2) * ((h + 1) // 2)


def check_one(avif, workdir):
    obu, rec = avif + ".obu", avif + ".recon.yuv"
    name = os.path.basename(avif)[:-5]
    if not (os.path.exists(obu) and os.path.exists(rec)):
        return dict(file=name, dav1d=False, aomdec=False, err="missing .obu/.recon.yuv")
    want = open(rec, "rb").read()
    res = dict(file=name, bytes=os.path.getsize(avif))
    si = seq_info(open(obu, "rb").read())
    if si is None:
        return dict(res, dav1d=False, aomdec=False, err="unreadable sequence header")
    res["format"] = si[0]
    if len(want) != plane_bytes(*si):
        return dict(res, dav1d=False, aomdec=False, err=f"recon has {len(want)} bytes, {si[0]} {si[1]}x{si[2]} needs {plane_bytes(*si)}")
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
    ap.add_argument("--expect-format", choices=("420", "444"), help="every colour image must be coded in this format")
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
        if a.expect_format:
            for r in results:
                if r.get("format") not in (a.expect_format, "mono", None):
                    r.update(dav1d=False, aomdec=False, err=f"coded {r['format']}, expected {a.expect_format}")
        bad = [r for r in results if not (r.get("dav1d") and r.get("aomdec"))]
        formats = {}
        for r in results:
            formats[r.get("format")] = formats.get(r.get("format"), 0) + 1
        run = dict(q=q, rc=p.returncode, wall_s=round(time.time() - t0, 1), batch=line.strip(), n_encoded=len(avifs),
                   n_pass=len(results) - len(bad), n_not_encoded=len(files) - len(avifs), formats=formats,
                   bytes=sum(r.get("bytes", 0) for r in results), failures=bad)
        summary["runs"].append(run)
        ok = p.returncode == 0 and not bad and len(avifs) > 0
        allok &= ok
        fmts = ", ".join(f"{k} {v}" for k, v in sorted(formats.items(), key=lambda kv: str(kv[0])))
        print(f"[gate] q{q}: {run['n_pass']}/{run['n_encoded']} identical in dav1d and aomdec ({fmts}; {run['bytes']} bytes), "
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
