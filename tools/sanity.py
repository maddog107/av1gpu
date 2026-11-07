#!/usr/bin/env python3
"""Quick sanity check of an av1gpu binary: a few batch runs, every output's SHA-256 against tools/sanity_golden.txt,
and one speed figure.  Meant for a release or a PR that shouldn't change output: a few minutes on one GPU.

  python3 tools/sanity.py BINARY [--ref BINARY] [--images DIR] [--update]

Inputs: the 100 photos of bench/openimages_v4_gamut.csv, 0.2 to 24 MP, 4:2:0 / 4:2:2 / 4:4:4, with ICC profiles and
EXIF rotations (bench/fetch_images.py --manifest bench/openimages_v4_gamut.csv --out bench/images_gamut fetches them and checks
their SHA-256), tests/gate_images, and a few PNGs this script writes itself (RGB, RGBA, 16-bit, a page with a border).
About 100 s on an L4.

- Output: av1gpu's output is deterministic (the same bytes from the CUDA 12 and 13 builds and across GPUs), so a
  matching hash means the same file as the golden run.  A hash that differs is reported; tests/decoder_gate.py then
  checks the new files in dav1d and aomdec.
- Speed: the 100 photos three times over in one batch run, best of 5.  With --ref (say, the last release's binary) its
  runs alternate with the reference's on the same machine; otherwise it is compared with the golden figure for the same
  GPU.  More than 5 % slower is reported as SLOWER (a warning: the exit status is for the outputs only).
- --update rewrites the golden file from this binary: do it only for a binary whose outputs passed the decoder gate.

Exit status: 0 when every hash matches, 1 otherwise."""
import argparse, glob, hashlib, json, os, shutil, struct, subprocess, sys, tempfile, time, zlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOLDEN = os.path.join(HERE, "sanity_golden.txt")

# Each case is one batch run over a list of inputs.  "all" = the 100 photos, "sub" = 8 of them plus the PNGs.
CASES = [
    ("default", "all", []),
    ("fast", "sub", ["--preset", "fast"]),
    ("turbo3", "sub", ["--preset", "turbo3"]),
    ("444_10", "sub", ["--yuv", "444", "--depth", "10"]),
    ("lossless", "sub", ["--lossless"]),
    ("lossless_ycgco", "sub", ["--lossless", "--ycgco-re"]),
    ("crop", "sub", ["--crop", "100x80+7+3"]),
    ("maxsize", "sub", ["--max-size", "300x300"]),
    ("palette", "sub", ["--palette", "on"]),
    ("cpu_entropy", "sub", ["--cpu-entropy"]),
    ("noalpha", "sub", ["--no-alpha"]),
    ("trim", "sub", ["--trim", "--fuzz", "10%"]),
    ("resize", "sub", ["--resize", "50%"]),
    ("tone", "sub", ["--level", "5%,95%", "--gamma", "1.2"]),
    ("autolevel", "sub", ["--auto-level"]),
    ("negate", "sub", ["--negate"]),
    ("gray", "sub", ["--grayscale", "Rec709Luma"]),
]
SINGLE = [("single_jpeg", 0, []), ("single_rgba", "rgba", [])]


def png(path, w, h, px, depth=8, alpha=False):
    """Write a PNG with the standard library; px(x, y) gives a tuple of channel values."""
    ch = 4 if alpha else 3
    fmt = ">" + "H" * ch if depth == 16 else "B" * ch
    rows = b"".join(b"\0" + b"".join(struct.pack(fmt, *px(x, y)) for x in range(w)) for y in range(h))
    ct = 6 if alpha else 2
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, depth, ct, 0, 0, 0))
                           + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def make_pngs(d):
    os.makedirs(d, exist_ok=True)
    out = {}
    def edge(x, y):  # saturated colour edges and a gradient
        return ((x * 255) // 210, 255 if (x // 16 + y // 16) % 2 else 0, (y * 255) // 148)
    out["rgb"] = os.path.join(d, "rgb_211x149.png"); png(out["rgb"], 211, 149, edge)
    out["rgba"] = os.path.join(d, "rgba_160x120.png")
    png(out["rgba"], 160, 120, lambda x, y: (200, (x * 3) % 256, 40, (x + y) * 255 // 278), alpha=True)
    out["deep"] = os.path.join(d, "deep_128x96.png")
    png(out["deep"], 128, 96, lambda x, y: ((x * 65535) // 127, (y * 65535) // 95, ((x ^ y) * 1021) % 65536), depth=16)
    def page(x, y):  # paper with a border of 23-37 px, dark text blocks inside
        if x < 31 or x > 600 - 23 or y < 37 or y > 800 - 29: return (250, 249, 245)
        return (20, 20, 25) if (y // 9) % 3 == 0 and (x // 7) % 5 else (238, 236, 230)
    out["page"] = os.path.join(d, "page_600x800.png"); png(out["page"], 600, 800, page)
    return out


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def run(binary, args, outdir):
    shutil.rmtree(outdir, ignore_errors=True); os.makedirs(outdir)
    t = time.time()
    r = subprocess.run([binary] + args, capture_output=True, text=True)
    st = {}
    for line in r.stdout.splitlines():
        if line.startswith("{"):
            try: st = json.loads(line)
            except ValueError: pass
    return r.returncode, time.time() - t, st, r.stderr


def gpu_name():
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                              text=True).stdout.splitlines()[0].strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("binary")
    ap.add_argument("--images", default=os.path.join(ROOT, "bench", "images_gamut"))
    ap.add_argument("--update", action="store_true")
    ap.add_argument("--ref", help="a reference binary (e.g. the last release) to time alternately with this one")
    ap.add_argument("--work", default=None)
    a = ap.parse_args()
    photos = sorted(glob.glob(os.path.join(a.images, "*.jpg")))
    if len(photos) != 100:
        sys.exit(f"expected the 100 photos in {a.images} (bench/fetch_images.py --manifest bench/openimages_v4_gamut.csv --out {a.images}), found {len(photos)}")
    work = a.work or tempfile.mkdtemp(prefix="av1gpu-sanity-")
    gen = make_pngs(os.path.join(work, "in"))
    sub = photos[::13] + sorted(glob.glob(os.path.join(ROOT, "tests", "gate_images", "*"))) + list(gen.values())
    lists = {}
    for name, files in (("all", photos), ("sub", sub)):
        lists[name] = os.path.join(work, name + ".lst"); open(lists[name], "w").write("".join(f + "\n" for f in files))

    got, t0 = {}, time.time()
    for case, inputs, opts in CASES:
        od = os.path.join(work, "out", case)
        rc, wall, _, _ = run(a.binary, ["--batch", lists[inputs], "--outdir", od] + opts, od)
        got[f"{case}/exit"] = str(rc)
        for p in sorted(glob.glob(os.path.join(od, "*"))): got[f"{case}/{os.path.basename(p)}"] = sha(p)
    for case, src, opts in SINGLE:
        f = photos[src] if isinstance(src, int) else gen[src]
        od = os.path.join(work, "out", case)
        rc, wall, _, _ = run(a.binary, [f, os.path.join(od, "o.avif")] + opts, od)
        got[f"{case}/exit"] = str(rc)
        if os.path.exists(os.path.join(od, "o.avif")): got[f"{case}/o.avif"] = sha(os.path.join(od, "o.avif"))
    print(f"outputs: {len(got)} entries in {time.time() - t0:.0f} s", flush=True)

    # Speed: the 100 photos three times over (1,434 MP, about 6 s on an L4), best of 5 runs; with --ref, the reference
    # binary's runs alternate with this one's on the same machine.
    sd = os.path.join(work, "speed"); os.makedirs(sd, exist_ok=True)
    for k in range(3):
        for f in photos:
            l = os.path.join(sd, f"{k}_{os.path.basename(f)}")
            if not os.path.exists(l): os.symlink(f, l)
    sl = os.path.join(work, "speed.lst"); open(sl, "w").write("".join(os.path.join(sd, x) + "\n" for x in sorted(os.listdir(sd))))
    speeds = {"new": [], "ref": []}
    for rep in range(5):
        for who, b in (("ref", a.ref), ("new", a.binary)) if rep % 2 == 0 else (("new", a.binary), ("ref", a.ref)):
            if not b: continue
            rc, wall, st, _ = run(b, ["--batch", sl, "--outdir", os.path.join(work, "out_speed"), "--stats"], os.path.join(work, "out_speed"))
            if st.get("mp_s"): speeds[who].append(st["mp_s"])
    gpu, best = gpu_name(), max(speeds["new"], default=0.0)
    print(f"speed: {best:.1f} MP/s on {gpu} (best of {len(speeds['new'])}: {', '.join(f'{x:.0f}' for x in speeds['new'])})")
    if a.ref and speeds["ref"]:
        r = best / max(speeds["ref"])
        print(f"{'SLOWER' if r < 0.95 else 'speed'} against --ref: {r - 1:+.1%} (ref best {max(speeds['ref']):.1f} MP/s)")

    gold, gspeed = {}, {}
    if os.path.exists(GOLDEN):
        for line in open(GOLDEN):
            if line.startswith("#speed\t"):
                _, g, v = line.rstrip("\n").split("\t"); gspeed[g] = float(v)
            elif line.strip() and not line.startswith("#"):
                k, v = line.split(); gold[k] = v

    if a.update:
        gspeed[gpu] = round(best, 1)
        with open(GOLDEN, "w") as f:
            f.write("# tools/sanity.py golden outputs: case/file sha256; '#speed GPU MP/s' per GPU.  Regenerate with --update.\n")
            for g, v in sorted(gspeed.items()): f.write(f"#speed\t{g}\t{v}\n")
            for k in sorted(got): f.write(f"{k} {got[k]}\n")
        print(f"wrote {GOLDEN}: {len(got)} entries"); return

    if not a.ref and gpu in gspeed:
        r = best / gspeed[gpu]
        print(f"{'SLOWER' if r < 0.95 else 'speed'} against the golden figure for {gpu}: {r - 1:+.1%} ({gspeed[gpu]} MP/s)")
    diff = {k for k in set(got) | set(gold) if got.get(k) != gold.get(k)}
    for case in [c[0] for c in CASES] + [c[0] for c in SINGLE]:
        ks = sorted(k for k in diff if k.split("/")[0] == case)
        n = len([k for k in gold if k.split("/")[0] == case and not k.endswith("/exit")])
        if ks:
            ex = f"exit {gold.get(case + '/exit')} -> {got.get(case + '/exit')}" if case + "/exit" in ks else ""
            print(f"DIFF {case}: {len([k for k in ks if not k.endswith('/exit')])} of {n} files {ex}")
    print(f"{'PASS' if not diff else 'FAIL'}: {len(set(got) - diff)}/{len(set(got) | set(gold))} match the golden outputs"
          + f"; {time.time() - t0:.0f} s in all")
    if not a.work: shutil.rmtree(work, ignore_errors=True)
    sys.exit(1 if diff else 0)


if __name__ == "__main__":
    main()
