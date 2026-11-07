#!/usr/bin/env python3
"""4:2:0, 4:4:4 and 10-bit tone / gray against real ImageMagick, PNG and JPEG sources.  The PR's own metric (match8.py): --dump-source planes against magick's output converted to
BT.601 full range and rounded; mean abs. difference and largest, in steps of the plane's depth.  Adds p99.9, and chroma against magick's 2x2 mean at 4:2:0."""
import concurrent.futures as cf, glob, json, os, shutil, subprocess, sys, tempfile, collections
import numpy as np
from PIL import Image, ImageDraw
H = os.path.expanduser("~"); BIN = os.environ.get("BIN", f"{H}/pr/build/av1gpu"); MAG = f"{H}/im/bin/magick"
env = dict(os.environ, LD_LIBRARY_PATH=f"{H}/im/lib")
A = os.environ.get("TONEDIR", f"{H}/tone2"); shutil.rmtree(A, ignore_errors=True); os.makedirs(A)
# coloured line art, as the contrarian's model
im = Image.new("RGB", (400, 600), (255, 255, 255)); d = ImageDraw.Draw(im); cols = [(220, 20, 30), (20, 60, 200), (240, 200, 20), (20, 160, 70)]
for i in range(120):
    c = cols[i % 4]; x = (i * 37) % 380; y = (i * 53) % 580
    d.line((x, y, x + 40, y + 25), fill=c, width=1 + (i % 3)); d.ellipse((x, y + 10, x + 14, y + 24), outline=c)
im.save(f"{A}/lineart.png"); im.save(f"{A}/lineart_q90_420.jpg", quality=90, subsampling=2); im.save(f"{A}/lineart_q90_444.jpg", quality=90, subsampling=0)
comics = sorted(glob.glob(f"{H}/comics/*.jpg"))
photos = sorted(glob.glob(f"{H}/images_std/*.jpg"))[:30]
pages = []
for f in comics[:6]:
    p = f"{A}/page_{os.path.basename(f)[:-4]}.png"
    subprocess.run([MAG, f, "-resize", "1050x1500", "-depth", "8", f"PNG24:{p}"], check=True, env=env); pages.append(p)
groups = {"line art PNG": [f"{A}/lineart.png"], "line art JPEG 4:2:0 q90": [f"{A}/lineart_q90_420.jpg"], "line art JPEG 4:4:4 q90": [f"{A}/lineart_q90_444.jpg"],
          "comic pages JPEG": comics, "comic pages PNG 1050x1500": pages, "photos JPEG": photos, "gate PNG 211x149": sorted(glob.glob(f"{H}/pr/tests/gate_images/*.png"))[:1]}
ops = [("--colorspace Gray", "-colorspace Gray"), ("--grayscale Rec709Luma", "-grayscale Rec709Luma"), ("--grayscale Rec601Luma", "-grayscale Rec601Luma"), ("--grayscale Average", "-grayscale Average"),
       ("--grayscale Lightness", "-grayscale Lightness"), ("--level 0%,99%", "-level 0%,99%"), ("--level 10%,90%", "-level 10%,90%"), ("--level 20%,80%,0.5", "-level 20%,80%,0.5"),
       ("--gamma 0.7", "-gamma 0.7"), ("--gamma 2", "-gamma 2"), ("--negate", "-negate"), ("--auto-level", "-auto-level")]
modes = {"4:2:0": [], "4:4:4": ["--yuv", "444"], "4:4:4 10-bit": ["--yuv", "444", "--depth", "10"], "4:2:0 10-bit": ["--depth", "10"]}
if os.environ.get("MODES"): modes = {k: v for k, v in modes.items() if k in os.environ["MODES"].split(",")}
sys.path.insert(0, os.path.expanduser("~/t"))
from trimref_lib import chroma_up
def luma(a): return .299 * a[..., 0] + .587 * a[..., 1] + .114 * a[..., 2]
def cb(a): return -0.168735892 * a[..., 0] - 0.331264108 * a[..., 1] + 0.5 * a[..., 2]
def cr(a): return 0.5 * a[..., 0] - 0.418687589 * a[..., 1] - 0.081312411 * a[..., 2]
def stats(e): e = np.abs(e); return float(e.mean()), float(np.percentile(e, 99.9)), float(e.max())
def one(j):
    grp, src, av, imo, mn = j
    d = tempfile.mkdtemp(dir="/dev/shm")
    try:
        ten = "10-bit" in mn
        depth = "16" if ten else "8"
        ref = f"{d}/ref.png"
        r = subprocess.run([MAG, src] + imo.split() + ["-colorspace", "sRGB", "-type", "TrueColor", "-depth", depth, f"PNG24:{ref}" if not ten else f"PNG48:{ref}"], capture_output=True, text=True, env=env)
        if r.returncode: return (grp, av, mn, "magick", r.stderr[:100])
        # read 16-bit through magick raw to keep the depth
        raw = subprocess.run([MAG, ref, "-depth", depth, "rgb:-"], capture_output=True, env=env).stdout
        wh = subprocess.run([MAG, ref, "-format", "%w %h", "info:"], capture_output=True, text=True, env=env).stdout.split()
        w, h = int(wh[0]), int(wh[1])
        a = np.frombuffer(raw, "<u2" if ten else np.uint8).reshape(h, w, 3).astype(np.float64)
        m = 1023 / 65535 if ten else 1.0
        a = a * m
        ds = f"{d}/s.yuv"
        p = subprocess.run([BIN, src, f"{d}/o.avif", "--dump-source", ds, "--small-preset", "none", "--preset", "turbo3", "--max-mp", "60"] + modes[mn] + av.split(), capture_output=True, text=True, timeout=200)
        if p.returncode: return (grp, av, mn, "rc", p.stderr[-100:])
        b = np.frombuffer(open(ds, "rb").read(), "<u2" if ten else np.uint8)
        top = 1023 if ten else 255
        Y = np.clip(np.rint(luma(a)), 0, top)
        y = b[:w * h].reshape(h, w).astype(np.float64)
        sy = stats(y - Y)
        out = dict(y=sy)
        rest = b[w * h:]
        if len(rest) and "4:2:0" in mn:
            cw, ch = (w + 1) // 2, (h + 1) // 2
            mid = 512 if ten else 128
            def blk(c):
                c2 = np.pad(c, ((0, h % 2), (0, w % 2)), mode="edge")
                return c2.reshape(ch, 2, cw, 2).mean((1, 3))
            U = rest[:cw * ch].reshape(ch, cw).astype(np.float64) - mid; V = rest[cw * ch:2 * cw * ch].reshape(ch, cw).astype(np.float64) - mid
            out["u"] = stats(U - blk(cb(a))); out["v"] = stats(V - blk(cr(a)))
        elif len(rest):
            mid = 512 if ten else 128
            U = rest[:w * h].reshape(h, w).astype(np.float64) - mid; V = rest[w * h:2 * w * h].reshape(h, w).astype(np.float64) - mid
            out["u"] = stats(U - cb(a)); out["v"] = stats(V - cr(a))
        # what a viewer sees: the planes to RGB as a decoder does (4:2:0: triangle upsampling, as libjpeg / libyuv's
        # bilinear), clipped; its luma against magick's
        if len(rest):
            if "4:2:0" in mn:
                cw, ch = (w + 1) // 2, (h + 1) // 2
                Uf = chroma_up(rest[:cw * ch].reshape(ch, cw).astype(np.int64), w, h).astype(np.float64); Vf = chroma_up(rest[cw * ch:2 * cw * ch].reshape(ch, cw).astype(np.int64), w, h).astype(np.float64)
            else:
                Uf = rest[:w * h].reshape(h, w).astype(np.float64); Vf = rest[w * h:2 * w * h].reshape(h, w).astype(np.float64)
            mid = 512 if ten else 128
            yy, cbb, crr = y / top, (Uf - mid) / top, (Vf - mid) / top
            dr = np.clip(np.dstack([yy + 1.402 * crr, yy - 0.344136286 * cbb - 0.714136286 * crr, yy + 1.772 * cbb]), 0, 1) * top
        else:
            dr = np.dstack([y, y, y])
        out["rgbluma"] = stats(luma(dr) - luma(a)); out["rgbmax"] = stats(np.abs(dr - a).max(2))
        return (grp, av, mn, "ok", out, os.path.basename(src))
    except Exception as e:
        return (grp, av, mn, "err", repr(e)[:160])
    finally:
        shutil.rmtree(d, ignore_errors=True)
jobs = [(g, s, a, b, m) for g, ss in groups.items() for s in ss for a, b in ops for m in modes]
agg = collections.defaultdict(list); errs = []
with cf.ThreadPoolExecutor(int(os.environ.get('WORKERS', '6'))) as tp:
    for r in tp.map(one, jobs):
        if r[3] == "ok": agg[(r[0], r[1], r[2])].append((r[4], r[5]))
        else: errs.append(r)
print("| source | operation | format | images | luma mean | luma p99.9 | luma max | worst image | chroma mean | chroma max | decoded luma mean | decoded luma max | decoded RGB channel mean | decoded RGB channel max |"); print("|---|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|")
order = list(groups)
for k in sorted(agg, key=lambda k: (order.index(k[0]), [a for a, _ in ops].index(k[1]), list(modes).index(k[2]))):
    v = agg[k]; ym = np.mean([x[0]["y"][0] for x in v]); yp = max(x[0]["y"][1] for x in v); ymx = max(x[0]["y"][2] for x in v); w = max(v, key=lambda x: x[0]["y"][2])[1]
    cm = [x[0][c][0] for x in v for c in ("u", "v") if c in x[0]]; cx = [x[0][c][2] for x in v for c in ("u", "v") if c in x[0]]
    rl = np.mean([x[0]["rgbluma"][0] for x in v]); rlx = max(x[0]["rgbluma"][2] for x in v); rc = np.mean([x[0]["rgbmax"][0] for x in v]); rcx = max(x[0]["rgbmax"][2] for x in v)
    print(f"| {k[0]} | `{k[1]}` | {k[2]} | {len(v)} | {ym:.2f} | {yp:.1f} | {ymx:.1f} | {w[:32]} | {np.mean(cm) if cm else 0:.2f} | {max(cx) if cx else 0:.1f} | {rl:.2f} | {rlx:.1f} | {rc:.2f} | {rcx:.1f} |")
for e in errs[:12]: print("ERR", e)
print("TONE2_DONE", len(errs), "errors")
