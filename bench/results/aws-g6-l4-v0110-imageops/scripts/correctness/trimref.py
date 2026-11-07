#!/usr/bin/env python3
"""--trim boxes and pixels against (a) magick, (b) a Python copy of ImageMagick's GetImageBoundingBox on the exact RGB, (c) the documented algorithm on the
planes the encoder sees (--dump-source).  Every output is decoded in dav1d + aomdec."""
import concurrent.futures as cf, glob, json, os, re, subprocess, sys, tempfile, shutil, collections, random
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.expanduser("~/t"))
import avt
H = os.path.expanduser("~"); MAG = f"{H}/im/bin/magick"; BIN = os.environ.get("BIN", f"{H}/pr/build/av1gpu")
env = dict(os.environ, LD_LIBRARY_PATH=f"{H}/im/lib")
TD = tempfile.mkdtemp(prefix="trimref_", dir=H)
FUZZ = [0, 1, 3, 10, 30, 100]

def im_real(src, f):
    r = subprocess.run([MAG, src, "-fuzz", f"{f}%", "-format", "%@ %w %h", "info:"], capture_output=True, text=True, env=env)
    m = re.match(r"(\d+)x(\d+)\+(-?\d+)\+(-?\d+) (\d+) (\d+)", r.stdout)
    if not m: return None, r.stdout + r.stderr
    bw, bh, x, y, w, h = (int(v) for v in m.groups())
    return (x, y, bw, bh), (w, h)

def im_py(rgb, alpha, f):
    """ImageMagick 7.1.2's GetImageBoundingBox (attribute.c), one thread, on 8-bit samples: rgb H x W x 3 uint8; (x, y, w, h) or None for 0x0"""
    H_, W_ = rgb.shape[:2]
    p = rgb.astype(np.float64) / 255.0
    a = None if alpha is None else alpha.astype(np.float64) / 255.0
    fz2 = 2 * f * f
    def differs(q, qa, Q, Qa):
        # IsFuzzyEquivalencePixelInfo (MagickSafeSignificantError ~ >) vectorised: q H x W x 3, Q 3
        dist = np.zeros(q.shape[:2]); scale = 1.0; fu = fz2
        out = np.zeros(q.shape[:2], bool)
        if a is not None:
            da = qa - Qa
            dist = da * da
            out |= (dist * dist) * 65535.0 ** 2 > fu   # IsFuzzyEquivalencePixelInfo: distance*distance against fuzz (in quantum units)
            scale = qa * Qa
        d3 = dist * 3
        s = scale * ((q - Q) ** 2).sum(2)
        res = out | ((d3 + s) > 3 * fu)
        if a is not None: res = np.where(out, True, np.where(scale <= 1e-12, False, res))
        return res
    c = [(0, 0), (0, W_ - 1), (H_ - 1, 0), (H_ - 1, W_ - 1)]
    dd = [differs(p, a, p[y, x], None if a is None else a[y, x]) for y, x in c]
    bx, bw, by, bh = W_, 0, H_, 0
    for y in range(H_):
        lx, lw, ly, lh = bx, bw, by, bh
        for x in np.nonzero(dd[0][y] | dd[1][y] | dd[2][y] | dd[3][y])[0]:
            x = int(x)
            if x < lx and dd[0][y, x]: lx = x
            if x > lw and dd[1][y, x]: lw = x
            if y < ly and dd[0][y, x]: ly = y
            if y > lh and dd[2][y, x]: lh = y
            if x < lw and y > lh and dd[3][y, x]: lw = x; lh = y
        bx = min(bx, lx); by = min(by, ly); bw = max(bw, lw); bh = max(bh, lh)
    if bw == 0 or bh == 0: return None
    w_, h_ = bw - (bx - 1), bh - (by - 1)
    return (bx, by, w_ if w_ < (1 << 40) else 0, h_ if h_ < (1 << 40) else 0)

def chroma_up(C, w, h):
    """libjpeg-turbo h2v2 fancy upsampling of a (h+1)//2 x (w+1)//2 plane to w x h (the fix head's chroma_up), integers"""
    C = C.astype(np.int64); ch, cw = C.shape
    x = np.arange(w); y = np.arange(h)
    cx = np.minimum(x >> 1, cw - 1); cy = np.minimum(y >> 1, ch - 1)
    nx = np.clip(cx + np.where(x & 1, 1, -1), 0, cw - 1); ny = np.clip(cy + np.where(y & 1, 1, -1), 0, ch - 1)
    here = 3 * C[cy][:, cx] + C[ny][:, cx]; side = 3 * C[cy][:, nx] + C[ny][:, nx]
    return (3 * here + side + np.where(x & 1, 7, 8)[None, :]) >> 4

UPSRC = {}
def jpeg420(path):
    """a JPEG whose chroma is 2x2-subsampled (h2v2), as ingest's INGEST_SUB_420"""
    try:
        im = Image.open(path)
        if im.format != "JPEG" or len(im.layer) < 3: return False
        (_, h0, v0, _), (_, h1, v1, _) = im.layer[0], im.layer[1]
        return h0 // h1 == 2 and v0 // v1 == 2 and im.layer[2][1:3] == im.layer[1][1:3]
    except Exception:
        return False

def plane_rgb(pl, fmt, depth):
    """trim_rgb(): RGB rebuilt from the planes in float32, rounded to whole samples; H x W x 3 float32"""
    m = np.float32(1023 if depth == 10 else 255); mid = np.float32(512 if depth == 10 else 128)
    if fmt == "mono":
        y = pl[0].astype(np.float32) / m; return np.dstack([y, y, y])
    U, V = pl[1], pl[2]
    if fmt == "420":
        hh, ww = pl[0].shape
        if os.environ.get("TRIM_UP") == "1" or (os.environ.get("TRIM_UP") == "auto" and UPSRC.get("up") and (ww + 1) // 2 > 2):
            U = chroma_up(U, ww, hh); V = chroma_up(V, ww, hh)
        else:
            U = np.repeat(np.repeat(U, 2, 0), 2, 1)[:hh, :ww]; V = np.repeat(np.repeat(V, 2, 0), 2, 1)[:hh, :ww]
    yy = pl[0].astype(np.float32) / m; cb = (U.astype(np.float32) - mid) / m; cr = (V.astype(np.float32) - mid) / m
    r = np.clip(yy + np.float32(1.402) * cr, 0, 1); g = np.clip(yy - np.float32(0.344136286) * cb - np.float32(0.714136286) * cr, 0, 1); b = np.clip(yy + np.float32(1.772) * cb, 0, 1)
    rgb = np.dstack([r, g, b])
    return (np.rint(rgb * m) / m).astype(np.float32)

def doc_box(rgb, f, alpha=None):
    """the PR's documented algorithm: (l, t, r, b) as measured, on float RGB"""
    thr = np.float32(6.0 * f * f)
    H_, W_ = rgb.shape[:2]
    def diff(Q, Qa):
        s = 1.0
        d = ((rgb - Q) ** 2).sum(2, dtype=np.float32)
        if alpha is None: return d > thr
        hasa = (alpha != 1) | (Qa != 1)
        neq = alpha != Qa
        s = (alpha * Qa).astype(np.float32)
        return np.where(hasa, np.where(neq, True, np.where(s <= 1e-6, False, s * d > thr)), d > thr)
    corners = [(0, 0), (0, W_ - 1), (H_ - 1, 0), (H_ - 1, W_ - 1)]
    dd = [diff(rgb[y, x], None if alpha is None else alpha[y, x]) for y, x in corners]
    ys, xs = np.nonzero(dd[0])
    l = int(xs.min()) if len(xs) else None; t = int(ys.min()) if len(ys) else None
    ys1, xs1 = np.nonzero(dd[1]); r = int(xs1.max()) if len(xs1) else -1
    ys2, xs2 = np.nonzero(dd[2]); b = int(ys2.max()) if len(ys2) else -1
    ys3, xs3 = np.nonzero(dd[3] & (np.arange(W_)[None, :] < r))
    if len(ys3): b = max(b, int(ys3.max()))
    return l, t, r, b

def decide(box, w, h):
    l, t, r, b = box
    L = w if l is None else l; T = h if t is None else t
    bw = max(r, 0) - L + 1; bh = max(b, 0) - T + 1
    if (r < 0 and b < 0 and l is None) or bw <= 0 or bh <= 0: return (0, 0, w, h)
    return (L, T, bw, bh)

def run(img, tag, opts):
    d = tempfile.mkdtemp(dir=TD)
    o, ds = f"{d}/o.avif", f"{d}/d.yuv"
    p = subprocess.run([BIN, img, o, "--dump-source", ds, "--small-preset", "none", "--preset", "turbo3"] + opts, capture_output=True, text=True, timeout=180)
    return d, o, ds, p

def planes_of(ds, fmt, w, h, depth):
    b = open(ds, "rb").read()
    pl, rest = avt.split(b, fmt, w, h, depth)
    return [x.copy() for x in pl], rest

def one(job):
    img, fz, mode = job
    rec = dict(img=img, fuzz=fz, mode=mode, issues=[])
    try:
        im = Image.open(img); w, h = im.size
        isrgb8 = im.mode in ("RGB", "RGBA") and img.endswith(".png")
        opts = {"ll": ["--lossless"], "444": ["--yuv", "444"], "420": [], "444_10": ["--yuv", "444", "--depth", "10"]}[mode]
        # plain planes
        d0, o0, ds0, p0 = run(img, "plain", opts)
        if p0.returncode: rec["issues"].append(f"plain rc {p0.returncode}: {p0.stderr[-120:]}"); return rec
        c0 = avt.check(o0, d0)
        fmt, sw, sh, depth = c0["si"][:4]
        d1, o1, ds1, p1 = run(img, "trim", opts + ["--trim", "--fuzz", f"{fz}%"])
        if p1.returncode: rec["issues"].append(f"trim rc {p1.returncode}: {p1.stderr[-120:]}"); return rec
        c1 = avt.check(o1, d1)
        kw, kh = c1["si"][1], c1["si"][2]
        rec["kept"] = (kw, kh)
        real, _ = im_real(img, fz)
        im_kept = (w, h) if real is None or real[2] == 0 or real[3] == 0 else (real[2], real[3])
        rec["im"] = real
        if (kw, kh) != im_kept:
            rec["issues"].append(f"size differs from magick: av1gpu {kw}x{kh}, magick {real}")
        if mode == "ll" and isrgb8:
            a = np.array(im.convert("RGBA")); rgb = a[..., :3]; al = a[..., 3] if im.mode == "RGBA" else None
            pyb = im_py(rgb, al, fz / 100)
            rec["py"] = pyb
            if (pyb or (0, 0, 0, 0))[2:] != ((real or (0, 0, 0, 0))[2:]) or (pyb and real and pyb[:2] != real[:2] and real[2]):
                rec["issues"].append(f"my python model differs from magick: {pyb} vs {real}")
            # the decoded pixels must be a crop of the source at the kept rectangle
            dec = np.rint(avt.to_rgb(c1["si"], c1["planes"]) * 255).astype(np.uint8)
            target = None if real is None or real[2] == 0 else real
            if target is not None and (kw, kh) == target[2:]:
                x, y, bw, bh = target
                if not np.array_equal(dec, rgb[y:y + bh, x:x + bw]): rec["issues"].append(f"pixels are not the source crop at {target}")
        # the documented algorithm on the planes
        UPSRC["up"] = jpeg420(img)
        pl, rest = planes_of(ds0, fmt, sw, sh, depth)
        if mode == "ll":
            r_ = plane_rgb(pl, fmt, depth) if fmt == "mono" else (np.dstack([pl[2], pl[0], pl[1]]).astype(np.float32) / (1023 if depth == 10 else 255))
        else:
            r_ = plane_rgb(pl, fmt, depth)
        alpha = None
        if len(rest):
            alpha = rest.reshape(sh, sw).astype(np.float32) / (1023 if depth == 10 else 255)
        b = doc_box(r_, fz / 100, alpha)
        x, y, bw, bh = decide(b, sw, sh)
        rec["doc"] = (x, y, bw, bh)
        if (kw, kh) != (bw, bh): rec["issues"].append(f"size differs from the documented algorithm on the planes: {kw}x{kh} vs {bw}x{bh} (box {b})")
        else:
            pk, _ = planes_of(ds1, fmt, kw, kh, depth)
            if not np.array_equal(pk[0], pl[0][y:y + bh, x:x + bw]): rec["issues"].append(f"Y plane is not the crop at {(x, y, bw, bh)}")
            if fmt == "444" and not np.array_equal(pk[1], pl[1][y:y + bh, x:x + bw]): rec["issues"].append("U plane is not the crop")
    except avt.Bad as e:
        rec["issues"].append("BAD " + str(e))
    except Exception as e:
        rec["issues"].append("harness " + repr(e))
    finally:
        pass
    for dd in ("d0", "d1"):
        v = locals().get(dd)
        if v: shutil.rmtree(v, ignore_errors=True)
    return rec

def main():
    sets = sys.argv[1:] or ["trim"]
    jobs = []
    rnd = random.Random(7)
    if "trim" in sets:
        imgs = sorted(glob.glob(f"{H}/adv/trim/*.png"))
        fl = sorted(glob.glob(f"{H}/adv/flat/*.png")); rnd.shuffle(fl); imgs += fl[:60]
        jobs += [(i, f, m) for i in imgs for f in FUZZ for m in ("ll", "444", "420")]
    if "alpha" in sets:
        imgs = [f for f in sorted(glob.glob(f"{H}/adv/alpha/*.png")) if "16" not in f and "/la" not in f]
        jobs += [(i, f, m) for i in imgs for f in FUZZ for m in ("ll", "444", "420")]
    if "deep" in sets:
        imgs = sorted(glob.glob(f"{H}/adv/deep/*.png")) + sorted(glob.glob(f"{H}/adv/alpha/*16*.png"))
        jobs += [(i, f, m) for i in imgs for f in (0, 3, 10) for m in ("444", "420", "444_10")]
    if "jpeg" in sets:
        imgs = [f for f in sorted(glob.glob(f"{H}/adv/jpeg/*.jpg")) if "cmyk" not in f]
        jobs += [(i, f, m) for i in imgs for f in (0, 3, 10) for m in ("444", "420", "444_10")]
    if "photos" in sets:
        imgs = sorted(glob.glob(f"{H}/comics/*.jpg")) + sorted(glob.glob(f"{H}/images_std/*.jpg")) + sorted(glob.glob(f"{H}/images/*.jpg"))[:60]
        jobs += [(i, f, m) for i in imgs for f in (0, 3, 10) for m in ("444", "420")]
    rnd.shuffle(jobs)
    print(len(jobs), "jobs", flush=True)
    cnt = collections.Counter(); ex = {}
    out = open(f"{H}/results/trimref_{'_'.join(sets)}{os.environ.get('TAG', '')}.jsonl", "w")
    with cf.ThreadPoolExecutor(int(os.environ.get("JOBS", "4"))) as tp:
        for i, r in enumerate(tp.map(one, jobs)):
            out.write(json.dumps(r) + "\n"); out.flush()
            cnt["jobs"] += 1
            for iss in r["issues"]:
                k = (r["mode"], re.sub(r"[\d.]+", "N", iss)[:70])
                cnt[k] += 1
                ex.setdefault(k, (os.path.basename(r["img"]), r["fuzz"], iss[:200]))
            if i % 200 == 199: print(i + 1, flush=True)
    print(cnt["jobs"], "jobs")
    for k, v in sorted(((k, v) for k, v in cnt.items() if k != "jobs"), key=lambda kv: -kv[1]):
        print(v, k, ex[k])
    shutil.rmtree(TD, ignore_errors=True)
main()
