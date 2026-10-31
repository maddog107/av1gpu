"""Score NVENC AVIFs with SSIMULACRA2 exactly as the published NVENC run did (scratchpad/l4n/score.py: avifdec -c dav1d,
padding cropped, ssimulacra2 against the source JPEG) and with CVVDP through bench.Cvvdp on the same decoded pixels.
nvenc_score.py REPO QDIR OUT.json DISPLAY"""
import concurrent.futures as cf, json, os, sys, tempfile
from PIL import Image, ImageOps
REPO, Q, OUT, DISPLAY = sys.argv[1:5]
sys.path.insert(0, os.path.join(REPO, "bench"))
import bench
SRC = os.path.expanduser("~/images_gamut")
WORK = os.path.expanduser("~/work_nvscore"); os.makedirs(WORK, exist_ok=True)
AVIFDEC = os.path.expanduser("~/opt/bin/avifdec"); SSIM2 = os.path.expanduser("~/opt/bin/ssimulacra2")
cv = bench.Cvvdp(DISPLAY)
notes = []


def score(job):
    img, cq, path = job
    with tempfile.TemporaryDirectory(dir=WORK) as td:
        png = os.path.join(td, "d.png")
        if bench.run([AVIFDEC, "-c", "dav1d", path, png]).returncode:
            notes.append(f"{img} q{cq}: avifdec failed")
            return dict(image=img, q=cq, bytes=os.path.getsize(path), ssimulacra2=None, cvvdp=None)
        ref = os.path.join(SRC, img)
        src, out = Image.open(ref), Image.open(png).convert("RGB")
        if out.size != src.size:
            if (out.width - src.width) in (0, 1) and (out.height - src.height) in (0, 1):
                out = out.crop((0, 0, src.width, src.height))       # NVENC pads odd sizes to even
            else:                                                     # ffmpeg applied EXIF rotation
                notes.append(f"{img} q{cq}: rotated {src.size} -> {out.size}")
                icc = src.info.get("icc_profile")
                src = ImageOps.exif_transpose(src); ref = os.path.join(td, "r.png")
                src.convert("RGB").save(ref)                          # as published (no ICC) for SSIMULACRA2
                out = out.crop((0, 0, src.width, src.height))
                if icc:                                               # CVVDP keeps the source's profile
                    src.convert("RGB").save(os.path.join(td, "ri.png"), icc_profile=icc)
        o = os.path.join(td, "o.png"); out.save(o)
        p = bench.run([SSIM2, ref, o])
        try:
            s = float(p.stdout.split()[-1])
        except (ValueError, IndexError):
            s = None
        cref = os.path.join(td, "ri.png") if os.path.exists(os.path.join(td, "ri.png")) else ref
        j = cv(cref, o, path)
    return dict(image=img, q=cq, bytes=os.path.getsize(path), ssimulacra2=s, cvvdp=j)


jobs = []
for f in sorted(os.listdir(Q)):
    img, cq = f.rsplit(".q", 1)
    jobs.append((img, int(cq[:-5]), os.path.join(Q, f)))
with cf.ThreadPoolExecutor(16) as ex:
    rows = list(ex.map(score, jobs))
json.dump(dict(cvvdp=cv.info, notes=notes, rows=rows), open(OUT, "w"), indent=1)
print(len(rows), "rows;", len(notes), "notes", notes[:5])
