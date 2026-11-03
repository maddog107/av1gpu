# Re-score the standard set's WebP row with each .webp's ICC profile (bench.py's fix), against the L4 run's libaom s6.
# usage: webp_icc_std.py REPO IMAGES WORK
import concurrent.futures as cf, json, os, statistics, subprocess, sys, tempfile, types
repo, imgs, work = sys.argv[1:4]
sys.path.insert(0, os.path.join(repo, "bench"))
import bench

a = types.SimpleNamespace(dwebp="dwebp", ssimulacra2="ssimulacra2", avifdec="avifdec")
res = json.load(open(os.path.join(repo, "bench/results/aws-g6-l4/results.json")))
old = {(r["image"], r["q"]): r for r in res["quality"]["webp"]}
os.makedirs(work, exist_ok=True)


def one(job):
    im, q = job
    src, dst = os.path.join(imgs, im), os.path.join(work, f"{im}.q{q}.webp")
    subprocess.run(["cwebp", "-quiet", "-q", str(q), "-m", "4", "-metadata", "icc", src, "-o", dst], check=True)
    fixed = bench.score(a, src, dst, work)
    with tempfile.TemporaryDirectory(dir=work) as td:  # the first published run: dwebp's PNG as it is
        png = os.path.join(td, "d.png")
        subprocess.run(["dwebp", "-quiet", dst, "-o", png], check=True)
        p = subprocess.run(["ssimulacra2", src, png], capture_output=True, text=True)
        nofix = float(p.stdout.strip().split()[-1])
    return dict(image=im, q=q, bytes=os.path.getsize(dst), fixed=fixed, nofix=nofix, icc=bool(bench.file_icc(dst)))


jobs = sorted(old)
with cf.ThreadPoolExecutor(os.cpu_count()) as ex:
    rows = list(ex.map(one, jobs))
json.dump(rows, open(os.path.join(work, "rows.json"), "w"))
same_bytes = sum(r["bytes"] == old[(r["image"], r["q"])]["bytes"] for r in rows)
same_nofix = sum(abs(r["nofix"] - old[(r["image"], r["q"])]["ssimulacra2"]) < 1e-3 for r in rows)
print(f"{len(rows)} encodes: bytes equal to the L4 run's {same_bytes}, unfixed score equal {same_nofix}")

aom = {}
for r in res["quality"]["aom_s6"]:
    aom.setdefault(r["image"], []).append((r["bytes"], r["ssimulacra2"]))
per = {}
for r in rows:
    per.setdefault(r["image"], {"old": [], "nofix": [], "fixed": [], "icc": r["icc"]})
    per[r["image"]]["old"].append((old[(r["image"], r["q"])]["bytes"], old[(r["image"], r["q"])]["ssimulacra2"]))
    per[r["image"]]["nofix"].append((r["bytes"], r["nofix"]))
    per[r["image"]]["fixed"].append((r["bytes"], r["fixed"]))
out = {}
for k in ("old", "nofix", "fixed"):
    v = {im: bench.bd_rate(p[k], aom[im]) for im, p in per.items()}
    v = {im: x for im, x in v.items() if x is not None}
    out[k] = v
    print(f"webp vs aom_s6 [{k}]: n={len(v)} median {100 * statistics.median(v.values()):+.1f} % "
          f"mean {100 * statistics.fmean(v.values()):+.1f} %")
print("images whose .webp has a profile:", sum(p["icc"] for p in per.values()))
print("largest changes (old -> fixed):")
ch = sorted(((out["fixed"].get(im, 0) - out["old"].get(im, 0)), im) for im in per)
for d, im in ch[:16]:
    print(f"  {im} {100 * out['old'].get(im, float('nan')):+.1f} -> {100 * out['fixed'].get(im, float('nan')):+.1f} icc={per[im]['icc']}")
