#!/usr/bin/env python3
"""spd_crop.py: speeds for the crop / resize port, the spd.py method (bench.speed_set repeated to >= 1,000 MP, a
warm-up then 3 timed runs per binary, --stats MP/s, rounds alternating the binary order, median of the runs).
  ab     no new options: main d7d94b3 (~/main) against the branch (~/cr), size range default, 2 rounds
  pages  40 3840x5586 4:4:4 JPEG pages (q94, from 40 size-range photos) repeated 4x, --max-height 2960 on the GPU,
         against the same pages shrunk beforehand by ImageMagick (2035x2960, q94 4:4:4); --yuv 444 --depth 10
         --preset turbo (pickae's settings); output MP/s, GPU memory peak, batches
~/results/spd_crop.jsonl, summary on stdout."""
import glob, json, os, shutil, statistics, subprocess, sys, time

H = os.path.expanduser("~")
sys.path.insert(0, f"{H}/main/bench")
import bench  # noqa: E402
from PIL import Image  # noqa: E402

R, W = f"{H}/results", f"{H}/work/spd"
os.makedirs(W, exist_ok=True)
BIN = {"main": f"{H}/main/build/av1gpu", "cr": f"{H}/cr/build/av1gpu"}
IM = f"{H}/im7/AppRun"


def one(phase, b, src, opts, timed):
    out = f"{W}/out"
    shutil.rmtree(out, ignore_errors=True)
    mem = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "100"],
                           stdout=subprocess.PIPE, text=True)
    t0 = time.time()
    p = subprocess.run([BIN[b], "--batch", src, "--outdir", out, "--stats"] + opts, capture_output=True, text=True)
    t1 = time.time()
    mem.terminate()
    peak = max((int(x) for x in mem.communicate()[0].split() if x.isdigit()), default=0)
    try:
        st = json.loads(p.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        st = {}
    line = next((l for l in p.stderr.splitlines() if "MP/s" in l), "")
    r = dict(phase=phase, bin=b, opts=opts, timed=timed, rc=p.returncode, wall_s=round(t1 - t0, 3), gpu_peak_mib=peak, stats=st, line=line)
    with open(f"{R}/spd_crop.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")
    if p.returncode:
        print("FAILED", phase, b, p.stderr[-500:], flush=True)
    return r


def speed(phase, b, src, opts):
    runs = [one(phase, b, src, opts, i > 0) for i in range(4)][1:]
    mps = [r["stats"].get("mp_s", 0) for r in runs]
    print(f"{phase} {b}: {statistics.median(mps):.1f} MP/s (runs {', '.join(f'{x:.1f}' for x in mps)}), GPU peak "
          f"{max(r['gpu_peak_mib'] for r in runs)} MiB, {runs[0]['line'][:120]}", flush=True)
    return mps


def ab():
    files = sorted(glob.glob(f"{H}/images/*.jpg"))
    sizes = {os.path.basename(f): Image.open(f).size[0] * Image.open(f).size[1] / 1e6 for f in files}
    d, _, mp, rep = bench.speed_set(files, sizes, 1000, f"{W}/sr")
    print(f"size range: {len(files)} photos x {rep} = {mp:.0f} MP", flush=True)
    res = {"main": [], "cr": []}
    for rnd, order in enumerate((("main", "cr"), ("cr", "main"))):
        for b in order:
            res[b] += speed(f"ab_r{rnd + 1}", b, d, [])
    for b in res:
        print(f"ab {b}: median of {len(res[b])} runs {statistics.median(res[b]):.1f} MP/s", flush=True)
    print(f"ab: branch / main = {statistics.median(res['cr']) / statistics.median(res['main']):.4f}", flush=True)


def pages():
    src = sorted(glob.glob(f"{H}/images/*.jpg"), key=lambda f: -Image.open(f).size[0] * Image.open(f).size[1])[:40]
    full, pre = f"{W}/pages_full", f"{W}/pages_pre"
    for d in (full, pre):
        os.makedirs(d, exist_ok=True)
    jobs = []
    for i, f in enumerate(src):
        a, b = f"{full}/p{i:02d}.jpg", f"{pre}/p{i:02d}.jpg"
        if not os.path.exists(a):
            subprocess.run([IM, f, "-resize", "3840x5586!", "-sampling-factor", "1x1", "-quality", "94", a], check=True)
        if not os.path.exists(b):
            subprocess.run([IM, a, "-resize", "x2960", "-sampling-factor", "1x1", "-quality", "94", b], check=True)
    for name, d in (("full", full), ("pre", pre)):
        rd = f"{W}/rep_{name}"
        shutil.rmtree(rd, ignore_errors=True)
        os.makedirs(rd)
        for r in range(4):
            for f in sorted(os.listdir(d)):
                os.symlink(f"{d}/{f}", f"{rd}/r{r}_{f}")
    print(f"pages: {Image.open(full + '/p00.jpg').size} -> {Image.open(pre + '/p00.jpg').size}, 40 x 4", flush=True)
    o = ["--yuv", "444", "--depth", "10", "--preset", "turbo"]
    res = {"gpu_shrink": [], "pre_shrunk": []}
    for rnd in range(2):
        for k in (("gpu_shrink", "pre_shrunk") if rnd == 0 else ("pre_shrunk", "gpu_shrink")):
            if k == "gpu_shrink":
                res[k] += speed(f"pages_{k}_r{rnd + 1}", "cr", f"{W}/rep_full", o + ["--max-height", "2960"])
            else:
                res[k] += speed(f"pages_{k}_r{rnd + 1}", "cr", f"{W}/rep_pre", o)
    for k in res:
        print(f"pages {k}: median of {len(res[k])} runs {statistics.median(res[k]):.1f} MP/s (output)", flush=True)


if __name__ == "__main__":
    for ph in sys.argv[1:]:
        {"ab": ab, "pages": pages}[ph]()
    print("SPD_DONE", flush=True)
