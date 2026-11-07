#!/usr/bin/env python3
"""mem65.py TAG RUNS SPECS(NAME=PATH,...): peak GPU memory of each binary per set: nvidia-smi samples memory.used every 50 ms during
the run, minus the idle value before it; plus the run's own peak_dev_mb and peak RSS.  Not a timing run (the sampler costs a little).
Sets as spd65.py (the speed sets left in ~/work/set_*), and the single files photo.jpg and comic.jpg."""
import json, os, statistics as st, subprocess, sys, time, shutil
H = os.path.expanduser("~"); TAG, N = sys.argv[1], int(sys.argv[2])
SPECS = [tuple(s.split("=", 1)) for s in sys.argv[3].split(",")]
C444 = "--yuv 444 --depth 10 --tiles 2,2,2 --max-mp 100 --max-input-mp 100".split()
def used():
    return int(subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout.strip())
def run(cmd):
    idle = used()
    smp = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-lms", "50"], stdout=subprocess.PIPE, text=True)
    time.sleep(0.2)
    so = open(f"{H}/work/mo.txt", "wb"); se = open(f"{H}/work/me.txt", "wb")
    p = subprocess.Popen(cmd, stdout=so, stderr=se); _, status, ru = os.wait4(p.pid, 0)
    time.sleep(0.2); smp.terminate(); out = smp.communicate()[0]; so.close(); se.close()
    v = [int(x) for x in out.split() if x.isdigit()]
    try: s = json.loads(open(f"{H}/work/mo.txt").read().strip().splitlines()[-1])
    except Exception: s = {}
    return max(v) - idle, s.get("peak_dev_mb", 0), ru.ru_maxrss / 1024
CASES = [(n, ["--batch", f"{H}/work/set_{d}/speedset", "--outdir", "/dev/shm/mo", "--stats"] + ex) for n, d, ex in
         (("sr", "sr", []), ("std", "std", []), ("thumbs", "thumbs", []), ("comics", "comics", []), ("pngwebp", "pngwebp", []), ("c444", "c444", C444))]
CASES += [("photo", [f"{H}/sets/st/photo.jpg", "/dev/shm/mo.avif"]), ("comic", [f"{H}/sets/st/comic.jpg", "/dev/shm/mo.avif"])]
for name, a in CASES:
    line = f"{name:8s}"
    res = {n: [] for n, _ in SPECS}
    for r in range(N):
        o = SPECS if r % 2 == 0 else SPECS[::-1]
        for n, p in o:
            shutil.rmtree("/dev/shm/mo", ignore_errors=True)
            res[n].append(run([p] + a))
    for n, _ in SPECS:
        g = [x[0] for x in res[n]]; d = [x[1] for x in res[n]]; m = [x[2] for x in res[n]]
        line += f" | {n}: gpu used {st.median(g):6.0f} MiB (max {max(g)}) peak_dev {st.median(d):7.0f} rss {st.median(m):6.0f}"
    print(line, flush=True)
print("MEM_DONE", TAG, flush=True)
