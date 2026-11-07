#!/usr/bin/env python3
"""cold65.py TAG N SPECS: cold-process timing.  SPECS: NAME=PATH,...  Each case runs one process per run, N timed runs per
binary, binaries rotating run by run (and reversed on odd rounds), one warm-up each.  Cases:
 single files (`bin IN OUT`), wall = time to the output file, since the process ends right after writing it;
 batch_two: `--batch` of the photo and its thumbnail (the upload case, EAGER module loading), wall and time to first output (polled at 0.2 ms).
Run with nothing else on the GPU.  JSON lines in ~/results/cold_TAG.jsonl."""
import json, os, shutil, statistics as st, subprocess, sys, time
H = os.path.expanduser("~"); TAG, N = sys.argv[1], int(sys.argv[2])
SPECS = [tuple(s.split("=", 1)) for s in sys.argv[3].split(";" if ";" in sys.argv[3] else ",")]
ONLY = sys.argv[4:]   # case names; none: all
I = f"{H}/sets/st"; OD = "/dev/shm/cold"; LOG = f"{H}/results/cold_{TAG}.jsonl"
C = [("photo_2.5MP", [f"{I}/photo.jpg"], []), ("big_10.5MP", [f"{I}/big.jpg"], []), ("d512", [f"{I}/d512.jpg"], []), ("t320", [f"{I}/t320.jpg"], []),
     ("comic", [f"{I}/comic.jpg"], []), ("comic_444_10", [f"{I}/comic.jpg"], "--yuv 444 --depth 10 --tiles 2,2,2 --max-mp 100 --max-input-mp 100".split()),
     ("page_png", [f"{I}/page.png"], []), ("alpha_png", [f"{I}/alpha.png"], [])]
LIST = f"{H}/work/two.txt"
open(LIST, "w").write(f"{I}/photo.jpg\n{I}/d512.jpg\n")
def run_single(path, inp, ex):
    os.makedirs(OD, exist_ok=True)
    o = f"{OD}/o.avif"
    if os.path.exists(o): os.remove(o)
    t = time.perf_counter()
    b, _, xo = path.partition("|")
    r = subprocess.run([b, inp, o] + ex + xo.split(), stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    d = time.perf_counter() - t
    if r.returncode not in (0, 3): print("FAIL", path, inp, r.returncode, r.stderr[-300:], flush=True)
    return dict(wall=d * 1e3)
def run_batch(path):
    shutil.rmtree(OD, ignore_errors=True); os.makedirs(OD)
    t = time.perf_counter()
    p = subprocess.Popen([path, "--batch", LIST, "--outdir", OD], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    first = None
    while p.poll() is None:
        if first is None and any(True for e in os.scandir(OD) if e.name.endswith(".avif")):
            first = time.perf_counter() - t
        time.sleep(0.0002)
    d = time.perf_counter() - t
    return dict(wall=d * 1e3, first=(first or d) * 1e3)
def case(name, fn):
    res = {n: [] for n, _ in SPECS}
    for n, p in SPECS: fn(p)
    for r in range(N):
        o = SPECS[r % len(SPECS):] + SPECS[:r % len(SPECS)]
        if r % 2: o = o[::-1]
        for n, p in o:
            x = fn(p); x.update(case=name, spec=n, rnd=r); res[n].append(x)
            with open(LOG, "a") as f: f.write(json.dumps(x) + "\n")
    base = SPECS[0][0]; m0 = st.median(x["wall"] for x in res[base])
    line = f"{name:14s}"
    for n, _ in SPECS:
        m = st.median(x["wall"] for x in res[n]); s = sorted(x["wall"] for x in res[n])
        line += f" | {n} {m:7.1f} ({s[len(s) // 4]:.1f}-{s[3 * len(s) // 4]:.1f}) {m - m0:+6.1f}"
        if "first" in res[n][0]: line += f" first {st.median(x['first'] for x in res[n]):7.1f}"
    print(line, flush=True)
for name, inp, ex in C:
    if not ONLY or name in ONLY:
        case(name, lambda p, inp=inp, ex=ex: run_single(p, inp[0], ex))
if not ONLY or "batch_two" in ONLY:
    case("batch_two", run_batch)
print("COLD_DONE", TAG, flush=True)
