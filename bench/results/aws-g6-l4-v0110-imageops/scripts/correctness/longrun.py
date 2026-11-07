#!/usr/bin/env python3
"""long lists: thousands of tiny images, GPU memory over a long list, with the new options and without, PR and main.  Run alone."""
import glob, json, os, random, re, shutil, subprocess, sys, threading, time
import numpy as np
from PIL import Image
sys.path.insert(0, os.path.expanduser("~/t"))
from gen_lib import wpng
import avt
H = os.path.expanduser("~"); PR, MAIN = f"{H}/pr/build/av1gpu", f"{H}/main/build/av1gpu"
W = f"{H}/lr"; shutil.rmtree(W, ignore_errors=True); os.makedirs(W)
rng = np.random.default_rng(3)
# tiny images
tiny = f"{W}/tiny"; os.makedirs(tiny)
files = []
for i in range(6000):
    w, h = int(rng.integers(1, 34)), int(rng.integers(1, 34))
    a = np.full((h, w, 3), 255, np.uint8)
    k = i % 5
    if k: a[h // 4: max(h // 4 + 1, 3 * h // 4), w // 4: max(w // 4 + 1, 3 * w // 4)] = rng.integers(0, 256, 3)
    p = f"{tiny}/{i:05d}" + (".png" if i % 3 else ".jpg")
    if i % 3: wpng(p, a)
    else: Image.fromarray(a).save(p, quality=95)
    files.append(p)
open(f"{W}/tiny.lst", "w").write("\n".join(files) + "\n")
# a long list of real pages: photos + comics repeated through links
lng = f"{W}/long"; os.makedirs(lng)
src = sorted(glob.glob(f"{H}/images_std/*.jpg")) + sorted(glob.glob(f"{H}/images/*.jpg"))[:40] + sorted(glob.glob(f"{H}/comics/*.jpg"))[:10] + sorted(glob.glob(f"{H}/adv/alpha/*120x80.png"))
ls = []
for r in range(14):
    for i, f in enumerate(src):
        l = f"{lng}/{r:02d}_{i:03d}_{os.path.basename(f)}"; os.symlink(f, l); ls.append(l)
open(f"{W}/long.lst", "w").write("\n".join(ls) + "\n")
print(len(ls), "files in the long list", flush=True)

def watch(p, samples):
    while p.poll() is None:
        r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        try:
            ps = open(f"/proc/{p.pid}/status").read()
            rss = int(re.search(r"VmRSS:\s+(\d+)", ps)[1]) // 1024
        except Exception: rss = 0
        samples.append((time.time(), int(r.stdout.strip()), rss))
        time.sleep(2)

def run(tag, b, lst, opts):
    od = f"{W}/o_{tag}"; shutil.rmtree(od, ignore_errors=True)
    t = time.time()
    p = subprocess.Popen([b, "--batch", lst, "--outdir", od, "--stats"] + opts, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    s = []; th = threading.Thread(target=watch, args=(p, s)); th.start()
    o, e = p.communicate(timeout=3000); th.join()
    nf = len(os.listdir(od)) if os.path.isdir(od) else 0
    m = [x[1] for x in s]; r = [x[2] for x in s]
    third = max(1, len(m) // 3)
    print(f"{tag}: rc {p.returncode} {time.time() - t:.0f} s files {nf}; GPU MiB first-third max {max(m[:third], default=0)} last-third max {max(m[-third:], default=0)}; RSS MiB first {max(r[:third], default=0)} last {max(r[-third:], default=0)}; " + (re.search(r"\[av1gpu\] trim:.*", e) or [""])[0] + " " + (re.search(r"peak dev \d+ MB", e) or [""])[0], flush=True)
    json.dump(dict(tag=tag, samples=s, stderr=e[-3000:]), open(f"{H}/results/lr_{tag}.json", "w"))
    if p.returncode: print(e[-500:])
    return od

run("tiny_main", MAIN, f"{W}/tiny.lst", [])
run("tiny_pr", PR, f"{W}/tiny.lst", [])
od = run("tiny_opts", PR, f"{W}/tiny.lst", ["--trim", "--fuzz", "5%", "--skip-blank", "99", "--auto-level", "--resize", "200%", "--grayscale", "Average"])
# sample decode
bad = 0
for f in random.Random(1).sample(sorted(os.listdir(od)), 200):
    try: avt.check(f"{od}/{f}")
    except avt.Bad as e: bad += 1; print("BAD", f, e)
print("tiny: 200 sampled outputs decoded,", bad, "bad", flush=True)
run("long_main", MAIN, f"{W}/long.lst", ["--preset", "turbo3"])
run("long_pr", PR, f"{W}/long.lst", ["--preset", "turbo3"])
run("long_trim", PR, f"{W}/long.lst", ["--preset", "turbo3", "--trim", "--fuzz", "10%", "--skip-blank", "99", "--resize", "1000x1000>"])
run("long_tone", PR, f"{W}/long.lst", ["--preset", "turbo3", "--auto-level", "--level", "5%,95%,1.1", "--grayscale", "Rec709Luma", "--yuv", "444", "--depth", "10"])
run("long_enl", PR, f"{W}/long.lst", ["--preset", "turbo3", "--resize", "1500x1500", "--trim", "--fuzz", "10%"])
print("LR_DONE")
