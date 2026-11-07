#!/usr/bin/env python3
"""GPU memory on a list 5x longer: is the growth a plateau?"""
import json, os, re, shutil, subprocess, sys, threading, time
H = os.path.expanduser("~"); PR, MAIN = f"{H}/pr/build/av1gpu", f"{H}/main/build/av1gpu"; W = f"{H}/lr"
ls = open(f"{W}/long.lst").read().split()
os.makedirs(f"{W}/l5", exist_ok=True)
out = []
for r in range(5):
    for l in ls:
        p = f"{W}/l5/{r}_{os.path.basename(l)}"
        if not os.path.lexists(p): os.symlink(os.path.realpath(l), p)
        out.append(p)
open(f"{W}/l5.lst", "w").write("\n".join(out) + "\n")
def run(tag, b, opts):
    od = f"{W}/o5_{tag}"; shutil.rmtree(od, ignore_errors=True)
    t = time.time()
    p = subprocess.Popen([b, "--batch", f"{W}/l5.lst", "--outdir", od, "--stats"] + opts, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    s = []
    def w():
        while p.poll() is None:
            r = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True)
            try: rss = int(re.search(r"VmRSS:\s+(\d+)", open(f"/proc/{p.pid}/status").read())[1]) // 1024
            except Exception: rss = 0
            s.append((int(r.stdout.strip()), rss)); time.sleep(1)
    th = threading.Thread(target=w); th.start()
    o, e = p.communicate(timeout=1500); th.join()
    q = max(1, len(s) // 4)
    print(tag, "rc", p.returncode, f"{time.time() - t:.0f} s", "files", len(os.listdir(od)), "GPU MiB by quarter", [max(x[0] for x in s[i * q:(i + 1) * q]) for i in range(4)], "RSS by quarter", [max(x[1] for x in s[i * q:(i + 1) * q]) for i in range(4)], flush=True)
    shutil.rmtree(od, ignore_errors=True)
run("main", MAIN, ["--preset", "turbo3"])
run("trim", PR, ["--preset", "turbo3", "--trim", "--fuzz", "10%", "--skip-blank", "99", "--resize", "1000x1000>"])
run("tone", PR, ["--preset", "turbo3", "--auto-level", "--level", "5%,95%,1.1", "--grayscale", "Rec709Luma", "--yuv", "444", "--depth", "10"])
