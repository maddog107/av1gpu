#!/usr/bin/env python3
"""CPU-thread tests (instance).  tthr.py hang|aff|sweep [rounds]

Every run goes through `timeout`; results -> ~/results/thr/<phase>.jsonl.  Output md5s are compared per
(list, args) with the first main run at its defaults (the reference), which is saved to ~/results/thr/ref_*.json.
"""
import glob, hashlib, json, os, re, shutil, subprocess, sys, threading, time

H = os.path.expanduser("~")
D, R, TMP = f"{H}/data", f"{H}/results/thr", f"{H}/tmp/thr"
BIN = {"main": f"{H}/bin/av1gpu-main", "new": f"{H}/bin/av1gpu-new", "bf": f"{H}/bin/av1gpu-bf"}
os.makedirs(R, exist_ok=True)
os.makedirs(TMP, exist_ok=True)
N = len(os.sched_getaffinity(0))
SUM = re.compile(r"\[av1gpu\] (\d+) images \(([\d.]+) MP\) in ([\d.]+) ms: [\d.]+ img/s, ([\d.]+) MP/s; (\d+) batches;.*"
                 r"CPU [\d.]+ ms = ([\d.]+) cores; skipped \(unsupported\): (\d+); WebP/PNG via CPU decode: (\d+); write failures (\d+)")
REFS = {}


def md5s(d):
    return {os.path.basename(p): hashlib.md5(open(p, "rb").read()).hexdigest() for p in glob.glob(f"{d}/*.avif")}


def gpu_mem_sampler(stop, out):
    p = subprocess.Popen(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits", "-lms", "250"],
                         stdout=subprocess.PIPE, text=True)
    mx, ut = 0, []
    for line in p.stdout:
        try:
            m, u = (int(x) for x in line.split(","))
            mx = max(mx, m)
            ut.append(u)
        except ValueError:
            pass
        if stop.is_set():
            break
    p.terminate()
    out["gpu_mem_max_mb"] = mx
    out["gpu_util_mean"] = round(sum(ut) / len(ut), 1) if ut else None


def run(b, lst, args=(), env=None, timeout=900, prefix=(), tag="run", hash_=True):
    out = f"{TMP}/{b}_{tag}"
    shutil.rmtree(out, ignore_errors=True)
    os.makedirs(out)
    e = dict(os.environ)
    e.update(env or {})
    cmd = list(prefix) + ["timeout", "-k", "10", str(timeout), BIN[b], "--batch", f"{D}/{lst}.txt", "--outdir", out, "--stats"] + list(args)
    stop, g = threading.Event(), {}
    th = threading.Thread(target=gpu_mem_sampler, args=(stop, g))
    th.start()
    t0 = time.time()
    p = subprocess.run(cmd, env=e, capture_output=True, text=True)
    wall = time.time() - t0
    stop.set()
    th.join()
    r = {"n": N, "bin": b, "list": lst, "args": list(args), "env": env or {}, "prefix": list(prefix), "rc": p.returncode,
         "timeout": p.returncode in (124, 137), "wall_s": round(wall, 2), **g}
    m = SUM.search(p.stderr)
    if m:
        r.update(images=int(m[1]), mp=float(m[2]), ms=float(m[3]), mp_s=float(m[4]), batches=int(m[5]), cores=float(m[6]),
                 skipped=int(m[7]), cpu_decoded=int(m[8]), write_fail=int(m[9]))
    for line in p.stdout.splitlines():
        if line.startswith("{"):
            try:
                s = json.loads(line)
                for k in ("cpus", "ingest_workers", "fallback_workers", "peak_dev_mb", "bytes"):
                    if k in s:
                        r[k] = s[k]
            except ValueError:
                pass
    if p.returncode != 0:
        r["stderr_tail"] = p.stderr.strip().splitlines()[-4:]
    if hash_:
        h = md5s(out)
        key = lst + " " + " ".join(a for a in args if a != "--cpu-entropy")
        r["n_out"] = len(h)
        if key not in REFS:
            rf = f"{R}/ref_{re.sub('[^a-z0-9]+', '_', key.strip())}.json"
            if os.path.exists(rf):
                REFS[key] = json.load(open(rf))
            elif b == "main" and not env and p.returncode == 0:
                REFS[key] = h
                json.dump(h, open(rf, "w"))
        if key in REFS:
            ref = REFS[key]
            r["md5_mismatch"] = sum(1 for k, v in ref.items() if h.get(k) != v) + sum(1 for k in h if k not in ref)
    shutil.rmtree(out, ignore_errors=True)
    return r


def log(phase, r):
    with open(f"{R}/{phase}.jsonl", "a") as f:
        f.write(json.dumps(r) + "\n")
    print(f"{r['bin']:4} {r['list']:6} {' '.join(r['args']):42} {json.dumps(r['env']):60} rc={r['rc']} t={r['wall_s']:6.1f}s "
          f"mp/s={r.get('mp_s')} cores={r.get('cores')} thr={r.get('cpus')}/{r.get('ingest_workers')}/{r.get('fallback_workers')} "
          f"gmem={r.get('gpu_mem_max_mb')} util={r.get('gpu_util_mean')} mism={r.get('md5_mismatch')}", flush=True)


def hang():
    for b, envs in (("main", [{"GP_ENCODERS": "24"}]), ("new", [{"GP_ENCODERS": "24"}, {"GP_ENCODERS": "48"}, {"GP_INGEST_INFLIGHT": "1"}])):
        for lst in ("fewpng", "fewjpg"):
            for env in envs:
                log("hang", run(b, lst, env=env, timeout=60, tag="hang", hash_=False))


def aff():
    log("aff", run("new", "fewjpg", prefix=["taskset", "-c", "0-7"], tag="aff", hash_=False))
    log("aff", run("new", "fewjpg", prefix=["taskset", "-c", "0,2,4"], tag="aff", hash_=False))
    log("aff", run("new", "fewjpg", prefix=["sudo", "systemd-run", "--quiet", "--scope", "-p", "CPUQuota=400%", "--uid=ubuntu",
                                          "--gid=ubuntu", "--setenv=HOME=" + H], tag="aff", hash_=False))
    log("aff", run("new", "fewjpg", prefix=["sudo", "systemd-run", "--quiet", "--scope", "-p", "CPUQuota=250%", "--uid=ubuntu",
                                          "--gid=ubuntu", "--setenv=HOME=" + H], tag="aff", hash_=False))
    log("aff", run("new", "fewjpg", tag="aff", hash_=False))


TILES = ["--tiles", "2,2,2"]   # 4 x 4 tiles for turbo (the coming default): every sweep run uses it


def configs():
    n, h = N, max(1, N // 2)
    P4 = ["--yuv", "444", "--depth", "10"]
    JC = {"GP_INGEST_DECODER": "1"}
    c = []
    for lst, a, e in (("jpeg", [], {}), ("jpeg", [], JC), ("png", [], {}), ("png", P4, {}), ("mix", [], {})):
        c += [("main", lst, a, e), ("new", lst, a, e)]
    c += [("new", "jpeg", [], {"GP_INGEST_WORKERS": "4"}),
          ("main", "jpeg", ["--cpu-entropy"], {}), ("new", "jpeg", ["--cpu-entropy"], {}),
          ("new", "jpeg", [], {"GP_ENCODERS": "4"})]
    c += [("new", "jpeg", [], {**JC, "GP_INGEST_WORKERS": str(w)}) for w in sorted({4, h, 2 * n}) if w != n]
    c += [("new", "jpeg", [], {**JC, "GP_INGEST_NICE": "0"})]
    c += [("new", "png", [], {"GP_FALLBACK_WORKERS": str(w)}) for w in sorted({4, max(1, n // 4), h, 2 * n}) if w != n]
    c += [("new", "png", [], {"GP_FALLBACK_QUEUE_MP": "100000"}), ("new", "png", [], {"GP_ENCODERS": "4"}),
          ("bf", "png", [], {"GP_FALLBACK_WORKERS": str(n)})]
    c += [("new", "png", P4, {"GP_FALLBACK_WORKERS": str(w)}) for w in sorted({4, h, 2 * n}) if w != n]
    c += [("bf", "png", P4, {"GP_FALLBACK_WORKERS": str(n)})]
    c += [("new", "mix", [], {"GP_INGEST_NICE": "0"}), ("new", "mix", [], {"GP_FALLBACK_WORKERS": str(h), "GP_INGEST_WORKERS": str(h)}),
          ("new", "mix", [], {"GP_INGEST_WORKERS": "4"}), ("bf", "mix", [], {"GP_FALLBACK_WORKERS": str(n)})]
    c = [(b, lst, TILES + a, e) for b, lst, a, e in c]
    # plus before / after at main's current tiling (2 x 2)
    for lst, a, e in (("jpeg", [], {}), ("jpeg", [], JC), ("png", [], {}), ("png", P4, {}), ("mix", [], {})):
        c += [("main", lst, a, e), ("new", lst, a, e)]
    seen, out = set(), []
    for x in c:
        k = json.dumps(x)
        if k not in seen:
            seen.add(k)
            out.append(x)
    return out


def sweep(rounds=3):
    cs = configs()
    print(f"{len(cs)} configs x {rounds} rounds, N={N}", flush=True)
    # reference first (main at its defaults, every list)
    for b, lst, a, e in cs:
        if b == "main" and not e:
            log("sweep_warm", run(b, lst, a, e, tag="ref"))
    for rd in range(rounds):
        for b, lst, a, e in (cs if rd % 2 == 0 else cs[::-1]):
            r = run(b, lst, a, e, tag="sw")
            r["round"] = rd
            log("sweep", r)
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    {"hang": hang, "aff": aff, "sweep": lambda: sweep(int(sys.argv[2]) if len(sys.argv) > 2 else 3)}[sys.argv[1]]()
