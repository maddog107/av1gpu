#!/usr/bin/env python3
"""seeded option-matrix fuzzer for av1gpu (PR build): random options and inputs, batch and single.  Checks: no crash, no hang, exit status,
outputs decode in dav1d and aomdec to the same pixels, no missing / surprise files, the counts in the batch summary, output size against a magick oracle.
usage: fuzz.py SEED ITERATIONS [--workers N]"""
import concurrent.futures as cf, glob, hashlib, json, os, random, re, shlex, shutil, signal, subprocess, sys, tempfile, threading, time, collections, struct
from functools import lru_cache
sys.path.insert(0, os.path.expanduser("~/t"))
import avt
H = os.path.expanduser("~"); MAG = f"{H}/im/bin/magick"; BIN = os.environ.get("BIN", f"{H}/pr/build/av1gpu")
ENVM = dict(os.environ, LD_LIBRARY_PATH=f"{H}/im/lib")
g = lambda *ps: sorted(f for p in ps for f in glob.glob(f"{H}/{p}"))
PNG = g("adv/trim/*.png", "adv/alpha/*.png", "adv/deep/*.png")
FLAT = g("adv/flat/*.png")
JPG = [f for f in g("adv/jpeg/*.jpg")]
WEBP = g("adv/webp/*.webp")
GATE = g("pr/tests/gate_images/*")
PHOTO = g("images_std/*.jpg")
COMIC = g("comics/*.jpg")
BROKEN = g("adv/broken/*")
RAND = g("adv/rand/*")
TALL = g("adv/limits/l64x6553[56].png", "adv/limits/l64x65472.png", "adv/limits/l65536x64.png")
LOG = os.environ.get("FZLOG", f"{H}/results/fuzz")
os.makedirs(LOG, exist_ok=True)
TO = int(os.environ.get("TO", "120"))

def geometry(r):
    W = r.choice([1, 2, 3, 7, 8, 9, 16, 17, 31, 32, 33, 64, 65, 100, 127, 128, 129, 200, 256, 300, 512, 1000])
    Hh = r.choice([1, 2, 3, 7, 8, 9, 16, 17, 31, 32, 33, 64, 65, 100, 127, 128, 129, 200, 256, 300, 512, 1000])
    P = r.choice([1, 5, 10, 25, 33.3, 50, 66, 75, 99, 100, 101, 150, 173, 200, 300])
    Q = r.choice([5, 25, 50, 100, 150, 200])
    A = r.choice([1, 10, 100, 1000, 5000, 20000, 100000, 500000])
    s = r.choice(["", "", "", "!", ">", "<", "^", "^>", "!>"])
    if r.random() < .03: return r.choice(["0.5", "0.4x0.4", "x0.9", "1x65536!", "65536x1", "70000x1!", "1048576x1", "2000%", "1000x1000^", "4000x4000!", "@40000000", "0.001%", "1x1^", "65535x1^", "1x65535^"])
    F = r.choice([0.4, 0.5, 0.6, 1.4, 99.5, 100.4, 100.6, 33.3])
    return r.choice([f"{W}", f"x{Hh}", f"{W}x", f"{W}x{Hh}{s}", f"{W}x{Hh}{s}", f"{W}x{Hh}", f"{P}%", f"{P}%", f"{P}%x{Q}%", f"{W}x{Q}%", f"x{P}%", f"{P}%x", f"@{A}", f"@{A}>", f"{W}{s}", f"x{Hh}{s}", f"{P}%{s}",
                     f"{W}!", f"x{Hh}!", f"{W}x!", f"{W}^", f"x{Hh}^", f"{W}x^", f"{A}@", f"@{A}<", f"{P}%>", f"{P}%<", f"{F}x", f"x{F}", f"{F}x{F}!", f"{W}.5x{Hh}.5", f"@{A * 37}"])

def crop(r):
    w, h = r.choice([1, 2, 8, 9, 16, 17, 33, 40, 64, 100, 200]), r.choice([1, 2, 8, 9, 16, 17, 33, 40, 64, 100, 200])
    x, y = r.choice([0, 0, 1, 2, 3, 7, 8, 13, 50]), r.choice([0, 0, 1, 2, 3, 7, 8, 13, 50])
    return f"{w}x{h}+{x}+{y}"

def options(r):
    o = []
    lossless = r.random() < .18
    if lossless:
        o += ["--lossless"]
        if r.random() < .4: o += ["--ycgco-re"]
        if r.random() < .3: o += ["--ll-tile-area", r.choice(["0", "1024", "65536"])]
    else:
        if r.random() < .35: o += ["--yuv", "444"]
        if r.random() < .35: o += ["--depth", "10"]
        o += r.choice([[], ["--preset", "turbo3"], ["--preset", "turbo3"], ["--preset", "turbo2"], ["--preset", "fast"], ["--preset", "base", "--small-preset", "none"]])
        o += r.choice([[], [], ["--small-preset", "none"], ["--small-preset", "turbo3"]])
        o += r.choice([[], [], [], ["--qindex", str(r.choice([1, 60, 120, 200, 255]))], ["--quality", str(r.choice([0, 30, 60, 90, 100]))]])
        o += r.choice([[], [], [], [], ["--tune", "none"]])
    if r.random() < .15: o += ["--palette", r.choice(["on", "off", "auto"])]
    if r.random() < .10: o += ["--no-alpha"]
    if r.random() < .10: o += ["--alpha-quality", str(r.choice([0, 40, 80, 100]))]
    if r.random() < .08: o += ["--tiles", r.choice(["0,0,0", "0.001,1,1", "0.01,2,1", "0.001,0,2"])]
    if r.random() < .08: o += ["--cpu-entropy"]
    if r.random() < .08: o += ["--max-mp", r.choice(["0.001", "0.01", "0.1", "1", "40"])]
    if r.random() < .05: o += ["--max-input-mp", r.choice(["0.01", "1", "50"])]
    if r.random() < .15: o += ["--crop", crop(r)]
    k = r.random()
    if k < .10: o += ["--max-size", r.choice(["100x100", "40x", "x40", "8x8", "1x1", "300x20"])]
    elif k < .15: o += ["--max-height", r.choice(["1", "16", "100"])]
    elif k < .55: o += ["--resize", geometry(r)]
    if r.random() < .55:
        o += ["--trim"]
        if r.random() < .75: o += ["--fuzz", r.choice(["0", "0%", "0.5%", "1%", "3%", "5%", "10%", "30%", "50%", "100%", "6553", "65535"])]
        if r.random() < .35: o += ["--trim-margin", r.choice(["0", "1", "5", "100", "1%", "10%", "50%", "100%"])]
        if r.random() < .25: o += ["--trim-limits", r.choice(["0%,100%", "1%,20%", "5%,5%", "50%,100%", "0,0", "100%,100%", "1,99"])]
        if r.random() < .25: o += ["--trim-lightness", r.choice(["0", "50", "80", "100"])]
        if r.random() < .35: o += ["--skip-blank", r.choice(["0", "1", "50", "99", "100"])]
    if r.random() < .25: o += ["--auto-level"]
    if r.random() < .25: o += ["--level", r.choice(["0,100%", "10%,90%", "5%", "20%,80%,0.5", "0%,50%,2", "1000,60000", "99%,100%", "0%,1%", "0,65535,1"])]
    if r.random() < .25: o += ["--gamma", r.choice(["0.01", "0.1", "0.5", "1", "1.4", "2", "10", "100"])]
    if r.random() < .2: o += ["--negate"]
    if r.random() < .3: o += ["--grayscale", r.choice(["Rec601Luma", "Rec709Luma", "Average", "Brightness", "Lightness", "RMS"])] if r.random() < .8 else ["--colorspace", "Gray"]
    return o

def pick(r):
    if TALL and r.random() < .01: return r.choice(TALL)
    if RAND and r.random() < .6: return r.choice(RAND)
    c = r.random()
    if c < .35: return r.choice(PNG)
    if c < .42: return r.choice(FLAT)
    if c < .70: return r.choice(JPG)
    if c < .78: return r.choice(WEBP)
    if c < .85: return r.choice(GATE)
    if c < .97: return r.choice(PHOTO)
    return r.choice(COMIC)

def run(argv, to=TO, cwd=None):
    t = time.time()
    p = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True, cwd=cwd, errors="replace")
    try:
        o, e = p.communicate(timeout=to)
        return p.returncode, o, e, time.time() - t
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        o, e = p.communicate()
        return "timeout", o, e, time.time() - t

def norm(s):
    s = re.sub(r"/[^ :']*", "PATH", s)
    s = re.sub(r"[\d.]+", "N", s)
    return s[:110]

def decode_check(path, sigs, tag):
    try:
        c = avt.check(path)
    except avt.Bad as e:
        sigs.append(("decode", norm(str(e)), tag)); return None
    except Exception as e:
        sigs.append(("decode-harness", norm(repr(e)), tag)); return None
    d = tempfile.mkdtemp(prefix="ad_", dir=os.environ.get("AVT_TMP"))
    try:
        if c["si"][4] != 16 and max(c["si"][1:3]) <= 32768:   # libavif 1.0.4: no YCgCo-Re, 32768 pixels per side at most
            r = subprocess.run(["avifdec", "--ignore-icc", path, f"{d}/o.png"], capture_output=True, text=True, timeout=120)
            if r.returncode: sigs.append(("avifdec", norm(r.stderr + r.stdout), tag))
    finally:
        shutil.rmtree(d, ignore_errors=True)
    return c

STATUS_OK = {0, 2, 3}
def check_single(cmd, wd, inp):
    out = f"{wd}/o.avif"
    if os.path.exists(out): os.remove(out)
    rc, so, se, dt = run(cmd)
    sigs = []
    if rc == "timeout": sigs.append(("hang", "", ""))
    elif rc < 0 or rc >= 128: sigs.append(("crash", f"rc {rc}", norm(se[-200:])))
    elif rc not in STATUS_OK:
        sigs.append(("status", f"rc {rc}", norm(se[-160:])))
    if rc == 2 and "bad " not in se and "need" not in se and "cannot" not in se: sigs.append(("rc2 without a reason", norm(se[:100]), ""))
    if rc == 0:
        if not os.path.exists(out): sigs.append(("nooutput", "", ""))
        else:
            c = decode_check(out, sigs, "single")
            if c is not None and os.environ.get("SIZE_ORACLE") == "1" and "--trim" not in cmd and "--resize" in cmd:
                im = []
                if "--crop" in cmd: im += ["-crop", cmd[cmd.index("--crop") + 1], "+repage"]
                im += ["-auto-orient", "-resize", cmd[cmd.index("--resize") + 1]]
                q = subprocess.run([MAG, inp] + im + ["-format", "%w %h", "info:"], capture_output=True, text=True, env=ENVM)
                mm = re.match(r"(\d+) (\d+)", q.stdout)
                if mm:
                    sw, sh = c["si"][1], c["si"][2]
                    disp = (sh, sw) if c["irot"] in (1, 3) else (sw, sh)
                    if disp != (int(mm[1]), int(mm[2])): sigs.append(("size vs magick", cmd[cmd.index("--resize") + 1].strip("0123456789."), ""))
            if c is not None and "/adv/flat/" in inp and "--negate" not in cmd:
                # a flat picture stays flat through crop / trim / resize (and the tone curve), whatever the filter
                y = c["planes"][0]
                if int(y.max()) - int(y.min()) > (8 if c["si"][3] == 10 else 2): sigs.append(("flat not flat", "", ""))
    elif os.path.exists(out): sigs.append(("output on failure", f"rc {rc}", ""))
    return sigs, rc, se

def check_batch(cmd, wd, lst_lines, outdir):
    shutil.rmtree(outdir, ignore_errors=True)
    rc, so, se, dt = run(cmd)
    sigs = []
    if rc == "timeout": sigs.append(("hang", "batch", ""))
    elif rc < 0 or rc >= 128: sigs.append(("crash", f"rc {rc}", norm(se[-200:])))
    elif rc not in (0, 2): sigs.append(("status", f"batch rc {rc}", norm(se[-160:])))
    if rc in (0,):
        ins = [l.split("\t")[0] for l in lst_lines]
        names = [os.path.basename(p) + ".avif" for p in ins]
        have = set(os.listdir(outdir)) if os.path.isdir(outdir) else set()
        surprise = have - set(names)
        if surprise: sigs.append(("surprise files", "", ""))
        missing = [n for n in names if n not in have]
        for p, n in zip(ins, names):
            if n not in have and f"skipped {p}" not in se:
                sigs.append(("missing without message", "", ""))
                break
        skipped = sum(1 for p, n in zip(ins, names) if f"skipped {p}" in se)
        both = [n for p, n in zip(ins, names) if n in have and f"skipped {p}" in se]
        if both: sigs.append(("file written and skipped", "", ""))
        m = re.search(r"\[av1gpu\] (\d+) images", se)
        if m and int(m[1]) != len(have): sigs.append(("count images != files", f"{m[1]} vs {len(have)}", ""))
        m2 = re.search(r"trim: (\d+) images trimmed, (\d+) skipped as blank", se)
        if m2:
            blanks = len(re.findall(r"skipped \S+: blank", se))
            if blanks != int(m2[2]): sigs.append(("blank count", f"{blanks} vs {m2[2]}", ""))
        with cf.ThreadPoolExecutor(4) as tp:
            for s2 in tp.map(lambda n: (lambda s: (decode_check(f"{outdir}/{n}", s, "batch"), s)[1])([]), sorted(have)):
                sigs += s2
    return sigs, rc, se

def signature(sigs):
    return sorted({"|".join(s[:3]) for s in sigs})

def minimize(mk, opts, inputs, pred):
    """drop options, then inputs, while pred(opts, inputs) holds"""
    changed = True
    while changed:
        changed = False
        i = 0
        while i < len(opts):
            for n in (2, 1):
                t = opts[:i] + opts[i + n:]
                if i + n <= len(opts) and (n == 1 or not opts[i + 1].startswith("--")) and opts[i].startswith("--") and pred(t, inputs):
                    opts = t; changed = True; break
            else:
                i += 1
    i = 0
    while len(inputs) > 1 and i < len(inputs):
        t = inputs[:i] + inputs[i + 1:]
        if pred(opts, t): inputs = t
        else: i += 1
    return opts, inputs

seen = {}; lk = threading.Lock(); stats = collections.Counter()
def report(kind, sigset, opts, inputs, rc, se, seed, mkpred, it):
    new = [s for s in sigset if s not in seen]
    with lk:
        for s in sigset: seen.setdefault(s, 0); seen[s] += 1
        first = bool(new) and all(s not in stats for s in new)
        for s in new: stats[s] = 1
    if not new: return
    t0 = time.time()
    so, si = opts, inputs
    try:
        target = new[0]
        so, si = minimize(None, list(opts), list(inputs), lambda o, i: target in mkpred(o, i))
    except Exception as e:
        print("minimize failed", repr(e), flush=True)
    line = dict(iter=it, kind=kind, sig=new, seed=seed, opts=so, inputs=si, rc=str(rc), stderr=se[-400:], full_opts=opts, full_inputs=inputs)
    with lk:
        open(f"{LOG}/failures.jsonl", "a").write(json.dumps(line) + "\n")
        print(f"[{it}] NEW {new}  -> {kind}: {shlex.join(so)}  inputs {[os.path.relpath(x, H) for x in si][:6]}  ({time.time() - t0:.0f} s to minimise)", flush=True)

def single_case(seed, it, wd):
    r = random.Random(seed)
    o = options(r); f = pick(r)
    def pred(oo, ii):
        s, rc, se = check_single([BIN, ii[0], f"{wd}/o.avif"] + oo, wd, ii[0])
        return signature(s)
    sigs, rc, se = check_single([BIN, f, f"{wd}/o.avif"] + o, wd, f)
    sg = signature(sigs)
    if sg: report("single", sg, o, [f], rc, se, seed, pred, it)
    return rc

def batch_case(seed, it, wd):
    r = random.Random(seed)
    o = options(r)
    n = r.choice([2, 3, 5, 8, 13, 21, 34])
    files = [pick(r) for _ in range(n)]
    if r.random() < .15: files.insert(r.randrange(len(files)), r.choice(BROKEN))
    if r.random() < .3:
        files += r.sample(FLAT, 3) + [r.choice(g("adv/jpeg/white_*.jpg"))]
        r.shuffle(files)
    # per-line options on some lines
    lines = []
    ldir = f"{wd}/in"; shutil.rmtree(ldir, ignore_errors=True); os.makedirs(ldir)
    for i, f in enumerate(files):
        link = f"{ldir}/{i:03d}_{os.path.basename(f)}"; os.symlink(f, link)
        l = link
        if r.random() < .12: l += "\tcrop=" + crop(r)
        if r.random() < .12: l += "\tresize=" + geometry(r)
        if r.random() < .05: l += "\tmax-size=" + r.choice(["50x50", "x40", "40x", "1x1"])
        lines.append(l)
    outdir = f"{wd}/out"
    def bcmd(oo, ll):
        p = f"{wd}/list"; open(p, "w").write("".join(x + "\n" for x in ll))
        return [BIN, "--batch", p, "--outdir", outdir] + oo
    def pred(oo, ll):
        s, rc, se = check_batch(bcmd(oo, ll), wd, ll, outdir)
        return signature(s)
    sigs, rc, se = check_batch(bcmd(o, lines), wd, lines, outdir)
    sg = signature(sigs)
    if sg:
        # keep copies of the links' targets in the log
        tgt = [os.path.realpath(l.split("\t")[0]) + "\t" + "\t".join(l.split("\t")[1:]) for l in lines]
        report("batch", sg, o, lines, rc, se, seed, pred, it)
    return rc

def main():
    seed0, iters = int(sys.argv[1]), int(sys.argv[2])
    workers = int(os.environ.get("WORKERS", "4"))
    os.makedirs("/dev/shm/fz", exist_ok=True)
    os.environ["AVT_TMP"] = "/dev/shm/fz"
    t0 = time.time(); rcs = collections.Counter()
    def job(i):
        wd = tempfile.mkdtemp(prefix=f"w{i}_", dir="/dev/shm/fz")
        try:
            if i % 3 == 2: rc = batch_case(seed0 * 100000 + i, i, wd)
            else: rc = single_case(seed0 * 100000 + i, i, wd)
            return ("batch" if i % 3 == 2 else "single", rc)
        except Exception as e:
            print("HARNESS", i, repr(e), flush=True); return ("err", "harness")
        finally:
            shutil.rmtree(wd, ignore_errors=True)
    with cf.ThreadPoolExecutor(workers) as tp:
        for k, rc in enumerate(tp.map(job, range(iters))):
            rcs[rc] += 1
            if k % 200 == 199:
                print(f"-- {k + 1} iterations, {time.time() - t0:.0f} s, rc {dict(rcs)}, signatures {len(seen)}", flush=True)
    print("DONE", dict(rcs), "signatures:")
    for s, n in sorted(seen.items(), key=lambda kv: -kv[1]): print(n, s)
main()
