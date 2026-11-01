#!/usr/bin/env python3
"""follow-up tests (instance): t2.py instr|speed|jpeg|emu ...   results -> ~/results/thr/t2_<phase>.jsonl"""
import os, sys, tthr
H = tthr.H
T = ["--tiles", "2,2,2"]
P4 = T + ["--yuv", "444", "--depth", "10"]
LOG = f"{H}/results/blog"
os.makedirs(LOG, exist_ok=True)


def log(phase, r):
    tthr.log("t2_" + phase, r)


def warm():
    for lst, a in (("mix", T), ("png", T), ("png", P4), ("jpeg", T)):
        log("warm", tthr.run("main", lst, a, tag="w"))


def instr():
    k = 0
    for b, e in (("main", {}), ("new", {}), ("main", {}), ("new", {}), ("new", {"GP_FB_POLICY": "1"}), ("new", {"GP_FB_POLICY": "2"}),
                 ("new", {"GP_FB_POLICY": "3"}), ("new", {"GP_FB_POLICY": "4"})):
        k += 1
        tag = f"{b}_p{e.get('GP_FB_POLICY', '0')}_{k}"
        r = tthr.run(b, "mix", T, {**e, "GP_BATCH_LOG": f"{LOG}/mix_{tag}.txt"}, tag="i")
        r["blog"] = f"mix_{tag}.txt"
        log("instr", r)
    for b in ("main", "new"):
        r = tthr.run(b, "png", T, {"GP_BATCH_LOG": f"{LOG}/png_{b}.txt"}, tag="i")
        r["blog"] = f"png_{b}.txt"
        log("instr", r)


def rounds(phase, cs, n, prefix=()):
    for rd in range(int(n)):
        for b, l, a, e in (cs if rd % 2 == 0 else cs[::-1]):
            r = tthr.run(b, l, a, e, prefix=prefix, tag=phase)
            r["round"] = rd
            log(phase, r)


def speed(n=5):
    cs = [("main", "mix", T, {})] + [("new", "mix", T, {"GP_FB_POLICY": str(p)} if p else {}) for p in range(5)]
    cs += [("main", "png", T, {})] + [("new", "png", T, {"GP_FB_POLICY": str(p)} if p else {}) for p in (0, 1, 2, 4)]
    rounds("speed", cs, n)


def jpeg(n=5):
    cs = [("main", "jpeg", T, {})] + [("new", "jpeg", T, {"GP_INGEST_WORKERS": str(w)}) for w in (4, 8, 16)] + [("new", "jpeg", T, {})]
    rounds("jpeg", cs, n)
    JC = {"GP_INGEST_DECODER": "1"}
    cs = [("main", "jpeg", T, JC)] + [("new", "jpeg", T, {**JC, "GP_INGEST_WORKERS": str(w)}) for w in (4, 8, 16)] + [("new", "jpeg", T, JC)]
    rounds("jpegcpu", cs, 3)


def emu(cpus, n, *envs):
    """emu CPULIST ROUNDS [ENV=V,ENV=V ...]: main vs new (+ env) under taskset on every list"""
    e = dict(kv.split("=") for env in envs for kv in env.split(",") if kv)
    cs = []
    for lst, a, x in (("mix", T, {}), ("png", T, {}), ("png", P4, {}), ("jpeg", T, {})):
        cs += [("main", lst, a, x), ("new", lst, a, {**x, **e})]
    rounds("emu", cs, int(n), prefix=["taskset", "-c", cpus])


if __name__ == "__main__":
    f = {"warm": warm, "instr": instr, "speed": speed, "jpeg": jpeg, "emu": emu}[sys.argv[1]]
    f(*sys.argv[2:])
    print("T2_DONE", sys.argv[1], flush=True)
