#!/usr/bin/env python3
"""blog.py FILE...: batch composition and GPU-busy coverage from GP_BATCH_LOG files"""
import sys


def union(iv):
    iv = sorted(iv)
    tot, cur = 0.0, None
    for a, b in iv:
        if cur is None or a > cur[1]:
            if cur: tot += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur: tot += cur[1] - cur[0]
    return tot


for f in sys.argv[1:]:
    E, N, W = [], [], None
    for l in open(f):
        p = l.split()
        if p[0] == "W":
            W = float(p[1]); continue
        r = dict(kind=p[0], enc=int(p[1]), tg=float(p[2]), t0=float(p[3]), t1=float(p[4]), tw=float(p[5]), nfb=int(p[6]), fbmp=float(p[7]),
                 ning=int(p[8]), nheld=int(p[9]), ingmp=float(p[10]), maxmp=float(p[11]), minmp=float(p[12]), nsmall=int(p[13]), cls=int(p[14]))
        (E if r["kind"] == "E" else N).append(r)
    if not E:
        continue
    start, end = min(r["t0"] for r in E), max(r["t1"] for r in E + N)
    pure_fb = [r for r in E if r["nfb"] and not r["ning"]]
    pure_in = [r for r in E if r["ning"] and not r["nfb"]]
    mixed = [r for r in E if r["ning"] and r["nfb"]]
    mp = [r["fbmp"] + r["ingmp"] for r in E]
    fbmp = sum(r["fbmp"] for r in E); inmp = sum(r["ingmp"] for r in E)
    encbusy = union([(r["t0"], r["t1"]) for r in E])
    allbusy = union([(r["t0"], r["t1"]) for r in E] + [(r["t0"], r["t1"]) for r in N])
    gather = sum(r["t0"] - r["tg"] for r in E)
    # when were fallback MP encoded: share of fallback MP finished in the first half of the run
    half = start + (end - start) / 2
    fb_first_half = sum(r["fbmp"] for r in E if r["t1"] <= half) / fbmp if fbmp else 0
    in_first_half = sum(r["ingmp"] for r in E if r["t1"] <= half) / inmp if inmp else 0
    # >= 1 encoder busy vs all 3 busy (time-sampled)
    import bisect
    step = 1.0
    t, c1, c3 = start, 0, 0
    while t < end:
        k = sum(1 for r in E if r["t0"] <= t < r["t1"])
        c1 += k >= 1; c3 += k >= 3
        t += step
    n = max(1, int((end - start) / step))
    print(f"{f.split('/')[-1]:28} wall={W:7.0f} batches={len(E):3} pureFB={len(pure_fb):3} pureJPG={len(pure_in):3} mixed={len(mixed):3} "
          f"MP/batch={sum(mp)/len(mp):5.1f} spread(max/min)={sum(r['maxmp']/max(r['minmp'],0.01) for r in E)/len(E):6.1f} "
          f"enc-busy={encbusy/(end-start)*100:5.1f}% >=1enc={c1/n*100:5.1f}% all3={c3/n*100:5.1f}% gather/batch={gather/len(E):6.1f}ms "
          f"ingwait/batch={sum(r['tw'] for r in E)/len(E):6.1f}ms fbMP-done-1st-half={fb_first_half*100:4.0f}% jpgMP-1st-half={in_first_half*100:4.0f}% "
          f"tail(last enc end - last start)={end - max(r['t0'] for r in E):6.0f}ms")
