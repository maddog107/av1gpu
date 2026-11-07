#!/usr/bin/env python3
"""an_off.py spd_TAG.jsonl [BASE...] -- TEST [KEY]: paired analysis of an interleaved run.  Rounds are the unit: resampling rounds
with replacement (10,000 times) keeps each round's specs together.  Prints, per set, median throughput per spec, TEST against
the pooled BASE specs (ratio of medians, 95 % bootstrap CI), and the noise: BASE specs against one another."""
import json, random, statistics as st, sys, collections
a = sys.argv[1:]; f = a[0]; k = a.index("--"); bases = a[1:k]; test = a[k + 1]
random.seed(65)
rows = collections.defaultdict(lambda: collections.defaultdict(dict))
for l in open(f):
    r = json.loads(l)
    if r.get("timed") and not r.get("rc"):
        rows[r["tag"]][r["rnd"]][r["spec"]] = r
def val(r, s):
    return r["stats"].get("img_s" if s == "thumbs" else "mp_s", 0)
def ci(s, R, top, bot):
    def est(rs):
        t = [val(R[x][top_], s) for x in rs for top_ in top if top_ in R[x]]
        b = [val(R[x][bot_], s) for x in rs for bot_ in bot if bot_ in R[x]]
        return st.median(t) / st.median(b)
    keys = sorted(x for x in R if all(t in R[x] for t in top) and any(b in R[x] for b in bot))
    bs = sorted(est([random.choice(keys) for _ in keys]) for _ in range(10000))
    return est(keys), bs[250], bs[9749]
print(f"{'set':8s} {'n':>3s} " + " ".join(f"{b:>9s}" for b in bases + [test]) + "   test/base [95 % CI]        base noise [95 % CI]        rss test/base  peakdev test/base  wall test/base")
for s, R in rows.items():
    R = {x: v for x, v in R.items() if test in v and all(b in v for b in bases)}
    if len(R) < 3:
        continue
    keys = sorted(R)
    med = {b: st.median(val(R[x][b], s) for x in keys if b in R[x]) for b in bases + [test]}
    m, lo, hi = ci(s, R, [test], bases)
    if len(bases) > 1:
        nm, nlo, nhi = ci(s, R, [bases[-1]], [bases[0]])
    else:
        nm = nlo = nhi = 1.0
    g = lambda spec, fn: st.median(fn(R[x][spec]) for x in keys if spec in R[x])
    rss = g(test, lambda r: r["rss_mb"]) / st.median(R[x][b]["rss_mb"] for x in keys for b in bases if b in R[x])
    pk = g(test, lambda r: r["stats"].get("peak_dev_mb", 0)) / st.median(R[x][b]["stats"].get("peak_dev_mb", 0) for x in keys for b in bases if b in R[x])
    wl = g(test, lambda r: r["wall_s"]) / st.median(R[x][b]["wall_s"] for x in keys for b in bases if b in R[x])
    print(f"{s:8s} {len(keys):3d} " + " ".join(f"{med[b]:9.2f}" for b in bases + [test]) +
          f"   {m:.4f} [{lo:.4f}, {hi:.4f}]    {nm:.4f} [{nlo:.4f}, {nhi:.4f}]    {rss:.4f}   {pk:.4f}   {wl:.4f}")
