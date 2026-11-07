#!/usr/bin/env python3
"""prof_an.py REPORT.sqlite [LABEL]: one nsys run.  Kernel time by group; union busy of the GPU (any kernel) against the span of
the process's kernels, i.e. how long the GPU had nothing to run; per thread the time blocked in cudaStreamSynchronize, and of
that, the waits that directly follow a k_trim_measure / k_tone_range launch (the new synchronous waits)."""
import collections, sqlite3, sys
db = sqlite3.connect(sys.argv[1]); label = sys.argv[2] if len(sys.argv) > 2 else ""
names = dict(db.execute("select id, value from StringIds"))
tabs = {r[0] for r in db.execute("select name from sqlite_master where type='table'")}
K = db.execute("select start,end,streamId,shortName,gridX*gridY*gridZ,blockX*blockY*blockZ,correlationId from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
def grp(n):
    for p, g in (("k_trim_measure", "trim"), ("k_tone_range", "tone_range"), ("k_tone", "tone"), ("k_gray", "gray"), ("k_rz_", "resize"), ("k_to10", "to10"),
                 ("k_entropy", "entropy"), ("k_sym", "entropy")):
        if n.startswith(p): return g
    return "rest"
tot = collections.Counter(); cnt = collections.Counter(); byn = collections.Counter(); nbn = collections.Counter(); iv = collections.defaultdict(list)
for st, en, s, n, g, b, c in K:
    nm = names[n]; gr = grp(nm)
    tot[gr] += en - st; cnt[gr] += 1; byn[nm] += en - st; nbn[nm] += 1
    iv[gr].append((st, en)); iv["all"].append((st, en))
def union(v):
    v = sorted(v); u = 0; cs = ce = None
    for a, b in v:
        if cs is None or a > ce:
            if cs is not None: u += ce - cs
            cs, ce = a, b
        else:
            ce = max(ce, b)
    return (u + ce - cs) if cs is not None else 0
t0 = min(k[0] for k in K); t1 = max(k[1] for k in K); ua = union(iv["all"])
print(f"{label}: span {(t1 - t0) / 1e6:.1f} ms, any-kernel busy {ua / 1e6:.1f} ms = {100 * ua / (t1 - t0):.1f} %, idle {(t1 - t0 - ua) / 1e6:.1f} ms")
print("  group sum ms (count; union ms): " + "; ".join(f"{g} {tot[g] / 1e6:.1f} ({cnt[g]}; {union(iv[g]) / 1e6:.1f})" for g, _ in tot.most_common()))
for n, v in byn.most_common(60):
    if grp(n) != "rest":
        print(f"  {v / 1e6:9.1f} ms n={nbn[n]:6d} avg {v / nbn[n] / 1e3:9.1f} us {n[:80]}")
R = db.execute("select start,end,nameId,globalTid,correlationId from CUPTI_ACTIVITY_KIND_RUNTIME").fetchall()
kc = {k[6]: grp(names[k[3]]) for k in K}
per = collections.defaultdict(list)
for r in R: per[r[3]].append(r)
syncall = syncnew = 0; n_new = 0; bytid = {}
for tid, v in per.items():
    v.sort(); last = None; s = 0; open_wait = False
    for st, en, nm, _, c in v:
        nn = names.get(nm, "")
        if c in kc:
            if open_wait and kc[c] not in ("trim", "tone_range"): n_new += 1
            last = kc[c]; open_wait = False
        blocking = nn.startswith(("cudaStreamSynchronize", "cudaDeviceSynchronize"))
        if blocking:
            s += en - st; syncall += en - st
        # the host wait for a trim / auto-level result: every call after the launch up to the next kernel that blocks.  A
        # cudaMemcpyAsync to pageable memory returns only when the stream has reached it, so it holds most of the wait
        if last in ("trim", "tone_range") and c not in kc and (blocking or nn.startswith("cudaMemcpyAsync")):
            syncnew += en - st; open_wait = True
    if open_wait: n_new += 1
    bytid[tid] = s
print(f"  blocked in sync: all threads {syncall / 1e6:.1f} ms; host wait for trim / tone_range results {syncnew / 1e6:.1f} ms in {n_new} waits"
      f" (avg {syncnew / max(n_new, 1) / 1e6:.2f} ms)")
M = db.execute("select bytes,end-start from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
print(f"  memcpy {len(M)} ops, {sum(m[0] for m in M) / 1e9:.2f} GB, {sum(m[1] for m in M) / 1e6:.1f} ms")
