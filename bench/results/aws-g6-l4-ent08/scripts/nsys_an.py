#!/usr/bin/env python3
"""nsys_an.py REPORT.sqlite: kernel time per kernel, per stream class (entropy streams = streams that ran
k_entropy_win; the rest = encoder streams), and the critical-path view: busy (union) time of each class, the time
both overlap, and the tail after the last encoder-stream kernel (entropy only)."""
import sqlite3, sys, collections

db = sqlite3.connect(sys.argv[1])
names = dict(db.execute("select id, value from StringIds"))
K = db.execute("select start, end, streamId, shortName, gridX*gridY*gridZ, blockX*blockY*blockZ from CUPTI_ACTIVITY_KIND_KERNEL").fetchall()
ent_streams = {s for (_, _, s, n, _, _) in K if names[n].startswith("k_entropy_win")}
pf_names = ("k_deblock", "k_cdef", "k_lf_", "k_select_small", "k_mark", "k_count_active", "k_rd_pick", "k_assign", "k_finalize",
            "k_wmap", "k_wnorm", "k_cdef_idx_norm", "k_prune", "k_gather", "k_cur", "k_pairs", "k_argmin", "k_shift", "k_sse")
def cls(s, n):
    if s in ent_streams:
        return "entropy"
    return "postfilter" if n.startswith(pf_names) else "encode"
tot = collections.Counter(); cnt = collections.Counter(); byc = collections.Counter()
iv = collections.defaultdict(list)
for st, en, s, n, g, b in K:
    nm = names[n]; c = cls(s, nm)
    tot[(c, nm)] += en - st; cnt[(c, nm)] += 1; byc[c] += en - st
    iv["entropy" if c == "entropy" else "enc"].append((st, en))
t0 = min(k[0] for k in K); t1 = max(k[1] for k in K)
def union(v):
    v = sorted(v); u = 0; cs, ce = None, None; out = []
    for a, b in v:
        if cs is None or a > ce:
            if cs is not None: out.append((cs, ce))
            cs, ce = a, b
        else:
            ce = max(ce, b)
    if cs is not None: out.append((cs, ce))
    return out
ue = union(iv["enc"]); un = union(iv["entropy"]); ua = union(iv["enc"] + iv["entropy"])
L = lambda u: sum(b - a for a, b in u) / 1e6
# overlap of entropy busy with encoder busy
def inter(a, b):
    i = j = 0; s = 0
    while i < len(a) and j < len(b):
        lo = max(a[i][0], b[j][0]); hi = min(a[i][1], b[j][1])
        if hi > lo: s += hi - lo
        if a[i][1] < b[j][1]: i += 1
        else: j += 1
    return s / 1e6
last_enc = max(b for a, b in ue); first_enc = min(a for a, b in ue)
print(f"GPU span {(t1 - t0) / 1e6:.1f} ms; any-kernel busy {L(ua):.1f} ms; encoder-stream busy {L(ue):.1f}; entropy busy {L(un):.1f};"
      f" entropy busy while encoders busy {inter(ue, un):.1f}; entropy-only time {L(ua) - L(ue):.1f}; tail after last encoder kernel"
      f" {(t1 - last_enc) / 1e6:.1f} ms; head before first encoder kernel {(first_enc - t0) / 1e6:.1f}")
print("kernel ms by class:", {k: round(v / 1e6, 1) for k, v in byc.items()})
for (c, nm), v in tot.most_common(45):
    print(f"{c:10s} {v / 1e6:9.1f} ms n={cnt[(c, nm)]:6d} avg {v / cnt[(c, nm)] / 1e3:9.1f} us  {nm[:90]}")
# memcpy / memset on entropy streams
try:
    M = db.execute("select start, end, streamId, bytes, copyKind from CUPTI_ACTIVITY_KIND_MEMCPY").fetchall()
    me = [m for m in M if m[2] in ent_streams]
    print(f"memcpy: all {len(M)} ({sum(m[1] - m[0] for m in M) / 1e6:.1f} ms), entropy streams {len(me)} ({sum(m[1] - m[0] for m in me) / 1e6:.1f} ms)")
except Exception as e:
    print("memcpy:", e)
