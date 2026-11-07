#!/usr/bin/env python3
"""an_on.py spd_TAG.jsonl: per set and spec the medians of --stats wall (ms), MP, bytes, and the cost in time against plain with the
noise of plain2.  Time is the honest metric here: options change the pixels coded, the content, or both."""
import collections, json, statistics as st, sys
R = collections.defaultdict(lambda: collections.defaultdict(list))
for l in open(sys.argv[1]):
    r = json.loads(l)
    if r.get("timed") and not r.get("rc"): R[r["tag"]][r["spec"]].append(r)
for t, S in R.items():
    w = lambda s: st.median(x["stats"]["wall_ms"] for x in S[s])
    p = (w("plain") + w("plain2")) / 2
    print(f"--- {t}: plain {w('plain'):.0f} ms, plain2 {w('plain2'):.0f} ms; noise {100 * (w('plain2') / w('plain') - 1):+.2f} %")
    for s, v in S.items():
        a = v[0]["stats"]; tr = v[0].get("trim", "")
        print(f"{s:10s} wall {w(s):8.0f} ms {100 * (w(s) / p - 1):+7.2f} % | MP {a['mp']:7.1f} ms/MP {w(s) / a['mp']:6.3f} | bytes {a['bytes'] / 1e6:7.1f} MB | ingest_wait {st.median(x['stats']['ingest_wait_ms'] for x in v):7.0f} anal {st.median(x['stats']['analysis_ms'] for x in v):6.0f} recon {st.median(x['stats']['recon_ms'] for x in v):6.0f} gpu {st.median(x['stats']['gpu_total_ms'] for x in v):6.0f} | {tr[10:48]}")
