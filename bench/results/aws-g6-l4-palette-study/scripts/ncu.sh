#!/bin/bash
# GPU time per kernel (ncu, serialised) of one photo encode, main vs branch: which kernel got slower
f=${1:-$(ls -S ~/pal/sizerange/*.jpg | sed -n 40p)}
echo "image $f"
for b in av1gpu br; do
  sudo /usr/local/cuda/bin/ncu --metrics gpu__time_duration.sum --csv --print-units base ~/$b/build/av1gpu "$f" /tmp/ncu_$b.avif > /tmp/ncu_$b.csv 2>/dev/null
  python3 - /tmp/ncu_$b.csv $b <<'EOF'
import csv, sys, collections
tot = collections.defaultdict(float); cnt = collections.Counter()
rows = [r for r in csv.reader(open(sys.argv[1])) if len(r) > 10]
hdr = rows[0]
ki, vi = hdr.index("Kernel Name"), hdr.index("Metric Value")
for r in rows[1:]:
    try: v = float(r[vi].replace(",", ""))
    except ValueError: continue
    k = r[ki].split("(")[0].split("<")[0].replace("void ", "")
    tot[k] += v; cnt[k] += 1
print(sys.argv[2], "total %.3f ms" % (sum(tot.values()) / 1e6))
for k, v in sorted(tot.items(), key=lambda x: -x[1])[:14]:
    print(f"  {k:40s} {v / 1e6:9.3f} ms  x{cnt[k]}")
EOF
done
