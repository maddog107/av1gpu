#!/bin/bash
# Gather the study's results on the instance into ~/collect.tgz: rows, set list, source sizes, logs, crops, versions.
set -e
C=~/collect; rm -rf $C; mkdir -p $C/rows $C/logs $C/crops
cp ~/pal/rows/*.jsonl $C/rows/
cp ~/pal/sets.json $C/
python3 - <<'EOF'
import json, os
H = os.path.expanduser("~")
d = {}
for s in sorted(os.listdir(f"{H}/pal/sets")):
    d[s] = {f: os.path.getsize(f"{H}/pal/sets/{s}/{f}") for f in sorted(os.listdir(f"{H}/pal/sets/{s}"))}
json.dump(d, open(f"{H}/collect/rows/sizes.json", "w"), indent=1)
EOF
cp ~/pal/*.log $C/logs/ 2>/dev/null || true
cp ~/results/env.txt $C/logs/ 2>/dev/null || true
cp ~/setup.out $C/logs/setup.out 2>/dev/null || true
cp ~/pal/crops/*.png ~/pal/crops/crops.json $C/crops/ 2>/dev/null || true
cp -r ~/switch $C/switch_test 2>/dev/null && rm -f $C/switch_test/*.avif $C/switch_test/*.ivf $C/switch_test/*.png || true
{ ~/opt/bin/avifenc --version 2>&1 | head -3; ~/opt/bin/avifdec --version 2>&1 | head -3; ~/av1gpu/build/av1gpu --help 2>&1 | head -1;
  cat ~/av1gpu/COMMIT; cwebp -version; dav1d --version; rsvg-convert --version; nvidia-smi --query-gpu=name,driver_version --format=csv,noheader; } > $C/logs/versions.txt 2>&1
tar -czf ~/collect.tgz -C ~ collect
ls -la ~/collect.tgz
