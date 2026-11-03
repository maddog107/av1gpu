#!/bin/bash
# after2.sh: after after.sh (sanitizers and module-load timing done, GPU idle): spd_paloff.py.
until grep -q AFTER_DONE ~/results/after.out 2>/dev/null; do sleep 10; done
echo "start $(date -u +%T); GPU apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
python3 ~/spd_paloff.py
echo "AFTER2_DONE $(date -u +%T)"
