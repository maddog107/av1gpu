#!/bin/bash
# speed8.sh: waits until the identity runs and the checks are done (the GPU must be idle), then the speed A/B of the
# PR head (br2, both commits) against main, the noise run (main against a copy of main), and the single-file wall
# time of commit 1 alone (br1).  nvidia-smi dmon alongside.  ~/results/spd_*.txt, ~/results/speed.log
L=~/results/speed.log
until grep -q IDENTITY_DONE ~/results/identity.txt 2>/dev/null && grep -q IDENTITY_DONE ~/results/identity_single_br1.txt 2>/dev/null &&
      grep -q IDENTITY_DONE ~/results/identity_single_br2.txt 2>/dev/null && grep -q SANITIZER_DONE ~/results/sanitizer.txt 2>/dev/null; do
  sleep 5; done
echo "$(date -u +%H:%M:%S) start; GPU processes: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" >> $L
cp ~/main/build/av1gpu ~/av1gpu_main2
nvidia-smi dmon -s puc -o DT > ~/results/dmon.txt 2>&1 &
DM=$!
python3 ~/spd8.py ab ~/main/build/av1gpu ~/br2/build/av1gpu sr std thumbs single > ~/results/spd_ab.txt 2>&1
echo "$(date -u +%H:%M:%S) ab done" >> $L
python3 ~/spd8.py noise ~/main/build/av1gpu ~/av1gpu_main2 sr std thumbs single > ~/results/spd_noise.txt 2>&1
echo "$(date -u +%H:%M:%S) noise done" >> $L
python3 ~/spd8.py single_br1 ~/main/build/av1gpu ~/br/build/av1gpu single > ~/results/spd_single_br1.txt 2>&1
echo "$(date -u +%H:%M:%S) single br1 done" >> $L
kill $DM
echo SPEED_DONE >> $L
