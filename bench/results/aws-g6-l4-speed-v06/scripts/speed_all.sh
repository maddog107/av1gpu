#!/bin/bash
# speed_all.sh: after pipe6.sh, with nothing else on the GPU: every speed phase (speed.sh: dmon around spd.py), then
# single-file mode timing and the README demo recording.  ~/results/main.log (MAIN_DONE), ~/results/speed_all.log.
L=~/results/speed_all.log; exec >> $L 2>&1
until grep -q PIPE6_DONE ~/results/pipe6.log 2>/dev/null; do sleep 10; done
# guard: no difference (after the reruns of sets that failed to run), every gate and test passed, and racecheck
# reports only the known pairs (k_entropy_win's ring, and k_recon.cuh:1373/1374)
R=~/results; bad=""
while read -r l; do
  n=${l%%:*}
  echo "$l" | grep -qE 'differ 0, .*rc 0/0|^single_file: .*differ 0, failed runs 0' && continue
  grep -E "^$n: " $R/identity_rerun.txt 2>/dev/null | tail -1 | grep -qE 'differ 0, .*rc 0/0' || bad="$bad identity:$n"
done < <(grep -E '^[a-z0-9_]+: inputs' $R/identity.txt)
grep -E '^rc ' $R/gate.txt | grep -vq '^rc 0 ' && bad="$bad gate"
grep -q '100% tests passed' $R/ctest.txt || bad="$bad ctest"
grep -E ' rc [0-9]+$' $R/ctest.txt | grep -vq ' rc 0$' && bad="$bad gputests"
grep -hE ' access at ' $R/race/new_0.log $R/race/new_1.log |
  grep -vE '(k_entropy_win|vload|vstore|win_group|__nanosleep)\b.* in gpu_entropy.cu:[0-9]+|rc_chroma.* in k_recon.cuh:137[34]' |
  grep -q . && bad="$bad racecheck"
grep -q 'RACECHECK SUMMARY' $R/race/new_0.log && grep -q 'RACECHECK SUMMARY' $R/race/new_1.log || bad="$bad racecheck-incomplete"
if [ -n "$bad" ]; then echo "SPEED_BLOCKED:$bad"; echo MAIN_DONE >> ~/results/main.log; exit 1; fi
echo "start $(date -u +%T); GPU apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
bash ~/speed.sh jpeg formats thumbs kodak lossless alpha
echo "phases done $(date -u +%T)"
python3 ~/single_time.py > ~/results/single.txt 2>&1
python3 ~/spd.py demo > ~/results/demo.log 2>&1
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit,enforced.power.limit,clocks_throttle_reasons.active,temperature.gpu --format=csv >> ~/results/clocks_end.txt 2>&1
echo "SPEED_ALL_DONE $(date -u +%T)"
