#!/bin/bash
# pipe6.sh: switch NEW to main 6a4956e and redo the checks.  The ef2522f results move to *_ef2522f; its build to ~/m34.
# Then: build (tests ON), the mono set, the identity matrix (alone on the GPU), reruns of sets that failed to run,
# decoder gate + ctest + memcheck / synccheck, then racecheck (main, and the #36 / #37 merges for the k_recon
# report).  ~/results/pipe6.log; PIPE6_DONE at the end.
L=~/results/pipe6.log; exec >> $L 2>&1
echo "start $(date -u +%T)"
for p in $(ps -eo pid,args | grep -E 'work/race_new_' | grep -v grep | awk '{print $1}'); do kill $p; done
sleep 2
cd ~/results
for f in identity.txt identity_rerun.txt gate.txt ctest.txt sanitizer.txt race_new.txt; do [ -e $f ] && mv $f ${f%.txt}_ef2522f.txt; done
for d in san race; do [ -d $d ] && mv $d ${d}_ef2522f; done
mkdir -p race; cp race_ef2522f/old_*.log race_ef2522f/m3* race/ 2>/dev/null
for f in gate_*.log; do mv $f ef2522f_$f; done
cd ~
mv ~/new ~/m34
mkdir -p ~/new; tar -xzf ~/new6.tar.gz -C ~/new
s=$(date +%s)
cmake -S ~/new -B ~/new/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DAV1GPU_TESTS=ON > ~/results/setup/new6.cfg.log 2>&1
cmake --build ~/new/build -j 8 > ~/results/setup/new6.log 2>&1; rc=$?
echo "build new $(cat ~/new/COMMIT) rc $rc after $(( $(date +%s) - s )) s"
[ $rc = 0 ] || { echo PIPE6_DONE; exit 1; }
bash ~/mk_mono.sh
bash ~/identity.sh
echo "identity done $(date -u +%T): $(grep -cE 'differ 0,' ~/results/identity.txt) sets with 0 differences of $(grep -cE '^[a-z0-9_]+: inputs' ~/results/identity.txt)"
bash ~/rerun_failed.sh
( bash ~/checks.sh ) &
( bash ~/sanitizer.sh ) &
wait
echo "checks + memcheck/synccheck done $(date -u +%T)"
( bash ~/race_ab.sh new ) &
( KS=0 bash ~/race_ab.sh m36 m37 ) &
wait
echo "racecheck done $(date -u +%T)"
echo PIPE6_DONE
