#!/bin/bash
# post.sh: after pipe.sh (PIPE_DONE).  The GPU unit tests of checks.sh (78e374c's build), then the speed A/B alone on
# the GPU (the first speed attempt overlapped the SASS extraction and was stopped; it is in ~/results/aborted_speed).
# ~/results/post.log; POST_DONE at the end.
L=~/results/post.log; exec >> $L 2>&1
echo "start $(date -u +%T)"
G=~/results/gputests.txt; : > $G
{ cd ~/new/build
  ./gpu_fuzz_test --seed 1 --count 400; echo "gpu_fuzz_test rc $?"
  ./gpu_fuzz_test --seed 77 --count 200 --yuv444 60 --depth10 60 --lossless 30; echo "gpu_fuzz_test (444/10/ll) rc $?"
  ./gpu_postfilter_test --seed 1 --count 200; echo "gpu_postfilter_test rc $?"
  ./gpu_dsp_test; echo "gpu_dsp_test rc $?"
  ./varboost_test; echo "varboost_test rc $?"; } >> $G 2>&1
echo "gpu tests done $(date -u +%T): $(grep -E ' rc [0-9]+$' $G | tr '\n' ';')"
echo "GPU apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); other jobs: $(ps -eo args | grep -cE 'compute-sanitizer|nvdisasm|ctest' )"
bash ~/speed.sh race noise
echo "speed done $(date -u +%T)"
echo POST_DONE
