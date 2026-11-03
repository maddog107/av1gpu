#!/bin/bash
# br2: gpu_fuzz_test (exercises run_batch, which commit 2 changes) and the 8 MP single-file check-entropy
cd ~/br2/build
nohup bash -c '{ ./gpu_fuzz_test --seed 1 --count 400; echo "gpu_fuzz_test rc $?"
  ./gpu_fuzz_test --seed 77 --count 200 --yuv444 60 --depth10 60 --lossless 30; echo "gpu_fuzz_test (444/10/ll) rc $?"
  for f in ~/sets/single/*; do ./av1gpu $f ~/work/ce.avif --check-entropy 2>&1 | grep -E "check-entropy|rror" | sed "s#^#$(basename $f): #"; done
  echo FUZZ2_DONE; } > ~/results/fuzz_br2.txt 2>&1' > /dev/null 2>&1 &
sleep 1
echo started
