#!/bin/bash
# speed9.sh: every speed run, with nothing else on the GPU; nvidia-smi dmon (power, utilisation, clocks; 1 s) around it.
#  1. spd9.py single: cold single-process wall times (B13, A13, A13c, A13 with GP_MAX_CONNECTIONS=1)
#  2. spd9.py batch: size range, standard, thumbnails, size range 4:4:4 10-bit (B13, A13, A13c = noise)
#  3. (L4 only, "docs") speed-v06's spd.py, unchanged except for the binaries (new = B13, old = A13) and the paths:
#     every av1gpu speed of README.md / docs/RESULTS.md (jpeg, formats, thumbs, kodak, lossless, alpha);
#     ~/results/docs/<phase>.jsonl
# ~/results/speed9.log; prints SPEED9_DONE.
L=~/results/speed9.log; exec >> $L 2>&1
cd ~
echo "start $(date -u +%T); GPU apps: $(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)"
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit,enforced.power.limit,clocks_throttle_reasons.active,temperature.gpu --format=csv > ~/results/clocks_start.txt 2>&1
nvidia-smi dmon -s puc -o DT > ~/results/dmon.txt 2>&1 &
DM=$!
python3 ~/spd9.py single; echo "single rc $? $(date -u +%T)"
python3 ~/spd9.py batch; echo "batch rc $? $(date -u +%T)"
if [ "$1" = docs ]; then
  mkdir -p ~/results/docs
  sed -e 's#f"{H}/new/build/av1gpu"#f"{H}/bin/B13"#; s#f"{H}/old/build/av1gpu"#f"{H}/bin/A13"#' \
      -e 's#{H}/new/bench#{H}/src/bench#; s#^R = f"{H}/results"#R = f"{H}/results/docs"#' \
      ~/src/bench/results/aws-g6-l4-speed-v06/scripts/spd.py > ~/spd_docs.py
  grep -nE 'BIN = |^R = |sys.path' ~/spd_docs.py
  python3 ~/spd_docs.py jpeg formats thumbs kodak lossless alpha; echo "docs rc $? $(date -u +%T)"
fi
kill $DM
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit,enforced.power.limit,clocks_throttle_reasons.active,temperature.gpu --format=csv > ~/results/clocks_end.txt 2>&1
echo "SPEED9_DONE $(date -u +%T)"
