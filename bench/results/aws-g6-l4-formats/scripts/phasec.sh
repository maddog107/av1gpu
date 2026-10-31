#!/bin/bash
# phasec.sh speed|quality : bench/formats.py on the gamut set with the fixed build (outputs ~/results/formats)
cd ~/fix
export GP_ENCODERS=4 PATH=$HOME/opt/bin:$PATH
X=--skip-quality; [ "$1" = quality ] && X=--skip-speed
python3 bench/formats.py --images ~/images_gamut --out ~/results/formats --av1gpu ~/build-fix/av1gpu --avifdec-codec dav1d \
  --anchor bench/results/aws-g6-l4-gamut/results.json --svt-preset 6 --svt-tune 3 \
  --svt-anchor bench/results/aws-c6a-gamut-svt/results.json $X > ~/results/formats_$1.log 2>&1
echo "PHASE_DONE formats_$1 rc=$? $(date -u +%H:%M:%S)" >> ~/results/phases.txt
