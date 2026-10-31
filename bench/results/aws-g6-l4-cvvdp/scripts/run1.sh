#!/bin/bash
# Phase 1 on the instance: the published quality points for every encoder.  NVENC encodes run in the background
# while bench.py encodes and scores the rest (SSIMULACRA2 + CVVDP standard_fhd); then NVENC is scored.
set -x
cd ~/av1gpu
P=~/venv/bin/python
# the AMI's LD_LIBRARY_PATH holds CUDA 13.2's own cuDNN 9.23, which cuDNN 9.26 (torch's wheel) would dlopen parts of
export LD_LIBRARY_PATH=$HOME/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
ph() { echo "PHASE $1 $(date -u +%FT%TZ)" >> ~/results/phases.txt; }
ph p1_start
( $P ~/nvenc_encode.py ~/nv/q 18,25,32,38,44,50 > ~/results/nvenc_encode_p1.log 2>&1; echo "PHASE nvenc_enc_p1_done $(date -u +%FT%TZ)" >> ~/results/phases.txt ) &
$P bench/bench.py --images ~/images_gamut --out ~/results/p1 --av1gpu ~/w4.sh --avifenc ~/opt/bin/avifenc \
  --avifdec ~/opt/bin/avifdec --ssimulacra2 ~/opt/bin/ssimulacra2 --skip-speed --aom-speeds 6,9 --svt-presets 4,6,9 \
  --svt-tune 3 --avifdec-codec dav1d --cvvdp standard_fhd --keep > ~/results/p1.log 2>&1
echo "bench exit $?" >> ~/results/p1.log
ph p1_bench_done
wait
$P ~/nvenc_score.py ~/av1gpu ~/nv/q ~/results/nvenc_p1.json standard_fhd > ~/results/nvenc_score_p1.log 2>&1
ph p1_nvenc_score_done
$P ~/bytes_check.py ~/av1gpu ~/results/p1/files ~/nv/q ~/results/bytes_p1.json > ~/results/bytes_p1.txt 2>&1
ph p1_done
