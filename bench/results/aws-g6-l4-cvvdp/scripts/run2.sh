#!/bin/bash
# Phase 2: two extra high-quality points per encoder so the curves reach 9.5 JOD: avifenc/av1gpu q85,90;
# WebP q98,100; NVENC CQ 12,6.  Same commands as phase 1 otherwise.  Then the WebP ICC side check.
set -x
cd ~/av1gpu
P=~/venv/bin/python
export LD_LIBRARY_PATH=$HOME/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
ph() { echo "PHASE $1 $(date -u +%FT%TZ)" >> ~/results/phases.txt; }
ph p2_start
( $P ~/nvenc_encode.py ~/nv/q2 12,6 > ~/results/nvenc_encode_p2.log 2>&1; echo "PHASE nvenc_enc_p2_done $(date -u +%FT%TZ)" >> ~/results/phases.txt ) &
$P bench/bench.py --images ~/images_gamut --out ~/results/p2 --av1gpu ~/w4.sh --avifenc ~/opt/bin/avifenc \
  --avifdec ~/opt/bin/avifdec --ssimulacra2 ~/opt/bin/ssimulacra2 --skip-speed --aom-speeds 6,9 --svt-presets 4,6,9 \
  --svt-tune 3 --avifdec-codec dav1d --cvvdp standard_fhd --quality 85,90 --webp-quality 98,100 --keep > ~/results/p2.log 2>&1
echo "bench exit $?" >> ~/results/p2.log
ph p2_bench_done
wait
$P ~/nvenc_score.py ~/av1gpu ~/nv/q2 ~/results/nvenc_p2.json standard_fhd > ~/results/nvenc_score_p2.log 2>&1
ph p2_nvenc_score_done
$P ~/webp_icc.py ~/results/p1/files/webp ~/results/p2/files/webp ~/results/webp_icc.json > ~/results/webp_icc.log 2>&1
$P ~/bytes_check.py ~/av1gpu ~/results/p2/files ~/nv/q2 ~/results/bytes_p2.json > ~/results/bytes_p2.txt 2>&1
ph p2_done
