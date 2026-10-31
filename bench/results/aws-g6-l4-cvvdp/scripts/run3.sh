#!/bin/bash
# Phase 3 (second instance): re-score the 4 photos whose JPEG ICC profile Pillow missed (APP2 after SOF0), with the
# fixed bench.srgb8, at every quality point of phases 1 and 2.  The encodes must have the same bytes and SSIMULACRA2.
set -x
cd ~/av1gpu
P=~/venv/bin/python
export LD_LIBRARY_PATH=$HOME/venv/lib/python3.12/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
ph() { echo "PHASE $1 $(date -u +%FT%TZ)" >> ~/results/phases.txt; }
mkdir -p ~/images4
for f in 370f7c330a75892f_large.jpg bd4ff3b451c3ba6b_large.jpg ca12d94f3a1dca2c_large.jpg fe00a8d898a01166_large.jpg; do
  ln -f ~/images_gamut/$f ~/images4/$f
done
ph p3_start
( NV_SRC=$HOME/images4 $P ~/nvenc_encode.py ~/nv/q3 6,12,18,25,32,38,44,50 > ~/results/nvenc_encode_p3.log 2>&1 ) &
$P bench/bench.py --images ~/images4 --out ~/results/p3 --av1gpu ~/w4.sh --avifenc ~/opt/bin/avifenc \
  --avifdec ~/opt/bin/avifdec --ssimulacra2 ~/opt/bin/ssimulacra2 --skip-speed --aom-speeds 6,9 --svt-presets 4,6,9 \
  --svt-tune 3 --avifdec-codec dav1d --cvvdp standard_fhd --quality 30,40,50,60,70,80,85,90 \
  --webp-quality 50,60,70,80,90,95,98,100 --keep > ~/results/p3.log 2>&1
echo "bench exit $?" >> ~/results/p3.log
wait
$P ~/nvenc_score.py ~/av1gpu ~/nv/q3 ~/results/nvenc_p3.json standard_fhd > ~/results/nvenc_score_p3.log 2>&1
$P ~/cvvdp_luma.py ~/results/cvvdp_luma_p3.json av1gpu=$HOME/results/p3/files/av1gpu aom_s6=$HOME/results/p3/files/aom_s6 \
  > ~/results/cvvdp_luma_p3.log 2>&1
ph p3_done
