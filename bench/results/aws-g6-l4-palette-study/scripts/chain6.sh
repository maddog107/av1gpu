#!/bin/bash
# experiment: GP_PAL_FAST=1 (k-means only at the largest palette size, stop at the first size without improvement):
# bytes / quality on the screen and flat PNG sets, and screen-content speed; ~/pal/chain.log
L=~/pal/chain.log
step() { echo "$(date +%H:%M:%S) $*" >> $L; }
cd ~
step "chain6 build"; (cd ~/br && cmake --build build -j 8 > ~/pal/chain_build.log 2>&1) || { step BUILD FAILED; exit 1; }
bash scripts/recheck.sh > pal/recheck6.out 2>&1; step "$(grep RECHECK pal/recheck6.out | tr '\n' ' ')"
rm -rf ~/pal/out_fast ~/pal/rows_fast
GP_PAL_FAST=1 STUDY_SETS="flat_png screen" STUDY_OUT=~/pal/out_fast STUDY_ROWS=~/pal/rows_fast STUDY_AV1GPU=~/br/build/av1gpu \
  python3 scripts/study.py encode-gpu > pal/fast_enc.log 2>&1
STUDY_OUT=~/pal/out_fast STUDY_ROWS=~/pal/rows_fast STUDY_SCORE_SETS="flat_png screen" python3 scripts/study.py score > pal/fast_score.log 2>&1
step "fast scored"
GP_PAL_FAST=1 python3 scripts/spd_pal_fast.py graphics > pal/spd_screen_fast.out 2>&1
step "$(grep SUMMARY pal/spd_screen_fast.out | tr '\n' ' ')"
step CHAIN_DONE
