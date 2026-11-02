#!/bin/bash
# branch checks in order (one GPU job at a time); progress lines in ~/pal/chain.log
L=~/pal/chain.log
step() { echo "$(date +%H:%M:%S) $*" >> $L; }
cd ~
step build; (cd ~/br && cmake --build build -j 8 > ~/pal/chain_build.log 2>&1) || { step BUILD FAILED; exit 1; }
step ident; bash scripts/ident.sh > pal/ident.out 2>&1; step "$(grep -c '^IDENT' pal/ident.out) ident lines"
step lltest; bash scripts/lltest.sh screen > pal/lltest_screen.out 2>&1; bash scripts/lltest.sh flat_png > pal/lltest_flat.out 2>&1
bash scripts/lltest.sh screen --ycgco-re > pal/lltest_screen_yc.out 2>&1
step encode-gpu; STUDY_OUT=~/pal/out_br STUDY_ROWS=~/pal/rows_br STUDY_AV1GPU=~/br/build/av1gpu python3 scripts/study.py encode-gpu > pal/br_enc.log 2>&1
step gate; bash scripts/gate.sh > pal/gate2.out 2>&1
step score; STUDY_OUT=~/pal/out_br STUDY_ROWS=~/pal/rows_br STUDY_SCORE_SETS="flat_png screen" python3 scripts/study.py score > pal/br_score.log 2>&1
step CHAIN_DONE
