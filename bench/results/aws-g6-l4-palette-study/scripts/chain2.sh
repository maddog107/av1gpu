#!/bin/bash
# after a lossy palette change: identity, study encodes, cmp, gates, scores (one GPU job at a time); ~/pal/chain.log
L=~/pal/chain.log
step() { echo "$(date +%H:%M:%S) $*" >> $L; }
cd ~
TAG=${1:-br2}
step "chain2 $TAG build"; (cd ~/br && cmake --build build -j 8 > ~/pal/chain_build.log 2>&1) || { step BUILD FAILED; exit 1; }
step ident; bash scripts/ident.sh > pal/ident_$TAG.out 2>&1; step "$(grep '^IDENT' pal/ident_$TAG.out | tr '\n' ' ')"
rm -rf ~/pal/out_br ~/pal/rows_br
step encode-gpu; STUDY_OUT=~/pal/out_br STUDY_ROWS=~/pal/rows_br STUDY_AV1GPU=~/br/build/av1gpu python3 scripts/study.py encode-gpu > pal/br_enc_$TAG.log 2>&1
bash scripts/cmpsets.sh > pal/cmpsets_$TAG.out 2>&1
step gate; bash scripts/gate.sh > pal/gate_$TAG.out 2>&1; step "$(grep -c PASS pal/gate_$TAG.out) gates pass"
step score; STUDY_OUT=~/pal/out_br STUDY_ROWS=~/pal/rows_br STUDY_SCORE_SETS="flat_png screen" python3 scripts/study.py score > pal/br_score_$TAG.log 2>&1
(cd scripts && python3 paluse.py ~/pal/out_br 60 > ~/pal/paluse_$TAG.out 2>&1)
step CHAIN_DONE
