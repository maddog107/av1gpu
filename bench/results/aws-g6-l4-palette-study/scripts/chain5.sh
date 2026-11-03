#!/bin/bash
# final build's checks after the photo race: gates, --palette off identity, ctest (aomdec on PATH), screen-content
# speed, --palette on encodes + scores, crops; ~/pal/chain.log
L=~/pal/chain.log
step() { echo "$(date +%H:%M:%S) $*" >> $L; }
cd ~
step "chain5 ident"; bash scripts/ident.sh > pal/ident_final.out 2>&1; step "$(grep '^IDENT' pal/ident_final.out | tr '\n' ' ')"
rm -rf pal/spd
step "chain5 spd-race"; python3 scripts/spd_pal_off.py race > pal/spd_race.out 2>&1; step "$(grep SUMMARY pal/spd_race.out | tr '\n' ' ')"
step "chain5 spd-noise"; python3 scripts/spd_pal_off.py noise > pal/spd_noise.out 2>&1; step "$(grep SUMMARY pal/spd_noise.out)"
step "chain5 spd-off"; python3 scripts/spd_pal_off.py off > pal/spd_off2.out 2>&1; step "$(grep SUMMARY pal/spd_off2.out)"
step "chain5 gate"; bash scripts/gate.sh > pal/gate_final.out 2>&1; step "$(grep -c PASS pal/gate_final.out) of $(grep -c GATE pal/gate_final.out) gates pass"
step ident-off; bash scripts/ident_off.sh > pal/ident_off.out 2>&1; step "$(grep IDENT_OFF pal/ident_off.out | tr '\n' ' ')"
step ctest; (cd ~/br/build && PATH=$HOME/src/aom-build:$PATH ctest --output-on-failure -j 4 > ~/pal/ctest2.out 2>&1); step "ctest: $(grep 'tests passed' ~/pal/ctest2.out)"
step spd-screen; python3 scripts/spd_pal_off.py screen > pal/spd_screen.out 2>&1; step "$(grep SUMMARY pal/spd_screen.out | tr '\n' ' ')"
step palette-on; rm -rf ~/pal/out_on ~/pal/rows_on
STUDY_SETS="flat_png flat_jpg" STUDY_GPU_EXTRA="--palette on" STUDY_OUT=~/pal/out_on STUDY_ROWS=~/pal/rows_on STUDY_AV1GPU=~/br/build/av1gpu \
  python3 scripts/study.py encode-gpu > pal/on_enc.log 2>&1
STUDY_OUT=~/pal/out_on STUDY_ROWS=~/pal/rows_on STUDY_SCORE_SETS="flat_png flat_jpg" python3 scripts/study.py score > pal/on_score.log 2>&1
step crops; AV1GPU_PAL=~/br/build/av1gpu python3 scripts/crops.py > pal/crops.out 2>&1; step "crops exit $?"
step CHAIN_DONE
