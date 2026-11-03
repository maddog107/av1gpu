#!/bin/bash
# size range identity, spd.py-method speed (photos, noise, screen content), --palette on encodes + scores; ~/pal/chain.log
L=~/pal/chain.log
step() { echo "$(date +%H:%M:%S) $*" >> $L; }
cd ~
step fetch; python3 ~/br/bench/fetch_images.py --manifest ~/br/bench/openimages_v4_gamut.csv --out ~/pal/sizerange > pal/sizerange.log 2>&1
step "$(tail -1 pal/sizerange.log)"
step ident; bash scripts/ident.sh > pal/ident_final.out 2>&1; step "$(grep '^IDENT' pal/ident_final.out | tr '\n' ' ')"
# lossless palettes: the same files as the lltest run (screen, GBR and YCgCo-Re; flat PNG)
for t in "screen:" "screen--ycgco-re:--ycgco-re" "flat_png:"; do
  d=${t%%:*}; x=${t#*:}; rm -rf ~/pal/llre; mkdir -p ~/pal/llre
  ~/br/build/av1gpu --batch ~/pal/lltest/$d/list.txt --outdir ~/pal/llre --lossless $x > /dev/null 2>&1
  n=0; same=0; for f in ~/pal/lltest/$d/auto/*.avif; do n=$((n + 1)); cmp -s $f ~/pal/llre/$(basename $f) && same=$((same + 1)); done
  step "LLRECHECK $d: $same/$n identical to the lltest run"
done
rm -rf pal/spd
step spd-race; python3 scripts/spd_pal.py race > pal/spd_race.out 2>&1
step spd-noise; python3 scripts/spd_pal.py noise > pal/spd_noise.out 2>&1
step spd-screen; python3 scripts/spd_pal.py screen > pal/spd_screen.out 2>&1
step "$(grep -h SUMMARY pal/spd_*.out | tr '\n' ' ')"
step palette-on; rm -rf ~/pal/out_on ~/pal/rows_on
STUDY_SETS="flat_png flat_jpg" STUDY_GPU_EXTRA="--palette on" STUDY_OUT=~/pal/out_on STUDY_ROWS=~/pal/rows_on STUDY_AV1GPU=~/br/build/av1gpu \
  python3 scripts/study.py encode-gpu > pal/on_enc.log 2>&1
STUDY_OUT=~/pal/out_on STUDY_ROWS=~/pal/rows_on STUDY_SCORE_SETS="flat_png flat_jpg" python3 scripts/study.py score > pal/on_score.log 2>&1
step ident-off; bash scripts/ident_off.sh > pal/ident_off.out 2>&1; step "$(grep IDENT_OFF pal/ident_off.out | tr '\n' ' ')"
step ctest2; (cd ~/br/build && PATH=$HOME/src/aom-build:$PATH ctest --output-on-failure -j 4 > ~/pal/ctest2.out 2>&1)
step "ctest2: $(grep 'tests passed' ~/pal/ctest2.out)"
step crops; AV1GPU_PAL=~/br/build/av1gpu python3 scripts/crops.py > pal/crops.out 2>&1; step "crops exit $?"
step CHAIN_DONE
