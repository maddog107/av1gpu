#!/bin/bash
# after the identity of the candidate: its wall time, the longer option-off rerun, release packages, modules, cold starts
L=~/results/after.log; A=$HOME/main/build/av1gpu; C=$HOME/bin/main_copy; B=$HOME/pr/build/av1gpu; P=$HOME/proto/pd/build/av1gpu
CONV="--trim --fuzz 10% --trim-margin 1% --trim-limits 1%,20% --trim-lightness 80 --skip-blank 99"; NV="--trim --fuzz 10% --trim-limits 100%,100%"
until grep -q AFTER6_DONE $L; do sleep 5; done
if grep -q "IDENTITY_DONE pd: [0-9]* files, 0 differ" ~/results/idn_pd.txt; then
  echo "$(date -u +%H:%M:%S) wall pd" >> $L
  python3 ~/spd65.py wpd 12 "plain=$B;neverPR=$B|$NV;neverPD=$P|$NV;t10PR=$B|--trim --fuzz 10%;t10PD=$P|--trim --fuzz 10%;convPR=$B|$CONV;convPD=$P|$CONV;autoPR=$B|--auto-level;autoPD=$P|--auto-level" -- border b444 comics sr thumbs > ~/results/spd_wpd.txt 2>&1
else echo "$(date -u +%H:%M:%S) pd NOT IDENTICAL: no wall run" >> $L; fi
echo "$(date -u +%H:%M:%S) off2" >> $L; python3 ~/spd65.py off2 24 "A=$A;C=$C;B=$B" -- std thumbs pngwebp > ~/results/spd_off2.txt 2>&1
echo "$(date -u +%H:%M:%S) coldon" >> $L
python3 ~/cold65.py on 16 "A=$A;B=$B;Btrim=$B|--trim --fuzz 10%;Blevel=$B|--level 5%,95%,1.1;Bauto=$B|--auto-level;Bgray=$B|--grayscale Rec709Luma" photo_2.5MP comic comic_444_10 page_png > ~/results/cold_on.txt 2>&1
echo "$(date -u +%H:%M:%S) rel build" >> $L; bash ~/rel65.sh > ~/results/rel65.out 2>&1
echo "$(date -u +%H:%M:%S) mods" >> $L; bash ~/mods65.sh > ~/results/mods65.out 2>&1
R=$HOME/rel/av1gpu-0.10.0-linux-x86_64/av1gpu; M=$HOME/rel13/main.bin; Q=$HOME/pkg/pr13/av1gpu-dev-e779036-linux-x86_64/av1gpu
if [ -x $M ] && [ -x $Q ]; then
  cp $R ~/bin/v0100_copy
  echo "$(date -u +%H:%M:%S) cold rel" >> $L; python3 ~/cold65.py rel 24 "R=$R;Rc=$HOME/bin/v0100_copy;M=$M;Q=$Q" > ~/results/cold_rel.txt 2>&1
  echo "$(date -u +%H:%M:%S) spd rel" >> $L; python3 ~/spd65.py rel 8 "R=$R;Rc=$HOME/bin/v0100_copy;M=$M;Q=$Q" -- sr comics c444 thumbs > ~/results/spd_rel.txt 2>&1
fi
echo "$(date -u +%H:%M:%S) AFTER7_DONE" >> $L
