#!/bin/bash
# gather the branch checks into ~/checks.tgz (checks/ of the results dir)
C=~/checks; rm -rf $C; mkdir -p $C/spd $C/rows
cd ~/pal
cp ident_final.out ident_off.out gate_final.out ctest2.out lltest_screen.out lltest_flat.out lltest_screen_yc.out \
   cmpsets_br2.out recheck5.out paluse_br2.out chain.log $C/ 2>/dev/null
cp spd_race.out spd_noise.out spd_off2.out spd_screen.out $C/ 2>/dev/null
cat detect_sizerange.txt | sed 's/^/sizerange /' > $C/detect_photos.txt
cat detect_std100.txt | sed 's/^/standard /' >> $C/detect_photos.txt
cat detect_photo.txt | sed 's/^/study_photos /' >> $C/detect_photos.txt
cp spd/*.jsonl spd/*_summary.json $C/spd/ 2>/dev/null
mkdir -p $C/gate; cp gate/*.log $C/gate/ 2>/dev/null
[ -f rows_br/scored_lossy.jsonl ] && gzip -9 -c rows_br/scored_lossy.jsonl > $C/rows/branch_scored_lossy.jsonl.gz
[ -f rows_on/scored_lossy.jsonl ] && gzip -9 -c rows_on/scored_lossy.jsonl > $C/rows/palette_on_scored_lossy.jsonl.gz
cp crops.out $C/ 2>/dev/null
mkdir -p $C/crops; cp crops/*.png crops/crops.json $C/crops/ 2>/dev/null
tar -czf ~/checks.tgz -C ~ checks
ls -la ~/checks.tgz
