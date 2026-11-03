#!/bin/bash
# collect.sh: the small result files -> ~/collect.tgz (no images, no binaries, no big ncu CSVs)
C=~/collect; rm -rf $C; mkdir -p $C/prof $C/race $C/v1 $C/v2pre $C/runs $C/sanitizer
cd ~/results
cp env.txt identity_new2.txt identity_pal_new2.txt identity_rerun_new2.txt gate_new2.txt ctest_new2.txt \
   sanitizer_new2b.txt speed2.log clocks_start2.txt clocks_end2.txt occ2.txt quick_ef1.txt quick_ef2.txt $C/ 2>/dev/null
cp ab3*.json* noise*.json* single.jsonl $C/runs/ 2>/dev/null
[ -f dmon2.txt ] && cp dmon2.txt $C/
cp race_new2.txt $C/race/ 2>/dev/null
for f in race/new2_0.log race/new2_1.log; do [ -f $f ] && grep -E "RACECHECK SUMMARY|Race reported|^rc " $f > $C/race/$(basename $f .log)_pairs.txt; done
cp v1/* $C/v1/ 2>/dev/null
cp v2pre/identity_new2_partial.txt v2pre/gate_new2.txt $C/v2pre/ 2>/dev/null
[ -f v2pre/race_new2_0_partial.log ] && grep -E "Race reported|and Write access" v2pre/race_new2_0_partial.log | head -40 > $C/v2pre/race_new2_0_partial_head.txt
for f in san_new2b/*.log; do [ -f $f ] && tail -n 5 $f > $C/sanitizer/$(basename $f); done
cd prof
cp *_ksum.txt *_stage.txt ptxv_*.txt $C/prof/ 2>/dev/null
for k in std_168 sr_6464 sr_1616 std_small th_816 th_s416; do
  [ -f ${k}_sass.csv ] && python3 ~/sass_sum.py ${k}_sass.csv > $C/prof/${k}_sass_summary.txt 2>&1
  [ -f ${k}_detail.txt ] && cp ${k}_detail.txt $C/prof/
done
for k in n2_3232 o_3232 n2_6464 n2_6432; do [ -f ${k}_detail.txt ] && cp ${k}_detail.txt $C/prof/; done
cd ~ && tar -czf collect.tgz collect && du -sh collect collect.tgz
