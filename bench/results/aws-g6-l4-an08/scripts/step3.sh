#!/bin/bash
# profiling step 3 (lineinfo build): source-line profiles of k_analysis<8,16> and k_analysis_small<4,16> on the
# thumbnails, k_analysis<16,16> on the size range
B=~/new/build_li/av1gpu
R=~/results/prof
for spec in "th_816:$HOME/sets/thumbs:k_analysis<.int.8, .int.16," "th_s416:$HOME/sets/thumbs:k_analysis_small<.int.4, .int.16," "sr_li1616:$HOME/images:k_analysis<.int.16, .int.16,"; do
  T=${spec%%:*}; rest=${spec#*:}; SET=${rest%%:*}; K=${rest#*:}
  bash ~/ncu_detail.sh $B $T $SET "$K" 1
  sudo /usr/local/cuda/bin/ncu -i /tmp/ncud_$T.ncu-rep --page source --csv --print-source cuda > $R/${T}_cuda.csv 2>&1
done
echo STEP3_DONE > $R/step3_done
