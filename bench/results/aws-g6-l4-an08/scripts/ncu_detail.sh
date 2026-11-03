#!/bin/bash
# ncu_detail.sh BIN TAG SET KREGEX [N]: detailed ncu sections (SOL, occupancy, launch, warp state, scheduler, memory,
# source counters) for the first N launches matching KREGEX on one pass of SET -> ~/results/prof/TAG_detail.txt
BIN=$1; T=$2; SET=$3; K=$4; N=${5:-1}
R=~/results/prof; mkdir -p $R
sudo rm -rf /tmp/ncud_$T
sudo /usr/local/cuda/bin/ncu --kernel-name-base demangled --kernel-name "regex:$K" --launch-count $N \
  --section SpeedOfLight --section Occupancy --section LaunchStats --section WarpStateStats --section SchedulerStats \
  --section MemoryWorkloadAnalysis --section SourceCounters --section ComputeWorkloadAnalysis --section InstructionStats \
  -o /tmp/ncud_$T -f $BIN --batch $SET --outdir /tmp/ncud_out_$T > $R/${T}_detail.out 2>&1
sudo /usr/local/cuda/bin/ncu -i /tmp/ncud_$T.ncu-rep --page details > $R/${T}_detail.txt 2>&1
sudo /usr/local/cuda/bin/ncu -i /tmp/ncud_$T.ncu-rep --page source --csv --print-source sass > $R/${T}_sass.csv 2>&1
echo DETAIL_DONE >> $R/${T}_detail.txt
