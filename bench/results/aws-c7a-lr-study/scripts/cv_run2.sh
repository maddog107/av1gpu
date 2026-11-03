#!/bin/bash
# The large-photo band of cv_run.sh again, wider: 7 workers x 9 threads used only ~3 cores each (40 min for 600
# jobs), so 14 workers x 4 threads, s6 and p6full first, then p6keep; cvscore.py skips the rows already done.
# Same memory watchdog.  Log ~/lr/cvvdp.log.
L=~/lr/cvvdp.log
( while sleep 2; do
    a=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
    if [ "$a" -lt 6000 ]; then echo "WATCHDOG: $a MB available, stopping" >> $L; pkill -f cvscore.py; fi
  done ) &
W=$!
CV_WORKERS=14 CV_THREADS=4 ~/cv/bin/python ~/bench/cvscore.py 5 100 s6 p6full p6keep >> $L 2>&1
echo CV_RUN2_DONE >> $L
kill $W
