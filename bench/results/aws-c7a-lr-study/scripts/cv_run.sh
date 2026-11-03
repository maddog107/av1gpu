#!/bin/bash
# CVVDP scoring in three size bands (cvscore.py), with a memory watchdog: below 6 GB available it stops the run
# rather than letting the OOM killer lock the instance up (as butteraugli did).  Log ~/lr/cvvdp.log.
L=~/lr/cvvdp.log
( while sleep 2; do
    a=$(awk '/MemAvailable/ {print int($2 / 1024)}' /proc/meminfo)
    if [ "$a" -lt 6000 ]; then echo "WATCHDOG: $a MB available, stopping" >> $L; pkill -f cvscore.py; fi
  done ) &
W=$!
PY=~/cv/bin/python
CV_WORKERS=64 CV_THREADS=1 $PY ~/bench/cvscore.py 0 1.1 >> $L 2>&1
CV_WORKERS=32 CV_THREADS=2 $PY ~/bench/cvscore.py 1.1 5 >> $L 2>&1
CV_WORKERS=7 CV_THREADS=9 $PY ~/bench/cvscore.py 5 100 s6 p6keep p6full >> $L 2>&1
echo CV_RUN_DONE >> $L
kill $W
