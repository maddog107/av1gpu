#!/bin/bash
# Alongside cv_run2.sh (which used only ~28 of 64 cores): p6keep's large photos in a second cvscore.py, 8 workers x
# 4 threads.  cv_run2.sh is stopped once its s6 and p6full rows are in (its p6keep jobs would repeat these).
CV_WORKERS=8 CV_THREADS=4 ~/cv/bin/python ~/bench/cvscore.py 5 100 p6keep >> ~/lr/cvvdp3.log 2>&1
echo CV_RUN3_DONE >> ~/lr/cvvdp3.log
