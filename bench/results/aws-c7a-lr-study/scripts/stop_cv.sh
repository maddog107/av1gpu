#!/bin/bash
# stop cv_run.sh (and its watchdog) and every cvscore.py worker; run as a file so pkill does not match the caller
pkill -f "bench/cv_run.sh"
pkill -f "bench/cvscore.py"
sleep 3
pgrep -fa "bench/cv" || echo "none running"
