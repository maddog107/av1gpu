#!/bin/bash
# primary host, after the main timing phase: a second interleaved round, then the optional --autotiling timings
set -x
while kill -0 $(cat ~/timing_main.pid) 2>/dev/null; do sleep 10; done
python3 -u ~/timing.py r2
python3 -u ~/timing.py tiles
echo TJOB_DONE
