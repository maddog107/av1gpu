#!/bin/bash
# wait for the running `lrstudy.py encode`, then score everything (so scoring does not disturb the encode timings)
while pgrep -f "lrstudy.py encode" > /dev/null; do sleep 5; done
python3 ~/bench/lrstudy.py score > ~/lr/score.log 2>&1
echo CHAIN_DONE >> ~/lr/score.log
