#!/bin/bash
# the release cold start again: the first run overlapped a large copy off the instance
L=~/results/after.log; R=$HOME/rel/av1gpu-0.10.0-linux-x86_64/av1gpu; M=$HOME/rel13/main.bin; Q=$HOME/pkg/pr13/av1gpu-dev-e779036-linux-x86_64/av1gpu
until grep -q AFTER9_DONE $L; do sleep 5; done
echo "$(date -u +%H:%M:%S) cold rel2" >> $L; python3 ~/cold65.py rel2 24 "R=$R;Rc=$HOME/bin/v0100_copy;M=$M;Q=$Q" > ~/results/cold_rel2.txt 2>&1
echo "$(date -u +%H:%M:%S) AFTER10_DONE" >> $L
