#!/bin/bash
L=~/results/after.log
until grep -q AFTER8_DONE $L; do sleep 5; done
echo "$(date -u +%H:%M:%S) jit" >> $L; bash ~/jit65.sh
echo "$(date -u +%H:%M:%S) AFTER9_DONE" >> $L
