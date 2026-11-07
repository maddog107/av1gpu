#!/bin/bash
L=~/results/after.log; B=$HOME/pr/build/av1gpu
until grep -q AFTER7_DONE $L; do sleep 5; done
echo "$(date -u +%H:%M:%S) idn border" >> $L
bash ~/idnb.sh $B $HOME/proto/pd/build/av1gpu pd; bash ~/idnb.sh $B $HOME/proto/p1/build/av1gpu p1
echo "$(date -u +%H:%M:%S) idn p1" >> $L; bash ~/idn65.sh $B $HOME/proto/p1/build/av1gpu p1 > ~/results/idn_p1.out 2>&1
echo "$(date -u +%H:%M:%S) AFTER8_DONE" >> $L
