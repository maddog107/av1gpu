#!/bin/bash
# the final candidate: build, identity against the PR, then wall time
L=~/results/after.log; B=$HOME/pr/build/av1gpu; P=$HOME/proto/pd/build/av1gpu
until grep -q AFTER5_DONE $L; do sleep 5; done
echo "$(date -u +%H:%M:%S) pd build" >> $L; bash ~/mkproto.sh pd > ~/results/proto_pd.txt 2>&1
echo "$(date -u +%H:%M:%S) idn pd" >> $L; bash ~/idn65.sh $B $P pd > ~/results/idn_pd.out 2>&1
echo "$(date -u +%H:%M:%S) AFTER6_DONE" >> $L
