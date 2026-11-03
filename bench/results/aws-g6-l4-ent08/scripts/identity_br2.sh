#!/bin/bash
# identity of br2 (both commits) against main, concurrently with br1's run: own work dirs (~/work2) and result file
cd ~
sed -e 's#~/work/cmp#~/work2/cmp#' ~/cmpset.sh > ~/cmpset2.sh
sed -e 's#cmpset.sh#cmpset2.sh#; s#~/work/single#~/work2/single#' ~/identity8.sh > ~/identity8b.sh
mkdir -p ~/work2
grep -n 'work2\|cmpset2' ~/identity8b.sh ~/cmpset2.sh | head
nohup bash ~/identity8b.sh ~/main/build/av1gpu ~/br2/build/av1gpu br2 > ~/identity_br2.out 2>&1 &
sleep 1
echo started
