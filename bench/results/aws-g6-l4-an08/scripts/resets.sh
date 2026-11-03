#!/bin/bash
# rerun the set builders after linking ~/br to the main tree (mk_ll_sets.py reads ~/br/docs/charts and encoder.cpp)
ln -sfn ~/old ~/br
L=~/results/setup
cd ~
python3 ~/mk_ll_sets.py > $L/llsets.log 2>&1; echo "exit $?" >> $L/llsets.log
python3 ~/mk_sets.py > $L/sets.log 2>&1; echo "exit $?" >> $L/sets.log
bash ~/mk_mono.sh > $L/mono.log 2>&1; echo "exit $?" >> $L/mono.log
echo SETS_DONE >> $L/mono.log
