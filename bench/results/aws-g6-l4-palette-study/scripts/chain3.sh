#!/bin/bash
# final branch checks: speed (main vs branch), ctest, GPU fuzz, crops at equal size; ~/pal/chain.log
L=~/pal/chain.log
step() { echo "$(date +%H:%M:%S) $*" >> $L; }
cd ~
step speed; bash scripts/speed.sh > pal/speed.out 2>&1; step "$(grep -c SPEED pal/speed.out) speed lines"
step ctest; (cd ~/br/build && ctest --output-on-failure -j 4 > ~/pal/ctest.out 2>&1); step "ctest: $(grep 'tests passed' pal/ctest.out)"
step crops; AV1GPU_PAL=~/br/build/av1gpu python3 scripts/crops.py > pal/crops.out 2>&1; step "crops exit $?"
step CHAIN_DONE
