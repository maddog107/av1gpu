#!/bin/bash
# round 3: the round-1 / round-2 checks on the fix head, in sequence; one log per check in ~/results/r3/
export BIN=$HOME/fix/build/av1gpu; R=~/results/r3; mkdir -p $R; cd ~/t
export LD_LIBRARY_PATH_IM=$HOME/im/lib
step() { n=$1; shift; s=$(date +%s); "$@" > $R/$n.txt 2>&1; echo "$n: exit $? in $(( $(date +%s) - s )) s: $(tail -1 $R/$n.txt | cut -c1-150)"; }
step exifgeo python3 exifgeo.py
step exifpix python3 exifpix.py
step geom env LD_LIBRARY_PATH=$HOME/im/lib python3 geom.py
step combo python3 combo.py 1200
step cond env CROSSED_KEEP=1 python3 cond.py 2000
step bvs python3 bvs.py
step batchp python3 batchp.py
step limits env LIMBINS=main,fix python3 limits.py
step alpharz python3 alpharz.py
step n3g5 bash n3g5.sh
step n3b python3 n3b.py
step repro bash ~/repro_for_aws.sh $BIN $HOME/im/bin/magick
step tonemodel python3 tonemodel.py
step alr python3 alr.py
step rsz python3 rsz.py
echo R3SUITE_DONE
