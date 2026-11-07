#!/bin/bash
# the final head b08c856: the checks again, one log each in ~/results/r4/
export BIN=$HOME/fix3/build/av1gpu; R=~/results/r4; mkdir -p $R; cd ~/t
step() { n=$1; shift; s=$(date +%s); "$@" > $R/$n.txt 2>&1; echo "$n: exit $? in $(( $(date +%s) - s )) s: $(tail -1 $R/$n.txt | cut -c1-150)"; }
step limits env LIMBINS=main,fix3 python3 limits.py
step tonemodel env UPRULE=jpeg420 python3 tonemodel.py
step alr python3 alr.py
step alr2 python3 alr2.py
step exifgeo python3 exifgeo.py
step exifpix python3 exifpix.py
step geom env LD_LIBRARY_PATH=$HOME/im/lib python3 geom.py
step combo python3 combo.py 1200
step cond env CROSSED_KEEP=1 python3 cond.py 2000
step bvs python3 bvs.py
step batchp python3 batchp.py
step alpharz python3 alpharz.py
step n3g5 bash n3g5.sh
step two bash two.sh
step repro bash ~/repro_for_aws.sh $BIN $HOME/im/bin/magick
step rsz python3 rsz.py
echo R4SUITE_DONE
