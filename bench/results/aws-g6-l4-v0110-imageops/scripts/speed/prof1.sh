#!/bin/bash
# prof1.sh BIN TAG: kernel cost without contention: one encoder stream (GP_ENCODERS=1), the comic pages at both formats and the
# size range; per config the kernel times and counts from nsys.  ~/prof/TAG1/
B=$1; TAG=$2; D=~/prof/${TAG}1; mkdir -p $D; export GP_ENCODERS=1
CONV="--trim --fuzz 10% --trim-margin 1% --trim-limits 1%,20% --trim-lightness 80 --skip-blank 99"
NV="--trim --fuzz 10% --trim-limits 100%,100%"
C444="--yuv 444 --depth 10 --tiles 2,2,2 --max-mp 100 --max-input-mp 100"
run() { local s=$1 src=$2 c=$3; shift 3; rm -rf /dev/shm/out
  nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true -o $D/${s}_$c $B --batch $src --outdir /dev/shm/out --stats "$@" > $D/${s}_$c.out 2> $D/${s}_$c.err
  nsys export --type sqlite --force-overwrite=true -o $D/${s}_$c.sqlite $D/${s}_$c.nsys-rep > /dev/null 2>&1
  python3 ~/prof_an.py $D/${s}_$c.sqlite ${s}_$c | tee $D/${s}_$c.an; rm -f $D/${s}_$c.nsys-rep $D/${s}_$c.sqlite; }
for cfg in "plain|" "never|$NV" "conv|$CONV" "trim0|--trim --fuzz 0%" "level|--level 5%,95%,1.1" "levelnog|--level 5%,95%" "auto|--auto-level" "gray|--grayscale Rec709Luma"; do
  c=${cfg%%|*}; o=${cfg#*|}
  run b444 ~/sets/border $c $C444 $o
  run border ~/sets/border $c $o
  run c444 ~/sets/comics $c $C444 $o
  run comics ~/sets/comics $c $o
  run sr ~/images $c $o
done
echo PROF1_DONE $TAG
