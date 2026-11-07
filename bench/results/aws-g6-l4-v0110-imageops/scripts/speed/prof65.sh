#!/bin/bash
# prof65.sh BIN TAG: nsys traces (CUDA only) of one pass per set and config, plus --stats; the .sqlite goes to ~/prof/TAG/.
B=$1; TAG=$2; D=~/prof/$TAG; mkdir -p $D
CONV="--trim --fuzz 10% --trim-margin 1% --trim-limits 1%,20% --trim-lightness 80 --skip-blank 99"
NV="--trim --fuzz 10% --trim-limits 100%,100%"
C444="--yuv 444 --depth 10 --tiles 2,2,2 --max-mp 100 --max-input-mp 100"
run() { # set src cfg opts...
  local s=$1 src=$2 c=$3; shift 3; rm -rf /dev/shm/out
  nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true -o $D/${s}_$c $B --batch $src --outdir /dev/shm/out --stats "$@" > $D/${s}_$c.out 2> $D/${s}_$c.err
  nsys export --type sqlite --force-overwrite=true -o $D/${s}_$c.sqlite $D/${s}_$c.nsys-rep > /dev/null 2>&1
  python3 ~/prof_an.py $D/${s}_$c.sqlite ${s}_$c | tee $D/${s}_$c.an
  rm -f $D/${s}_$c.nsys-rep
}
[ -n "$ONLY_BORDER" ] && { for cfg in "plain|" "never|$NV" "conv|$CONV" "trim10|--trim --fuzz 10%"; do c=${cfg%%|*}; o=${cfg#*|}; run b444 ~/work/set_b444/speedset $c $C444 $o; run border ~/work/set_border/speedset $c $o; done; echo PROF_DONE $TAG; exit 0; }
for cfg in "plain|" "never|$NV" "conv|$CONV" "trim10|--trim --fuzz 10%" "level|--level 5%,95%,1.1" "auto|--auto-level" "gray|--grayscale Rec709Luma" "negate|--negate"; do
  c=${cfg%%|*}; o=${cfg#*|}
  run c444 ~/work/set_c444/speedset $c $C444 $o
  run comics ~/work/set_comics/speedset $c $o
  run sr ~/work/set_sr/speedset $c $o
  run thumbs ~/work/set_thumbs/speedset $c $o
done
echo PROF_DONE $TAG
