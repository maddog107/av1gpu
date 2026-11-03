#!/bin/bash
# --palette off against main on the images palette is tried on (screen + flat PNG): every file cmp'd
M=~/av1gpu/build/av1gpu; BR=~/br/build/av1gpu
O=~/pal/ident_off; rm -rf $O; mkdir -p $O
ls ~/pal/sets/screen/*.png ~/pal/sets/flat_png/*.png > $O/list.txt
run() {
  local n=$1; shift
  mkdir -p $O/$n/main $O/$n/br
  $M --batch $O/list.txt --outdir $O/$n/main "$@" > /dev/null 2>&1
  $BR --batch $O/list.txt --outdir $O/$n/br --palette off "$@" > /dev/null 2>&1
  local tot=0 same=0
  for f in $O/$n/main/*.avif; do tot=$((tot + 1)); cmp -s $f $O/$n/br/$(basename $f) && same=$((same + 1)); done
  echo "IDENT_OFF $n: $same/$tot identical"
}
run default
run 444_10 --yuv 444 --depth 10
run lossless --lossless
run lossless_ycgco --lossless --ycgco-re
