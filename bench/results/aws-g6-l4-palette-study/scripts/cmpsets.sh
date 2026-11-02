#!/bin/bash
# branch vs main av1gpu study outputs: identical files and total bytes per format and set
for fmt in 420 444_10; do
  for s in comics flat_png flat_jpg painted screen photo; do
    m=~/pal/out/gpu/$fmt/$s; b=~/pal/out_br/gpu/$fmt/$s
    n=0; same=0; mb=0; bb=0
    for f in $m/*.avif; do
      n=$((n + 1)); g=$b/$(basename $f)
      cmp -s $f $g && same=$((same + 1))
      mb=$((mb + $(stat -c %s $f))); bb=$((bb + $(stat -c %s $g)))
    done
    echo "$fmt $s identical $same/$n bytes main $mb branch $bb ratio $(echo "scale=4; $bb / $mb" | bc)"
  done
done
grep -h "av1gpu " ~/pal/gpu.log ~/pal/enc_gpu.log 2>/dev/null | sed 's/^/main: /'
