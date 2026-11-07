#!/bin/bash
export T=/dev/shm/p5; source ~/t/try.sh
cd $T; G=$(ls ~/bp/e/*.png | head -8); rm -f l*
i=0; for g in 0.5 0.4x0.4! x0.4 1x100000! 100000x1! 1x65537! 65537x1! 1x65536! 65536x1! 30000x1! 1x30000! 32x32000! 32000x32! 2x65000! 65000x2!; do
  { echo "$G" | head -3; echo "$(echo "$G" | sed -n 4p)	resize=$g"; echo "$G" | tail -4; } > l
  rm -rf o; timeout 120 $B --batch l --outdir o > out.txt 2> err.txt; rc=$?
  echo "resize=$g: rc=$rc files=$(ls o 2>/dev/null | wc -l)/8: $(grep -v 'stage ms' err.txt | grep -v '^\[av1gpu\] [0-9]' | tail -2 | cut -c1-170 | tr '\n' ' ')"
done
echo "== truncated JPEG, main vs PR (single, batch)"
try $M ~/adv/broken/trunc.jpg $T/o.avif; try $B ~/adv/broken/trunc.jpg $T/o.avif
