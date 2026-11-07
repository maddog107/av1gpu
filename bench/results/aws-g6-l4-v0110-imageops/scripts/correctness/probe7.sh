#!/bin/bash
export T=/dev/shm/p7; source ~/t/try.sh
export LD_LIBRARY_PATH=$HOME/im/lib
echo "== two-tone pictures and --skip-blank 99"
for f in adv/trim/vsplit_256x64.png adv/trim/hsplit_256x64.png pr/tests/gate_images/cfl_edge_211x149.png; do
  ~/im/bin/magick ~/$f -format "$f magick box fuzz0 %@ " info: 2>/dev/null; ~/im/bin/magick ~/$f -fuzz 10% -format "fuzz10 %@\n" info: 2>/dev/null
  try $B ~/$f $T/o.avif --trim --skip-blank 99
  try $B ~/$f $T/o.avif --trim --fuzz 10% --skip-blank 99
done
echo "== real images flagged blank"
for s in images_std images comics; do for z in 3 10 20 30; do
  rm -rf $T/o; $B --batch ~/$s --outdir $T/o --preset turbo3 --trim --fuzz $z% --skip-blank 99 > $T/out.txt 2> $T/err.txt; echo "$s fuzz $z: rc=$? $(grep -c ': blank' $T/err.txt) blank of $(ls ~/$s | wc -l): $(grep ': blank' $T/err.txt | sed 's/.*skipped //; s/: blank.*//' | xargs -n1 basename 2>/dev/null | head -4 | tr '\n' ' ')"
done; done
