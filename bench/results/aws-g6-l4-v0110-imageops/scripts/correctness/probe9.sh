#!/bin/bash
export T=/dev/shm/p9; source ~/t/try.sh
export LD_LIBRARY_PATH=$HOME/im/lib
P=~/images_std/06abed7823b7f37f.jpg
~/im/bin/magick $P -format "input %wx%h\n" info:
for g in @100000 @1048576 @1048577 @2000000 @4000000; do ~/im/bin/magick $P -resize $g -format "magick $g: %wx%h\n" info:; try $B $P $T/o.avif --resize $g --max-mp 100; tail -3 $T/err.txt | head -1 | cut -c1-120; grep -m1 'bad --resize' $T/err.txt | cut -c1-160; done
rm -rf $T/o; printf '%s\tresize=@2000000\n' $P > $T/l; echo ~/images_std/07e1062253462b85.jpg >> $T/l; $B --batch $T/l --outdir $T/o 2>&1 | grep -E 'skipped|bad' | cut -c1-200; ls $T/o | wc -l
