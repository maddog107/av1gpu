#!/bin/bash
# av1gpu's screen-content detection score per image (GP_SC_DEBUG; > 1: palette search on), one image per run
BR=~/br/build/av1gpu
O=~/pal/detect; mkdir -p $O
for s in comics flat_png flat_jpg painted photo screen; do
  for f in ~/pal/sets/$s/*; do
    GP_SC_DEBUG=1 $BR "$f" $O/t.avif 2>&1 | grep sc-detect | head -1 | sed "s|^|$s $(basename $f) |"
  done
done
for f in ~/pal/std100/*.jpg; do
  GP_SC_DEBUG=1 $BR "$f" $O/t.avif 2>&1 | grep sc-detect | head -1 | sed "s|^|std100 $(basename $f) |"
done
