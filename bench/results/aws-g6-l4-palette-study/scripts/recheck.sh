#!/bin/bash
# after a detection-kernel change: the study's flat PNG + screen encodes again, cmp'd with the previous branch build's;
# detection scores of the photo sets in batch mode (GP_SC_DEBUG)
cd ~
rm -rf ~/pal/out_br3 ~/pal/rows_br3
STUDY_SETS="flat_png screen" STUDY_OUT=~/pal/out_br3 STUDY_ROWS=~/pal/rows_br3 STUDY_AV1GPU=~/br/build/av1gpu python3 scripts/study.py encode-gpu > pal/br3_enc.log 2>&1
for fmt in 420 444_10; do
  for s in flat_png screen; do
    n=0; same=0
    for f in ~/pal/out_br/gpu/$fmt/$s/*.avif; do n=$((n + 1)); cmp -s $f ~/pal/out_br3/gpu/$fmt/$s/$(basename $f) && same=$((same + 1)); done
    echo "RECHECK $fmt $s: $same/$n identical to the previous branch build"
  done
done
for s in sizerange std100; do
  ls ~/pal/$s/*.jpg > /tmp/l.txt
  GP_SC_DEBUG=1 ~/br/build/av1gpu --batch /tmp/l.txt --outdir /tmp/o_$s 2>&1 | grep sc-detect > pal/detect_$s.txt
  echo "DETECT $s: $(wc -l < pal/detect_$s.txt) images, $(awk '$NF > 1' pal/detect_$s.txt | wc -l) pass, max score $(awk '{print $NF}' pal/detect_$s.txt | sort -g | tail -1), libaom's rule would pass $(awk '$9 > 1' pal/detect_$s.txt | wc -l)"
done
ls ~/pal/sets/photo/*.jpg ~/pal/sets/photo/*.png > /tmp/l.txt
GP_SC_DEBUG=1 ~/br/build/av1gpu --batch /tmp/l.txt --outdir /tmp/o_photo 2>&1 | grep sc-detect > pal/detect_photo.txt
echo "DETECT study photos: $(wc -l < pal/detect_photo.txt) images, $(awk '$NF > 1' pal/detect_photo.txt | wc -l) pass, libaom's rule would pass $(awk '$9 > 1' pal/detect_photo.txt | wc -l)"
