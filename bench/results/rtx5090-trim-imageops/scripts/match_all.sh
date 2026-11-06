#!/bin/bash
# The match table: --dump-source planes against ImageMagick 7 for each operation.
#   match_all.sh BIN GATE_PNG PAGE_PNG PNG16 WORKDIR
B=$1; G=$2; P=$3; P16=$4; W=$5; D=$(dirname "$0")
row() { printf "| %s | %s | " "$1" "$2"; python3 $D/match8.py $B "$3" "$4" -- $5 | sed "s/: / | /; s/ | \([UV]\)/; \1/g; s/$/ |/"; }
echo "| image | ImageMagick | size / planes: mean abs. difference (8-bit steps), PSNR, max |"; echo "|---|---|---|"
for c in "-resize 150%:--resize 150%" "-resize 50%:--resize 50%" "-resize 300x300^:--resize 300x300^" "-resize 100x100!:--resize 100x100!" \
         "-colorspace Gray:--colorspace Gray" "-grayscale Rec601Luma:--grayscale Rec601Luma" "-grayscale Average:--grayscale Average" \
         "-grayscale Brightness:--grayscale Brightness" "-grayscale Lightness:--grayscale Lightness" "-grayscale RMS:--grayscale RMS" \
         "-level 10%,90%:--level 10%,90%" "-level 20%,80%,0.5:--level 20%,80%,0.5" "-gamma 0.7:--gamma 0.7" "-gamma 2:--gamma 2" \
         "-negate:--negate" "-auto-level:--auto-level" "-level 10%,90% -gamma 1.5 -negate:--level 10%,90% --gamma 1.5 --negate"; do
  row "211x149 PNG" "${c%%:*}" "$G" "${c%%:*}" "${c#*:}"
done
for c in "-resize 150%:--resize 150%" "-resize x2960:--resize x2960" "-colorspace Gray:--colorspace Gray" "-level 10%,90%:--level 10%,90%" \
         "-gamma 2:--gamma 2" "-gamma 0.7:--gamma 0.7" "-auto-level:--auto-level"; do
  row "1050x1500 page PNG" "${c%%:*}" "$P" "${c%%:*}" "${c#*:}"
done
echo; echo "16-bit PNG, 10-bit planes (10-bit steps):"; echo
for c in "-gamma 2:--gamma 2" "-gamma 0.6:--gamma 0.6" "-level 10%,90%:--level 10%,90%" "-level 10%,90%,0.7:--level 10%,90%,0.7" \
         "-auto-level:--auto-level" "-level 5%,95% -gamma 1.2 -negate:--level 5%,95% --gamma 1.2 --negate"; do
  printf "%-44s " "${c%%:*}"; python3 $D/match16.py $W $B "${c%%:*}" "${c#*:}" $P16
done
