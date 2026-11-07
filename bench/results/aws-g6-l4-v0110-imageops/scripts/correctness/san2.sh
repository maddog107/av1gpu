#!/bin/bash
# racecheck / initcheck reports grouped by kernel: new options on the PR, baseline on main
CS=/usr/local/cuda/bin/compute-sanitizer; PR=~/pr/build/av1gpu; M=~/main/build/av1gpu; R=~/results/san2; mkdir -p $R; D=/dev/shm/san
go() { tool=$1; tag=$2; bin=$3; shift 3; rm -rf $D/o; mkdir -p $D/o
  timeout 1800 $CS --tool $tool --print-limit 1000000 --error-exitcode 99 $bin "$@" > $R/$tool.$tag.txt 2>&1; echo "$tool $tag rc=$?"; 
  grep -oE '(k_[a-z0-9_]+|win_group|gpu_[a-z_]+)[<(]' $R/$tool.$tag.txt | sort | uniq -c | sort -rn | head -12 > $R/$tool.$tag.kern; }
for tool in racecheck initcheck; do
 go $tool main1 $M --batch $D/in --outdir $D/o --small-preset none --preset turbo3 --yuv 444 --depth 10
 go $tool main2 $M --batch $D/in --outdir $D/o --small-preset none --preset turbo3
 go $tool pr1 $PR --batch $D/in --outdir $D/o --small-preset none --preset turbo3 --yuv 444 --depth 10 --trim --fuzz 5% --level 5%,95%,1.2 --negate --resize 150%
 go $tool pr2 $PR --batch $D/in --outdir $D/o --small-preset none --preset turbo3 --trim --auto-level --grayscale Lightness --resize x200
 go $tool pr3 $PR --batch $D/in --outdir $D/o --lossless --trim --auto-level --gamma 1.3 --grayscale RMS
 go $tool pr4 $PR --batch $D/in --outdir $D/o --lossless --ycgco-re --trim --fuzz 10% --negate --resize 50%
done
for f in $R/*.kern; do echo "--- $f"; cat $f; done
grep -lE 'k_trim_measure|k_tone|k_gray' $R/*.txt
echo; grep -hE 'k_trim_measure|k_tone|k_gray' $R/*.txt | cut -c1-240 | sort | uniq -c | sort -rn | head -20
echo SAN2_DONE
