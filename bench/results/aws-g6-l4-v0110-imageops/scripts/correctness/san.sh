#!/bin/bash
# compute-sanitizer on runs that use the new options; same runs without them on main for the pre-existing baseline
CS=/usr/local/cuda/bin/compute-sanitizer; PR=~/pr/build/av1gpu; M=~/main/build/av1gpu; R=~/results/san; mkdir -p $R
D=/dev/shm/san; rm -rf $D; mkdir -p $D/in
cp ~/pr/tests/gate_images/* $D/in/ ; cp ~/adv/trim/bar_mid_200x120.png ~/adv/trim/corners_1_65x33.png ~/adv/trim/pix_c_17x33.png ~/adv/trim/hgrad_257x65.png ~/adv/flat/white_64x64.png ~/adv/flat/white_1x1.png ~/adv/alpha/hidden_33x17.png ~/adv/alpha/agrad_64x64.png ~/adv/alpha/rgba16_33x17.png ~/adv/deep/g16_65x33.png ~/adv/jpeg/ex6_420_64x48.jpg ~/adv/jpeg/ex3_444_48x64.jpg ~/adv/jpeg/gray_64x48.jpg ~/adv/jpeg/s440_64x48.jpg ~/adv/jpeg/s411_64x48.jpg ~/adv/jpeg/white_64x64.jpg ~/adv/webp/rgba_64x64.webp ~/adv/webp/lossy_65x33.webp $D/in/
ls $D/in | wc -l
run() { # tool tag bin args...
  tool=$1; tag=$2; bin=$3; shift 3
  rm -rf $D/o; mkdir -p $D/o
  s=$(date +%s)
  timeout 1500 $CS --tool $tool --error-exitcode 99 --print-limit 10 $bin "$@" > $R/$tool.$tag.txt 2>&1
  rc=$?
  echo "$tool $tag rc=$rc $(( $(date +%s) - s ))s files=$(ls $D/o | wc -l): $(grep -E 'ERROR SUMMARY|RACECHECK SUMMARY|LEAK SUMMARY' $R/$tool.$tag.txt | tail -2 | tr '\n' ' ')"
}
OPTS=(
 "t1|--trim --fuzz 10% --trim-margin 1% --trim-limits 1%,40% --trim-lightness 50 --skip-blank 99"
 "t2|--yuv 444 --depth 10 --trim --fuzz 5% --level 5%,95%,1.2 --negate --resize 150%"
 "t3|--auto-level --grayscale Lightness --resize x200"
 "t4|--lossless --trim --auto-level --resize 50%"
 "t5|--lossless --ycgco-re --gamma 1.4 --colorspace Gray"
 "t6|--resize 100x100^ --gamma 0.6 --negate --grayscale RMS --depth 10"
 "t7|--trim --resize 300%x50% --auto-level"
 "t8|--yuv 444 --trim --fuzz 30% --skip-blank 50 --grayscale Average --resize @5000"
)
for tool in ${TOOLS:-memcheck racecheck synccheck initcheck}; do
  for o in "${OPTS[@]}"; do
    tag=${o%%|*}; args=${o#*|}
    run $tool $tag $PR --batch $D/in --outdir $D/o --small-preset none --preset turbo3 $args
  done
  for s in rgba hidden; do :; done
  run $tool s1 $PR $D/in/hidden_33x17.png $D/o/s.avif --trim --level 10%,90% --resize 77x77^
  run $tool s2 $PR $D/in/ex6_420_64x48.jpg $D/o/s.avif --depth 10 --yuv 444 --trim --fuzz 10% --auto-level --grayscale RMS --resize @30000
  run $tool s3 $PR $D/in/white_1x1.png $D/o/s.avif --trim --resize 200x200! --negate
  # baseline, same lists, no new options
  run $tool main_b1 $M --batch $D/in --outdir $D/o --small-preset none --preset turbo3
  run $tool main_b2 $M --batch $D/in --outdir $D/o --small-preset none --preset turbo3 --yuv 444 --depth 10 --max-size 100x100
  run $tool main_b3 $M --batch $D/in --outdir $D/o --lossless
  run $tool pr_b1 $PR --batch $D/in --outdir $D/o --small-preset none --preset turbo3
done
echo SAN_DONE
