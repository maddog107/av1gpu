#!/bin/bash
# byte identity: main vs branch, without the new options
S=$1; A=$HOME/av1gpu/build/av1gpu; B=$HOME/av1gpu/.claude/worktrees/feat-trim/build/av1gpu
tot=0; bad=0
run() {  # name dir opts...
  local name=$1 dir=$2; shift 2
  rm -rf $S/ident/oa $S/ident/ob
  $A --batch $dir --outdir $S/ident/oa "$@" >/dev/null 2>&1; $B --batch $dir --outdir $S/ident/ob "$@" >/dev/null 2>&1
  local n=0 d=0
  for f in $S/ident/oa/*.avif; do n=$((n+1)); cmp -s "$f" "$S/ident/ob/$(basename "$f")" || d=$((d+1)); done
  local nb=$(ls $S/ident/ob/*.avif 2>/dev/null | wc -l)
  [ $nb -ne $n ] && d=$((d+1))
  printf "%-48s %4d files, %d differ\n" "$name" $n $d; tot=$((tot+n)); bad=$((bad+d))
}
run "pages default" $S/ident/jpg
run "pages --yuv 444 --depth 10" $S/ident/jpg --yuv 444 --depth 10
run "pages 444 10-bit turbo3 --max-height 2960" $S/ident/jpg --yuv 444 --depth 10 --preset turbo3 --max-height 2960
run "pages --crop 1000x1500+77+33" $S/ident/jpg --crop 1000x1500+77+33
run "photos default" $S/ident/photo
run "photos --preset fast" $S/ident/photo --preset fast
GP_INGEST_DECODER=1 run "photos via CPU route (GP_INGEST_DECODER=1)" $S/ident/photo
run "PNG/WebP default" $S/ident/png
run "PNG/WebP --yuv 444 --depth 10" $S/ident/png --yuv 444 --depth 10
run "PNG/WebP --lossless" $S/ident/png --lossless
run "PNG/WebP --lossless --ycgco-re" $S/ident/png --lossless --ycgco-re
run "PNG/WebP --max-size 97x97" $S/ident/png --max-size 97x97
# single-image mode
n=0; d=0
for f in $S/ident/png/*.png $(ls $S/ident/jpg/*.jpg | head -3) $(ls $S/ident/photo/*.jpg | head -2); do
  for o in "" "--yuv 444 --depth 10"; do n=$((n+1)); $A "$f" $S/ident/sa.avif $o >/dev/null 2>&1; $B "$f" $S/ident/sb.avif $o >/dev/null 2>&1; cmp -s $S/ident/sa.avif $S/ident/sb.avif || d=$((d+1)); done
done
printf "%-48s %4d files, %d differ\n" "single-image mode" $n $d; tot=$((tot+n)); bad=$((bad+d))
echo "TOTAL $tot files, $bad differ"
