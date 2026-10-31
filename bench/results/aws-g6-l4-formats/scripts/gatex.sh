#!/bin/bash
# gatex.sh: extra decoder gates on the fixed build (CPU JPEG route, CPU entropy, presets on the gamut set)
export AV1G_DAV1D=dav1d AV1G_AOMDEC=aomdec TMPDIR=$HOME/tmp
B=$HOME/build-fix; G="python3 $HOME/br/tests/decoder_gate.py"
gate() {   # label images expect-format expect-depth q [-- options]
  local l=$1 im=$2 ef=$3 ed=$4 q=$5; shift 5
  echo "-- $l ($(basename $im), q$q)"
  local t0=$(date +%s)
  $G --bin $B/av1gpu --images $im --q $q --jobs 14 --expect-format $ef --expect-depth $ed "$@" 2>&1 | grep "^\[gate\]"
  echo "   ($(( $(date +%s) - t0 )) s)"
}
{
for f in "420 8" "444 8" "420 10" "444 10"; do
  set -- $f; x=(); [ $1 = 444 ] && x+=(--yuv 444); [ $2 = 10 ] && x+=(--depth 10)
  GP_INGEST_DECODER=1 gate "CPU JPEG route $2-bit $1" ~/images $1 $2 120 -- "${x[@]}"
  gate "CPU entropy $2-bit $1" ~/images $1 $2 120 -- "${x[@]}" --cpu-entropy
  gate "turbo3 $2-bit $1" ~/images_gamut $1 $2 120 -- "${x[@]}" --preset turbo3
  gate "slow small-preset none $2-bit $1" ~/images_gamut $1 $2 120 -- "${x[@]}" --preset slow --small-preset none
done
} > ~/results/gatex.txt 2>&1
echo "PHASE_DONE gatex $(date -u +%H:%M:%S)" >> ~/results/phases.txt
