#!/bin/bash
# The cost of the new stages: one --batch call per configuration over DIR, interleaved, ROUNDS rounds; prints the
# --stats wall time and MP/s (megapixels encoded, i.e. after the trim / resize).
#   speed.sh BIN DIR ROUNDS
B=$1; D=$2; N=${3:-3}; O=$(mktemp -d)
cfgs=("plain|"
      "trim --fuzz 0%|--trim --fuzz 0%"
      "trim as convert-comics asks|--trim --fuzz 10% --trim-margin 1% --trim-limits 1%,20% --trim-lightness 80 --skip-blank 99"
      "level 5%,95%,1.1|--level 5%,95%,1.1"
      "auto-level|--auto-level"
      "grayscale Rec709Luma|--grayscale Rec709Luma"
      "resize x2960|--resize x2960")
for r in $(seq $N); do
  for c in "${cfgs[@]}"; do
    name=${c%%|*}; opts=${c#*|}
    rm -rf $O/out
    s=$($B --batch $D --outdir $O/out --yuv 444 --depth 10 --tiles 2,2,2 --stats --max-mp 100 --max-input-mp 100 $opts 2>/dev/null | tail -1)
    python3 -c "import json,sys; j=json.loads(sys.argv[1]); print('%-32s round %s: %6.0f ms  %6.1f MP/s  %.0f MP' % (sys.argv[2], sys.argv[3], j['wall_ms'], j['mp_s'], j['mp']))" "$s" "$name" $r
  done
done
rm -rf $O
