#!/bin/bash
# Options on: every config against plain, same binary, interleaved rounds.  plain2 is a second plain spec: the noise.
# "never" measures every frame and cuts none (--trim-limits 100%,100%): its output must equal plain's, so it isolates
# the measurement and the wait from any saving in pixels coded.
B=${1:-$HOME/pr/build/av1gpu}; TAG=${2:-on}; R=${3:-8}; shift 3
SETS=${@:-comics c444 sr std thumbs pngwebp}
CONV="--trim --fuzz 10% --trim-margin 1% --trim-limits 1%,20% --trim-lightness 80 --skip-blank 99"
SP="plain=$B;plain2=$B"
SP="$SP;never=$B|--trim --fuzz 10% --trim-limits 100%,100%"
SP="$SP;trim0=$B|--trim --fuzz 0%"
SP="$SP;trim10=$B|--trim --fuzz 10%"
SP="$SP;conv=$B|$CONV"
SP="$SP;level=$B|--level 5%,95%,1.1"
SP="$SP;levelnog=$B|--level 5%,95%"
SP="$SP;gamma=$B|--gamma 1.2"
SP="$SP;negate=$B|--negate"
SP="$SP;auto=$B|--auto-level"
SP="$SP;gray709=$B|--grayscale Rec709Luma"
SP="$SP;gray601=$B|--grayscale Rec601Luma"
SP="$SP;rs50=$B|--resize 50%"
SP="$SP;rs2960=$B|--resize x2960"
python3 ~/spd65.py $TAG $R "$SP" -- $SETS
