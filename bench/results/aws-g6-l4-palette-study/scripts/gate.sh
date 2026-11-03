#!/bin/bash
# decoder gate (dav1d + aomdec) of the branch on palette-coded output: screen + flat sets, lossy 4:2:0 / 4:4:4 / 10-bit,
# lossless GBR / YCgCo-R.  usage: gate.sh [what...]   (default: all)
export AV1G_AOMDEC=${AV1G_AOMDEC:-~/src/aom-build/aomdec}
BR=~/br/build/av1gpu
G=~/br/tests/decoder_gate.py
O=~/pal/gate; mkdir -p $O
mkimgs() {   # dir with the screen set + the flat PNGs (symlinks)
  local d=$O/imgs_$1; rm -rf $d; mkdir -p $d
  for s in $2; do for f in ~/pal/sets/$s/*; do ln -s $f $d/; done; done
  echo $d
}
SCR=$(mkimgs scr "screen flat_png")
SCRJ=$(mkimgs jpg "flat_jpg painted")
g() {   # name images args...
  local n=$1 d=$2; shift 2
  python3 $G --bin $BR --images $d --jobs 8 "$@" > $O/$n.log 2>&1
  echo "GATE $n exit $?: $(grep -a -E '[0-9]+/[0-9]+' $O/$n.log | tail -1)"
}
W=${*:-all}
case " $W " in *" all "*|*" l420 "*) g l420 $SCR --q 120,60,20 --expect-format 420 ;; esac
case " $W " in *" all "*|*" l444 "*) g l444 $SCR --q 120,40 --expect-format 444 -- --yuv 444 ;; esac
case " $W " in *" all "*|*" l10 "*) g l10 $SCR --q 120,40 --expect-depth 10 -- --depth 10 ;; esac
case " $W " in *" all "*|*" l444_10 "*) g l444_10 $SCR --q 80 --expect-format 444 --expect-depth 10 -- --yuv 444 --depth 10 ;; esac
case " $W " in *" all "*|*" ljpg "*) g ljpg $SCRJ --q 120,60 ;; esac
case " $W " in *" all "*|*" ll "*) g ll $SCR --lossless ;; esac
case " $W " in *" all "*|*" llyc "*) g llyc $SCR --lossless -- --ycgco-re ;; esac
