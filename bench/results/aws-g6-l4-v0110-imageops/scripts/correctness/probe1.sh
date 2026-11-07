#!/bin/bash
B=~/pr/build/av1gpu; M=~/main/build/av1gpu; T=/dev/shm/p; mkdir -p $T; P=~/adv/trim/bar_mid_200x120.png; J=~/adv/jpeg/ex0_420_160x96.jpg
try() { # label, cmd...
  rm -f $T/o.avif; s=$(date +%s.%N); timeout 120 "$@" > $T/out.txt 2> $T/err.txt; rc=$?; e=$(date +%s.%N)
  sz=""; if [ -f $T/o.avif ]; then sz=$(python3 - <<PY
import sys; sys.path.insert(0,'/home/ubuntu/t'); import avt
try:
    c=avt.check('$T/o.avif'); print(c['si'][:5], 'ok')
except Exception as ex: print('BAD', ex)
PY
); fi
  printf 'rc=%s %.1fs [%s] %s | %s\n' $rc $(echo "$e-$s"|bc) "$sz" "${*:2}" "$(tail -1 $T/err.txt | cut -c1-150)"
}
echo "== extreme geometries"
for g in 0.5 0.4 0.4x0.4! 0.4x0.4 0.6x0.6! 0.4x100! 100x0.4! x0.4 0.4% 0.001% 0.4%x0.4% @0.4 @0.001 1048576 x1048576 1048576x1048576 1048576x1048576! 1048576x1 1x1048576 1048576x1! 1x1048576! 1048576x1^ 100000% 1048576% 1048577 1048577x1 @1048576 @1 0 0x0 0x100 100x0 -1 abc 100x100+5+5 100x100x100 "" 'x' '%' '!' '50%%' '50%x25%x10%' '^' '100x100^!<>' ; do
  try $B $P $T/o.avif --resize "$g"
done
for g in 1048576x1^ 1x1048576^ 100000% 1048576%; do try $B ~/adv/tall/t_1x4097.png $T/o.avif --resize "$g"; done
try $B ~/adv/jpeg/ex0_420_64x48.jpg $T/o.avif --resize 1048576%
try $B ~/adv/jpeg/ex0_420_64x48.jpg $T/o.avif --resize 8000x8000!
try $B ~/adv/jpeg/ex0_420_64x48.jpg $T/o.avif --resize 8000x8000! --max-mp 100
try $B ~/adv/jpeg/ex0_420_64x48.jpg $T/o.avif --resize 20000x20000!
echo "== numbers in options"
for v in "--gamma 0" "--gamma -1" "--gamma nan" "--gamma inf" "--gamma 1e-30" "--gamma 1e-5" "--gamma 101" "--gamma 100" "--level 50%,50%" "--level 60%,40%" "--level 0%,100%,0" "--level 0%,100%,nan" "--level 0%,100%,1e-9" "--level 0%,100%,1e9" "--level 0.001%,0.002%" "--level 70000,80000" "--level 0,1" "--level 5" "--level ,50%" "--level 10%x90%" "--trim --fuzz nan" "--trim --fuzz 1e3" "--trim --fuzz 65536" "--trim --fuzz -1" "--trim --trim-margin 1048576" "--trim --trim-margin 1048577" "--trim --trim-margin 100%" "--trim --trim-margin 101%" "--trim --trim-limits 0,0" "--trim --trim-limits 100,100" "--trim --trim-limits 50%" "--trim --trim-lightness 0" "--trim --trim-lightness 100.0" "--trim --trim-lightness 100.1" "--trim --skip-blank 100" "--trim --skip-blank 100.1" "--trim --skip-blank %" "--grayscale foo" "--grayscale rec709luma" "--colorspace gray" "--colorspace GRAY" "--colorspace sRGB"; do
  try $B $P $T/o.avif $v
done
