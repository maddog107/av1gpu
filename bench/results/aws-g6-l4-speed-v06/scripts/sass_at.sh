#!/bin/bash
# sass_at.sh TREE KERNEL_SUBSTR OFFSET...: the SASS (with line info) of the k_recon kernel instance whose mangled
# name contains KERNEL_SUBSTR in ~/TREE/build/av1gpu, around each OFFSET (hex, as racecheck prints it);
# ~/results/sass_TREE.txt
T=$1; K=$2; shift 2
B=~/$T/build/av1gpu; W=~/work/sass_$T; rm -rf $W; mkdir -p $W; cd $W
/usr/local/cuda/bin/cuobjdump -lelf $B > elfs.txt
/usr/local/cuda/bin/cuobjdump -xelf all $B > /dev/null 2>&1
O=~/results/sass_$T.txt; : > $O
for e in *.cubin; do
  names=$(/usr/local/cuda/bin/cuobjdump -elf $e 2>/dev/null | grep -oE '\.text\.[A-Za-z0-9_]+' | sed 's/^\.text\.//' | grep "$K" | sort -u)
  for n in $names; do
    echo "== $e $n" >> $O
    /usr/local/cuda/bin/nvdisasm -g -c -fun $n $e > $W/$n.sass 2>&1
    for off in "$@"; do
      a=$(printf '%04x' $((16#${off#0x})))
      echo "-- offset $off" >> $O
      grep -n -i "/\*0*$a\*/" $W/$n.sass | head -1 | cut -d: -f1 | while read L; do
        sed -n "$((L - 40)),$((L + 12))p" $W/$n.sass >> $O; done
    done
  done
done
echo SASS_DONE >> $O
