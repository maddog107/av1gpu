#!/bin/bash
# s6r1 (stock libaom, speed 6, -a enable-restoration=1) against s6 (enable-restoration=0): cmp every file
cd ~/lr/out/420/s6
n=0; same=0
for f in */*.avif; do n=$((n + 1)); cmp -s "$f" ../s6r1/"$f" && same=$((same + 1)); done
echo "s6r1 vs s6 (420): $same/$n identical"
