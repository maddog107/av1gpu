#!/bin/bash
# fix_llsets.sh: setup.sh ran mk_ll_sets.py before ~/br existed (it reads ~/br/docs/charts and ~/br/av1core), so only
# ll/kodak was made.  Build the other lossless sets into ~/ll_new (kodak, in use by the identity run, untouched),
# move them into ~/ll, and add what mk_sets.py takes from them (sets/gray, sets/webp) (mk_mono.sh and the identity reruns: rerun8.sh, after the identity run).  ~/ll_new stays (ll/cmp links into it).
cd ~
sed -e 's#R = f"{H}/ll"#R = f"{H}/ll_new"#' ~/mk_ll_sets.py > ~/mk_ll_sets_new.py
grep -n 'll_new' ~/mk_ll_sets_new.py | head -2
python3 ~/mk_ll_sets_new.py > ~/results/setup/llsets_new.log 2>&1; echo "mk_ll_sets exit $?" >> ~/results/setup/llsets_new.log
for d in cmp graphics gray jpeg misc png16_10 png16_8 png16_full webp; do rm -rf ~/ll/$d; mv ~/ll_new/$d ~/ll/$d; done
for f in ~/ll/gray/*.png; do ln -sf $f ~/sets/gray/; done   # mk_sets.py links them
for f in ~/ll/webp/*.webp; do ln -sf $f ~/sets/webp/; done
for d in ~/ll/*/ ~/sets/gray ~/sets/webp; do echo "$d $(ls $d | wc -l)"; done
echo FIX_DONE
