# bnew.sh: copy ~/r/src/* (encoder/src files) into ~/new, rebuild av1gpu (+ tests when ALL=1); quick identity on
# the size range (default) and thumbnails against ~/old
set -e
cp ~/r/src/* ~/new/encoder/src/
echo "${TAG:-wip}" > ~/new/COMMIT
s=$(date +%s)
if [ "$ALL" = 1 ]; then cmake --build ~/new/build -j 24 > ~/r/bnew_build.log 2>&1
else cmake --build ~/new/build -j 24 --target av1gpu > ~/r/bnew_build.log 2>&1; fi
echo "built after $(( $(date +%s) - s )) s"
/usr/local/cuda/bin/cuobjdump --dump-resource-usage ~/new/build/av1gpu 2>/dev/null | grep -A1 "k_recon" | grep -o 'Function [^ ]*\|REG:[0-9]*\|STACK:[0-9]*\|SHARED:[0-9]*\|LOCAL:[0-9]*' | paste - - - - - > ~/r/res_new.txt
/usr/local/cuda/bin/cuobjdump --dump-resource-usage ~/old/build/av1gpu 2>/dev/null | grep -A1 "k_recon" | grep -o 'Function [^ ]*\|REG:[0-9]*\|STACK:[0-9]*\|SHARED:[0-9]*\|LOCAL:[0-9]*' | paste - - - - - > ~/r/res_old.txt
paste ~/r/res_old.txt ~/r/res_new.txt
[ "$QUICK" = 0 ] && exit 0
bash ~/cmpset.sh q_sr_default ~/images ~/old/build/av1gpu ~/new/build/av1gpu | tail -3
bash ~/cmpset.sh q_thumbs ~/sets/thumbs ~/old/build/av1gpu ~/new/build/av1gpu | tail -3
bash ~/cmpset.sh q_std ~/images_std ~/old/build/av1gpu ~/new/build/av1gpu | tail -3
grep -h "stage ms" ~/work/cmp/q_sr_default/*.log ~/work/cmp/q_std/*.log
echo BNEW_DONE
