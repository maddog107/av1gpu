# resusage.sh: ptxas resource usage (registers, stack, spills, shared) of the four k_recon instantiations, main
# (~/old) and perf/recon (~/new): the build's own nvcc command plus --resource-usage, into a scratch object.
# ~/results/resusage.txt
O=~/results/resusage.txt; : > $O
for v in old new; do
  for f in k_recon.cu k_recon_444_8.cu k_recon_420_10.cu k_recon_444_10.cu; do
    cmd=$(ninja -C ~/$v/build -t commands av1gpu_recon | grep -F "encoder/src/$f " | grep -v ccache | tail -1)
    [ -n "$cmd" ] || cmd=$(ninja -C ~/$v/build -t commands av1gpu_recon | grep -F "encoder/src/$f " | tail -1 | sed 's/^[^ ]*ccache //')
    cmd=$(echo "$cmd" | sed "s# -o [^ ]*\.o# -o /tmp/ru_${v}_$f.o#")
    ( cd ~/$v/build && eval "$cmd --resource-usage" > /tmp/ru_${v}_$f.full 2>&1; grep -A3 "_Z7k_recon" /tmp/ru_${v}_$f.full ) > /tmp/ru_${v}_$f.txt 2>&1 &
  done
  wait   # 4 k_recon compiles at once (about 18 GB each)
done
for v in old new; do for f in k_recon.cu k_recon_444_8.cu k_recon_420_10.cu k_recon_444_10.cu; do
  echo "== $v $f"; cat /tmp/ru_${v}_$f.txt; done; done >> $O
echo RESUSAGE_DONE >> $O
