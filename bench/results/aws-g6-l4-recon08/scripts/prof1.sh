# prof1.sh: baseline profile of main (~/old): stage lines and recon phases on the size range, thumbnails, standard;
# k_recon resource usage.  ~/results/prof1/
B=${B:-~/old/build/av1gpu}
O=~/results/prof1${TAG:+_$TAG}; mkdir -p $O ~/work/p
cd ~
/usr/local/cuda/bin/cuobjdump --dump-resource-usage $B 2>/dev/null | grep -A1 k_recon | grep -o 'Function [^ ]*\|REG:[0-9]*\|STACK:[0-9]*\|SHARED:[0-9]*\|LOCAL:[0-9]*' | paste - - - - - > $O/resources.txt
for set in images sets/thumbs images_std; do
  n=$(basename $set)
  for enc in "" 1; do
    for prof in "" --prof; do
      t0=$(date +%s.%N)
      GP_ENCODERS=$enc $B --batch ~/$set --outdir ~/work/p/$n $prof > $O/$n${enc:+_enc$enc}$prof.log 2>&1
      t1=$(date +%s.%N)
      echo "$n enc=${enc:-default} $prof wall $(echo "$t1 - $t0" | bc)" >> $O/walls.txt
    done
  done
done
grep -h "stage ms\|recon phases\|CTA-cycles\|MP/s" $O/*.log > $O/summary.txt
echo PROF1_DONE
