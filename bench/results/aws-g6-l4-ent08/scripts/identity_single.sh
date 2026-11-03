#!/bin/bash
# identity_single.sh A B TAG: the single-file part of identity8.sh (10 files one at a time, 3 option sets), cmp of
# every output.  Used for br2 (commit 2 changes only run_batch, which only single-file mode calls; batch mode codes
# the device records with run_batch_device, so br2's batch outputs are br1's).  ~/results/identity_single_TAG.txt
A=$1; B=$2; R=~/results/identity_single_$3.txt; : > $R
{
echo "A=$A B=$B"
for opts in "" "--yuv 444 --depth 10" "--preset slow"; do
  O=~/work3/single; rm -rf $O; mkdir -p $O/a $O/b; n=0; d=0; fa=0
  for f in ~/sets/single/*; do
    b=$(basename $f); n=$((n + 1))
    $A $f $O/a/$b.avif $opts > $O/a.log 2>&1 || fa=$((fa + 1))
    $B $f $O/b/$b.avif $opts > $O/b.log 2>&1 || fa=$((fa + 1))
    cmp -s $O/a/$b.avif $O/b/$b.avif || { d=$((d + 1)); echo "DIFF single $b [$opts]"; }
  done
  echo "single_file: inputs $n, files $(ls $O/a | wc -l) / $(ls $O/b | wc -l), differ $d, failed runs $fa, digest $(cd $O/a && sha256sum * | sha256sum | cut -c1-16), opts [$opts]"
done
} >> $R 2>&1
echo IDENTITY_DONE >> $R
