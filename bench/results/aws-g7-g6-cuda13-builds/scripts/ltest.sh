#!/bin/bash
# ltest.sh: the release workflow's Linux tarballs on this L4: av1gpu-*-linux-x86_64.tar.gz (CUDA 13.4, ~/rel/c13) and
# av1gpu-*-linux-x86_64-cuda12.tar.gz (CUDA 12.9, ~/rel/c12).  cmp of every output (size range default and --yuv 444
# --depth 10, standard set, Kodak); sha256 lists of the CUDA 13 outputs for the Windows comparison; start-up; speed.
# Prints LTEST_DONE.
set -u
mkdir -p ~/rel/c13 ~/rel/c12 ~/res
for t in ~/av1gpu-*-linux-x86_64.tar.gz; do (cd ~ && sha256sum -c $t.sha256) && tar -xzf $t -C ~/rel/c13; done
for t in ~/av1gpu-*-linux-x86_64-cuda12.tar.gz; do (cd ~ && sha256sum -c $t.sha256) && tar -xzf $t -C ~/rel/c12; done
A=$(ls ~/rel/c12/*/av1gpu); B=$(ls ~/rel/c13/*/av1gpu)
echo "c12 $A $(stat -c %s $A)"; echo "c13 $B $(stat -c %s $B)"
W=~/cmp/rel; rm -rf $W; mkdir -p $W
for s in sr sr444 std kodak; do
  case $s in sr) src=~/images; o=();; sr444) src=~/images; o=(--yuv 444 --depth 10);; std) src=~/images_std; o=();;
             kodak) src=~/kodak; o=();; esac
  for x in a b; do
    [ $x = a ] && bin=$A || bin=$B
    $bin --batch $src --outdir $W/$s/$x "${o[@]}" > $W/$s.$x.log 2>&1 || echo "$s $x rc=$?"
  done
  n=$(ls $W/$s/a | wc -l); m=$(ls $W/$s/b | wc -l); d=0
  for f in $W/$s/a/*; do cmp -s $f $W/$s/b/$(basename $f) || d=$((d+1)); done
  echo "cuda12 vs cuda13 $s: $n vs $m files, $d differ ($(du -sk $W/$s/a | cut -f1) kB)" | tee -a ~/res/ident.txt
  (cd $W/$s/b && sha256sum * | awk '{print $2, $1}') > ~/res/lin_c13_$s.sha256
  (cd $W/$s/a && sha256sum * | awk '{print $2, $1}') > ~/res/lin_c12_$s.sha256
done
bash ~/startup.sh $A c12 2 | tee -a ~/res/startup.txt
bash ~/startup.sh $B c13 2 | tee -a ~/res/startup.txt
bash ~/spdab.sh $A $B ~/sr3 2 CUDA_CACHE_DISABLE=1 | tee ~/res/speed.txt | tail -2
echo LTEST_DONE
