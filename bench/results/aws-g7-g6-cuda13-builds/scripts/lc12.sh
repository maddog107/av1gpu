#!/bin/bash
# lc12.sh: the -cuda12 tarball alone (before the CUDA 13 one is built): sha256 lists of its size range (default,
# --yuv 444 --depth 10) and Kodak outputs, for the Windows comparison.  Prints LC12_DONE.
mkdir -p ~/rel/c12 ~/res
(cd ~ && sha256sum -c av1gpu-*-linux-x86_64-cuda12.tar.gz.sha256) && tar -xzf ~/av1gpu-*-linux-x86_64-cuda12.tar.gz -C ~/rel/c12
A=$(ls ~/rel/c12/*/av1gpu)
for s in sr sr444 kodak; do
  case $s in sr) src=~/images; o=();; sr444) src=~/images; o=(--yuv 444 --depth 10);; kodak) src=~/kodak; o=();; esac
  rm -rf ~/o12/$s; $A --batch $src --outdir ~/o12/$s "${o[@]}" > ~/o12-$s.log 2>&1; echo "$s rc=$? $(ls ~/o12/$s | wc -l) files"
  (cd ~/o12/$s && sha256sum * | awk '{print $2, $1}') > ~/res/lin_c12_$s.sha256
done
echo LC12_DONE
