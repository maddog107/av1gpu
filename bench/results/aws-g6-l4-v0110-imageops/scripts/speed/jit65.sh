#!/bin/bash
# PTX JIT at start-up on a GPU without a matching cubin, measured on the L4 with the sm_89 PTX the dev build embeds: every module of the
# PR build, compute cache off, eager loading (batch mode), 2 passes.  k_imageops' share is what the PR adds to such a start.
L=~/results/mods; cd $L/x_pr89
ls -l *.ptx | awk '{print $5, $NF}' > ../ptx_sizes.txt
CUDA_CACHE_DISABLE=1 CUDA_MODULE_LOADING=EAGER ~/modload k_imageops.sm_89.ptx k_imageops.sm_89.ptx > ../jit_imageops.txt 2>&1
CUDA_CACHE_DISABLE=1 CUDA_MODULE_LOADING=EAGER timeout 1800 ~/modload $(ls *.ptx) > ../jit_all.txt 2>&1
echo JIT_DONE >> ../jit_all.txt
