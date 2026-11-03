#!/bin/bash
# profiling step 1: ncu details of the small kernels on the standard set (background), ptxas -v of the analysis files
nohup bash ~/ncu_detail.sh ~/new/build/av1gpu std_small ~/images_std "k_analysis_small" 6 > /dev/null 2>&1 < /dev/null &
bash ~/ptxv.sh new k_an_a.cu k_an_b.cu k_an_c.cu k_an_d.cu
cat ~/results/prof/ptxv_new_k_an_*.txt | grep -v ", 10>" | cut -c1-200
