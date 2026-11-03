#!/bin/bash
# profiling step 2: ncu details of k_analysis<16,8> (standard set: base preset), k_analysis_small<8,8>, and
# k_analysis<64,64> (size range: turbo), one launch each
bash ~/ncu_detail.sh ~/new/build/av1gpu std_168 ~/images_std "k_analysis<.int.16, .int.8," 1
bash ~/ncu_detail.sh ~/new/build/av1gpu sr_6464 ~/images "k_analysis<.int.64, .int.64," 1
bash ~/ncu_detail.sh ~/new/build/av1gpu sr_1616 ~/images "k_analysis<.int.16, .int.16," 1
bash ~/ncu_detail.sh ~/new/build/av1gpu std_small ~/images_std "k_analysis_small<.int.8, .int.8," 1
echo STEP2_DONE > ~/results/prof/step2_done
