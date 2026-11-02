#!/bin/bash
# extra.sh: the CPU JPEG decode route (GP_INGEST_DECODER=1) and WebP inputs with crop / resize (decoder gate), and
# byte identity on that route without the options.  ~/results/extra.txt
B=~/cr/build/av1gpu; A=~/main/build/av1gpu; G=~/cr/tests/decoder_gate.py
R=~/results/extra.txt; : > $R
{
GP_INGEST_DECODER=1 python3 $G --bin $B --jobs 16 --images ~/sets/gate30 --q 120 -- --crop 65536x65536+1+1 --max-size 1201x997 > ~/results/gate_cpu_route.log 2>&1
echo "gate GP_INGEST_DECODER=1 photos30 crop + resize: rc $?"; tail -n 1 ~/results/gate_cpu_route.log
GP_INGEST_DECODER=1 python3 $G --bin $B --jobs 16 --images ~/sets/gate30 --q 120 --expect-format 444 --expect-depth 10 -- --yuv 444 --depth 10 --max-height 1000 > ~/results/gate_cpu_route_444.log 2>&1
echo "gate GP_INGEST_DECODER=1 photos30 4:4:4 10-bit --max-height 1000: rc $?"; tail -n 1 ~/results/gate_cpu_route_444.log
python3 $G --bin $B --jobs 16 --images ~/sets/webp --q 120 --expect-alpha -- --crop 65536x65536+1+1 --max-size 300x300 > ~/results/gate_webp.log 2>&1
echo "gate webp crop + resize: rc $?"; tail -n 1 ~/results/gate_webp.log
python3 $G --bin $B --jobs 16 --images ~/ll/kodak --q 120 --expect-format 444 -- --yuv 444 --crop 700x500+33+5 > ~/results/gate_kodak_444_crop.log 2>&1
echo "gate kodak 4:4:4 crop only: rc $?"; tail -n 1 ~/results/gate_kodak_444_crop.log
GP_INGEST_DECODER=1 bash ~/cmpset.sh sr_default_cpu_route ~/images $A $B
GP_INGEST_DECODER=1 bash ~/cmpset.sh sr_444_10_cpu_route ~/images $A $B --yuv 444 --depth 10
bash ~/cmpset.sh std_slow ~/images_std $A $B --preset slow
bash ~/cmpset.sh webp_lossless ~/sets/webp $A $B --lossless
} >> $R 2>&1
echo EXTRA_DONE >> $R
