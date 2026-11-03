# q1.sh: quick A/B of ~/new (as c1) against main on the size range; keeps a copy of the binary as ~/bins/c1
mkdir -p ~/bins; cp ~/new/build/av1gpu ~/bins/${TAG:-c1}
nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,temperature.gpu,clocks_throttle_reasons.active --format=csv
nvidia-smi --query-compute-apps=pid,name --format=csv
TAG=q_${TAG:-c1} SETS=${SETS:-sr} python3 ~/r/spd8.py main=$HOME/old/build/av1gpu ${TAG:-c1}=$HOME/bins/${TAG:-c1}
