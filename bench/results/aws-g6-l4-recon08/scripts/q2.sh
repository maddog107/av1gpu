# q2.sh: wait for bnew.sh, then the quick A/B of TAG against main and c1 (size range)
TAG=${TAG:-c3}
until grep -q BNEW_DONE ~/r/bnew.sh.log; do sleep 3; done
mkdir -p ~/bins; cp ~/new/build/av1gpu ~/bins/$TAG
TAG=q_$TAG SETS=${SETS:-sr} python3 ~/r/spd8.py main=$HOME/old/build/av1gpu c1=$HOME/bins/c1 $TAG=$HOME/bins/$TAG
