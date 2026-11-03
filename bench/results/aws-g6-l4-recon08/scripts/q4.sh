# q4.sh: wait for bnew.sh, then the quick A/B of c4 (c1 + compile-time fwd loops) against main and c1
until grep -q BNEW_DONE ~/r/bnew.sh.log; do sleep 3; done
mkdir -p ~/bins; cp ~/new/build/av1gpu ~/bins/c4
TAG=q_c4 SETS=sr python3 ~/r/spd8.py main=$HOME/old/build/av1gpu c1=$HOME/bins/c1 c4=$HOME/bins/c4
TAG=q_c4 MINMP=250 SETS=std python3 ~/r/spd8.py main=$HOME/old/build/av1gpu c1=$HOME/bins/c1 c4=$HOME/bins/c4
echo Q4_DONE
