#!/bin/bash
# sol.sh: speed-of-light experiment (not byte-identical, exp_patch.py build): main vs the experiment binary with
# k_select_small replaced by memsets (AV1X_SKIP_SEL) or k_entropy_win not coding (AV1X_SKIP_ENT); 2 runs each, the sets
# of spd8.py (>= 1,000 MP).  ~/results/sol.txt
cd ~
python3 - <<'EOF'
import glob, os, sys
sys.path.insert(0, os.path.expanduser("~/main/bench"))
import bench
from PIL import Image
H = os.path.expanduser("~")
for name, pat in (("sr", "images/*.jpg"), ("std", "images_std/*.jpg"), ("thumbs", "sets/thumbs/*.jpg")):
    files = sorted(glob.glob(f"{H}/{pat}"))
    sizes = {os.path.basename(f): Image.open(f).size[0] * Image.open(f).size[1] / 1e6 for f in files}
    bench.speed_set(files, sizes, 1000, f"{H}/work/{name}")
EOF
R=~/results/sol.txt
for s in sr std thumbs; do
  for rnd in 1 2; do
    for cfg in main exp exp_sel exp_ent; do
      case $cfg in main) B=~/main/build/av1gpu; EV=;; exp) B=~/exp/build/av1gpu; EV=;; exp_sel) B=~/exp/build/av1gpu; EV=AV1X_SKIP_SEL=1;; exp_ent) B=~/exp/build/av1gpu; EV=AV1X_SKIP_ENT=1;; esac
      rm -rf ~/work/o_sol
      L=$(env $EV $B --batch ~/work/$s/speedset --outdir ~/work/o_sol --stats 2>/dev/null | tail -1)
      echo "$s $rnd $cfg $(echo $L | python3 -c 'import json,sys; d=json.loads(sys.stdin.read()); print(d["mp_s"], d["img_s"], d["entropy_wall_ms"], d["postfilter_ms"])')" >> $R
    done
  done
done
echo SOL_DONE >> $R
