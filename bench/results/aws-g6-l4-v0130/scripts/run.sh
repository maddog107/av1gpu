#!/bin/bash
# v0.13.0 docs refresh driver (on the instance, after setup.sh printed SETUP_DONE).  Unattended, one part after another:
#
#   run.sh                 every part not yet done (~/results/done/<part>), in the order below
#   run.sh PART ...        exactly these parts, in the order given (a part that ran before starts afresh;
#                          the md5 score cache is kept, so a re-scored part only scores what is missing)
#   SKIP="cvvdp speed_extra" run.sh     every part not yet done except these
#
# Parts (most quoted numbers first; minutes are estimates for the L4, see REFRESH_MAP.md / the summary):
#   speed        ~13  size range default (GP_ENCODERS 4 and 3), nosmall, turbo3, turbo3 nosmall; 4:4:4 / 10-bit formats
#   quality      ~14  encode every lossy config (+ lossless, RGBA) with NEW and OLD; SSIMULACRA2 of NEW (+ OLD default)
#   speed_small  ~12  thumbnails and Kodak: default, nosmall (+ turbo3 nosmall on Kodak)
#   speed_std    ~12  standard set: default with 4 / 2 / 3 encoders, nosmall with 4 / 3
#   speed_ll      ~4  lossless comparison set: one pass and 4x, with and without --ycgco-re
#   gates         ~6  tests/decoder_gate.py, NEW: default q60/120/180, 4:4:4 10-bit q120, turbo3 nosmall q120
#   startup       ~2  cold single-process wall time, NEW vs OLD (and OLD again: noise), median of 12 (wall9.py)
#   alpha         ~7  RGBA set speed (x20) and alpha rows -- only with ~/data/rgba (else SKIPPED in a second)
#   cvvdp        ~10  CVVDP of NEW default (99 photos x 8 points) and of OLD on 8 control photos
#   quality_old   ~5  SSIMULACRA2 of the other OLD configs (NEW vs OLD BD-rate per config)
#   speed_extra   ~7  size range: default with 2 encoders, nosmall with 3, --tiles 2,1,1 and 2,3,3
# Speed parts wait for an idle GPU (no compute process) and run with nothing else on it; nvidia-smi dmon samples
# power / SM clock for the whole run (~/results/dmon.txt).  One "## part: result, seconds" line per part in
# ~/results/progress.log, the part's output in ~/results/logs/<part>.log; RUN_DONE at the end.
S=$(cd "$(dirname "$0")" && pwd)
R=~/results; mkdir -p $R/logs $R/done
ALL="speed quality speed_small speed_std speed_ll gates startup alpha cvvdp quality_old speed_extra"
PY=~/venv/bin/python
exec 9> /tmp/av1gpu-run13.lock
flock -n 9 || { echo "another run.sh is running"; exit 1; }
[ -x ~/bin/new ] && [ -x ~/bin/old ] && [ -x $PY ] || { echo "run setup.sh first (no ~/bin/new, ~/bin/old or ~/venv)"; exit 1; }
for p in "$@"; do case " $ALL " in *" $p "*) ;; *) echo "unknown part '$p' (parts: $ALL)"; exit 1;; esac; done

log() { echo "$*" | tee -a $R/progress.log; }
gpu_idle() {   # wait (up to 5 min) until no process uses the GPU
  local i n
  for i in $(seq 60); do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -c .)
    [ "$n" = 0 ] && return 0
    sleep 5
  done
  log "WARNING: $n other GPU process(es) still running at $(date -u +%T)"; nvidia-smi
}
clocks() { echo "$1 $(date -u +%FT%T) $(nvidia-smi --query-gpu=clocks.sm,clocks.max.sm,power.draw,power.limit,enforced.power.limit,clocks_throttle_reasons.active,temperature.gpu --format=csv,noheader)" >> $R/clocks.txt; }
speed() { gpu_idle; clocks "start $1"; python3 $S/spd13.py $1; local rc=$?; clocks "end $1"; return $rc; }

part_speed() { speed core; }
part_speed_small() { speed small; }
part_speed_std() { speed std; }
part_speed_ll() { speed ll; }
part_speed_extra() { speed extra; }
part_quality() { $PY $S/qual13.py encode score; }
part_quality_old() { $PY $S/qual13.py score_old; }
# (without the AMI's LD_LIBRARY_PATH: its CUDA 13.2 libraries shadow the venv's nvrtc, and cuDNN then fails to load its
# runtime-compiled engines: "CUDNN_STATUS_SUBLIBRARY_LOADING_FAILED")
part_cvvdp() { env -u LD_LIBRARY_PATH $PY $S/qual13.py cvvdp; }
part_gates() { $PY $S/qual13.py gates; }
part_alpha() {
  if [ -z "$(ls ~/data/rgba/set/*.png 2>/dev/null)" ]; then echo "SKIPPED: no ~/data/rgba/set"; return 3; fi
  speed alpha && $PY $S/qual13.py alpha
}
part_startup() {   # as the v0.9.0 check's spd9.py single: `av1gpu IN OUT` per image, and the production pair as a batch
  gpu_idle
  local o=$R/startup.txt; : > $o
  local W="python3 $S/wall9.py 12 $HOME/bin/new $HOME/bin/old $HOME/bin/old2 --"
  for f in photo.jpg big.jpg thumb512.jpg t320.jpg; do
    { echo "== $f"; $W ~/st/$f /tmp/s{bin}.avif; } >> $o 2>&1 || return 1
  done
  { echo "== pair batch (photo.jpg + thumb512.jpg)"; $W --batch ~/st/pair.txt --outdir /tmp/pb{bin}; } >> $o 2>&1 || return 1
  grep -v '^{' $o
  ! grep -q '^FAIL' $o
}

nvidia-smi dmon -s puc -o DT >> $R/dmon.txt 2>&1 &
DM=$!
trap 'kill $DM 2>/dev/null' EXIT
if [ $# -gt 0 ]; then PARTS="$*"; else PARTS=""; for p in $ALL; do
  [ -e $R/done/$p ] && continue; case " $SKIP " in *" $p "*) continue;; esac; PARTS="$PARTS $p"; done; fi
log "# run.sh $(date -u +%FT%T): parts$PARTS"
for p in $PARTS; do
  t0=$(date +%s)
  echo "=== $p $(date -u +%FT%T)" >> $R/logs/$p.log
  part_$p >> $R/logs/$p.log 2>&1
  rc=$?; dt=$(( $(date +%s) - t0 ))
  if [ $rc = 0 ]; then res=OK; touch $R/done/$p
  elif [ $rc = 3 ]; then res="SKIPPED ($(grep -m1 SKIPPED $R/logs/$p.log | cut -c1-80))"; touch $R/done/$p
  else res="FAILED rc=$rc (tail: $(tail -n 1 $R/logs/$p.log | cut -c1-120))"; fi
  log "## $p: $res, $dt s"
done
log "RUN_DONE $(date -u +%FT%T)"
