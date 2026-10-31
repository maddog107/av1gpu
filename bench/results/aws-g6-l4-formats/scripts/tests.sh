#!/bin/bash
# On the instance: tests.sh PHASE   (outputs in ~/results/<phase>[_TAG].txt)
#   sets | timing | rounds | unit | regress | gate8 | gate444 | gate10 | gateimg
export AV1G_DAV1D=dav1d AV1G_AOMDEC=aomdec TMPDIR=$HOME/tmp
mkdir -p ~/tmp ~/results
cd ~
B=${BRB:-$HOME/build-br}; M=~/build-main; G="python3 $HOME/br/tests/decoder_gate.py"
T=${TAG:+_$TAG}
gate() {   # label images expect-format expect-depth q [-- options]
  local l=$1 im=$2 ef=$3 ed=$4 q=$5; shift 5
  echo "-- $l ($(basename $im), q$q)"
  local x=(); [ "$ef" != - ] && x+=(--expect-format $ef); [ "$ed" != - ] && x+=(--expect-depth $ed)
  local t0=$(date +%s)
  $G --bin $B/av1gpu --images $im --q $q --jobs 14 "${x[@]}" --json ~/results/gate_json/$(echo "$l" | tr ' /:' '___')_$(basename $im).json "$@" 2>&1 | grep "^\[gate\]"
  echo "   ($(( $(date +%s) - t0 )) s)"
}
mkdir -p ~/results/gate_json
case "$1" in
sets)
  python3 ~/mk_sets.py > ~/results/sets.txt 2>&1
  echo "images $(ls images | wc -l), gamut $(ls images_gamut | wc -l)" >> ~/results/sets.txt
  ;;
timing)   # Phase B: main vs branch default (+ VARIANTS), GP_ENCODERS=4, gamut set, bench.gpu_speed, alternating rounds
  mk() { printf '#!/bin/sh\nexport GP_ENCODERS=4\nexec %s "$@"\n' $2 > ~/w_$1.sh; chmod +x ~/w_$1.sh; }
  for v in ${NAMES:-main br}; do mk $v ~/build-$v/av1gpu; done
  NAMES="${NAMES:-main br}" ROUNDS=${ROUNDS:-3} TAG=${TAG:-b} python3 - > ~/results/timing$T.txt 2>&1 <<'PY'
import json, os, statistics, sys, types
sys.path.insert(0, os.path.expanduser("~/br/bench"))
import bench
from PIL import Image
D = os.path.expanduser("~/images_gamut"); work = os.path.expanduser("~/work_t"); os.makedirs(work, exist_ok=True)
files = sorted(os.path.join(D, f) for f in os.listdir(D))
sizes = {os.path.basename(f): (lambda im: im.width * im.height / 1e6)(Image.open(f)) for f in files}
sdir, sfiles, smp, reps = bench.speed_set(files, sizes, 1000, work)
names = os.environ["NAMES"].split()
print(len(files), "images;", f"{smp:.0f} MP per run, GP_ENCODERS=4, bench.gpu_speed (warm-up + 3 timed runs), "
      f"{os.environ['ROUNDS']} rounds, order alternating", flush=True)
out = {}
for rnd in range(int(os.environ["ROUNDS"])):
    for name in (names if rnd % 2 == 0 else names[::-1]):
        a = types.SimpleNamespace(av1gpu=os.path.expanduser(f"~/w_{name}.sh"), repeats=3, gpu_label=name)
        r = bench.gpu_speed(a, sdir, work)
        out.setdefault(name, []).append(r)
        print(rnd, name, f"{r['mp_s']:.1f} MP/s", " ".join(f"{x['mp_s']:.1f}" for x in r["runs"]),
              f"{r['cpu_cores']:.2f} cores", r["bytes"], "bytes", flush=True)
med = {}
for name, rs in out.items():
    allr = [x["mp_s"] for r in rs for x in r["runs"]]
    med[name] = statistics.median(allr)
    print(f"{name}: median of {len(allr)} runs {med[name]:.2f} MP/s (min {min(allr):.1f}, max {max(allr):.1f}); "
          f"round medians {' '.join('%.1f' % r['mp_s'] for r in rs)}", flush=True)
for name in names[1:]:
    print(f"{name} vs {names[0]}: {100 * (med[name] / med[names[0]] - 1):+.2f} %")
json.dump(out, open(os.path.expanduser(f"~/results/timing_{os.environ['TAG']}.json"), "w"), indent=1)
PY
  ;;
rounds)   # diagnostic: N alternating single-encoder runs per build on the gamut set; median recon_ms / MP/s
  N=${N:-5}; V=${NAMES:-main br}
  for r in $(seq 1 $N); do
    for v in $V; do
      rm -rf ~/work_r/s1
      GP_ENCODERS=1 ~/build-$v/av1gpu --batch ~/images_gamut --outdir ~/work_r/s1 --stats 2>/dev/null | tail -1 | \
        python3 -c "import json,sys; s=json.loads(sys.stdin.read()); print('$v', s['recon_ms'], s['mp_s'], s['analysis_ms'], s['bytes'])"
    done
  done > ~/results/rounds$T.raw
  python3 - ~/results/rounds$T.raw > ~/results/rounds$T.txt <<'EOF'
import statistics, collections, sys
d = collections.defaultdict(list)
for l in open(sys.argv[1]):
    v, r, m, a, b = l.split(); d[v].append((float(r), float(m), float(a), b))
base = None
for v, xs in d.items():
    rm = statistics.median(x[0] for x in xs)
    base = base or rm
    print(f"{v:8s} n={len(xs)} recon_ms median {rm:.0f} (min {min(x[0] for x in xs):.0f}, {100 * (rm / base - 1):+.2f} %)  "
          f"mp_s median {statistics.median(x[1] for x in xs):.1f}  analysis median {statistics.median(x[2] for x in xs):.0f}  "
          f"bytes {sorted(set(x[3] for x in xs))}")
EOF
  ;;
unit)
  {
  echo "== gpu_dsp_test"; $B/gpu_dsp_test 2>&1 | tail -7
  echo "== varboost_test"; $B/varboost_test 2>&1 | tail -8
  echo "== gpu_fuzz_test mixed (defaults: 40 % 10-bit, 40 % of colour 4:4:4)"; $B/gpu_fuzz_test --seed 1 --count 3000 2>&1 | tail -2
  echo "== gpu_fuzz_test 4:4:4 only (8-bit)"; $B/gpu_fuzz_test --seed 500001 --count 1500 --yuv444 100 --depth10 0 2>&1 | tail -2
  echo "== gpu_fuzz_test 10-bit only"; $B/gpu_fuzz_test --seed 700001 --count 1500 --depth10 100 2>&1 | tail -2
  echo "== gpu_fuzz_test 10-bit 4:4:4 only"; $B/gpu_fuzz_test --seed 800001 --count 800 --depth10 100 --yuv444 100 2>&1 | tail -2
  echo "== gpu_fuzz_test 8-bit 4:2:0 stream (--depth10 0 --yuv444 0)"; $B/gpu_fuzz_test --seed 1 --count 400 --depth10 0 --yuv444 0 2>&1 | tail -2
  echo "== gpu_postfilter_test mixed"; $B/gpu_postfilter_test --seed 1 --count 2000 2>&1 | tail -2
  echo "== gpu_postfilter_test 4:4:4 only (8-bit)"; $B/gpu_postfilter_test --seed 300001 --count 1000 --yuv444 100 --depth10 0 2>&1 | tail -2
  echo "== gpu_postfilter_test 10-bit only"; $B/gpu_postfilter_test --seed 310001 --count 1500 --depth10 100 2>&1 | tail -2
  echo "== gpu_postfilter_test 10-bit 4:4:4 only"; $B/gpu_postfilter_test --seed 400001 --count 600 --depth10 100 --yuv444 100 2>&1 | tail -2
  echo "== ctest"; (cd $B && ctest --output-on-failure 2>&1 | tail -6)
  echo "== av1g_fuzz 2000 cases (instance dav1d 1.4.1 / aomdec 3.8.2)"; $B/av1g_fuzz --seed 900001 --count 2000 --quiet --out ~/tmp/fz 2>&1 | grep -E "SUMMARY|missing"
  } > ~/results/unit$T.txt 2>&1
  ;;
regress)   # every 8-bit 4:2:0 .avif of main and the branch byte-identical: default, turbo3, slow; bench + gamut sets
  {
  for set in images images_gamut; do
    for p in default turbo3 slow; do
      for b in main br; do
        o=~/reg/$set-$b-$p; rm -rf $o; mkdir -p $o
        x=""; [ $p != default ] && x="--preset $p"
        bin=$M/av1gpu; [ $b = br ] && bin=$B/av1gpu
        $bin --batch ~/$set --outdir $o $x > $o.log 2>&1
        echo "$set $b $p rc=$? $(ls $o | wc -l) files $(cat $o/*.avif | wc -c) bytes"
      done
      n=0; d=0
      for f in ~/reg/$set-main-$p/*.avif; do
        n=$((n+1)); cmp -s "$f" ~/reg/$set-br-$p/$(basename "$f") || { d=$((d+1)); echo "DIFF $(basename $f)"; }
      done
      echo "REGRESSION $set $p: $n files compared, $d differ"
    done
  done
  } > ~/results/regress$T.txt 2>&1
  ;;
regressx)   # 4:4:4 / 10-bit outputs of the speed fix (BRB) byte-identical to the unmodified branch (~/build-br)
  {
  for set in images images_gamut; do
    for f in 444_8 420_10 444_10; do
      for b in head fix; do
        o=~/regx/$set-$b-$f; rm -rf $o; mkdir -p $o
        x="--yuv 444"; [ $f = 420_10 ] && x="--depth 10"; [ $f = 444_10 ] && x="--depth 10 --yuv 444"
        bin=~/build-br/av1gpu; [ $b = fix ] && bin=$B/av1gpu
        $bin --batch ~/$set --outdir $o $x > $o.log 2>&1
        echo "$set $b $f rc=$? $(ls $o | wc -l) files $(cat $o/*.avif | wc -c) bytes"
      done
      n=0; d=0
      for f2 in ~/regx/$set-head-$f/*.avif; do
        n=$((n+1)); cmp -s "$f2" ~/regx/$set-fix-$f/$(basename "$f2") || { d=$((d+1)); echo "DIFF $(basename $f2)"; }
      done
      echo "REGRESSION (fix vs unmodified branch) $set $f: $n files compared, $d differ"
    done
  done
  } > ~/results/regressx$T.txt 2>&1
  ;;
gate8)
  {
  for im in ~/images ~/images_gamut; do
    gate "8-bit default" $im 420 8 60,120
    gate "8-bit turbo3" $im 420 8 60,120 -- --preset turbo3
    gate "8-bit slow" $im 420 8 60,120 -- --preset slow
    gate "8-bit slow small-preset none" $im 420 8 60,120 -- --preset slow --small-preset none
  done
  } > ~/results/gate8$T.txt 2>&1
  ;;
gate444)
  {
  for im in ~/images ~/images_gamut; do
    gate "8-bit 4:4:4" $im 444 8 60,120 -- --yuv 444
  done
  gate "8-bit 4:4:4 slow small-preset none" ~/images 444 8 60,120 -- --yuv 444 --preset slow --small-preset none
  gate "8-bit 4:4:4 turbo3" ~/images 444 8 120 -- --yuv 444 --preset turbo3
  } > ~/results/gate444$T.txt 2>&1
  ;;
gate10)
  {
  for im in ~/images ~/images_gamut; do
    gate "10-bit 4:2:0" $im 420 10 60,120 -- --depth 10
    gate "10-bit 4:4:4" $im 444 10 60,120 -- --depth 10 --yuv 444
  done
  gate "10-bit slow small-preset none" ~/images 420 10 60,120 -- --depth 10 --preset slow --small-preset none
  gate "10-bit 4:4:4 slow small-preset none" ~/images 444 10 120 -- --depth 10 --yuv 444 --preset slow --small-preset none
  gate "10-bit turbo3" ~/images 420 10 120 -- --depth 10 --preset turbo3
  } > ~/results/gate10$T.txt 2>&1
  ;;
gateimg)   # 16-bit PNGs, 8-bit PNG / WebP (CPU-decoded inputs) in every format
  {
  for im in ~/images_png16 ~/images_png; do
    gate "8-bit 4:2:0" $im 420 8 60,120
    gate "8-bit 4:4:4" $im 444 8 60,120 -- --yuv 444
    gate "10-bit 4:2:0" $im 420 10 60,120 -- --depth 10
    gate "10-bit 4:4:4" $im 444 10 60,120 -- --depth 10 --yuv 444
  done
  } > ~/results/gateimg$T.txt 2>&1
  ;;
esac
echo "PHASE_DONE $1$T $(date -u +%H:%M:%S)" >> ~/results/phases.txt
