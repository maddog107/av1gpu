#!/bin/bash
# ptxv.sh TREE FILE...: recompile encoder/src/FILE of TREE's build with -Xptxas -v into /tmp; registers / spills per
# kernel -> ~/results/prof/ptxv_<tree>_<file>.txt
T=$1; shift
cd ~/$T/build
for f in "$@"; do
  cmd=$(ninja -t commands CMakeFiles/av1gpu_encoder.dir/encoder/src/$f.o | tail -1)
  cmd=$(echo "$cmd" | sed -e "s# -o [^ ]*$f.o# -o /tmp/ptxv_$f.o#" -e 's#-MD -MT [^ ]* -MF [^ ]*##')
  eval "$cmd -Xptxas -v" > ~/results/prof/ptxv_${T}_$f.raw 2>&1
  python3 - ~/results/prof/ptxv_${T}_$f.raw > ~/results/prof/ptxv_${T}_$f.txt <<'EOF'
import re, sys, subprocess
fn = None
for l in open(sys.argv[1]):
    m = re.search(r"Compiling entry function '([^']+)'", l)
    if m:
        fn = subprocess.run(["/usr/local/cuda/bin/cu++filt", m.group(1)], capture_output=True, text=True).stdout.strip()
    m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", l)
    if m and fn: st = m.groups()
    m = re.search(r"Used (\d+) registers", l)
    if m and fn:
        print(f"{fn[:90]:90s} regs {m.group(1):>4s} stack {st[0]:>5s} spillST {st[1]:>5s} spillLD {st[2]:>5s} | {l.strip()[-60:]}")
        fn = None
EOF
done
