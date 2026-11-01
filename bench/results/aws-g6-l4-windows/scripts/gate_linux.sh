#!/bin/bash
# decoder_gate.py runs (dav1d + aomdec) with the feat/windows Linux build, outputs kept in ~/gate/<run>/;
# gates.txt is also what gate.ps1 replays on Windows (run<TAB>images<TAB>q|lossless<TAB>av1gpu options)
cd ~/data
mkdir -p gate_img && cp ~/av1gpu/tests/gate_images/*.png gate_img/
rm -rf ~/gate; mkdir -p ~/gate
G=~/av1gpu/tests/decoder_gate.py; B=~/bin/av1gpu-feat-windows
while IFS=$'\t' read -r run imgs q extra opts; do
  [ -z "$run" ] && continue
  [ "$opts" = - ] && opts=
  if [ "$q" = lossless ]; then qa=(--lossless); else qa=(--q $q); fi
  [ "$extra" = - ] && extra=
  python3 $G --bin $B --images $imgs "${qa[@]}" $extra --keep ~/gate/$run --json ~/gate/$run.json -- $opts > ~/gate/$run.log 2>&1
  echo "$run exit $? $(grep -E '^(PASS|FAIL|ok|[0-9]+/[0-9]+)' ~/gate/$run.log | tail -1)"
done < ~/test/gates.txt
cd ~/gate && find . -type f ! -name '*.json' ! -name '*.log' | LC_ALL=C sort | while read -r f; do echo "${f#./} $(md5sum < "$f" | cut -c1-32)"; done > ~/gate_md5.txt
wc -l ~/gate_md5.txt
