#!/bin/bash
# why is --level 5%,95%,1.1 20 % slower on the standard set when --level 5%,95% and --gamma 1.2 are not?
B=$HOME/pr/build/av1gpu; L=~/results/after.log; D=~/prof/stdlevel; mkdir -p $D
until grep -q AFTER3_DONE $L; do sleep 5; done
echo "$(date -u +%H:%M:%S) stdlevel" >> $L
python3 ~/spd65.py stdlv 5 "plain=$B;plain2=$B;level=$B|--level 5%,95%,1.1;levelnog=$B|--level 5%,95%;levelP0=$B|--level 5%,95%,1.1 --palette off;nogP0=$B|--level 5%,95% --palette off;plainP0=$B|--palette off;lv105=$B|--level 5%,95%,1.05;lv130=$B|--level 5%,95%,1.3;lvb=$B|--level 2%,98%,1.1" -- std > ~/results/spd_stdlv.txt 2>&1
for cfg in "plain|" "level|--level 5%,95%,1.1" "levelnog|--level 5%,95%"; do c=${cfg%%|*}; o=${cfg#*|}; rm -rf /dev/shm/out
  nsys profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true -o $D/$c $B --batch ~/work/set_std/speedset --outdir /dev/shm/out --stats $o > $D/$c.out 2> $D/$c.err
  nsys export --type sqlite --force-overwrite=true -o $D/$c.sqlite $D/$c.nsys-rep > /dev/null 2>&1
  python3 ~/prof_an.py $D/$c.sqlite std_$c | tee $D/$c.an
  python3 - $D/$c.sqlite <<'PY' | tee -a $D/$c.an
import collections, sqlite3, sys
db = sqlite3.connect(sys.argv[1]); names = dict(db.execute("select id, value from StringIds"))
t = collections.Counter(); n = collections.Counter()
for st, en, s in db.execute("select start,end,shortName from CUPTI_ACTIVITY_KIND_KERNEL"):
    t[names[s]] += en - st; n[names[s]] += 1
for k, v in t.most_common(14): print(f"  {v / 1e6:9.1f} ms n={n[k]:6d} {k[:70]}")
PY
  rm -f $D/$c.nsys-rep $D/$c.sqlite; done
echo "$(date -u +%H:%M:%S) AFTER4_DONE" >> $L
