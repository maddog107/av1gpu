#!/bin/bash
# gen.sh TAG BIN GROUP: every run of ~/sets9.tsv in GROUP (full | sub), encoded with BIN into ~/out/TAG/<name>/:
# o/ = the output files, rc = exit status (single-file runs: one "file status" line per input), log / err_*.txt = stderr.
# Single-file outputs are numbered (two inputs share a basename).  Prints one line per run and GEN_DONE.
TAG=$1; BIN=$2; GRP=$3
R=~/out/$TAG; mkdir -p $R
echo "TAG $TAG BIN $BIN ($(sha256sum < $BIN | cut -c1-16)) GROUP $GRP start $(date -u +%H:%M:%S)"
t0=$(date +%s)
while IFS='|' read -r kind name src envs opts groups; do
  [[ " $groups " == *" $GRP "* ]] || continue
  O=$R/$name; rm -rf $O; mkdir -p $O/o; t=$(date +%s)
  case $kind in
    batch) env $envs $BIN --batch $src --outdir $O/o $opts > $O/log 2>&1 < /dev/null; echo $? > $O/rc ;;
    single) i=0; : > $O/rc
      while read -r f; do i=$((i + 1))
        case "$f" in *.jpg|*.jpeg) [[ " $opts " == *" --lossless "* ]] && continue;; esac
        b=$(printf %03d_%s $i "$(basename $f)")
        env $envs $BIN $f $O/o/$b.avif $opts > $O/err_$b.txt 2>&1 < /dev/null; echo "$b $?" >> $O/rc
      done < $src ;;
    err) env $envs $BIN $src $O/o/out.avif $opts > $O/log 2>&1 < /dev/null; echo $? > $O/rc ;;
  esac
  echo "$name: $(ls $O/o | wc -l) files, rc $(awk '{print $NF}' $O/rc | sort | uniq -c | tr -s ' \n' ' '), $(( $(date +%s) - t )) s"
done < ~/sets9.tsv
echo "GEN_DONE $TAG $(( $(date +%s) - t0 )) s"
