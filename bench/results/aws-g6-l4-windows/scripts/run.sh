#!/bin/bash
# run.sh BIN TAG: every case of cases.txt (in ~/data) -> ~/res/TAG/{md5.txt,exit.txt,logs/}
BIN=$1; TAG=$2
cd ~/data
R=~/res/$TAG; rm -rf $R out; mkdir -p $R/logs out
while IFS=$'\t' read -r name envs args; do
  [ -z "$name" ] && continue
  case $name in s_*) mkdir -p out/$name;; esac
  [ "$envs" = - ] && envs=
  env $envs $BIN $args > $R/logs/$name.out 2> $R/logs/$name.err
  echo "$name $?" >> $R/exit.txt
done < cases.txt
( cd out && find . -type f | LC_ALL=C sort | while read -r f; do echo "${f#./} $(md5sum < "$f" | cut -c1-32)"; done ) > $R/md5.txt
echo "$(wc -l < $R/md5.txt) files"
