#!/bin/bash
# rebuild.sh: rebuild ~/new (the uploaded working-tree files are already in place) -> ~/results/rebuild.log
s=$(date +%s)
cmake --build ~/new/build -j 32 > ~/results/rebuild.log 2>&1
echo "exit $? after $(( $(date +%s) - s )) s" >> ~/results/rebuild.log
tail -n 3 ~/results/rebuild.log
