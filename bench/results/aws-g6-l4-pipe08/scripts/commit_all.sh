#!/bin/bash
# commit_all.sh: (re)create the branch's commits 1..6 on top of cde1f74 from stage.py's states, messages in msg/N.txt;
# then the results commit if bench/results/aws-g6-l4-pipe08 exists (msg/7.txt).  Run in the worktree.
set -e
P=/private/tmp/claude-501/-Users-mei-git-av1gpu/3ff16727-2f42-41f2-b891-0efc6c7767ab/scratchpad/pipe08
cd /Users/mei/git/av1gpu/.claude/worktrees/agent-a3ea690e9d945437e
git reset -q --soft cde1f74
git restore --staged . 2>/dev/null || true
for n in 1 2 3 4 5 6; do
  python3 $P/scripts/stage.py $n > /dev/null
  git add encoder/apps/av1gpu.cpp encoder/src/batch.cpp encoder/src/pipeline.cpp ingest/src/ingest_api.h ingest/src/pipeline.cu
  git commit -q -F $P/msg/$n.txt
done
if [ -d bench/results/aws-g6-l4-pipe08 ]; then
  git add bench/results/aws-g6-l4-pipe08
  git commit -q -F $P/msg/7.txt
fi
git log --oneline cde1f74..HEAD
git status --short | head
