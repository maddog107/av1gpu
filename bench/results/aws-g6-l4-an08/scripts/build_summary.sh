#!/bin/bash
# build_summary.sh: identity table + speed tables + checks -> bench/results/aws-g6-l4-an08/summary.md
P=/private/tmp/claude-501/-Users-mei-git-av1gpu/3ff16727-2f42-41f2-b891-0efc6c7767ab/scratchpad/an08/pay
D=/Users/mei/git/av1gpu/.claude/worktrees/agent-acbf441b11e05f2c5/bench/results/aws-g6-l4-an08
L=$D/logs
T=/private/tmp/claude-501/-Users-mei-git-av1gpu/3ff16727-2f42-41f2-b891-0efc6c7767ab/scratchpad/an08/fetched
python3 $P/mk_idtable.py $L/identity_new2.txt $L/identity_pal_new2.txt > $T/idtable.md
python3 $P/mk_speed.py $L/runs/ab3.jsonl $L/runs/noise.jsonl $L/runs/single.jsonl $L/runs/ab3_sets.json > $T/speed.md
python3 $P/checks_text.py $L/sanitizer_new2b.txt $T/checks.md
python3 $P/mk_summary.py $D/summary.md $T/idtable.md $T/speed.md $T/checks.md
cat $T/speed.md
tail -n 2 $T/idtable.md
