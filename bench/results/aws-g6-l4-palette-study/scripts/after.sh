#!/bin/bash
# after.sh PID CMD...: run CMD once process PID has exited
p=$1; shift
while kill -0 $p 2>/dev/null; do sleep 5; done
"$@"
