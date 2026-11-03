#!/bin/bash
# bg.sh SCRIPT OUT [ARGS...]: run ~/SCRIPT detached on the instance (own session, no inherited fds), output in ~/results/OUT
S=$1; O=$2; shift 2
mkdir -p ~/results
cd ~
setsid nohup bash ~/$S "$@" > ~/results/$O 2>&1 < /dev/null &
disown
echo "started $S"
exit 0
