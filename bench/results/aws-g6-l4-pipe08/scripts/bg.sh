#!/bin/bash
# bg.sh SCRIPT [OUT]: run ~/SCRIPT detached on the instance (own session, no inherited fds), output in ~/results/OUT
S=$1; O=${2:-$(basename $S .sh).out}
cd ~ && setsid nohup bash ~/$S > ~/results/$O 2>&1 < /dev/null &
disown
echo "started $S"
exit 0
