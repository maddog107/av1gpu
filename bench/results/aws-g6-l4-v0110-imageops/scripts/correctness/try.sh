B=${BIN:-$HOME/pr/build/av1gpu}; M=~/main/build/av1gpu; T=${T:-/dev/shm/p$$}; mkdir -p $T
try() { rm -f $T/o.avif; s=$(date +%s.%N); timeout 200 "$@" > $T/out.txt 2> $T/err.txt; rc=$?; e=$(date +%s.%N)
  sz=""; if [ -f $T/o.avif ]; then sz=$(python3 - <<PY
import sys; sys.path.insert(0,'/home/ubuntu/t'); import avt
import numpy as np
try:
    c=avt.check('$T/o.avif'); y=c['planes'][0]; print(c['si'][:5], 'ok Y',int(y.min()),int(y.max()), 'alpha' if c['alpha'] is not None else '', 'irot',c['irot'],'imir',c['imir'])
except Exception as ex: print('BAD', ex)
PY
); fi
  printf 'rc=%s %.1fs [%s] %s | %s\n' $rc $(echo "$e-$s"|bc) "$sz" "${*:1}" "$(grep -v 'stage ms' $T/err.txt | tail -1 | cut -c1-170)" | sed "s#$T/o.avif##; s#/home/ubuntu/##g"
}
