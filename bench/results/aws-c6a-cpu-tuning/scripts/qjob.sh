#!/bin/bash
# Quality work (host-independent: bytes and SSIMULACRA2 scores are deterministic), run on the replica host after
# its timing phase: published-row check, q60 scores of the libaom -j 2 outputs, full sweeps at -j 2 and with
# --autotiling at 4 x -j 4.
set -x
while kill -0 $(cat ~/timing_main.pid) 2>/dev/null; do sleep 10; done
python3 -u ~/quality.py check
python3 -u ~/quality.py splits A_stock_aom_s6_j2 A_stock_aom_s9_j2
python3 -u ~/quality.py sweep stock aom_s6,aom_s9 2
python3 -u ~/quality.py sweep stock aom_s6,svt_p6,svt_p4 4 tiles
echo QJOB_DONE
