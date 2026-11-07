#!/bin/bash
pkill -f 'python3 batchp' ; sleep 1
cd ~/t; nohup python3 batchp.py > ~/results/batchp.txt 2>&1 < /dev/null &
