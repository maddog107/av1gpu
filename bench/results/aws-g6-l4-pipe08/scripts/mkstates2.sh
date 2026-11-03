#!/bin/bash
# mkstates2.sh N...: worktree files as of commit N -> ~/cN on the instance, then a build script ~/bld_states2.sh for them;
# the worktree is left in the final state (6)
P=/private/tmp/claude-501/-Users-mei-git-av1gpu/3ff16727-2f42-41f2-b891-0efc6c7767ab/scratchpad/pipe08
FILES="encoder/apps/av1gpu.cpp encoder/src/batch.cpp encoder/src/pipeline.cpp ingest/src/ingest_api.h ingest/src/pipeline.cu"
: > /tmp/pipe08_bld2.sh
for n in "$@"; do
  python3 $P/scripts/stage.py $n
  $P/ssh.sh "rm -rf ~/c$n && mkdir -p ~/c$n && tar -xzf ~/main.tar.gz -C ~/c$n && echo c$n > ~/c$n/COMMIT"
  $P/push.sh c$n $FILES
  cat >> /tmp/pipe08_bld2.sh <<EOF
cmake -S ~/c$n -B ~/c$n/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DAV1GPU_TESTS=OFF \
  -DAV1GPU_RECON_OBJECTS=\$HOME/old/build/CMakeFiles/av1gpu_recon.dir/encoder/src > ~/results/setup/c$n.cfg.log 2>&1
cmake --build ~/c$n/build -j 8 > ~/results/setup/c$n.log 2>&1; echo "c$n exit \$?" >> ~/results/setup/cstates.log
EOF
done
echo "echo CSTATES2_DONE >> ~/results/setup/cstates.log" >> /tmp/pipe08_bld2.sh
python3 $P/scripts/stage.py 6
$P/scp.sh -q /tmp/pipe08_bld2.sh h:~/bld_states2.sh
echo pushed
