#!/bin/bash
# mkstates.sh: worktree files as of commits 1..3 -> ~/cN on the instance (copies of ~/old's tree + those files), built
# like ~/new (k_recon objects from ~/old); the worktree is left in the final state.
P=/private/tmp/claude-501/-Users-mei-git-av1gpu/3ff16727-2f42-41f2-b891-0efc6c7767ab/scratchpad/pipe08
W=/Users/mei/git/av1gpu/.claude/worktrees/agent-a3ea690e9d945437e
FILES="encoder/apps/av1gpu.cpp encoder/src/batch.cpp encoder/src/pipeline.cpp ingest/src/ingest_api.h ingest/src/pipeline.cu"
for n in 1 2 3; do
  python3 $P/scripts/stage.py $n
  $P/ssh.sh "rm -rf ~/c$n && mkdir -p ~/c$n && tar -xzf ~/main.tar.gz -C ~/c$n && echo c$n > ~/c$n/COMMIT"
  $P/push.sh c$n $FILES
done
python3 $P/scripts/stage.py 4
cat > /tmp/pipe08_bld.sh <<'EOF'
for n in 1 2 3; do
  cmake -S ~/c$n -B ~/c$n/build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89 \
    -DCMAKE_CUDA_COMPILER=/usr/local/cuda/bin/nvcc -DAV1GPU_TESTS=OFF \
    -DAV1GPU_RECON_OBJECTS=$HOME/old/build/CMakeFiles/av1gpu_recon.dir/encoder/src > ~/results/setup/c$n.cfg.log 2>&1
  cmake --build ~/c$n/build -j 8 > ~/results/setup/c$n.log 2>&1; echo "c$n exit $?" >> ~/results/setup/cstates.log
done
echo CSTATES_DONE >> ~/results/setup/cstates.log
EOF
$P/scp.sh -q /tmp/pipe08_bld.sh h:~/bld_states.sh
echo states pushed
