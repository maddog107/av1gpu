#!/bin/bash
# after.sh ARCH: after post.sh, with the GPU otherwise idle: the module-load timing (modload.sh ARCH).
until grep -q POST_DONE ~/results/post.log 2>/dev/null; do sleep 10; done
[ -x /usr/local/cuda-13.4/bin/cuobjdump ] || sudo DEBIAN_FRONTEND=noninteractive apt-get install -yq cuda-cuobjdump-13-4 > ~/results/setup/cuobjdump_apt.log 2>&1
bash ~/modload.sh $1
echo AFTER_DONE
