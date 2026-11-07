#!/bin/bash
# Release-style packages.  ~/pkg holds the PR's Release-run artifacts (run 42298013588); v0.10.0 is in ~/rel.
# main' and PR' are built here in the release container with the Release run's k_recon objects, so main' differs from the PR's
# package only by the PR.  Not a timing stage.
set -x
L=~/results/rel; mkdir -p $L; cd ~/pkg
mkdir -p pr13 pr12 recon13 recon12
unzip -qo pr_cuda13.zip -d pr13; tar xzf pr13/*.tar.gz -C pr13; unzip -qo pr_cuda12.zip -d pr12; tar xzf pr12/*.tar.gz -C pr12
for i in 0 1 2 3; do unzip -qo recon$i.zip -d recon13; done; unzip -qo recon12a.zip -d recon12; unzip -qo recon12b.zip -d recon12
ls -la recon13 recon12 pr13 pr12 > $L/ls.txt
{ sha256sum pr13/*.tar.gz pr12/*.tar.gz; cat pr13/*.sha256 pr12/*.sha256; } > $L/sha.txt
cat > ~/pkg/inner.sh <<'IN'
set -x
export DEBIAN_FRONTEND=noninteractive
apt-get update -q >/dev/null; apt-get install -yq python3-pip pkg-config g++ file curl libjpeg-turbo8-dev libpng-dev libwebp-dev zlib1g-dev >/dev/null 2>&1
pip3 install -q cmake==3.31.6 ninja
nvcc --version | tail -2 | head -1
for t in main pr; do
  s=$(date +%s)
  cmake -S /h/$t -B /h/rel13/$t -G Ninja -DCMAKE_BUILD_TYPE=Release -DAV1GPU_STATIC_DEPS=ON -DAV1GPU_TESTS=OFF -DAV1GPU_RECON_OBJECTS=/h/pkg/recon13 > /h/results/rel/$t.cfg.log 2>&1
  cmake --build /h/rel13/$t -j 14 --target av1gpu > /h/results/rel/$t.build.log 2>&1
  echo "$t exit $? after $(( $(date +%s) - s )) s" | tee -a /h/results/rel/build.txt
  cp /h/rel13/$t/av1gpu /h/rel13/$t.bin; strip --strip-debug /h/rel13/$t.bin
done
chown -R 1000:1000 /h/rel13
IN
docker pull -q nvidia/cuda:13.4.3-devel-ubuntu22.04 > $L/docker.log 2>&1 || sudo docker pull -q nvidia/cuda:13.4.3-devel-ubuntu22.04 >> $L/docker.log 2>&1
sudo docker run --rm -v $HOME:/h nvidia/cuda:13.4.3-devel-ubuntu22.04 bash /h/pkg/inner.sh > $L/container.log 2>&1
echo REL_BUILD_DONE
