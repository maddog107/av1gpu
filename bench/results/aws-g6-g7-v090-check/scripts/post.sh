#!/bin/bash
# post.sh l4|bw: everything after main c553ffb's packages arrive in ~/rel (scp from the Release run's artifacts).
#  1. sha256sum -c and unpack: ~/bin/B13 (CUDA 13.4 package), ~/bin/B12 (-cuda12)
#  2. (bw) resusage9.py: k_recon's registers / stack / spills for sm_89 and sm_120, v0.8.0 against v0.9.0
#  3. identity: B13 on the full list against A13; B12 on the CUDA 12 subset against A12 and against B13;
#     manifests (SHA-256 per output) of A13 and B13 for the comparison across GPUs
#  4. checks9.sh B13 (decoder gate, compute-sanitizer)
#  5. speed9.sh (l4: plus every speed of the docs)
# ~/results/post.log; prints POST_DONE.
K=$1
L=~/results/post.log; exec >> $L 2>&1
until grep -q PRE_DONE ~/results/pre.out 2>/dev/null && [ -e ~/rel/B_READY ]; do sleep 5; done   # B_READY: touched after the scp
echo "start $(date -u +%T)"
cd ~/rel
for s in "" -cuda12; do f=av1gpu-dev-c553ffb-linux-x86_64$s.tar.gz; sha256sum -c $f.sha256 || { echo "SHA_FAIL $f"; exit 1; }; tar -xzf $f; done
cp ~/rel/av1gpu-dev-c553ffb-linux-x86_64/av1gpu ~/bin/B13; cp ~/rel/av1gpu-dev-c553ffb-linux-x86_64-cuda12/av1gpu ~/bin/B12
sha256sum ~/rel/av1gpu-dev-*.tar.gz ~/bin/B13 ~/bin/B12 | tee ~/results/b_sha256.txt
ls -la ~/rel/av1gpu-dev-c553ffb-linux-x86_64 ~/rel/av1gpu-dev-c553ffb-linux-x86_64-cuda12
cd ~
[ $K = bw ] && { python3 ~/resusage9.py > ~/results/resusage.out 2>&1; echo "resusage rc $? $(date -u +%T)"; }
bash ~/gen.sh B13 ~/bin/B13 full > ~/results/gen_B13.txt 2>&1
python3 ~/cmp9.py cmp ~/out/A13 ~/out/B13 A13_B13 full > ~/results/cmp_A13_B13.txt 2>&1
tail -1 ~/results/cmp_A13_B13.txt
python3 ~/cmp9.py man ~/out/A13 ~/results/man_${K}_A13.json; python3 ~/cmp9.py man ~/out/B13 ~/results/man_${K}_B13.json
if [ $K = bw ]; then bash ~/gen.sh A12 ~/bin/A12 sub > ~/results/gen_A12.txt 2>&1; fi
bash ~/gen.sh B12 ~/bin/B12 sub > ~/results/gen_B12.txt 2>&1
python3 ~/cmp9.py cmp ~/out/A12 ~/out/B12 A12_B12 sub > ~/results/cmp_A12_B12.txt 2>&1
python3 ~/cmp9.py cmp ~/out/B13 ~/out/B12 B13_B12 sub > ~/results/cmp_B13_B12.txt 2>&1
tail -1 ~/results/cmp_A12_B12.txt ~/results/cmp_B13_B12.txt
echo "identity done $(date -u +%T)"
bash ~/checks9.sh ~/bin/B13 gate > ~/results/checks9_gate.out 2>&1
echo "gate done $(date -u +%T)"
grep -E '^rc ' ~/results/gate.txt | sort | uniq -c
if [ $K = l4 ]; then bash ~/speed9.sh docs; else bash ~/speed9.sh; fi
echo "speed done $(date -u +%T)"
bash ~/checks9.sh ~/bin/B13 san > ~/results/checks9_san.out 2>&1
echo "sanitizers done $(date -u +%T)"
cat ~/results/sanitizer.txt
echo "POST_DONE $(date -u +%T)"
