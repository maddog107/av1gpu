#!/bin/bash
# sizeof(RcShared<8>) / <10> and the RcUnion in main and the branch (static shared memory of k_recon)
for t in av1gpu br; do
  R=~/$t
  cat > /tmp/rcsize_$t.cu <<'EOF'
#include <cstdio>
#include "k_recon.cuh"
int main() { printf("RcShared<8> %zu RcShared<10> %zu RcUnion %zu\n", sizeof(RcShared<8>), sizeof(RcShared<10>), sizeof(RcUnion)); }
EOF
  nice /usr/local/cuda/bin/nvcc -std=c++17 --expt-relaxed-constexpr -I$R/encoder/src -I$R/av1f/include -I$R/av1core/include -I$R/av1core/src \
    -I$R/ingest/src -o /tmp/rcsize_$t /tmp/rcsize_$t.cu 2>&1 | grep -v warning | head -5
  echo "$t: $(/tmp/rcsize_$t)"
done
