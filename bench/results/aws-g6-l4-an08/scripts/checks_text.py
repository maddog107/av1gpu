#!/usr/bin/env python3
"""checks_text.py SANITIZER.txt OUT: the checks section of summary.md (decoder gate, ctest, sanitizers)."""
import sys

san = open(sys.argv[1]).read().strip().splitlines()
san = [l for l in san if l.startswith(("memcheck", "synccheck"))]
T = """## Decoder gate, tests, sanitizers (branch head)

- **Decoder gate** (`tests/decoder_gate.py`, dav1d 1.4.1 and aomdec, `logs/gate_new2.txt`): 12 runs, **497/497
  encodes and 3/3 alpha items identical** in both decoders, all PASS: 30 size-range photos (3 of every 10) at qindex
  60 / 120 / 200, 4:4:4 10-bit, `--preset slow`; 28 thumbnails at 60 / 120; the 30 odd sizes in 4:2:0 and 4:4:4
  10-bit; the gray and gray + alpha PNGs; the 47 monochrome inputs at 26 / 120; the palette sets screen (also 4:4:4
  10-bit), flat_png and flat_jpg.
- **ctest** 9/9 passed; `gpu_fuzz_test` seeds 1-400 and 77-276 (more 4:4:4, 10-bit, lossless): 0 mismatches;
  `gpu_dsp_test` passed (`logs/ctest_new2.txt`).
- **compute-sanitizer racecheck** (analysis mode, `race_ab.sh`: 3 size-range photos, default and `--yuv 444 --depth
  10`; `logs/race/`): 8 hazards per run, all in `k_entropy_win` / `vload` / `vstore` (gpu_entropy.cu, the documented
  producer / consumer hand-off of docs/RESULTS.md); nothing in the analysis kernels. A first build of change 2 without
  the `__syncwarp` before `warp_residual`'s writes showed reports between `coeff_bits` reads of the levels and the next
  residual's writes (`logs/v2pre/race_new2_0_partial_head.txt`); the fix went in before the final runs.
- **memcheck / synccheck** (`sanitizer.sh`, alone on the GPU, `logs/sanitizer_new2b.txt`):
"""
T += "\n".join(f"  - {l}" for l in san) + "\n"
T += """  The leak lines of memcheck `--leak-check full` are the variance-boost tables of v0.6's summary (pipeline.cpp, the
  same in main).
"""
open(sys.argv[2], "w").write(T)
