#!/usr/bin/env python3
"""Patch libaom 3.13.1's av1/encoder/speed_features.c with an environment switch LRSTUDY for this study.
Unset (or empty), the encoder is unchanged (checked: byte-identical files at speed 6).  In all-intra mode:

  LRSTUDY=keep    undo the speed >= 5 rule that turns off both loop-restoration filters (so speed 5/6 keep the
                  Wiener and self-guided search libaom has at speed 4, with the speed 1-4 pruning)
  LRSTUDY=full    keep, and every LR pruning speed feature back at its speed-0 value, plus the speed-0
                  restoration-unit size search (64, 128 and 256); the ceiling of what libaom's LR search finds
  LRSTUDY=wiener  full, Wiener only (self-guided off)
  LRSTUDY=sgr     full, self-guided only (Wiener off)

usage: patch_lrstudy.py AOM_SOURCE_DIR
"""
import sys

path = sys.argv[1] + "/av1/encoder/speed_features.c"
s = open(path).read()

HOOK = r'''
#include <stdlib.h>
#include <string.h>
// LRSTUDY (av1gpu loop-restoration study): see bench/results/aws-c7a-lr-study/scripts/patch_lrstudy.py
static const char *lrstudy_mode(void) {
  const char *e = getenv("LRSTUDY");
  return e && *e ? e : NULL;
}
static void lrstudy_apply(SPEED_FEATURES *const sf) {
  const char *m = lrstudy_mode();
  if (!m) return;
  sf->lpf_sf.disable_wiener_filter = false;
  sf->lpf_sf.disable_sgr_filter = false;
  if (!strcmp(m, "wiener")) sf->lpf_sf.disable_sgr_filter = true;
  if (!strcmp(m, "sgr")) sf->lpf_sf.disable_wiener_filter = true;
  if (!strcmp(m, "full") || !strcmp(m, "wiener") || !strcmp(m, "sgr")) {
    sf->lpf_sf.disable_loop_restoration_chroma = 0;
    sf->lpf_sf.disable_loop_restoration_luma = 0;
    sf->lpf_sf.prune_wiener_based_on_src_var = 0;
    sf->lpf_sf.prune_sgr_based_on_wiener = 0;
    sf->lpf_sf.enable_sgr_ep_pruning = 0;
    sf->lpf_sf.reduce_wiener_window_size = 0;
    sf->lpf_sf.dual_sgr_penalty_level = 0;
    sf->lpf_sf.disable_wiener_coeff_refine_search = false;
    sf->lpf_sf.use_downsampled_wiener_stats = 0;
    sf->lpf_sf.switchable_lr_with_bias_level = 0;
  }
}

void av1_set_speed_features_framesize_independent(AV1_COMP *cpi, int speed) {'''

a = "\nvoid av1_set_speed_features_framesize_independent(AV1_COMP *cpi, int speed) {"
assert s.count(a) == 1
s = s.replace(a, HOOK)

b = "      set_allintra_speed_features_framesize_independent(cpi, sf, speed);\n"
assert s.count(b) == 1
s = s.replace(b, b + "      lrstudy_apply(sf);\n")

c = """  set_subpel_search_method(&cpi->mv_search_params,
                           cpi->oxcf.unit_test_cfg.motion_vector_unit_test,
                           sf->mv_sf.subpel_search_method);

  if (cpi->oxcf.enable_low_complexity_decode && cpi->oxcf.mode == GOOD)"""
assert s.count(c) == 1
s = s.replace(c, """  if (cpi->oxcf.mode == ALLINTRA && lrstudy_mode() && strcmp(lrstudy_mode(), "keep")) {
    sf->lpf_sf.min_lr_unit_size = RESTORATION_PROC_UNIT_SIZE;
    sf->lpf_sf.max_lr_unit_size = RESTORATION_UNITSIZE_MAX;
    sf->lpf_sf.disable_loop_restoration_luma = 0;
  }

""" + c)
open(path, "w").write(s)
print("patched", path)
