#!/usr/bin/env python3
"""rename_logs.py LOGDIR: clearer names for two log groups (ptxv_new_* was built from main's sources; ef2 is the
dropped small-kernel experiment)."""
import glob
import os
import sys

d = sys.argv[1]
for f in glob.glob(f"{d}/prof/ptxv_new_k_an_*"):
    os.rename(f, f.replace("ptxv_new_", "ptxv_main_"))
for f in glob.glob(f"{d}/prof/ef2_*"):
    os.rename(f, f.replace("/ef2_", "/dropped_small_"))
if os.path.exists(f"{d}/quick_ef2.txt"):
    os.rename(f"{d}/quick_ef2.txt", f"{d}/quick_dropped_small.txt")
print(sorted(os.listdir(f"{d}/prof")))
