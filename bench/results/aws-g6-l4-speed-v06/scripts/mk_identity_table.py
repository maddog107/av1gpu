#!/usr/bin/env python3
"""mk_identity_table.py IDENTITY_TXT: markdown table of every identity set (inputs, files per binary, differences)."""
import re, sys
SRC = {"sr": "size range (100 JPEGs)", "std": "standard (100 JPEGs)", "thumbs": "thumbnails (280 JPEGs)",
       "mono": "monochrome (47: 40 gray JPEGs of the size range, gray PNGs and JPEGs)", "kodak": "Kodak (24 PNGs)",
       "rgba": "RGBA (39 PNGs)", "gray": "gray and gray + alpha PNGs (9)", "odd": "odd sizes (30: PNG, 4:2:0 and 4:4:4 JPEG)",
       "misc": "ll/misc (13: palette, RGBA, odd, 4100 × 40, 6144 × 4096)", "png16": "16-bit PNGs with 10 bits (8)",
       "llgraphics": "graphics PNGs (23)", "webp": "WebP (7: lossless, lossy, with alpha)", "mixed": "mixed `--batch` LIST (33)",
       "lljpeg": "ll/jpeg (25: 4:2:0, 4:4:4, gray, 4:2:2, odd)", "single": "single-file mode, no `--batch` (10 files)"}
rows, tot = [], [0, 0, 0]
for l in open(sys.argv[1]):
    m = re.match(r"(\w+): inputs (\d+), files (\d+) / (\d+), differ (\d+)(?:, skipped ([^,]+))?.*opts \[([^\]]*)\]\s*(GP_ENCODERS=\d+)?", l)
    if not m:
        continue
    name, nin, a, b, d, sk, opts, enc = m.groups()
    key = name.split("_")[0]
    opts = (opts.strip() or "default") + (f", `{enc}`" if enc else "")
    if not opts.startswith("default"):
        opts = f"`{opts}`"
    rows.append((SRC.get(key, key), opts, nin, a, b, d, sk or ""))
    tot = [tot[0] + int(a), tot[1] + int(b), tot[2] + int(d)]
print("| set | options | inputs | v0.5.0 files | v0.6 files | differ | skipped (both) |\n|---|---|---:|---:|---:|---:|---:|")
for r in rows:
    print("| " + " | ".join(r) + " |")
print(f"| **all ({len(rows)} runs: {sum(1 for r in rows if not r[0].startswith('single'))} batch, {sum(1 for r in rows if r[0].startswith('single'))} single-file)** | | | **{tot[0]:,}** | **{tot[1]:,}** | **{tot[2]}** | |")
