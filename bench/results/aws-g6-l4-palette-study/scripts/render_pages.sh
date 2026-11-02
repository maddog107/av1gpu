#!/bin/bash
# Real screenshots: English Wikipedia articles rendered by headless Chrome at 1920x1080, device scale 1, light
# theme (2026-11-02, macOS, Google Chrome stable).  Text CC BY-SA 4.0; these articles' figures are free files from
# Wikimedia Commons.  Not reproducible byte for byte (the pages and fonts change), so the PNGs' SHA-256 are listed
# in the study's sets.json instead of the manifest.
#   render_pages.sh OUTDIR
OUT=${1:?outdir}
mkdir -p "$OUT"
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
for a in Huffman_coding Discrete_cosine_transform Color_quantization Run-length_encoding Arithmetic_coding; do
  "$CHROME" --headless=new --disable-gpu --hide-scrollbars --force-device-scale-factor=1 --window-size=1920,1080 \
    --screenshot="$OUT/screen_wikipedia_$(echo $a | tr 'A-Z-' 'a-z_').png" "https://en.wikipedia.org/wiki/$a?banner=none" 2>/dev/null
done
ls -la "$OUT"
