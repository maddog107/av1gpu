#!/bin/sh
# Convert every JPEG/PNG/WebP in a directory to AVIF on the GPU; inputs the GPU path does not handle
# (CMYK JPEGs, images over 35.6 MP, ...) are encoded with avifenc on the CPU instead.
#
#   examples/encode_dir.sh SRC_DIR OUT_DIR [QUALITY]
#
# Outputs are named OUT_DIR/<input file name>.avif (photo.jpg -> photo.jpg.avif), which suits serving
# them next to the originals (see nginx-avif.conf).  QUALITY uses the avifenc -q scale (default 60).
set -eu
src=$1
out=$2
q=${3:-60}
av1gpu=${AV1GPU:-av1gpu}

mkdir -p "$out"
"$av1gpu" --batch "$src" --outdir "$out" --quality "$q" 2> "$out/av1gpu.log" || {
    cat "$out/av1gpu.log" >&2
    exit 1
}
grep 'MP/s' "$out/av1gpu.log" >&2 || true

# skipped inputs are reported as "[av1gpu] skipped <path>: <reason>"
grep '^\[av1gpu\] skipped ' "$out/av1gpu.log" | sed 's/^\[av1gpu\] skipped \(.*\): [^:]*$/\1/' |
while IFS= read -r f; do
    echo "CPU fallback: $f" >&2
    # with libaom >= 3.12, add -a tune=iq to match av1gpu's tuning
    avifenc -q "$q" -s 6 "$f" "$out/$(basename "$f").avif" > /dev/null || echo "failed: $f" >&2
done
