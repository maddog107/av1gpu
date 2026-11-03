#!/bin/bash
# mk_mono.sh: ~/sets/mono = monochrome inputs: 40 of the size-range photos (every 5th from the 1st and 3rd) as gray
# JPEGs (ImageMagick -colorspace Gray, q90; 0.18-24 MP), the gray PNGs of ~/sets/gray without alpha, and ll/jpeg's
# gray JPEGs.  Prints the count.
M=~/sets/mono; rm -rf $M; mkdir -p $M
ls ~/images/*.jpg | awk 'NR % 5 == 1 || NR % 5 == 3' | xargs -P 16 -I{} sh -c 'convert "{}" -colorspace Gray -quality 90 '$M'/gray_$(basename "{}")'
for f in ~/ll/gray/*.png ~/ll/jpeg/*gray*.jpg; do ln -sf $(readlink -f $f) $M/; done
echo "mono $(ls $M | wc -l) files; $(identify -format '%[colorspace] %r\n' $M/gray_*.jpg 2>/dev/null | sort | uniq -c | tr '\n' ';')"
