| case | source | output | ImageMagick | Y MAD / PSNR | U MAD / PSNR | V MAD / PSNR | alpha MAD / PSNR |
|---|---|---|---|---:|---:|---:|---:|
| crop_only_444_10 | 25d19e868909e5e2_huge.jpg | 1500x1000 | 1500x1000 | 0.727 / 60.54 dB | 0.659 / 61.85 dB | 0.526 / 62.5 dB |  |
| resize_only_444_10 | 25d19e868909e5e2_huge.jpg | 3008x2000 | 3008x2000 | 0.521 / 62.42 dB | 0.5 / 62.58 dB | 0.461 / 62.82 dB |  |
| page_crop_only_444_10 | page_3840x5586.jpg | 3000x4000 | 3000x4000 | 0.65 / 61.06 dB | 0.631 / 62.11 dB | 0.517 / 62.35 dB |  |
| page_max_height_2960_444_10 | page_3840x5586.jpg | 2035x2960 | 2035x2960 | 0.499 / 62.64 dB | 0.498 / 62.52 dB | 0.447 / 63.07 dB |  |
| photo_png16_resize_444_10 | photo16.png | 3008x2000 | 3008x2000 | 0.191 / 67.15 dB | 0.161 / 67.08 dB | 0.164 / 67.29 dB |  |
| photo_png16_odd_crop_resize_420_10 | photo16.png | 1000x750 | 1000x750 | 0.189 / 66.96 dB | 0.455 / 61.71 dB | 0.428 / 62.19 dB |  |
| odd_crop_resize_420_8 | 25d19e868909e5e2_huge.jpg | 1000x750 | 1000x750 | 0.1 / 58.09 dB | 0.184 / 55.37 dB | 0.158 / 56.06 dB |  |
| odd_crop_resize_420_10 | 25d19e868909e5e2_huge.jpg | 1000x750 | 1000x750 | 0.528 / 62.23 dB | 0.682 / 60.06 dB | 0.6 / 60.66 dB |  |
| png8_resize_420_8 | kodim23.png | 500x333 | 500x333 | 0.173 / 55.68 dB | 0.278 / 50.75 dB | 0.272 / 52.3 dB |  |
| png8_odd_crop_resize_420_8 | kodim23.png | 333x222 | 333x222 | 0.177 / 55.54 dB | 0.269 / 53.39 dB | 0.288 / 52.58 dB |  |
| png16_crop_resize_444_10 | k2_10in16.png | 300x198 | 300x198 | 0.201 / 62.72 dB | 0.187 / 65.04 dB | 0.179 / 64.61 dB |  |
| rgba_resize_444_8 | ui_hud_1080p.png | 960x540 | 960x540 | 0.056 / 55.4 dB | 0.023 / 56.26 dB | 0.028 / 52.29 dB | 0.0 / 99.0 dB |
| rgba_crop_resize_420_8 | ui_hud_1080p.png | 640x359 | 640x359 | 0.008 / 52.54 dB | 0.009 / 63.67 dB | 0.269 / 48.4 dB | 0.0 / 99.0 dB |
| exif6_resize_420_8 | rot6.jpg | 1000x666 | 1000x666 | 0.199 / 54.66 dB | 0.587 / 47.62 dB | 0.526 / 48.78 dB |  |
