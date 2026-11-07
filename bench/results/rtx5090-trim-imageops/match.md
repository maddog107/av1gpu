| image | ImageMagick | size | planes: mean abs. difference (8-bit steps), PSNR, max |
|---|---|---|---|
| 211x149 PNG | -resize 150% | 317x224 | Y mad 0.231 psnr 54.5 max 1; U mad 0.221 psnr 54.7 max 1; V mad 0.244 psnr 54.3 max 1 |
| 211x149 PNG | -resize 50% | 106x75 | Y mad 0.223 psnr 54.6 max 1; U mad 0.205 psnr 55.0 max 1; V mad 0.235 psnr 54.4 max 1 |
| 211x149 PNG | -resize 300x300^ | 425x300 | Y mad 0.231 psnr 54.5 max 1; U mad 0.218 psnr 54.7 max 1; V mad 0.245 psnr 54.2 max 1 |
| 211x149 PNG | -resize 100x100! | 100x100 | Y mad 0.223 psnr 54.7 max 1; U mad 0.207 psnr 55.0 max 1; V mad 0.240 psnr 54.3 max 1 |
| 211x149 PNG | -colorspace Gray | 211x149 mono | Y mad 0.498 psnr 50.9 max 2 |
| 211x149 PNG | -grayscale Rec601Luma | 211x149 mono | Y mad 0.468 psnr 51.4 max 1 |
| 211x149 PNG | -grayscale Average | 211x149 mono | Y mad 0.295 psnr 53.4 max 1 |
| 211x149 PNG | -grayscale Brightness | 211x149 mono | Y mad 0.332 psnr 52.9 max 1 |
| 211x149 PNG | -grayscale Lightness | 211x149 mono | Y mad 0.281 psnr 53.6 max 1 |
| 211x149 PNG | -grayscale RMS | 211x149 mono | Y mad 0.504 psnr 51.1 max 2 |
| 211x149 PNG | -level 10%,90% | 211x149 | Y mad 0.198 psnr 55.2 max 1; U mad 0.328 psnr 53.0 max 1; V mad 0.261 psnr 54.0 max 1 |
| 211x149 PNG | -level 20%,80%,0.5 | 211x149 | Y mad 0.007 psnr 69.8 max 1; U mad 0.004 psnr 72.2 max 1; V mad 0.016 psnr 66.0 max 1 |
| 211x149 PNG | -gamma 0.7 | 211x149 | Y mad 0.296 psnr 53.4 max 1; U mad 0.184 psnr 55.5 max 1; V mad 0.202 psnr 55.1 max 1 |
| 211x149 PNG | -gamma 2 | 211x149 | Y mad 1.831 psnr 41.1 max 5; U mad 1.066 psnr 45.6 max 3; V mad 1.325 psnr 43.8 max 4 |
| 211x149 PNG | -negate | 211x149 | Y mad 0.000 psnr 99.0 max 0; U mad 0.000 psnr 99.0 max 0; V mad 0.000 psnr 99.0 max 0 |
| 211x149 PNG | -auto-level | 211x149 | Y mad 2.216 psnr 40.4 max 5; U mad 0.799 psnr 47.9 max 2; V mad 0.803 psnr 47.4 max 2 |
| 211x149 PNG | -level 10%,90% -gamma 1.5 -negate | 211x149 | Y mad 0.447 psnr 51.6 max 1; U mad 0.481 psnr 49.8 max 3; V mad 0.374 psnr 52.2 max 2 |
| 1050x1500 page PNG | -resize 150% | 1575x2250 | Y mad 0.280 psnr 53.4 max 11; U mad 0.078 psnr 59.1 max 6; V mad 0.093 psnr 58.2 max 6 |

| 1050x1500 page PNG | -resize x2960 | 2072x2960 | Y mad 0.283 psnr 53.4 max 9; U mad 0.079 psnr 59.0 max 6; V mad 0.096 psnr 58.1 max 6 |

| 1050x1500 page PNG | -colorspace Gray | 1050x1500 mono | Y mad 0.215 psnr 54.8 max 2 |

| 1050x1500 page PNG | -level 10%,90% | 1050x1500 | Y mad 0.177 psnr 55.5 max 2; U mad 0.172 psnr 55.7 max 2; V mad 0.086 psnr 58.8 max 2 |

| 1050x1500 page PNG | -gamma 2 | 1050x1500 | Y mad 0.434 psnr 47.9 max 10; U mad 0.339 psnr 47.7 max 9; V mad 0.314 psnr 48.4 max 9 |

| 1050x1500 page PNG | -gamma 0.7 | 1050x1500 | Y mad 0.253 psnr 54.0 max 2; U mad 0.151 psnr 56.3 max 2; V mad 0.126 psnr 57.1 max 2 |

| 1050x1500 page PNG | -auto-level | 1050x1500 | Y mad 0.000 psnr 99.0 max 0; U mad 0.000 psnr 99.0 max 0; V mad 0.000 psnr 99.0 max 0 |


16-bit PNG, 10-bit planes (10-bit steps):

-gamma 2                                     Y mad 1.866 max 4 | U mad 1.024 max 3 | V mad 1.218 max 3
-gamma 0.6                                   Y mad 0.063 max 1 | U mad 0.090 max 1 | V mad 0.104 max 1
-level 10%,90%                               Y mad 0.111 max 1 | U mad 0.130 max 1 | V mad 0.169 max 1
-level 10%,90%,0.7                           Y mad 0.031 max 1 | U mad 0.145 max 1 | V mad 0.036 max 1
-auto-level                                  Y mad 1.790 max 4 | U mad 0.924 max 2 | V mad 1.066 max 3
-level 5%,95% -gamma 1.2 -negate             Y mad 0.235 max 1 | U mad 0.266 max 1 | V mad 0.223 max 2
