#!/usr/bin/env python3
"""mk_tables.py RESULTS_JSON: the summary.md speed tables (v0.6 against v0.5.0), markdown on stdout."""
import json, sys
R = json.load(open(sys.argv[1]))
R = R.get("speed", R)


def g(k, b):
    return R.get(f"{k}/{b}")


def row(label, k, wall=False):
    n, o = g(k, "new"), g(k, "old")
    if not n:
        return None
    m = "mp_s_wall" if wall else "mp_s"
    rn = "rounds_wall" if wall else "rounds"
    ch = f"{(n[m] / o[m] - 1) * 100:+.1f} %" if o else ""
    w = f"{n['watts']:.1f}" if "watts" in n and not wall else ""
    clk = f"{n['sm_mhz']}" if "sm_mhz" in n and not wall else ""
    wall_cols = "" if wall else f" {n['mp_s_wall']:.1f} | {o['mp_s_wall']:.1f} |" if o else f" {n['mp_s_wall']:.1f} | |"
    return (f"| {label} | {n[m]:.1f} | {o[m]:.1f} | {ch} | {' / '.join(f'{x:.1f}' for x in n[rn])} | "
            f"{' / '.join(f'{x:.1f}' for x in o[rn])} |{wall_cols} {n['cpu_cores']:.2f} | {o['cpu_cores']:.2f} | {w} | {clk} |"
            if o else f"| {label} | {n[m]:.1f} | | | | |{wall_cols} {n['cpu_cores']:.2f} | | {w} | {clk} |")


HEAD = ("| setting | v0.6 MP/s | v0.5.0 MP/s | change | v0.6 rounds | v0.5.0 rounds | v0.6 wall MP/s | v0.5.0 wall MP/s "
        "| v0.6 CPU cores | v0.5.0 CPU cores | v0.6 W | v0.6 SM MHz |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
HEADW = ("| setting | v0.6 MP/s | v0.5.0 MP/s | change | v0.6 rounds | v0.5.0 rounds | v0.6 CPU cores | v0.5.0 CPU cores | | |\n"
         "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
T = [
    ("JPEG, size range", "jpeg/size_range", [
        ("default (3 encoders)", "default_enc3"), ("`GP_ENCODERS=2`", "default_enc2"), ("`GP_ENCODERS=4`", "default_enc4"),
        ("`--small-preset none`, 3 encoders", "nosmall_enc3"), ("`--small-preset none`, 4 encoders", "nosmall_enc4"),
        ("`--preset turbo3`, 4 encoders", "turbo3_enc4"), ("`--preset turbo3 --small-preset none`, 4 encoders", "turbo3ns_enc4"),
        ("`--tiles 2,1,1` (2 × 2 tiles), 3 encoders", "tiles211_enc3"), ("`--tiles 2,3,3` (8 × 8 tiles), 3 encoders", "tiles233_enc3")]),
    ("JPEG, standard", "jpeg/standard", [
        ("default (3 encoders)", "default_enc3"), ("`GP_ENCODERS=2`", "default_enc2"), ("`GP_ENCODERS=4`", "default_enc4"),
        ("`--small-preset none`, 3 encoders", "nosmall_enc3"), ("`--small-preset none`, 4 encoders", "nosmall_enc4")]),
    ("Thumbnails", "thumbs", [("default", "default"), ("`--small-preset none`", "nosmall")]),
    ("Kodak", "kodak", [("default", "default"), ("`--small-preset none`", "nosmall"),
                        ("`--preset turbo3 --small-preset none`", "turbo3_nosmall")]),
    ("Formats, size range, 4 encoders", "formats", [("8-bit 4:4:4", "444_8"), ("10-bit 4:2:0", "420_10"), ("10-bit 4:4:4", "444_10")]),
]
for title, pre, rows in T:
    print(f"\n## {title}\n\n{HEAD}")
    for label, k in rows:
        r = row(label, f"{pre}/{k}")
        if r:
            print(r)
for title, pre, rows in [
        ("Lossless (wall incl. start-up for one pass; av1gpu's MP/s for the set 4×)", "lossless",
         [("`--lossless`, one pass (wall)", "cmp_1pass", True), ("`--lossless --ycgco-re`, one pass (wall)", "cmp_1pass_ycgco", True),
          ("`--lossless`, set 4×", "cmp_4x", False), ("`--lossless --ycgco-re`, set 4×", "cmp_4x_ycgco", False)]),
        ("RGBA, 4 encoders, wall incl. start-up", "alpha",
         [("default", "default", True), ("`--alpha-quality 100`", "q100", True), ("`--lossless`", "lossless", True), ("`--no-alpha`", "no_alpha", True)])]:
    print(f"\n## {title}\n\n{HEADW}")
    for label, k, wall in rows:
        n, o = g(f"{pre}/{k}", "new"), g(f"{pre}/{k}", "old")
        if not n:
            continue
        m, rn = ("mp_s_wall", "rounds_wall") if wall else ("mp_s", "rounds")
        print(f"| {label} | {n[m]:.1f} | {o[m]:.1f} | {(n[m] / o[m] - 1) * 100:+.1f} % | {' / '.join(f'{x:.1f}' for x in n[rn])} | "
              f"{' / '.join(f'{x:.1f}' for x in o[rn])} | {n['cpu_cores']:.2f} | {o['cpu_cores']:.2f} | | |")

# Kodak in images per second (the FPGA comparison): both binaries, both rounds, wall-clock incl. start-up
print("\n## Kodak, images per second\n")
print("| setting | v0.6 img/s | v0.5.0 img/s | v0.6 rounds | v0.6 wall img/s | v0.5.0 wall img/s | v0.6 MP/s | v0.5.0 MP/s "
      "| v0.6 CPU cores | v0.5.0 CPU cores |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
for label, k in (("default", "default"), ("`--small-preset none`", "nosmall"),
                 ("`--preset turbo3 --small-preset none`", "turbo3_nosmall")):
    n, o = g(f"kodak/{k}", "new"), g(f"kodak/{k}", "old")
    if not n:
        continue
    print(f"| {label} | {n['img_s']:.1f} | {o['img_s']:.1f} | {' / '.join(f'{x:.1f}' for x in n['rounds_img'])} | "
          f"{n['img_s_wall']:.1f} | {o['img_s_wall']:.1f} | {n['mp_s']:.1f} | {o['mp_s']:.1f} | {n['cpu_cores']:.2f} | {o['cpu_cores']:.2f} |")
