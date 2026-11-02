#!/usr/bin/env python3
"""derive.py RUN_SUMMARY_JSON [REPO]: every ratio the README / RESULTS.md quote, from the new av1gpu speeds and the
stored CPU / NVENC speeds (the method of RESULTS.md: speed at equal file size, interpolating SVT-AV1 presets
log-linearly in MP/s against BD-rate)."""
import json, math, os, sys
S = json.load(open(sys.argv[1]))
REPO = sys.argv[2] if len(sys.argv) > 2 else "."
RES = os.path.join(REPO, "bench", "results")
ld = lambda n: json.load(open(os.path.join(RES, n, "results.json")))  # noqa: E731
n = lambda k: S[k + "/new"]  # noqa: E731
o = lambda k: S[k + "/old"]  # noqa: E731
d4, d3, d2 = n("jpeg/size_range/default_enc4"), n("jpeg/size_range/default_enc3"), n("jpeg/size_range/default_enc2")
ns4, ns3 = n("jpeg/size_range/nosmall_enc4"), n("jpeg/size_range/nosmall_enc3")
t3ns, t3 = n("jpeg/size_range/turbo3ns_enc4"), n("jpeg/size_range/turbo3_enc4")
enc2 = ld("aws-g6-l4-gamut-enc2")["speed"]
s6 = next(x for x in enc2 if "speed 6" in x["codec"])["mp_s"]
s9 = next(x for x in enc2 if "speed 9" in x["codec"])["mp_s"]
nv = max(ld("aws-g6-l4-nvenc")["speed"], key=lambda x: x["mp_s"])["mp_s"]
svt = ld("aws-c6a-gamut-svt")
sv = {int(x["codec"].split()[2].rstrip(",")): x["mp_s"] for x in svt["speed"] if x["codec"].startswith("SVT")}
svb = {p: svt["bd_rate"][f"svt_p{p}_vs_aom_s6"]["median"] for p in sv}
t4 = ld("aws-g6-l4-tiles4x4")["bd_rate"]
bd = dict(default=t4["av1gpu_vs_aom_s6"]["median"], nosmall=t4["av1gpu_nosmall_vs_aom_s6"]["median"],
          turbo3=t4["av1gpu_turbo3_nosmall_vs_aom_s6"]["median"])


def svt_at(b):
    ps = sorted(sv, key=lambda p: svb[p])
    for a, c in zip(ps, ps[1:]):
        if svb[a] <= b <= svb[c]:
            f = (b - svb[a]) / (svb[c] - svb[a])
            return math.exp(math.log(sv[a]) + f * (math.log(sv[c]) - math.log(sv[a])))


P = print
P(f"libaom s6 {s6} s9 {s9} NVENC best {nv}; SVT {sv} BD {svb}; av1gpu BD {bd}")
P(f"size range: enc4 {d4['mp_s']:.1f} enc3 {d3['mp_s']:.1f} enc2 {d2['mp_s']:.1f} nosmall4 {ns4['mp_s']:.1f} nosmall3 {ns3['mp_s']:.1f} "
  f"turbo3ns {t3ns['mp_s']:.1f} turbo3 {t3['mp_s']:.1f}; cores {d4['cpu_cores']:.2f} {ns4['cpu_cores']:.2f} {t3ns['cpu_cores']:.2f}")
D = d4["mp_s"]
P(f"x libaom s6 (28.8 RESULTS / chart {s6}): {D / 28.8:.2f} / {D / s6:.2f}; tuned 29.7: {D / 29.7:.2f}; tuned 8x-j2 31.1: {D / 31.1:.2f}")
P(f"SVT untiled interp 33.3: {D / 33.3:.2f}; PGO 37.0: {D / 37.0:.2f}; tiled+PGO+8x-j2 34.5: {D / 34.5:.2f}")
P(f"x NVENC {nv}: {D / nv:.2f} (15.7: {D / 15.7:.2f})")
for k, m in (("default", D), ("nosmall", ns4["mp_s"]), ("turbo3", t3ns["mp_s"])):
    s = svt_at(bd[k])
    P(f"  {k}: BD {bd[k] * 100:+.2f} %, SVT at same size {s:.1f} MP/s -> {m / s:.2f}x")
P(f"nosmall vs preset 6 (49.4): {ns4['mp_s'] / 49.4:.2f}; turbo3ns vs p9 182.8: {t3ns['mp_s'] / 182.8:.2f}, vs libaom s9 187.6: {t3ns['mp_s'] / 187.6:.2f}")
P(f"turbo3ns / default: {t3ns['mp_s'] / D:.2f}; libaom s9 187.6 / default: {187.6 / D:.2f}")
P(f"CVVDP: SVT 73 -> {D / 73:.2f}x, libaom 49 -> {D / 49:.2f}x")
lib = {"420_8": 28.6, "444_8": 24.3, "420_10": 22.0, "444_10": 18.9}
fm = {"420_8": D, "444_8": n("formats/444_8")["mp_s"], "420_10": n("formats/420_10")["mp_s"], "444_10": n("formats/444_10")["mp_s"]}
P("formats:", {k: f"{fm[k]:.1f} ({fm[k] / lib[k]:.2f}x)" for k in lib},
  f"444 cost {1 - fm['444_8'] / D:.1%}, 10-bit cost {1 - fm['420_10'] / D:.1%} (444_10 vs 444_8 {1 - fm['444_10'] / fm['444_8']:.1%})")
l4, l4y = n("lossless/cmp_4x"), n("lossless/cmp_4x_ycgco")
l1, l1y = n("lossless/cmp_1pass"), n("lossless/cmp_1pass_ycgco")
P(f"lossless: 4x {l4['mp_s']:.1f} / ycgco {l4y['mp_s']:.1f}; 1 pass wall {l1['mp_s_wall']:.1f} / {l1y['mp_s_wall']:.1f}; "
  f"x avifenc -l 9.3: {l4['mp_s'] / 9.3:.2f}, ycgco x cicp 8.8: {l4y['mp_s'] / 8.8:.2f}")
P("alpha (wall):", {k: round(n(f"alpha/{k}")["mp_s_wall"], 1) for k in ("default", "q100", "lossless", "no_alpha")})
P("standard:", {k: round(n(f"jpeg/standard/{k}")["mp_s"], 1) for k in ("default_enc3", "default_enc2", "default_enc4", "nosmall_enc4", "nosmall_enc3")},
  "nosmall/default enc4", round(n("jpeg/standard/nosmall_enc4")["mp_s"] / n("jpeg/standard/default_enc4")["mp_s"], 2))
for k in ("thumbs/default", "thumbs/nosmall", "kodak/default", "kodak/nosmall"):
    if k + "/new" in S:
        P(k, round(n(k)["mp_s"], 1), "old", round(o(k)["mp_s"], 1), "wall", round(n(k)["mp_s_wall"], 1))
for k in ("jpeg/size_range/tiles211_enc3", "jpeg/size_range/tiles233_enc3"):
    if k + "/new" in S:
        P(k, round(n(k)["mp_s"], 1), "old", round(o(k)["mp_s"], 1), "vs default enc3", f"{n(k)['mp_s'] / d3['mp_s'] - 1:+.1%}")
P("J/MP:", {k: S.get(k, {}).get("j_per_mp") for k in ("jpeg/size_range/default_enc3/new", "jpeg/size_range/default_enc4/new",
                                               "jpeg/standard/default_enc3/new", "jpeg/size_range/default_enc3/old", "jpeg/standard/default_enc3/old")})
