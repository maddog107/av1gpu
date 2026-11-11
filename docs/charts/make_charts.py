#!/usr/bin/env python3
"""Draw the README charts from the NVIDIA L4 results in bench/results/ (needs numpy, like bench/bench.py).

  docs/charts/make_charts.py

av1gpu's speeds and file sizes come from bench/results/aws-g6-l4-v0130 (v0.13.0: its scripts/analyze.py writes
results.json); the CPU and NVENC baselines stay in their own result directories.

Writes <chart>-light.svg and <chart>-dark.svg next to this script; the README picks one with <picture>.
"""
import csv
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
RES = os.path.join(ROOT, "bench", "results")
sys.path.insert(0, os.path.join(ROOT, "bench"))
from bench import SIZE_CLASSES, bd_rate  # noqa: E402

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781", grid="#e1e0d9",
                  axis="#c3c2b7", border="rgba(11,11,11,0.10)",
                  av1gpu="#2a78d6", aom_s6="#eb6834", aom_s9="#1baf7a", webp="#eda100", nvenc="#e87ba4", svt="#008300"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", border="rgba(255,255,255,0.10)",
                 av1gpu="#3987e5", aom_s6="#d95926", aom_s9="#199e70", webp="#c98500", nvenc="#d55181", svt="#008300"),
}
FONT = "system-ui,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"
NAMES = dict(av1gpu="av1gpu", aom_s6="libaom speed 6", aom_s9="libaom speed 9", webp="WebP", nvenc="NVENC AV1", svt="SVT-AV1")


def load(name):
    return json.load(open(os.path.join(RES, name, "results.json")))


AV1GPU = "aws-g6-l4-v0130"   # av1gpu's speeds and file sizes (v0.13.0)


def av1gpu_l4():
    """av1gpu v0.13.0 on the L4 (bench/results/aws-g6-l4-v0130): speeds in "speed" (the keys of aws-g6-g7-v090-check),
    its files re-encoded and scored against the stored CPU baselines in "quality" / "bd_rate" / ... (the keys of
    aws-g6-l4-tiles4x4)"""
    p = os.path.join(RES, AV1GPU, "results.json")
    if not os.path.exists(p):
        sys.exit(f"make_charts.py: {p} is missing: run the v0.13.0 benchmark ({AV1GPU}/scripts: setup.sh, run.sh on the "
                 "instance, then analyze.py on the pulled results) before drawing the charts")
    return json.load(open(p))


def need(r, *keys):
    """r[k0][k1]...: or a clear failure naming what the v0.13.0 results lack (a part of run.sh that was skipped)"""
    for i, k in enumerate(keys):
        if not isinstance(r, dict) or k not in r:
            sys.exit(f"make_charts.py: {AV1GPU}/results.json has no {'/'.join(keys[:i + 1])} (was that part of run.sh "
                     "skipped? see its not_measured list)")
        r = r[k]
    return r


def nvenc():
    """NVENC AV1 on the same L4 (bench/results/aws-g6-l4-nvenc): best speed, per-image scores, BD-rate vs speed 6."""
    r = load("aws-g6-l4-nvenc")
    best = max(r["speed"], key=lambda x: x["mp_s"])
    return r, best


def tiles4x4():
    """av1gpu's file sizes: v0.13.0's files re-encoded on the L4 and scored against the stored CPU baselines
    (bench/results/aws-g6-l4-v0130; until v0.12.0 they came from aws-g6-l4-tiles4x4, same keys)"""
    return av1gpu_l4()


def svt():
    """SVT-AV1 presets on the same 16-vCPU EPYC 7R13 (bench/results/aws-c6a-gamut-svt): {preset: (MP/s, BD vs s6)}.

    That run's libaom speed 6 encodes and scores are identical to the L4 run's, so the BD-rates compare directly."""
    r = load("aws-c6a-gamut-svt")
    mps = {int(x["codec"].split()[2].rstrip(",")): x["mp_s"] for x in r["speed"] if x["codec"].startswith("SVT")}
    return {p: (v, r["bd_rate"][f"svt_p{p}_vs_aom_s6"]["median"]) for p, v in mps.items()}


def esc(s):
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


class Svg:
    def __init__(self, w, h, t, title):
        self.w, self.h, self.t, self.out = w, h, t, []
        self.title = title

    def add(self, s):
        self.out.append(s)

    def text(self, x, y, s, size=13, fill="ink", anchor="start", weight=400):
        self.add(f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{self.t[fill]}" '
                 f'text-anchor="{anchor}" font-weight="{weight}">{esc(s)}</text>')

    def line(self, x1, y1, x2, y2, stroke="grid", width=1):
        self.add(f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                 f'stroke="{self.t[stroke]}" stroke-width="{width}"/>')

    def bar_h(self, x0, x1, y, h, color, r=4):
        """Horizontal bar from baseline x0 to x1, rounded at the data end only."""
        r = min(r, (x1 - x0) / 2, h / 2)
        self.add(f'<path d="M{x0:.1f},{y:.1f} H{x1 - r:.1f} Q{x1:.1f},{y:.1f} {x1:.1f},{y + r:.1f} '
                 f'V{y + h - r:.1f} Q{x1:.1f},{y + h:.1f} {x1 - r:.1f},{y + h:.1f} H{x0:.1f} Z" fill="{color}"/>')

    def bar_v(self, x, w, y0, y1, color, r=4):
        """Vertical bar from baseline y0 to data end y1 (above or below the baseline)."""
        r = min(r, w / 2, abs(y1 - y0) / 2)
        s = -1 if y1 < y0 else 1         # direction from baseline to data end
        self.add(f'<path d="M{x:.1f},{y0:.1f} V{y1 - s * r:.1f} Q{x:.1f},{y1:.1f} {x + r:.1f},{y1:.1f} '
                 f'H{x + w - r:.1f} Q{x + w:.1f},{y1:.1f} {x + w:.1f},{y1 - s * r:.1f} V{y0:.1f} Z" fill="{color}"/>')

    def header(self, title, subtitle):
        self.text(24, 34, title, 17, weight=600)
        for i, s in enumerate(subtitle):
            self.text(24, 56 + 18 * i, s, 13, "ink2")

    def legend(self, x, y, items):
        for key, label in items:
            self.add(f'<rect x="{x}" y="{y - 9}" width="10" height="10" rx="2" fill="{self.t[key]}"/>')
            self.text(x + 16, y, label, 13, "ink2")
            x += 16 + 7.4 * len(label) + 22

    def save(self, path):
        body = "\n".join(self.out)
        open(path, "w").write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
            f'viewBox="0 0 {self.w} {self.h}" font-family="{FONT}" role="img" aria-label="{esc(self.title)}">\n'
            f'<title>{esc(self.title)}</title>\n'
            f'<rect x="0.5" y="0.5" width="{self.w - 1}" height="{self.h - 1}" rx="12" '
            f'fill="{self.t["surface"]}" stroke="{self.t["border"]}"/>\n{body}\n</svg>\n')


def pct(bd):
    """a median BD-rate as the docs print it: +0.5 %"""
    return f"{bd * 100:+.1f} %".replace("-", "−")


def span(a, b):
    """'2.2–2.5', or one figure when both round alike"""
    lo, hi = f"{min(a, b):.1f}", f"{max(a, b):.1f}"
    return lo if lo == hi else f"{lo}–{hi}"


def files(bd):
    return f"files {bd * 100:+.1f} % vs speed 6".replace("-", "−")


def l4_speed(setting):
    """av1gpu's speed with v0.13.0 on the L4, as (MP/s, CPU cores): the median of 6 runs
    (bench/results/aws-g6-l4-v0130). setting: e.g. "jpeg/size_range/default_enc4", "formats/444_10"."""
    r = need(av1gpu_l4(), "speed", setting + "/new")
    return r["mp_s"], r["cpu_cores"]


def matchup_chart(t, name, title, subtitle, rows, sizes=True, speed=(250, 50), size=(16, 5)):
    """Encoders side by side: speed (higher is better) and, if sizes, file size (lower is better) beside it.

    speed and size are (axis maximum, tick step)."""
    lx, top, band = 24, 100 + 18 * len(subtitle), 56
    s = Svg(760, top + 20 + band * len(rows), t, title)
    s.header(title, subtitle)
    panels = [(250, 470, *speed[:1], "Speed, MP/s  (higher is better)", speed[1]),
              (520, 700, *size[:1], "Files vs libaom speed 6  (lower is better)", size[1])]
    if not sizes:
        panels = [(250, 680, *speed[:1], "Speed, MP/s  (higher is better)", speed[1])]
    for i, (x0, x1, vmax, head, step) in enumerate(panels):
        s.text(x0, top - 22, head, 12, "ink2", weight=600)
        for v in range(0, vmax + 1, step):
            x = x0 + (x1 - x0) * v / vmax
            s.line(x, top - 6, x, top + band * len(rows) - 14)
            s.text(x, top + band * len(rows) + 4, f"{v}" + (" %" if i and v else ""), 12, "muted", "middle")
        s.line(x0, top - 6, x0, top + band * len(rows) - 14, "axis")
    for i, (label, note, key, mps, bd) in enumerate(rows):
        y = top + band * i
        s.text(lx, y + 13, label, 14, weight=600 if key == "av1gpu" else 400)
        s.text(lx, y + 31, note, 12, "muted")
        x0, x1, vmax = panels[0][:3]
        xe = x0 + (x1 - x0) * mps / vmax
        s.bar_h(x0, xe, y + 4, 24, t[key])
        s.text(xe + 8, y + 21, f"{mps:.0f}", 14, weight=600)
        if not sizes:
            continue
        x0, x1, vmax = panels[1][:3]
        xe = x0 + (x1 - x0) * max(bd, 0) * 100 / vmax
        if xe - x0 > 1:
            s.bar_h(x0, xe, y + 4, 24, t[key])
        s.text(xe + 8, y + 21, "reference" if key == "aom_s6" else f"{bd * 100:+.1f} %".replace("-", "−"), 14,
               weight=600)
    s.save(os.path.join(HERE, f"{name}-{t['name']}.svg"))


def matchup_charts(t):
    gamut = load("aws-g6-l4-gamut")["bd_rate"]
    t4 = tiles4x4()["bd_rate"]
    t3 = t4["av1gpu_turbo3_nosmall_vs_aom_s6"]["median"]
    enc2 = load("aws-g6-l4-gamut-enc2")["speed"]
    s6 = next(x for x in enc2 if "speed 6" in x["codec"])["mp_s"]
    s9 = next(x for x in enc2 if "speed 9" in x["codec"])["mp_s"]
    default, cores = l4_speed("jpeg/size_range/default_enc4")
    turbo3 = l4_speed("jpeg/size_range/turbo3ns_enc4")[0]
    nv, nv_best = nvenc()
    nv_bd = nv["bd_rate"]["nvenc_vs_aom_s6"]["median"]
    sv = svt()
    cpu = "16 CPU threads at once"
    svt_note = lambda p: f"16 CPU threads; files {sv[p][1] * 100:+.1f} %".replace("-", "−")
    matchup_chart(t, "match-small", "Smallest files: av1gpu default vs libaom speed 6 and SVT-AV1",
                  ["100 photos of 0.18–24 MP on an AWS g6.4xlarge: the CPU encoders on all 16 vCPUs, av1gpu",
                   f"on one NVIDIA L4 and about {cores:.1f} CPU cores. Files {pct(t4['av1gpu_vs_aom_s6']['median'])} against "
                   f"libaom speed 6, {default / s6:.1f}× the speed;",
                   "SVT-AV1 reaches those sizes between its presets 4 and 6."],
                  [("av1gpu default", "1 GPU, 4 encoders", "av1gpu", default,
                    t4["av1gpu_vs_aom_s6"]["median"]),
                   ("libaom speed 6", cpu, "aom_s6", s6, 0.0),
                   ("SVT-AV1 preset 6", svt_note(6), "svt", sv[6][0], sv[6][1]),
                   ("SVT-AV1 preset 4", svt_note(4), "svt", sv[4][0], sv[4][1]),
                   ("NVENC AV1, p7", f"same L4; files {nv_bd * 100:+.0f} %", "nvenc", nv_best["mp_s"], nv_bd)],
                  sizes=False, speed=(400, 50))
    matchup_chart(t, "match-nvenc", "Same GPU: av1gpu vs the NVENC AV1 hardware encoder",
                  ["The same photos through ffmpeg av1_nvenc (preset p7, tune uhq) on the same L4. NVENC is",
                   f"one fixed-function block; av1gpu runs on the CUDA cores, {default / nv_best['mp_s']:.0f}× faster "
                   "with much smaller files."],
                  [("av1gpu default", "1 GPU, 4 encoders", "av1gpu", default,
                    t4["av1gpu_vs_aom_s6"]["median"]),
                   ("NVENC AV1, p7", f"{nv_best['workers']} at once, {nv_best['cpu_cores']:.0f} CPU cores", "nvenc", nv_best["mp_s"], nv_bd)],
                  speed=(400, 50), size=(30, 10))
    matchup_chart(t, "match-fast", "Fastest: av1gpu turbo3 vs SVT-AV1 preset 9 and libaom speed 9",
                  ["The same photos at each encoder's fastest useful setting. av1gpu is "
                   f"{span(turbo3 / sv[9][0], turbo3 / s9)}× as fast,",
                   f"and its files grow {t3 * 100:.0f} % where SVT-AV1's grow {sv[9][1] * 100:.0f} % and libaom's "
                   f"{gamut['aom_s9_vs_aom_s6']['median'] * 100:.0f} %."],
                  [("av1gpu turbo3", "--small-preset none", "av1gpu", turbo3, t3),
                   ("SVT-AV1 preset 9", cpu, "svt", sv[9][0], sv[9][1]),
                   ("libaom speed 9", cpu, "aom_s9", s9, gamut["aom_s9_vs_aom_s6"]["median"])],
                  speed=(500, 100))


def tradeoff_chart(t):
    """Every measured setting: file size against libaom speed 6 (x) and speed (y)."""
    gamut = load("aws-g6-l4-gamut")["bd_rate"]
    enc2 = load("aws-g6-l4-gamut-enc2")["speed"]
    t4 = tiles4x4()["bd_rate"]
    default = l4_speed("jpeg/size_range/default_enc4")[0]
    s6 = next(x for x in enc2 if "speed 6" in x["codec"])["mp_s"]
    nv, nv_best = nvenc()
    sv = svt()
    series = [  # key, legend, marker, dashed, points (BD, MP/s, label, label anchor)
        ("av1gpu", "av1gpu (one L4)", "circle", False,
         [(t4["av1gpu_vs_aom_s6"]["median"], default, "default", "w"),
          (t4["av1gpu_nosmall_vs_aom_s6"]["median"], l4_speed("jpeg/size_range/nosmall_enc4")[0], "no small preset", "nw"),
          (t4["av1gpu_turbo3_nosmall_vs_aom_s6"]["median"], l4_speed("jpeg/size_range/turbo3ns_enc4")[0], "turbo3", "n")]),
        ("svt", "SVT-AV1 (16 threads)", "triangle", True,
         [(sv[p][1], sv[p][0], f"preset {p}", {2: "n", 4: "n", 8: "w", 9: "n"}.get(p, "e")) for p in sorted(sv)]),
        ("aom_s6", "libaom (16 threads)", "square", False,
         [(0.0, next(x for x in enc2 if "speed 6" in x["codec"])["mp_s"], "speed 6", "e"),
          (gamut["aom_s9_vs_aom_s6"]["median"], next(x for x in enc2 if "speed 9" in x["codec"])["mp_s"], "speed 9", "e")]),
        ("nvenc", "NVENC AV1 (same L4)", "diamond", False,
         [(nv["bd_rate"]["nvenc_vs_aom_s6"]["median"], nv_best["mp_s"], "p7, uhq", "n")]),
    ]
    s = Svg(760, 590, t, "Speed against file size: av1gpu, SVT-AV1, libaom, NVENC")
    s.header("Speed against file size",
             ["Each point is one setting. Up is faster, left is smaller files at equal quality (median BD-rate",
              "on SSIMULACRA2 against libaom speed 6). The CPU encoders use all 16 threads of an EPYC 7R13."])
    lx = 24
    for key, label, mk, dashed, pts in series:
        if len(pts) > 1:
            s.add(f'<line x1="{lx}" y1="{106}" x2="{lx + 22}" y2="{106}" stroke="{t[key]}" stroke-width="2"'
                  + (' stroke-dasharray="5 3"' if dashed else "") + '/>')
        marker(s, t, mk, lx + 11, 106, t[key])
        s.text(lx + 30, 110, label, 13, "ink2")
        lx += 30 + 7.0 * len(label) + 22
    L, R, T, B = 70, 730, 136, 530
    xmin, xmax, ymin, ymax = -6, 30, 0, 500
    X = lambda v: L + (R - L) * (v * 100 - xmin) / (xmax - xmin)
    Y = lambda v: B - (B - T) * (v - ymin) / (ymax - ymin)
    for v in range(ymin, ymax + 1, 50):
        s.line(L, Y(v), R, Y(v), "axis" if v == 0 else "grid")
        s.text(L - 8, Y(v) + 4, str(v), 12, "muted", "end")
    for v in range(-5, xmax + 1, 5):
        s.line(X(v / 100), T, X(v / 100), B, "axis" if v == 0 else "grid")
        s.text(X(v / 100), B + 18, (f"{v:+d} %" if v else "0").replace("-", "−"), 12, "muted", "middle")
    s.text((L + R) / 2, B + 38, "files vs libaom speed 6 (smaller is better)", 12, "muted", "middle")
    s.add(f'<text x="18" y="{(T + B) / 2:.1f}" font-size="12" fill="{t["muted"]}" text-anchor="middle" '
          f'transform="rotate(-90 18 {(T + B) / 2:.1f})">MP/s (faster is higher)</text>')
    # the headline gap: av1gpu default against libaom speed 6, at the same file size
    ax, lo, hi = X(0) - 14, Y(s6) - 10, Y(default) + 10
    s.add(f'<line x1="{ax:.1f}" y1="{lo:.1f}" x2="{ax:.1f}" y2="{hi + 6:.1f}" stroke="{t["ink2"]}" stroke-width="1.5"/>')
    s.add(f'<path d="M{ax - 5:.1f},{hi + 7:.1f} L{ax:.1f},{hi:.1f} L{ax + 5:.1f},{hi + 7:.1f}" fill="none" '
          f'stroke="{t["ink2"]}" stroke-width="1.5" stroke-linejoin="round"/>')
    s.text(ax - 8, (lo + hi) / 2 + 2, f"{default / s6:.1f}×", 20, "ink", "end", 700)
    s.text(ax - 8, (lo + hi) / 2 + 18, "libaom speed 6's speed,", 12, "ink2", "end")
    s.text(ax - 8, (lo + hi) / 2 + 33, f"files {pct(t4['av1gpu_vs_aom_s6']['median'])}", 12, "ink2", "end")
    off = dict(n=(0, -12, "middle"), s=(0, 20, "middle"), e=(10, 4, "start"), w=(-10, 4, "end"), se=(8, 18, "start"), nw=(-6, -10, "end"))
    for key, _, mk, dashed, pts in reversed(series):
        pts = sorted(pts)
        if len(pts) > 1:
            d = " ".join(f"{'M' if i == 0 else 'L'}{X(b):.1f},{Y(m):.1f}" for i, (b, m, _, _) in enumerate(pts))
            s.add(f'<path d="{d}" fill="none" stroke="{t[key]}" stroke-width="2" stroke-linejoin="round"'
                  + (' stroke-dasharray="5 3"' if dashed else "") + '/>')
        for b, m, label, anc in pts:
            marker(s, t, mk, X(b), Y(m), t[key])
            dx, dy, ta = off[anc]
            s.text(X(b) + dx, Y(m) + dy, label, 12, "ink2", ta)
    s.save(os.path.join(HERE, f"tradeoff-{t['name']}.svg"))


def formats_chart(t):
    """av1gpu against libaom speed 6 in each output format (bench/results/aws-g6-l4-formats)"""
    r = load("aws-g6-l4-formats")
    t4 = tiles4x4()["bd_rate"]
    names = {"420_8": "8-bit 4:2:0", "444_8": "8-bit 4:4:4", "420_10": "10-bit 4:2:0", "444_10": "10-bit 4:4:4"}
    sp = {(x["encoder"], x["format"]): x["mp_s"] for x in r["speed"] if x["encoder"] != "av1gpu"}   # libaom
    sp[("av1gpu", "420_8")] = l4_speed("jpeg/size_range/default_enc4")[0]
    for f in ("444_8", "420_10", "444_10"):
        sp[("av1gpu", f)] = l4_speed(f"formats/{f}")[0]
    rows = []
    within = max(abs(t4[f"av1gpu_{f}_vs_aom_s6_{f}"]["median"]) for f in names) * 100
    within = round(within + 0.049, 1)   # rounded up
    for f, name in names.items():
        b = t4[f"av1gpu_{f}_vs_aom_s6_{f}"]["median"]
        rows.append((f"av1gpu, {name}", f"one L4; files {b * 100:+.1f} % vs libaom".replace("-", "−"), "av1gpu",
                     sp[("av1gpu", f)], 0))
        rows.append((f"libaom speed 6, {name}", "16 CPU threads", "aom_s6", sp[("aom_s6", f)], 0))
    matchup_chart(t, "formats", "Every output format: av1gpu vs libaom speed 6",
                  ["Speed at quality 60 on the 100 size-range photos. In each format av1gpu's files are within",
                   f"{within:.1f} % of libaom speed 6's in the same format (median BD-rate, SSIMULACRA2)."],
                  rows, sizes=False, speed=(400, 50))


def metrics_chart(t):
    """median BD-rate vs libaom speed 6 on SSIMULACRA2 and on CVVDP (bench/results/aws-g6-l4-cvvdp)"""
    r, t4 = load("aws-g6-l4-cvvdp"), tiles4x4()
    s2 = dict(r["bd_rate_ssimulacra2_as_published"], av1gpu_vs_aom_s6=need(t4, "bd_rate", "av1gpu_vs_aom_s6"))
    jod = dict(r["bd_rate_cvvdp"], av1gpu_vs_aom_s6=need(t4, "bd_rate_cvvdp", "av1gpu_vs_aom_s6"))
    enc = [("av1gpu", "av1gpu default", "av1gpu"), ("svt_p4", "SVT-AV1 preset 4", "svt"),
           ("svt_p6", "SVT-AV1 preset 6", "svt"), ("svt_p9", "SVT-AV1 preset 9", "svt"),
           ("aom_s9", "libaom speed 9", "aom_s9"), ("nvenc", "NVENC AV1", "nvenc")]
    band, top = 40, 150
    s = Svg(760, top + band * len(enc) + 50, t, "BD-rate against libaom speed 6 on SSIMULACRA2 and on CVVDP")
    s.header("Two metrics: SSIMULACRA2 and CVVDP",
             ["Median BD-rate against libaom speed 6, same files. CVVDP (standard_fhd display) agrees with",
              "SSIMULACRA2 within about a point for every encoder except av1gpu."])
    marker(s, t, "circle", 30, 112, t["ink2"])
    s.text(42, 116, "SSIMULACRA2", 13, "ink2")
    marker(s, t, "diamond", 160, 112, t["ink2"])
    s.text(172, 116, "CVVDP (JOD)", 13, "ink2")
    L, R, xmin, xmax = 200, 730, -5, 30
    X = lambda v: L + (R - L) * (v * 100 - xmin) / (xmax - xmin)
    for v in range(xmin, xmax + 1, 5):
        s.line(X(v / 100), top - 10, X(v / 100), top + band * len(enc) - 10, "axis" if v == 0 else "grid")
        s.text(X(v / 100), top + band * len(enc) + 8, (f"{v:+d} %" if v else "0").replace("-", "−"), 12, "muted",
               "middle")
    s.text((L + R) / 2, top + band * len(enc) + 28, "files vs libaom speed 6 (smaller is better)", 12, "muted",
           "middle")
    for i, (k, label, col) in enumerate(enc):
        y = top + band * i + 10
        a, b = s2[f"{k}_vs_aom_s6"]["median"], jod[f"{k}_vs_aom_s6"]["median"]
        s.text(24, y + 4, label, 13, "ink", weight=600 if k == "av1gpu" else 400)
        s.add(f'<line x1="{X(a):.1f}" y1="{y}" x2="{X(b):.1f}" y2="{y}" stroke="{t[col]}" stroke-width="2"/>')
        marker(s, t, "circle", X(a), y, t[col])
        marker(s, t, "diamond", X(b), y, t[col])
        lo, hi = sorted((a, b))
        lab = f"{a * 100:+.1f} / {b * 100:+.1f} %".replace("-", "−")
        if X(hi) + 12 + 7 * len(lab) > 745:  # no room on the right: label left of the pair
            s.text(X(lo) - 12, y + 4, lab, 12, "ink2", "end")
        else:
            s.text(X(hi) + 12, y + 4, lab, 12, "ink2")
    s.save(os.path.join(HERE, f"metrics-{t['name']}.svg"))


def marker(s, t, kind, x, y, color, r=5):
    ring = f'stroke="{t["surface"]}" stroke-width="2"'
    if kind == "circle":
        s.add(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="{r}" fill="{color}" {ring}/>')
    elif kind == "square":
        s.add(f'<rect x="{x - r:.1f}" y="{y - r:.1f}" width="{2 * r}" height="{2 * r}" rx="1.5" fill="{color}" {ring}/>')
    elif kind == "triangle":
        s.add(f'<path d="M{x:.1f},{y - r - 1:.1f} L{x + r + 1:.1f},{y + r:.1f} L{x - r - 1:.1f},{y + r:.1f} Z" '
              f'fill="{color}" {ring}/>')
    else:
        s.add(f'<path d="M{x:.1f},{y - r - 1:.1f} L{x + r + 1:.1f},{y:.1f} L{x:.1f},{y + r + 1:.1f} '
              f'L{x - r - 1:.1f},{y:.1f} Z" fill="{color}" {ring}/>')


def webp_icc_fixed(rows):
    """the published WebP rows with SSIMULACRA2 re-scored with the .webp's ICC profile (bench/results/aws-g6-l4-cvvdp:
    dwebp drops it, so 13 photos with non-sRGB profiles were scored as colour-shifted)"""
    fix = {(x["image"], x["q"]): x["ssimulacra2_icc"] for x in load("aws-g6-l4-cvvdp")["webp_icc_check"]["rows"]}
    return [dict(x, ssimulacra2=fix.get((x["image"], x["q"]), x["ssimulacra2"])) for x in rows]


def quality_chart(t):
    r = load("aws-g6-l4-gamut")
    r["quality"]["av1gpu"] = tiles4x4()["quality"]["av1gpu"]
    r["quality"]["webp"] = webp_icc_fixed(r["quality"]["webp"])
    man = {m["file"]: int(m["width"]) * int(m["height"])
           for m in csv.DictReader(open(os.path.join(ROOT, "bench", "openimages_v4_gamut.csv")))}
    curves = {}
    for codec, rows in r["quality"].items():
        pts = []
        for q in sorted({x["q"] for x in rows}):
            sel = [x for x in rows if x["q"] == q]
            bpp = sum(x["bytes"] for x in sel) * 8 / sum(man[x["image"]] for x in sel)
            pts.append((bpp, statistics.fmean(x["ssimulacra2"] for x in sel)))
        curves[codec] = pts
    nv = nvenc()[0]["quality"]["nvenc"]
    curves["nvenc"] = []
    for q in sorted({x["q"] for x in nv}):
        sel = [x for x in nv if x["q"] == q]
        bpp = sum(x["bytes"] for x in sel) * 8 / sum(man[x["image"]] for x in sel)
        curves["nvenc"].append((bpp, statistics.fmean(x["ssimulacra2"] for x in sel)))
    curves["nvenc"] = [p for p in curves["nvenc"] if p[0] <= 1.8]
    s = Svg(760, 420, t, "Quality against file size: av1gpu, libaom speed 6 and 9, WebP, NVENC AV1")
    s.header("Quality against file size",
             ["Mean SSIMULACRA2 (higher is better) against bits per pixel, over the same 100 photos.",
              "Up and to the left is better; av1gpu lies on top of libaom speed 6, NVENC well below it."])
    order = ["av1gpu", "aom_s6", "aom_s9", "webp", "nvenc"]
    s.legend(24, 110, [(k, NAMES[k]) for k in order])
    L, R, T, B = 64, 730, 132, 370
    xmax, ymin, ymax = 1.8, 20, 90
    X = lambda v: L + (R - L) * v / xmax
    Y = lambda v: B - (B - T) * (v - ymin) / (ymax - ymin)
    for v in range(ymin, ymax + 1, 10):
        s.line(L, Y(v), R, Y(v))
        s.text(L - 8, Y(v) + 4, str(v), 12, "muted", "end")
    for i in range(0, 10):
        v = i * 0.2
        s.text(X(v), B + 18, f"{v:.1f}", 12, "muted", "middle")
    s.line(L, B, R, B, "axis")
    s.text((L + R) / 2, B + 38, "bits per pixel", 12, "muted", "middle")
    for k in reversed(order):       # av1gpu drawn last, on top
        pts = curves[k]
        d = " ".join(f"{'M' if i == 0 else 'L'}{X(a):.1f},{Y(b):.1f}" for i, (a, b) in enumerate(pts))
        s.add(f'<path d="{d}" fill="none" stroke="{t[k]}" stroke-width="2" stroke-linejoin="round" '
              f'stroke-linecap="round"/>')
        for a, b in pts:
            s.add(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="4" fill="{t[k]}" stroke="{t["surface"]}" '
                  f'stroke-width="2"/>')
    s.save(os.path.join(HERE, f"quality-{t['name']}.svg"))


def by_size_chart(t):
    r = load("aws-g6-l4-gamut")
    r["quality"]["av1gpu"] = tiles4x4()["quality"]["av1gpu"]
    mp = {m["file"]: int(m["width"]) * int(m["height"]) / 1e6
          for m in csv.DictReader(open(os.path.join(ROOT, "bench", "openimages_v4_gamut.csv")))}
    per = {}
    for codec, rows in r["quality"].items():
        for x in rows:
            per.setdefault(codec, {}).setdefault(x["image"], []).append((x["bytes"], x["ssimulacra2"]))
    by = {}
    for x in nvenc()[0]["quality"]["nvenc"]:
        per.setdefault("nvenc", {}).setdefault(x["image"], []).append((x["bytes"], x["ssimulacra2"]))
    for k in ("av1gpu", "aom_s9", "nvenc"):
        for cls, lo, hi in SIZE_CLASSES:
            v = [bd_rate(per[k][im], per["aom_s6"][im]) for im in per[k] if lo <= mp[im] < hi]
            v = [x for x in v if x is not None]
            by.setdefault(f"{k}_vs_aom_s6", {})[cls] = dict(n=len(v), median=statistics.median(v))
    classes = [c for c, _, _ in SIZE_CLASSES]
    series = [("av1gpu", "av1gpu default"), ("aom_s9", "libaom speed 9"), ("nvenc", "NVENC AV1")]
    s = Svg(760, 380, t, "File size against libaom speed 6 by image size")
    s.header("File size against libaom speed 6, by image size",
             ["Median BD-rate on SSIMULACRA2: how much bigger (+) or smaller (−) the files are at equal quality.",
              "av1gpu matches speed 6 and does better as images get larger; NVENC is 22–36 % bigger."])
    s.legend(24, 110, [(k, n) for k, n in series])
    L, R, T, B = 64, 730, 136, 330
    ymin, ymax = -5, 40
    Y = lambda v: B - (B - T) * (v - ymin) / (ymax - ymin)
    for v in range(ymin, ymax + 1, 5):
        s.line(L, Y(v), R, Y(v), "axis" if v == 0 else "grid")
        s.text(L - 8, Y(v) + 4, f"{v:+d} %" if v else "0", 12, "muted", "end")
    gw = (R - L) / len(classes)
    for i, cls in enumerate(classes):
        cx = L + gw * (i + 0.5)
        n = by["av1gpu_vs_aom_s6"][cls]["n"]
        s.text(cx, B + 20, cls.replace(">=", "≥").replace("-", "–"), 13, "ink2", "middle")
        s.text(cx, B + 37, f"{n} images", 12, "muted", "middle")
        for j, (k, _) in enumerate(series):
            v = by[f"{k}_vs_aom_s6"][cls]["median"] * 100
            x = cx - 12 - 14 * (len(series) - 1) + j * 28
            s.bar_v(x, 24, Y(0), Y(v), t[k])
            s.text(x + 12, Y(v) - 6 if v >= 0 else Y(v) + 16, f"{v:+.1f}".replace("-", "−"), 12, "ink", "middle")
    s.save(os.path.join(HERE, f"by-size-{t['name']}.svg"))


def pipeline_diagram(t):
    s = Svg(900, 330, t, "av1gpu pipeline: the CPU parses JPEGs and writes AVIF files; the GPU does everything between")
    s.header("Where the work happens",
             ["The CPU only parses JPEG headers and writes files. Everything from the JPEG's pixels to the finished",
              "AV1 bitstream stays in GPU memory, with many images in flight at once."])
    s.add(f'<defs><marker id="a" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
          f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{t["muted"]}"/></marker></defs>')
    gpu = [("Decode JPEG", "nvJPEG or GPU IDCT"), ("Mode search", "every block shape"),
           ("Partition", "DP per superblock"), ("Reconstruction", "exact, closed loop"),
           ("Loop filters", "deblock + CDEF"), ("Entropy coding", "AV1 range coder")]
    bw, gap, x0, bh = 124, 20, 34, 52
    xs = [x0 + i * (bw + gap) for i in range(len(gpu))]
    cpu_y, gpu_y = 116, 232

    def box(x, y, a, b):
        s.add(f'<rect x="{x}" y="{y}" width="{bw}" height="{bh}" rx="8" fill="{t["surface"]}" '
              f'stroke="{t["axis"]}"/>')
        s.text(x + bw / 2, y + 22, a, 13, "ink", "middle", 600)
        s.text(x + bw / 2, y + 39, b, 12, "muted", "middle")

    def arrow(x1, y1, x2, y2):
        s.add(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{t["muted"]}" stroke-width="1.5" '
              f'marker-end="url(#a)"/>')

    s.add(f'<rect x="{x0 - 16}" y="{cpu_y - 26}" width="{900 - 2 * (x0 - 16)}" height="{bh + 40}" rx="10" '
          f'fill="{t["muted"]}" fill-opacity="0.08"/>')
    s.add(f'<rect x="{x0 - 16}" y="{gpu_y - 26}" width="{900 - 2 * (x0 - 16)}" height="{bh + 40}" rx="10" '
          f'fill="{t["av1gpu"]}" fill-opacity="0.10"/>')
    s.text(xs[1], cpu_y - 10, f"CPU · about {l4_speed('jpeg/size_range/default_enc4')[1]:.1f} cores", 12, "ink2", weight=600)
    s.text(xs[1], gpu_y - 10, "GPU · CUDA cores, no NVENC", 12, "ink2", weight=600)
    box(xs[0], cpu_y, "Read JPEGs", "parse headers")
    box(xs[-1], cpu_y, "Write AVIF", "OBUs + container")
    s.text((xs[1] + xs[-1] + bw) / 2 - bw / 2, cpu_y + 31, "free for other work", 13, "muted", "middle")
    for i, (a, b) in enumerate(gpu):
        box(xs[i], gpu_y, a, b)
        if i:
            arrow(xs[i - 1] + bw + 2, gpu_y + bh / 2, xs[i] - 3, gpu_y + bh / 2)
    arrow(xs[0] + bw / 2, cpu_y + bh + 2, xs[0] + bw / 2, gpu_y - 3)
    arrow(xs[-1] + bw / 2, gpu_y - 2, xs[-1] + bw / 2, cpu_y + bh + 3)
    s.save(os.path.join(HERE, f"pipeline-{t['name']}.svg"))


def banner(t):
    """Wordmark beside a superblock split the way AV1 partitions one: big blocks on flat areas, small on detail."""
    import random
    rnd = random.Random(19)
    ramp = (["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95"] if t["name"] == "light" else
            ["#104281", "#184f95", "#1c5cab", "#256abf", "#3987e5", "#6da7ec"])
    s = Svg(900, 220, t, "av1gpu: AVIF encoding on the GPU")
    blocks = []

    def split(x, y, n, depth):
        # detail grows toward the lower right, like an edge running through the block
        if depth < 3 and rnd.random() < (0.8, 0.5, 0.3)[depth] * (0.5 + ((x + y) / 2 + n / 2) / 192):
            h = n / 2
            for dx, dy in ((0, 0), (h, 0), (0, h), (h, h)):
                split(x + dx, y + dy, h, depth + 1)
        else:
            blocks.append((x, y, n))

    for gx in range(3):
        for gy in range(2):
            split(gx * 96, gy * 96, 96, 0)
    ox, oy, sc = 40, 24, 172 / 192
    for x, y, n in blocks:
        level = min(5, int(round(5 - (n / 96) * 5 + rnd.random())))
        s.add(f'<rect x="{ox + x * sc + 1:.1f}" y="{oy + y * sc + 1:.1f}" width="{n * sc - 2:.1f}" '
              f'height="{n * sc - 2:.1f}" rx="{2 if n < 24 else 4}" fill="{ramp[level]}"/>')
    s.add(f'<text x="350" y="112" font-size="64" font-weight="700" fill="{t["ink"]}" '
          f'letter-spacing="-1">av1gpu</text>')
    s.text(352, 148, "AVIF encoding on the GPU", 22, "ink2")
    ratio = l4_speed("jpeg/size_range/default_enc4")[0] / next(
        x for x in load("aws-g6-l4-gamut-enc2")["speed"] if "speed 6" in x["codec"])["mp_s"]
    s.text(352, 178, f"about libaom speed 6's file sizes, {ratio:.1f}× its speed on all 16 CPU threads", 14, "muted")
    s.save(os.path.join(HERE, f"banner-{t['name']}.svg"))


if __name__ == "__main__":
    for name, t in THEMES.items():
        t = dict(t, name=name)
        matchup_charts(t)
        tradeoff_chart(t)
        quality_chart(t)
        by_size_chart(t)
        formats_chart(t)
        metrics_chart(t)
        pipeline_diagram(t)
        banner(t)
    print("wrote", ", ".join(sorted(f for f in os.listdir(HERE) if f.endswith(".svg"))))
