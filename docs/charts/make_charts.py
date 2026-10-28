#!/usr/bin/env python3
"""Draw the README charts from the NVIDIA L4 results in bench/results/ (needs numpy, like bench/bench.py).

  docs/charts/make_charts.py

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
                  av1gpu="#2a78d6", aom_s6="#eb6834", aom_s9="#1baf7a", webp="#eda100"),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781", grid="#2c2c2a",
                 axis="#383835", border="rgba(255,255,255,0.10)",
                 av1gpu="#3987e5", aom_s6="#d95926", aom_s9="#199e70", webp="#c98500"),
}
FONT = "system-ui,-apple-system,'Segoe UI',Helvetica,Arial,sans-serif"
NAMES = dict(av1gpu="av1gpu", aom_s6="libaom speed 6", aom_s9="libaom speed 9", webp="WebP")


def load(name):
    return json.load(open(os.path.join(RES, name, "results.json")))


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


def files(bd):
    return f"files {bd * 100:+.1f} % vs speed 6".replace("-", "−")


def speed_chart(t):
    gamut = load("aws-g6-l4-gamut")["bd_rate"]
    rows = [  # (label, note, colour key, MP/s)
        ("libaom speed 9", files(gamut["aom_s9_vs_aom_s6"]["median"]), "aom_s9",
         next(s for s in load("aws-g6-l4-gamut-enc2")["speed"] if "speed 9" in s["codec"])["mp_s"]),
        ("av1gpu, small-image preset off", "files +2.1 % vs speed 6", "av1gpu",
         load("aws-g6-l4-gamut-turbo-nosmall")["speed"][0]["mp_s"]),
        ("av1gpu default (4 encoders)", files(gamut["av1gpu_vs_aom_s6"]["median"]),
         "av1gpu", load("aws-g6-l4-gamut-enc4")["speed"][0]["mp_s"]),
        ("libaom speed 6", "the reference for file size", "aom_s6",
         next(s for s in load("aws-g6-l4-gamut-enc2")["speed"] if "speed 6" in s["codec"])["mp_s"]),
    ]
    s = Svg(760, 334, t, "Encoding speed on an NVIDIA L4, megapixels per second")
    s.header("Encoding speed, megapixels per second",
             ["100 photos of 0.18–24 MP on an AWS g6.4xlarge. libaom runs one avifenc per vCPU (16 at once);",
              "av1gpu uses one NVIDIA L4 and about 0.1 CPU core."])
    x0, x1, top, band = 250, 680, 104, 54
    vmax = 200
    for v in range(0, vmax + 1, 50):
        x = x0 + (x1 - x0) * v / vmax
        s.line(x, top - 6, x, top + band * len(rows) - 14)
        s.text(x, top + band * len(rows) + 4, str(v), 12, "muted", "middle")
    for i, (label, note, key, v) in enumerate(rows):
        y = top + band * i
        s.text(24, y + 13, label, 14, weight=600 if key == "av1gpu" else 400)
        s.text(24, y + 31, note, 12, "muted")
        xe = x0 + (x1 - x0) * v / vmax
        s.bar_h(x0, xe, y + 4, 24, t[key])
        s.text(xe + 8, y + 21, f"{v:.0f}", 14, weight=600)
    s.line(x0, top - 6, x0, top + band * len(rows) - 14, "axis")
    s.save(os.path.join(HERE, f"speed-{t['name']}.svg"))


def quality_chart(t):
    r = load("aws-g6-l4-gamut")
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
    s = Svg(760, 420, t, "Quality against file size: av1gpu, libaom speed 6 and 9, WebP")
    s.header("Quality against file size",
             ["Mean SSIMULACRA2 (higher is better) against bits per pixel, over the same 100 photos.",
              "Up and to the left is better; av1gpu lies on top of libaom speed 6."])
    order = ["av1gpu", "aom_s6", "aom_s9", "webp"]
    s.legend(24, 110, [(k, NAMES[k]) for k in order])
    L, R, T, B = 64, 730, 132, 370
    xmax, ymin, ymax = 1.8, 30, 90
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
    mp = {m["file"]: int(m["width"]) * int(m["height"]) / 1e6
          for m in csv.DictReader(open(os.path.join(ROOT, "bench", "openimages_v4_gamut.csv")))}
    per = {}
    for codec, rows in r["quality"].items():
        for x in rows:
            per.setdefault(codec, {}).setdefault(x["image"], []).append((x["bytes"], x["ssimulacra2"]))
    by = {}
    for k in ("av1gpu", "aom_s9"):
        for cls, lo, hi in SIZE_CLASSES:
            v = [bd_rate(per[k][im], per["aom_s6"][im]) for im in per[k] if lo <= mp[im] < hi]
            v = [x for x in v if x is not None]
            by.setdefault(f"{k}_vs_aom_s6", {})[cls] = dict(n=len(v), median=statistics.median(v))
    classes = [c for c, _, _ in SIZE_CLASSES]
    series = [("av1gpu", "av1gpu default"), ("aom_s9", "libaom speed 9")]
    s = Svg(760, 380, t, "File size against libaom speed 6 by image size")
    s.header("File size against libaom speed 6, by image size",
             ["Median BD-rate on SSIMULACRA2: how much bigger (+) or smaller (−) the files are at equal quality.",
              "av1gpu matches speed 6 and does better as images get larger."])
    s.legend(24, 110, [(k, n) for k, n in series])
    L, R, T, B = 64, 730, 136, 330
    ymin, ymax = -5, 25
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
            x = cx - 26 + j * 28
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
    s.text(xs[1], cpu_y - 10, "CPU · about 0.1 core", 12, "ink2", weight=600)
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
    s.text(352, 178, "libaom-speed-6 file sizes · 4× the speed of a 16-thread CPU · 0.1 CPU cores", 14, "muted")
    s.save(os.path.join(HERE, f"banner-{t['name']}.svg"))


if __name__ == "__main__":
    for name, t in THEMES.items():
        t = dict(t, name=name)
        speed_chart(t)
        quality_chart(t)
        by_size_chart(t)
        pipeline_diagram(t)
        banner(t)
    print("wrote", ", ".join(sorted(f for f in os.listdir(HERE) if f.endswith(".svg"))))
