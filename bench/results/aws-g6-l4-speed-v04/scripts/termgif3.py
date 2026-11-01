"""Render a recorded session (rec.py JSON) as an animated terminal GIF, with the real waiting times."""
import json, re, sys, textwrap
from PIL import Image, ImageDraw, ImageFont

rec = json.load(open(sys.argv[1]))
OUT = sys.argv[2]
W, COLS, LH, PAD, TOP = 960, 104, 20, 18, 40
FONT = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14)
BOLD = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 14, index=1)
BG, FG, DIM = (22, 24, 29), (220, 223, 228), (120, 126, 138)
GREEN, BLUE, YELLOW, PINK = (126, 211, 129), (110, 170, 245), (240, 200, 100), (240, 130, 170)
HIGHLIGHT = [(r"\d+(\.\d+)? MP/s", GREEN), (r"\d+(\.\d+)? cores", YELLOW), (r"real\s+\S+", GREEN),
             (r"\b\d+M\b", PINK), (r"99 images", BLUE), (r"\b99\b", BLUE)]

lines = []          # (segments, dim)
frames = []


def wrap(text):
    return textwrap.wrap(text.replace("\t", "  "), COLS, subsequent_indent="  ") or [""]


clock = None      # (seconds, running) for the stopwatch in the title bar


def render(cursor=False, typing=None, spin=None):
    shown = lines + ([typing] if typing else [])
    if spin is not None:                 # live "still working" line while the command runs
        shown = shown + [([("    encoding on the GPU", DIM, False), ("." * (1 + int(spin * 2) % 3), DIM, False)], False)]
    rows = max(16, len(shown) + 1)
    H = TOP + PAD + rows * LH + PAD
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.rectangle((0, 0, W, 30), fill=(38, 41, 48))
    for i, c in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        d.ellipse((14 + i * 20, 9, 26 + i * 20, 21), fill=c)
    d.text((W / 2, 15), "ubuntu@g6.4xlarge: ~/av1gpu-readme  (NVIDIA L4)", font=FONT,
           fill=DIM, anchor="mm")
    if clock is not None:
        sec, running = clock
        f = BOLD; txt = f"{sec:4.1f} s"; col = YELLOW if running else GREEN
        tw = d.textlength(txt, font=f); cx, cy = W - PAD - tw - 16, 15
        d.ellipse((cx - 7, cy - 7, cx + 7, cy + 7), outline=col, width=2)      # stopwatch face
        d.rectangle((cx - 2, cy - 11, cx + 2, cy - 9), fill=col)
        hand = (sec % 1.0) * 6.2832 if running else 0
        import math
        d.line((cx, cy, cx + 5 * math.sin(hand), cy - 5 * math.cos(hand)), fill=col, width=2)
        d.text((W - PAD - tw, cy), txt, font=f, fill=col, anchor="lm")
    y = TOP + PAD
    x = PAD
    if spin is not None:                 # spinner: eight dots, one bright, rotating
        import math
        sy = TOP + PAD + (len(shown) - 1) * LH + LH // 2 - 2
        for k in range(8):
            a = k * math.pi / 4; on = (int(spin * 10) % 8) == k
            d.ellipse((PAD + 10 + 7 * math.cos(a) - 2, sy + 7 * math.sin(a) - 2, PAD + 10 + 7 * math.cos(a) + 2,
                       sy + 7 * math.sin(a) + 2), fill=GREEN if on else (70, 75, 86))
    for segs, dim in shown:
        x = PAD
        for text, color, bold in segs:
            d.text((x, y), text, font=BOLD if bold else FONT, fill=DIM if dim else color)
            x += d.textlength(text, font=FONT)
        y += LH
    if cursor:
        d.rectangle((x + 1, y - LH + 3, x + 9, y - 2), fill=FG)
    return im


def frame(ms, **kw):
    frames.append((render(**kw), ms))


def colored(text):
    marks = []
    for pat, col in HIGHLIGHT:
        for m in re.finditer(pat, text):
            if not any(a < m.end() and m.start() < b for a, b, _ in marks):
                marks.append((m.start(), m.end(), col))
    segs, i = [], 0
    for a, b, col in sorted(marks):
        segs.append((text[i:a], FG, False)); segs.append((text[a:b], col, True)); i = b
    segs.append((text[i:], FG, False))
    return segs


def prompt(cmd):
    p = [("$ ", GREEN, True)]
    for n in range(0, len(cmd) + 1, 2):
        frame(45, cursor=True, typing=(p + [(cmd[:n], FG, False)], False))
    lines.append((p + [(cmd, FG, False)], False))


frame(900, cursor=True, typing=([("$ ", GREEN, True)], False))
for c in rec:
    cmd = c["cmd"].replace("bash -c 'time ", "time ").rstrip("'")
    prompt(cmd)
    t = 0.0
    for when, text in c["lines"]:
        while when - t > 0.1:            # real waiting time: spinner and stopwatch, 10 frames a second
            clock = (t, True); frame(100, spin=t); t += 0.1
        dim = text.startswith("[av1gpu] stage ms") or text.startswith(("user", "sys"))
        for w in wrap(text):
            lines.append((colored(w), dim))
        t = when
    end = c["lines"][-1][0] if c["lines"] else 0
    real = [re.search(r"(\d+)m([\d.]+)s", x) for _, x in c["lines"] if x.startswith("real")]
    if real:
        end = int(real[0][1]) * 60 + float(real[0][2])
    clock = (end, False) if end >= 1 else clock   # instant commands keep the last time
    frame(700 if end < 1 else 2500)
_du = [x for c in rec for _, x in c["lines"] if re.match(r"\d+M\s", x)]
_real = [re.search(r"(\d+)m([\d.]+)s", x) for c in rec for _, x in c["lines"] if x.startswith("real")][0]
_secs = int(_real[1]) * 60 + float(_real[2])
prompt(f"# {_du[0].split()[0][:-1]} MB of JPEGs -> {_du[-1].split()[0][:-1]} MB of AVIF in {_secs:.1f} s, with the CPU almost idle")
frame(4000, cursor=True)

# one shared palette keeps the GIF small; pad all frames to the final height
H = max(f.height for f, _ in frames)
pal = frames[-1][0].convert("P", palette=Image.ADAPTIVE, colors=64)
out = []
for f, ms in frames:
    full = Image.new("RGB", (W, H), BG); full.paste(f, (0, 0))
    out.append(full.quantize(palette=pal, dither=Image.NONE))
out[0].save(OUT, save_all=True, append_images=out[1:], duration=[ms for _, ms in frames], loop=0,
            optimize=True)
print(len(out), "frames", sum(ms for _, ms in frames) / 1000, "s")
