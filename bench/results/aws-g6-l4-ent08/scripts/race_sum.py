#!/usr/bin/env python3
"""race_sum.py LOG...: the distinct racecheck analysis reports per log: (kernel, file:line) of the first access and
of each conflicting access, hazard counts summed over launches; template arguments and parameter lists dropped."""
import re, sys


def short(s):
    s = re.sub(r"\+0x[0-9a-f]+", "", s)
    while True:
        b = re.sub(r"<[^<>]*>", "", s)
        if b == s:
            break
        s = b
    s = re.sub(r"\([^()]*\)", "", s)
    s = re.sub(r"/home/ubuntu/[a-z0-9]+/", "", s)
    s = s.replace("void ", "").replace("av1g_gpu::", "")
    return re.sub(r"\s+", " ", s).strip()


for p in sys.argv[1:]:
    t = [l.replace("=========", "").strip() for l in open(p, errors="replace").read().splitlines()]
    out = {}
    summ = [l for l in t if "RACECHECK SUMMARY" in l or l.startswith("rc ")]
    i = 0
    while i < len(t):
        m = re.match(r"(Error|Warning): Race reported between (\w+) access at (.*) in (\S+)$", t[i])
        if m:
            first = f"{m.group(1)}: {m.group(2)} {short(m.group(3))} {m.group(4)}"
            i += 1
            while i < len(t) and t[i].startswith("and "):
                m2 = re.match(r"and (\w+) access at (.*) in (\S+) \[(\d+) hazards?\]", t[i])
                if m2:
                    k = f"{first}  <->  {m2.group(1)} {short(m2.group(2))} {m2.group(3)}"
                    out[k] = out.get(k, 0) + int(m2.group(4))
                i += 1
            continue
        i += 1
    print(f"== {p}: {' | '.join(summ)}")
    for k, v in sorted(out.items(), key=lambda x: -x[1]):
        print(f"  {v:9d}  {k}")
