import sys, json, re
rows = []; cur = {}
for l in open(sys.argv[1]):
    if l.startswith('{"args"'):
        a = " ".join(x.split("/")[-1] for x in json.loads(l)["args"] if "{bin}" not in x and x != "--outdir")
        rows.append((a, cur)); cur = {}
    else:
        m = re.match(r"(\S+): median ([\d.]+)", l)
        if not m: continue
        k = m.group(1); k = "A" if "/rel/" in k else ("Ac" if k.endswith("/Ac") else "B"); cur[k] = float(m.group(2))
for a, c in rows:
    A, Ac, B = c["A"], c["Ac"], c["B"]
    print("%-36s A %6.1f  Ac %6.1f  B %6.1f  saved %6.1f ms (%4.1f %%)  A/Ac %.1f %%" % (a, A, Ac, B, A - B, (A - B) / A * 100, abs(A - Ac) / A * 100))
