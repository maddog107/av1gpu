#!/usr/bin/env python3
"""cmp9.py: byte identity of two gen.sh output trees (~/out/TAG), run by run, in the order of ~/sets9.tsv.

  cmp9.py man ROOT OUT.json          manifest of ROOT: per run the SHA-256 of every output file, the exit status(es),
                                     the 'skipped (unsupported)' count and the error messages (for another machine)
  cmp9.py cmp A B LABEL [GROUP]      A, B = output roots or manifests.  Two roots: every file pair is also compared
                                     byte by byte (cmp).  Prints one line per run, a total, and writes
                                     ~/results/ident_LABEL.json and ~/results/ident_LABEL.md (table).
A run counts as identical when both have the same files, every file is byte-identical, the exit statuses are the
same, and (failed single-file runs) the error messages are the same."""
import filecmp, hashlib, json, os, re, sys

H = os.path.expanduser("~")


def table():
    rows = []
    for line in open(f"{H}/sets9.tsv"):
        kind, name, src, envs, opts, groups = line.rstrip("\n").split("|")
        if kind == "batch":
            n = len(os.listdir(src)) if os.path.isdir(src) else sum(1 for _ in open(src))
        elif kind == "single":
            fs = [l.strip() for l in open(src) if l.strip()]
            if "--lossless" in opts.split():
                fs = [f for f in fs if not f.lower().endswith((".jpg", ".jpeg"))]
            n = len(fs)
        else:
            n = 1
        rows.append(dict(kind=kind, name=name, src=src, env=envs, opts=opts, groups=groups.split(), inputs=n))
    return rows


def manifest(root):
    m = {}
    for r in table():
        d = os.path.join(root, r["name"])
        if not os.path.isdir(d):
            continue
        o = os.path.join(d, "o")
        files = {f: hashlib.sha256(open(os.path.join(o, f), "rb").read()).hexdigest() for f in sorted(os.listdir(o))}
        rc = open(os.path.join(d, "rc")).read().strip() if os.path.exists(os.path.join(d, "rc")) else "missing"
        sk, err = None, {}
        if r["kind"] == "batch" and os.path.exists(os.path.join(d, "log")):
            x = re.findall(r"skipped \(unsupported\): (\d+)", open(os.path.join(d, "log"), errors="replace").read())
            sk = int(x[-1]) if x else None
        if r["kind"] == "err":
            err["log"] = open(os.path.join(d, "log"), errors="replace").read()
        if r["kind"] == "single":
            for line in rc.splitlines():
                b, s = line.rsplit(" ", 1)
                if s != "0":
                    err[b] = open(os.path.join(d, f"err_{b}.txt"), errors="replace").read()
        m[r["name"]] = dict(files=files, rc=rc, skipped=sk, err=err, root=root)
    return m


def load(x):
    return json.load(open(x)) if x.endswith(".json") else manifest(x)


def main():
    if sys.argv[1] == "man":
        json.dump(manifest(sys.argv[2]), open(sys.argv[3], "w"))
        return
    A, B, label = sys.argv[2:5]
    group = sys.argv[5] if len(sys.argv) > 5 else None
    ma, mb = load(A), load(B)
    both_dirs = os.path.isdir(A) and os.path.isdir(B)
    out, tot = [], dict(runs=0, ok=0, fa=0, fb=0, differ=0)
    for r in table():
        n = r["name"]
        if group and group not in r["groups"]:
            continue
        if n not in ma or n not in mb:
            if n in ma or n in mb:
                print(f"MISSING {n}: in {'A' if n in ma else 'B'} only")
            continue
        a, b = ma[n], mb[n]
        fa, fb = set(a["files"]), set(b["files"])
        differ = sorted(f for f in fa & fb if a["files"][f] != b["files"][f])
        if both_dirs:   # cmp as well
            for f in sorted(fa & fb):
                same = filecmp.cmp(os.path.join(A, n, "o", f), os.path.join(B, n, "o", f), shallow=False)
                if same != (f not in differ):
                    print(f"HASH/CMP DISAGREE {n} {f}")
                if not same and f not in differ:
                    differ.append(f)
        only_a, only_b = sorted(fa - fb), sorted(fb - fa)
        rc_same = a["rc"] == b["rc"]
        err_same = a["err"] == b["err"]
        ok = not differ and not only_a and not only_b and rc_same and err_same
        for f in differ:
            print(f"DIFF {n} {f}")
        for f in only_a:
            print(f"ONLY_A {n} {f}")
        for f in only_b:
            print(f"ONLY_B {n} {f}")
        if not rc_same:
            print(f"RC {n}: A [{a['rc'][:200]}] B [{b['rc'][:200]}]")
        if not err_same:
            print(f"ERRMSG {n}: A {json.dumps(a['err'])[:300]} B {json.dumps(b['err'])[:300]}")
        rcs = sorted(set(l.split()[-1] for l in a["rc"].splitlines())) if a["rc"] else []
        row = dict(name=n, kind=r["kind"], opts=r["opts"], env=r["env"], inputs=r["inputs"], files_a=len(fa),
                   files_b=len(fb), identical=len(fa & fb) - len(differ), differ=len(differ) + len(only_a) + len(only_b),
                   rc_same=rc_same, err_same=err_same, rc=",".join(rcs), skipped_a=a["skipped"], skipped_b=b["skipped"], ok=ok)
        out.append(row)
        tot["runs"] += 1; tot["ok"] += ok; tot["fa"] += len(fa); tot["fb"] += len(fb); tot["differ"] += row["differ"]
        print(f"{'OK  ' if ok else 'FAIL'} {n}: inputs {r['inputs']}, files {len(fa)} / {len(fb)}, identical "
              f"{row['identical']}/{len(fa | fb)}, rc {'same' if rc_same else 'DIFFERENT'} ({row['rc']}), skipped "
              f"{a['skipped']} / {b['skipped']}{', error messages same' if r['kind'] != 'batch' and err_same and a['err'] else ''}"
              f"{'' if err_same else ', ERROR MESSAGES DIFFER'}  [{r['env']} {r['opts']}]")
    print(f"TOTAL {label}: {tot['ok']}/{tot['runs']} runs identical; files {tot['fa']} / {tot['fb']}, differing or missing "
          f"{tot['differ']}")
    json.dump(dict(label=label, A=A, B=B, total=tot, runs=out), open(f"{H}/results/ident_{label}.json", "w"), indent=1)
    with open(f"{H}/results/ident_{label}.md", "w") as f:
        f.write("| run | options | inputs | A files | B files | identical | differ | exit status | skipped (A / B) |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---|---|\n")
        for x in out:
            o = " ".join(t for t in (x["env"], x["opts"]) if t) or "default"
            f.write(f"| {x['name']} | `{o}` | {x['inputs']} | {x['files_a']} | {x['files_b']} | {x['identical']} | "
                    f"{x['differ']} | {'same' if x['rc_same'] else 'DIFFERENT'} ({x['rc']}) | "
                    f"{x['skipped_a'] if x['skipped_a'] is not None else ''} / {x['skipped_b'] if x['skipped_b'] is not None else ''} |\n")
        f.write(f"| **all ({tot['runs']} runs)** | | | **{tot['fa']}** | **{tot['fb']}** | | **{tot['differ']}** | | |\n")


main()
