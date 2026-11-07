#!/usr/bin/env python3
"""--help goes to stdout with status 0 and names every option the command line takes.

  tests/help_test.py --bin build/av1gpu --source encoder/apps/av1gpu.cpp
"""
import argparse
import re
import subprocess
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--source", required=True)
    a = ap.parse_args()
    p = subprocess.run([a.bin, "--help"], capture_output=True, text=True, timeout=60)
    src = open(a.source, encoding="utf8").read()
    parser = src[src.index("struct Opts"):]
    options = sorted(set(re.findall(r'a == "(--[A-Za-z0-9-]+)"', parser)))
    missing = [o for o in options if not re.search(re.escape(o) + r"(?![A-Za-z0-9-])", p.stdout)]
    ok = p.returncode == 0 and len(options) > 50 and not missing
    print(f"{len(options)} options, status {p.returncode}, missing from --help: {missing}")
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
