#!/usr/bin/env python3
"""Generate av1core C headers from the AV1 specification markdown.

Provenance: every table and enum value in the generated headers is parsed from
the AV1 bitstream specification sources (github.com/AOMediaCodec/av1-spec, *.md) (git rev noted in
the header).  Nothing is hand-typed.  Sanity checks below cross-check a handful of
well-known values and the element counts against the declared dimensions.

Outputs (into ../av1core/include/av1g/):
  enums_gen.h      - spec constants (03.symbols.md) + enum names (07 semantics tables)
  spec_tables.h    - non-CDF lookup tables, host+device copies
  cdf_defaults.h   - default CDF tables (10.additional.tables.md), host+device copies

Usage:  python3 tools/gen_tables.py [--spec DIR]
"""
import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "av1core", "include", "av1g")
DEFAULT_SPEC = os.environ.get("AV1_SPEC_DIR", "av1-spec")  # git clone https://github.com/AOMediaCodec/av1-spec

# ---------------------------------------------------------------------------
# parsing helpers
# ---------------------------------------------------------------------------

def read(spec, name):
    with open(os.path.join(spec, name), encoding="utf-8") as f:
        return f.read()


# NOTE: table elements / dimensions are small integer expressions taken from the
# (trusted, local, read-only) spec markdown, e.g. "SIG_COEF_CONTEXTS_2D + 5" or
# "1 << 14".  They are evaluated with builtins disabled and only spec constants in
# scope.
SAFE = {"__builtins__": {}}


def unescape(s):
    return s.replace("\\<", "<").replace("\\>", ">").replace("\\|", "|").replace("\\-", "-")


def parse_constants(text):
    """| `NAME` | value | desc  rows of 03.symbols.md"""
    consts = {}
    for m in re.finditer(r"^\|\s*`([A-Z0-9_]+)`\s*\|\s*([^|]+?)\s*\|", text, re.M):
        name, val = m.group(1), unescape(m.group(2))
        consts[name] = val
    out = {}
    # evaluate (some refer to others)
    for _ in range(4):
        for k, v in consts.items():
            if k in out:
                continue
            try:
                out[k] = int(eval(v, SAFE, dict(out)))
            except Exception:
                pass
    missing = [k for k in consts if k not in out]
    if missing:
        raise SystemExit("could not evaluate constants: %s" % missing)
    return out


def parse_enum_rows(text):
    """| <int> | NAME  rows (two-column semantic tables) -> name->value."""
    enums = {}
    for m in re.finditer(r"^\|\s*(\d+)\s*\|\s*([A-Z][A-Z0-9_]+)\s*$", text, re.M):
        v, name = int(m.group(1)), m.group(2)
        if name in enums and enums[name] != v:
            raise SystemExit("conflicting enum %s: %d vs %d" % (name, enums[name], v))
        enums[name] = v
    return enums


TABLE_RE = re.compile(r"^([A-Z][A-Za-z0-9_]*)\s*((?:\[[^\]\n]*\]\s*)+)=\s*\n?\s*\{", re.M)


def find_tables(text):
    """Yield (name, [dim exprs], body) for every `Name[..]..= { ... }` definition."""
    for m in TABLE_RE.finditer(text):
        name = m.group(1)
        dims = [d.strip() for d in re.findall(r"\[([^\]]*)\]", m.group(2))]
        # match braces
        i = m.end() - 1
        depth = 0
        j = i
        while True:
            c = text[j]
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
            j += 1
        yield name, dims, text[i:j + 1]


def flatten_body(body):
    body = re.sub(r"//[^\n]*", "", body)
    body = re.sub(r"/\*.*?\*/", "", body, flags=re.S)
    body = unescape(body)
    body = body.replace("{", ",").replace("}", ",")
    items = []
    for line in body.split("\n"):
        for piece in line.split(","):
            p = piece.strip()
            if p:
                items.append(p)
    return items


# ---------------------------------------------------------------------------

# Tables emitted into spec_tables.h  (name -> optional forced C type)
SPEC_TABLES = [
    # 10.additional.tables.md : scans / conversions / predictors
    r"(Default|Mrow|Mcol)_Scan_\d+x\d+",
    "Mi_Width_Log2", "Mi_Height_Log2", "Num_4x4_Blocks_Wide", "Num_4x4_Blocks_High",
    "Size_Group", "Max_Tx_Size_Rect", "Partition_Subsize", "Split_Tx_Size", "Mode_To_Txfm",
    r"Sm_Weights_Tx_\d+x\d+", "Mode_To_Angle", "Dr_Intra_Derivative", "Intra_Filter_Taps",
    "Tx_Size_Sqr", "Tx_Size_Sqr_Up", "Tx_Width", "Tx_Height", "Tx_Width_Log2", "Tx_Height_Log2",
    "Sig_Ref_Diff_Offset", "Adjusted_Tx_Size", "Qm_Offset", "Quantizer_Matrix",
    # 08.decoding.process.md
    "Dc_Qlookup", "Ac_Qlookup", "Cos128_Lookup", "Transform_Row_Shift", "Intra_Edge_Kernel",
    # 06.bitstream.syntax.md
    "Max_Tx_Depth", "Subsampled_Size", "Tx_Type_In_Set_Intra", "Tx_Type_In_Set_Inter",
    "Tx_Type_Intra_Inv_Set1", "Tx_Type_Intra_Inv_Set2",
    # 09.parsing.process.md
    "Intra_Mode_Context", "Coeff_Base_Ctx_Offset", "Coeff_Base_Pos_Ctx_Offset",
    "Mag_Ref_Offset_With_Tx_Class", "Filter_Intra_Mode_To_Intra_Dir",
]

# Default CDFs needed for intra-only coding (+ reserved M5 tools)
CDF_TABLES = [
    "Default_Intra_Frame_Y_Mode_Cdf", "Default_Uv_Mode_Cfl_Not_Allowed_Cdf",
    "Default_Uv_Mode_Cfl_Allowed_Cdf", "Default_Angle_Delta_Cdf",
    "Default_Partition_W8_Cdf", "Default_Partition_W16_Cdf", "Default_Partition_W32_Cdf",
    "Default_Partition_W64_Cdf", "Default_Partition_W128_Cdf",
    "Default_Tx_8x8_Cdf", "Default_Tx_16x16_Cdf", "Default_Tx_32x32_Cdf", "Default_Tx_64x64_Cdf",
    "Default_Filter_Intra_Mode_Cdf", "Default_Filter_Intra_Cdf",
    "Default_Skip_Cdf", "Default_Delta_Q_Cdf", "Default_Delta_Lf_Cdf",
    "Default_Intra_Tx_Type_Set1_Cdf", "Default_Intra_Tx_Type_Set2_Cdf",
    "Default_Cfl_Sign_Cdf", "Default_Cfl_Alpha_Cdf",
    "Default_Palette_Y_Mode_Cdf", "Default_Palette_Uv_Mode_Cdf",
    "Default_Txb_Skip_Cdf", "Default_Eob_Pt_16_Cdf", "Default_Eob_Pt_32_Cdf",
    "Default_Eob_Pt_64_Cdf", "Default_Eob_Pt_128_Cdf", "Default_Eob_Pt_256_Cdf",
    "Default_Eob_Pt_512_Cdf", "Default_Eob_Pt_1024_Cdf", "Default_Eob_Extra_Cdf",
    "Default_Dc_Sign_Cdf", "Default_Coeff_Base_Eob_Cdf", "Default_Coeff_Base_Cdf",
    "Default_Coeff_Br_Cdf",
]


def ctype_for(vals):
    lo, hi = min(vals), max(vals)
    for t, a, b in (("uint8_t", 0, 255), ("int8_t", -128, 127), ("uint16_t", 0, 65535),
                    ("int16_t", -32768, 32767)):
        if a <= lo and hi <= b:
            return t
    return "int32_t"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", default=DEFAULT_SPEC)
    args = ap.parse_args()
    spec = args.spec
    try:
        rev = subprocess.check_output(["git", "-C", spec, "log", "-1", "--format=%H %cs"],
                                      text=True).strip()
    except Exception:
        rev = "unknown"

    files = ["03.symbols.md", "06.bitstream.syntax.md", "07.bitstream.semantics.md",
             "08.decoding.process.md", "09.parsing.process.md", "10.additional.tables.md"]
    texts = {f: read(spec, f) for f in files}

    consts = parse_constants(texts["03.symbols.md"])
    enums = parse_enum_rows(texts["07.bitstream.semantics.md"])
    # names needed for evaluation
    ns = dict(consts)
    ns.update(enums)
    ns["BLOCK_INVALID"] = consts["BLOCK_INVALID"]
    # transform set names are in a 3-column table: | is_inter | set | NAME |
    for m in re.finditer(r"^\|[^|\n]*\|\s*(\d+)\s*\|\s*(TX_SET_[A-Z0-9_]+)\s*\|", texts["07.bitstream.semantics.md"], re.M):
        ns[m.group(2)] = int(m.group(1))

    tables = {}
    for f in files:
        for name, dims, body in find_tables(texts[f]):
            if name in tables:
                continue
            tables[name] = (f, dims, body)

    def evaluate(name):
        f, dims, body = tables[name]
        d = [int(eval(unescape(x), SAFE, ns)) for x in dims]
        items = flatten_body(body)
        vals = []
        for it in items:
            try:
                vals.append(int(eval(it, SAFE, ns)))
            except Exception as e:
                raise SystemExit("table %s: cannot evaluate %r (%s)" % (name, it, e))
        n = 1
        for x in d:
            n *= x
        if len(vals) != n:
            raise SystemExit("table %s (%s): %d values for dims %s" % (name, f, len(vals), d))
        return f, d, vals

    def select(patterns):
        out = []
        for p in patterns:
            hits = sorted(k for k in tables if re.fullmatch(p, k))
            if not hits:
                raise SystemExit("table pattern %s not found in spec" % p)
            out.extend(h for h in hits if h not in out)
        return out

    os.makedirs(OUT, exist_ok=True)
    banner = ("// GENERATED by tools/gen_tables.py from the AV1 specification markdown\n"
              "// (%s, rev %s). DO NOT EDIT.\n" % ("github.com/AOMediaCodec/av1-spec", rev))

    # ---- enums_gen.h ------------------------------------------------------
    with open(os.path.join(OUT, "enums_gen.h"), "w") as fo:
        fo.write(banner)
        fo.write("#ifndef AV1G_ENUMS_GEN_H\n#define AV1G_ENUMS_GEN_H\n\n")
        fo.write("/* constants from 03.symbols.md */\n")
        for k in sorted(consts):
            fo.write("#define AV1G_%s %d\n" % (k, consts[k]))
        fo.write("\n/* enum values from the semantics tables in 07.bitstream.semantics.md */\n")
        for k in sorted(set(enums) | {k for k in ns if k.startswith("TX_SET_")}):
            if k in consts:
                continue
            fo.write("#define AV1G_%s %d\n" % (k, ns[k]))
        fo.write("\n#endif\n")

    def emit(fname, guard, names, extra_doc):
        with open(os.path.join(OUT, fname), "w") as fo:
            fo.write(banner)
            fo.write("#ifndef %s\n#define %s\n#include <stdint.h>\n#include \"av1g/hd.h\"\n\n" % (guard, guard))
            fo.write(extra_doc)
            for name in names:
                f, d, vals = evaluate(name)
                t = ctype_for(vals)
                if re.fullmatch(r"(Default|Mrow|Mcol)_Scan_\d+x\d+", name):
                    t = "uint16_t"   # uniform type so get_scan() can return one pointer type
                dimstr = "".join("[%d]" % x for x in d)
                body = ", ".join(str(v) for v in vals)
                fo.write("/* %s : %s */\n" % (name, f))
                fo.write("AV1G_TABLE(%s, %s, %s, {%s})\n" % (t, name, dimstr, body))
            fo.write("\n#endif\n")

    spec_names = select(SPEC_TABLES)
    cdf_names = select(CDF_TABLES)

    # sanity checks against well-known values
    def val(name, *idx):
        f, d, vals = evaluate(name)
        off = 0
        for i, x in zip(idx, d):
            off = off * x + i
        return vals[off]
    assert val("Dc_Qlookup", 0, 0) == 4 and val("Dc_Qlookup", 0, 255) == 1336
    assert val("Ac_Qlookup", 0, 0) == 4 and val("Ac_Qlookup", 0, 255) == 1828
    assert val("Cos128_Lookup", 0) == 4096 and val("Cos128_Lookup", 64) == 0
    assert val("Mode_To_Angle", 8) == 67
    assert val("Default_Scan_4x4", 2) == 4
    assert val("Split_Tx_Size", 12) == ns["TX_32X32"] and val("Split_Tx_Size", 13) == ns["TX_4X8"]
    assert val("Default_Skip_Cdf", 0, 1) == 32768
    assert ns["UV_CFL_PRED"] == 13 and ns["BLOCK_64X16"] == 21 and ns["TX_64X16"] == 18
    assert ns["PARTITION_VERT_4"] == 9 and ns["TX_SET_INTRA_2"] == 2
    assert val("Quantizer_Matrix", 0, 0, 0) == 32 and val("Quantizer_Matrix", 0, 0, 15) == 200
    assert val("Qm_Offset", 4) == 336 and val("Qm_Offset", 18) == 2192

    emit("spec_tables.h", "AV1G_SPEC_TABLES_H", spec_names,
         "/* Lookup tables used by the decoding process; access with AV1G_T(Name)[...]\n"
         "   so that host and device code each read their own copy. */\n\n")
    emit("cdf_defaults.h", "AV1G_CDF_DEFAULTS_H", cdf_names,
         "/* Default CDFs in spec format (cumulative, last entry 32768 followed by the\n"
         "   adaptation counter 0). */\n\n")
    # ---- levels_gen.h (annex.a.levels.md) ----------------------------------
    lv = read(spec, "annex.a.levels.md")
    rows1 = {}
    rows2 = {}
    for m in re.finditer(r"^\|\s*(\d)\.(\d)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", lv, re.M):
        rows1[(int(m.group(1)), int(m.group(2)))] = (int(m.group(3)), int(m.group(4)), int(m.group(5)))
    for m in re.finditer(r"^\|\s*(\d)\.(\d)\s*\|\s*\d+\s*\|[^|]*\|[^|]*\|[^|]*\|[^|]*\|\s*(\d+)\s*\|\s*(\d+)\s*\|", lv, re.M):
        rows2[(int(m.group(1)), int(m.group(2)))] = (int(m.group(3)), int(m.group(4)))
    levels = sorted(set(rows1) & set(rows2))
    if len(levels) < 14 or rows1[(2, 0)] != (147456, 2048, 1152) or rows2[(6, 0)] != (128, 16):
        raise SystemExit("level table parse failed: %s" % levels)
    with open(os.path.join(OUT, "levels_gen.h"), "w") as fo:
        fo.write(banner.replace("specification markdown", "specification markdown (annex.a.levels.md)"))
        fo.write("#ifndef AV1G_LEVELS_GEN_H\n#define AV1G_LEVELS_GEN_H\n#include <stdint.h>\n\n")
        fo.write("/* defined levels: seq_level_idx = ((X-2)<<2)|Y, MaxPicSize, MaxHSize, MaxVSize, MaxTiles, MaxTileCols */\n")
        fo.write("static const uint32_t av1g_levels[%d][6] = {\n" % len(levels))
        for (X, Y) in levels:
            fo.write("    {%d, %d, %d, %d, %d, %d}, /* %d.%d */\n" % (((X - 2) << 2) | Y, *rows1[(X, Y)], *rows2[(X, Y)], X, Y))
        fo.write("};\n#define AV1G_NUM_LEVELS %d\n\n#endif\n" % len(levels))

    print("generated %d spec tables, %d cdf tables, %d constants, %d enum names"
          % (len(spec_names), len(cdf_names), len(consts), len(enums)))


if __name__ == "__main__":
    main()
