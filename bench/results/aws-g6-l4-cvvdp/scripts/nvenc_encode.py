"""NVENC AV1 on the L4 through the hybrid_nvenc_libavif pipeline, unchanged (as scratchpad/l4n/run3.py):
nvenc_encode.py OUTDIR CQ[,CQ...] -> OUTDIR/<image>.q<CQ>.avif, 8 encodes at a time; prints one JSON line per CQ."""
import concurrent.futures as cf, json, os, sys, time
sys.path.insert(0, os.path.expanduser("~/hybrid_nvenc_libavif"))
from encode import encode_single
SRC = os.path.expanduser(os.environ.get("NV_SRC", "~/images_gamut"))
out, cqs = sys.argv[1], [int(x) for x in sys.argv[2].split(",")]
files = sorted(os.path.join(SRC, f) for f in os.listdir(SRC))
os.makedirs(out, exist_ok=True)
for cq in cqs:
    t0 = time.monotonic()
    with cf.ThreadPoolExecutor(8) as ex:
        rs = list(ex.map(lambda f: encode_single(f, os.path.join(out, os.path.basename(f) + f".q{cq}.avif"), cq=cq), files))
    bad = [(r.get("src"), r.get("error")[:200]) for r in rs if not r["ok"]]
    print(json.dumps(dict(cq=cq, files=len(rs), wall_s=round(time.monotonic() - t0, 2),
                          bytes=sum(r["dst_size"] for r in rs), failed=bad)), flush=True)
