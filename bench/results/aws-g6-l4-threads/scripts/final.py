import tthr
H = tthr.H
tthr.BIN.update(main2=f"{H}/bin/av1gpu-main2", reb=f"{H}/bin/av1gpu-reb")
T = ["--tiles", "2,2,2"]
for rd in range(2):
    for b in (("main2", "reb") if rd == 0 else ("reb", "main2")):
        for lst, a, e in (("jpeg", T, {}), ("jpeg", T, {"GP_INGEST_DECODER": "1"}), ("png", T, {}),
                          ("png", T + ["--yuv", "444", "--depth", "10"], {}), ("mix", T, {}), ("png", T + ["--lossless"], {})):
            r = tthr.run(b, lst, a, e, tag="fin")
            r["round"] = rd
            tthr.log("final", r)
for lst in ("fewpng", "fewjpg"):
    tthr.log("final_hang", tthr.run("reb", lst, env={"GP_ENCODERS": "24"}, timeout=60, tag="hang", hash_=False))
print("FINAL_DONE", flush=True)
