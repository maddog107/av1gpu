import sys, tthr
T = ["--tiles", "2,2,2"]
P4 = T + ["--yuv", "444", "--depth", "10"]
cs = [("main", "mix", T, {}), ("new", "mix", T, {}), ("new", "mix", T, {"GP_FALLBACK_QUEUE_MP": "24"}),
      ("new", "mix", T, {"GP_FALLBACK_QUEUE_MP": "48"}), ("new", "mix", T, {"GP_FALLBACK_QUEUE_MP": "12"}),
      ("new", "png", T, {}), ("new", "png", T, {"GP_FALLBACK_QUEUE_MP": "24"}), ("new", "png", T, {"GP_FALLBACK_QUEUE_MP": "48"}),
      ("new", "png", P4, {}), ("new", "png", P4, {"GP_FALLBACK_QUEUE_MP": "24"})]
for rd in range(int(sys.argv[1])):
    for b, l, a, e in (cs if rd % 2 == 0 else cs[::-1]):
        r = tthr.run(b, l, a, e, tag="q")
        r["round"] = rd
        tthr.log("qtest", r)
print("QTEST_DONE", flush=True)
