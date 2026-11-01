import sys, tthr
n = tthr.N
T = ["--tiles", "2,2,2"]
P4 = T + ["--yuv", "444", "--depth", "10"]
cs = []
for lst, a in (("mix", T), ("png", T), ("png", P4)):
    cs += [("main", lst, a, {}), ("new", lst, a, {}), ("new", lst, a, {"GP_FALLBACK_NICE": "10"}), ("new", lst, a, {"GP_FALLBACK_NICE": "19"})]
for rd in range(int(sys.argv[1])):
    for b, l, a, e in (cs if rd % 2 == 0 else cs[::-1]):
        r = tthr.run(b, l, a, e, tag="cf")
        r["round"] = rd
        tthr.log("confirm", r)
print("CONFIRM_DONE", flush=True)
