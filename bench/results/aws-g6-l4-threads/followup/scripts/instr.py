"""instr.py FILE: add GP_BATCH_LOG instrumentation (batch composition + encode/entropy intervals) to batch.cpp"""
import sys
p = sys.argv[1]
s = open(p).read()
def rep(a, b):
    global s
    assert s.count(a) == 1, a[:60]
    s = s.replace(a, b)
rep('''    Agg A;
    int failures = 0;''', '''    Agg A;
    int failures = 0;
    // GP_BATCH_LOG=FILE (diagnostic): one line per encoder batch (composition, gather / encode times) and per entropy job
    struct BLog { char kind; int enc; double tg, t0, t1, twIng; int nfb; double fbMp; int nIng, nHeld; double ingMp, maxMp, minMp; int nSmall, cls; };
    std::vector<BLog> blog;
    std::atomic<int> encIdx{0};
    const double tBase = now_ms();''')
rep('''                    const double tm = now_ms();
                    uint64_t bytes = 0;''', '''                    const double tm = now_ms();
                    {
                        std::lock_guard<std::mutex> lk3(aggMu);
                        blog.push_back({'N', -1, te - tBase, te - tBase, tm - tBase, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0});
                    }
                    uint64_t bytes = 0;''')
rep('''        std::string eerr;
        AlphaEnc ea;
        for (;;) {
            std::vector<ingest_batch> held;''', '''        std::string eerr;
        AlphaEnc ea;
        const int myIdx = encIdx++;
        for (;;) {
            const double tg = now_ms();
            double twIng = 0;
            std::vector<ingest_batch> held;''')
rep('''                        A.ingestWait += now_ms() - tw;''', '''                        A.ingestWait += now_ms() - tw;
                        twIng += now_ms() - tw;''')
rep('''            GpTimes tm;
            const double ts = now_ms();''', '''            GpTimes tm;
            const double ts = now_ms();
            BLog bl{'E', myIdx, tg - tBase, ts - tBase, 0, twIng, 0, 0, 0, (int)held.size(), 0, 0, 1e9, 0, -1};
            {
                const int nfb = (int)fbHeld.size() - fbConv;
                for (size_t i = 0; i < fin.size(); i++) {
                    const double m = fin[i].W * (double)fin[i].H / 1e6;
                    if ((int)i < nfb) { bl.nfb++; bl.fbMp += m; } else { bl.nIng++; bl.ingMp += m; }
                    bl.maxMp = std::max(bl.maxMp, m); bl.minMp = std::min(bl.minMp, m);
                    if (m < bcfg.smallMp) bl.nSmall++;
                }
                for (auto &b : held) bl.cls = bl.cls < 0 ? b.cls : (bl.cls == b.cls ? bl.cls : 9);
            }''')
rep('''            push(job);
        }
    };''', '''            bl.t1 = now_ms() - tBase;
            {
                std::lock_guard<std::mutex> lk3(aggMu);
                blog.push_back(bl);
            }
            push(job);
        }
    };''')
rep('''    if (stats)
        printf(''', '''    if (const char *bf = getenv("GP_BATCH_LOG"))
        if (FILE *f = fopen(bf, "w")) {
            for (auto &b : blog)
                fprintf(f, "%c %d %.2f %.2f %.2f %.2f %d %.3f %d %d %.3f %.3f %.3f %d %d\\n", b.kind, b.enc, b.tg, b.t0, b.t1, b.twIng, b.nfb,
                        b.fbMp, b.nIng, b.nHeld, b.ingMp, b.maxMp, b.minMp, b.nSmall, b.cls);
            fprintf(f, "W %.2f\\n", wall);
            fclose(f);
        }
    if (stats)
        printf(''')
open(p, "w").write(s)
print("ok")
