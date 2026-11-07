#!/usr/bin/env python3
"""suggested fixes, applied to a throw-away copy on the instance (~/prfix) to prove them"""
import re, os
R = os.path.expanduser("~/prfix/encoder/src/")
def sub(f, old, new, n=1):
    p = R + f if not f.startswith("/") else f
    s = open(p).read()
    assert s.count(old) >= 1, old
    s = s.replace(old, new) if n == 0 else s.replace(old, new, n)
    open(p, "w").write(s)
# 1: percent geometries carry both sides, so turn() and x50% work; AREA is not limited to 2^20; whole sides as IM rounds them
sub("pipeline.cpp", """        return e && !*e && d > 0 && d <= (1 << 20);
    };""", """        return e && !*e && d > 0 && d <= ((f & GpXform::RS_AREA) ? 1e12 : 1 << 20);
    };""")
sub("pipeline.cpp", """    x.rsFlags = f;
    x.rsW = w;
    x.rsH = h;
""", """    if (f & GpXform::RS_PCT) {   // ImageMagick: N% scales both sides, xN% only the height
        if (!(f & GpXform::RS_W)) w = 100;
        if (!(f & GpXform::RS_H)) h = w;
        f |= GpXform::RS_W | GpXform::RS_H;
    }
    x.rsFlags = f;
    x.rsW = w;
    x.rsH = h;
""")
sub("pipeline.cpp", """            tw = (int)x.rsW;
            th = (int)x.rsH;
""", """            tw = rnd(x.rsW);
            th = rnd(x.rsH);
""")
sub("pipeline.cpp", """            tw = (int)x.rsW;
            th = rnd(h * (x.rsW / w));
""", """            tw = rnd(x.rsW);
            th = rnd(h * (std::floor(x.rsW + 0.5) / w));
""")
sub("pipeline.cpp", """            th = (int)x.rsH;
            tw = rnd(w * (x.rsH / h));
""", """            th = rnd(x.rsH);
            tw = rnd(w * (std::floor(x.rsH + 0.5) / h));
""")
# 2: a size that cannot be encoded is an error of this image, not of the list
sub("batch.cpp", """        if ((double)*ow * *oh <= encMaxMp * 1e6) return "";
""", """        if (*ow != W || *oh != H) {   // AV1 sides, and the resize kernels' grid (aligned height <= 65535 rows)
            if (*ow > 65536 || ((*oh + 63) & ~63) > 65535)
                return std::to_string(*ow) + "x" + std::to_string(*oh) + " after crop / resize is too large for AV1 / the resize";
        }
        if ((double)*ow * *oh <= encMaxMp * 1e6) return "";
""")
sub("batch.cpp", """                    gp_xform_size(xfin[i], fin[i].W, fin[i].H, &ow, &oh);
                    if ((double)ow * oh > encMaxMp * 1e6) {""", """                    gp_xform_size(xfin[i], fin[i].W, fin[i].H, &ow, &oh);
                    if (ow > 65536 || ((oh + 63) & ~63) > 65535 || (double)ow * oh > encMaxMp * 1e6) {""")
sub("/home/ubuntu/prfix/encoder/apps/av1gpu.cpp", """        std::vector<GpFrameIn> v = {fin}, al = {afin};
        if (afin.W > 0) v.push_back(afin);
""", """        if ((ow != fin.W || oh != fin.H) && (ow > 65536 || ((oh + 63) & ~63) > 65535)) {
            fprintf(stderr, "%s: %dx%d after crop / resize is too large for AV1 / the resize\\n", o.in.c_str(), ow, oh);
            return 3;
        }
        std::vector<GpFrameIn> v = {fin}, al = {afin};
        if (afin.W > 0) v.push_back(afin);
""")
# 3: 65536 rows do not fit gridDim.y: the rows of k_tone / k_gray over y and z
sub("k_imageops.cu", """    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y;
    if (x >= aw || y >= ah) return;
    const Rgba p = trim_px<T>(f, x, y);""", """    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y + blockIdx.z * gridDim.y;
    if (x >= aw || y >= ah) return;
    const Rgba p = trim_px<T>(f, x, y);""")
sub("k_imageops.cu", """    const int cx = blockIdx.x * blockDim.x + threadIdx.x, cy = blockIdx.y;
""", """    const int cx = blockIdx.x * blockDim.x + threadIdx.x, cy = blockIdx.y + blockIdx.z * gridDim.y;
""")
sub("k_imageops.cu", """    const dim3 grid(((aw >> s) + 127) / 128, ah >> s);
""", """    const dim3 grid(((aw >> s) + 127) / 128, std::min(ah >> s, 32768), ((ah >> s) + 32767) / 32768);
""")
sub("k_imageops.cu", """    const dim3 grid((aw + 127) / 128, ah);
""", """    const dim3 grid((aw + 127) / 128, std::min(ah, 32768), (ah + 32767) / 32768);
""")
# 4: a box that crosses over (corners of two colours) is not a blank image
sub("imageops.cpp", """    if ((b.right < 0 && b.bottom < 0 && b.left == INT_MAX) || bw <= 0 || bh <= 0) bw = bh = 0;
""", """    const bool nothing = b.right < 0 && b.bottom < 0 && b.left == INT_MAX;   // every pixel within --fuzz of every corner
    if (!nothing && (bw <= 0 || bh <= 0)) return GP_TRIM_WHOLE;              // the box crossed over: not blank, nothing to trim
    if (nothing) bw = bh = 0;
""")
print("patched")
