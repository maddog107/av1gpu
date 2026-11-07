import numpy as np
def chroma_up(C, w, h):
    """libjpeg-turbo h2v2 fancy upsampling of a (h+1)//2 x (w+1)//2 plane to w x h (the fix head's chroma_up), integers"""
    C = C.astype(np.int64); ch, cw = C.shape
    x = np.arange(w); y = np.arange(h)
    cx = np.minimum(x >> 1, cw - 1); cy = np.minimum(y >> 1, ch - 1)
    nx = np.clip(cx + np.where(x & 1, 1, -1), 0, cw - 1); ny = np.clip(cy + np.where(y & 1, 1, -1), 0, ch - 1)
    here = 3 * C[cy][:, cx] + C[ny][:, cx]; side = 3 * C[cy][:, nx] + C[ny][:, nx]
    return (3 * here + side + np.where(x & 1, 7, 8)[None, :]) >> 4

