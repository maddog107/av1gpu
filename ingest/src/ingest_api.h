/* av1gpu ingest: JPEG -> GPU-resident 4:2:0 / 4:4:4 / mono planes.
 *
 * Plain C ABI (fixed-layout structs), so it can also be bound from other languages, e.g. ctypes.
 *
 * Data flow
 *   ingest_submit(paths) -> N CPU workers: read file, parse header/markers, route:
 *     - hardware route (decoder=0, structurally clean baseline Huffman 420/422/440/444/gray):
 *       keep the bitstream; the batcher decodes it with nvJPEG on the GPU's NVJPG engine
 *     - CPU route (everything else): libjpeg-turbo entropy decode with the IDCT replaced by a
 *       hook that stores the quantized DCT blocks straight into a pinned arena; one H2D per image
 *       and GPU dequant + islow IDCT
 *   Both routes produce native planes bit-exact with libjpeg-turbo's islow decode.  A finalize
 *   kernel then converts native sampling -> 4:2:0 (or mono), or with cfg.out_444 to full-resolution
 *   chroma (4:4:4 sources as coded, subsampled ones upsampled exactly like libjpeg-turbo's decoder:
 *   jdsample.c with do_fancy_upsampling), replicates edges, optionally applies the EXIF rotation, and
 *   writes the output pool + activity maps.  Batches are packed by size class; ingest_next_batch()
 *   hands them out in dispatch order.
 *
 * Output layout (one device allocation per batch, "pool"):
 *   for each image i:  Y   at pool + y_off,  aw x ah bytes, row pitch y_pitch
 *                      Cb  at pool + cb_off, aw/2 x ah/2,   row pitch c_pitch   (cb_off = -1 for mono;
 *                                            aw x ah when chroma_full = 1, i.e. cfg.out_444)
 *                      Cr  at pool + cr_off, same as Cb
 *                      ACT at pool + act_off, float4 [act_h][act_w] per 8x8 luma block (if enabled)
 *   width/height = visible size; w8/h8 = padded to 8 (the AV1 coded frame area);
 *   aw/ah = padded to cfg.align (default 64 = superblock).  Every byte of the aw x ah area is valid:
 *   pixels outside width x height replicate the nearest visible edge pixel.
 *   All offsets are multiples of 256, pitches multiples of max(align,8) (luma) / half that (chroma).
 */
#ifndef GPU_JPEG_INGEST_API_H
#define GPU_JPEG_INGEST_API_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define INGEST_ABI_VERSION 6   /* 6: ingest_config.out_444, ingest_image.chroma_full */

/* source subsampling codes (ingest_image.src_sub) */
enum {
  INGEST_SUB_420 = 0, INGEST_SUB_444 = 1, INGEST_SUB_422 = 2, INGEST_SUB_440 = 3,
  INGEST_SUB_411 = 4, INGEST_SUB_GRAY = 5, INGEST_SUB_OTHER = 6
};

/* size buckets (ingest_image.bucket) */
enum {
  INGEST_BUCKET_TINY = 0,    /* < 0.3 MP */
  INGEST_BUCKET_SMALL = 1,   /* 0.3 - 1 MP */
  INGEST_BUCKET_TYPICAL = 2, /* 1 - 4 MP */
  INGEST_BUCKET_LARGE = 3,   /* 4 - 9.4 MP, single tile */
  INGEST_BUCKET_HUGE = 4     /* needs AV1 tiles (width > 4096 or > 2304 superblocks); <= max_mp */
};

/* batch classes (ingest_batch.cls) */
enum { INGEST_CLASS_SMALL = 0, INGEST_CLASS_REGULAR = 1, INGEST_CLASS_TILED = 2 };

/* fallback reasons (-> CPU avifenc) */
enum {
  INGEST_FB_NONE = 0,
  INGEST_FB_READ_ERROR = 1,   /* could not read file */
  INGEST_FB_DECODE_ERROR = 2, /* libjpeg error (not a JPEG, lossless, truncated header, ...) */
  INGEST_FB_GIANT = 3,        /* > max_mp */
  INGEST_FB_CMYK = 4,         /* CMYK or YCCK */
  INGEST_FB_COLORSPACE = 5,   /* RGB-coded / unknown colour space / unexpected component count */
  INGEST_FB_PRECISION = 6,    /* not 8-bit */
  INGEST_FB_TOO_BIG = 7       /* does not fit the pinned arena / device budget */
};

typedef struct ingest_config {
  int n_workers;          /* CPU entropy-decode threads (default 2) */
  int device;             /* CUDA device (default 0) */
  int max_inflight;       /* batches dispatched but not yet released (default 3) */
  int max_wait_ms;        /* dispatch a partially filled batch after this long (default 30) */
  int max_batch_images;   /* per batch (default 256) */
  int apply_orientation;  /* 0 (default): keep stored orientation, signal irot/imir in AVIF; 1: rotate on GPU */
  int activity;           /* 1 (default): emit per-8x8 activity maps */
  int keep_native;        /* debug: keep native (un-resampled) planes until release */
  int align;              /* luma alloc alignment, power of 2 >= 8 (default 64) */
  int nice_workers;       /* setpriority() for worker threads (default 19) */
  int legacy_coef;        /* 1: jpeg_read_coefficients + copy instead of the IDCT-hook fast path */
  int decoder;            /* 0 (default) auto: baseline JPEGs -> nvJPEG on the GPU's hardware JPEG engine
                             (NVJPG), everything else -> CPU entropy decode + GPU IDCT; 1: CPU path only */
  int hw_batch;           /* max images per nvJPEG hardware call (default 32) */
  int out_444;            /* 0 (default): 4:2:0 output; 1: full-resolution chroma (4:4:4 output); 2: the source's
                             format: 4:4:4 JPEGs full-resolution chroma, the rest 4:2:0 (4:2:0 JPEGs: the native
                             chroma samples exactly, plus edge replication; lossless encoding) */
  int64_t pinned_bytes;   /* pinned staging arena (default 512 MiB) */
  int64_t batch_pixels;   /* luma pixels per batch, small/regular classes (default 16e6) */
  double max_mp;          /* fallback above this (default 35.6) */
  double small_mp;        /* images below this go to the SMALL class (default 1.0) */
} ingest_config;

typedef struct ingest_image {
  int32_t index;              /* position in the submitted list */
  int32_t width, height;      /* visible size (after orientation if applied) */
  int32_t w8, h8;             /* padded to 8 */
  int32_t aw, ah;             /* allocated / filled luma size (multiple of align) */
  int32_t y_pitch, c_pitch;
  int64_t y_off, cb_off, cr_off;   /* byte offsets into pool; cb_off = cr_off = -1 for mono */
  int64_t act_off;                 /* float4 map offset, -1 if disabled */
  int32_t act_w, act_h;            /* = w8/8, h8/8 */
  int32_t mono, src_sub;
  int32_t orientation;        /* EXIF orientation (1..8, 1 if absent) */
  int32_t oriented;           /* 1 if the pixels were rotated on GPU */
  int32_t quality;            /* nearest IJG quality of the quant tables (1..100) */
  int32_t quality_exact;      /* 1 if the tables are exactly IJG-scaled */
  int32_t progressive, arithmetic, restart_interval, warnings;
  int32_t bucket, tiled, tile_cols_log2, tile_rows_log2;
  int32_t icc_size, exif_size;
  int32_t decoder_used;       /* 0 CPU entropy decode + GPU IDCT, 1 NVJPG hardware, 2 hardware failed -> CPU */
  int32_t chroma_full;        /* 1: Cb/Cr planes are full size (aw x ah, cfg.out_444); 0: aw/2 x ah/2 (or mono) */
  const uint8_t *icc;         /* host pointers, valid until ingest_release_batch */
  const uint8_t *exif;        /* APP1 payload after "Exif\0\0" (TIFF header first) */
  float cpu_ms_read, cpu_ms_decode, cpu_ms_copy;   /* worker thread CPU time */
  /* native planes (keep_native only; otherwise offsets -1). In the batch "native" buffer. */
  int32_t ncomp;
  int32_t h_samp[3], v_samp[3];
  int32_t nat_w[3], nat_h[3], nat_pitch[3];   /* coded size = width_in_blocks*8 x height_in_blocks*8 */
  int64_t nat_off[3];
  uint16_t qt[3][64];         /* dequant tables per component, natural order */
} ingest_image;

typedef struct ingest_batch {
  int64_t batch_id;
  int32_t n_images;
  int32_t cls;
  uint8_t *pool;              /* device pointer */
  int64_t pool_bytes;
  uint8_t *native;            /* device pointer (keep_native only) */
  int64_t native_bytes;
  ingest_image *images;       /* host array [n_images], valid until release */
  void *stream;               /* cudaStream_t the batch was produced on (already synchronized) */
  float h2d_ms, gpu_ms;       /* device time: H2D copies, all decode/finalize work after the copies */
  float hw_ms;                /* part of gpu_ms spent in nvJPEG hardware decode calls */
  int32_t n_hw;               /* images decoded by the hardware engine */
  int64_t coef_bytes;         /* bytes moved H2D */
  int64_t total_pixels;
} ingest_batch;

typedef struct ingest_stats {
  int64_t submitted, decoded, fallbacks, batches, released;
  double worker_cpu_ms_read, worker_cpu_ms_decode, worker_cpu_ms_copy;
  double gpu_ms, h2d_ms;
  int64_t pixels, coef_bytes;
  int64_t pinned_peak, device_peak;
  int64_t n_hw, n_hw_failed;
  double hw_ms;
  double batcher_cpu_ms;      /* CPU time of the batcher thread (descriptor build, CUDA + nvJPEG host work) */
  int32_t hw_available, hw_engines;
} ingest_stats;

typedef struct ingest_fallback {
  int32_t index;
  int32_t reason;
  int32_t width, height;
} ingest_fallback;

typedef struct ingest_ctx ingest_ctx;

int ingest_abi_version(void);
/* sizeof of the ABI structs, for binding self-checks: 0 config, 1 image, 2 batch, 3 stats, 4 fallback */
int ingest_sizeof(int which);
void ingest_default_config(ingest_config *cfg);
ingest_ctx *ingest_create(const ingest_config *cfg, char *err, int errlen);
/* Queue files (copied).  May be called repeatedly until ingest_finish_submit. */
int ingest_submit(ingest_ctx *c, const char *const *paths, int n);
void ingest_finish_submit(ingest_ctx *c);
/* 1 = batch returned, 0 = all done, -1 = timeout (timeout_ms < 0 waits forever), -2 error */
int ingest_next_batch(ingest_ctx *c, ingest_batch *out, int timeout_ms);
/* Frees the batch's device memory, ordered after work already queued on `stream`
 * (NULL = the batch's own stream). */
void ingest_release_batch(ingest_ctx *c, int64_t batch_id, void *stream);
/* Copy up to cap fallback records; returns the total count. */
int ingest_get_fallbacks(ingest_ctx *c, ingest_fallback *out, int cap);
void ingest_get_stats(ingest_ctx *c, ingest_stats *out);
/* Copies the first pipeline error message (empty if none); returns its length. */
int ingest_last_error(ingest_ctx *c, char *buf, int len);
void ingest_destroy(ingest_ctx *c);

/* ---- reference / utility entry points (CPU, used by tests and benches) ---- */

/* libjpeg-turbo jpeg_read_raw_data (islow): native planes, each comp cropped to its coded
 * area (width_in_blocks*8 x height_in_blocks*8), packed one after another.  info[0]=ncomp,
 * info[1]=W, info[2]=H, then for each comp: w, h, h_samp, v_samp.  Returns bytes written
 * or <0 on error (-needed if cap is too small). */
int64_t ingest_ref_decode_raw(const uint8_t *jpg, int64_t n, uint8_t *out, int64_t cap, int32_t *info);

/* Header-only probe (no entropy decode): fills an ingest_image's metadata fields
 * (width/height/sub/quality/orientation/progressive/restart/icc_size/exif_size/...);
 * icc/exif pointers are NULL.  Returns fallback reason (0 = would be handled on GPU). */
int ingest_probe(const uint8_t *jpg, int64_t n, double max_mp, ingest_image *out);

/* IJG quality estimate from a luma (and optional chroma) table in natural order. */
int ingest_estimate_quality(const uint16_t *luma, const uint16_t *chroma, int *exact);

#ifdef __cplusplus
}
#endif
#endif
