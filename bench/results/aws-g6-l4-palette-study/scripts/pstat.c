/*
 * pstat: per-frame screen-content statistics of an AV1 stream (IVF), via libaom's inspection hook.
 * Built in place of libaom's examples/inspect.c (CONFIG_INSPECTION=1), so it links like that example.
 * Prints one JSON line per shown frame: frame-header flags (allow_screen_content_tools, allow_intrabc,
 * lossless, loop filter / CDEF / LR state) and the share of 4x4 units in blocks that use a luma palette,
 * a chroma palette or IntraBC, plus the palette sizes used.
 * usage: pstat FILE.ivf
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "config/aom_config.h"

#include "aom/aom_decoder.h"
#include "aom/aomdx.h"
#include "av1/common/av1_common_int.h"
#include "av1/decoder/decoder.h"
#include "av1/decoder/inspection.h"
#include "common/tools_common.h"
#include "common/video_reader.h"

static insp_frame_data frame_data;
static aom_codec_ctx_t codec;

void usage_exit(void) {   // tools_common.c's die() calls it
  fprintf(stderr, "usage: pstat FILE.ivf\n");
  exit(EXIT_FAILURE);
}

static void inspect(void *pbi, void *data) {
  (void)data;
  ifd_inspect(&frame_data, pbi, 0);
  if (frame_data.show_existing_frame) return;
  const AV1Decoder *const d = (const AV1Decoder *)pbi;
  const AV1_COMMON *const cm = &d->common;
  const CommonModeInfoParams *const mp = &cm->mi_params;
  long n = 0, pal_y = 0, pal_uv = 0, ibc = 0, pal_y_hist[9] = { 0 };
  for (int r = 0; r < mp->mi_rows; r++) {
    for (int c = 0; c < mp->mi_cols; c++) {
      const insp_mi_data *mi = &frame_data.mi_grid[r * mp->mi_cols + c];
      n++;
      if (mi->palette > 0) {
        pal_y++;
        if (mi->palette <= 8) pal_y_hist[mi->palette]++;
      }
      if (mi->uv_palette > 0) pal_uv++;
      if (mi->intrabc) ibc++;
    }
  }
  int lr = 0;
  for (int p = 0; p < 3; p++)
    lr |= (cm->rst_info[p].frame_restoration_type != RESTORE_NONE) << p;
  printf("{\"w\": %d, \"h\": %d, \"bitdepth\": %d, \"ss_x\": %d, \"ss_y\": %d, \"qindex\": %d, "
         "\"sct\": %d, \"ibc_allowed\": %d, \"coded_lossless\": %d, \"lf\": [%d, %d, %d, %d], "
         "\"cdef_bits\": %d, \"cdef_y0\": %d, \"lr\": %d, \"mi\": %ld, \"pal_y\": %ld, \"pal_uv\": %ld, "
         "\"ibc\": %ld, \"pal_y_sizes\": [%ld, %ld, %ld, %ld, %ld, %ld, %ld]}\n",
         cm->superres_upscaled_width, cm->superres_upscaled_height, cm->seq_params->bit_depth,
         cm->seq_params->subsampling_x, cm->seq_params->subsampling_y, cm->quant_params.base_qindex,
         cm->features.allow_screen_content_tools, cm->features.allow_intrabc, cm->features.coded_lossless,
         cm->lf.filter_level[0], cm->lf.filter_level[1], cm->lf.filter_level_u, cm->lf.filter_level_v,
         cm->cdef_info.cdef_bits, cm->cdef_info.cdef_strengths[0], lr, n, pal_y, pal_uv, ibc, pal_y_hist[2],
         pal_y_hist[3], pal_y_hist[4], pal_y_hist[5], pal_y_hist[6], pal_y_hist[7], pal_y_hist[8]);
}

int main(int argc, char **argv) {
  if (argc != 2) {
    fprintf(stderr, "usage: %s FILE.ivf\n", argv[0]);
    return 2;
  }
  AvxVideoReader *reader = aom_video_reader_open(argv[1]);
  if (!reader) die("Failed to open %s for reading.", argv[1]);
  const AvxVideoInfo *info = aom_video_reader_get_info(reader);
  aom_codec_iface_t *decoder = get_aom_decoder_by_fourcc(info->codec_fourcc);
  if (!decoder) die("Unknown input codec.");
  if (aom_codec_dec_init(&codec, decoder, NULL, 0)) die("Failed to initialize decoder.");
  ifd_init(&frame_data, info->frame_width, info->frame_height);
  aom_inspect_init ii;
  ii.inspect_cb = inspect;
  ii.inspect_ctx = NULL;
  aom_codec_control(&codec, AV1_SET_INSPECTION_CALLBACK, &ii);
  while (aom_video_reader_read_frame(reader)) {
    size_t frame_size = 0;
    const unsigned char *frame = aom_video_reader_get_frame(reader, &frame_size);
    const unsigned char *end = frame + frame_size;
    // a non-NULL user_priv selects libaom's inspection path (one frame per call, the callback fires per frame)
    while (frame < end) {
      Av1DecodeReturn adr;
      if (aom_codec_decode(&codec, frame, (size_t)(end - frame), &adr) != AOM_CODEC_OK) {
        fprintf(stderr, "decode failed: %s\n", aom_codec_error(&codec));
        return 1;
      }
      if (adr.buf <= frame) break;
      frame = adr.buf;
    }
  }
  ifd_clear(&frame_data);
  aom_codec_destroy(&codec);
  aom_video_reader_close(reader);
  return 0;
}
