/*
 * lrstat: per-frame loop-restoration statistics of an AV1 stream (IVF), via libaom's inspection hook.
 * Built in place of libaom's examples/inspect.c (CONFIG_INSPECTION=1), as the palette study's pstat.c.
 * Prints one JSON line per shown frame: the sequence header's enable_restoration, qindex, and per plane the
 * frame_restoration_type (0 none, 1 Wiener, 2 self-guided, 3 switchable), the restoration-unit size, the
 * number of units and how many of them use no filter / Wiener / self-guided.
 * usage: lrstat FILE.ivf
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
  fprintf(stderr, "usage: lrstat FILE.ivf\n");
  exit(EXIT_FAILURE);
}

static void inspect(void *pbi, void *data) {
  (void)data;
  ifd_inspect(&frame_data, pbi, 0);
  if (frame_data.show_existing_frame) return;
  const AV1Decoder *const d = (const AV1Decoder *)pbi;
  const AV1_COMMON *const cm = &d->common;
  const int np = cm->seq_params->monochrome ? 1 : 3;
  printf("{\"w\": %d, \"h\": %d, \"bitdepth\": %d, \"ss_x\": %d, \"ss_y\": %d, \"qindex\": %d, "
         "\"seq_lr\": %d, \"sct\": %d, \"cdef_bits\": %d, \"planes\": [",
         cm->superres_upscaled_width, cm->superres_upscaled_height, cm->seq_params->bit_depth,
         cm->seq_params->subsampling_x, cm->seq_params->subsampling_y, cm->quant_params.base_qindex,
         cm->seq_params->enable_restoration, cm->features.allow_screen_content_tools, cm->cdef_info.cdef_bits);
  for (int p = 0; p < np; p++) {
    const RestorationInfo *ri = &cm->rst_info[p];
    long cnt[3] = { 0, 0, 0 };
    if (ri->frame_restoration_type != RESTORE_NONE && ri->unit_info)
      for (int u = 0; u < ri->num_rest_units; u++) {
        const int t = ri->unit_info[u].restoration_type;
        if (t >= 0 && t < 3) cnt[t]++;
      }
    printf("%s{\"type\": %d, \"unit\": %d, \"units\": %d, \"none\": %ld, \"wiener\": %ld, \"sgr\": %ld}", p ? ", " : "",
           (int)ri->frame_restoration_type, ri->restoration_unit_size,
           ri->frame_restoration_type != RESTORE_NONE ? ri->num_rest_units : 0, cnt[0], cnt[1], cnt[2]);
  }
  printf("]}\n");
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
