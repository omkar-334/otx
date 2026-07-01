"""Track objects in a video with a getitune RF-DETR detector and getitrack ByteTrack."""

from itertools import islice
from pathlib import Path

import torch
from getitune.backend.lightning.models.detection.rfdetr import RFDETR
from getitune.backend.lightning.models.utils.utils import load_checkpoint

from getitrack import (
    COCO_CLASSES,
    BaseTracker,
    ByteTrackConfig,
    VideoAnnotator,
    VideoReader,
    to_h264,
)
from getitrack.adapters import GetiAdapter

VIDEO = Path("videos/vehicles.mp4")
OUTPUT = Path("videos/tracked/vehicles_getitrack.mp4")
MODEL_NAME = "rfdetr_small"

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

MAX_FRAMES = 0
SCORE_THRESHOLD = 0.4
VERBOSE = True


model = RFDETR(label_info=91, model_name=MODEL_NAME).eval().to(DEVICE)
# getitune builds RF-DETR for fine-tuning and reinitializes the classification
# head; reload the checkpoint to restore the trained COCO head for zero-shot use.
load_checkpoint(model.model.lwdetr, model._pretrained_weights[MODEL_NAME], map_location="cpu")  # noqa: SLF001
model.hparams["best_confidence_threshold"] = SCORE_THRESHOLD

adapter = GetiAdapter(model, device=DEVICE)
# Zero-shot models carry placeholder label names, so fall back to the COCO table.
class_names = adapter.class_names or COCO_CLASSES

config = ByteTrackConfig(verbose=VERBOSE, score_threshold=SCORE_THRESHOLD)
# Track vehicles only: config.class_filter = [2, 3, 4, 6, 8]

tracker = BaseTracker.from_config(config)

track_ids: set[int] = set()

with (
    VideoReader(VIDEO) as reader,
    VideoAnnotator(
        OUTPUT,
        fps=reader.fps or 30.0,
        frame_size=(reader.width, reader.height),
        class_names=class_names,
    ) as writer,
):
    for frame_id, frame in enumerate(islice(reader, MAX_FRAMES or None)):
        detections = adapter.detect(frame, frame_id)
        tracked = tracker.update(detections)
        track_ids.update(int(t) for t in tracked.track_ids)
        writer.write(frame, tracked)


to_h264(OUTPUT)


print(f"Tracked {len(track_ids)} objects over {writer.frames_written} frames -> {OUTPUT}")
