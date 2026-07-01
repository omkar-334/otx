"""Track objects in a video with a getitune YOLOX detector and getitrack ByteTrack."""

from itertools import islice
from pathlib import Path

import torch
from getitune.backend.lightning.models.detection.yolox import YOLOX

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
OUTPUT = Path("videos/tracked/vehicles_getitrack_yolox.mp4")
MODEL_NAME = "yolox_tiny"

DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

MAX_FRAMES = 0
SCORE_THRESHOLD = 0.1

VERBOSE = True

# YOLOX predicts contiguous 80-class COCO ids (0-79); map them to names.
COCO80_CLASSES = dict(enumerate(COCO_CLASSES.values()))

model = YOLOX(label_info=80, model_name=MODEL_NAME).eval().to(DEVICE)
model.hparams["best_confidence_threshold"] = SCORE_THRESHOLD

adapter = GetiAdapter(model, device=DEVICE)
# Zero-shot models carry placeholder label names, so fall back to the COCO table.
class_names = adapter.class_names or COCO80_CLASSES

config = ByteTrackConfig(verbose=VERBOSE, score_threshold=SCORE_THRESHOLD)

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
