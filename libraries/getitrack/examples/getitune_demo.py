# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Track real videos with a getitune RF-DETR detector and getitrack ByteTrack.

Builds a getitune RF-DETR model, wraps it with getitrack's `GetiAdapter`, and
runs detection plus tracking over a video, writing an annotated H.264 result.
Classes are selected with `ByteTrackConfig.class_filter`. Run directly to
process the demo videos.

Requires ``getitune`` and its dependencies. Run with an environment that has
both ``getitune`` and ``getitrack`` installed.
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import torch

from getitrack import BaseTracker, ByteTrackConfig, TrackAnnotator, to_h264
from getitrack.adapters import DetectionAdapter, GetiAdapter
from getitrack.io import VideoReader, VideoWriter
from getitrack.utils import COCO_CLASSES

# COCO class ids per demo video.
CLASS_SETS = {
    "bikes": [2, 4],  # bicycle, motorcycle
    "jets": [5],  # airplane
    "apples": [53],  # apple
}


def build_detector(
    model_name: str = "rfdetr_small",
    score_thresh: float = 0.4,
    device: str | None = None,
) -> GetiAdapter:
    """Build a getitune RF-DETR detector wrapped as a getitrack `GetiAdapter`."""
    from getitune.backend.lightning.models.detection.rfdetr import RFDETR
    from getitune.backend.lightning.models.utils.utils import load_checkpoint

    device = device or ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    model = RFDETR(label_info=91, model_name=model_name).eval().to(device)
    # getitune reinitializes the classification head; reload the COCO weights.
    load_checkpoint(model.model.lwdetr, model._pretrained_weights[model_name], map_location="cpu")  # noqa: SLF001
    model.hparams["best_confidence_threshold"] = score_thresh
    return GetiAdapter(model, device=device)


def track_video(
    video: str | Path,
    class_ids: list[int],
    output: str | Path,
    adapter: DetectionAdapter,
    *,
    width: int = 960,
    max_frames: int = 0,
) -> Path:
    """Detect and track ``class_ids`` over ``video``, writing an annotated H.264 clip.

    Returns the path to the final H.264 video.
    """
    video, output = Path(video), Path(output)
    class_names = adapter.class_names or COCO_CLASSES
    tracker = BaseTracker.from_config(ByteTrackConfig(class_filter=class_ids))
    annotator = TrackAnnotator(show_score=True, class_names=class_names)

    with VideoReader(video) as reader:
        size = (width, round(reader.height * width / reader.width))
        raw = output.with_name(f"{output.stem}.raw.mp4")
        track_ids: set[int] = set()
        started = time.time()
        with VideoWriter(raw, fps=reader.fps or 30.0, frame_size=size) as writer:
            for frame_id, frame in enumerate(reader):
                if max_frames and frame_id >= max_frames:
                    break
                resized = cv2.resize(frame, size)
                tracked = tracker.update(adapter.detect(resized, frame_id))
                track_ids.update(int(t) for t in tracked.track_ids)
                writer.write(annotator.annotate(resized, tracked))
                if frame_id % 25 == 0:
                    print(f"  frame {frame_id}: {len(tracked)} tracks")
        n_frames = writer.frames_written

    final = to_h264(raw, output)
    raw.unlink(missing_ok=True)
    print(f"{video.name}: {n_frames} frames, {len(track_ids)} unique ids, {time.time() - started:.0f}s -> {final}")
    return final


def main() -> None:
    """Track the demo videos into ``results/`` with one shared detector."""
    root = Path(__file__).resolve().parents[1]
    results = root / "results"
    results.mkdir(exist_ok=True)
    adapter = build_detector()
    for name, class_ids in CLASS_SETS.items():
        src = next(root.glob(f"videos/{name}*.mp4"))
        track_video(src, class_ids, results / f"{name}_tracked.mp4", adapter)


if __name__ == "__main__":
    main()
