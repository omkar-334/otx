# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""ByteTrack multi-object tracker for OTX.

Detector-agnostic: works with any OTXDetectionModel (YOLOX, RTDETR, SSD, etc.).
The detector is passed to ``track()`` or ``track_frame()`` at call time.

Usage::

    from otx.backend.native.models.detection.yolox import YOLOX
    from otx.backend.native.models.tracking.bytetrack import ByteTrack

    model = YOLOX(label_info=80, ...)
    model.eval().to(device)

    tracker = ByteTrack(track_thresh=0.5, track_buffer=30, match_thresh=0.8)
    tracker.track(model, "video.mp4", output_dir="videos/tracked", device=device)
"""

from __future__ import annotations

from typing import Any

import numpy as np

from otx.backend.native.models.tracking.base import OTXTracker


# TODO: do we name this ByteTrack or ByteTrackTracker?
class ByteTrack(OTXTracker):
    """ByteTrack multi-object tracker.

    Uses two-stage association (high-confidence + low-confidence detections)
    with Kalman filter prediction for robust multi-object tracking.

    Args:
        track_thresh: Confidence threshold for primary association.
        track_buffer: Frames to keep lost tracks alive.
        match_thresh: IoU threshold for association.
        frame_rate: Video frame rate (used for track buffer calculation).
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
    ) -> None:
        super().__init__(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
        )
        self.frame_rate = frame_rate

    # for now i am relying on the original implementation for state management and core logic, so this is just a thin wrapper to adapt to our OTXDataBatch and numpy inputs/outputs
    # TODO: refactor this to implement logic ourselves
    def _create_tracker(self) -> Any:
        """Create a BYTETracker instance."""
        from .tracker.byte_tracker import BYTETracker

        class _Args:
            pass

        args = _Args()
        args.track_thresh = self.track_thresh
        args.track_buffer = self.track_buffer
        args.match_thresh = self.match_thresh
        args.mot20 = False

        return BYTETracker(args, frame_rate=self.frame_rate)

    def update(
        self,
        bboxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,  # TODO: we shud pass labels and class names here -> match each tracked bbox back to the input detections by IoU to recover the labels for display
        img_h: int,
        img_w: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Update ByteTrack with a single frame's detections.

        Returns:
            Tuple of (bboxes, scores, labels, track_ids) for active tracks.
        """
        if len(scores) == 0:
            return (
                np.empty((0, 4)),
                np.empty(0),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )

        # BYTETracker.update expects (N, 5): [x1, y1, x2, y2, score]
        dets = np.column_stack([bboxes, scores])
        online_targets = self.tracker.update(dets, [img_h, img_w], [img_h, img_w])

        out_bboxes = []
        out_scores = []
        out_labels = []
        out_ids = []

        for t in online_targets:
            x1, y1, w, h = t.tlwh
            out_bboxes.append([x1, y1, x1 + w, y1 + h])
            out_scores.append(t.score)
            out_ids.append(t.track_id)
            out_labels.append(0)

        if len(out_bboxes) == 0:
            return (
                np.empty((0, 4)),
                np.empty(0),
                np.empty(0, dtype=np.int64),
                np.empty(0, dtype=np.int64),
            )

        return (
            np.array(out_bboxes),
            np.array(out_scores),
            np.array(out_labels, dtype=np.int64),
            np.array(out_ids, dtype=np.int64),
        )
