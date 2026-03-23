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


class ByteTrack(OTXTracker):
    """ByteTrack multi-object tracker.

    Uses two-stage association (high-confidence + low-confidence detections)
    with Kalman filter prediction for robust multi-object tracking.

    Args:
        track_thresh: Confidence threshold for primary (high-confidence) association.
        track_buffer: Frames to keep lost tracks alive before removal.
        match_thresh: IoU threshold for matching detections to existing tracks.
        frame_rate: Video frame rate (used for track buffer scaling).
        low_thresh: Minimum confidence for second-stage (low-confidence) association.
            Detections between low_thresh and track_thresh are used in the second
            matching stage to recover occluded tracks.
        new_track_thresh: Minimum confidence to initialize a new track.
            Defaults to track_thresh + 0.1 to avoid creating tracks from noise.
        second_match_thresh: IoU threshold for second-stage association.
        unconfirmed_match_thresh: IoU threshold for matching unconfirmed
            (single-frame) tracks.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
        frame_rate: int = 30,
        low_thresh: float = 0.1,
        new_track_thresh: float | None = None,
        second_match_thresh: float = 0.5,
        unconfirmed_match_thresh: float = 0.7,
    ) -> None:
        super().__init__(
            track_thresh=track_thresh,
            track_buffer=track_buffer,
            match_thresh=match_thresh,
        )
        self.frame_rate = frame_rate
        self.low_thresh = low_thresh
        self.new_track_thresh = new_track_thresh if new_track_thresh is not None else track_thresh + 0.1
        self.second_match_thresh = second_match_thresh
        self.unconfirmed_match_thresh = unconfirmed_match_thresh

    def _create_tracker(self) -> Any:  # noqa: ANN401
        """Create a BYTETracker instance."""
        from types import SimpleNamespace

        from .tracker.byte_tracker import BYTETracker

        args = SimpleNamespace(
            track_thresh=self.track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            mot20=False,
            low_thresh=self.low_thresh,
            new_track_thresh=self.new_track_thresh,
            second_match_thresh=self.second_match_thresh,
            unconfirmed_match_thresh=self.unconfirmed_match_thresh,
        )
        return BYTETracker(args, frame_rate=self.frame_rate)

    @staticmethod
    def _compute_iou(bbox: np.ndarray, bboxes: np.ndarray) -> np.ndarray:
        """Compute IoU between one bbox and an array of bboxes.

        Args:
            bbox: Single box [x1, y1, x2, y2].
            bboxes: (N, 4) array of [x1, y1, x2, y2].

        Returns:
            (N,) array of IoU values.
        """
        x1 = np.maximum(bbox[0], bboxes[:, 0])
        y1 = np.maximum(bbox[1], bboxes[:, 1])
        x2 = np.minimum(bbox[2], bboxes[:, 2])
        y2 = np.minimum(bbox[3], bboxes[:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_a = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        area_b = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
        return inter / (area_a + area_b - inter + 1e-6)

    def update(
        self,
        bboxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
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
            t_bbox = np.array([x1, y1, x1 + w, y1 + h])
            out_bboxes.append(t_bbox)
            out_scores.append(t.score)
            out_ids.append(t.track_id)

            # Recover class label by matching tracked bbox to input detections via IoU
            if len(bboxes) > 0:
                ious = self._compute_iou(t_bbox, bboxes)
                out_labels.append(labels[ious.argmax()])
            else:
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
