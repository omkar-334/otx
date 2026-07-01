# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Minimal ByteTrack demo on a synthetic detection sequence.

Runs the tracker over generated detections (no video, no rendering) to show
that track ids stay stable across frames and that a track recovers its original
id after a missed detection. Uses only getitrack core and the ByteTrack
algorithm, so it runs without the video or visualization extras.
"""

from __future__ import annotations

import numpy as np

from getitrack.algorithms import ByteTrackTracker
from getitrack.algorithms.bytetrack import ByteTrackConfig
from getitrack.core.detection import Detections

# Two objects, each drifting right by a fixed step every frame.
_OBJECTS = {0: (10.0, 20.0), 1: (200.0, 60.0)}
_STEP = 8.0
_SIZE = 40.0
_DROP = {3: (0,)}  # object 0 is missed on frame 3; it should recover id 1 on frame 4


def _frame(frame_id: int) -> Detections:
    dropped = _DROP.get(frame_id, ())
    rows = [
        [x + frame_id * _STEP, y, x + frame_id * _STEP + _SIZE, y + _SIZE]
        for obj, (x, y) in _OBJECTS.items()
        if obj not in dropped
    ]
    boxes = np.asarray(rows, dtype=np.float32).reshape(len(rows), 4)
    return Detections(
        bboxes=boxes,
        scores=np.full(len(rows), 0.9, dtype=np.float32),
        class_ids=np.zeros(len(rows), dtype=np.int64),
        frame_id=frame_id,
    )


def main() -> None:
    """Track the synthetic sequence and print the active ids per frame."""
    tracker = ByteTrackTracker(ByteTrackConfig())
    for frame_id in range(6):
        tracked = tracker.update(_frame(frame_id))
        print(f"frame {frame_id}: ids={sorted(tracked.track_ids.tolist())}")


if __name__ == "__main__":
    main()
