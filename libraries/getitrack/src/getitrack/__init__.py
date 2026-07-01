# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""getitrack: Multi-object tracking toolkit."""

from loguru import logger as _logger

import getitrack.algorithms  # noqa: F401  -> registers the bundled algorithms on import
from getitrack.algorithms.bytetrack import ByteTrackConfig
from getitrack.config import AlgorithmType, InterpolationMethod, TrackerConfig
from getitrack.core import BaseTracker, Detections, Track, TrackedDetections, TrackState
from getitrack.io import VideoReader, VideoWriter
from getitrack.utils import COCO_CLASSES, to_h264
from getitrack.visualization import TrackAnnotator, VideoAnnotator, color_for_track

# Silent by default; the application opts in with ``logger.enable("getitrack")``.
_logger.disable("getitrack")

__version__ = "0.1.0"

__all__ = [
    "COCO_CLASSES",
    "AlgorithmType",
    "BaseTracker",
    "ByteTrackConfig",
    "Detections",
    "InterpolationMethod",
    "Track",
    "TrackAnnotator",
    "TrackState",
    "TrackedDetections",
    "TrackerConfig",
    "VideoAnnotator",
    "VideoReader",
    "VideoWriter",
    "color_for_track",
    "to_h264",
]
