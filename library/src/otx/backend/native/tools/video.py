# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Video processing utilities for OTX models (detection, tracking, etc.)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import torch

from otx.data.entity.base import ImageInfo
from otx.data.entity.torch import OTXDataBatch

if TYPE_CHECKING:
    from otx.backend.native.models.base import OTXModel

# i added 20 distinct colors for visualization
TRACK_COLORS = [
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (128, 0, 0),
    (0, 128, 0),
    (0, 0, 128),
    (128, 128, 0),
    (0, 128, 128),
    (128, 0, 128),
    (255, 128, 0),
    (128, 255, 0),
    (0, 128, 255),
    (255, 0, 128),
    (128, 0, 255),
    (0, 255, 128),
    (64, 224, 208),
    (255, 165, 0),
]


def preprocess_frame(
    frame_bgr: np.ndarray,
    model: OTXModel,
    device: str | torch.device = "cpu",
) -> OTXDataBatch:
    """Preprocess a BGR frame into an OTXDataBatch ready for model inference.

    Handles resizing, normalization, and proper scale_factor so bboxes
    are returned in original image coordinates.

    Args:
        frame_bgr: Input frame in BGR format (from cv2).
        model: OTX model (uses data_input_params for size/mean/std).
        device: Target device.

    Returns:
        OTXDataBatch with a single image.
    """
    inp_h, inp_w = model.data_input_params.input_size
    mean = model.data_input_params.mean
    std = model.data_input_params.std

    ori_h, ori_w = frame_bgr.shape[:2]

    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (inp_w, inp_h))
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float()
    tensor = (tensor - torch.tensor(mean).view(3, 1, 1)) / torch.tensor(std).view(3, 1, 1)

    img_info = ImageInfo(
        img_idx=0,
        img_shape=(inp_h, inp_w),
        ori_shape=(ori_h, ori_w),
        scale_factor=(inp_h / ori_h, inp_w / ori_w),
    )
    return OTXDataBatch(
        batch_size=1,
        images=tensor.unsqueeze(0).to(device),
        imgs_info=[img_info],
    )


def draw_detections(
    frame_bgr: np.ndarray,
    bboxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray | None = None,
    track_ids: np.ndarray | None = None,
    class_names: list[str] | None = None,
    conf_thresh: float = 0.0,
) -> np.ndarray:
    """Draw bounding boxes on a frame.

    Args:
        frame_bgr: Frame to draw on (modified in-place and returned).
        bboxes: (N, 4) array [x1, y1, x2, y2].
        scores: (N,) confidence scores.
        labels: (N,) class label indices (optional).
        track_ids: (N,) track IDs for consistent coloring (optional).
        class_names: List of class names for label display.
        conf_thresh: Only draw detections above this threshold.

    Returns:
        The frame with drawn boxes.
    """
    for i in range(len(bboxes)):
        if scores[i] < conf_thresh:
            continue

        x1, y1, x2, y2 = map(int, bboxes[i])
        tid = int(track_ids[i]) if track_ids is not None else i
        color = TRACK_COLORS[tid % len(TRACK_COLORS)]

        cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)

        parts = []
        if track_ids is not None:
            parts.append(f"ID:{tid}")
        if class_names is not None and labels is not None:
            cls_id = int(labels[i])
            if cls_id < len(class_names):
                parts.append(class_names[cls_id])
        parts.append(f"{scores[i]:.2f}")
        label = " ".join(parts)

        cv2.putText(frame_bgr, label, (x1, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame_bgr


# this is a common issue where the defult encoding doesnt work in browsers/vscode
def to_h264(src: str | Path, dst: str | Path) -> Path:
    """Re-encode a video to H.264 for browser/VSCode playback.

    Args:
        src: Source video path.
        dst: Destination path.

    Returns:
        Path to the H.264 encoded video.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                str(src),
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                "-an",
                str(dst),
            ],
            capture_output=True,
            check=True,
        )
    except FileNotFoundError:
        msg = "ffmpeg not found. Install it with: apt-get install ffmpeg (Linux) or brew install ffmpeg (macOS)"
        raise RuntimeError(msg) from None
    except subprocess.CalledProcessError as e:
        msg = f"ffmpeg failed to encode {src}: {e.stderr.decode()}"
        raise RuntimeError(msg) from e
    return dst
