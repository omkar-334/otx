# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Miscellaneous helpers."""

from __future__ import annotations

import subprocess
from pathlib import Path


def to_h264(src: str | Path, dst: str | Path | None = None) -> Path:
    """Re-encode a video to H.264 so browsers and VSCode can play it.

    OpenCV's default ``mp4v`` codec is not playable in most browsers.
    When ``dst`` is omitted (or equals ``src``), the file is re-encoded in
    place via a temporary file; otherwise the source is left untouched.

    Args:
        src: Source video path.
        dst: Destination path for the H.264 video. Parent directories are
            created as needed. Defaults to re-encoding ``src`` in place.

    Returns:
        The destination path.

    Raises:
        RuntimeError: If ffmpeg is not installed or the encode fails.
    """
    src = Path(src)
    dst = Path(dst) if dst is not None else src
    in_place = dst == src
    target = src.with_name(f"{src.stem}.h264.tmp.mp4") if in_place else dst
    target.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
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
        str(target),
    ]
    try:
        subprocess.run(cmd, capture_output=True, check=True)  # noqa: S603
    except FileNotFoundError:
        msg = "ffmpeg not found. Install it with 'brew install ffmpeg' (macOS) or 'apt-get install ffmpeg' (Linux)."
        raise RuntimeError(msg) from None
    except subprocess.CalledProcessError as e:
        target.unlink(missing_ok=True)
        msg = f"ffmpeg failed to encode {src}: {e.stderr.decode(errors='replace')[-500:]}"
        raise RuntimeError(msg) from e
    if in_place:
        target.replace(src)
    return dst


# COCO category names in the standard 91-index label space (sparse: ids
# unused by COCO are absent).
COCO_CLASSES: dict[int, str] = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "airplane",
    6: "bus",
    7: "train",
    8: "truck",
    9: "boat",
    10: "traffic light",
    11: "fire hydrant",
    13: "stop sign",
    14: "parking meter",
    15: "bench",
    16: "bird",
    17: "cat",
    18: "dog",
    19: "horse",
    20: "sheep",
    21: "cow",
    22: "elephant",
    23: "bear",
    24: "zebra",
    25: "giraffe",
    27: "backpack",
    28: "umbrella",
    31: "handbag",
    32: "tie",
    33: "suitcase",
    34: "frisbee",
    35: "skis",
    36: "snowboard",
    37: "sports ball",
    38: "kite",
    39: "baseball bat",
    40: "baseball glove",
    41: "skateboard",
    42: "surfboard",
    43: "tennis racket",
    44: "bottle",
    46: "wine glass",
    47: "cup",
    48: "fork",
    49: "knife",
    50: "spoon",
    51: "bowl",
    52: "banana",
    53: "apple",
    54: "sandwich",
    55: "orange",
    56: "broccoli",
    57: "carrot",
    58: "hot dog",
    59: "pizza",
    60: "donut",
    61: "cake",
    62: "chair",
    63: "couch",
    64: "potted plant",
    65: "bed",
    67: "dining table",
    70: "toilet",
    72: "tv",
    73: "laptop",
    74: "mouse",
    75: "remote",
    76: "keyboard",
    77: "cell phone",
    78: "microwave",
    79: "oven",
    80: "toaster",
    81: "sink",
    82: "refrigerator",
    84: "book",
    85: "clock",
    86: "vase",
    87: "scissors",
    88: "teddy bear",
    89: "hair drier",
    90: "toothbrush",
}
