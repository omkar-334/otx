# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Typer command-line interface.

Two commands are exposed:

- ``getitrack run``: track precomputed detections (MOT text format) over a
  video and write per-frame track results, optionally with an annotated video.
- ``getitrack demo``: generate a synthetic bouncing-box scene, track it, and
  write an annotated video. Useful as a smoke test without any input data.
"""

from collections import defaultdict
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
import typer
import yaml

import getitrack.algorithms  # noqa: F401  -> registers the bundled algorithms
from getitrack.config import TrackerConfig
from getitrack.core.base import BaseTracker
from getitrack.core.detection import Detections, TrackedDetections
from getitrack.core.registry import resolve_tracker_config
from getitrack.io import VideoReader, VideoWriter
from getitrack.visualization import TrackAnnotator, color_for_track

app = typer.Typer(name="getitrack", help="Multi-object tracking toolkit.", no_args_is_help=True)

_MOT_MIN_COLS = 6
_MOT_SCORE_COL = 6
_MOT_CLASS_COL = 7


def _load_config(config: Path | None, algorithm: str | None = None) -> TrackerConfig:
    # Apply the algorithm override to the raw mapping, then validate.
    data: dict[str, Any] = {}
    if config is not None:
        with Path(config).open(encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    if algorithm is not None:
        data["algorithm"] = algorithm
    return resolve_tracker_config(data)


def _load_mot_detections(path: Path) -> dict[int, list[list[float]]]:
    """Parse a MOT-format detection file grouped by 1-based frame number.

    Each line is ``frame,id,x,y,w,h[,score[,class]]`` with ``x, y`` the
    top-left corner. Scores are clipped into ``[0, 1]``; a missing score
    defaults to 1.0 and a missing class to 0.
    """
    per_frame: dict[int, list[list[float]]] = defaultdict(list)
    for lineno, raw in enumerate(path.read_text().splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split(",")
        if len(parts) < _MOT_MIN_COLS:
            msg = f"{path}:{lineno}: expected at least {_MOT_MIN_COLS} comma-separated values; got {len(parts)}"
            raise ValueError(msg)
        frame = int(float(parts[0]))
        x, y, w, h = (float(v) for v in parts[2:6])
        score = float(parts[_MOT_SCORE_COL]) if len(parts) > _MOT_SCORE_COL else 1.0
        class_id = int(float(parts[_MOT_CLASS_COL])) if len(parts) > _MOT_CLASS_COL else 0
        score = min(max(score, 0.0), 1.0)
        per_frame[frame].append([x, y, x + w, y + h, score, float(class_id)])
    return per_frame


def _detections_for_frame(per_frame: dict[int, list[list[float]]], frame_id: int) -> Detections:
    rows = per_frame.get(frame_id)
    if not rows:
        return Detections.create_empty(frame_id=frame_id)
    arr = np.asarray(rows, dtype=np.float64)
    return Detections(
        bboxes=arr[:, 0:4].astype(np.float32),
        scores=arr[:, 4].astype(np.float32),
        class_ids=arr[:, 5].astype(np.int64),
        frame_id=frame_id,
    )


def _mot_rows(tracked: TrackedDetections) -> list[str]:
    rows = []
    for bbox, track_id, score, class_id in zip(
        tracked.bboxes,
        tracked.track_ids,
        tracked.scores,
        tracked.class_ids,
        strict=True,
    ):
        x1, y1, x2, y2 = (float(v) for v in bbox)
        rows.append(
            f"{tracked.frame_id},{int(track_id)},{x1:.2f},{y1:.2f},{x2 - x1:.2f},{y2 - y1:.2f}"
            f",{float(score):.4f},{int(class_id)},-1",
        )
    return rows


@app.command()
def run(
    video: Annotated[Path, typer.Argument(exists=True, dir_okay=False, help="Input video file.")],
    detections: Annotated[
        Path,
        typer.Option("--detections", "-d", exists=True, dir_okay=False, help="MOT-format detection file."),
    ],
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", exists=True, dir_okay=False, help="Tracker YAML config."),
    ] = None,
    algorithm: Annotated[
        str | None,
        typer.Option("--algorithm", "-a", help="Tracking algorithm. Overrides the config value."),
    ] = None,
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Track results file (MOT format). Defaults to '<video stem>_tracks.txt'."),
    ] = None,
    output_video: Annotated[
        Path | None,
        typer.Option("--output-video", help="Annotated video path. Skipped when omitted."),
    ] = None,
) -> None:
    """Track precomputed detections over a video and write MOT-format results."""
    tracker = BaseTracker.from_config(_load_config(config, algorithm))
    per_frame = _load_mot_detections(detections)
    if output is None:
        output = video.with_name(f"{video.stem}_tracks.txt")

    annotator = TrackAnnotator()
    rows: list[str] = []
    track_ids: set[int] = set()
    n_frames = 0
    with VideoReader(video) as reader:
        writer = None
        if output_video is not None:
            writer = VideoWriter(output_video, fps=reader.fps or 30.0, frame_size=(reader.width, reader.height))
        try:
            for index, frame in enumerate(reader):
                frame_id = index + 1  # MOT frame numbers are 1-based.
                tracked = tracker.update(_detections_for_frame(per_frame, frame_id))
                rows.extend(_mot_rows(tracked))
                track_ids.update(int(t) for t in tracked.track_ids)
                if writer is not None:
                    writer.write(annotator.annotate(frame, tracked))
                n_frames += 1
        finally:
            if writer is not None:
                writer.close()

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(rows) + "\n" if rows else "")
    typer.echo(f"Processed {n_frames} frames, {len(track_ids)} tracks -> {output}")
    if output_video is not None:
        typer.echo(f"Annotated video -> {output_video}")


@app.command()
def demo(
    output: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="Annotated video path. Defaults to 'getitrack_demo.mp4'."),
    ] = None,
    config: Annotated[
        Path | None,
        typer.Option("--config", "-c", exists=True, dir_okay=False, help="Tracker YAML config."),
    ] = None,
    algorithm: Annotated[
        str | None,
        typer.Option("--algorithm", "-a", help="Tracking algorithm. Overrides the config value."),
    ] = None,
    num_objects: Annotated[int, typer.Option(min=1, help="Number of moving boxes.")] = 4,
    num_frames: Annotated[int, typer.Option(min=1, help="Number of frames to generate.")] = 150,
    width: Annotated[int, typer.Option(min=64, help="Frame width in pixels.")] = 640,
    height: Annotated[int, typer.Option(min=64, help="Frame height in pixels.")] = 480,
    fps: Annotated[float, typer.Option(min=1.0, help="Output frame rate.")] = 30.0,
    seed: Annotated[int, typer.Option(help="Random seed for scene generation.")] = 0,
) -> None:
    """Track a synthetic bouncing-box scene and write an annotated video."""
    if output is None:
        output = Path("getitrack_demo.mp4")
    tracker = BaseTracker.from_config(_load_config(config, algorithm))
    scene = _SyntheticScene(width=width, height=height, num_objects=num_objects, seed=seed)
    annotator = TrackAnnotator()
    track_ids: set[int] = set()
    with VideoWriter(output, fps=fps, frame_size=(width, height)) as writer:
        for frame_id in range(1, num_frames + 1):
            scene.advance()
            frame = scene.render()
            tracked = tracker.update(scene.noisy_detections(frame_id))
            track_ids.update(int(t) for t in tracked.track_ids)
            writer.write(annotator.annotate(frame, tracked))
    typer.echo(f"Wrote {num_frames} frames, {len(track_ids)} tracks -> {output}")


class _SyntheticScene:
    """Constant-velocity boxes bouncing off the frame edges.

    Detections are the true boxes with Gaussian corner noise, jittered
    scores, and random dropouts, so the tracker sees realistic imperfect
    input (including low-score detections for ByteTrack's second stage).
    """

    _MIN_BOX = 8.0
    _DROPOUT_PROB = 0.05
    _CORNER_NOISE_STD = 1.5

    def __init__(self, width: int, height: int, num_objects: int, seed: int) -> None:
        self._w = float(width)
        self._h = float(height)
        self._rng = np.random.default_rng(seed)
        base = min(self._w, self._h)
        self._sizes = self._rng.uniform(0.08, 0.15, size=(num_objects, 2)) * base
        max_xy = np.array([self._w, self._h]) - self._sizes
        self._tl = self._rng.uniform(0.0, 1.0, size=(num_objects, 2)) * max_xy
        self._vel = self._rng.uniform(-4.0, 4.0, size=(num_objects, 2))

    def advance(self) -> None:
        """Move every box one step and bounce off the frame edges."""
        self._tl += self._vel
        max_xy = np.array([self._w, self._h]) - self._sizes
        for axis in range(2):
            low = self._tl[:, axis] < 0.0
            high = self._tl[:, axis] > max_xy[:, axis]
            self._vel[low | high, axis] *= -1.0
            self._tl[:, axis] = np.clip(self._tl[:, axis], 0.0, max_xy[:, axis])

    def boxes(self) -> np.ndarray:
        """Return the true ``(N, 4)`` xyxy boxes."""
        return np.concatenate([self._tl, self._tl + self._sizes], axis=1).astype(np.float32)

    def render(self) -> np.ndarray:
        """Render the scene as a BGR uint8 frame."""
        frame = np.full((int(self._h), int(self._w), 3), 30, dtype=np.uint8)
        for i, box in enumerate(self.boxes()):
            x1, y1, x2, y2 = (round(float(v)) for v in box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color_for_track(i + 1), -1)
        return frame

    def noisy_detections(self, frame_id: int) -> Detections:
        """Return the true boxes perturbed into detector-like output."""
        boxes = self.boxes().astype(np.float64)
        keep = self._rng.random(boxes.shape[0]) >= self._DROPOUT_PROB
        boxes = boxes[keep]
        boxes += self._rng.normal(0.0, self._CORNER_NOISE_STD, size=boxes.shape)
        boxes[:, 0] = np.clip(boxes[:, 0], 0.0, self._w - self._MIN_BOX)
        boxes[:, 1] = np.clip(boxes[:, 1], 0.0, self._h - self._MIN_BOX)
        boxes[:, 2] = np.clip(np.maximum(boxes[:, 2], boxes[:, 0] + self._MIN_BOX), 0.0, self._w)
        boxes[:, 3] = np.clip(np.maximum(boxes[:, 3], boxes[:, 1] + self._MIN_BOX), 0.0, self._h)
        scores = np.clip(self._rng.normal(0.85, 0.12, size=boxes.shape[0]), 0.05, 0.99)
        return Detections(
            bboxes=boxes.astype(np.float32),
            scores=scores.astype(np.float32),
            class_ids=np.zeros(boxes.shape[0], dtype=np.int64),
            frame_id=frame_id,
        )


if __name__ == "__main__":
    app()
