# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Base class for multi-object trackers in OTX."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import cv2
import torch
from torchvision.tv_tensors import BoundingBoxes

from otx.backend.native.tools.video import draw_detections, preprocess_frame, to_h264
from otx.data.entity.torch import OTXPredBatch

if TYPE_CHECKING:
    import numpy as np

    from otx.backend.native.models.detection.base import OTXDetectionModel


class OTXTracker(ABC):
    """Base class for multi-object trackers.

    A tracker is detector-agnostic. It takes per-frame detections from any
    OTXDetectionModel and associates them across frames to produce consistent
    track IDs. The model's internal confidence threshold is automatically
    synced to track_thresh so detections aren't silently filtered.

    Args:
        track_thresh: Minimum confidence for a detection to be considered in
            the primary (first-stage) association. Also auto-syncs the model's
            internal best_confidence_threshold to this value.
        track_buffer: Number of frames to keep lost tracks alive before
            removing them. A lost track can be re-identified if the object
            reappears within this window.
        match_thresh: IoU threshold for the primary association stage.
            Higher values require tighter spatial overlap between predictions
            and existing tracks.
    """

    def __init__(
        self,
        track_thresh: float = 0.5,
        track_buffer: int = 30,
        match_thresh: float = 0.8,
    ) -> None:
        self.track_thresh = track_thresh
        self.track_buffer = track_buffer
        self.match_thresh = match_thresh
        self._tracker_impl: Any = None

    @abstractmethod
    def _create_tracker(self) -> Any:  # noqa: ANN401
        """Create the underlying tracker implementation."""

    @property
    def tracker(self) -> Any:  # noqa: ANN401
        """Lazy-init tracker on first access."""
        if self._tracker_impl is None:
            self._tracker_impl = self._create_tracker()
        return self._tracker_impl

    def reset(self) -> None:
        """Reset tracker state (call between videos)."""
        from .tracker.basetrack import BaseTrack

        BaseTrack._count = 0  # noqa: SLF001
        self._tracker_impl = None

    @abstractmethod
    def update(
        self,
        bboxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        img_h: int,
        img_w: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Update tracker with a single frame's detections.

        Args:
            bboxes: (N, 4) array of [x1, y1, x2, y2].
            scores: (N,) detection scores.
            labels: (N,) class labels.
            img_h: Image height.
            img_w: Image width.

        Returns:
            Tuple of (bboxes, scores, labels, track_ids) for active tracks.
        """

    @staticmethod
    def _sync_model_threshold(model: OTXDetectionModel, conf_thresh: float) -> None:
        """Set model's internal confidence threshold to match the tracker's.

        OTX models filter detections internally via best_confidence_threshold
        (default 0.5) before returning predictions. If the tracker wants a lower
        threshold, the model would silently discard detections the tracker needs.
        This syncs the two so the model passes through everything above conf_thresh.
        """
        if hasattr(model, "hparams"):
            current = model.hparams.get("best_confidence_threshold")
            if current is None or current > conf_thresh:
                model.hparams["best_confidence_threshold"] = conf_thresh

    def track_frame(
        self,
        model: OTXDetectionModel,
        frame_bgr: np.ndarray,
        device: str | torch.device = "cpu",
    ) -> OTXPredBatch:
        """Run detection + tracking on a single BGR frame.

        Args:
            model: Any OTX detection model (YOLOX, RTDETR, SSD, etc.).
            frame_bgr: Input frame in BGR format.
            device: Device the model is on.

        Returns:
            OTXPredBatch with bboxes, scores, labels, and track_ids.
        """
        self._sync_model_threshold(model, self.track_thresh)
        batch = preprocess_frame(frame_bgr, model, device)

        with torch.no_grad():
            preds = model.predict_step(batch, batch_idx=0)

        assert preds.bboxes is not None  # noqa: S101
        assert preds.scores is not None  # noqa: S101
        assert preds.labels is not None  # noqa: S101
        bboxes = preds.bboxes[0].cpu().numpy()
        scores = preds.scores[0].cpu().numpy()
        labels = preds.labels[0].cpu().numpy()

        ori_h, ori_w = frame_bgr.shape[:2]
        t_bboxes, t_scores, t_labels, t_ids = self.update(
            bboxes,
            scores,
            labels,
            img_h=ori_h,
            img_w=ori_w,
        )

        return OTXPredBatch(
            batch_size=1,
            images=preds.images,
            imgs_info=preds.imgs_info,
            bboxes=[
                BoundingBoxes(
                    torch.from_numpy(t_bboxes).float(),
                    format="XYXY",
                    canvas_size=(ori_h, ori_w),
                )
            ],
            scores=[torch.from_numpy(t_scores).float()],
            labels=[torch.from_numpy(t_labels).long()],
            track_ids=[torch.from_numpy(t_ids).long()],
        )

    def track(
        self,
        model: OTXDetectionModel,
        video_path: str | Path,
        output_dir: str | Path = "videos/tracked",
        device: str | torch.device = "cpu",
        conf_thresh: float = 0.3,
        class_names: list[str] | None = None,
        verbose: bool = False,
    ) -> Path:
        """Run detection + tracking on a video and save the result.

        Args:
            model: Any OTX detection model in eval mode.
            video_path: Path to input video.
            output_dir: Directory for output video.
            device: Device the model is on.
            conf_thresh: Confidence threshold for visualization only — controls
                which tracked boxes are drawn on the output video. Does NOT affect
                tracking logic (that's controlled by track_thresh).
            class_names: List of class names (e.g. COCO 80 classes) for labeling
                boxes on the video and in verbose logs. If None, only IDs are shown.
            verbose: If True, log per-frame detection counts, track IDs, and labels.

        Returns:
            Path to saved H.264 video.
        """
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        raw_path = output_dir / f"{video_path.stem}_raw.mp4"
        h264_path = output_dir / f"{video_path.stem}.mp4"
        writer = cv2.VideoWriter(
            str(raw_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (w_orig, h_orig),
        )

        self.reset()
        self._sync_model_threshold(model, self.track_thresh)

        frame_idx = 0
        max_id = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            batch = preprocess_frame(frame, model, device)

            with torch.no_grad():
                preds = model.predict_step(batch, batch_idx=0)

            assert preds.bboxes is not None  # noqa: S101
            assert preds.scores is not None  # noqa: S101
            assert preds.labels is not None  # noqa: S101
            bboxes = preds.bboxes[0].cpu().numpy()
            scores = preds.scores[0].cpu().numpy()
            labels = preds.labels[0].cpu().numpy()

            t_bboxes, t_scores, t_labels, t_ids = self.update(
                bboxes,
                scores,
                labels,
                img_h=h_orig,
                img_w=w_orig,
            )

            cur_max = int(t_ids.max()) if len(t_ids) > 0 else 0
            max_id = max(max_id, cur_max)

            if verbose:
                n_dets = (scores > conf_thresh).sum()
                new_max = cur_max > (max_id - 1)
                id_str = f" NEW max_id={max_id}" if new_max else ""
                # Build ID:label pairs for readable logging
                if class_names is not None and len(t_ids) > 0:
                    id_labels = [
                        f"{tid}:{class_names[lbl] if lbl < len(class_names) else lbl}"
                        for tid, lbl in zip(t_ids.tolist(), t_labels.tolist())
                    ]
                    tracks_str = f"[{', '.join(id_labels)}]"
                else:
                    tracks_str = str(t_ids.tolist())
                print(
                    f"  Frame {frame_idx:>4d}: "
                    f"{n_dets} dets (>{conf_thresh}), "
                    f"{len(t_ids)} tracks, "
                    f"IDs={tracks_str}"
                    f"{id_str}"
                )

            draw_detections(
                frame,
                t_bboxes,
                t_scores,
                labels=t_labels,
                track_ids=t_ids,
                class_names=class_names,
                conf_thresh=conf_thresh,
            )

            writer.write(frame)
            frame_idx += 1
            if not verbose and frame_idx % 50 == 0:
                print(f"  {frame_idx}/{total} frames")

        cap.release()
        writer.release()

        to_h264(raw_path, h264_path)
        raw_path.unlink()
        print(f"Saved: {h264_path} ({frame_idx} frames, max_id={max_id})")
        return h264_path
