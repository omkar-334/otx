# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for ByteTrack multi-object tracker."""

import numpy as np
import pytest
import torch

from getitune.backend.lightning.models.tracking.base import OTXTracker
from getitune.backend.lightning.models.tracking.bytetrack import ByteTrack


class TestByteTrackInit:
    def test_default_params(self) -> None:
        tracker = ByteTrack()
        assert tracker.track_thresh == 0.5
        assert tracker.track_buffer == 30
        assert tracker.match_thresh == 0.8
        assert tracker.frame_rate == 30
        assert tracker.tracker is not None

    def test_custom_params(self) -> None:
        tracker = ByteTrack(
            track_thresh=0.3,
            track_buffer=60,
            match_thresh=0.7,
            frame_rate=15,
        )
        assert tracker.track_thresh == 0.3
        assert tracker.track_buffer == 60
        assert tracker.match_thresh == 0.7
        assert tracker.frame_rate == 15

    def test_is_otx_tracker(self) -> None:
        tracker = ByteTrack()
        assert isinstance(tracker, OTXTracker)


class TestByteTrackUpdate:
    @pytest.fixture()
    def tracker(self) -> ByteTrack:
        t = ByteTrack(track_thresh=0.3)
        t.reset()
        return t

    def test_empty_detections(self, tracker: ByteTrack) -> None:
        bboxes = np.empty((0, 4))
        scores = np.empty(0)
        labels = np.empty(0, dtype=np.int64)

        out_bboxes, out_scores, out_labels, out_ids = tracker.update(
            bboxes, scores, labels, img_h=480, img_w=640,
        )

        assert out_bboxes.shape == (0, 4)
        assert out_scores.shape == (0,)
        assert out_labels.shape == (0,)
        assert out_ids.shape == (0,)

    def test_single_detection(self, tracker: ByteTrack) -> None:
        bboxes = np.array([[100, 100, 200, 200]])
        scores = np.array([0.9])
        labels = np.array([0])

        out_bboxes, out_scores, out_labels, out_ids = tracker.update(
            bboxes, scores, labels, img_h=480, img_w=640,
        )

        assert len(out_ids) == 1
        assert out_ids[0] >= 1  # track IDs start at 1
        assert out_bboxes.shape == (1, 4)

    def test_multiple_detections(self, tracker: ByteTrack) -> None:
        bboxes = np.array([
            [10, 10, 50, 50],
            [200, 200, 300, 300],
            [400, 100, 500, 200],
        ])
        scores = np.array([0.9, 0.8, 0.7])
        labels = np.array([0, 1, 2])

        out_bboxes, out_scores, out_labels, out_ids = tracker.update(
            bboxes, scores, labels, img_h=480, img_w=640,
        )

        assert len(out_ids) >= 1
        # All IDs should be unique
        assert len(set(out_ids)) == len(out_ids)

    def test_id_persistence_across_frames(self, tracker: ByteTrack) -> None:
        """Same object in same location should keep same ID across frames."""
        bbox = np.array([[100, 100, 200, 200]])
        score = np.array([0.9])
        label = np.array([0])

        _, _, _, ids_frame1 = tracker.update(bbox, score, label, 480, 640)

        # Slightly shifted bbox (same object moving)
        bbox2 = np.array([[105, 105, 205, 205]])
        _, _, _, ids_frame2 = tracker.update(bbox2, score, label, 480, 640)

        assert ids_frame1[0] == ids_frame2[0], "Same object should keep same track ID"

    def test_new_object_gets_new_id(self, tracker: ByteTrack) -> None:
        # Feed both objects for several frames so tracker establishes tracks
        bboxes = np.array([
            [100, 100, 200, 200],
            [400, 400, 500, 500],
        ])
        scores = np.array([0.9, 0.9])
        labels = np.array([0, 0])

        for _ in range(3):
            _, _, _, ids = tracker.update(bboxes, scores, labels, 480, 640)

        assert len(ids) == 2
        assert ids[0] != ids[1], "Different objects should have different IDs"

    def test_output_types(self, tracker: ByteTrack) -> None:
        bboxes = np.array([[100, 100, 200, 200]])
        scores = np.array([0.9])
        labels = np.array([0])

        out_bboxes, out_scores, out_labels, out_ids = tracker.update(
            bboxes, scores, labels, img_h=480, img_w=640,
        )

        assert out_bboxes.dtype == np.float64
        assert out_labels.dtype == np.int64
        assert out_ids.dtype == np.int64


class TestByteTrackReset:
    def test_reset_clears_state(self) -> None:
        tracker = ByteTrack(track_thresh=0.3)

        bbox = np.array([[100, 100, 200, 200]])
        score = np.array([0.9])
        label = np.array([0])

        tracker.update(bbox, score, label, 480, 640)
        tracker.update(bbox, score, label, 480, 640)

        tracker.reset()

        # After reset, IDs should start from 1 again
        _, _, _, ids = tracker.update(bbox, score, label, 480, 640)
        assert ids[0] == 1, "After reset, track IDs should restart from 1"

    def test_reset_creates_new_tracker(self) -> None:
        tracker = ByteTrack()
        old_internal = tracker.tracker

        tracker.reset()
        assert tracker.tracker is not old_internal


class TestSyncModelThreshold:
    """Test automatic model confidence threshold synchronization."""

    def test_syncs_when_model_threshold_is_higher(self) -> None:
        model = type("MockModel", (), {"hparams": {"best_confidence_threshold": 0.5}})()
        OTXTracker._sync_model_threshold(model, 0.3)
        assert model.hparams["best_confidence_threshold"] == 0.3

    def test_keeps_lower_model_threshold(self) -> None:
        model = type("MockModel", (), {"hparams": {"best_confidence_threshold": 0.1}})()
        OTXTracker._sync_model_threshold(model, 0.3)
        assert model.hparams["best_confidence_threshold"] == 0.1

    def test_syncs_when_threshold_is_none(self) -> None:
        model = type("MockModel", (), {"hparams": {"best_confidence_threshold": None}})()
        OTXTracker._sync_model_threshold(model, 0.3)
        assert model.hparams["best_confidence_threshold"] == 0.3

    def test_no_crash_without_hparams(self) -> None:
        model = type("MockModel", (), {})()
        OTXTracker._sync_model_threshold(model, 0.3)  # should not raise


class TestByteTrackLabelRecovery:
    """Test that tracked objects recover correct class labels."""

    @pytest.fixture()
    def tracker(self) -> ByteTrack:
        t = ByteTrack(track_thresh=0.3)
        t.reset()
        return t

    def test_labels_recovered_from_detections(self, tracker: ByteTrack) -> None:
        bboxes = np.array([
            [10, 10, 50, 50],
            [200, 200, 300, 300],
        ])
        scores = np.array([0.9, 0.8])
        labels = np.array([5, 12])  # car=5, dog=12

        _, _, out_labels, _ = tracker.update(bboxes, scores, labels, 480, 640)

        # Labels should be recovered from input detections, not all zeros
        assert not np.all(out_labels == 0) or len(out_labels) == 0
        for lbl in out_labels:
            assert lbl in [5, 12], f"Label {lbl} not in input labels"

    def test_empty_detections_returns_empty_labels(self, tracker: ByteTrack) -> None:
        bboxes = np.empty((0, 4))
        scores = np.empty(0)
        labels = np.empty(0, dtype=np.int64)

        _, _, out_labels, _ = tracker.update(bboxes, scores, labels, 480, 640)
        assert len(out_labels) == 0


class TestByteTrackTrackFrame:
    """Test track_frame returns PredictionBatch with correct fields."""

    def test_returns_prediction_batch(self) -> None:
        """Verify track_frame returns PredictionBatch with track_ids."""
        from unittest.mock import MagicMock

        from torchvision.tv_tensors import BoundingBoxes

        from getitune.data.entity.base import ImageInfo
        from getitune.data.entity.sample import PredictionBatch

        tracker = ByteTrack(track_thresh=0.3)
        tracker.reset()

        model = MagicMock()
        model.data_input_params.input_size = (416, 416)
        model.data_input_params.mean = (0.0, 0.0, 0.0)
        model.data_input_params.std = (1.0, 1.0, 1.0)
        model.hparams = {"best_confidence_threshold": 0.5}

        mock_preds = PredictionBatch(
            images=torch.randn(1, 3, 416, 416),
            imgs_info=[ImageInfo(
                img_idx=0,
                img_shape=(416, 416),
                ori_shape=(480, 640),
                scale_factor=(416 / 480, 416 / 640),
            )],
            bboxes=[BoundingBoxes(
                torch.tensor([[100.0, 100.0, 200.0, 200.0]]),
                format="XYXY",
                canvas_size=(480, 640),
            )],
            scores=[torch.tensor([0.9])],
            labels=[torch.tensor([0])],
        )
        model.predict_step.return_value = mock_preds

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = tracker.track_frame(model, frame, device="cpu")

        assert isinstance(result, PredictionBatch)
        assert result.track_ids is not None
        assert len(result.track_ids) == 1
        assert result.track_ids[0].dtype == torch.long
        assert result.bboxes is not None
        assert result.scores is not None
        assert result.labels is not None
