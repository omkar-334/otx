# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for getitrack.utils."""

import shutil

import numpy as np
import pytest

from getitrack.io import VideoReader, VideoWriter
from getitrack.utils import COCO_CLASSES, to_h264

_HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _write_video(path, n_frames=5) -> None:
    with VideoWriter(path, fps=30.0, frame_size=(64, 48)) as writer:
        for _ in range(n_frames):
            writer.write(np.full((48, 64, 3), 40, dtype=np.uint8))


class TestToH264:
    @pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not installed")
    def test_reencodes_and_keeps_source(self, tmp_path):
        src = tmp_path / "raw.mp4"
        dst = tmp_path / "out" / "final.mp4"
        _write_video(src, n_frames=5)
        result = to_h264(src, dst)
        assert result == dst
        assert src.is_file()  # source untouched
        with VideoReader(dst) as reader:
            assert len(list(reader)) == 5

    @pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not installed")
    def test_garbage_input_raises(self, tmp_path):
        src = tmp_path / "bogus.mp4"
        src.write_text("not a video")
        with pytest.raises(RuntimeError, match="failed to encode"):
            to_h264(src, tmp_path / "out.mp4")

    def test_missing_ffmpeg_raises(self, tmp_path, monkeypatch):
        monkeypatch.setenv("PATH", str(tmp_path))
        src = tmp_path / "raw.mp4"
        src.write_bytes(b"")
        with pytest.raises(RuntimeError, match="ffmpeg not found"):
            to_h264(src, tmp_path / "out.mp4")


class TestCocoClasses:
    def test_well_known_ids(self):
        assert COCO_CLASSES[1] == "person"
        assert COCO_CLASSES[3] == "car"
        assert COCO_CLASSES[8] == "truck"

    def test_sparse_91_index_space(self):
        assert len(COCO_CLASSES) == 80
        assert max(COCO_CLASSES) == 90
        assert 12 not in COCO_CLASSES  # gap id in the 91-index space


class TestToH264InPlace:
    @pytest.mark.skipif(not _HAS_FFMPEG, reason="ffmpeg not installed")
    def test_reencodes_in_place(self, tmp_path):
        path = tmp_path / "video.mp4"
        _write_video(path, n_frames=4)
        result = to_h264(path)
        assert result == path
        assert path.is_file()
        assert not list(tmp_path.glob("*.tmp.mp4"))  # temp file cleaned up
        with VideoReader(path) as reader:
            assert len(list(reader)) == 4
