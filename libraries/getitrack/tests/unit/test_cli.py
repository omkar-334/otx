# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Tests for getitrack.cli."""

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from getitrack.cli import app
from getitrack.io import VideoReader, VideoWriter

runner = CliRunner()

_W, _H = 128, 96


def _make_video(path, n_frames) -> Path:
    with VideoWriter(path, fps=30.0, frame_size=(_W, _H)) as writer:
        for _ in range(n_frames):
            writer.write(np.full((_H, _W, 3), 30, dtype=np.uint8))
    return path


def _make_mot_detections(path, n_frames) -> Path:
    """Two boxes drifting horizontally in opposite directions, score 0.9."""
    lines = []
    for frame in range(1, n_frames + 1):
        x1 = 5 + 2 * frame
        x2 = 80 - 2 * frame
        lines.append(f"{frame},-1,{x1},10,20,20,0.9,0")
        lines.append(f"{frame},-1,{x2},50,20,20,0.9,0")
    path.write_text("\n".join(lines) + "\n")
    return path


def _make_config(path) -> Path:
    path.write_text("lifecycle:\n  min_hits: 1\n")
    return path


class TestDemo:
    def test_writes_annotated_video(self, tmp_path):
        out = tmp_path / "demo.mp4"
        result = runner.invoke(
            app,
            [
                "demo",
                "--output",
                str(out),
                "--num-frames",
                "12",
                "--num-objects",
                "3",
                "--width",
                "128",
                "--height",
                "96",
            ],
        )
        assert result.exit_code == 0, result.output
        assert out.is_file()
        with VideoReader(out) as reader:
            assert len(list(reader)) == 12
        assert "12 frames" in result.output

    def test_tracks_are_created(self, tmp_path):
        out = tmp_path / "demo.mp4"
        result = runner.invoke(app, ["demo", "--output", str(out), "--num-frames", "20", "--num-objects", "2"])
        assert result.exit_code == 0, result.output
        assert "0 tracks" not in result.output


class TestRun:
    def test_writes_mot_results(self, tmp_path):
        n_frames = 8
        video = _make_video(tmp_path / "clip.mp4", n_frames)
        dets = _make_mot_detections(tmp_path / "det.txt", n_frames)
        cfg = _make_config(tmp_path / "cfg.yaml")
        out = tmp_path / "tracks.txt"
        result = runner.invoke(
            app,
            ["run", str(video), "--detections", str(dets), "--config", str(cfg), "--output", str(out)],
        )
        assert result.exit_code == 0, result.output
        rows = [line.split(",") for line in out.read_text().splitlines()]
        # min_hits=1 promotes immediately: 2 tracks on every frame.
        assert len(rows) == 2 * n_frames
        track_ids = {int(r[1]) for r in rows}
        assert track_ids == {1, 2}
        assert "2 tracks" in result.output

    def test_default_output_path(self, tmp_path):
        n_frames = 4
        video = _make_video(tmp_path / "clip.mp4", n_frames)
        dets = _make_mot_detections(tmp_path / "det.txt", n_frames)
        result = runner.invoke(app, ["run", str(video), "--detections", str(dets)])
        assert result.exit_code == 0, result.output
        assert (tmp_path / "clip_tracks.txt").is_file()

    def test_output_video(self, tmp_path):
        n_frames = 6
        video = _make_video(tmp_path / "clip.mp4", n_frames)
        dets = _make_mot_detections(tmp_path / "det.txt", n_frames)
        out_video = tmp_path / "annotated.mp4"
        result = runner.invoke(
            app,
            ["run", str(video), "--detections", str(dets), "--output-video", str(out_video)],
        )
        assert result.exit_code == 0, result.output
        with VideoReader(out_video) as reader:
            assert len(list(reader)) == n_frames

    def test_missing_detections_file_fails(self, tmp_path):
        video = _make_video(tmp_path / "clip.mp4", 2)
        result = runner.invoke(app, ["run", str(video), "--detections", str(tmp_path / "nope.txt")])
        assert result.exit_code != 0

    def test_malformed_detection_line_fails(self, tmp_path):
        video = _make_video(tmp_path / "clip.mp4", 2)
        bad = tmp_path / "det.txt"
        bad.write_text("1,2,3\n")
        result = runner.invoke(app, ["run", str(video), "--detections", str(bad)])
        assert result.exit_code != 0


class TestHelp:
    def test_lists_commands(self):
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "demo" in result.output


class TestAlgorithmOption:
    def test_run_with_explicit_algorithm(self, tmp_path):
        video = _make_video(tmp_path / "clip.mp4", 3)
        dets = _make_mot_detections(tmp_path / "det.txt", 3)
        result = runner.invoke(app, ["run", str(video), "--detections", str(dets), "--algorithm", "bytetrack"])
        assert result.exit_code == 0, result.output

    def test_run_with_unknown_algorithm_fails(self, tmp_path):
        video = _make_video(tmp_path / "clip.mp4", 2)
        dets = _make_mot_detections(tmp_path / "det.txt", 2)
        result = runner.invoke(app, ["run", str(video), "--detections", str(dets), "--algorithm", "nope"])
        assert result.exit_code != 0

    def test_demo_with_explicit_algorithm(self, tmp_path):
        out = tmp_path / "demo.mp4"
        result = runner.invoke(app, ["demo", "--output", str(out), "--num-frames", "5", "-a", "bytetrack"])
        assert result.exit_code == 0, result.output
        assert out.is_file()
