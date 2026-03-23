from __future__ import annotations

from collections import OrderedDict
from typing import Any

import numpy as np


class TrackState:
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    _count: int = 0

    track_id: int = 0
    is_activated: bool = False
    state: int = TrackState.New

    history: OrderedDict[int, Any] = OrderedDict()
    features: list[Any] = []
    curr_feature: Any = None
    score: float = 0
    start_frame: int = 0
    frame_id: int = 0
    time_since_update: int = 0

    # multi-camera
    location: tuple[float, float] = (np.inf, np.inf)

    @property
    def end_frame(self) -> int:
        return self.frame_id

    @staticmethod
    def next_id() -> int:
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args: Any) -> None:
        raise NotImplementedError

    def predict(self) -> None:
        raise NotImplementedError

    def update(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    def mark_lost(self) -> None:
        self.state = TrackState.Lost

    def mark_removed(self) -> None:
        self.state = TrackState.Removed
