# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Multi-object trackers for OTX."""

from .base import OTXTracker
from .bytetrack import ByteTrack

__all__ = ["OTXTracker", "ByteTrack"]
