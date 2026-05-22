from __future__ import annotations

from .gold_analyzer import multi_timeframe_summary


def analyze_mtf(frames: dict):
    return multi_timeframe_summary(frames)
